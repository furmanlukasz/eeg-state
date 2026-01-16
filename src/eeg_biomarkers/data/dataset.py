"""PyTorch Dataset and DataModule for EEG data."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig
from tqdm import tqdm

from eeg_biomarkers.data.preprocessing import (
    load_eeg_file,
    prepare_phase_chunks,
)

logger = logging.getLogger(__name__)


def _init_worker():
    """Initialize worker process - suppress MNE verbose output."""
    import mne
    mne.set_log_level("ERROR")


def _load_single_file(args: tuple, preprocess_cfg: dict, model_cfg: dict) -> dict | None:
    """
    Load and preprocess a single EEG file. Used for parallel processing.

    Args:
        args: Tuple of (fif_file_path, subject_id, group_idx, group_name)
        preprocess_cfg: Preprocessing config dict
        model_cfg: Model config dict

    Returns:
        Dict with chunks, mask, subject_id, group_idx or None if failed
    """
    import mne
    mne.set_log_level("ERROR")  # Suppress MNE output in worker

    fif_file, subject_id, group_idx, group_name = args
    fif_file = Path(fif_file)  # Convert back from string

    try:
        # Load and preprocess
        raw = load_eeg_file(
            fif_file,
            filter_low=preprocess_cfg["filter_low"],
            filter_high=preprocess_cfg["filter_high"],
            reference=preprocess_cfg["reference"],
            notch_freq=preprocess_cfg["notch_freq"],
        )

        # Extract phase chunks
        chunks, mask, _ = prepare_phase_chunks(
            raw,
            chunk_duration=preprocess_cfg["chunk_duration"],
            chunk_overlap=preprocess_cfg["chunk_overlap"],
            include_amplitude=model_cfg["include_amplitude"],
        )

        # Convert to float32 to save memory
        chunks = chunks.astype(np.float32)

        return {
            "chunks": chunks,
            "mask": mask,
            "subject_id": subject_id,
            "group_idx": group_idx,
            "n_chunks": chunks.shape[0],
        }

    except Exception as e:
        return {"error": str(e), "file": fif_file.name}


class EEGDataset(Dataset):
    """
    PyTorch Dataset for EEG phase chunks.

    Args:
        chunks: Phase data (n_samples, n_features, time)
        masks: Validity masks (n_samples, time)
        labels: Condition labels (n_samples,)
        subject_ids: Subject identifiers (n_samples,)
    """

    def __init__(
        self,
        chunks: np.ndarray,
        masks: np.ndarray,
        labels: np.ndarray | None = None,
        subject_ids: np.ndarray | None = None,
    ):
        self.chunks = torch.from_numpy(chunks).float()
        self.masks = torch.from_numpy(masks).bool()
        self.labels = torch.from_numpy(labels).long() if labels is not None else None
        self.subject_ids = subject_ids

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "data": self.chunks[idx],
            "mask": self.masks[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        if self.subject_ids is not None:
            item["subject_id"] = self.subject_ids[idx]
        return item


class EEGDataModule:
    """
    Data module for loading and preparing EEG datasets.

    Handles:
    - Loading data from multiple groups/conditions
    - Preprocessing and phase extraction
    - Train/test splitting with proper subject-level separation
    - DataLoader creation

    Args:
        cfg: Hydra configuration
        data_dir: Root directory containing group folders
    """

    def __init__(self, cfg: DictConfig, data_dir: str | Path):
        self.cfg = cfg
        self.data_dir = Path(data_dir)

        self.train_dataset: EEGDataset | None = None
        self.val_dataset: EEGDataset | None = None
        self.test_dataset: EEGDataset | None = None

        self._label_map: dict[str, int] = {}

    def setup(self, stage: Literal["fit", "test", "predict"] = "fit") -> None:
        """
        Load and prepare data for the specified stage.

        Args:
            stage: One of "fit", "test", or "predict"
        """
        if stage == "fit":
            self._load_and_split_data()
        elif stage == "test":
            if self.test_dataset is None:
                self._load_and_split_data()
        elif stage == "predict":
            self._load_all_data()

    def _load_and_split_data(self) -> None:
        """Load data and create train/val/test splits."""
        all_chunks = []
        all_masks = []
        all_labels = []
        all_subject_ids = []

        # First pass: collect all files to process (for progress bar)
        files_to_process = []
        for group_idx, group in enumerate(self.cfg.data.groups):
            group_name = group.name
            group_path = self.data_dir / group.path
            self._label_map[group_name] = group_idx

            if not group_path.exists():
                logger.warning(f"Group path does not exist: {group_path}")
                continue

            # Get subject directories
            subject_dirs = sorted([d for d in group_path.iterdir() if d.is_dir()])

            # Randomize subject sampling to avoid systematic bias
            n_subjects = self.cfg.data.sampling.n_subjects_per_group
            if n_subjects is not None and n_subjects < len(subject_dirs):
                rng = np.random.RandomState(self.cfg.data.sampling.random_seed)
                indices = rng.permutation(len(subject_dirs))[:n_subjects]
                subject_dirs = [subject_dirs[i] for i in sorted(indices)]
            elif n_subjects is not None:
                subject_dirs = subject_dirs[:n_subjects]

            for subject_dir in subject_dirs:
                subject_id = subject_dir.name
                # Filter out macOS metadata files (._*)
                fif_files = [
                    f for f in subject_dir.rglob("*_eeg.fif")
                    if not f.name.startswith("._")
                ]
                for fif_file in fif_files:
                    files_to_process.append((fif_file, subject_id, group_idx, group_name))

        logger.info(f"Found {len(files_to_process)} EEG files to process")

        # Prepare config dicts for worker processes (OmegaConf can't be pickled)
        preprocess_cfg = {
            "filter_low": self.cfg.data.preprocessing.filter_low,
            "filter_high": self.cfg.data.preprocessing.filter_high,
            "reference": self.cfg.data.preprocessing.reference,
            "notch_freq": self.cfg.data.preprocessing.notch_freq,
            "chunk_duration": self.cfg.data.preprocessing.chunk_duration,
            "chunk_overlap": self.cfg.data.preprocessing.chunk_overlap,
        }
        model_cfg = {
            "include_amplitude": self.cfg.model.phase.include_amplitude,
        }

        # Convert Path objects to strings for pickling
        files_to_process_str = [
            (str(f), sid, gidx, gname) for f, sid, gidx, gname in files_to_process
        ]

        # Determine number of workers (use all CPUs, but cap at 8 to avoid memory issues)
        # Lower cap because each worker loads full EEG files into memory
        n_workers = min(os.cpu_count() or 4, 8)
        logger.info(f"Loading data with {n_workers} parallel workers")

        # Load files in parallel using spawn context (safer with MNE)
        n_failed = 0
        load_func = partial(_load_single_file, preprocess_cfg=preprocess_cfg, model_cfg=model_cfg)

        import multiprocessing as mp
        ctx = mp.get_context("spawn")

        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_init_worker,
        ) as executor:
            # Submit all jobs
            futures = {executor.submit(load_func, args): args for args in files_to_process_str}

            # Process results as they complete with progress bar
            with tqdm(total=len(futures), desc="Loading EEG data", unit="file") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=120)  # 2 min timeout per file
                    except Exception as e:
                        n_failed += 1
                        if n_failed <= 5:
                            logger.warning(f"Worker failed: {e}")
                        result = None

                    if result is None:
                        n_failed += 1
                    elif "error" in result:
                        n_failed += 1
                        if n_failed <= 5:
                            logger.warning(f"Failed to load {result['file']}: {result['error']}")
                        elif n_failed == 6:
                            logger.warning("Suppressing further load warnings...")
                    else:
                        all_chunks.append(result["chunks"])
                        all_masks.append(result["mask"])
                        all_labels.extend([result["group_idx"]] * result["n_chunks"])
                        all_subject_ids.extend([result["subject_id"]] * result["n_chunks"])

                    pbar.update(1)

        if n_failed > 0:
            logger.warning(f"Total files failed to load: {n_failed}")

        if not all_chunks:
            raise RuntimeError("No data loaded! Check your data paths and file formats.")

        # Concatenate all data
        logger.info("Concatenating data...")
        all_chunks = np.concatenate(all_chunks, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)
        all_labels = np.array(all_labels)
        all_subject_ids = np.array(all_subject_ids)

        logger.info(f"Loaded {len(all_chunks)} chunks from {len(set(all_subject_ids))} subjects")

        # Split by subject (critical: prevents label leakage)
        unique_subjects = np.unique(all_subject_ids)

        # Get subject-level labels for stratification
        subject_labels = {}
        for subj, label in zip(all_subject_ids, all_labels):
            subject_labels[subj] = label
        stratify_labels = [subject_labels[s] for s in unique_subjects]

        # Train/test split at subject level
        train_subjects, test_subjects = train_test_split(
            unique_subjects,
            test_size=self.cfg.data.splitting.test_size,
            random_state=self.cfg.data.sampling.random_seed,
            stratify=stratify_labels,
        )

        # Further split train into train/val
        train_subject_labels = [subject_labels[s] for s in train_subjects]
        train_subjects, val_subjects = train_test_split(
            train_subjects,
            test_size=0.15,  # 15% of training for validation
            random_state=self.cfg.data.sampling.random_seed,
            stratify=train_subject_labels,
        )

        # Create masks for each split
        train_mask = np.isin(all_subject_ids, train_subjects)
        val_mask = np.isin(all_subject_ids, val_subjects)
        test_mask = np.isin(all_subject_ids, test_subjects)

        # Create datasets
        self.train_dataset = EEGDataset(
            all_chunks[train_mask],
            all_masks[train_mask],
            all_labels[train_mask],
            all_subject_ids[train_mask],
        )
        self.val_dataset = EEGDataset(
            all_chunks[val_mask],
            all_masks[val_mask],
            all_labels[val_mask],
            all_subject_ids[val_mask],
        )
        self.test_dataset = EEGDataset(
            all_chunks[test_mask],
            all_masks[test_mask],
            all_labels[test_mask],
            all_subject_ids[test_mask],
        )

        logger.info(
            f"Split: {len(self.train_dataset)} train, "
            f"{len(self.val_dataset)} val, {len(self.test_dataset)} test"
        )

    def _load_all_data(self) -> None:
        """Load all data without splitting (for prediction)."""
        # Similar to _load_and_split_data but puts everything in train_dataset
        # Implementation omitted for brevity
        pass

    def train_dataloader(self, batch_size: int | None = None) -> DataLoader:
        """Get training DataLoader."""
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') first")

        bs = batch_size or self.cfg.training.batch_size
        return DataLoader(
            self.train_dataset,
            batch_size=bs,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self, batch_size: int | None = None) -> DataLoader:
        """Get validation DataLoader."""
        if self.val_dataset is None:
            raise RuntimeError("Call setup('fit') first")

        bs = batch_size or self.cfg.training.batch_size
        return DataLoader(
            self.val_dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    def test_dataloader(self, batch_size: int | None = None) -> DataLoader:
        """Get test DataLoader."""
        if self.test_dataset is None:
            raise RuntimeError("Call setup('test') first")

        bs = batch_size or self.cfg.training.batch_size
        return DataLoader(
            self.test_dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    @property
    def n_channels(self) -> int:
        """Number of EEG channels."""
        if self.train_dataset is not None:
            # Infer from data shape
            n_features = self.train_dataset.chunks.shape[1]
            phase_channels = 3 if self.cfg.model.phase.include_amplitude else 2
            return n_features // phase_channels
        raise RuntimeError("Call setup() first to determine n_channels")

    @property
    def label_map(self) -> dict[str, int]:
        """Mapping from group names to label indices."""
        return self._label_map
