"""PyTorch Dataset and DataModule for EEG data."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
import logging
import hashlib

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig
from tqdm import tqdm

from eeg_biomarkers.data.preprocessing import (
    load_eeg_file,
    prepare_phase_chunks,
)

logger = logging.getLogger(__name__)


def get_cache_key(cfg: DictConfig) -> str:
    """Generate a unique cache key based on preprocessing config."""
    key_parts = [
        f"filter_{cfg.data.preprocessing.filter_low}_{cfg.data.preprocessing.filter_high}",
        f"ref_{cfg.data.preprocessing.reference}",
        f"notch_{cfg.data.preprocessing.notch_freq}",
        f"chunk_{cfg.data.preprocessing.chunk_duration}_{cfg.data.preprocessing.chunk_overlap}",
        f"amp_{cfg.model.phase.include_amplitude}",
    ]
    key_str = "_".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()[:8]


class CachedFileDataset(Dataset):
    """
    Dataset that loads preprocessed chunks from a single cached .pt file.

    Each file is preprocessed once and saved as a tensor file.
    Loading tensors is extremely fast compared to MNE preprocessing.
    """

    def __init__(self, cache_path: Path, label: int, subject_id: str):
        self.cache_path = cache_path
        self.label = label
        self.subject_id = subject_id

        # Load cached data (fast - just tensor loading)
        data = torch.load(cache_path, weights_only=True)
        self.chunks = data["chunks"]  # (n_chunks, features, time)
        self.masks = data["masks"]    # (n_chunks, time)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict:
        return {
            "data": self.chunks[idx],
            "mask": self.masks[idx],
            "label": torch.tensor(self.label, dtype=torch.long),
            "subject_id": self.subject_id,
        }


class EEGDataset(Dataset):
    """
    In-memory PyTorch Dataset for EEG phase chunks.

    Use for smaller datasets or when all data fits in memory.

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
    - Preprocessing and phase extraction with disk caching
    - Train/test splitting with proper subject-level separation
    - DataLoader creation

    The preprocessing cache ensures that expensive CSD/filtering operations
    are only done once per file. Subsequent runs load fast tensor files.

    Args:
        cfg: Hydra configuration
        data_dir: Root directory containing group folders
    """

    def __init__(self, cfg: DictConfig, data_dir: str | Path):
        self.cfg = cfg
        self.data_dir = Path(data_dir)

        # Cache directory for preprocessed data
        cache_dir = cfg.data.get("caching", {}).get("cache_dir", "preprocessed_cache")
        self.cache_dir = self.data_dir / cache_dir
        self.cache_key = get_cache_key(cfg)

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

        self._label_map: dict[str, int] = {}
        self._n_channels: int | None = None

    def setup(self, stage: Literal["fit", "test", "predict"] = "fit") -> None:
        """
        Load and prepare data for the specified stage.

        Args:
            stage: One of "fit", "test", or "predict"
        """
        if stage == "fit":
            self._setup_cached_datasets()
        elif stage == "test":
            if self.test_dataset is None:
                self._setup_cached_datasets()
        elif stage == "predict":
            self._setup_cached_datasets()

    def _preprocess_and_cache_file(
        self,
        fif_file: Path,
        cache_path: Path
    ) -> bool:
        """
        Preprocess a single file and save to cache.

        Returns True if successful, False otherwise.
        """
        try:
            raw = load_eeg_file(
                fif_file,
                filter_low=self.cfg.data.preprocessing.filter_low,
                filter_high=self.cfg.data.preprocessing.filter_high,
                reference=self.cfg.data.preprocessing.reference,
                notch_freq=self.cfg.data.preprocessing.notch_freq,
            )

            chunks, mask, info = prepare_phase_chunks(
                raw,
                chunk_duration=self.cfg.data.preprocessing.chunk_duration,
                chunk_overlap=self.cfg.data.preprocessing.chunk_overlap,
                include_amplitude=self.cfg.model.phase.include_amplitude,
            )

            # Store n_channels for later
            if self._n_channels is None:
                self._n_channels = info["n_channels"]

            # Save as tensors (fast to load)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "chunks": torch.from_numpy(chunks.astype(np.float32)),
                "masks": torch.from_numpy(mask),
                "info": info,
            }, cache_path)

            del raw
            return True

        except Exception as e:
            logger.warning(f"Failed to preprocess {fif_file.name}: {e}")
            return False

    def _setup_cached_datasets(self) -> None:
        """
        Set up datasets with disk caching.

        Strategy:
        1. Discover all subjects and their files
        2. Split subjects into train/val/test
        3. For each split, preprocess files (if not cached) and create dataset
        """
        # Step 1: Discover all subjects and their files
        subject_files: dict[str, list[Path]] = {}
        subject_labels: dict[str, int] = {}

        for group_idx, group in enumerate(self.cfg.data.groups):
            group_name = group.name
            group_path = self.data_dir / group.path
            self._label_map[group_name] = group_idx

            if not group_path.exists():
                logger.warning(f"Group path does not exist: {group_path}")
                continue

            subject_dirs = sorted([d for d in group_path.iterdir() if d.is_dir()])

            # Sampling
            n_subjects = self.cfg.data.sampling.n_subjects_per_group
            if n_subjects is not None and n_subjects < len(subject_dirs):
                rng = np.random.RandomState(self.cfg.data.sampling.random_seed)
                indices = rng.permutation(len(subject_dirs))[:n_subjects]
                subject_dirs = [subject_dirs[i] for i in sorted(indices)]
            elif n_subjects is not None:
                subject_dirs = subject_dirs[:n_subjects]

            for subject_dir in subject_dirs:
                subject_id = subject_dir.name
                fif_files = [
                    f for f in subject_dir.rglob("*_eeg.fif")
                    if not f.name.startswith("._")
                ]
                if fif_files:
                    subject_files[subject_id] = fif_files
                    subject_labels[subject_id] = group_idx

        all_subjects = list(subject_files.keys())
        total_files = sum(len(files) for files in subject_files.values())
        logger.info(f"Found {len(all_subjects)} subjects with {total_files} EEG files")

        # Step 2: Split subjects
        stratify = [subject_labels[s] for s in all_subjects]

        train_subjects, test_subjects = train_test_split(
            all_subjects,
            test_size=self.cfg.data.splitting.test_size,
            random_state=self.cfg.data.sampling.random_seed,
            stratify=stratify,
        )

        train_stratify = [subject_labels[s] for s in train_subjects]
        train_subjects, val_subjects = train_test_split(
            train_subjects,
            test_size=0.15,
            random_state=self.cfg.data.sampling.random_seed,
            stratify=train_stratify,
        )

        logger.info(
            f"Split: {len(train_subjects)} train, {len(val_subjects)} val, "
            f"{len(test_subjects)} test subjects"
        )

        # Step 3: Preprocess and cache files, then create datasets
        def get_cache_path(fif_file: Path) -> Path:
            """Get cache path for a .fif file."""
            # Use relative path from data_dir to create unique cache path
            rel_path = fif_file.relative_to(self.data_dir)
            cache_name = f"{rel_path.stem}_{self.cache_key}.pt"
            return self.cache_dir / rel_path.parent / cache_name

        def build_dataset(subjects: list[str], desc: str) -> Dataset:
            """Build dataset for a set of subjects, using cache."""
            datasets = []
            n_cached = 0
            n_processed = 0
            n_failed = 0

            # Collect all files for these subjects
            all_files = []
            for subj in subjects:
                for fif_file in subject_files[subj]:
                    all_files.append((fif_file, subj, subject_labels[subj]))

            with tqdm(all_files, desc=f"Preparing {desc}", unit="file") as pbar:
                for fif_file, subject_id, label in pbar:
                    cache_path = get_cache_path(fif_file)
                    pbar.set_postfix(subj=subject_id[:10])

                    # Check if cached
                    if cache_path.exists():
                        n_cached += 1
                        # Extract n_channels from cached file if not yet set
                        if self._n_channels is None:
                            try:
                                cached_data = torch.load(cache_path, weights_only=False)
                                if "info" in cached_data and "n_channels" in cached_data["info"]:
                                    self._n_channels = cached_data["info"]["n_channels"]
                            except Exception:
                                pass  # Will be set during preprocessing if needed
                    else:
                        # Preprocess and cache
                        if self._preprocess_and_cache_file(fif_file, cache_path):
                            n_processed += 1
                        else:
                            n_failed += 1
                            continue

                    # Create dataset from cached file
                    try:
                        ds = CachedFileDataset(cache_path, label, subject_id)
                        datasets.append(ds)
                    except Exception as e:
                        logger.warning(f"Failed to load cache {cache_path}: {e}")
                        n_failed += 1

            logger.info(
                f"{desc}: {n_cached} cached, {n_processed} processed, "
                f"{n_failed} failed, {sum(len(d) for d in datasets)} total chunks"
            )

            if not datasets:
                raise RuntimeError(f"No data loaded for {desc}!")

            return ConcatDataset(datasets)

        self.train_dataset = build_dataset(train_subjects, "train")
        self.val_dataset = build_dataset(val_subjects, "val")
        self.test_dataset = build_dataset(test_subjects, "test")

        logger.info(
            f"Total: {len(self.train_dataset)} train, "
            f"{len(self.val_dataset)} val, {len(self.test_dataset)} test chunks"
        )

    def train_dataloader(self, batch_size: int | None = None) -> DataLoader:
        """Get training DataLoader."""
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') first")

        bs = batch_size or self.cfg.training.batch_size
        return DataLoader(
            self.train_dataset,
            batch_size=bs,
            shuffle=True,
            num_workers=4,  # Can use workers now - loading cached tensors is fast
            pin_memory=True,
            persistent_workers=True,
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
            persistent_workers=True,
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
            persistent_workers=True,
        )

    @property
    def n_channels(self) -> int:
        """Number of EEG channels."""
        if self._n_channels is not None:
            return self._n_channels
        raise RuntimeError("Call setup() first to determine n_channels")

    @property
    def label_map(self) -> dict[str, int]:
        """Mapping from group names to label indices."""
        return self._label_map
