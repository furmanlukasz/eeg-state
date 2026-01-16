"""PyTorch Dataset and DataModule for EEG data."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
import logging

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


class EEGChunkDataset(Dataset):
    """
    Lazy-loading PyTorch Dataset for EEG phase chunks.

    Loads and preprocesses files on-demand rather than pre-loading everything.
    Caches processed chunks in memory as they are accessed.

    Args:
        file_infos: List of (file_path, subject_id, label) tuples
        cfg: Hydra configuration for preprocessing
    """

    def __init__(
        self,
        file_infos: list[tuple[Path, str, int]],
        cfg: DictConfig,
    ):
        self.file_infos = file_infos
        self.cfg = cfg

        # Build index: map global chunk index -> (file_idx, local_chunk_idx)
        # We need to know how many chunks each file produces
        self._file_chunks: list[np.ndarray] = [None] * len(file_infos)  # Lazy cache
        self._file_masks: list[np.ndarray] = [None] * len(file_infos)
        self._chunk_counts: list[int] = []
        self._cumsum: list[int] = [0]

        # Pre-scan files to get chunk counts (fast - just reads headers)
        logger.info(f"Scanning {len(file_infos)} files for chunk counts...")
        for fif_file, _, _ in tqdm(file_infos, desc="Scanning files", unit="file"):
            try:
                # Quick scan: load raw just to get duration
                import mne
                raw = mne.io.read_raw_fif(fif_file, preload=False, verbose=False)
                n_samples = raw.n_times
                sfreq = raw.info["sfreq"]
                raw.close()

                # Calculate chunks
                chunk_samples = int(cfg.data.preprocessing.chunk_duration * sfreq)
                overlap_samples = int(cfg.data.preprocessing.chunk_overlap * sfreq)
                step = chunk_samples - overlap_samples
                n_chunks = int(np.ceil((n_samples - overlap_samples) / step))

                self._chunk_counts.append(n_chunks)
                self._cumsum.append(self._cumsum[-1] + n_chunks)
            except Exception as e:
                logger.warning(f"Failed to scan {fif_file.name}: {e}")
                self._chunk_counts.append(0)
                self._cumsum.append(self._cumsum[-1])

        self._total_chunks = self._cumsum[-1]
        logger.info(f"Total chunks: {self._total_chunks} from {len(file_infos)} files")

    def __len__(self) -> int:
        return self._total_chunks

    def _load_file(self, file_idx: int) -> None:
        """Load and preprocess a single file, caching the results."""
        if self._file_chunks[file_idx] is not None:
            return  # Already loaded

        fif_file, subject_id, label = self.file_infos[file_idx]

        try:
            raw = load_eeg_file(
                fif_file,
                filter_low=self.cfg.data.preprocessing.filter_low,
                filter_high=self.cfg.data.preprocessing.filter_high,
                reference=self.cfg.data.preprocessing.reference,
                notch_freq=self.cfg.data.preprocessing.notch_freq,
            )

            chunks, mask, _ = prepare_phase_chunks(
                raw,
                chunk_duration=self.cfg.data.preprocessing.chunk_duration,
                chunk_overlap=self.cfg.data.preprocessing.chunk_overlap,
                include_amplitude=self.cfg.model.phase.include_amplitude,
            )

            self._file_chunks[file_idx] = chunks.astype(np.float32)
            self._file_masks[file_idx] = mask
            del raw

        except Exception as e:
            logger.error(f"Failed to load {fif_file.name}: {e}")
            # Create empty placeholder
            self._file_chunks[file_idx] = np.zeros((0, 1, 1), dtype=np.float32)
            self._file_masks[file_idx] = np.zeros((0, 1), dtype=bool)

    def _global_to_local(self, global_idx: int) -> tuple[int, int]:
        """Convert global chunk index to (file_idx, local_chunk_idx)."""
        # Binary search for the file
        file_idx = np.searchsorted(self._cumsum[1:], global_idx, side='right')
        local_idx = global_idx - self._cumsum[file_idx]
        return file_idx, local_idx

    def __getitem__(self, idx: int) -> dict:
        file_idx, local_idx = self._global_to_local(idx)

        # Lazy load if needed
        self._load_file(file_idx)

        _, subject_id, label = self.file_infos[file_idx]

        chunks = self._file_chunks[file_idx]
        masks = self._file_masks[file_idx]

        # Handle edge case where file failed to load properly
        if local_idx >= len(chunks):
            # Return zeros - shouldn't happen often
            chunk_samples = int(self.cfg.data.preprocessing.chunk_duration * 250)  # Assume 250Hz
            n_features = 256 * (3 if self.cfg.model.phase.include_amplitude else 2)
            return {
                "data": torch.zeros(n_features, chunk_samples),
                "mask": torch.zeros(chunk_samples, dtype=torch.bool),
                "label": torch.tensor(label, dtype=torch.long),
                "subject_id": subject_id,
            }

        return {
            "data": torch.from_numpy(chunks[local_idx]).float(),
            "mask": torch.from_numpy(masks[local_idx]).bool(),
            "label": torch.tensor(label, dtype=torch.long),
            "subject_id": subject_id,
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

        self.train_dataset: EEGChunkDataset | None = None
        self.val_dataset: EEGChunkDataset | None = None
        self.test_dataset: EEGChunkDataset | None = None

        self._label_map: dict[str, int] = {}
        self._n_channels: int | None = None

    def setup(self, stage: Literal["fit", "test", "predict"] = "fit") -> None:
        """
        Load and prepare data for the specified stage.

        Args:
            stage: One of "fit", "test", or "predict"
        """
        if stage == "fit":
            self._setup_lazy_datasets()
        elif stage == "test":
            if self.test_dataset is None:
                self._setup_lazy_datasets()
        elif stage == "predict":
            self._setup_lazy_datasets()

    def _setup_lazy_datasets(self) -> None:
        """
        Set up lazy-loading datasets.

        Memory-efficient approach:
        1. Discover all subjects and their files (no data loading)
        2. Split subjects into train/val/test
        3. Create lazy-loading datasets for each split
        """
        # Step 1: Discover all subjects and their files (without loading data)
        subject_files: dict[str, list[Path]] = {}  # subject_id -> list of files
        subject_labels: dict[str, int] = {}  # subject_id -> group_idx

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
                if fif_files:
                    subject_files[subject_id] = fif_files
                    subject_labels[subject_id] = group_idx

        all_subjects = list(subject_files.keys())
        total_files = sum(len(files) for files in subject_files.values())
        logger.info(f"Found {len(all_subjects)} subjects with {total_files} EEG files")

        # Step 2: Split subjects BEFORE creating datasets
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

        # Step 3: Create file info lists for each split
        def get_file_infos(subjects: list[str]) -> list[tuple[Path, str, int]]:
            """Get (file_path, subject_id, label) tuples for subjects."""
            infos = []
            for subj in subjects:
                for fif_file in subject_files[subj]:
                    infos.append((fif_file, subj, subject_labels[subj]))
            return infos

        train_infos = get_file_infos(train_subjects)
        val_infos = get_file_infos(val_subjects)
        test_infos = get_file_infos(test_subjects)

        logger.info(
            f"Files per split: {len(train_infos)} train, {len(val_infos)} val, "
            f"{len(test_infos)} test"
        )

        # Step 4: Create lazy-loading datasets
        self.train_dataset = EEGChunkDataset(train_infos, self.cfg)
        self.val_dataset = EEGChunkDataset(val_infos, self.cfg)
        self.test_dataset = EEGChunkDataset(test_infos, self.cfg)

        logger.info(
            f"Total chunks: {len(self.train_dataset)} train, "
            f"{len(self.val_dataset)} val, {len(self.test_dataset)} test"
        )

        # Get n_channels from first file
        if train_infos:
            import mne
            raw = mne.io.read_raw_fif(train_infos[0][0], preload=False, verbose=False)
            self._n_channels = len(raw.ch_names)
            raw.close()

    def train_dataloader(self, batch_size: int | None = None) -> DataLoader:
        """Get training DataLoader."""
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') first")

        bs = batch_size or self.cfg.training.batch_size
        return DataLoader(
            self.train_dataset,
            batch_size=bs,
            shuffle=True,
            num_workers=0,  # Must be 0 for lazy loading with caching
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
            num_workers=0,  # Must be 0 for lazy loading with caching
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
            num_workers=0,  # Must be 0 for lazy loading with caching
            pin_memory=True,
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
