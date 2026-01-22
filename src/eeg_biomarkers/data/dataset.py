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
    preprocess_raw,
    prepare_phase_chunks,
)
from eeg_biomarkers.data.dataset_config import (
    DatasetConfig,
    dataset_from_hydra_config,
)

logger = logging.getLogger(__name__)


def get_cache_key(cfg: DictConfig) -> str:
    """Generate a unique cache key based on dataset and preprocessing config."""
    dataset_name = cfg.data.get("dataset", "greek_resting")
    key_parts = [
        f"ds_{dataset_name}",
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
    Uses lazy loading to avoid loading all data into memory at once.
    """

    def __init__(self, cache_path: Path, label: int, subject_id: str, lazy: bool = True):
        self.cache_path = cache_path
        self.label = label
        self.subject_id = subject_id
        self.lazy = lazy

        if lazy:
            # Only load metadata - actual data loaded on demand
            data = torch.load(cache_path, weights_only=True)
            self._len = len(data["chunks"])
            self._chunks = None
            self._masks = None
            del data  # Free memory immediately
        else:
            # Load cached data fully (original behavior)
            data = torch.load(cache_path, weights_only=True)
            self._chunks = data["chunks"]  # (n_chunks, features, time)
            self._masks = data["masks"]    # (n_chunks, time)
            self._len = len(self._chunks)

    def _load_data(self) -> None:
        """Lazily load data when first accessed."""
        if self._chunks is None:
            data = torch.load(self.cache_path, weights_only=True)
            self._chunks = data["chunks"]
            self._masks = data["masks"]

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict:
        if self.lazy and self._chunks is None:
            self._load_data()
        return {
            "data": self._chunks[idx],
            "mask": self._masks[idx],
            "label": torch.tensor(self.label, dtype=torch.long),
            "subject_id": self.subject_id,
        }


class MemoryEfficientConcatDataset(Dataset):
    """
    Memory-efficient dataset that loads files on-demand or preloads all.

    Modes:
    - preload=True: Load all data into memory at init (fast training, high memory)
    - preload=False: Load files on-demand with LRU cache (slow training, low memory)

    For datasets where total size < available RAM, preload=True is recommended.
    """

    def __init__(self, file_infos: list[tuple[Path, int, str]], preload: bool = True):
        """
        Args:
            file_infos: List of (cache_path, label, subject_id) tuples
            preload: If True, load all data into memory at init (recommended)
        """
        self.file_infos = file_infos
        self.preload = preload

        # Build index: (file_idx, chunk_idx) for each sample
        self._index: list[tuple[int, int]] = []
        self._file_lengths: list[int] = []

        if preload:
            # Preload all files into memory - slow init but fast training
            logger.info(f"Preloading {len(file_infos)} files into memory...")
            self._all_chunks = []
            self._all_masks = []
            self._all_labels = []
            self._all_subject_ids = []

            for file_idx, (cache_path, label, subject_id) in enumerate(
                tqdm(file_infos, desc="Loading data", unit="file")
            ):
                try:
                    data = torch.load(cache_path, weights_only=True)
                    n_chunks = len(data["chunks"])
                    self._file_lengths.append(n_chunks)

                    # Store all chunks from this file
                    for chunk_idx in range(n_chunks):
                        self._all_chunks.append(data["chunks"][chunk_idx])
                        self._all_masks.append(data["masks"][chunk_idx])
                        self._all_labels.append(label)
                        self._all_subject_ids.append(subject_id)
                        self._index.append((file_idx, chunk_idx))

                    del data
                except Exception as e:
                    logger.warning(f"Failed to load {cache_path}: {e}")
                    self._file_lengths.append(0)

            # Stack into tensors for faster access
            logger.info("Stacking tensors...")
            self._all_chunks = torch.stack(self._all_chunks)
            self._all_masks = torch.stack(self._all_masks)
            self._all_labels = torch.tensor(self._all_labels, dtype=torch.long)

            logger.info(
                f"Loaded {len(self._index)} chunks, "
                f"shape: {self._all_chunks.shape}, "
                f"memory: {self._all_chunks.nbytes / 1024**3:.1f} GB"
            )
        else:
            # On-demand loading with LRU cache
            logger.info(f"Building index for {len(file_infos)} files (on-demand loading)...")
            for file_idx, (cache_path, label, subject_id) in enumerate(file_infos):
                try:
                    data = torch.load(cache_path, weights_only=True)
                    n_chunks = len(data["chunks"])
                    self._file_lengths.append(n_chunks)
                    for chunk_idx in range(n_chunks):
                        self._index.append((file_idx, chunk_idx))
                    del data
                except Exception as e:
                    logger.warning(f"Failed to index {cache_path}: {e}")
                    self._file_lengths.append(0)

            logger.info(f"Indexed {len(self._index)} total chunks from {len(file_infos)} files")

            # LRU cache
            self._cache: dict[int, dict] = {}
            self._cache_order: list[int] = []
            self._max_cache_size = min(len(file_infos), 15)
            logger.info(f"LRU cache size: {self._max_cache_size} files")

    def _get_file_data(self, file_idx: int) -> dict:
        """Load file data with simple LRU caching (only used when preload=False)."""
        if file_idx in self._cache:
            return self._cache[file_idx]

        cache_path, _, _ = self.file_infos[file_idx]
        data = torch.load(cache_path, weights_only=True)

        self._cache[file_idx] = data
        self._cache_order.append(file_idx)

        while len(self._cache_order) > self._max_cache_size:
            old_idx = self._cache_order.pop(0)
            if old_idx in self._cache:
                del self._cache[old_idx]

        return data

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        if self.preload:
            return {
                "data": self._all_chunks[idx],
                "mask": self._all_masks[idx],
                "label": self._all_labels[idx],
                "subject_id": self._all_subject_ids[idx],
            }
        else:
            file_idx, chunk_idx = self._index[idx]
            cache_path, label, subject_id = self.file_infos[file_idx]
            data = self._get_file_data(file_idx)
            return {
                "data": data["chunks"][chunk_idx],
                "mask": data["masks"][chunk_idx],
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
    - Loading data from multiple groups/conditions (via DatasetConfig)
    - Preprocessing and phase extraction with disk caching
    - Train/test splitting with proper subject-level separation
    - DataLoader creation

    The preprocessing cache ensures that expensive CSD/filtering operations
    are only done once per file. Subsequent runs load fast tensor files.

    Supports multiple datasets through the DatasetConfig abstraction:
    - Greek resting-state (HC/MCI/AD)
    - OpenNeuro meditation BIDS (expert/novice)
    - Custom datasets via register_dataset decorator

    Args:
        cfg: Hydra configuration
        data_dir: Root directory containing data
    """

    def __init__(self, cfg: DictConfig, data_dir: str | Path):
        self.cfg = cfg
        self.data_dir = Path(data_dir)

        # Load dataset configuration based on cfg.data.dataset
        self.dataset_config: DatasetConfig = dataset_from_hydra_config(cfg)
        logger.info(f"Using dataset config: {self.dataset_config.name}")

        # Validate dataset exists
        self.dataset_config.validate(self.data_dir)

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
        data_file: Path,
        cache_path: Path
    ) -> bool:
        """
        Preprocess a single file and save to cache.

        Uses the dataset config to load the raw data (supporting FIF, BDF, etc.)
        and then applies standard preprocessing pipeline.

        Returns True if successful, False otherwise.
        """
        try:
            # Load raw data using dataset-specific loader
            raw = self.dataset_config.load_raw(data_file)

            # Apply preprocessing (filtering, referencing)
            raw = preprocess_raw(
                raw,
                filter_low=self.cfg.data.preprocessing.filter_low,
                filter_high=self.cfg.data.preprocessing.filter_high,
                reference=self.cfg.data.preprocessing.reference,
                notch_freq=self.cfg.data.preprocessing.notch_freq,
                resample_freq=self.cfg.data.preprocessing.get("resample_freq"),
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
            logger.warning(f"Failed to preprocess {data_file.name}: {e}")
            return False

    def _setup_cached_datasets(self) -> None:
        """
        Set up datasets with disk caching.

        Strategy:
        1. Discover all subjects and their files (via DatasetConfig)
        2. Split subjects into train/val/test
        3. For each split, preprocess files (if not cached) and create dataset
        """
        # Step 1: Discover all subjects and their files using DatasetConfig
        subject_files: dict[str, list[Path]] = {}
        subject_labels: dict[str, int] = {}

        # Get label map from dataset config
        self._label_map = self.dataset_config.get_label_map()

        for group in self.dataset_config.groups:
            # Use DatasetConfig to discover files for this group
            group_files = self.dataset_config.get_files_for_group(self.data_dir, group)

            if not group_files:
                logger.warning(f"No files found for group {group.name}")
                continue

            # Group files by subject using DatasetConfig's subject ID extraction
            for data_file in group_files:
                subject_id = self.dataset_config.get_subject_id(data_file)

                if subject_id not in subject_files:
                    subject_files[subject_id] = []
                    subject_labels[subject_id] = group.label

                subject_files[subject_id].append(data_file)

            logger.info(f"Group {group.name}: {len(group_files)} files")

        # Apply sampling if requested
        n_subjects = self.cfg.data.sampling.n_subjects_per_group
        if n_subjects is not None:
            # Sample within each group
            sampled_subject_files = {}
            sampled_subject_labels = {}

            for group in self.dataset_config.groups:
                # Get subjects for this group
                group_subjects = [
                    s for s, label in subject_labels.items()
                    if label == group.label
                ]

                if n_subjects < len(group_subjects):
                    rng = np.random.RandomState(self.cfg.data.sampling.random_seed)
                    indices = rng.permutation(len(group_subjects))[:n_subjects]
                    group_subjects = [group_subjects[i] for i in sorted(indices)]

                for subj in group_subjects:
                    sampled_subject_files[subj] = subject_files[subj]
                    sampled_subject_labels[subj] = subject_labels[subj]

            subject_files = sampled_subject_files
            subject_labels = sampled_subject_labels

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
        def get_cache_path(data_file: Path) -> Path:
            """Get cache path for a data file (FIF, BDF, etc.)."""
            # Use relative path from data_dir to create unique cache path
            try:
                rel_path = data_file.relative_to(self.data_dir)
            except ValueError:
                # File is not under data_dir, use absolute path hash
                rel_path = Path(hashlib.md5(str(data_file).encode()).hexdigest()[:16])
            cache_name = f"{rel_path.stem}_{self.cache_key}.pt"
            return self.cache_dir / rel_path.parent / cache_name

        # Use preloading dataset for fast training on any GPU (MPS or CUDA)
        # This loads all data into RAM once, then training is fast
        # On CPU-only, use the older ConcatDataset approach
        use_preload = torch.backends.mps.is_available() or torch.cuda.is_available()
        if use_preload:
            device_name = "MPS" if torch.backends.mps.is_available() else "CUDA"
            logger.info(f"Using preloaded dataset for fast training ({device_name} detected)")

        def build_dataset(subjects: list[str], desc: str) -> Dataset:
            """Build dataset for a set of subjects, using cache."""
            file_infos: list[tuple[Path, int, str]] = []  # For memory-efficient mode
            datasets = []  # For standard mode
            n_cached = 0
            n_processed = 0
            n_failed = 0

            # Collect all files for these subjects
            all_files = []
            for subj in subjects:
                for data_file in subject_files[subj]:
                    all_files.append((data_file, subj, subject_labels[subj]))

            with tqdm(all_files, desc=f"Preparing {desc}", unit="file") as pbar:
                for data_file, subject_id, label in pbar:
                    cache_path = get_cache_path(data_file)
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
                                del cached_data
                            except Exception:
                                pass  # Will be set during preprocessing if needed
                    else:
                        # Preprocess and cache
                        if self._preprocess_and_cache_file(data_file, cache_path):
                            n_processed += 1
                        else:
                            n_failed += 1
                            continue

                    # Add to appropriate structure
                    if use_preload:
                        file_infos.append((cache_path, label, subject_id))
                    else:
                        try:
                            ds = CachedFileDataset(cache_path, label, subject_id)
                            datasets.append(ds)
                        except Exception as e:
                            logger.warning(f"Failed to load cache {cache_path}: {e}")
                            n_failed += 1

            logger.info(
                f"{desc}: {n_cached} cached, {n_processed} processed, {n_failed} failed"
            )

            if use_preload:
                if not file_infos:
                    raise RuntimeError(f"No data loaded for {desc}!")
                return MemoryEfficientConcatDataset(file_infos)
            else:
                if not datasets:
                    raise RuntimeError(f"No data loaded for {desc}!")
                logger.info(f"{desc}: {sum(len(d) for d in datasets)} total chunks")
                return ConcatDataset(datasets)

        self.train_dataset = build_dataset(train_subjects, "train")
        self.val_dataset = build_dataset(val_subjects, "val")
        self.test_dataset = build_dataset(test_subjects, "test")

        logger.info(
            f"Total: {len(self.train_dataset)} train, "
            f"{len(self.val_dataset)} val, {len(self.test_dataset)} test chunks"
        )

    def _get_dataloader_kwargs(self) -> dict:
        """Get optimal DataLoader kwargs based on device/platform."""
        # Check if MPS (Apple Silicon) - pin_memory not supported
        use_mps = torch.backends.mps.is_available()
        use_cuda = torch.cuda.is_available()

        if use_mps:
            # MPS: No pin_memory, fewer workers to avoid memory duplication
            return {
                "num_workers": 0,  # Main process only - avoids memory copies
                "pin_memory": False,
                "persistent_workers": False,
            }
        elif use_cuda:
            # CUDA: Full optimization
            return {
                "num_workers": 4,
                "pin_memory": True,
                "persistent_workers": True,
            }
        else:
            # CPU: Moderate workers
            return {
                "num_workers": 2,
                "pin_memory": False,
                "persistent_workers": True,
            }

    def train_dataloader(self, batch_size: int | None = None) -> DataLoader:
        """Get training DataLoader."""
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') first")

        bs = batch_size or self.cfg.training.batch_size
        kwargs = self._get_dataloader_kwargs()
        return DataLoader(
            self.train_dataset,
            batch_size=bs,
            shuffle=True,
            **kwargs,
        )

    def val_dataloader(self, batch_size: int | None = None) -> DataLoader:
        """Get validation DataLoader."""
        if self.val_dataset is None:
            raise RuntimeError("Call setup('fit') first")

        bs = batch_size or self.cfg.training.batch_size
        kwargs = self._get_dataloader_kwargs()
        return DataLoader(
            self.val_dataset,
            batch_size=bs,
            shuffle=False,
            **kwargs,
        )

    def test_dataloader(self, batch_size: int | None = None) -> DataLoader:
        """Get test DataLoader."""
        if self.test_dataset is None:
            raise RuntimeError("Call setup('test') first")

        bs = batch_size or self.cfg.training.batch_size
        kwargs = self._get_dataloader_kwargs()
        return DataLoader(
            self.test_dataset,
            batch_size=bs,
            shuffle=False,
            **kwargs,
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
