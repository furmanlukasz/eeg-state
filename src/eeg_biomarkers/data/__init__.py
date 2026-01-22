"""Data loading and preprocessing for EEG signals."""

from eeg_biomarkers.data.dataset import EEGDataset, EEGDataModule
from eeg_biomarkers.data.preprocessing import (
    extract_phase_circular,
    chunk_signal,
    load_eeg_file,
    preprocess_raw,
)
from eeg_biomarkers.data.dataset_config import (
    DatasetConfig,
    GroupConfig,
    PreprocessingConfig,
    GreekRestingConfig,
    MeditationBIDSConfig,
    get_dataset_config,
    register_dataset,
    list_datasets,
    dataset_from_hydra_config,
)

__all__ = [
    # Dataset classes
    "EEGDataset",
    "EEGDataModule",
    # Preprocessing
    "extract_phase_circular",
    "chunk_signal",
    "load_eeg_file",
    "preprocess_raw",
    # Dataset configuration
    "DatasetConfig",
    "GroupConfig",
    "PreprocessingConfig",
    "GreekRestingConfig",
    "MeditationBIDSConfig",
    "get_dataset_config",
    "register_dataset",
    "list_datasets",
    "dataset_from_hydra_config",
]
