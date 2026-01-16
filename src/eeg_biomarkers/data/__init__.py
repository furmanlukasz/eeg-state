"""Data loading and preprocessing for EEG signals."""

from eeg_biomarkers.data.dataset import EEGDataset, EEGDataModule
from eeg_biomarkers.data.preprocessing import (
    extract_phase_circular,
    chunk_signal,
    load_eeg_file,
)

__all__ = [
    "EEGDataset",
    "EEGDataModule",
    "extract_phase_circular",
    "chunk_signal",
    "load_eeg_file",
]
