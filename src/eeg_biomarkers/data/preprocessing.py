"""EEG preprocessing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import mne
from scipy.signal import hilbert


def preprocess_raw(
    raw: mne.io.Raw,
    filter_low: float = 3.0,
    filter_high: float = 48.0,
    reference: Literal["csd", "average"] = "csd",
    notch_freq: float | None = None,
    resample_freq: float | None = None,
    verbose: bool = False,
) -> mne.io.Raw:
    """
    Preprocess an already-loaded MNE Raw object.

    Args:
        raw: MNE Raw object (already loaded with preload=True)
        filter_low: Low cutoff frequency (Hz)
        filter_high: High cutoff frequency (Hz)
        reference: Referencing method ("csd" or "average")
        notch_freq: Notch filter frequency for line noise (50/60 Hz), or None
        resample_freq: Target sampling rate, or None to keep original
        verbose: Whether to print MNE output

    Returns:
        Preprocessed Raw object
    """
    # Resample if requested (do this first to speed up filtering)
    if resample_freq is not None and raw.info["sfreq"] != resample_freq:
        raw.resample(resample_freq, verbose=verbose)

    # Apply notch filter if specified
    if notch_freq is not None:
        raw.notch_filter(notch_freq, verbose=verbose)

    # Band-pass filter
    raw.filter(filter_low, filter_high, verbose=verbose)

    # Apply referencing
    if reference == "csd":
        raw = mne.preprocessing.compute_current_source_density(raw, verbose=verbose)
    elif reference == "average":
        raw.set_eeg_reference("average", verbose=verbose)

    return raw


def load_eeg_file(
    filepath: str | Path,
    filter_low: float = 3.0,
    filter_high: float = 48.0,
    reference: Literal["csd", "average"] = "csd",
    notch_freq: float | None = None,
    verbose: bool = False,
) -> mne.io.Raw:
    """
    Load and preprocess an EEG file (FIF format).

    This is a convenience function for loading .fif files. For other formats
    or more control, use preprocess_raw() directly.

    Args:
        filepath: Path to .fif file
        filter_low: Low cutoff frequency (Hz)
        filter_high: High cutoff frequency (Hz)
        reference: Referencing method ("csd" or "average")
        notch_freq: Notch filter frequency for line noise (50/60 Hz), or None
        verbose: Whether to print MNE output

    Returns:
        Preprocessed Raw object
    """
    # Load raw data
    raw = mne.io.read_raw_fif(filepath, preload=True, verbose=verbose)

    return preprocess_raw(
        raw,
        filter_low=filter_low,
        filter_high=filter_high,
        reference=reference,
        notch_freq=notch_freq,
        verbose=verbose,
    )


def extract_phase_circular(
    signal: np.ndarray,
    include_amplitude: bool = False,
    normalize_amplitude: bool = True,
) -> np.ndarray:
    """
    Extract phase using Hilbert transform with circular (cos, sin) representation.

    This avoids the mathematical issues with raw phase angles where
    values near ±π appear maximally different but are actually close.

    Args:
        signal: EEG signal array (n_channels, n_samples)
        include_amplitude: Whether to include log-amplitude as third channel
        normalize_amplitude: Whether to z-score normalize log-amplitude per channel
                            (recommended to prevent amplitude dominating the loss)

    Returns:
        Phase representation:
        - If include_amplitude=False: (n_channels, 2, n_samples) for cos, sin
        - If include_amplitude=True: (n_channels, 3, n_samples) for cos, sin, log_amp
    """
    n_channels, n_samples = signal.shape

    # Compute analytic signal via Hilbert transform
    analytic = hilbert(signal, axis=1)

    # Extract phase
    phase = np.angle(analytic)

    # Convert to circular representation
    cos_phase = np.cos(phase)
    sin_phase = np.sin(phase)

    if include_amplitude:
        # Log amplitude for numerical stability
        amplitude = np.abs(analytic)
        log_amplitude = np.log(amplitude + 1e-8)

        if normalize_amplitude:
            # Z-score normalize per channel to prevent amplitude dominating MSE loss
            # This ensures cos/sin and amplitude contribute roughly equally
            mean_amp = log_amplitude.mean(axis=1, keepdims=True)
            std_amp = log_amplitude.std(axis=1, keepdims=True) + 1e-8
            log_amplitude = (log_amplitude - mean_amp) / std_amp

        # Stack: (n_channels, 3, n_samples)
        result = np.stack([cos_phase, sin_phase, log_amplitude], axis=1)
    else:
        # Stack: (n_channels, 2, n_samples)
        result = np.stack([cos_phase, sin_phase], axis=1)

    return result


def chunk_signal(
    signal: np.ndarray,
    chunk_samples: int,
    overlap_samples: int = 0,
    pad_last: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split signal into fixed-duration chunks.

    Args:
        signal: Signal array (..., n_samples)
        chunk_samples: Number of samples per chunk
        overlap_samples: Overlap between consecutive chunks
        pad_last: Whether to zero-pad the last chunk if incomplete

    Returns:
        chunks: Array of chunks (..., n_chunks, chunk_samples)
        mask: Boolean mask indicating valid (non-padded) samples
    """
    *leading_dims, n_samples = signal.shape

    # Calculate step size and number of chunks
    step = chunk_samples - overlap_samples
    if pad_last:
        n_chunks = int(np.ceil((n_samples - overlap_samples) / step))
    else:
        n_chunks = int(np.floor((n_samples - chunk_samples) / step)) + 1

    # Pre-allocate output
    chunks = np.zeros((*leading_dims, n_chunks, chunk_samples), dtype=signal.dtype)
    mask = np.zeros((n_chunks, chunk_samples), dtype=bool)

    for i in range(n_chunks):
        start = i * step
        end = start + chunk_samples
        actual_end = min(end, n_samples)
        valid_samples = actual_end - start

        chunks[..., i, :valid_samples] = signal[..., start:actual_end]
        mask[i, :valid_samples] = True

    return chunks, mask


def prepare_phase_chunks(
    raw: mne.io.Raw,
    chunk_duration: float = 5.0,
    chunk_overlap: float = 0.0,
    include_amplitude: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Prepare EEG data as phase chunks for model input.

    Combines phase extraction and chunking in correct order.

    Args:
        raw: Preprocessed MNE Raw object
        chunk_duration: Duration of each chunk in seconds
        chunk_overlap: Overlap between chunks in seconds
        include_amplitude: Whether to include amplitude channel

    Returns:
        chunks: Phase chunks (n_chunks, n_channels * phase_channels, chunk_samples)
        mask: Validity mask (n_chunks, chunk_samples)
        info: Dictionary with metadata
    """
    sfreq = raw.info["sfreq"]
    chunk_samples = int(chunk_duration * sfreq)
    overlap_samples = int(chunk_overlap * sfreq)

    # Get data
    data = raw.get_data()  # (n_channels, n_samples)
    n_channels = data.shape[0]

    # Extract phase with circular representation
    phase_data = extract_phase_circular(data, include_amplitude=include_amplitude)
    # Shape: (n_channels, phase_channels, n_samples)

    phase_channels = phase_data.shape[1]

    # Reshape to (n_channels * phase_channels, n_samples) for chunking
    phase_data = phase_data.reshape(n_channels * phase_channels, -1)

    # Chunk the data
    chunks, mask = chunk_signal(phase_data, chunk_samples, overlap_samples)
    # Shape: (n_channels * phase_channels, n_chunks, chunk_samples)

    # Transpose to (n_chunks, n_channels * phase_channels, chunk_samples)
    chunks = np.transpose(chunks, (1, 0, 2))

    info = {
        "n_channels": n_channels,
        "phase_channels": phase_channels,
        "sfreq": sfreq,
        "chunk_duration": chunk_duration,
        "chunk_samples": chunk_samples,
    }

    return chunks, mask, info
