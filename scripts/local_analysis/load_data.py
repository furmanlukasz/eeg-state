"""
Data Loading Utilities for Local Analysis

Handles loading .fif files and extracting phase data.
"""

from pathlib import Path
import numpy as np
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def load_eeg_from_fif(fif_path: Path, verbose: bool = True):
    """
    Load EEG data from .fif file.

    Args:
        fif_path: Path to .fif file
        verbose: Whether to print info

    Returns:
        Tuple of (raw_data, sfreq, channel_names)
        raw_data: (n_channels, n_samples) numpy array
    """
    import mne

    # Suppress MNE info messages
    mne.set_log_level("WARNING")

    raw = mne.io.read_raw_fif(fif_path, preload=True)

    if verbose:
        print(f"Loaded: {fif_path.name}")
        print(f"  Channels: {len(raw.ch_names)}")
        print(f"  Duration: {raw.times[-1]:.1f}s")
        print(f"  Sfreq: {raw.info['sfreq']} Hz")

    return raw.get_data(), raw.info["sfreq"], raw.ch_names


def extract_phase_circular(
    data: np.ndarray,
    sfreq: float,
    filter_low: float = 1.0,
    filter_high: float = 30.0,
    include_amplitude: bool = True,
) -> np.ndarray:
    """
    Extract circular phase representation (cos, sin) and optionally amplitude.

    This is the CORRECT phase representation that avoids wraparound issues.

    Args:
        data: (n_channels, n_samples) raw EEG data
        sfreq: Sampling frequency
        filter_low: Low cutoff for bandpass filter
        filter_high: High cutoff for bandpass filter
        include_amplitude: Whether to include log-amplitude as third channel

    Returns:
        phase_data: (n_channels * phase_channels, n_samples) where phase_channels is 2 or 3
    """
    from scipy.signal import hilbert, butter, filtfilt

    n_channels, n_samples = data.shape

    # Bandpass filter
    nyq = sfreq / 2
    low = filter_low / nyq
    high = filter_high / nyq
    b, a = butter(4, [low, high], btype="band")

    filtered = filtfilt(b, a, data, axis=1)

    # Hilbert transform for analytic signal
    analytic = hilbert(filtered, axis=1)

    # Extract phase and amplitude
    phase = np.angle(analytic)  # [-pi, pi]
    amplitude = np.abs(analytic)

    # Circular representation: (cos(phase), sin(phase))
    cos_phase = np.cos(phase)
    sin_phase = np.sin(phase)

    if include_amplitude:
        # Log-amplitude (more stable for neural networks)
        log_amplitude = np.log1p(amplitude)
        # Stack: (n_channels*3, n_samples)
        phase_data = np.vstack([cos_phase, sin_phase, log_amplitude])
    else:
        # Stack: (n_channels*2, n_samples)
        phase_data = np.vstack([cos_phase, sin_phase])

    return phase_data.astype(np.float32)


def chunk_data(data: np.ndarray, chunk_samples: int, overlap: float = 0.0):
    """
    Split data into chunks.

    Args:
        data: (n_features, n_samples) array
        chunk_samples: Number of samples per chunk
        overlap: Overlap fraction (0.0 = no overlap, 0.5 = 50% overlap)

    Returns:
        List of (n_features, chunk_samples) arrays
    """
    n_features, n_samples = data.shape
    step = int(chunk_samples * (1 - overlap))

    chunks = []
    for start in range(0, n_samples - chunk_samples + 1, step):
        end = start + chunk_samples
        chunks.append(data[:, start:end])

    return chunks


def load_and_preprocess_fif(
    fif_path: Path,
    filter_low: float = 1.0,
    filter_high: float = 30.0,
    chunk_duration: float = 5.0,
    include_amplitude: bool = True,
    verbose: bool = True,
):
    """
    Load .fif file and extract phase chunks ready for model.

    Args:
        fif_path: Path to .fif file
        filter_low: Bandpass low cutoff
        filter_high: Bandpass high cutoff
        chunk_duration: Chunk duration in seconds
        include_amplitude: Include amplitude in phase representation
        verbose: Print progress info

    Returns:
        Dict with:
            - chunks: List of (n_features, n_samples) arrays
            - n_channels: Number of EEG channels
            - sfreq: Sampling frequency
            - channel_names: List of channel names
            - subject_id: Extracted subject ID
    """
    # Load raw data
    raw_data, sfreq, channel_names = load_eeg_from_fif(fif_path, verbose)
    n_channels = len(channel_names)

    # Extract phase
    if verbose:
        print(f"  Extracting phase ({filter_low}-{filter_high} Hz)...")
    phase_data = extract_phase_circular(
        raw_data, sfreq, filter_low, filter_high, include_amplitude
    )

    # Chunk
    chunk_samples = int(chunk_duration * sfreq)
    chunks = chunk_data(phase_data, chunk_samples)

    if verbose:
        print(f"  Created {len(chunks)} chunks of {chunk_duration}s")
        print(f"  Phase shape per chunk: {chunks[0].shape}")

    # Extract subject ID
    subject_id = fif_path.parent.name.split()[0]

    return {
        "chunks": chunks,
        "n_channels": n_channels,
        "sfreq": sfreq,
        "channel_names": channel_names,
        "subject_id": subject_id,
        "fif_path": fif_path,
    }


if __name__ == "__main__":
    # Quick test with example file
    from config import DATA_DIR, FILTER_LOW, FILTER_HIGH, CHUNK_DURATION

    # Find first available .fif file
    fif_files = list(DATA_DIR.rglob("*.fif"))
    if fif_files:
        result = load_and_preprocess_fif(
            fif_files[0],
            FILTER_LOW,
            FILTER_HIGH,
            CHUNK_DURATION,
            include_amplitude=True,
        )
        print(f"\nLoaded {result['subject_id']}: {len(result['chunks'])} chunks")
    else:
        print(f"No .fif files found in {DATA_DIR}")
