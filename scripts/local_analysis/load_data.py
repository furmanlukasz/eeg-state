"""
Data Loading Utilities for Local Analysis

Handles loading .fif files and extracting phase data.
"""

from pathlib import Path
import numpy as np
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def load_eeg_from_file(file_path: Path, verbose: bool = True, apply_preprocessing: bool = True):
    """
    Load EEG data from file (FIF or BDF format).

    Dataset-specific preprocessing is applied:
    - BDF files (meditation): No re-referencing, 2-48 Hz filter
    - FIF files (greek): No re-referencing in this function (done separately if needed)

    Args:
        file_path: Path to .fif or .bdf file
        verbose: Whether to print info
        apply_preprocessing: Whether to apply MNE preprocessing (reference, filter)

    Returns:
        Tuple of (raw_data, sfreq, channel_names)
        raw_data: (n_channels, n_samples) numpy array
    """
    import mne

    # Suppress MNE info messages
    mne.set_log_level("WARNING")

    # Detect file format
    suffix = file_path.suffix.lower()
    is_meditation = suffix == ".bdf"

    if suffix == ".fif":
        raw = mne.io.read_raw_fif(file_path, preload=True)
    elif suffix == ".bdf":
        raw = mne.io.read_raw_bdf(file_path, preload=True)
        # For BDF files, select only EEG channels (exclude GSR, respiration, etc.)
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        if len(eeg_picks) > 0:
            raw = raw.pick(eeg_picks)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    # Apply dataset-specific preprocessing
    if apply_preprocessing and is_meditation:
        # Meditation BDF: No re-referencing, 2-48 Hz filter (matches training)
        if verbose:
            print(f"  Preprocessing: no reference, 2-48 Hz filter")
        raw.filter(2.0, 48.0, verbose=False)

    if verbose:
        print(f"Loaded: {file_path.name}")
        print(f"  Channels: {len(raw.ch_names)}")
        print(f"  Duration: {raw.times[-1]:.1f}s")
        print(f"  Sfreq: {raw.info['sfreq']} Hz")

    return raw.get_data(), raw.info["sfreq"], raw.ch_names


# Legacy alias for backwards compatibility
def load_eeg_from_fif(fif_path: Path, verbose: bool = True):
    """Legacy wrapper - use load_eeg_from_file instead."""
    return load_eeg_from_file(fif_path, verbose)


def extract_phase_circular(
    data: np.ndarray,
    sfreq: float,
    filter_low: float = 1.0,
    filter_high: float = 30.0,
    include_amplitude: bool = True,
    skip_filter: bool = False,
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
        skip_filter: If True, skip bandpass filtering (use when data is pre-filtered)

    Returns:
        phase_data: (n_channels * phase_channels, n_samples) where phase_channels is 2 or 3
    """
    from scipy.signal import hilbert, butter, filtfilt

    n_channels, n_samples = data.shape

    if skip_filter:
        # Data already filtered, use as-is
        filtered = data
    else:
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


def load_and_preprocess_file(
    file_path: Path,
    filter_low: float = None,
    filter_high: float = None,
    chunk_duration: float = 5.0,
    include_amplitude: bool = True,
    verbose: bool = True,
):
    """
    Load EEG file (FIF or BDF) and extract phase chunks ready for model.

    Dataset-specific defaults:
    - BDF files (meditation): 2-48 Hz filter, no reference (applied in load_eeg_from_file)
    - FIF files (greek): 1-30 Hz filter (legacy default)

    Args:
        file_path: Path to .fif or .bdf file
        filter_low: Bandpass low cutoff (None = dataset-specific default)
        filter_high: Bandpass high cutoff (None = dataset-specific default)
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
    # Detect dataset type
    is_meditation = file_path.suffix.lower() == ".bdf"

    # Dataset-specific filter defaults
    if filter_low is None:
        filter_low = 2.0 if is_meditation else 1.0
    if filter_high is None:
        filter_high = 48.0 if is_meditation else 30.0

    # Load raw data (MNE preprocessing applied for meditation in load_eeg_from_file)
    # For meditation, filtering is done in MNE, so we skip scipy filtering in extract_phase
    raw_data, sfreq, channel_names = load_eeg_from_file(file_path, verbose, apply_preprocessing=is_meditation)
    n_channels = len(channel_names)

    # Extract phase
    # For meditation: data already filtered by MNE (2-48 Hz), skip scipy filtering
    # For Greek: apply scipy filter here
    if is_meditation:
        if verbose:
            print(f"  Extracting phase (data pre-filtered 2-48 Hz)...")
        phase_data = extract_phase_circular(
            raw_data, sfreq, include_amplitude=include_amplitude, skip_filter=True
        )
    else:
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
        if len(chunks) > 0:
            print(f"  Phase shape per chunk: {chunks[0].shape}")

    # Extract subject ID (handle both formats)
    # BIDS: sub-001 from path; Greek: i002 from folder name
    subject_id = _extract_subject_id(file_path)

    return {
        "chunks": chunks,
        "n_channels": n_channels,
        "sfreq": sfreq,
        "channel_names": channel_names,
        "subject_id": subject_id,
        "file_path": file_path,
    }


def _extract_subject_id(file_path: Path) -> str:
    """Extract subject ID from file path (handles both BIDS and Greek formats)."""
    # Try BIDS format first: look for sub-XXX in path
    for parent in file_path.parents:
        if parent.name.startswith("sub-"):
            return parent.name

    # Greek format: extract from parent folder name
    folder_name = file_path.parent.name
    if " " in folder_name:
        return folder_name.split()[0]
    elif "_" in folder_name:
        return folder_name.split("_")[0]
    return folder_name


# Legacy alias for backwards compatibility
def load_and_preprocess_fif(
    fif_path: Path,
    filter_low: float = 1.0,
    filter_high: float = 30.0,
    chunk_duration: float = 5.0,
    include_amplitude: bool = True,
    verbose: bool = True,
):
    """Legacy wrapper - use load_and_preprocess_file instead."""
    return load_and_preprocess_file(
        fif_path, filter_low, filter_high, chunk_duration, include_amplitude, verbose
    )


def load_model_and_compute_trajectories(
    checkpoint_path: Path,
    device: str = "mps",
    return_amplitudes: bool = False,
    max_subjects_per_group: int = None,
):
    """
    Load model and compute latent trajectories (and optionally amplitudes) for all subjects.

    This is the main entry point for amplitude ablation analysis.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device for inference
        return_amplitudes: If True, also return per-timepoint amplitude values
        max_subjects_per_group: Limit subjects per group (for testing)

    Returns:
        If return_amplitudes:
            (trajectories_by_group, amplitudes_by_group)
        Else:
            trajectories_by_group

        Where each is dict[group_name -> list[np.ndarray]]
    """
    import torch
    import ast
    from tqdm import tqdm
    from config import (
        DATASET, DATA_PATHS, get_dataset_config, get_data_files_via_config,
        get_fif_files, CHUNK_DURATION,
    )

    # Dynamically import model
    try:
        from eeg_biomarkers.models import TransformerAutoencoder
    except ImportError:
        print("Warning: eeg_biomarkers not installed, trying legacy import")
        from scripts.utils import ConvLSTMEEGAutoencoder as TransformerAutoencoder

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config
    config = checkpoint.get("config", {})
    model_config_raw = config.get("model", {})
    if isinstance(model_config_raw, str):
        model_config = ast.literal_eval(model_config_raw)
    else:
        model_config = model_config_raw

    encoder_config = model_config.get("encoder", {})
    phase_config = model_config.get("phase", {})

    # Determine model parameters
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    include_amplitude = phase_config.get("include_amplitude", True)
    phase_channels = 3 if include_amplitude else 2

    # Infer n_channels from conv layer
    n_channels = 79  # Default
    for key in state_dict:
        if "conv_layers.0.0.weight" in key:
            input_dim = state_dict[key].shape[1]
            n_channels = input_dim // phase_channels
            break

    print(f"  Model: n_channels={n_channels}, phase_channels={phase_channels}, "
          f"include_amplitude={include_amplitude}")

    # Create model
    model = TransformerAutoencoder(
        n_channels=n_channels,
        hidden_size=encoder_config.get("hidden_size", 64),
        n_heads=encoder_config.get("n_heads", 4),
        n_transformer_layers=encoder_config.get("n_transformer_layers", 2),
        dim_feedforward=encoder_config.get("dim_feedforward", 256),
        dropout=encoder_config.get("dropout", 0.1),
        phase_channels=phase_channels,
    )
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    # Get data files
    data_dir = DATA_PATHS.get(DATASET)
    dataset_config = get_dataset_config()

    if dataset_config is not None:
        data_files = []
        for group in dataset_config.groups:
            group_files = dataset_config.get_files_for_group(data_dir, group)
            for f in group_files:
                data_files.append((f, group.label, group.name))
    else:
        data_files = get_fif_files()

    print(f"  Found {len(data_files)} files")

    # Process each subject
    trajectories_by_group = {}
    amplitudes_by_group = {}
    subjects_per_group = {}

    for file_path, label, group_name in tqdm(data_files, desc="Processing subjects"):
        # Limit subjects per group if specified
        if max_subjects_per_group:
            if subjects_per_group.get(group_name, 0) >= max_subjects_per_group:
                continue

        try:
            # Load and preprocess
            result = load_and_preprocess_file(
                file_path,
                chunk_duration=CHUNK_DURATION,
                include_amplitude=include_amplitude,
                verbose=False,
            )

            if len(result["chunks"]) == 0:
                continue

            # Compute latent trajectories
            subject_latents = []
            subject_amplitudes = []

            with torch.no_grad():
                for chunk in result["chunks"]:
                    # Prepare input
                    x = torch.from_numpy(chunk).unsqueeze(0).to(device)
                    x = x.permute(0, 2, 1)  # (batch, time, features)

                    # Get latent
                    latent = model.encode(x)  # (batch, time, hidden)
                    subject_latents.append(latent.squeeze(0).cpu().numpy())

                    if return_amplitudes and include_amplitude:
                        # Extract amplitude from input features
                        # Features are [cos*C, sin*C, log_amp*C] = 3*C total
                        # Amplitude is last third
                        amp_start = 2 * n_channels
                        amp_features = chunk[amp_start:, :]  # (C, T)
                        mean_amp = np.mean(amp_features, axis=0)  # (T,)
                        subject_amplitudes.append(mean_amp)

            # Concatenate chunks into trajectory
            if subject_latents:
                trajectory = np.concatenate(subject_latents, axis=0)

                if group_name not in trajectories_by_group:
                    trajectories_by_group[group_name] = []
                trajectories_by_group[group_name].append(trajectory)

                if return_amplitudes and subject_amplitudes:
                    amplitude = np.concatenate(subject_amplitudes, axis=0)
                    if group_name not in amplitudes_by_group:
                        amplitudes_by_group[group_name] = []
                    amplitudes_by_group[group_name].append(amplitude)

                subjects_per_group[group_name] = subjects_per_group.get(group_name, 0) + 1

        except Exception as e:
            print(f"  Error processing {file_path.name}: {e}")
            continue

    if return_amplitudes:
        return trajectories_by_group, amplitudes_by_group
    return trajectories_by_group


if __name__ == "__main__":
    # Quick test with example file
    from config import DATA_DIR, FILTER_LOW, FILTER_HIGH, CHUNK_DURATION, DATASET

    # Find first available file (FIF or BDF)
    data_files = list(DATA_DIR.rglob("*.fif")) + list(DATA_DIR.rglob("*.bdf"))
    if data_files:
        print(f"Testing with {DATASET} dataset...")
        result = load_and_preprocess_file(
            data_files[0],
            FILTER_LOW,
            FILTER_HIGH,
            CHUNK_DURATION,
            include_amplitude=True,
        )
        print(f"\nLoaded {result['subject_id']}: {len(result['chunks'])} chunks")
    else:
        print(f"No .fif or .bdf files found in {DATA_DIR}")
