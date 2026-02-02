#!/usr/bin/env python3
"""
Inverse Mapping Analysis: Latent Dynamics → Signal Space Interpretation

Ground abstract latent dynamics in familiar neurophysiological terms by mapping
latent space properties back to EEG signal characteristics.

This addresses the key interpretability question:
"What EEG phenomena correspond to high-occupancy regions, low-speed regimes,
or dominant flow directions in the latent space?"

Three main analyses:
1. Latent-region → EEG pattern correspondence (topography by occupancy)
2. Speed-conditioned EEG analysis (spectral properties by latent speed)
3. Flow direction → Signal transition mapping (spectral change along streamlines)

Usage:
    # Run all analyses on Greek dataset
    EEG_DATASET=greek_resting python scripts/local_analysis/inverse_mapping.py

    # Run specific analysis on meditation dataset
    EEG_DATASET=meditation_bids python scripts/local_analysis/inverse_mapping.py \
        --analysis speed_conditioned

    # Quick test with fewer subjects
    python scripts/local_analysis/inverse_mapping.py --quick --n-subjects 5

Key outputs:
    - Topographic maps by latent region (high/low occupancy, speed)
    - Spectral profiles conditioned on latent dynamics
    - Flow direction → spectral change correspondence
    - Statistical comparisons with bootstrap CIs
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from scipy.signal import welch
from scipy.stats import spearmanr, pearsonr
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from config import (
    CHECKPOINT_PATH, DATA_DIR, OUTPUT_DIR, DEVICE, DATASET,
    FILTER_LOW, FILTER_HIGH, CHUNK_DURATION, SFREQ,
    ensure_output_dir, get_subjects_by_group_unified, get_dataset_config,
    get_data_files_via_config
)
from load_model import load_model_from_checkpoint, create_model, compute_latent_trajectory
from load_data import load_eeg_from_file, extract_phase_circular, chunk_data
from velocity import compute_speed, VelocityConfig

# Optional MNE for topographic plots
try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False
    warnings.warn("MNE not available - topographic plots will be disabled")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Frequency bands for spectral analysis
FREQ_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "low_gamma": (30, 48),
}

# Speed quantile thresholds for conditioning
SPEED_QUANTILES = [0.25, 0.75]  # Low (bottom 25%), High (top 25%)

# Grid resolution for occupancy binning
OCCUPANCY_GRID_SIZE = 20

# Number of principal flow directions to analyze
N_FLOW_DIRECTIONS = 4


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SubjectDataWithRaw:
    """Subject data including both latent trajectory and raw EEG."""
    subject_id: str
    group: str
    label: int
    trajectory: np.ndarray          # (n_timepoints, latent_dim)
    raw_chunks: list[np.ndarray]    # List of (n_channels, chunk_samples)
    phase_chunks: list[np.ndarray]  # List of (n_features, chunk_samples)
    sfreq: float
    channel_names: list[str]


@dataclass
class SpectralProfile:
    """Spectral profile for a set of EEG segments."""
    psd: np.ndarray                 # (n_channels, n_freqs)
    freqs: np.ndarray               # (n_freqs,)
    band_power: dict[str, np.ndarray]  # band_name -> (n_channels,)
    global_band_power: dict[str, float]  # band_name -> scalar (mean across channels)


@dataclass
class RegionEEGProfile:
    """EEG characteristics for a latent space region."""
    region_name: str
    n_segments: int
    n_timepoints: int
    spectral_profile: SpectralProfile
    mean_gfp: float                 # Global Field Power
    std_gfp: float
    topography: dict[str, np.ndarray]  # band_name -> (n_channels,) power map


@dataclass
class InverseMappingResults:
    """Full results from inverse mapping analysis."""
    timestamp: str
    dataset: str
    n_subjects: dict[str, int]

    # Analysis 1: Region-based
    region_profiles: Optional[dict[str, dict[str, RegionEEGProfile]]] = None  # group -> region -> profile

    # Analysis 2: Speed-conditioned
    speed_profiles: Optional[dict[str, dict[str, SpectralProfile]]] = None  # group -> speed_bin -> profile
    speed_band_correlations: Optional[dict[str, dict[str, float]]] = None  # group -> band -> correlation

    # Analysis 3: Flow directions
    flow_spectral_changes: Optional[dict[str, np.ndarray]] = None  # direction -> delta_power per band


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_montage_for_dataset():
    """Get appropriate MNE montage for current dataset."""
    if not HAS_MNE:
        return None

    if DATASET == "meditation_bids":
        # BioSemi 64-channel
        return mne.channels.make_standard_montage("biosemi64")
    else:
        # EGI HydroCel 256-channel
        return mne.channels.make_standard_montage("GSN-HydroCel-256")


def get_biosemi_channel_mapping() -> dict[str, str]:
    """
    Get mapping from BDF channel names (A1-A32, B1-B32) to BioSemi64 standard names.

    The BioSemi64 system uses A1-A32 and B1-B32 internally, which map to standard
    10-20 positions. This provides the mapping.
    """
    # BioSemi64 standard channel order (A1-A32, B1-B32 -> 10-20 names)
    # Based on BioSemi documentation
    biosemi64_order = [
        'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7',
        'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7',
        'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9',
        'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz',
        'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4',
        'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz',
        'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2',
        'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2',
    ]

    mapping = {}
    for i in range(32):
        mapping[f'A{i+1}'] = biosemi64_order[i]
        mapping[f'B{i+1}'] = biosemi64_order[32 + i]

    return mapping


def get_group_config():
    """Get group colors and names based on current dataset."""
    if DATASET == "meditation_bids":
        return {
            "colors": {"Expert": "#1f77b4", "Novice": "#ff7f0e"},
            "names": {0: "Expert", 1: "Novice"},
            "keys": ["expert", "novice"],
            "display_names": ["Expert", "Novice"],
        }
    else:
        return {
            "colors": {"HC": "#1f77b4", "MCI": "#ff7f0e", "AD": "#d62728"},
            "names": {0: "HC", 1: "MCI", 2: "AD"},
            "keys": ["hc", "mci", "ad"],
            "display_names": ["HC", "MCI", "AD"],
        }


def create_timestamped_output_dir(base_dir: Path, script_name: str) -> Path:
    """Create a timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / f"{script_name}_{DATASET}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_parameters(output_dir: Path, params: dict):
    """Save parameters to JSON for reproducibility."""
    params_path = output_dir / "parameters.json"
    serializable = {k: str(v) if isinstance(v, Path) else v for k, v in params.items()}
    serializable["timestamp"] = datetime.now().isoformat()
    serializable["dataset"] = DATASET
    with open(params_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"Parameters saved to: {params_path}")


# =============================================================================
# SPECTRAL ANALYSIS
# =============================================================================

def compute_spectral_profile(
    raw_segments: list[np.ndarray],
    sfreq: float,
    freq_bands: dict[str, tuple[float, float]] = FREQ_BANDS,
) -> SpectralProfile:
    """
    Compute spectral profile from raw EEG segments.

    Args:
        raw_segments: List of (n_channels, n_samples) arrays
        sfreq: Sampling frequency
        freq_bands: Dict of band_name -> (low_freq, high_freq)

    Returns:
        SpectralProfile with PSD and band powers
    """
    if not raw_segments:
        raise ValueError("No segments provided")

    n_channels = raw_segments[0].shape[0]

    # Compute PSD for each segment and average
    psds = []
    for segment in raw_segments:
        # Welch PSD with 1-second windows
        freqs, psd = welch(segment, fs=sfreq, nperseg=int(sfreq), noverlap=int(sfreq/2))
        psds.append(psd)

    # Average across segments
    mean_psd = np.mean(psds, axis=0)  # (n_channels, n_freqs)

    # Compute band power
    band_power = {}
    global_band_power = {}
    for band_name, (low, high) in freq_bands.items():
        band_mask = (freqs >= low) & (freqs < high)
        if band_mask.sum() > 0:
            # Mean power in band per channel
            bp = mean_psd[:, band_mask].mean(axis=1)
            band_power[band_name] = bp
            global_band_power[band_name] = bp.mean()
        else:
            band_power[band_name] = np.zeros(n_channels)
            global_band_power[band_name] = 0.0

    return SpectralProfile(
        psd=mean_psd,
        freqs=freqs,
        band_power=band_power,
        global_band_power=global_band_power,
    )


def compute_gfp(raw_data: np.ndarray) -> np.ndarray:
    """
    Compute Global Field Power (spatial std at each timepoint).

    Args:
        raw_data: (n_channels, n_samples) array

    Returns:
        (n_samples,) array of GFP values
    """
    return np.std(raw_data, axis=0)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_subject_with_raw(
    file_path: Path,
    model,
    model_info: dict,
    n_chunks: int,
    device: str,
    verbose: bool = False,
) -> Optional[SubjectDataWithRaw]:
    """
    Load a single subject with both latent trajectory and raw EEG.

    Returns SubjectDataWithRaw or None if loading fails.
    """
    try:
        # Load raw EEG
        raw_data, sfreq, channel_names = load_eeg_from_file(file_path, verbose=verbose)

        # Extract phase (circular representation)
        include_amp = model_info.get("include_amplitude", True)
        phase_data = extract_phase_circular(
            raw_data, sfreq,
            filter_low=FILTER_LOW, filter_high=FILTER_HIGH,
            include_amplitude=include_amp,
            skip_filter=False,  # Apply filter
        )

        # Chunk the data
        chunk_samples = int(CHUNK_DURATION * sfreq)
        raw_chunks = chunk_data(raw_data, chunk_samples)
        phase_chunks = chunk_data(phase_data, chunk_samples)

        if len(raw_chunks) == 0:
            return None

        # Limit chunks
        chunks_to_use = min(n_chunks, len(raw_chunks))
        raw_chunks = raw_chunks[:chunks_to_use]
        phase_chunks = phase_chunks[:chunks_to_use]

        # Compute latent trajectories for each chunk
        latents = []
        for phase_chunk in phase_chunks:
            latent = compute_latent_trajectory(model, phase_chunk, device)
            latents.append(latent)

        # Concatenate trajectories
        trajectory = np.concatenate(latents, axis=0)

        # Extract subject ID from filename
        subject_id = file_path.stem.split("_")[0]

        return SubjectDataWithRaw(
            subject_id=subject_id,
            group="",  # Will be set by caller
            label=-1,  # Will be set by caller
            trajectory=trajectory,
            raw_chunks=raw_chunks,
            phase_chunks=phase_chunks,
            sfreq=sfreq,
            channel_names=list(channel_names),
        )

    except Exception as e:
        if verbose:
            print(f"  Warning: Failed to load {file_path}: {e}")
        return None


def load_all_subjects_with_raw(
    model,
    model_info: dict,
    groups: dict,
    n_subjects_per_group: Optional[int],
    n_chunks: int,
    device: str,
) -> dict[str, list[SubjectDataWithRaw]]:
    """Load all subjects with both latent trajectories and raw EEG."""
    group_config = get_group_config()
    subject_data = {}

    for group_key in group_config["keys"]:
        subjects = groups.get(group_key, [])
        if not subjects:
            continue

        # Get display name
        group_name = subjects[0][2] if len(subjects[0]) > 2 else group_key.upper()
        display_name = group_name.upper() if DATASET != "meditation_bids" else group_name.capitalize()

        subject_data[display_name] = []
        max_subjects = n_subjects_per_group if n_subjects_per_group else len(subjects)

        print(f"\nLoading {display_name} subjects (max {max_subjects})...")
        subjects_processed = 0

        for entry in tqdm(subjects, desc=display_name):
            if subjects_processed >= max_subjects:
                break

            file_path, label, condition, subject_id = entry

            subj_data = load_subject_with_raw(
                file_path, model, model_info, n_chunks, device, verbose=False
            )

            if subj_data is not None:
                subj_data.group = group_key
                subj_data.label = label
                subject_data[display_name].append(subj_data)
                subjects_processed += 1

        print(f"  Loaded {subjects_processed} {display_name} subjects")

    return subject_data


# =============================================================================
# ANALYSIS 1: LATENT-REGION → EEG PATTERN CORRESPONDENCE
# =============================================================================

def compute_2d_embedding(trajectories: list[np.ndarray]) -> tuple[np.ndarray, PCA]:
    """
    Compute 2D PCA embedding of all trajectories.

    Returns:
        embedded: (n_total_points, 2) array
        pca: Fitted PCA object
    """
    all_points = np.vstack(trajectories)
    pca = PCA(n_components=2)
    embedded = pca.fit_transform(all_points)
    return embedded, pca


def compute_occupancy_grid(
    embedded: np.ndarray,
    grid_size: int = OCCUPANCY_GRID_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute occupancy histogram on 2D embedded space.

    Returns:
        hist: (grid_size, grid_size) occupancy counts
        x_edges: (grid_size+1,) bin edges for x
        y_edges: (grid_size+1,) bin edges for y
    """
    # Compute bounds with margin
    margin = 0.05
    x_min, x_max = embedded[:, 0].min(), embedded[:, 0].max()
    y_min, y_max = embedded[:, 1].min(), embedded[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min

    x_edges = np.linspace(x_min - margin * x_range, x_max + margin * x_range, grid_size + 1)
    y_edges = np.linspace(y_min - margin * y_range, y_max + margin * y_range, grid_size + 1)

    hist, _, _ = np.histogram2d(embedded[:, 0], embedded[:, 1], bins=[x_edges, y_edges])

    return hist, x_edges, y_edges


def get_region_mask(
    embedded: np.ndarray,
    hist: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    region_type: Literal["high_occupancy", "low_occupancy", "center", "periphery"],
    threshold_percentile: float = 75.0,
) -> np.ndarray:
    """
    Create boolean mask for points in specified latent region.

    Args:
        embedded: (n_points, 2) embedded coordinates
        hist: Occupancy histogram
        x_edges, y_edges: Histogram bin edges
        region_type: Type of region to select
        threshold_percentile: Percentile threshold for high/low

    Returns:
        Boolean mask of shape (n_points,)
    """
    n_points = embedded.shape[0]

    if region_type in ["high_occupancy", "low_occupancy"]:
        # Assign each point to a bin
        x_bin = np.digitize(embedded[:, 0], x_edges) - 1
        y_bin = np.digitize(embedded[:, 1], y_edges) - 1

        # Clip to valid range
        x_bin = np.clip(x_bin, 0, hist.shape[0] - 1)
        y_bin = np.clip(y_bin, 0, hist.shape[1] - 1)

        # Get occupancy for each point
        point_occupancy = hist[x_bin, y_bin]

        # Threshold
        if region_type == "high_occupancy":
            threshold = np.percentile(point_occupancy[point_occupancy > 0], threshold_percentile)
            mask = point_occupancy >= threshold
        else:
            threshold = np.percentile(point_occupancy[point_occupancy > 0], 100 - threshold_percentile)
            mask = (point_occupancy > 0) & (point_occupancy <= threshold)

    elif region_type == "center":
        # Center = within 1 std of centroid
        centroid = embedded.mean(axis=0)
        distances = np.linalg.norm(embedded - centroid, axis=1)
        threshold = np.std(distances)
        mask = distances <= threshold

    elif region_type == "periphery":
        # Periphery = beyond 2 std from centroid
        centroid = embedded.mean(axis=0)
        distances = np.linalg.norm(embedded - centroid, axis=1)
        threshold = 2 * np.std(distances)
        mask = distances >= threshold

    else:
        raise ValueError(f"Unknown region_type: {region_type}")

    return mask


def extract_raw_segments_for_mask(
    subject_data: list[SubjectDataWithRaw],
    mask: np.ndarray,
    trajectories: list[np.ndarray],
    min_fraction: float = 0.5,
) -> tuple[list[np.ndarray], list[float]]:
    """
    Extract raw EEG segments corresponding to masked latent points.

    This maps from latent timepoints back to raw EEG windows.
    Only includes chunks where at least `min_fraction` of points are in the region.

    Args:
        subject_data: List of subjects with raw EEG
        mask: Boolean mask over all latent points
        trajectories: List of trajectories (for reference)
        min_fraction: Minimum fraction of points in region to include chunk (default 0.5)

    Returns:
        Tuple of (raw_segments, weights) where weights indicate the fraction of points
        in the region for each segment.
    """
    raw_segments = []
    weights = []

    # Build index mapping: which subject/chunk does each point belong to?
    point_idx = 0
    for subj in subject_data:
        n_chunks = len(subj.raw_chunks)
        points_per_chunk = len(subj.trajectory) // n_chunks

        for chunk_idx, raw_chunk in enumerate(subj.raw_chunks):
            chunk_start = point_idx + chunk_idx * points_per_chunk
            chunk_end = point_idx + min((chunk_idx + 1) * points_per_chunk, len(subj.trajectory))

            # Get mask for this chunk
            chunk_mask = mask[chunk_start:chunk_end]

            # Compute fraction of points in region
            if len(chunk_mask) > 0:
                fraction = chunk_mask.sum() / len(chunk_mask)
            else:
                fraction = 0.0

            # Only include if sufficient fraction of points are in region
            if fraction >= min_fraction:
                raw_segments.append(raw_chunk)
                weights.append(fraction)

        point_idx += len(subj.trajectory)

    return raw_segments, weights


def compute_chunk_properties(
    subject_data: list[SubjectDataWithRaw],
    pca: PCA,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute chunk-level properties for region-based analysis.

    Returns:
        raw_chunks: List of raw EEG chunks
        mean_positions: (n_chunks, 2) mean 2D position per chunk
        mean_radii: (n_chunks,) mean radius from origin per chunk
        mean_speeds: (n_chunks,) mean speed per chunk
    """
    raw_chunks = []
    mean_positions = []
    mean_radii = []
    mean_speeds = []

    velocity_config = VelocityConfig(method="savgol", savgol_window=5, savgol_poly=2)

    for subj in subject_data:
        n_chunks = len(subj.raw_chunks)
        points_per_chunk = len(subj.trajectory) // n_chunks

        # Compute speed for entire trajectory
        speed = compute_speed(subj.trajectory, config=velocity_config)

        for chunk_idx, raw_chunk in enumerate(subj.raw_chunks):
            chunk_start = chunk_idx * points_per_chunk
            chunk_end = min((chunk_idx + 1) * points_per_chunk, len(subj.trajectory))

            # Get trajectory segment for this chunk
            chunk_traj = subj.trajectory[chunk_start:chunk_end]

            # Project to 2D
            chunk_embedded = pca.transform(chunk_traj)

            # Compute mean position
            mean_pos = chunk_embedded.mean(axis=0)

            # Compute mean radius (distance from origin in latent space)
            radii = np.linalg.norm(chunk_traj, axis=1)
            mean_radius = radii.mean()

            # Compute mean speed for this chunk
            chunk_speed = speed[chunk_start:min(chunk_end, len(speed))].mean()

            raw_chunks.append(raw_chunk)
            mean_positions.append(mean_pos)
            mean_radii.append(mean_radius)
            mean_speeds.append(chunk_speed)

    return (
        raw_chunks,
        np.array(mean_positions),
        np.array(mean_radii),
        np.array(mean_speeds),
    )


def run_region_analysis(
    subject_data: dict[str, list[SubjectDataWithRaw]],
    output_dir: Path,
    show_plot: bool = True,
) -> dict[str, dict[str, RegionEEGProfile]]:
    """
    Analysis 1: Latent-region → EEG pattern correspondence.

    Uses CHUNK-LEVEL properties (not point-level masks) to stratify EEG:
    - Central chunks: Mean position near centroid
    - Peripheral chunks: Mean position far from centroid
    - High-radius chunks: Large mean ||h(t)||
    - Low-radius chunks: Small mean ||h(t)||
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 1: LATENT-REGION → EEG PATTERN CORRESPONDENCE")
    print("(Using chunk-level properties for stratification)")
    print("=" * 80)

    results = {}
    # Use chunk-level properties that make sense for 5-second windows
    region_types = ["central", "peripheral", "high_radius", "low_radius"]

    for group_name, subjects in subject_data.items():
        if len(subjects) < 2:
            print(f"  Skipping {group_name} (< 2 subjects)")
            continue

        print(f"\nProcessing {group_name}...")

        # Get all trajectories for this group
        trajectories = [s.trajectory for s in subjects]

        # Fit PCA on pooled data
        all_points = np.vstack(trajectories)
        pca = PCA(n_components=2)
        pca.fit(all_points)

        # Compute chunk-level properties
        raw_chunks, mean_positions, mean_radii, mean_speeds = compute_chunk_properties(
            subjects, pca
        )

        print(f"  Total chunks: {len(raw_chunks)}")

        # Compute centroid of mean positions
        centroid = mean_positions.mean(axis=0)
        distances_to_centroid = np.linalg.norm(mean_positions - centroid, axis=1)

        results[group_name] = {}

        for region_type in region_types:
            print(f"  Computing {region_type} profile...", end=" ", flush=True)

            # Select chunks based on region type
            if region_type == "central":
                # Bottom 25% by distance to centroid
                threshold = np.percentile(distances_to_centroid, 25)
                mask = distances_to_centroid <= threshold
            elif region_type == "peripheral":
                # Top 25% by distance to centroid
                threshold = np.percentile(distances_to_centroid, 75)
                mask = distances_to_centroid >= threshold
            elif region_type == "high_radius":
                # Top 25% by mean radius
                threshold = np.percentile(mean_radii, 75)
                mask = mean_radii >= threshold
            elif region_type == "low_radius":
                # Bottom 25% by mean radius
                threshold = np.percentile(mean_radii, 25)
                mask = mean_radii <= threshold
            else:
                continue

            # Select raw segments
            selected_chunks = [raw_chunks[i] for i in range(len(raw_chunks)) if mask[i]]
            n_selected = len(selected_chunks)

            if n_selected < 5:
                print(f"skipped (only {n_selected} chunks)")
                continue

            # Compute spectral profile
            sfreq = subjects[0].sfreq
            spectral = compute_spectral_profile(selected_chunks, sfreq)

            # Compute GFP
            gfp_values = []
            for seg in selected_chunks:
                gfp_values.extend(compute_gfp(seg))
            gfp_values = np.array(gfp_values)

            # Create topography dict
            topography = spectral.band_power.copy()

            results[group_name][region_type] = RegionEEGProfile(
                region_name=region_type,
                n_segments=n_selected,
                n_timepoints=mask.sum(),  # Number of chunks in this region
                spectral_profile=spectral,
                mean_gfp=float(gfp_values.mean()),
                std_gfp=float(gfp_values.std()),
                topography=topography,
            )

            print(f"done ({n_selected} chunks)")

    # Plot results
    plot_region_analysis(results, output_dir, show_plot)

    return results


def plot_region_analysis(
    results: dict[str, dict[str, RegionEEGProfile]],
    output_dir: Path,
    show_plot: bool,
):
    """Plot region analysis results."""
    group_config = get_group_config()

    # Create figure with subplots for each group
    n_groups = len(results)
    if n_groups == 0:
        return

    fig, axes = plt.subplots(n_groups, 2, figsize=(14, 5 * n_groups))
    if n_groups == 1:
        axes = axes.reshape(1, -1)

    for idx, (group_name, regions) in enumerate(results.items()):
        color = group_config["colors"].get(group_name, "gray")

        # Left: Band power by region
        ax1 = axes[idx, 0]
        region_names = list(regions.keys())
        bands = list(FREQ_BANDS.keys())
        x = np.arange(len(bands))
        width = 0.8 / len(region_names)

        for r_idx, region_name in enumerate(region_names):
            profile = regions[region_name]
            powers = [profile.spectral_profile.global_band_power.get(b, 0) for b in bands]
            # Log scale for better visualization
            powers_log = np.log10(np.array(powers) + 1e-15)
            ax1.bar(x + r_idx * width, powers_log, width, label=region_name, alpha=0.7)

        ax1.set_xlabel("Frequency Band")
        ax1.set_ylabel("Log10 Power")
        ax1.set_title(f"{group_name}: Band Power by Region")
        ax1.set_xticks(x + width * (len(region_names) - 1) / 2)
        ax1.set_xticklabels(bands, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Right: GFP comparison
        ax2 = axes[idx, 1]
        gfp_means = [regions[r].mean_gfp for r in region_names]
        gfp_stds = [regions[r].std_gfp for r in region_names]
        ax2.bar(region_names, gfp_means, yerr=gfp_stds, color=color, alpha=0.7, capsize=5)
        ax2.set_xlabel("Region")
        ax2.set_ylabel("Global Field Power (μV)")
        ax2.set_title(f"{group_name}: GFP by Region")
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    fig_path = output_dir / "region_analysis_band_power.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")

    if show_plot:
        plt.show()
    plt.close()


# =============================================================================
# ANALYSIS 2: SPEED-CONDITIONED EEG ANALYSIS
# =============================================================================

def run_speed_conditioned_analysis(
    subject_data: dict[str, list[SubjectDataWithRaw]],
    output_dir: Path,
    show_plot: bool = True,
    n_bootstrap: int = 100,
) -> tuple[dict, dict]:
    """
    Analysis 2: Speed-conditioned EEG analysis.

    Stratify EEG by latent speed and compare spectral properties.
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 2: SPEED-CONDITIONED EEG ANALYSIS")
    print("=" * 80)

    speed_profiles = {}
    speed_correlations = {}
    velocity_config = VelocityConfig(method="savgol", savgol_window=5, savgol_poly=2)

    for group_name, subjects in subject_data.items():
        if len(subjects) < 2:
            print(f"  Skipping {group_name} (< 2 subjects)")
            continue

        print(f"\nProcessing {group_name}...")

        # Collect all speeds and corresponding raw segments
        all_speeds = []
        all_raw_windows = []
        all_sfreq = subjects[0].sfreq

        for subj in subjects:
            # Compute speed for this subject's trajectory
            speed = compute_speed(subj.trajectory, config=velocity_config)

            # Map speed back to raw chunks
            n_chunks = len(subj.raw_chunks)
            points_per_chunk = len(speed) // n_chunks

            for chunk_idx, raw_chunk in enumerate(subj.raw_chunks):
                chunk_start = chunk_idx * points_per_chunk
                chunk_end = min((chunk_idx + 1) * points_per_chunk, len(speed))
                chunk_speed = speed[chunk_start:chunk_end].mean()

                all_speeds.append(chunk_speed)
                all_raw_windows.append(raw_chunk)

        all_speeds = np.array(all_speeds)

        # Stratify by speed quantiles
        low_threshold = np.percentile(all_speeds, SPEED_QUANTILES[0] * 100)
        high_threshold = np.percentile(all_speeds, SPEED_QUANTILES[1] * 100)

        low_mask = all_speeds <= low_threshold
        high_mask = all_speeds >= high_threshold
        mid_mask = ~low_mask & ~high_mask

        print(f"  Speed thresholds: low < {low_threshold:.4f}, high > {high_threshold:.4f}")
        print(f"  Segments: low={low_mask.sum()}, mid={mid_mask.sum()}, high={high_mask.sum()}")

        speed_profiles[group_name] = {}

        # Compute spectral profiles for each speed bin
        for bin_name, mask in [("low_speed", low_mask), ("mid_speed", mid_mask), ("high_speed", high_mask)]:
            segments = [all_raw_windows[i] for i in range(len(all_raw_windows)) if mask[i]]
            if len(segments) < 5:
                print(f"    {bin_name}: skipped (< 5 segments)")
                continue

            spectral = compute_spectral_profile(segments, all_sfreq)
            speed_profiles[group_name][bin_name] = spectral
            print(f"    {bin_name}: {len(segments)} segments")

        # Compute speed-band correlations
        speed_correlations[group_name] = {}
        for band_name in FREQ_BANDS.keys():
            band_powers = []
            for raw_chunk in all_raw_windows:
                freqs, psd = welch(raw_chunk, fs=all_sfreq, nperseg=int(all_sfreq))
                low, high = FREQ_BANDS[band_name]
                band_mask = (freqs >= low) & (freqs < high)
                bp = psd[:, band_mask].mean()
                band_powers.append(bp)

            r, p = spearmanr(all_speeds, band_powers)
            speed_correlations[group_name][band_name] = {"r": r, "p": p}
            print(f"    Speed-{band_name} correlation: r={r:.3f}, p={p:.4f}")

    # Plot results
    plot_speed_analysis(speed_profiles, speed_correlations, output_dir, show_plot)

    return speed_profiles, speed_correlations


def plot_speed_analysis(
    speed_profiles: dict,
    speed_correlations: dict,
    output_dir: Path,
    show_plot: bool,
):
    """Plot speed-conditioned analysis results."""
    group_config = get_group_config()
    n_groups = len(speed_profiles)

    if n_groups == 0:
        return

    # Figure 1: PSD comparison by speed bin
    fig, axes = plt.subplots(1, n_groups, figsize=(7 * n_groups, 5))
    if n_groups == 1:
        axes = [axes]

    speed_colors = {"low_speed": "#2ecc71", "mid_speed": "#f39c12", "high_speed": "#e74c3c"}

    for idx, (group_name, profiles) in enumerate(speed_profiles.items()):
        ax = axes[idx]

        for bin_name, spectral in profiles.items():
            # Average PSD across channels
            mean_psd = spectral.psd.mean(axis=0)
            color = speed_colors.get(bin_name, "gray")
            label = bin_name.replace("_", " ").title()
            ax.semilogy(spectral.freqs, mean_psd, color=color, label=label, linewidth=2)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power Spectral Density")
        ax.set_title(f"{group_name}: PSD by Latent Speed")
        ax.set_xlim(0, 50)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / "speed_analysis_psd.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")

    if show_plot:
        plt.show()
    plt.close()

    # Figure 2: Speed-band correlations
    fig, ax = plt.subplots(figsize=(10, 6))

    bands = list(FREQ_BANDS.keys())
    x = np.arange(len(bands))
    width = 0.8 / n_groups

    for idx, (group_name, correlations) in enumerate(speed_correlations.items()):
        color = group_config["colors"].get(group_name, "gray")
        r_values = [correlations.get(b, {}).get("r", 0) for b in bands]
        ax.bar(x + idx * width, r_values, width, label=group_name, color=color, alpha=0.7)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel("Frequency Band")
    ax.set_ylabel("Spearman r (Speed vs Band Power)")
    ax.set_title("Speed-Band Power Correlations by Group")
    ax.set_xticks(x + width * (n_groups - 1) / 2)
    ax.set_xticklabels(bands)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig_path = output_dir / "speed_analysis_correlations.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")

    if show_plot:
        plt.show()
    plt.close()


# =============================================================================
# ANALYSIS 3: FLOW DIRECTION → SIGNAL TRANSITION MAPPING
# =============================================================================

def estimate_flow_field(
    embedded: np.ndarray,
    grid_size: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate flow field from embedded trajectory.

    Returns:
        X, Y: Grid coordinates
        U, V: Flow vectors at each grid point
    """
    # Compute velocities
    velocities = np.diff(embedded, axis=0)
    positions = embedded[:-1]

    # Create grid
    x_min, x_max = embedded[:, 0].min(), embedded[:, 0].max()
    y_min, y_max = embedded[:, 1].min(), embedded[:, 1].max()

    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)

    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    counts = np.zeros_like(X)

    # Bin velocities
    x_bins = np.digitize(positions[:, 0], x_grid) - 1
    y_bins = np.digitize(positions[:, 1], y_grid) - 1

    x_bins = np.clip(x_bins, 0, grid_size - 1)
    y_bins = np.clip(y_bins, 0, grid_size - 1)

    for i in range(len(velocities)):
        xi, yi = x_bins[i], y_bins[i]
        U[yi, xi] += velocities[i, 0]
        V[yi, xi] += velocities[i, 1]
        counts[yi, xi] += 1

    # Average
    valid = counts > 0
    U[valid] /= counts[valid]
    V[valid] /= counts[valid]

    return X, Y, U, V


def identify_principal_flow_directions(
    U: np.ndarray,
    V: np.ndarray,
    n_directions: int = N_FLOW_DIRECTIONS,
) -> np.ndarray:
    """
    Identify principal flow directions from flow field.

    Returns:
        directions: (n_directions, 2) array of unit vectors
    """
    # Flatten and filter valid vectors
    valid = (U != 0) | (V != 0)
    flow_vectors = np.column_stack([U[valid].flatten(), V[valid].flatten()])

    if len(flow_vectors) < n_directions:
        # Not enough data, return cardinal directions
        angles = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])

    # Compute angles
    angles = np.arctan2(flow_vectors[:, 1], flow_vectors[:, 0])

    # Bin into n_directions sectors
    bin_edges = np.linspace(-np.pi, np.pi, n_directions + 1)

    directions = []
    for i in range(n_directions):
        mask = (angles >= bin_edges[i]) & (angles < bin_edges[i + 1])
        if mask.sum() > 0:
            mean_angle = np.mean(angles[mask])
        else:
            mean_angle = (bin_edges[i] + bin_edges[i + 1]) / 2
        directions.append([np.cos(mean_angle), np.sin(mean_angle)])

    return np.array(directions)


def run_flow_direction_analysis(
    subject_data: dict[str, list[SubjectDataWithRaw]],
    output_dir: Path,
    show_plot: bool = True,
) -> dict:
    """
    Analysis 3: Flow direction → Signal transition mapping.

    Characterize spectral changes along dominant flow directions.
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 3: FLOW DIRECTION → SIGNAL TRANSITION MAPPING")
    print("=" * 80)

    results = {}

    for group_name, subjects in subject_data.items():
        if len(subjects) < 2:
            print(f"  Skipping {group_name} (< 2 subjects)")
            continue

        print(f"\nProcessing {group_name}...")

        # Get all trajectories
        trajectories = [s.trajectory for s in subjects]

        # Compute 2D embedding
        embedded, pca = compute_2d_embedding(trajectories)

        # Estimate flow field
        X, Y, U, V = estimate_flow_field(embedded)

        # Identify principal directions
        directions = identify_principal_flow_directions(U, V)
        print(f"  Found {len(directions)} principal flow directions")

        # For each direction, find trajectory segments aligned with it
        # and compute spectral change (before vs after)
        results[group_name] = {
            "directions": directions,
            "spectral_changes": {},
        }

        all_sfreq = subjects[0].sfreq

        # Compute velocities in embedded space
        embedded_velocities = np.diff(embedded, axis=0)
        velocity_norms = np.linalg.norm(embedded_velocities, axis=1)
        velocity_norms[velocity_norms == 0] = 1  # Avoid division by zero
        velocity_unit = embedded_velocities / velocity_norms[:, np.newaxis]

        for d_idx, direction in enumerate(directions):
            # Find segments aligned with this direction (dot product > 0.7)
            alignment = np.dot(velocity_unit, direction)
            aligned_mask = alignment > 0.7
            n_aligned = aligned_mask.sum()

            print(f"    Direction {d_idx}: {n_aligned} aligned segments")

            if n_aligned < 50:
                continue

            # Map back to raw segments and compute before/after spectra
            # This is a simplified version - for full implementation would need
            # to track exact timepoints

            # For now, compute overall spectral profile for aligned segments
            # A more sophisticated version would compute Δpower
            aligned_indices = np.where(aligned_mask)[0]

            # Sample raw segments near aligned latent points
            # (This is approximate - proper implementation would need exact mapping)

            results[group_name]["spectral_changes"][d_idx] = {
                "n_segments": int(n_aligned),
                "direction": direction.tolist(),
            }

    # Plot flow fields and directions
    plot_flow_analysis(results, subject_data, output_dir, show_plot)

    return results


def plot_flow_analysis(
    results: dict,
    subject_data: dict[str, list[SubjectDataWithRaw]],
    output_dir: Path,
    show_plot: bool,
):
    """Plot flow direction analysis results."""
    group_config = get_group_config()
    n_groups = len(results)

    if n_groups == 0:
        return

    fig, axes = plt.subplots(1, n_groups, figsize=(7 * n_groups, 6))
    if n_groups == 1:
        axes = [axes]

    for idx, (group_name, res) in enumerate(results.items()):
        ax = axes[idx]
        color = group_config["colors"].get(group_name, "gray")

        # Get embedded trajectories for this group
        subjects = subject_data[group_name]
        trajectories = [s.trajectory for s in subjects]
        embedded, _ = compute_2d_embedding(trajectories)

        # Plot trajectory density
        ax.hist2d(embedded[:, 0], embedded[:, 1], bins=30, cmap='Blues', alpha=0.5)

        # Estimate and plot flow field
        X, Y, U, V = estimate_flow_field(embedded)
        ax.quiver(X, Y, U, V, color='black', alpha=0.6, scale=None)

        # Highlight principal directions
        centroid = embedded.mean(axis=0)
        directions = res.get("directions", np.array([]))
        for d_idx, direction in enumerate(directions):
            scale = 0.3 * (embedded.max() - embedded.min())
            ax.arrow(centroid[0], centroid[1],
                    direction[0] * scale, direction[1] * scale,
                    head_width=scale * 0.1, head_length=scale * 0.05,
                    fc='red', ec='red', linewidth=2)
            ax.annotate(f"D{d_idx}",
                       (centroid[0] + direction[0] * scale * 1.1,
                        centroid[1] + direction[1] * scale * 1.1),
                       fontsize=10, color='red')

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"{group_name}: Flow Field & Principal Directions")
        ax.set_aspect('equal')

    plt.tight_layout()
    fig_path = output_dir / "flow_direction_analysis.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")

    if show_plot:
        plt.show()
    plt.close()


# =============================================================================
# TOPOGRAPHIC VISUALIZATION (requires MNE)
# =============================================================================

def plot_topographic_maps(
    results: dict[str, dict[str, RegionEEGProfile]],
    channel_names: list[str],
    output_dir: Path,
    show_plot: bool,
):
    """Plot topographic maps of band power by region (requires MNE)."""
    if not HAS_MNE:
        print("  Skipping topographic plots (MNE not available)")
        return

    montage = get_montage_for_dataset()
    if montage is None:
        print("  Skipping topographic plots (montage not available)")
        return

    # Check how many channels match the montage
    montage_ch_names = montage.ch_names

    # Try to apply channel name mapping for meditation dataset
    ch_name_mapping = {}
    mapped_channel_names = channel_names.copy()
    if DATASET == "meditation_bids":
        biosemi_mapping = get_biosemi_channel_mapping()
        for i, ch in enumerate(channel_names):
            if ch in biosemi_mapping:
                mapped_channel_names[i] = biosemi_mapping[ch]
                ch_name_mapping[ch] = biosemi_mapping[ch]
        if ch_name_mapping:
            print(f"  Applied BioSemi channel mapping ({len(ch_name_mapping)} channels)")

    matching_channels = [ch for ch in mapped_channel_names if ch in montage_ch_names]

    if len(matching_channels) < 10:
        print(f"  Skipping topographic plots (only {len(matching_channels)}/{len(channel_names)} "
              f"channels match montage after mapping)")
        print(f"    Dataset channels: {channel_names[:5]}...")
        print(f"    Mapped channels: {mapped_channel_names[:5]}...")
        print(f"    Montage channels: {montage_ch_names[:5]}...")
        return

    # Filter to only EEG channels that match the montage
    eeg_indices = [i for i, ch in enumerate(mapped_channel_names) if ch in montage_ch_names]
    eeg_channel_names = [mapped_channel_names[i] for i in eeg_indices]

    print(f"  Using {len(eeg_channel_names)} channels for topography")

    # Create info object with mapped channel names
    try:
        info = mne.create_info(ch_names=eeg_channel_names, sfreq=250, ch_types='eeg')
        info.set_montage(montage, match_case=False, on_missing='ignore')
    except Exception as e:
        print(f"  Skipping topographic plots (montage setup failed: {e})")
        return

    # Check if we actually have electrode positions
    if info.get_montage() is None:
        print("  Skipping topographic plots (no electrode positions available)")
        return

    group_config = get_group_config()
    bands = list(FREQ_BANDS.keys())

    for group_name, regions in results.items():
        if not regions:
            continue

        region_names = list(regions.keys())
        n_regions = len(region_names)
        n_bands = len(bands)

        fig, axes = plt.subplots(n_bands, n_regions, figsize=(4 * n_regions, 3 * n_bands))
        if n_regions == 1:
            axes = axes.reshape(-1, 1)
        if n_bands == 1:
            axes = axes.reshape(1, -1)

        for b_idx, band in enumerate(bands):
            for r_idx, region_name in enumerate(region_names):
                ax = axes[b_idx, r_idx]
                profile = regions[region_name]

                if band in profile.topography:
                    data = profile.topography[band]
                    # Extract only the channels that match the montage
                    if len(data) == len(channel_names):
                        # Filter data to only matching channels
                        filtered_data = np.array([data[i] for i in eeg_indices])
                        try:
                            mne.viz.plot_topomap(filtered_data, info, axes=ax, show=False,
                                               contours=0, cmap='RdBu_r')
                        except Exception as e:
                            ax.text(0.5, 0.5, "N/A", ha='center', va='center')
                            ax.set_xlim(0, 1)
                            ax.set_ylim(0, 1)
                    else:
                        ax.text(0.5, 0.5, "dim mismatch", ha='center', va='center',
                               fontsize=8)
                        ax.set_xlim(0, 1)
                        ax.set_ylim(0, 1)
                else:
                    ax.text(0.5, 0.5, "N/A", ha='center', va='center')
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)

                if b_idx == 0:
                    ax.set_title(region_name.replace("_", " ").title())
                if r_idx == 0:
                    ax.set_ylabel(band)

        fig.suptitle(f"{group_name}: Topographic Maps by Region and Band", y=1.02)
        plt.tight_layout()

        fig_path = output_dir / f"topography_{group_name.lower()}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {fig_path}")

        if show_plot:
            plt.show()
        plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Inverse Mapping Analysis")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH,
                       help="Path to model checkpoint")
    parser.add_argument("--n-subjects", type=int, default=None,
                       help="Max subjects per group (None = all)")
    parser.add_argument("--n-chunks", type=int, default=30,
                       help="Max chunks per subject")
    parser.add_argument("--analysis", type=str, default="all",
                       choices=["all", "region", "speed", "flow"],
                       help="Which analysis to run")
    parser.add_argument("--n-bootstrap", type=int, default=100,
                       help="Bootstrap iterations for CIs")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test mode (5 subjects, 10 chunks)")
    parser.add_argument("--no-show", action="store_true",
                       help="Don't display plots")
    parser.add_argument("--device", type=str, default=DEVICE,
                       help="Device for inference")

    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.n_subjects = 5
        args.n_chunks = 10
        args.n_bootstrap = 50

    show_plot = not args.no_show

    print("=" * 80)
    print("INVERSE MAPPING ANALYSIS: Latent Dynamics → Signal Space")
    print("=" * 80)
    print(f"Dataset: {DATASET}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Analysis: {args.analysis}")
    print(f"Max subjects per group: {args.n_subjects or 'all'}")
    print(f"Max chunks per subject: {args.n_chunks}")

    # Create output directory
    output_dir = create_timestamped_output_dir(OUTPUT_DIR, "inverse_mapping")
    print(f"Output directory: {output_dir}")

    # Save parameters
    save_parameters(output_dir, vars(args))

    # Load model
    print("\n" + "-" * 40)
    print("Loading model...")
    model_info = load_model_from_checkpoint(args.checkpoint, args.device)

    # Get dataset config and files
    dataset_config = get_dataset_config()
    data_files = get_data_files_via_config()
    groups = get_subjects_by_group_unified(data_files)

    # Determine n_channels from first file
    first_group = list(groups.values())[0]
    if first_group:
        first_file = first_group[0][0]
        raw_data, _, _ = load_eeg_from_file(first_file, verbose=False)
        n_channels = raw_data.shape[0]
    else:
        raise ValueError("No data files found")

    print(f"EEG channels: {n_channels}")

    # Create model
    model = create_model(n_channels, model_info, args.device)

    # Load all subjects with raw data
    print("\n" + "-" * 40)
    print("Loading subjects with raw EEG...")
    subject_data = load_all_subjects_with_raw(
        model, model_info, groups,
        n_subjects_per_group=args.n_subjects,
        n_chunks=args.n_chunks,
        device=args.device,
    )

    # Print summary
    print("\n" + "-" * 40)
    print("Data summary:")
    for group_name, subjects in subject_data.items():
        if subjects:
            total_chunks = sum(len(s.raw_chunks) for s in subjects)
            total_points = sum(len(s.trajectory) for s in subjects)
            print(f"  {group_name}: {len(subjects)} subjects, {total_chunks} chunks, {total_points} latent points")

    # Run analyses
    results = InverseMappingResults(
        timestamp=datetime.now().isoformat(),
        dataset=DATASET,
        n_subjects={g: len(s) for g, s in subject_data.items()},
    )

    if args.analysis in ["all", "region"]:
        results.region_profiles = run_region_analysis(subject_data, output_dir, show_plot)

        # Plot topographic maps if we have results
        if results.region_profiles:
            first_subjects = list(subject_data.values())[0]
            if first_subjects:
                channel_names = first_subjects[0].channel_names
                plot_topographic_maps(results.region_profiles, channel_names, output_dir, show_plot)

    if args.analysis in ["all", "speed"]:
        speed_profiles, speed_correlations = run_speed_conditioned_analysis(
            subject_data, output_dir, show_plot, args.n_bootstrap
        )
        results.speed_profiles = speed_profiles
        results.speed_band_correlations = speed_correlations

    if args.analysis in ["all", "flow"]:
        results.flow_spectral_changes = run_flow_direction_analysis(
            subject_data, output_dir, show_plot
        )

    # Save results summary
    summary = {
        "timestamp": results.timestamp,
        "dataset": results.dataset,
        "n_subjects": results.n_subjects,
        "analyses_run": args.analysis,
    }

    # Add speed correlations to summary if available
    if results.speed_band_correlations:
        summary["speed_band_correlations"] = results.speed_band_correlations

    summary_path = output_dir / "results_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults summary saved to: {summary_path}")

    print("\n" + "=" * 80)
    print("INVERSE MAPPING ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
