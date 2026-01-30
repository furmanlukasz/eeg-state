#!/usr/bin/env python3
"""
Decoding Sanity Check: Latent Space Grounding in Canonical EEG Summaries

Tests whether learned latent coordinates are grounded in canonical EEG features.
This is a legibility/sanity check, NOT biological interpretability.

Computes correlations between:
- Latent summaries: radius ||h(t)||, speed ||h(t+1)-h(t)||, PC1
- EEG summaries: broadband envelope, alpha envelope, GFP

Uses subject-level block bootstrap for confidence intervals.

Usage:
    python scripts/local_analysis/decoding_sanity_check.py
    python scripts/local_analysis/decoding_sanity_check.py --checkpoint models/best.pt
    python scripts/local_analysis/decoding_sanity_check.py --alpha_band 8 12 --n_bootstrap 500

Outputs:
    - decoding_sanity_correlations.png (violin/box plots)
    - decoding_sanity_example_scatter.png (representative subject)
    - decoding_sanity_results.json (per-subject correlations)
    - decoding_sanity_summary.csv (median + CI)
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
import multiprocessing as mp
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import config as cfg
from load_model import load_model_from_checkpoint, create_model, compute_latent_trajectory
from load_data import load_eeg_from_file
from velocity import compute_speed as _compute_speed


def get_n_channels_from_checkpoint(checkpoint_path: Path) -> tuple[int, bool]:
    """
    Infer n_channels from model checkpoint weights.

    Returns:
        Tuple of (n_channels, include_amplitude)
    """
    import torch
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    for key in state_dict:
        if "conv" in key and "weight" in key and state_dict[key].dim() == 3:
            in_features = state_dict[key].shape[1]
            if in_features % 3 == 0:
                return in_features // 3, True
            elif in_features % 2 == 0:
                return in_features // 2, False

    raise ValueError("Could not infer n_channels from checkpoint")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class LatentSummaries:
    """Latent space summary signals for a chunk or subject."""
    radius: np.ndarray          # ||h(t)||
    speed: np.ndarray           # ||h(t+1) - h(t)||
    pc1: Optional[np.ndarray] = None  # First PC projection

    @property
    def n_timepoints(self) -> int:
        return len(self.radius)


@dataclass
class EEGSummaries:
    """Canonical EEG summary signals."""
    broadband_envelope: np.ndarray   # Mean log amplitude across channels
    alpha_envelope: np.ndarray       # Alpha-band envelope (8-12 Hz)
    gfp: Optional[np.ndarray] = None # Global Field Power

    @property
    def n_timepoints(self) -> int:
        return len(self.broadband_envelope)


@dataclass
class CorrelationResult:
    """Correlation result with bootstrap CI."""
    spearman_r: float
    spearman_p: float
    pearson_r: float
    pearson_p: float
    n_samples: int

    def to_dict(self):
        return {
            "spearman_r": float(self.spearman_r),
            "spearman_p": float(self.spearman_p),
            "pearson_r": float(self.pearson_r),
            "pearson_p": float(self.pearson_p),
            "n_samples": self.n_samples,
        }


@dataclass
class SubjectResults:
    """Results for a single subject."""
    subject_id: str
    group: str
    n_chunks: int
    correlations: dict  # {(latent_name, eeg_name): CorrelationResult}

    def to_dict(self):
        return {
            "subject_id": self.subject_id,
            "group": self.group,
            "n_chunks": self.n_chunks,
            "correlations": {
                f"{k[0]}_vs_{k[1]}": v.to_dict()
                for k, v in self.correlations.items()
            }
        }


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval."""
    median: float
    ci_low: float
    ci_high: float
    mean: float
    std: float

    @classmethod
    def from_samples(cls, samples: np.ndarray, ci: float = 0.95):
        alpha = (1 - ci) / 2
        return cls(
            median=float(np.median(samples)),
            ci_low=float(np.percentile(samples, alpha * 100)),
            ci_high=float(np.percentile(samples, (1 - alpha) * 100)),
            mean=float(np.mean(samples)),
            std=float(np.std(samples)),
        )


# =============================================================================
# EEG FEATURE EXTRACTION
# =============================================================================

def compute_broadband_envelope(data: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Compute broadband amplitude envelope (mean log amplitude across channels).

    Args:
        data: (n_channels, n_samples) raw EEG data
        sfreq: Sampling frequency

    Returns:
        (n_samples,) broadband envelope
    """
    # Hilbert transform for analytic signal
    analytic = hilbert(data, axis=1)
    amplitude = np.abs(analytic)

    # Log amplitude (matches preprocessing)
    log_amplitude = np.log1p(amplitude)

    # Mean across channels
    return np.mean(log_amplitude, axis=0)


def compute_alpha_envelope(
    data: np.ndarray,
    sfreq: float,
    alpha_low: float = 8.0,
    alpha_high: float = 12.0,
) -> np.ndarray:
    """
    Compute alpha-band envelope (mean alpha amplitude across channels).

    Args:
        data: (n_channels, n_samples) raw EEG data
        sfreq: Sampling frequency
        alpha_low: Low cutoff for alpha band
        alpha_high: High cutoff for alpha band

    Returns:
        (n_samples,) alpha envelope
    """
    # Bandpass filter to alpha band
    nyq = sfreq / 2
    low = alpha_low / nyq
    high = alpha_high / nyq

    # Ensure valid filter bounds
    if high >= 1.0:
        high = 0.99
    if low <= 0:
        low = 0.01

    b, a = butter(4, [low, high], btype="band")
    filtered = filtfilt(b, a, data, axis=1)

    # Hilbert transform for envelope
    analytic = hilbert(filtered, axis=1)
    amplitude = np.abs(analytic)

    # Mean across channels
    return np.mean(amplitude, axis=0)


def compute_gfp(data: np.ndarray) -> np.ndarray:
    """
    Compute Global Field Power (spatial standard deviation at each timepoint).

    Args:
        data: (n_channels, n_samples) raw EEG data

    Returns:
        (n_samples,) GFP
    """
    # Average reference first (GFP is reference-dependent)
    data_avg_ref = data - np.mean(data, axis=0, keepdims=True)

    # Spatial std at each timepoint
    return np.std(data_avg_ref, axis=0)


def extract_eeg_summaries(
    data: np.ndarray,
    sfreq: float,
    alpha_low: float = 8.0,
    alpha_high: float = 12.0,
    compute_gfp_flag: bool = True,
) -> EEGSummaries:
    """
    Extract all canonical EEG summary signals.

    Args:
        data: (n_channels, n_samples) raw EEG data
        sfreq: Sampling frequency
        alpha_low: Low cutoff for alpha band
        alpha_high: High cutoff for alpha band
        compute_gfp_flag: Whether to compute GFP

    Returns:
        EEGSummaries dataclass
    """
    broadband = compute_broadband_envelope(data, sfreq)
    alpha = compute_alpha_envelope(data, sfreq, alpha_low, alpha_high)
    gfp = compute_gfp(data) if compute_gfp_flag else None

    return EEGSummaries(
        broadband_envelope=broadband,
        alpha_envelope=alpha,
        gfp=gfp,
    )


# =============================================================================
# LATENT FEATURE EXTRACTION
# =============================================================================

def compute_latent_radius(latent: np.ndarray) -> np.ndarray:
    """
    Compute latent radius ||h(t)|| at each timepoint.

    Args:
        latent: (n_timepoints, hidden_size) latent trajectory

    Returns:
        (n_timepoints,) radius
    """
    return np.linalg.norm(latent, axis=1)


def compute_latent_speed(latent: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    Compute latent speed ||h(t+1) - h(t)|| / dt.

    Args:
        latent: (n_timepoints, hidden_size) latent trajectory
        dt: Time step (for normalization, default=1 means per-sample speed)

    Returns:
        (n_timepoints-1,) speed (one less than input length)

    Note: Delegates to centralized velocity module for consistency
    and configurable Δt/Savitzky-Golay support.
    """
    return _compute_speed(latent, method="finite_diff", delta_t=1, dt_seconds=dt)


def compute_latent_pc1(latent: np.ndarray) -> np.ndarray:
    """
    Compute first principal component projection of latent trajectory.

    Args:
        latent: (n_timepoints, hidden_size) latent trajectory

    Returns:
        (n_timepoints,) PC1 projection
    """
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(latent).flatten()
    return pc1


def extract_latent_summaries(
    latent: np.ndarray,
    compute_pc1: bool = True,
) -> LatentSummaries:
    """
    Extract all latent summary signals.

    Args:
        latent: (n_timepoints, hidden_size) latent trajectory
        compute_pc1: Whether to compute PC1

    Returns:
        LatentSummaries dataclass
    """
    radius = compute_latent_radius(latent)
    speed = compute_latent_speed(latent)
    pc1 = compute_latent_pc1(latent) if compute_pc1 else None

    return LatentSummaries(
        radius=radius,
        speed=speed,
        pc1=pc1,
    )


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def compute_correlation(
    x: np.ndarray,
    y: np.ndarray,
) -> CorrelationResult:
    """
    Compute Spearman and Pearson correlations between two signals.

    Args:
        x: First signal
        y: Second signal (must be same length as x)

    Returns:
        CorrelationResult
    """
    # Ensure same length
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]

    # Remove any NaN/inf
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return CorrelationResult(
            spearman_r=np.nan,
            spearman_p=np.nan,
            pearson_r=np.nan,
            pearson_p=np.nan,
            n_samples=len(x),
        )

    spearman_r, spearman_p = spearmanr(x, y)
    pearson_r, pearson_p = pearsonr(x, y)

    return CorrelationResult(
        spearman_r=spearman_r,
        spearman_p=spearman_p,
        pearson_r=pearson_r,
        pearson_p=pearson_p,
        n_samples=len(x),
    )


def downsample_to_match(
    signal: np.ndarray,
    target_len: int,
) -> np.ndarray:
    """
    Downsample signal to match target length using averaging.

    Args:
        signal: Signal to downsample
        target_len: Target length

    Returns:
        Downsampled signal
    """
    if len(signal) == target_len:
        return signal

    if len(signal) < target_len:
        # Upsample via interpolation
        indices = np.linspace(0, len(signal) - 1, target_len)
        return np.interp(indices, np.arange(len(signal)), signal)

    # Downsample by averaging bins
    factor = len(signal) / target_len
    result = np.zeros(target_len)
    for i in range(target_len):
        start = int(i * factor)
        end = int((i + 1) * factor)
        result[i] = np.mean(signal[start:end])

    return result


def block_bootstrap_correlation(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 500,
    block_size: int = 10,
) -> BootstrapCI:
    """
    Compute bootstrap CI for correlation using block resampling.

    Args:
        x: First signal
        y: Second signal
        n_bootstrap: Number of bootstrap iterations
        block_size: Size of contiguous blocks to resample

    Returns:
        BootstrapCI for Spearman correlation
    """
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]

    n_blocks = len(x) // block_size
    if n_blocks < 2:
        # Not enough data for block bootstrap
        r, _ = spearmanr(x, y)
        return BootstrapCI(
            median=r, ci_low=r, ci_high=r, mean=r, std=0.0
        )

    bootstrap_rs = []
    for _ in range(n_bootstrap):
        # Sample blocks with replacement
        block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)

        # Reconstruct resampled signals
        x_boot = []
        y_boot = []
        for bi in block_indices:
            start = bi * block_size
            end = start + block_size
            x_boot.extend(x[start:end])
            y_boot.extend(y[start:end])

        x_boot = np.array(x_boot)
        y_boot = np.array(y_boot)

        r, _ = spearmanr(x_boot, y_boot)
        if np.isfinite(r):
            bootstrap_rs.append(r)

    if len(bootstrap_rs) < 10:
        r, _ = spearmanr(x, y)
        return BootstrapCI(
            median=r, ci_low=r, ci_high=r, mean=r, std=0.0
        )

    return BootstrapCI.from_samples(np.array(bootstrap_rs))


# =============================================================================
# SUBJECT-LEVEL PROCESSING
# =============================================================================

def process_subject(
    file_path: Path,
    model,
    model_info: dict,
    device: str,
    sfreq: float,
    chunk_duration: float,
    alpha_low: float,
    alpha_high: float,
    verbose: bool = False,
) -> tuple[list[LatentSummaries], list[EEGSummaries]]:
    """
    Process a single subject: load data, compute latent and EEG summaries per chunk.

    Args:
        file_path: Path to EEG file
        model: Trained autoencoder
        model_info: Model configuration
        device: Compute device
        sfreq: Sampling frequency
        chunk_duration: Duration of each chunk in seconds
        alpha_low: Alpha band low cutoff
        alpha_high: Alpha band high cutoff
        verbose: Print progress

    Returns:
        Tuple of (list of LatentSummaries, list of EEGSummaries) per chunk
    """
    # Load raw EEG data with preprocessing
    raw_data, actual_sfreq, channel_names = load_eeg_from_file(file_path, verbose=verbose)
    n_channels = len(channel_names)

    # Use actual sampling frequency
    sfreq = actual_sfreq
    chunk_samples = int(chunk_duration * sfreq)

    # Chunk the raw data
    n_samples = raw_data.shape[1]
    n_chunks = n_samples // chunk_samples

    latent_summaries_list = []
    eeg_summaries_list = []

    for i in range(n_chunks):
        start = i * chunk_samples
        end = start + chunk_samples
        chunk_data = raw_data[:, start:end]

        # Extract EEG summaries from raw chunk
        eeg_summ = extract_eeg_summaries(
            chunk_data, sfreq, alpha_low, alpha_high
        )

        # Convert to phase representation for model
        from load_data import extract_phase_circular
        is_meditation = file_path.suffix.lower() == ".bdf"

        if is_meditation:
            # Data already filtered 2-48 Hz in load_eeg_from_file
            phase_data = extract_phase_circular(
                chunk_data, sfreq,
                include_amplitude=model_info["include_amplitude"],
                skip_filter=True
            )
        else:
            phase_data = extract_phase_circular(
                chunk_data, sfreq,
                filter_low=1.0, filter_high=30.0,
                include_amplitude=model_info["include_amplitude"],
                skip_filter=False
            )

        # Compute latent trajectory
        latent = compute_latent_trajectory(model, phase_data, device)

        # Extract latent summaries
        latent_summ = extract_latent_summaries(latent, compute_pc1=True)

        latent_summaries_list.append(latent_summ)
        eeg_summaries_list.append(eeg_summ)

    return latent_summaries_list, eeg_summaries_list


def analyze_subject(
    latent_list: list[LatentSummaries],
    eeg_list: list[EEGSummaries],
    subject_id: str,
    group: str,
) -> SubjectResults:
    """
    Compute correlations for a subject across all chunks.

    Uses chunk-level summaries (mean per chunk) for correlation.

    Args:
        latent_list: List of LatentSummaries per chunk
        eeg_list: List of EEGSummaries per chunk
        subject_id: Subject identifier
        group: Group label

    Returns:
        SubjectResults
    """
    n_chunks = len(latent_list)

    # Aggregate chunk-level means
    latent_means = {
        "radius": np.array([np.mean(ls.radius) for ls in latent_list]),
        "speed": np.array([np.mean(ls.speed) for ls in latent_list]),
    }
    if latent_list[0].pc1 is not None:
        latent_means["pc1"] = np.array([np.mean(ls.pc1) for ls in latent_list])

    eeg_means = {
        "broadband": np.array([np.mean(es.broadband_envelope) for es in eeg_list]),
        "alpha": np.array([np.mean(es.alpha_envelope) for es in eeg_list]),
    }
    if eeg_list[0].gfp is not None:
        eeg_means["gfp"] = np.array([np.mean(es.gfp) for es in eeg_list])

    # Compute all pairwise correlations
    correlations = {}
    for latent_name, latent_signal in latent_means.items():
        for eeg_name, eeg_signal in eeg_means.items():
            corr = compute_correlation(latent_signal, eeg_signal)
            correlations[(latent_name, eeg_name)] = corr

    return SubjectResults(
        subject_id=subject_id,
        group=group,
        n_chunks=n_chunks,
        correlations=correlations,
    )


# =============================================================================
# PARALLEL PROCESSING
# =============================================================================

# Global model cache for worker processes
_worker_model = None
_worker_model_info = None


def _init_worker(checkpoint_path: str, n_channels: int, device: str):
    """Initialize model in worker process."""
    global _worker_model, _worker_model_info
    import torch

    # Use CPU in workers to avoid MPS/CUDA sharing issues
    worker_device = "cpu"

    _worker_model_info = load_model_from_checkpoint(Path(checkpoint_path), worker_device)
    _worker_model = create_model(n_channels, _worker_model_info, worker_device, load_weights=True)


def _process_subject_worker(args: tuple) -> tuple:
    """
    Worker function for parallel subject processing.

    Args:
        args: (file_path, label, group_name, subject_id, sfreq, chunk_duration, alpha_low, alpha_high)

    Returns:
        (subject_id, group_name, latent_list, eeg_list) or (subject_id, None, None, None) on error
    """
    global _worker_model, _worker_model_info

    file_path, label, group_name, subject_id, sfreq, chunk_duration, alpha_low, alpha_high = args

    try:
        latent_list, eeg_list = process_subject(
            file_path=file_path,
            model=_worker_model,
            model_info=_worker_model_info,
            device="cpu",  # Workers use CPU
            sfreq=sfreq,
            chunk_duration=chunk_duration,
            alpha_low=alpha_low,
            alpha_high=alpha_high,
            verbose=False,
        )
        return (subject_id, group_name, label, latent_list, eeg_list)
    except Exception as e:
        print(f"  Error processing {subject_id}: {e}")
        return (subject_id, None, None, None, None)


def process_subjects_parallel(
    all_subjects: list,
    checkpoint_path: Path,
    n_channels: int,
    device: str,
    sfreq: float,
    chunk_duration: float,
    alpha_low: float,
    alpha_high: float,
    n_workers: int = None,
) -> tuple[list[SubjectResults], list, list, str]:
    """
    Process all subjects in parallel using multiprocessing.

    Returns:
        Tuple of (subject_results, example_latent, example_eeg, example_subject_id)
    """
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    print(f"Using {n_workers} parallel workers")

    # Prepare args for each subject
    subject_args = [
        (fp, label, group, sid, sfreq, chunk_duration, alpha_low, alpha_high)
        for fp, label, group, sid in all_subjects
    ]

    subject_results = []
    example_latent = None
    example_eeg = None
    example_subject_id = None

    # Use spawn to avoid MPS/CUDA issues on macOS
    ctx = mp.get_context('spawn')

    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(str(checkpoint_path), n_channels, device),
    ) as executor:
        # Submit all tasks
        futures = {executor.submit(_process_subject_worker, args): args[3] for args in subject_args}

        # Collect results with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing subjects"):
            subject_id = futures[future]
            try:
                sid, group_name, label, latent_list, eeg_list = future.result()

                if latent_list is None or len(latent_list) < 3:
                    continue

                results = analyze_subject(latent_list, eeg_list, sid, group_name)
                subject_results.append(results)

                # Save first subject for example scatter
                if example_latent is None:
                    example_latent = latent_list
                    example_eeg = eeg_list
                    example_subject_id = sid

            except Exception as e:
                print(f"  Future error for {subject_id}: {e}")
                continue

    return subject_results, example_latent, example_eeg, example_subject_id


# =============================================================================
# BOOTSTRAP AGGREGATION
# =============================================================================

def bootstrap_across_subjects(
    subject_results: list[SubjectResults],
    n_bootstrap: int = 500,
) -> dict:
    """
    Bootstrap across subjects to get population-level CIs.

    Args:
        subject_results: List of SubjectResults
        n_bootstrap: Number of bootstrap iterations

    Returns:
        Dict of {(latent, eeg): BootstrapCI} for each correlation pair
    """
    if len(subject_results) == 0:
        return {}

    # Get all correlation pairs from first subject
    pairs = list(subject_results[0].correlations.keys())

    bootstrap_cis = {}
    for pair in pairs:
        # Collect Spearman r values across subjects
        rs = [
            sr.correlations[pair].spearman_r
            for sr in subject_results
            if np.isfinite(sr.correlations[pair].spearman_r)
        ]

        if len(rs) < 3:
            bootstrap_cis[pair] = BootstrapCI(
                median=np.nan, ci_low=np.nan, ci_high=np.nan,
                mean=np.nan, std=np.nan
            )
            continue

        rs = np.array(rs)

        # Bootstrap over subjects
        bootstrap_medians = []
        for _ in range(n_bootstrap):
            boot_rs = np.random.choice(rs, size=len(rs), replace=True)
            bootstrap_medians.append(np.median(boot_rs))

        bootstrap_cis[pair] = BootstrapCI.from_samples(np.array(bootstrap_medians))

    return bootstrap_cis


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_correlation_violins(
    subject_results: list[SubjectResults],
    bootstrap_cis: dict,
    output_path: Path,
):
    """
    Create violin/box plot of correlations across subjects.

    Args:
        subject_results: List of SubjectResults
        bootstrap_cis: Bootstrap CIs per correlation pair
        output_path: Path to save figure
    """
    if len(subject_results) == 0:
        print("No subjects to plot")
        return

    # Collect data for plotting
    pairs = list(subject_results[0].correlations.keys())

    # Create figure
    n_pairs = len(pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(4 * n_pairs, 5), squeeze=False)
    axes = axes[0]

    for i, pair in enumerate(pairs):
        ax = axes[i]
        latent_name, eeg_name = pair

        # Collect r values
        rs = [
            sr.correlations[pair].spearman_r
            for sr in subject_results
            if np.isfinite(sr.correlations[pair].spearman_r)
        ]

        if len(rs) > 0:
            # Violin plot
            parts = ax.violinplot([rs], positions=[0], showmeans=True, showmedians=True)

            # Color the violin
            for pc in parts['bodies']:
                pc.set_facecolor('#1f77b4')
                pc.set_alpha(0.6)

            # Add individual points
            jitter = np.random.normal(0, 0.03, size=len(rs))
            ax.scatter(jitter, rs, c='#1f77b4', alpha=0.5, s=30)

            # Add bootstrap CI annotation
            ci = bootstrap_cis.get(pair)
            if ci and np.isfinite(ci.median):
                ax.axhline(ci.median, color='red', linestyle='--', linewidth=1.5, label='Median')
                ax.fill_between(
                    [-0.4, 0.4], ci.ci_low, ci.ci_high,
                    alpha=0.2, color='red', label=f'95% CI'
                )

        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-1, 1)
        ax.set_xticks([])
        ax.set_ylabel("Spearman r")
        ax.set_title(f"{latent_name} vs {eeg_name}")
        ax.legend(fontsize=8, loc='lower right')

    plt.suptitle("Latent-EEG Correlations (per subject)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_example_scatter(
    latent_list: list[LatentSummaries],
    eeg_list: list[EEGSummaries],
    subject_id: str,
    output_path: Path,
):
    """
    Plot scatter of latent radius vs alpha envelope for one subject.

    Args:
        latent_list: LatentSummaries per chunk
        eeg_list: EEGSummaries per chunk
        subject_id: Subject ID for title
        output_path: Path to save figure
    """
    # Chunk-level means
    radius = np.array([np.mean(ls.radius) for ls in latent_list])
    alpha = np.array([np.mean(es.alpha_envelope) for es in eeg_list])

    # Compute correlation
    r, p = spearmanr(radius, alpha)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(alpha, radius, c='#1f77b4', alpha=0.6, s=40)

    # Fit line
    z = np.polyfit(alpha, radius, 1)
    p_line = np.poly1d(z)
    alpha_range = np.linspace(alpha.min(), alpha.max(), 100)
    ax.plot(alpha_range, p_line(alpha_range), 'r--', linewidth=2,
            label=f'Spearman r = {r:.3f}')

    ax.set_xlabel("Alpha envelope (chunk mean)")
    ax.set_ylabel("Latent radius ||h|| (chunk mean)")
    ax.set_title(f"Subject: {subject_id}")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# OUTPUT
# =============================================================================

def save_results_json(
    subject_results: list[SubjectResults],
    bootstrap_cis: dict,
    output_path: Path,
):
    """Save full results to JSON."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "n_subjects": len(subject_results),
        "subjects": [sr.to_dict() for sr in subject_results],
        "population_bootstrap_ci": {
            f"{k[0]}_vs_{k[1]}": {
                "median": v.median,
                "ci_low": v.ci_low,
                "ci_high": v.ci_high,
                "mean": v.mean,
                "std": v.std,
            }
            for k, v in bootstrap_cis.items()
        }
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {output_path}")


def save_summary_csv(
    subject_results: list[SubjectResults],
    bootstrap_cis: dict,
    output_path: Path,
):
    """Save summary table to CSV."""
    rows = []

    if len(subject_results) == 0:
        pd.DataFrame(rows).to_csv(output_path, index=False)
        return

    pairs = list(subject_results[0].correlations.keys())

    for pair in pairs:
        latent_name, eeg_name = pair

        # Collect r values
        rs = [
            sr.correlations[pair].spearman_r
            for sr in subject_results
            if np.isfinite(sr.correlations[pair].spearman_r)
        ]

        ci = bootstrap_cis.get(pair)

        rows.append({
            "latent_feature": latent_name,
            "eeg_feature": eeg_name,
            "n_subjects": len(rs),
            "median_r": np.median(rs) if rs else np.nan,
            "mean_r": np.mean(rs) if rs else np.nan,
            "std_r": np.std(rs) if rs else np.nan,
            "ci_low": ci.ci_low if ci else np.nan,
            "ci_high": ci.ci_high if ci else np.nan,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def create_output_dir(base_dir: Path) -> Path:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / f"decoding_sanity_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Decoding sanity check: correlate latent summaries with EEG summaries"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=cfg.CHECKPOINT_PATH,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data_dir", type=Path, default=cfg.DATA_DIR,
        help="Path to data directory"
    )
    parser.add_argument(
        "--out_dir", type=Path, default=None,
        help="Output directory (default: results/local_analysis/decoding_sanity_TIMESTAMP)"
    )
    parser.add_argument(
        "--alpha_band", type=float, nargs=2, default=[8.0, 12.0],
        help="Alpha band range in Hz (default: 8 12)"
    )
    parser.add_argument(
        "--n_bootstrap", type=int, default=500,
        help="Number of bootstrap iterations"
    )
    parser.add_argument(
        "--chunk_duration", type=float, default=cfg.CHUNK_DURATION,
        help="Chunk duration in seconds"
    )
    parser.add_argument(
        "--max_subjects", type=int, default=None,
        help="Maximum subjects to process (for testing)"
    )
    parser.add_argument(
        "--device", type=str, default=cfg.DEVICE,
        help="Compute device (mps, cuda, cpu)"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Dataset to use: greek_resting or meditation_bids (default: infer from checkpoint name)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print verbose output"
    )
    parser.add_argument(
        "--n_workers", type=int, default=None,
        help="Number of parallel workers (default: CPU count - 1)"
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Run sequentially (no parallelization)"
    )

    args = parser.parse_args()

    # Determine dataset - infer from checkpoint name if not specified
    if args.dataset is None:
        checkpoint_name = args.checkpoint.name.lower()
        if "meditation" in checkpoint_name:
            dataset = "meditation_bids"
        else:
            dataset = "greek_resting"
    else:
        dataset = args.dataset

    # Override config module's DATASET for this run
    cfg.DATASET = dataset

    # Update DATA_DIR based on dataset
    args.data_dir = cfg.DATA_PATHS.get(dataset, args.data_dir)

    # Setup output directory
    if args.out_dir is None:
        base_dir = cfg.ensure_output_dir()
        output_dir = create_output_dir(base_dir)
    else:
        output_dir = args.out_dir
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Dataset: {dataset}")
    print(f"Data directory: {args.data_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Alpha band: {args.alpha_band[0]}-{args.alpha_band[1]} Hz")
    print(f"Bootstrap iterations: {args.n_bootstrap}")

    # Save parameters
    params = {
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "dataset": dataset,
        "alpha_low": args.alpha_band[0],
        "alpha_high": args.alpha_band[1],
        "n_bootstrap": args.n_bootstrap,
        "chunk_duration": args.chunk_duration,
        "device": args.device,
    }
    with open(output_dir / "parameters.json", 'w') as f:
        json.dump(params, f, indent=2)

    # Load model info and infer n_channels from checkpoint
    print("\nLoading model...")
    model_info = load_model_from_checkpoint(args.checkpoint, args.device)

    # Infer n_channels from checkpoint weights (model was trained with specific channel count)
    n_channels, include_amp = get_n_channels_from_checkpoint(args.checkpoint)
    print(f"  Inferred n_channels from checkpoint: {n_channels}")

    # Get data files
    print("\nDiscovering data files...")
    data_files = cfg.get_data_files_via_config()
    groups = cfg.get_subjects_by_group_unified(data_files)

    # Flatten to list of (file_path, label, group_name, subject_id)
    all_subjects = []
    for group_key, subjects in groups.items():
        all_subjects.extend(subjects)

    if args.max_subjects:
        all_subjects = all_subjects[:args.max_subjects]

    print(f"Processing {len(all_subjects)} subjects...")

    # Get sfreq from first file
    first_file = all_subjects[0][0]
    _, sfreq, _ = load_eeg_from_file(first_file, verbose=False)

    # Process all subjects (parallel or sequential)
    if args.sequential:
        print("Running in sequential mode...")

        # Create model with n_channels from checkpoint
        model = create_model(n_channels, model_info, args.device)

        subject_results = []
        example_latent = None
        example_eeg = None
        example_subject_id = None

        for file_path, label, group_name, subject_id in tqdm(all_subjects, desc="Processing"):
            try:
                latent_list, eeg_list = process_subject(
                    file_path=file_path,
                    model=model,
                    model_info=model_info,
                    device=args.device,
                    sfreq=sfreq,
                    chunk_duration=args.chunk_duration,
                    alpha_low=args.alpha_band[0],
                    alpha_high=args.alpha_band[1],
                    verbose=args.verbose,
                )

                if len(latent_list) < 3:
                    if args.verbose:
                        print(f"  Skipping {subject_id}: only {len(latent_list)} chunks")
                    continue

                results = analyze_subject(latent_list, eeg_list, subject_id, group_name)
                subject_results.append(results)

                # Save first subject for example scatter
                if example_latent is None:
                    example_latent = latent_list
                    example_eeg = eeg_list
                    example_subject_id = subject_id

            except Exception as e:
                print(f"  Error processing {subject_id}: {e}")
                continue
    else:
        print("Running in parallel mode...")
        subject_results, example_latent, example_eeg, example_subject_id = process_subjects_parallel(
            all_subjects=all_subjects,
            checkpoint_path=args.checkpoint,
            n_channels=n_channels,
            device=args.device,
            sfreq=sfreq,
            chunk_duration=args.chunk_duration,
            alpha_low=args.alpha_band[0],
            alpha_high=args.alpha_band[1],
            n_workers=args.n_workers,
        )

    print(f"\nSuccessfully processed {len(subject_results)} subjects")

    if len(subject_results) == 0:
        print("No subjects processed successfully. Exiting.")
        return

    # Bootstrap across subjects
    print(f"\nBootstrapping across {len(subject_results)} subjects ({args.n_bootstrap} iterations)...")
    bootstrap_cis = bootstrap_across_subjects(subject_results, args.n_bootstrap)

    # Save outputs
    print("\nSaving outputs...")

    # JSON with full results
    save_results_json(
        subject_results, bootstrap_cis,
        output_dir / "decoding_sanity_results.json"
    )

    # CSV summary
    save_summary_csv(
        subject_results, bootstrap_cis,
        output_dir / "decoding_sanity_summary.csv"
    )

    # Violin plot
    plot_correlation_violins(
        subject_results, bootstrap_cis,
        output_dir / "decoding_sanity_correlations.png"
    )

    # Example scatter
    if example_latent is not None:
        plot_example_scatter(
            example_latent, example_eeg, example_subject_id,
            output_dir / "decoding_sanity_example_scatter.png"
        )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for pair, ci in bootstrap_cis.items():
        latent_name, eeg_name = pair
        if np.isfinite(ci.median):
            print(f"{latent_name} vs {eeg_name}:")
            print(f"  Median Spearman r = {ci.median:.3f} [{ci.ci_low:.3f}, {ci.ci_high:.3f}]")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
