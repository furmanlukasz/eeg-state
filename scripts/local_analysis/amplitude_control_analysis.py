#!/usr/bin/env python3
"""
Amplitude Control Analysis: Partial Correlations and PCA Baseline

This script addresses the key reviewer concern:
"Is your latent geometry just tracking signal amplitude/power?"

Two analyses:
A) Partial correlation: group differences in speed controlling for GFP/broadband
B) PCA-only baseline: compare AE latent correlations with simple PCA coordinates

Key insight: We want to show dynamics (speed, flow) are NOT reducible to amplitude.
Even if radius correlates with power, speed/flow differences should persist
after conditioning on magnitude.

Usage:
    python scripts/local_analysis/amplitude_control_analysis.py --checkpoint models/best.pt
    python scripts/local_analysis/amplitude_control_analysis.py --checkpoint models/best.pt --max_subjects 10

Outputs:
    - amplitude_control_partial_correlations.png
    - amplitude_control_pca_baseline.png
    - amplitude_control_results.json
    - amplitude_control_summary.csv
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from functools import partial
import multiprocessing as mp
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import config as cfg
from load_model import load_model_from_checkpoint, create_model, compute_latent_trajectory
from load_data import load_eeg_from_file, extract_phase_circular
from velocity import compute_speed as _compute_speed


def get_n_channels_from_checkpoint(checkpoint_path: Path) -> tuple[int, bool]:
    """Infer n_channels from model checkpoint weights."""
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
class SubjectSummary:
    """Summary statistics for one subject."""
    subject_id: str
    group: str
    label: int
    n_chunks: int

    # AE latent summaries (chunk means, then subject mean)
    ae_mean_radius: float
    ae_mean_speed: float
    ae_radius_std: float
    ae_speed_std: float

    # PCA latent summaries
    pca_mean_radius: float
    pca_mean_speed: float
    pca_radius_std: float
    pca_speed_std: float

    # EEG summaries
    mean_gfp: float
    mean_broadband: float
    mean_alpha: float

    def to_dict(self):
        return {
            "subject_id": self.subject_id,
            "group": self.group,
            "label": int(self.label),
            "n_chunks": int(self.n_chunks),
            "ae_mean_radius": float(self.ae_mean_radius),
            "ae_mean_speed": float(self.ae_mean_speed),
            "ae_radius_std": float(self.ae_radius_std),
            "ae_speed_std": float(self.ae_speed_std),
            "pca_mean_radius": float(self.pca_mean_radius),
            "pca_mean_speed": float(self.pca_mean_speed),
            "pca_radius_std": float(self.pca_radius_std),
            "pca_speed_std": float(self.pca_speed_std),
            "mean_gfp": float(self.mean_gfp),
            "mean_broadband": float(self.mean_broadband),
            "mean_alpha": float(self.mean_alpha),
        }


@dataclass
class PartialCorrelationResult:
    """Result of partial correlation analysis."""
    variable: str  # e.g., "speed"
    covariate: str  # e.g., "gfp"

    # Raw correlation (variable ~ group)
    raw_r: float
    raw_p: float

    # Partial correlation (variable ~ group | covariate)
    partial_r: float
    partial_p: float

    # Effect size change
    attenuation: float  # (raw_r - partial_r) / raw_r

    def to_dict(self):
        return {
            "variable": self.variable,
            "covariate": self.covariate,
            "raw_r": float(self.raw_r),
            "raw_p": float(self.raw_p),
            "partial_r": float(self.partial_r),
            "partial_p": float(self.partial_p),
            "attenuation": float(self.attenuation),
        }


# =============================================================================
# EEG FEATURE EXTRACTION
# =============================================================================

def compute_gfp(data: np.ndarray) -> np.ndarray:
    """Compute Global Field Power (spatial std at each timepoint)."""
    data_avg_ref = data - np.mean(data, axis=0, keepdims=True)
    return np.std(data_avg_ref, axis=0)


def compute_broadband_envelope(data: np.ndarray) -> np.ndarray:
    """Compute broadband amplitude envelope."""
    from scipy.signal import hilbert
    analytic = hilbert(data, axis=1)
    amplitude = np.abs(analytic)
    log_amplitude = np.log1p(amplitude)
    return np.mean(log_amplitude, axis=0)


def compute_alpha_envelope(
    data: np.ndarray,
    sfreq: float,
    alpha_low: float = 8.0,
    alpha_high: float = 12.0,
) -> np.ndarray:
    """Compute alpha-band envelope."""
    from scipy.signal import hilbert, butter, filtfilt

    nyq = sfreq / 2
    low = max(alpha_low / nyq, 0.01)
    high = min(alpha_high / nyq, 0.99)

    b, a = butter(4, [low, high], btype="band")
    filtered = filtfilt(b, a, data, axis=1)

    analytic = hilbert(filtered, axis=1)
    amplitude = np.abs(analytic)
    return np.mean(amplitude, axis=0)


# =============================================================================
# LATENT FEATURE EXTRACTION
# =============================================================================

def compute_latent_radius(latent: np.ndarray) -> np.ndarray:
    """Compute ||h(t)|| at each timepoint."""
    return np.linalg.norm(latent, axis=1)


def compute_latent_speed(latent: np.ndarray) -> np.ndarray:
    """Compute ||h(t+1) - h(t)||.

    Note: Delegates to centralized velocity module for consistency
    and configurable Δt/Savitzky-Golay support.
    """
    return _compute_speed(latent, method="finite_diff", delta_t=1)


def compute_pca_latent(phase_data: np.ndarray, n_components: int = 64) -> np.ndarray:
    """
    Compute PCA-based latent representation from phase features.

    This is the "PCA-only baseline" - same input features, but linear projection
    instead of neural network encoding.

    Args:
        phase_data: (n_features, n_samples) phase representation
        n_components: Number of PCA components (match AE hidden size)

    Returns:
        (n_samples, n_components) PCA latent trajectory
    """
    # Transpose to (n_samples, n_features) for PCA
    X = phase_data.T

    # Fit PCA
    n_components = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components)
    latent = pca.fit_transform(X)

    return latent


# =============================================================================
# PARTIAL CORRELATION
# =============================================================================

def partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """
    Compute partial correlation between x and y, controlling for z.

    Uses regression-based approach:
    1. Regress x on z, get residuals
    2. Regress y on z, get residuals
    3. Correlate residuals

    Args:
        x: First variable
        y: Second variable
        z: Covariate to control for

    Returns:
        (partial_r, p_value)
    """
    # Handle NaN/inf
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]

    if len(x) < 5:
        return np.nan, np.nan

    # Reshape for sklearn
    z = z.reshape(-1, 1)

    # Residualize x
    reg_x = LinearRegression().fit(z, x)
    x_resid = x - reg_x.predict(z)

    # Residualize y
    reg_y = LinearRegression().fit(z, y)
    y_resid = y - reg_y.predict(z)

    # Correlate residuals
    r, p = spearmanr(x_resid, y_resid)

    return r, p


def compute_group_correlation(values: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """
    Compute correlation between continuous values and binary group labels.

    Uses point-biserial correlation (equivalent to Spearman for binary).
    """
    mask = np.isfinite(values)
    values, labels = values[mask], labels[mask]

    if len(values) < 5:
        return np.nan, np.nan

    return spearmanr(values, labels)


# =============================================================================
# SUBJECT PROCESSING
# =============================================================================

def process_subject(
    file_path: Path,
    model,
    model_info: dict,
    device: str,
    chunk_duration: float,
    alpha_low: float,
    alpha_high: float,
    pca_n_components: int = 64,
    verbose: bool = False,
) -> dict:
    """
    Process a single subject: compute AE latent, PCA latent, and EEG summaries.

    Returns dict with per-chunk summaries.
    """
    # Load raw EEG data
    raw_data, sfreq, channel_names = load_eeg_from_file(file_path, verbose=verbose)
    n_channels = len(channel_names)
    chunk_samples = int(chunk_duration * sfreq)

    # Chunk the data
    n_samples = raw_data.shape[1]
    n_chunks = n_samples // chunk_samples

    results = {
        "ae_radius": [],
        "ae_speed": [],
        "pca_radius": [],
        "pca_speed": [],
        "gfp": [],
        "broadband": [],
        "alpha": [],
    }

    for i in range(n_chunks):
        start = i * chunk_samples
        end = start + chunk_samples
        chunk_data = raw_data[:, start:end]

        # === EEG summaries ===
        gfp = compute_gfp(chunk_data)
        broadband = compute_broadband_envelope(chunk_data)
        alpha = compute_alpha_envelope(chunk_data, sfreq, alpha_low, alpha_high)

        results["gfp"].append(np.mean(gfp))
        results["broadband"].append(np.mean(broadband))
        results["alpha"].append(np.mean(alpha))

        # === Phase representation ===
        is_meditation = file_path.suffix.lower() == ".bdf"
        phase_data = extract_phase_circular(
            chunk_data, sfreq,
            include_amplitude=model_info["include_amplitude"],
            skip_filter=is_meditation  # Already filtered for meditation
        )

        # === AE latent ===
        ae_latent = compute_latent_trajectory(model, phase_data, device)
        ae_radius = compute_latent_radius(ae_latent)
        ae_speed = compute_latent_speed(ae_latent)

        results["ae_radius"].append(np.mean(ae_radius))
        results["ae_speed"].append(np.mean(ae_speed))

        # === PCA latent (baseline) ===
        pca_latent = compute_pca_latent(phase_data, n_components=pca_n_components)
        pca_radius = compute_latent_radius(pca_latent)
        pca_speed = compute_latent_speed(pca_latent)

        results["pca_radius"].append(np.mean(pca_radius))
        results["pca_speed"].append(np.mean(pca_speed))

    return results


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
        args: (file_path, label, group_name, subject_id, chunk_duration, alpha_low, alpha_high, pca_n_components)

    Returns:
        (subject_id, group_name, label, chunk_results) or (subject_id, None, None, None) on error
    """
    global _worker_model, _worker_model_info

    file_path, label, group_name, subject_id, chunk_duration, alpha_low, alpha_high, pca_n_components = args

    try:
        chunk_results = process_subject(
            file_path=file_path,
            model=_worker_model,
            model_info=_worker_model_info,
            device="cpu",  # Workers use CPU
            chunk_duration=chunk_duration,
            alpha_low=alpha_low,
            alpha_high=alpha_high,
            pca_n_components=pca_n_components,
            verbose=False,
        )
        return (subject_id, group_name, label, chunk_results)
    except Exception as e:
        print(f"  Error processing {subject_id}: {e}")
        return (subject_id, None, None, None)


def process_subjects_parallel(
    all_subjects: list,
    checkpoint_path: Path,
    n_channels: int,
    device: str,
    chunk_duration: float,
    alpha_low: float,
    alpha_high: float,
    pca_n_components: int,
    n_workers: int = None,
) -> list[SubjectSummary]:
    """
    Process all subjects in parallel using multiprocessing with CPU workers.

    Note: Workers use CPU because MPS/CUDA contexts can't be shared across processes.
    For MPS acceleration, use --sequential mode instead (single process, GPU inference).

    Args:
        all_subjects: List of (file_path, label, group_name, subject_id) tuples
        checkpoint_path: Path to model checkpoint
        n_channels: Number of EEG channels
        device: Device (not used, workers use CPU)
        chunk_duration: Chunk duration in seconds
        alpha_low: Alpha band low frequency
        alpha_high: Alpha band high frequency
        pca_n_components: Number of PCA components
        n_workers: Number of parallel workers (default: CPU count - 1)

    Returns:
        List of SubjectSummary objects
    """
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    print(f"Using {n_workers} parallel workers (CPU inference)")
    print(f"  Note: For MPS/GPU inference, use --sequential mode")

    # Prepare args for each subject
    subject_args = [
        (fp, label, group, sid, chunk_duration, alpha_low, alpha_high, pca_n_components)
        for fp, label, group, sid in all_subjects
    ]

    summaries = []

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
                sid, group_name, label, chunk_results = future.result()

                if chunk_results is None or len(chunk_results.get("ae_radius", [])) < 3:
                    continue

                summary = create_subject_summary(sid, group_name, label, chunk_results)
                summaries.append(summary)

            except Exception as e:
                print(f"  Future error for {subject_id}: {e}")
                continue

    return summaries


def create_subject_summary(
    subject_id: str,
    group: str,
    label: int,
    chunk_results: dict,
) -> SubjectSummary:
    """Create SubjectSummary from per-chunk results."""
    return SubjectSummary(
        subject_id=subject_id,
        group=group,
        label=label,
        n_chunks=len(chunk_results["ae_radius"]),
        ae_mean_radius=np.mean(chunk_results["ae_radius"]),
        ae_mean_speed=np.mean(chunk_results["ae_speed"]),
        ae_radius_std=np.std(chunk_results["ae_radius"]),
        ae_speed_std=np.std(chunk_results["ae_speed"]),
        pca_mean_radius=np.mean(chunk_results["pca_radius"]),
        pca_mean_speed=np.mean(chunk_results["pca_speed"]),
        pca_radius_std=np.std(chunk_results["pca_radius"]),
        pca_speed_std=np.std(chunk_results["pca_speed"]),
        mean_gfp=np.mean(chunk_results["gfp"]),
        mean_broadband=np.mean(chunk_results["broadband"]),
        mean_alpha=np.mean(chunk_results["alpha"]),
    )


# =============================================================================
# ANALYSIS
# =============================================================================

def run_partial_correlation_analysis(
    summaries: list[SubjectSummary],
) -> list[PartialCorrelationResult]:
    """
    Run partial correlation analysis for speed controlling for magnitude.

    Tests whether group differences in speed persist after conditioning on GFP/broadband.
    """
    # Extract arrays
    labels = np.array([s.label for s in summaries])
    ae_speed = np.array([s.ae_mean_speed for s in summaries])
    pca_speed = np.array([s.pca_mean_speed for s in summaries])
    gfp = np.array([s.mean_gfp for s in summaries])
    broadband = np.array([s.mean_broadband for s in summaries])

    results = []

    # AE speed ~ group | GFP
    raw_r, raw_p = compute_group_correlation(ae_speed, labels)
    partial_r, partial_p = partial_correlation(ae_speed, labels.astype(float), gfp)
    attenuation = (abs(raw_r) - abs(partial_r)) / abs(raw_r) if abs(raw_r) > 0 else 0
    results.append(PartialCorrelationResult(
        variable="ae_speed",
        covariate="gfp",
        raw_r=raw_r, raw_p=raw_p,
        partial_r=partial_r, partial_p=partial_p,
        attenuation=attenuation,
    ))

    # AE speed ~ group | broadband
    partial_r, partial_p = partial_correlation(ae_speed, labels.astype(float), broadband)
    attenuation = (abs(raw_r) - abs(partial_r)) / abs(raw_r) if abs(raw_r) > 0 else 0
    results.append(PartialCorrelationResult(
        variable="ae_speed",
        covariate="broadband",
        raw_r=raw_r, raw_p=raw_p,
        partial_r=partial_r, partial_p=partial_p,
        attenuation=attenuation,
    ))

    # PCA speed ~ group | GFP
    raw_r, raw_p = compute_group_correlation(pca_speed, labels)
    partial_r, partial_p = partial_correlation(pca_speed, labels.astype(float), gfp)
    attenuation = (abs(raw_r) - abs(partial_r)) / abs(raw_r) if abs(raw_r) > 0 else 0
    results.append(PartialCorrelationResult(
        variable="pca_speed",
        covariate="gfp",
        raw_r=raw_r, raw_p=raw_p,
        partial_r=partial_r, partial_p=partial_p,
        attenuation=attenuation,
    ))

    # PCA speed ~ group | broadband
    partial_r, partial_p = partial_correlation(pca_speed, labels.astype(float), broadband)
    attenuation = (abs(raw_r) - abs(partial_r)) / abs(raw_r) if abs(raw_r) > 0 else 0
    results.append(PartialCorrelationResult(
        variable="pca_speed",
        covariate="broadband",
        raw_r=raw_r, raw_p=raw_p,
        partial_r=partial_r, partial_p=partial_p,
        attenuation=attenuation,
    ))

    return results


def compute_ae_vs_pca_correlations(
    summaries: list[SubjectSummary],
) -> dict:
    """
    Compare AE vs PCA correlations with EEG features.

    Key question: Does AE produce coordinates less aligned with raw power?
    """
    # Extract arrays
    ae_radius = np.array([s.ae_mean_radius for s in summaries])
    ae_speed = np.array([s.ae_mean_speed for s in summaries])
    pca_radius = np.array([s.pca_mean_radius for s in summaries])
    pca_speed = np.array([s.pca_mean_speed for s in summaries])
    gfp = np.array([s.mean_gfp for s in summaries])
    broadband = np.array([s.mean_broadband for s in summaries])

    results = {}

    # AE radius vs power
    r, p = spearmanr(ae_radius, gfp)
    results["ae_radius_vs_gfp"] = {"r": r, "p": p}
    r, p = spearmanr(ae_radius, broadband)
    results["ae_radius_vs_broadband"] = {"r": r, "p": p}

    # PCA radius vs power
    r, p = spearmanr(pca_radius, gfp)
    results["pca_radius_vs_gfp"] = {"r": r, "p": p}
    r, p = spearmanr(pca_radius, broadband)
    results["pca_radius_vs_broadband"] = {"r": r, "p": p}

    # AE speed vs power
    r, p = spearmanr(ae_speed, gfp)
    results["ae_speed_vs_gfp"] = {"r": r, "p": p}
    r, p = spearmanr(ae_speed, broadband)
    results["ae_speed_vs_broadband"] = {"r": r, "p": p}

    # PCA speed vs power
    r, p = spearmanr(pca_speed, gfp)
    results["pca_speed_vs_gfp"] = {"r": r, "p": p}
    r, p = spearmanr(pca_speed, broadband)
    results["pca_speed_vs_broadband"] = {"r": r, "p": p}

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_partial_correlations(
    partial_results: list[PartialCorrelationResult],
    output_path: Path,
):
    """
    Plot raw vs partial correlations showing effect of controlling for amplitude.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Organize by variable
    ae_results = [r for r in partial_results if r.variable == "ae_speed"]
    pca_results = [r for r in partial_results if r.variable == "pca_speed"]

    for ax, results, title in [
        (axes[0], ae_results, "AE Speed"),
        (axes[1], pca_results, "PCA Speed"),
    ]:
        covariates = [r.covariate for r in results]
        raw_rs = [r.raw_r for r in results]
        partial_rs = [r.partial_r for r in results]

        x = np.arange(len(covariates))
        width = 0.35

        bars1 = ax.bar(x - width/2, raw_rs, width, label='Raw', color='#1f77b4', alpha=0.8)
        bars2 = ax.bar(x + width/2, partial_rs, width, label='Partial', color='#ff7f0e', alpha=0.8)

        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Spearman r (speed ~ group)')
        ax.set_title(f'{title} vs Group\n(controlling for amplitude)')
        ax.set_xticks(x)
        ax.set_xticklabels([f'| {c}' for c in covariates])
        ax.legend()
        ax.set_ylim(-0.8, 0.8)

        # Add attenuation annotations
        for i, r in enumerate(results):
            if np.isfinite(r.attenuation):
                ax.annotate(f'{r.attenuation*100:.0f}% att.',
                           xy=(i + width/2, r.partial_r),
                           xytext=(0, 5 if r.partial_r > 0 else -15),
                           textcoords='offset points',
                           ha='center', fontsize=8, color='gray')

    plt.suptitle('Partial Correlation Control: Group Differences in Speed\nAfter Conditioning on Amplitude',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_ae_vs_pca_baseline(
    summaries: list[SubjectSummary],
    output_path: Path,
):
    """
    Plot AE vs PCA correlations with amplitude features.

    Shows whether AE coordinates are less "amplitude-aligned" than PCA.
    """
    # Extract arrays
    ae_radius = np.array([s.ae_mean_radius for s in summaries])
    pca_radius = np.array([s.pca_mean_radius for s in summaries])
    ae_speed = np.array([s.ae_mean_speed for s in summaries])
    pca_speed = np.array([s.pca_mean_speed for s in summaries])
    gfp = np.array([s.mean_gfp for s in summaries])
    broadband = np.array([s.mean_broadband for s in summaries])
    labels = np.array([s.label for s in summaries])
    groups = np.array([s.group for s in summaries])

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Get unique groups for coloring
    unique_groups = np.unique(groups)
    colors = {'expert': '#1f77b4', 'novice': '#ff7f0e', 'Expert': '#1f77b4', 'Novice': '#ff7f0e',
              'HC': '#1f77b4', 'MCI': '#ff7f0e', 'AD': '#d62728'}

    # Panel A: AE radius vs broadband
    ax = axes[0, 0]
    for g in unique_groups:
        mask = groups == g
        ax.scatter(broadband[mask], ae_radius[mask], c=colors.get(g, 'gray'),
                  label=g, alpha=0.6, s=50)
    r, p = spearmanr(broadband, ae_radius)
    ax.set_xlabel('Broadband Power')
    ax.set_ylabel('AE Radius ||h||')
    ax.set_title(f'AE Radius vs Amplitude\nr = {r:.3f}')
    ax.legend()

    # Panel B: PCA radius vs broadband
    ax = axes[0, 1]
    for g in unique_groups:
        mask = groups == g
        ax.scatter(broadband[mask], pca_radius[mask], c=colors.get(g, 'gray'),
                  label=g, alpha=0.6, s=50)
    r, p = spearmanr(broadband, pca_radius)
    ax.set_xlabel('Broadband Power')
    ax.set_ylabel('PCA Radius ||z_pca||')
    ax.set_title(f'PCA Radius vs Amplitude\nr = {r:.3f}')
    ax.legend()

    # Panel C: AE speed vs broadband
    ax = axes[1, 0]
    for g in unique_groups:
        mask = groups == g
        ax.scatter(broadband[mask], ae_speed[mask], c=colors.get(g, 'gray'),
                  label=g, alpha=0.6, s=50)
    r, p = spearmanr(broadband, ae_speed)
    ax.set_xlabel('Broadband Power')
    ax.set_ylabel('AE Speed ||dh/dt||')
    ax.set_title(f'AE Speed vs Amplitude\nr = {r:.3f}')
    ax.legend()

    # Panel D: PCA speed vs broadband
    ax = axes[1, 1]
    for g in unique_groups:
        mask = groups == g
        ax.scatter(broadband[mask], pca_speed[mask], c=colors.get(g, 'gray'),
                  label=g, alpha=0.6, s=50)
    r, p = spearmanr(broadband, pca_speed)
    ax.set_xlabel('Broadband Power')
    ax.set_ylabel('PCA Speed ||dz_pca/dt||')
    ax.set_title(f'PCA Speed vs Amplitude\nr = {r:.3f}')
    ax.legend()

    plt.suptitle('AE vs PCA Baseline: Correlation with Signal Amplitude',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_group_differences(
    summaries: list[SubjectSummary],
    output_path: Path,
):
    """
    Plot group differences in AE vs PCA metrics.
    """
    # Organize by group
    groups = {}
    for s in summaries:
        if s.group not in groups:
            groups[s.group] = []
        groups[s.group].append(s)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    group_names = list(groups.keys())
    colors = {'expert': '#1f77b4', 'novice': '#ff7f0e', 'Expert': '#1f77b4', 'Novice': '#ff7f0e',
              'HC': '#1f77b4', 'MCI': '#ff7f0e', 'AD': '#d62728'}

    # Panel A: Speed comparison
    ax = axes[0]
    x = np.arange(len(group_names))
    width = 0.35

    ae_means = [np.mean([s.ae_mean_speed for s in groups[g]]) for g in group_names]
    ae_stds = [np.std([s.ae_mean_speed for s in groups[g]]) / np.sqrt(len(groups[g])) for g in group_names]
    pca_means = [np.mean([s.pca_mean_speed for s in groups[g]]) for g in group_names]
    pca_stds = [np.std([s.pca_mean_speed for s in groups[g]]) / np.sqrt(len(groups[g])) for g in group_names]

    ax.bar(x - width/2, ae_means, width, yerr=ae_stds, label='AE', color='#1f77b4', alpha=0.8, capsize=5)
    ax.bar(x + width/2, pca_means, width, yerr=pca_stds, label='PCA', color='#ff7f0e', alpha=0.8, capsize=5)

    ax.set_ylabel('Mean Speed')
    ax.set_title('Latent Speed by Group')
    ax.set_xticks(x)
    ax.set_xticklabels(group_names)
    ax.legend()

    # Panel B: Radius comparison
    ax = axes[1]

    ae_means = [np.mean([s.ae_mean_radius for s in groups[g]]) for g in group_names]
    ae_stds = [np.std([s.ae_mean_radius for s in groups[g]]) / np.sqrt(len(groups[g])) for g in group_names]
    pca_means = [np.mean([s.pca_mean_radius for s in groups[g]]) for g in group_names]
    pca_stds = [np.std([s.pca_mean_radius for s in groups[g]]) / np.sqrt(len(groups[g])) for g in group_names]

    ax.bar(x - width/2, ae_means, width, yerr=ae_stds, label='AE', color='#1f77b4', alpha=0.8, capsize=5)
    ax.bar(x + width/2, pca_means, width, yerr=pca_stds, label='PCA', color='#ff7f0e', alpha=0.8, capsize=5)

    ax.set_ylabel('Mean Radius')
    ax.set_title('Latent Radius by Group')
    ax.set_xticks(x)
    ax.set_xticklabels(group_names)
    ax.legend()

    plt.suptitle('Group Differences: AE vs PCA Latent Metrics', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# OUTPUT
# =============================================================================

def save_results(
    summaries: list[SubjectSummary],
    partial_results: list[PartialCorrelationResult],
    ae_pca_correlations: dict,
    output_dir: Path,
):
    """Save all results to JSON and CSV."""

    # Full JSON
    results = {
        "timestamp": datetime.now().isoformat(),
        "n_subjects": len(summaries),
        "subjects": [s.to_dict() for s in summaries],
        "partial_correlations": [r.to_dict() for r in partial_results],
        "ae_vs_pca_correlations": {
            k: {"r": float(v["r"]), "p": float(v["p"])}
            for k, v in ae_pca_correlations.items()
        },
    }

    with open(output_dir / "amplitude_control_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {output_dir / 'amplitude_control_results.json'}")

    # Summary CSV
    rows = []
    for r in partial_results:
        rows.append({
            "analysis": "partial_correlation",
            "variable": r.variable,
            "covariate": r.covariate,
            "raw_r": r.raw_r,
            "partial_r": r.partial_r,
            "attenuation_pct": r.attenuation * 100,
        })

    for key, val in ae_pca_correlations.items():
        rows.append({
            "analysis": "ae_pca_comparison",
            "variable": key,
            "covariate": "",
            "raw_r": val["r"],
            "partial_r": np.nan,
            "attenuation_pct": np.nan,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "amplitude_control_summary.csv", index=False)
    print(f"Saved: {output_dir / 'amplitude_control_summary.csv'}")


# =============================================================================
# MAIN
# =============================================================================

def create_output_dir(base_dir: Path) -> Path:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / f"amplitude_control_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Amplitude control analysis: partial correlations and PCA baseline"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=cfg.CHECKPOINT_PATH,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--out_dir", type=Path, default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--alpha_band", type=float, nargs=2, default=[8.0, 12.0],
        help="Alpha band range in Hz"
    )
    parser.add_argument(
        "--chunk_duration", type=float, default=cfg.CHUNK_DURATION,
        help="Chunk duration in seconds"
    )
    parser.add_argument(
        "--max_subjects", type=int, default=None,
        help="Maximum subjects to process"
    )
    parser.add_argument(
        "--device", type=str, default=cfg.DEVICE,
        help="Compute device"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Dataset: greek_resting or meditation_bids"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Verbose output"
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

    # Determine dataset
    if args.dataset is None:
        checkpoint_name = args.checkpoint.name.lower()
        dataset = "meditation_bids" if "meditation" in checkpoint_name else "greek_resting"
    else:
        dataset = args.dataset

    cfg.DATASET = dataset
    data_dir = cfg.DATA_PATHS.get(dataset, cfg.DATA_DIR)

    # Setup output
    if args.out_dir is None:
        base_dir = cfg.ensure_output_dir()
        output_dir = create_output_dir(base_dir)
    else:
        output_dir = args.out_dir
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Dataset: {dataset}")
    print(f"Data directory: {data_dir}")
    print(f"Checkpoint: {args.checkpoint}")

    # Save parameters
    params = {
        "checkpoint": str(args.checkpoint),
        "dataset": dataset,
        "alpha_band": args.alpha_band,
        "chunk_duration": args.chunk_duration,
        "device": args.device,
    }
    with open(output_dir / "parameters.json", 'w') as f:
        json.dump(params, f, indent=2)

    # Load model
    print("\nLoading model...")
    model_info = load_model_from_checkpoint(args.checkpoint, args.device)
    n_channels, _ = get_n_channels_from_checkpoint(args.checkpoint)
    print(f"  Inferred n_channels: {n_channels}")

    model = create_model(n_channels, model_info, args.device)

    # Get data files
    print("\nDiscovering data files...")
    data_files = cfg.get_data_files_via_config()
    groups = cfg.get_subjects_by_group_unified(data_files)

    all_subjects = []
    for group_key, subjects in groups.items():
        all_subjects.extend(subjects)

    if args.max_subjects:
        all_subjects = all_subjects[:args.max_subjects]

    print(f"Processing {len(all_subjects)} subjects...")

    # Process all subjects (parallel or sequential)
    if args.sequential:
        print("Running in sequential mode...")
        summaries = []
        for file_path, label, group_name, subject_id in tqdm(all_subjects, desc="Processing"):
            try:
                chunk_results = process_subject(
                    file_path=file_path,
                    model=model,
                    model_info=model_info,
                    device=args.device,
                    chunk_duration=args.chunk_duration,
                    alpha_low=args.alpha_band[0],
                    alpha_high=args.alpha_band[1],
                    pca_n_components=model_info["hidden_size"],
                    verbose=args.verbose,
                )

                if len(chunk_results["ae_radius"]) < 3:
                    continue

                summary = create_subject_summary(subject_id, group_name, label, chunk_results)
                summaries.append(summary)

            except Exception as e:
                print(f"  Error processing {subject_id}: {e}")
                continue
    else:
        print("Running in parallel mode...")
        summaries = process_subjects_parallel(
            all_subjects=all_subjects,
            checkpoint_path=args.checkpoint,
            n_channels=n_channels,
            device=args.device,
            chunk_duration=args.chunk_duration,
            alpha_low=args.alpha_band[0],
            alpha_high=args.alpha_band[1],
            pca_n_components=model_info["hidden_size"],
            n_workers=args.n_workers,
        )

    print(f"\nSuccessfully processed {len(summaries)} subjects")

    if len(summaries) < 5:
        print("Not enough subjects for analysis. Exiting.")
        return

    # Run analyses
    print("\nRunning partial correlation analysis...")
    partial_results = run_partial_correlation_analysis(summaries)

    print("Computing AE vs PCA correlations...")
    ae_pca_corrs = compute_ae_vs_pca_correlations(summaries)

    # Save results
    print("\nSaving outputs...")
    save_results(summaries, partial_results, ae_pca_corrs, output_dir)

    # Generate plots
    plot_partial_correlations(partial_results, output_dir / "amplitude_control_partial_correlations.png")
    plot_ae_vs_pca_baseline(summaries, output_dir / "amplitude_control_pca_baseline.png")
    plot_group_differences(summaries, output_dir / "amplitude_control_group_differences.png")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY: Partial Correlation Control")
    print("=" * 70)
    print("\nKey question: Do group differences in speed persist after controlling for amplitude?\n")

    for r in partial_results:
        if r.variable == "ae_speed":
            print(f"{r.variable} ~ group | {r.covariate}:")
            print(f"  Raw r = {r.raw_r:.3f}")
            print(f"  Partial r = {r.partial_r:.3f}")
            print(f"  Attenuation = {r.attenuation*100:.1f}%")
            if abs(r.partial_r) > 0.1:
                print(f"  --> Effect PERSISTS after controlling for {r.covariate}")
            print()

    print("\n" + "=" * 70)
    print("SUMMARY: AE vs PCA Baseline Correlations with Amplitude")
    print("=" * 70)
    print("\nKey question: Is AE less 'amplitude-aligned' than raw PCA?\n")

    for key in ["ae_radius_vs_broadband", "pca_radius_vs_broadband",
                "ae_speed_vs_broadband", "pca_speed_vs_broadband"]:
        r = ae_pca_corrs[key]["r"]
        print(f"  {key}: r = {r:.3f}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
