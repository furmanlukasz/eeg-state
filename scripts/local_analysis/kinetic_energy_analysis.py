#!/usr/bin/env python3
"""
kinetic_energy_analysis.py

Kinetic Energy Proxy Analysis for Latent Trajectories

Computes dynamical activity metrics from latent space velocities, providing
an "energy-of-motion" proxy that characterizes how vigorously the system
traverses its manifold.

Theoretical motivation:
- NOT metabolic energy (scalp EEG is not a direct measure of metabolism)
- A dynamical activity proxy: relates to "neural reconfiguration effort"
- Interpretable as "state-space mobility" or "trajectory activity"

Key metrics:
1. Instantaneous kinetic energy: E_k(t) = ||v(t)||^2
2. Mean kinetic energy: E[E_k] (global activity level)
3. Energy variance / CV: intermittency of motion
4. Energy tail index: burstiness (rare high-energy episodes)
5. Energy landscape: spatial distribution of E_k across manifold

Integration:
- Works with any simulation script that produces latent trajectories
- Compatible with coupled_oscillator_sim.py and simulation_analysis.py
- Uses the same autoencoder and embedding infrastructure

Usage:
    # Run on existing simulation results
    python kinetic_energy_analysis.py --input results/simulations/coupled_sl_*/trajectories.npz

    # Run full pipeline with new simulation
    python kinetic_energy_analysis.py --simulate --duration 160 --coupling 5.0

    # Analyze with custom parameters
    python kinetic_energy_analysis.py --input trajectories.npz --velocity-method savgol --window 7
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import entropy as scipy_entropy, kurtosis, skew
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class KineticEnergyMetrics:
    """Container for kinetic energy proxy statistics."""
    # Basic statistics
    mean_energy: float          # E[||v||^2] - global activity level
    std_energy: float           # Std(||v||^2)
    cv_energy: float            # CV = std/mean - intermittency
    median_energy: float        # Median ||v||^2 - robust center

    # Distribution shape
    skewness: float             # Asymmetry (positive = right tail)
    kurtosis: float             # Tail heaviness (>3 = heavy tails)
    tail_index: float           # Fraction of energy in top 5%

    # Temporal structure
    autocorr_lag1: float        # First-order autocorrelation
    burst_frequency: float      # High-energy episodes per second
    mean_burst_duration: float  # Avg duration of high-energy bursts

    # Summary
    total_energy: float         # Sum(||v||^2) - total "effort"
    n_samples: int              # Number of time points


@dataclass
class EnergyLandscape:
    """Spatial distribution of kinetic energy on the manifold."""
    grid_x: np.ndarray          # X coordinates of grid
    grid_y: np.ndarray          # Y coordinates of grid
    mean_energy: np.ndarray     # Mean E_k per bin
    std_energy: np.ndarray      # Std E_k per bin
    occupancy: np.ndarray       # Sample count per bin
    bounds: Tuple[float, float, float, float]


# =============================================================================
# VELOCITY COMPUTATION
# =============================================================================

def compute_velocity(
    trajectory: np.ndarray,
    method: str = "savgol",
    delta_t: int = 1,
    savgol_window: int = 5,
    savgol_order: int = 2,
) -> np.ndarray:
    """
    Compute velocity vectors from trajectory.

    Args:
        trajectory: (n_samples, n_dims) trajectory in latent/embedded space
        method: "finite_diff" or "savgol" (Savitzky-Golay derivative)
        delta_t: Time step for finite differences (samples)
        savgol_window: Window size for Savitzky-Golay (must be odd)
        savgol_order: Polynomial order for Savitzky-Golay

    Returns:
        velocity: (n_samples - delta_t, n_dims) velocity vectors
    """
    if method == "finite_diff":
        # Simple finite difference: v(t) = x(t+dt) - x(t)
        velocity = np.diff(trajectory, n=delta_t, axis=0) / delta_t
    elif method == "savgol":
        # Savitzky-Golay derivative estimation (smoother)
        if savgol_window % 2 == 0:
            savgol_window += 1  # Must be odd
        if len(trajectory) <= savgol_window:
            # Fallback to finite diff if too short
            return compute_velocity(trajectory, method="finite_diff", delta_t=delta_t)
        velocity = savgol_filter(
            trajectory, savgol_window, savgol_order,
            deriv=1, axis=0, mode='interp'
        )
    else:
        raise ValueError(f"Unknown velocity method: {method}")

    return velocity


def compute_kinetic_energy(velocity: np.ndarray) -> np.ndarray:
    """
    Compute instantaneous kinetic energy proxy: E_k(t) = ||v(t)||^2

    Args:
        velocity: (n_samples, n_dims) velocity vectors

    Returns:
        energy: (n_samples,) kinetic energy at each time point
    """
    return np.sum(velocity ** 2, axis=1)


# =============================================================================
# KINETIC ENERGY METRICS
# =============================================================================

def detect_energy_bursts(
    energy: np.ndarray,
    threshold_percentile: float = 90,
    min_duration: int = 3,
) -> List[Tuple[int, int]]:
    """
    Detect high-energy burst episodes.

    Args:
        energy: (n_samples,) kinetic energy time series
        threshold_percentile: Percentile threshold for "high" energy
        min_duration: Minimum burst duration (samples)

    Returns:
        List of (start, end) tuples for each burst
    """
    threshold = np.percentile(energy, threshold_percentile)
    is_high = energy > threshold

    bursts = []
    in_burst = False
    start = 0

    for i, high in enumerate(is_high):
        if high and not in_burst:
            in_burst = True
            start = i
        elif not high and in_burst:
            in_burst = False
            if i - start >= min_duration:
                bursts.append((start, i))

    # Handle burst at end
    if in_burst and len(is_high) - start >= min_duration:
        bursts.append((start, len(is_high)))

    return bursts


def compute_autocorrelation(x: np.ndarray, lag: int = 1) -> float:
    """Compute autocorrelation at specified lag."""
    if len(x) <= lag:
        return 0.0
    x_centered = x - np.mean(x)
    var = np.var(x)
    if var < 1e-10:
        return 0.0
    autocov = np.mean(x_centered[:-lag] * x_centered[lag:])
    return autocov / var


def compute_kinetic_energy_metrics(
    trajectory: np.ndarray,
    sfreq: float = 250.0,
    velocity_method: str = "savgol",
    savgol_window: int = 5,
    burst_threshold_pct: float = 90,
) -> KineticEnergyMetrics:
    """
    Compute comprehensive kinetic energy metrics from trajectory.

    Args:
        trajectory: (n_samples, n_dims) latent/embedded trajectory
        sfreq: Sampling frequency (Hz) for temporal metrics
        velocity_method: "finite_diff" or "savgol"
        savgol_window: Window for Savitzky-Golay smoothing
        burst_threshold_pct: Percentile for burst detection

    Returns:
        KineticEnergyMetrics containing all computed statistics
    """
    # Compute velocity and kinetic energy
    velocity = compute_velocity(trajectory, method=velocity_method, savgol_window=savgol_window)
    energy = compute_kinetic_energy(velocity)

    # Basic statistics
    mean_e = float(np.mean(energy))
    std_e = float(np.std(energy))
    cv_e = std_e / mean_e if mean_e > 0 else 0.0
    median_e = float(np.median(energy))

    # Distribution shape
    skew_e = float(skew(energy))
    kurt_e = float(kurtosis(energy))  # Excess kurtosis (normal = 0)

    # Tail index: fraction of total energy in top 5%
    p95 = np.percentile(energy, 95)
    tail_energy = energy[energy >= p95].sum()
    tail_idx = tail_energy / energy.sum() if energy.sum() > 0 else 0.0

    # Temporal structure
    autocorr = compute_autocorrelation(energy, lag=1)

    # Burst detection
    bursts = detect_energy_bursts(energy, threshold_percentile=burst_threshold_pct)
    n_bursts = len(bursts)
    duration_s = len(energy) / sfreq
    burst_freq = n_bursts / duration_s if duration_s > 0 else 0.0

    if bursts:
        mean_burst_dur = np.mean([end - start for start, end in bursts]) / sfreq
    else:
        mean_burst_dur = 0.0

    return KineticEnergyMetrics(
        mean_energy=mean_e,
        std_energy=std_e,
        cv_energy=cv_e,
        median_energy=median_e,
        skewness=skew_e,
        kurtosis=kurt_e,
        tail_index=float(tail_idx),
        autocorr_lag1=float(autocorr),
        burst_frequency=float(burst_freq),
        mean_burst_duration=float(mean_burst_dur),
        total_energy=float(energy.sum()),
        n_samples=len(energy),
    )


# =============================================================================
# ENERGY LANDSCAPE
# =============================================================================

def compute_energy_landscape(
    embedded: np.ndarray,
    velocity: np.ndarray,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    grid_size: int = 20,
    min_samples: int = 5,
) -> EnergyLandscape:
    """
    Compute spatial distribution of kinetic energy on the 2D manifold.

    Args:
        embedded: (n_samples, 2) 2D embedding coordinates
        velocity: (n_samples, n_dims) velocity vectors (can be higher-dim)
        bounds: (xmin, xmax, ymin, ymax) or None to auto-compute
        grid_size: Number of bins per dimension
        min_samples: Minimum samples per bin for valid statistics

    Returns:
        EnergyLandscape with spatial energy distribution
    """
    # Handle size mismatch between embedded and velocity
    n_vel = len(velocity)
    n_emb = len(embedded)
    if n_vel != n_emb:
        # Trim embedded to match velocity (velocity is 1 shorter from diff)
        embedded = embedded[:n_vel]

    # Compute kinetic energy
    energy = compute_kinetic_energy(velocity)

    # Set bounds
    if bounds is None:
        margin = 0.05
        xmin, xmax = embedded[:, 0].min(), embedded[:, 0].max()
        ymin, ymax = embedded[:, 1].min(), embedded[:, 1].max()
        x_range = xmax - xmin
        y_range = ymax - ymin
        xmin -= margin * x_range
        xmax += margin * x_range
        ymin -= margin * y_range
        ymax += margin * y_range
        bounds = (xmin, xmax, ymin, ymax)

    xmin, xmax, ymin, ymax = bounds

    # Create grid
    x_edges = np.linspace(xmin, xmax, grid_size + 1)
    y_edges = np.linspace(ymin, ymax, grid_size + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    # Bin energies
    mean_grid = np.full((grid_size, grid_size), np.nan)
    std_grid = np.full((grid_size, grid_size), np.nan)
    count_grid = np.zeros((grid_size, grid_size), dtype=int)

    # Compute bin indices
    x_idx = np.clip(np.digitize(embedded[:, 0], x_edges) - 1, 0, grid_size - 1)
    y_idx = np.clip(np.digitize(embedded[:, 1], y_edges) - 1, 0, grid_size - 1)

    # Aggregate energy per bin
    for i in range(grid_size):
        for j in range(grid_size):
            mask = (x_idx == i) & (y_idx == j)
            count = mask.sum()
            count_grid[j, i] = count
            if count >= min_samples:
                bin_energy = energy[mask]
                mean_grid[j, i] = np.mean(bin_energy)
                std_grid[j, i] = np.std(bin_energy)

    return EnergyLandscape(
        grid_x=x_centers,
        grid_y=y_centers,
        mean_energy=mean_grid,
        std_energy=std_grid,
        occupancy=count_grid,
        bounds=bounds,
    )


# =============================================================================
# PER-REGIME ANALYSIS
# =============================================================================

def compute_regime_energy_metrics(
    trajectory: np.ndarray,
    regime_labels: np.ndarray,
    regime_names: List[str],
    sfreq: float = 250.0,
    velocity_method: str = "savgol",
) -> Dict[str, KineticEnergyMetrics]:
    """
    Compute kinetic energy metrics for each regime.

    Args:
        trajectory: (n_samples, n_dims) latent trajectory
        regime_labels: (n_samples,) regime label per sample
        regime_names: List of unique regime names
        sfreq: Sampling frequency
        velocity_method: Velocity computation method

    Returns:
        Dict mapping regime name -> KineticEnergyMetrics
    """
    results = {}

    # Get unique regime ids
    unique_ids = np.unique(regime_labels)

    # Map regime ids to names
    # Handle case where regime_names has duplicates (multiple cycles)
    id_to_name = {}
    for rid in unique_ids:
        if rid < len(regime_names):
            id_to_name[rid] = regime_names[rid]
        else:
            id_to_name[rid] = f"regime_{rid}"

    # Compute velocity once (we'll slice it per regime)
    velocity = compute_velocity(trajectory, method=velocity_method)

    # Aggregate by unique regime name
    name_to_metrics = {}

    for rid, name in id_to_name.items():
        mask = regime_labels == rid
        # Adjust mask for velocity length
        mask_vel = mask[:-1] if velocity.shape[0] == len(mask) - 1 else mask[:velocity.shape[0]]

        if mask_vel.sum() > 50:  # Minimum samples
            regime_velocity = velocity[mask_vel]
            regime_energy = compute_kinetic_energy(regime_velocity)

            # If this name already exists, combine the data
            if name in name_to_metrics:
                # Concatenate energies for combined statistics
                existing = name_to_metrics[name]["energies"]
                name_to_metrics[name]["energies"] = np.concatenate([existing, regime_energy])
            else:
                name_to_metrics[name] = {"energies": regime_energy}

    # Compute final metrics from combined energies
    for name, data in name_to_metrics.items():
        energy = data["energies"]

        # Basic statistics
        mean_e = float(np.mean(energy))
        std_e = float(np.std(energy))
        cv_e = std_e / mean_e if mean_e > 0 else 0.0
        median_e = float(np.median(energy))

        # Distribution shape
        skew_e = float(skew(energy)) if len(energy) > 2 else 0.0
        kurt_e = float(kurtosis(energy)) if len(energy) > 3 else 0.0

        # Tail index
        p95 = np.percentile(energy, 95)
        tail_energy = energy[energy >= p95].sum()
        tail_idx = tail_energy / energy.sum() if energy.sum() > 0 else 0.0

        # Temporal metrics (simplified for concatenated data)
        autocorr = compute_autocorrelation(energy, lag=1)
        bursts = detect_energy_bursts(energy)
        duration_s = len(energy) / sfreq
        burst_freq = len(bursts) / duration_s if duration_s > 0 else 0.0
        mean_burst_dur = np.mean([e - s for s, e in bursts]) / sfreq if bursts else 0.0

        results[name] = KineticEnergyMetrics(
            mean_energy=mean_e,
            std_energy=std_e,
            cv_energy=cv_e,
            median_energy=median_e,
            skewness=skew_e,
            kurtosis=kurt_e,
            tail_index=float(tail_idx),
            autocorr_lag1=float(autocorr),
            burst_frequency=float(burst_freq),
            mean_burst_duration=float(mean_burst_dur),
            total_energy=float(energy.sum()),
            n_samples=len(energy),
        )

    return results


# =============================================================================
# DISCRIMINABILITY ANALYSIS
# =============================================================================

def compute_energy_discriminability(
    trajectory: np.ndarray,
    regime_labels: np.ndarray,
    regime_names: List[str],
    window_size: int = 50,
    velocity_method: str = "savgol",
) -> Dict[str, Dict]:
    """
    Compute per-window energy statistics for discriminability analysis.

    Args:
        trajectory: (n_samples, n_dims) latent trajectory
        regime_labels: (n_samples,) regime label per sample
        regime_names: List of unique regime names
        window_size: Window size for per-window statistics
        velocity_method: Velocity computation method

    Returns:
        Dict with discriminability statistics
    """
    from scipy.stats import f_oneway

    # Compute velocity and energy
    velocity = compute_velocity(trajectory, method=velocity_method)
    energy = compute_kinetic_energy(velocity)

    # Adjust labels for velocity length
    labels = regime_labels[:len(energy)]

    # Get unique regime names (handle duplicates from cycles)
    unique_names = list(dict.fromkeys(regime_names))
    id_to_name = {i: regime_names[i] for i in range(len(regime_names))}

    # Collect per-window energies by regime
    window_energies = {name: [] for name in unique_names}

    for name in unique_names:
        # Find all regime IDs that map to this name
        matching_ids = [i for i, n in enumerate(regime_names) if n == name]
        mask = np.isin(labels, matching_ids)
        regime_energy = energy[mask]

        # Compute windowed statistics
        n_windows = len(regime_energy) // window_size
        for w in range(n_windows):
            window = regime_energy[w * window_size : (w + 1) * window_size]
            if len(window) == window_size:
                window_energies[name].append(float(np.mean(window)))

    # Convert to arrays
    for name in unique_names:
        window_energies[name] = np.array(window_energies[name])

    # ANOVA for discriminability
    groups = [window_energies[name] for name in unique_names if len(window_energies[name]) > 1]

    if len(groups) >= 2 and all(len(g) > 1 for g in groups):
        f_stat, p_val = f_oneway(*groups)

        # Eta-squared effect size
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)
        ss_total = np.sum((all_data - grand_mean) ** 2)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0

        effect_label = "large" if eta_sq > 0.14 else "medium" if eta_sq > 0.06 else "small"
    else:
        f_stat, p_val, eta_sq, effect_label = 0.0, 1.0, 0.0, "n/a"

    return {
        "window_energies": {name: vals.tolist() for name, vals in window_energies.items()},
        "f_statistic": float(f_stat),
        "p_value": float(p_val),
        "eta_squared": float(eta_sq),
        "effect_size": effect_label,
        "n_windows": {name: len(vals) for name, vals in window_energies.items()},
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_kinetic_energy_analysis(
    trajectory: np.ndarray,
    embedded: np.ndarray,
    regime_labels: np.ndarray,
    regime_names: List[str],
    regime_colors: Optional[Dict[str, str]] = None,
    sfreq: float = 250.0,
    velocity_method: str = "savgol",
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Create comprehensive kinetic energy analysis figure.

    Panels:
    A) Energy time series with regime coloring
    B) Energy distribution per regime (violin plots)
    C) Energy landscape on 2D embedding
    D) Energy vs speed scatter (sanity check)

    Args:
        trajectory: (n_samples, n_dims) full latent trajectory
        embedded: (n_samples, 2) 2D embedding
        regime_labels: (n_samples,) regime labels
        regime_names: List of regime names
        regime_colors: Dict mapping name -> color
        sfreq: Sampling frequency
        velocity_method: Velocity computation method
        output_path: Optional path to save figure
        show: Whether to display figure

    Returns:
        Matplotlib Figure
    """
    # Default colors
    if regime_colors is None:
        regime_colors = {
            "global": "#1f77b4",
            "cluster": "#ff7f0e",
            "sparse": "#2ca02c",
            "ring": "#d62728",
        }

    # Compute velocity and energy
    velocity = compute_velocity(trajectory, method=velocity_method)
    energy = compute_kinetic_energy(velocity)
    speed = np.sqrt(energy)  # For scatter plot

    # Get unique regime names
    unique_names = list(dict.fromkeys(regime_names))

    # Adjust labels for velocity length
    labels = regime_labels[:len(energy)]
    time = np.arange(len(energy)) / sfreq

    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

    # Panel A: Energy time series
    ax_a = fig.add_subplot(gs[0, 0])

    # Plot with regime coloring
    for name in unique_names:
        matching_ids = [i for i, n in enumerate(regime_names) if n == name]
        mask = np.isin(labels, matching_ids)
        color = regime_colors.get(name, "#888888")
        ax_a.scatter(time[mask], energy[mask], c=color, s=1, alpha=0.3, label=name)

    # Smoothed trend
    from scipy.ndimage import uniform_filter1d
    smoothed = uniform_filter1d(energy, size=int(sfreq * 0.5))  # 0.5s smoothing
    ax_a.plot(time, smoothed, 'k-', lw=1.5, alpha=0.8, label='Smoothed')

    ax_a.set_xlabel("Time (s)")
    ax_a.set_ylabel("Kinetic Energy (||v||²)")
    ax_a.set_title("A) Kinetic Energy Time Series", fontweight='bold')
    ax_a.legend(markerscale=5)

    # Panel B: Energy distribution per regime (violin plots)
    ax_b = fig.add_subplot(gs[0, 1])

    data_for_plot = []
    positions = []
    colors_for_plot = []

    for i, name in enumerate(unique_names):
        matching_ids = [j for j, n in enumerate(regime_names) if n == name]
        mask = np.isin(labels, matching_ids)
        regime_energy = energy[mask]
        if len(regime_energy) > 0:
            data_for_plot.append(regime_energy)
            positions.append(i)
            colors_for_plot.append(regime_colors.get(name, "#888888"))

    if data_for_plot:
        parts = ax_b.violinplot(data_for_plot, positions=positions, showmeans=True, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors_for_plot[i])
            pc.set_alpha(0.7)
        for partname in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
            if partname in parts:
                parts[partname].set_color('black')

    ax_b.set_xticks(range(len(unique_names)))
    ax_b.set_xticklabels(unique_names)
    ax_b.set_ylabel("Kinetic Energy (||v||²)")
    ax_b.set_title("B) Energy Distribution by Regime", fontweight='bold')

    # Panel C: Energy landscape
    ax_c = fig.add_subplot(gs[1, 0])

    # Compute velocity in embedded space for visualization
    embedded_vel = compute_velocity(embedded[:len(energy)+1], method=velocity_method)
    landscape = compute_energy_landscape(embedded[:len(energy)], embedded_vel)

    # Plot energy heatmap
    im = ax_c.imshow(
        landscape.mean_energy,
        origin='lower',
        extent=list(landscape.bounds),
        cmap='YlOrRd',
        aspect='equal',
    )
    plt.colorbar(im, ax=ax_c, label='Mean Energy')

    # Overlay trajectory
    step = max(1, len(embedded) // 2000)
    ax_c.scatter(embedded[::step, 0], embedded[::step, 1], c='blue', s=1, alpha=0.1)

    ax_c.set_xlabel("Dim 1")
    ax_c.set_ylabel("Dim 2")
    ax_c.set_title("C) Energy Landscape on Manifold", fontweight='bold')

    # Panel D: Summary bar chart of metrics per regime
    ax_d = fig.add_subplot(gs[1, 1])

    # Compute metrics per regime
    regime_metrics = compute_regime_energy_metrics(
        trajectory, regime_labels, regime_names, sfreq, velocity_method
    )

    # Bar chart of mean energy
    x = np.arange(len(unique_names))
    means = [regime_metrics.get(name, KineticEnergyMetrics(0,0,0,0,0,0,0,0,0,0,0,0)).mean_energy
             for name in unique_names]
    cvs = [regime_metrics.get(name, KineticEnergyMetrics(0,0,0,0,0,0,0,0,0,0,0,0)).cv_energy
           for name in unique_names]

    width = 0.35
    bars1 = ax_d.bar(x - width/2, means, width, label='Mean Energy',
                     color=[regime_colors.get(n, '#888') for n in unique_names], alpha=0.8)

    # Secondary axis for CV
    ax_d2 = ax_d.twinx()
    bars2 = ax_d2.bar(x + width/2, cvs, width, label='CV', color='gray', alpha=0.5)

    ax_d.set_xticks(x)
    ax_d.set_xticklabels(unique_names)
    ax_d.set_ylabel("Mean Energy (||v||²)")
    ax_d2.set_ylabel("Coefficient of Variation")
    ax_d.set_title("D) Energy Metrics by Regime", fontweight='bold')
    ax_d.legend(loc='upper left')
    ax_d2.legend(loc='upper right')

    fig.suptitle("Kinetic Energy Analysis: Dynamical Activity Proxy", fontsize=14, fontweight='bold')

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================

def run_kinetic_energy_analysis(
    trajectories_path: Optional[Path] = None,
    simulation_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    velocity_method: str = "savgol",
    savgol_window: int = 5,
    show_plots: bool = False,
    simulate: bool = False,
    simulation_params: Optional[Dict] = None,
) -> Dict:
    """
    Run full kinetic energy analysis pipeline.

    Args:
        trajectories_path: Path to trajectories.npz from previous simulation
        simulation_dir: Directory containing results from coupled_oscillator_sim.py
        output_dir: Directory for output files
        velocity_method: "finite_diff" or "savgol"
        savgol_window: Window size for Savitzky-Golay
        show_plots: Whether to display plots
        simulate: Whether to run new simulation
        simulation_params: Parameters for new simulation

    Returns:
        Dict with all computed metrics
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent.parent.parent / "results" / "kinetic_energy" / f"analysis_{timestamp}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Kinetic Energy Analysis Pipeline")
    print("=" * 70)
    print(f"Output directory: {output_dir}")

    # Load or generate data
    if simulate:
        print("\nRunning new simulation...")
        from coupled_oscillator_sim import run_full_analysis

        sim_params = simulation_params or {}
        sim_output = output_dir / "simulation"
        sim_results = run_full_analysis(
            output_dir=sim_output,
            show_plots=False,
            **sim_params,
        )

        # Load generated trajectories
        trajectories_path = sim_output / "trajectories.npz"
        simulation_dir = sim_output

    # Load trajectories
    if trajectories_path is None and simulation_dir is not None:
        trajectories_path = Path(simulation_dir) / "trajectories.npz"

    if trajectories_path is None:
        raise ValueError("Must provide trajectories_path, simulation_dir, or use --simulate")

    print(f"\nLoading trajectories from: {trajectories_path}")
    data = np.load(trajectories_path)

    latent = data["latent"]
    embedded = data["embedded"]
    regime_labels = data["regime_labels"]

    # Load regime names from results.json if available
    results_json = trajectories_path.parent / "results.json"
    if results_json.exists():
        with open(results_json) as f:
            sim_results = json.load(f)
        regime_names = sim_results.get("regime_names", [f"regime_{i}" for i in np.unique(regime_labels)])
        sfreq = sim_results.get("parameters", {}).get("sfreq", 250.0)
    else:
        regime_names = [f"regime_{i}" for i in np.unique(regime_labels)]
        sfreq = 250.0

    print(f"  Latent shape: {latent.shape}")
    print(f"  Embedded shape: {embedded.shape}")
    print(f"  Regime names: {list(dict.fromkeys(regime_names))}")

    # ==========================================================================
    # STEP 1: Global Kinetic Energy Metrics
    # ==========================================================================
    print("\n" + "-" * 50)
    print("Step 1: Computing Global Kinetic Energy Metrics")
    print("-" * 50)

    global_metrics = compute_kinetic_energy_metrics(
        latent, sfreq=sfreq, velocity_method=velocity_method, savgol_window=savgol_window
    )

    print(f"  Mean energy: {global_metrics.mean_energy:.4f}")
    print(f"  CV (intermittency): {global_metrics.cv_energy:.3f}")
    print(f"  Tail index (burstiness): {global_metrics.tail_index:.3f}")
    print(f"  Kurtosis: {global_metrics.kurtosis:.2f}")
    print(f"  Burst frequency: {global_metrics.burst_frequency:.2f} Hz")

    # ==========================================================================
    # STEP 2: Per-Regime Kinetic Energy Metrics
    # ==========================================================================
    print("\n" + "-" * 50)
    print("Step 2: Computing Per-Regime Kinetic Energy Metrics")
    print("-" * 50)

    regime_metrics = compute_regime_energy_metrics(
        latent, regime_labels, regime_names, sfreq=sfreq, velocity_method=velocity_method
    )

    for name, metrics in regime_metrics.items():
        print(f"  {name}: E={metrics.mean_energy:.4f}, CV={metrics.cv_energy:.3f}, "
              f"tail={metrics.tail_index:.3f}")

    # ==========================================================================
    # STEP 3: Discriminability Analysis
    # ==========================================================================
    print("\n" + "-" * 50)
    print("Step 3: Discriminability Analysis")
    print("-" * 50)

    discriminability = compute_energy_discriminability(
        latent, regime_labels, regime_names, velocity_method=velocity_method
    )

    print(f"  F-statistic: {discriminability['f_statistic']:.1f}")
    print(f"  p-value: {discriminability['p_value']:.2e}")
    print(f"  η² (effect size): {discriminability['eta_squared']:.3f} ({discriminability['effect_size']})")

    # ==========================================================================
    # STEP 4: Energy Landscape
    # ==========================================================================
    print("\n" + "-" * 50)
    print("Step 4: Computing Energy Landscape")
    print("-" * 50)

    velocity = compute_velocity(latent, method=velocity_method)
    embedded_for_landscape = embedded[:len(velocity)]
    landscape = compute_energy_landscape(embedded_for_landscape, velocity)

    valid_cells = np.sum(~np.isnan(landscape.mean_energy))
    print(f"  Grid size: {landscape.mean_energy.shape}")
    print(f"  Valid cells: {valid_cells}")

    # ==========================================================================
    # STEP 5: Generate Figures
    # ==========================================================================
    print("\n" + "-" * 50)
    print("Step 5: Generating Figures")
    print("-" * 50)

    # Main analysis figure
    fig = plot_kinetic_energy_analysis(
        latent, embedded, regime_labels, regime_names,
        sfreq=sfreq, velocity_method=velocity_method,
        output_path=output_dir / "fig_kinetic_energy.png",
        show=show_plots,
    )

    # Also save PDF
    fig.savefig(output_dir / "fig_kinetic_energy.pdf", dpi=300, bbox_inches='tight')
    print(f"  Saved: fig_kinetic_energy.png/pdf")

    # ==========================================================================
    # STEP 6: Save Results
    # ==========================================================================
    print("\n" + "-" * 50)
    print("Step 6: Saving Results")
    print("-" * 50)

    # Helper to convert numpy types to Python native types for JSON
    def to_native(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [to_native(v) for v in obj]
        return obj

    results = {
        "parameters": {
            "velocity_method": velocity_method,
            "savgol_window": savgol_window,
            "sfreq": float(sfreq),
            "source_file": str(trajectories_path),
        },
        "global_metrics": to_native(asdict(global_metrics)),
        "regime_metrics": {name: to_native(asdict(m)) for name, m in regime_metrics.items()},
        "discriminability": to_native(discriminability),
        "landscape": {
            "bounds": list(landscape.bounds),
            "grid_shape": list(landscape.mean_energy.shape),
            "valid_cells": int(valid_cells),
        },
    }

    # Custom JSON encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64, np.floating)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64, np.integer)):
                return int(obj)
            return super().default(obj)

    with open(output_dir / "kinetic_energy_results.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved: kinetic_energy_results.json")

    # Save energy time series
    energy = compute_kinetic_energy(velocity)
    np.savez_compressed(
        output_dir / "energy_timeseries.npz",
        energy=energy,
        velocity_norm=np.linalg.norm(velocity, axis=1),
        regime_labels=regime_labels[:len(energy)],
    )
    print(f"  Saved: energy_timeseries.npz")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nKey findings:")
    print(f"  - Global mean energy: {global_metrics.mean_energy:.4f}")
    print(f"  - Energy discriminability: η²={discriminability['eta_squared']:.3f} ({discriminability['effect_size']})")

    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kinetic Energy Analysis for Latent Trajectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input options
    input_group = parser.add_argument_group("Input options")
    input_group.add_argument(
        "--input", "-i", type=str, default=None,
        help="Path to trajectories.npz file"
    )
    input_group.add_argument(
        "--simulation-dir", type=str, default=None,
        help="Directory containing simulation results"
    )
    input_group.add_argument(
        "--simulate", action="store_true",
        help="Run new simulation before analysis"
    )

    # Simulation parameters (if --simulate)
    sim_group = parser.add_argument_group("Simulation parameters (if --simulate)")
    sim_group.add_argument("--duration", type=float, default=160.0, help="Simulation duration (s)")
    sim_group.add_argument("--coupling", type=float, default=5.0, help="Coupling strength")
    sim_group.add_argument("--cycles", type=int, default=4, help="Number of regime cycles")
    sim_group.add_argument("--regime-duration", type=float, default=10.0, help="Duration per regime (s)")

    # Analysis parameters
    analysis_group = parser.add_argument_group("Analysis parameters")
    analysis_group.add_argument(
        "--velocity-method", type=str, default="savgol",
        choices=["finite_diff", "savgol"],
        help="Velocity computation method"
    )
    analysis_group.add_argument(
        "--savgol-window", type=int, default=5,
        help="Savitzky-Golay window size (must be odd)"
    )

    # Output options
    output_group = parser.add_argument_group("Output options")
    output_group.add_argument("--output", "-o", type=str, default=None, help="Output directory")
    output_group.add_argument("--show", action="store_true", help="Show plots")

    args = parser.parse_args()

    # Build simulation params if simulating
    simulation_params = None
    if args.simulate:
        simulation_params = {
            "total_duration_s": args.duration,
            "coupling_strength": args.coupling,
            "n_cycles": args.cycles,
            "regime_duration_s": args.regime_duration,
        }

    # Run analysis
    run_kinetic_energy_analysis(
        trajectories_path=Path(args.input) if args.input else None,
        simulation_dir=Path(args.simulation_dir) if args.simulation_dir else None,
        output_dir=Path(args.output) if args.output else None,
        velocity_method=args.velocity_method,
        savgol_window=args.savgol_window,
        show_plots=args.show,
        simulate=args.simulate,
        simulation_params=simulation_params,
    )
