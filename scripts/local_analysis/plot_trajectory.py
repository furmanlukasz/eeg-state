#!/usr/bin/env python3
"""
Rabinovich-style Trajectory Visualization in Latent Phase Space

Visualize latent dynamics as trajectories (not points/clusters) to observe:
- Flow channels and preferred directions
- Metastable regions (slow/sticky areas)
- Recurrent visits and looping motifs
- Transition dynamics between states

Based on Rabinovich's Information Flow Phase Space (IFPS) framework.

Usage:
    python plot_trajectory.py                           # Default: first HC subject
    python plot_trajectory.py --subject S001            # Specific subject
    python plot_trajectory.py --n-chunks 5              # Use 5 consecutive chunks
    python plot_trajectory.py --mode all                # All visualizations
    python plot_trajectory.py --mode trajectory         # Just trajectory plots
    python plot_trajectory.py --mode speed              # Speed-colored + metastability
    python plot_trajectory.py --mode flow               # Flow field (quiver)
    python plot_trajectory.py --mode density            # Density heatmap
    python plot_trajectory.py --compare-random          # Compare trained vs random
    python plot_trajectory.py --list-subjects           # List available subjects
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CHECKPOINT_PATH, DATA_DIR, OUTPUT_DIR, DEVICE,
    FILTER_LOW, FILTER_HIGH, CHUNK_DURATION, SFREQ,
    ensure_output_dir, get_fif_files, get_subjects_by_group
)
from load_model import load_model_from_checkpoint, create_model, compute_latent_trajectory
from load_data import load_and_preprocess_fif

# Color mapping
GROUP_COLORS = {0: "blue", 1: "orange", 2: "red"}
GROUP_NAMES = {0: "HC", 1: "MCI", 2: "AD"}


def create_timestamped_output_dir(base_dir: Path, script_name: str) -> Path:
    """Create a timestamped output directory for versioned results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / f"{script_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_parameters(output_dir: Path, params: dict):
    """Save parameters to a JSON file for reproducibility."""
    params_path = output_dir / "parameters.json"

    # Convert Path objects to strings
    serializable_params = {}
    for k, v in params.items():
        if isinstance(v, Path):
            serializable_params[k] = str(v)
        else:
            serializable_params[k] = v

    serializable_params["timestamp"] = datetime.now().isoformat()

    with open(params_path, 'w') as f:
        json.dump(serializable_params, f, indent=2)
    print(f"Parameters saved to: {params_path}")


def extract_continuous_latent(
    model,
    fif_path: Path,
    model_info: dict,
    n_chunks: int = 5,
    device: str = "mps",
) -> tuple[np.ndarray, list[int]]:
    """
    Extract continuous latent trajectory from consecutive chunks.

    Returns:
        latent: (total_time, hidden_size) array - concatenated latent trajectory
        chunk_boundaries: list of indices where chunks start
    """
    data = load_and_preprocess_fif(
        fif_path, FILTER_LOW, FILTER_HIGH, CHUNK_DURATION,
        include_amplitude=model_info["include_amplitude"],
        verbose=False,
    )

    available_chunks = len(data["chunks"])
    if available_chunks == 0:
        return None, []

    chunks_to_use = min(n_chunks, available_chunks)

    latents = []
    chunk_boundaries = [0]

    for cidx in range(chunks_to_use):
        phase_data = data["chunks"][cidx]
        latent = compute_latent_trajectory(model, phase_data, device)
        latents.append(latent)
        chunk_boundaries.append(chunk_boundaries[-1] + latent.shape[0])

    # Concatenate all chunks
    continuous_latent = np.concatenate(latents, axis=0)

    return continuous_latent, chunk_boundaries[:-1]  # Remove last (total length)


def compute_instantaneous_speed(latent: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    Compute instantaneous speed in latent space.

    Args:
        latent: (T, D) trajectory
        dt: time step (default 1 sample)

    Returns:
        speed: (T-1,) array of speeds
    """
    diff = np.diff(latent, axis=0)
    speed = np.linalg.norm(diff, axis=1) / dt
    return speed


def detect_dwell_episodes(
    speed: np.ndarray,
    threshold_percentile: float = 20,
    min_duration: int = 10,
) -> list[tuple[int, int, float]]:
    """
    Detect dwell episodes (contiguous low-speed runs).

    Args:
        speed: instantaneous speed array
        threshold_percentile: consider bottom X% as "slow"
        min_duration: minimum samples to count as episode

    Returns:
        List of (start_idx, end_idx, mean_speed) tuples
    """
    threshold = np.percentile(speed, threshold_percentile)
    is_slow = speed < threshold

    episodes = []
    in_episode = False
    start = 0

    for i, slow in enumerate(is_slow):
        if slow and not in_episode:
            in_episode = True
            start = i
        elif not slow and in_episode:
            in_episode = False
            if i - start >= min_duration:
                episodes.append((start, i, speed[start:i].mean()))

    # Handle episode at end
    if in_episode and len(is_slow) - start >= min_duration:
        episodes.append((start, len(is_slow), speed[start:].mean()))

    return episodes


def plot_trajectory_2d(
    latent: np.ndarray,
    output_dir: Path,
    chunk_boundaries: list[int] = None,
    title_suffix: str = "",
    show_plot: bool = True,
):
    """
    Plot 2D PCA trajectory colored by time.
    """
    # PCA to 2D
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create line segments for color mapping
    points = latent_2d.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Color by time
    norm = Normalize(vmin=0, vmax=len(latent_2d))
    lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=0.5, alpha=0.7)
    lc.set_array(np.arange(len(latent_2d)))
    ax.add_collection(lc)

    # Mark start and end
    ax.scatter(latent_2d[0, 0], latent_2d[0, 1], c='green', s=100, marker='o',
               label='Start', zorder=5, edgecolors='black')
    ax.scatter(latent_2d[-1, 0], latent_2d[-1, 1], c='red', s=100, marker='s',
               label='End', zorder=5, edgecolors='black')

    # Mark chunk boundaries
    if chunk_boundaries:
        for i, boundary in enumerate(chunk_boundaries[1:], 1):
            if boundary < len(latent_2d):
                ax.scatter(latent_2d[boundary, 0], latent_2d[boundary, 1],
                          c='white', s=50, marker='|', edgecolors='black', zorder=4)

    ax.autoscale()
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title(f'Latent Trajectory (2D PCA){title_suffix}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Time (samples)')

    plt.tight_layout()

    save_path = output_dir / "trajectory_2d.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return pca, latent_2d


def plot_trajectory_3d(
    latent: np.ndarray,
    output_dir: Path,
    chunk_boundaries: list[int] = None,
    title_suffix: str = "",
    show_plot: bool = True,
):
    """
    Plot 3D PCA trajectory colored by time.
    """
    # PCA to 3D
    pca = PCA(n_components=3)
    latent_3d = pca.fit_transform(latent)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory colored by time
    colors = plt.cm.viridis(np.linspace(0, 1, len(latent_3d)))

    for i in range(len(latent_3d) - 1):
        ax.plot(latent_3d[i:i+2, 0], latent_3d[i:i+2, 1], latent_3d[i:i+2, 2],
                c=colors[i], alpha=0.7, linewidth=0.5)

    # Mark start and end
    ax.scatter(*latent_3d[0], c='green', s=100, marker='o', label='Start', edgecolors='black')
    ax.scatter(*latent_3d[-1], c='red', s=100, marker='s', label='End', edgecolors='black')

    var_explained = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}%)')
    ax.set_zlabel(f'PC3 ({var_explained[2]*100:.1f}%)')
    ax.set_title(f'Latent Trajectory (3D PCA){title_suffix}', fontsize=14, fontweight='bold')
    ax.legend()

    plt.tight_layout()

    save_path = output_dir / "trajectory_3d.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return pca, latent_3d


def plot_speed_trajectory(
    latent: np.ndarray,
    output_dir: Path,
    speed_threshold_percentile: float = 20,
    title_suffix: str = "",
    show_plot: bool = True,
):
    """
    Plot trajectory colored by instantaneous speed + mark metastable regions.
    """
    # PCA to 2D
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent)

    # Compute speed
    speed = compute_instantaneous_speed(latent)
    speed_threshold = np.percentile(speed, speed_threshold_percentile)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Speed-colored trajectory
    ax = axes[0]
    points = latent_2d[:-1].reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = Normalize(vmin=speed.min(), vmax=speed.max())
    lc = LineCollection(segments, cmap='coolwarm_r', norm=norm, linewidth=1, alpha=0.8)
    lc.set_array(speed[:-1])
    ax.add_collection(lc)

    ax.autoscale()
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Speed-colored trajectory\n(dark blue = slow/metastable)', fontsize=12)

    sm = plt.cm.ScalarMappable(cmap='coolwarm_r', norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Speed (latent units/sample)')

    # Right: Metastable markers
    ax = axes[1]

    # Plot full trajectory in gray
    ax.plot(latent_2d[:, 0], latent_2d[:, 1], 'gray', alpha=0.3, linewidth=0.5)

    # Mark slow (metastable) points
    is_slow = speed < speed_threshold
    slow_points = latent_2d[:-1][is_slow]
    ax.scatter(slow_points[:, 0], slow_points[:, 1], c='darkblue', s=3, alpha=0.5,
               label=f'Slow (<{speed_threshold_percentile}%ile)')

    # Mark fast points
    fast_points = latent_2d[:-1][~is_slow]
    ax.scatter(fast_points[:, 0], fast_points[:, 1], c='red', s=1, alpha=0.3,
               label=f'Fast (>{speed_threshold_percentile}%ile)')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title(f'Metastable regions (bottom {speed_threshold_percentile}% speed)', fontsize=12)
    ax.legend(loc='upper right')

    plt.suptitle(f'Metastability Analysis{title_suffix}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = output_dir / "trajectory_speed.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return speed


def plot_density_heatmap(
    latent: np.ndarray,
    output_dir: Path,
    bins: int = 50,
    title_suffix: str = "",
    show_plot: bool = True,
):
    """
    Plot 2D density heatmap showing where system spends time + trajectory overlay.
    """
    # PCA to 2D
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Density heatmap
    ax = axes[0]
    H, xedges, yedges = np.histogram2d(latent_2d[:, 0], latent_2d[:, 1], bins=bins)
    H = gaussian_filter(H.T, sigma=1)  # Smooth

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(H, origin='lower', extent=extent, cmap='hot', aspect='equal')
    plt.colorbar(im, ax=ax, label='Density (time spent)', shrink=0.8)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Density heatmap\n(where system spends time)', fontsize=12)
    ax.set_aspect('equal')

    # Right: Density + trajectory overlay
    ax = axes[1]
    im = ax.imshow(H, origin='lower', extent=extent, cmap='hot', aspect='equal', alpha=0.7)
    ax.plot(latent_2d[:, 0], latent_2d[:, 1], 'cyan', alpha=0.4, linewidth=0.3)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Density + trajectory overlay', fontsize=12)
    ax.set_aspect('equal')

    plt.suptitle(f'Density Analysis (Dwell Time Distribution){title_suffix}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = output_dir / "trajectory_density.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_flow_field(
    latent: np.ndarray,
    output_dir: Path,
    grid_size: int = 20,
    title_suffix: str = "",
    show_plot: bool = True,
):
    """
    Plot empirical flow field (quiver) showing local displacement vectors.
    """
    # PCA to 2D
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent)

    # Compute displacements
    dx = np.diff(latent_2d[:, 0])
    dy = np.diff(latent_2d[:, 1])

    # Create grid
    x_min, x_max = latent_2d[:, 0].min(), latent_2d[:, 0].max()
    y_min, y_max = latent_2d[:, 1].min(), latent_2d[:, 1].max()

    x_edges = np.linspace(x_min, x_max, grid_size + 1)
    y_edges = np.linspace(y_min, y_max, grid_size + 1)

    # Bin displacements
    flow_x = np.zeros((grid_size, grid_size))
    flow_y = np.zeros((grid_size, grid_size))
    counts = np.zeros((grid_size, grid_size))

    for i in range(len(dx)):
        x_bin = np.searchsorted(x_edges[:-1], latent_2d[i, 0]) - 1
        y_bin = np.searchsorted(y_edges[:-1], latent_2d[i, 1]) - 1

        x_bin = np.clip(x_bin, 0, grid_size - 1)
        y_bin = np.clip(y_bin, 0, grid_size - 1)

        flow_x[y_bin, x_bin] += dx[i]
        flow_y[y_bin, x_bin] += dy[i]
        counts[y_bin, x_bin] += 1

    # Average
    mask = counts > 0
    flow_x[mask] /= counts[mask]
    flow_y[mask] /= counts[mask]

    # Grid centers
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Flow field only
    ax = axes[0]
    magnitude = np.sqrt(flow_x**2 + flow_y**2)
    ax.quiver(X, Y, flow_x, flow_y, magnitude, cmap='plasma', alpha=0.8)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Flow field (local displacement vectors)', fontsize=12)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')

    # Right: Flow field + trajectory
    ax = axes[1]
    ax.plot(latent_2d[:, 0], latent_2d[:, 1], 'gray', alpha=0.3, linewidth=0.3)
    ax.quiver(X, Y, flow_x, flow_y, magnitude, cmap='plasma', alpha=0.8)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Flow field + trajectory overlay', fontsize=12)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')

    plt.suptitle(f'Empirical Flow Field (Rabinovich-style){title_suffix}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = output_dir / "trajectory_flow.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def compute_trajectory_stats(latent: np.ndarray, speed: np.ndarray = None) -> dict:
    """
    Compute summary statistics for trajectory dynamics.
    """
    if speed is None:
        speed = compute_instantaneous_speed(latent)

    # Dwell episodes
    episodes = detect_dwell_episodes(speed)

    # PCA for explored volume
    pca = PCA(n_components=min(3, latent.shape[1]))
    latent_pca = pca.fit_transform(latent)

    # Convex hull volume proxy (variance)
    explored_variance = np.var(latent_pca, axis=0).sum()

    # Path length
    path_length = speed.sum()

    # Tortuosity (path length / displacement)
    displacement = np.linalg.norm(latent[-1] - latent[0])
    tortuosity = path_length / displacement if displacement > 0 else np.inf

    return {
        "mean_speed": speed.mean(),
        "speed_std": speed.std(),
        "speed_cv": speed.std() / speed.mean() if speed.mean() > 0 else 0,
        "n_dwell_episodes": len(episodes),
        "total_dwell_time": sum(e[1] - e[0] for e in episodes),
        "mean_dwell_duration": np.mean([e[1] - e[0] for e in episodes]) if episodes else 0,
        "explored_variance": explored_variance,
        "path_length": path_length,
        "displacement": displacement,
        "tortuosity": tortuosity,
        "pca_var_explained": pca.explained_variance_ratio_.tolist(),
    }


def compare_trained_vs_random(
    fif_path: Path,
    model_info: dict,
    n_channels: int,
    n_chunks: int,
    output_dir: Path,
    show_plot: bool = True,
):
    """
    Compare trajectory properties between trained and random models.
    """
    print("\n" + "=" * 60)
    print("COMPARING TRAINED vs RANDOM WEIGHTS")
    print("=" * 60)

    results = {}

    for mode, load_weights in [("Trained", True), ("Random", False)]:
        print(f"\n--- {mode} model ---")
        model = create_model(n_channels, model_info, DEVICE, load_weights=load_weights)

        latent, boundaries = extract_continuous_latent(
            model, fif_path, model_info, n_chunks, DEVICE
        )

        if latent is None:
            print(f"  No data available")
            continue

        speed = compute_instantaneous_speed(latent)
        stats = compute_trajectory_stats(latent, speed)
        results[mode] = {"latent": latent, "speed": speed, "stats": stats}

        print(f"  Mean speed: {stats['mean_speed']:.4f}")
        print(f"  Speed CV: {stats['speed_cv']:.4f}")
        print(f"  Dwell episodes: {stats['n_dwell_episodes']}")
        print(f"  Tortuosity: {stats['tortuosity']:.2f}")
        print(f"  Explored variance: {stats['explored_variance']:.4f}")

    if len(results) < 2:
        print("Cannot compare - missing data")
        return

    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for col, (mode, data) in enumerate(results.items()):
        latent = data["latent"]
        speed = data["speed"]

        # PCA
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latent)

        # Row 0: Trajectory
        ax = axes[0, col]
        points = latent_2d.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = Normalize(vmin=0, vmax=len(latent_2d))
        lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=0.5, alpha=0.7)
        lc.set_array(np.arange(len(latent_2d)))
        ax.add_collection(lc)
        ax.autoscale()
        ax.set_title(f'{mode}: Trajectory', fontsize=12)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

        # Row 1: Speed distribution
        ax = axes[1, col]
        ax.hist(speed, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(np.percentile(speed, 20), color='red', linestyle='--', label='20%ile')
        ax.set_xlabel('Speed')
        ax.set_ylabel('Count')
        ax.set_title(f'{mode}: Speed distribution', fontsize=12)
        ax.legend()

    # Right column: comparison stats
    ax = axes[0, 2]
    ax.axis('off')

    stats_text = "COMPARISON SUMMARY\n" + "=" * 40 + "\n\n"
    for metric in ["mean_speed", "speed_cv", "n_dwell_episodes", "tortuosity", "explored_variance"]:
        trained_val = results["Trained"]["stats"][metric]
        random_val = results["Random"]["stats"][metric]
        ratio = trained_val / random_val if random_val != 0 else np.inf
        stats_text += f"{metric}:\n"
        stats_text += f"  Trained: {trained_val:.4f}\n"
        stats_text += f"  Random:  {random_val:.4f}\n"
        stats_text += f"  Ratio:   {ratio:.2f}\n\n"

    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax = axes[1, 2]
    ax.axis('off')

    interpretation = """
INTERPRETATION GUIDE
====================

If Trained shows:
- Lower speed CV → more consistent dynamics
- More dwell episodes → richer metastable structure
- Higher tortuosity → more complex paths
- Lower explored variance → more constrained dynamics

vs Random:
- Similar values → learned features may not add much
- Very different → learned representations capture structure
"""
    ax.text(0.1, 0.9, interpretation, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace')

    plt.suptitle('Trained vs Random Weights: Trajectory Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = output_dir / "trajectory_trained_vs_random.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return results


def list_subjects(groups: dict):
    """List available subjects by group."""
    print("\n" + "=" * 60)
    print("AVAILABLE SUBJECTS")
    print("=" * 60)

    for group_key in ["hc", "mci", "ad"]:
        subjects = groups.get(group_key, [])
        if not subjects:
            continue

        group_name = GROUP_NAMES.get(subjects[0][1], group_key.upper())
        print(f"\n{group_name} subjects ({len(subjects)}):")
        for i, (fif_path, label, condition, subject_id) in enumerate(subjects):
            print(f"  {i+1:3d}. {subject_id:10s} ({condition})")

    total = sum(len(groups.get(k, [])) for k in ["hc", "mci", "ad"])
    print(f"\nTotal: {total} unique subjects")


def main():
    parser = argparse.ArgumentParser(
        description="Rabinovich-style trajectory visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_trajectory.py                           # Default: first HC subject
  python plot_trajectory.py --subject S001            # Specific subject
  python plot_trajectory.py --n-chunks 10             # Use 10 consecutive chunks
  python plot_trajectory.py --mode all                # All visualizations
  python plot_trajectory.py --mode trajectory         # Just 2D/3D trajectory
  python plot_trajectory.py --mode speed              # Speed + metastability
  python plot_trajectory.py --mode flow               # Flow field (quiver)
  python plot_trajectory.py --mode density            # Density heatmap
  python plot_trajectory.py --compare-random          # Trained vs random comparison
  python plot_trajectory.py --conditions MCI          # Use MCI subject
  python plot_trajectory.py --list-subjects           # List available subjects
        """
    )
    parser.add_argument("--subject", type=str, default=None,
                        help="Specific subject ID (default: first available)")
    parser.add_argument("--n-chunks", type=int, default=5,
                        help="Number of consecutive chunks to use (default: 5)")
    parser.add_argument("--conditions", type=str, nargs="+", default=["HID", "MCI"],
                        help="Conditions to include (default: HID MCI)")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "trajectory", "speed", "flow", "density"],
                        help="Visualization mode (default: all)")
    parser.add_argument("--compare-random", action="store_true",
                        help="Compare trained vs random weights")
    parser.add_argument("--list-subjects", action="store_true",
                        help="List available subjects and exit")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't display plots interactively")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Custom output directory (default: timestamped folder)")
    args = parser.parse_args()

    # Create timestamped output directory
    base_output_dir = ensure_output_dir()
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = create_timestamped_output_dir(base_output_dir, "plot_trajectory")

    # Get subjects
    fif_files = get_fif_files(args.conditions)
    groups = get_subjects_by_group(fif_files)

    # Print summary
    group_counts = []
    for key in ["hc", "mci", "ad"]:
        if groups.get(key):
            group_counts.append(f"{len(groups[key])} {key.upper()}")
    print(f"Found {', '.join(group_counts)} subjects for conditions: {args.conditions}")

    # Handle --list-subjects
    if args.list_subjects:
        list_subjects(groups)
        return 0

    # Select subject
    all_subjects = []
    for key in ["hc", "mci", "ad"]:
        all_subjects.extend(groups.get(key, []))

    if not all_subjects:
        print("No subjects found!")
        return 1

    if args.subject:
        # Find specific subject
        selected = None
        for s in all_subjects:
            if args.subject in s[3]:  # subject_id is at index 3
                selected = s
                break
        if not selected:
            print(f"Subject {args.subject} not found!")
            return 1
    else:
        # Use first subject
        selected = all_subjects[0]

    fif_path, label, condition, subject_id = selected
    group_name = GROUP_NAMES.get(label, "Unknown")

    print(f"\nSelected subject: {subject_id} ({condition}, {group_name})")

    # Save parameters
    params = {
        "subject": subject_id,
        "n_chunks": args.n_chunks,
        "conditions": args.conditions,
        "mode": args.mode,
        "compare_random": args.compare_random,
        "group": group_name,
        "fif_path": str(fif_path),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "filter_low": FILTER_LOW,
        "filter_high": FILTER_HIGH,
        "chunk_duration": CHUNK_DURATION,
        "sfreq": SFREQ,
    }
    save_parameters(output_dir, params)

    # Load model
    print("\nLoading model...")
    model_info = load_model_from_checkpoint(CHECKPOINT_PATH, DEVICE)

    # Get n_channels from first file
    first_data = load_and_preprocess_fif(
        fif_path, FILTER_LOW, FILTER_HIGH, CHUNK_DURATION,
        include_amplitude=model_info["include_amplitude"],
        verbose=False,
    )
    n_channels = first_data["n_channels"]

    # Handle --compare-random
    if args.compare_random:
        compare_trained_vs_random(
            fif_path, model_info, n_channels, args.n_chunks,
            output_dir, not args.no_show
        )
        return 0

    # Create model with trained weights
    model = create_model(n_channels, model_info, DEVICE, load_weights=True)

    # Extract continuous latent
    print(f"\nExtracting latent trajectory from {args.n_chunks} chunks...")
    latent, boundaries = extract_continuous_latent(
        model, fif_path, model_info, args.n_chunks, DEVICE
    )

    if latent is None:
        print("No data available!")
        return 1

    print(f"Latent trajectory shape: {latent.shape}")
    print(f"Duration: {latent.shape[0] / SFREQ:.1f} seconds")

    title_suffix = f"\n{subject_id} ({group_name})"

    # Generate visualizations based on mode
    if args.mode in ["all", "trajectory"]:
        print("\n--- 2D Trajectory ---")
        plot_trajectory_2d(latent, output_dir, boundaries, title_suffix, not args.no_show)

        print("\n--- 3D Trajectory ---")
        plot_trajectory_3d(latent, output_dir, boundaries, title_suffix, not args.no_show)

    if args.mode in ["all", "speed"]:
        print("\n--- Speed/Metastability Analysis ---")
        speed = plot_speed_trajectory(latent, output_dir, 20, title_suffix, not args.no_show)

    if args.mode in ["all", "density"]:
        print("\n--- Density Heatmap ---")
        plot_density_heatmap(latent, output_dir, 50, title_suffix, not args.no_show)

    if args.mode in ["all", "flow"]:
        print("\n--- Flow Field ---")
        plot_flow_field(latent, output_dir, 20, title_suffix, not args.no_show)

    # Compute and print summary stats
    print("\n" + "=" * 60)
    print("TRAJECTORY STATISTICS")
    print("=" * 60)

    stats = compute_trajectory_stats(latent)
    print(f"Mean speed:        {stats['mean_speed']:.4f}")
    print(f"Speed std:         {stats['speed_std']:.4f}")
    print(f"Speed CV:          {stats['speed_cv']:.4f}")
    print(f"Dwell episodes:    {stats['n_dwell_episodes']}")
    print(f"Total dwell time:  {stats['total_dwell_time']} samples ({stats['total_dwell_time']/SFREQ:.1f}s)")
    print(f"Mean dwell dur:    {stats['mean_dwell_duration']:.1f} samples")
    print(f"Path length:       {stats['path_length']:.2f}")
    print(f"Displacement:      {stats['displacement']:.4f}")
    print(f"Tortuosity:        {stats['tortuosity']:.2f}")
    print(f"Explored variance: {stats['explored_variance']:.4f}")
    print(f"PCA var explained: {[f'{v*100:.1f}%' for v in stats['pca_var_explained']]}")

    print(f"\nAll plots saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
