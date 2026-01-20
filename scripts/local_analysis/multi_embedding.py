#!/usr/bin/env python3
"""
Multi-Embedding Trajectory Analysis

Compare trajectory dynamics across multiple embedding methods:
- PCA (baseline linear projection)
- Time-lagged PCA (sensitive to temporal structure)
- Diffusion Maps (metastability-aware nonlinear embedding)
- UMAP (visualization only - explicitly qualified)

Theory-aligned with Rabinovich's Information Flow Phase Space (IFPS).
Focus on FLOW GEOMETRY, not classification.

Usage:
    python multi_embedding.py                           # Default: first HC subject
    python multi_embedding.py --subject S001            # Specific subject
    python multi_embedding.py --n-chunks 10             # Use 10 consecutive chunks
    python multi_embedding.py --compare-groups          # Compare HC vs MCI
    python multi_embedding.py --embedding pca           # Single embedding
    python multi_embedding.py --embedding diffusion     # Diffusion maps only
    python multi_embedding.py --list-subjects           # List available subjects
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, entropy
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

# Optional imports
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

GROUP_COLORS = {0: "blue", 1: "orange", 2: "red"}
GROUP_NAMES = {0: "HC", 1: "MCI", 2: "AD"}


@dataclass
class FlowMetrics:
    """Flow geometry metrics for a trajectory."""
    mean_speed: float
    speed_std: float
    speed_cv: float
    n_dwell_episodes: int
    total_dwell_time: int
    mean_dwell_duration: float
    occupancy_entropy: float
    path_tortuosity: float
    explored_variance: float


# =============================================================================
# EMBEDDING METHODS
# =============================================================================

def pca_embedding(z: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, dict]:
    """
    Standard PCA embedding.

    Returns:
        embedded: (T, n_components) array
        info: dict with explained variance etc.
    """
    pca = PCA(n_components=n_components)
    embedded = pca.fit_transform(z)

    return embedded, {
        "method": "PCA",
        "var_explained": pca.explained_variance_ratio_.tolist(),
        "total_var": sum(pca.explained_variance_ratio_),
    }


def time_lagged_pca(z: np.ndarray, tau: int = 5, n_components: int = 2) -> tuple[np.ndarray, dict]:
    """
    Time-lagged PCA: PCA on [z(t), z(t+τ)] pairs.

    Sensitive to temporal structure, reveals directions of maximal
    predictive change rather than static variance.

    Args:
        z: (T, D) latent trajectory
        tau: time lag in samples
        n_components: output dimensions

    Returns:
        embedded: (T-tau, n_components) array
        info: dict with metadata
    """
    T, D = z.shape

    # Create time-lagged pairs: [z(t), z(t+τ)]
    z_lagged = np.hstack([z[:-tau], z[tau:]])  # (T-tau, 2D)

    pca = PCA(n_components=n_components)
    embedded = pca.fit_transform(z_lagged)

    return embedded, {
        "method": f"tPCA (τ={tau})",
        "tau": tau,
        "var_explained": pca.explained_variance_ratio_.tolist(),
        "total_var": sum(pca.explained_variance_ratio_),
    }


def diffusion_maps(
    z: np.ndarray,
    n_components: int = 2,
    k: int = 10,
    epsilon: str = "median",
) -> tuple[np.ndarray, dict]:
    """
    Diffusion maps embedding.

    Uncovers slow collective variables, directly connected to metastability.
    Embedding organized by transition probabilities.

    Args:
        z: (T, D) latent trajectory
        n_components: output dimensions
        k: number of nearest neighbors for kernel
        epsilon: kernel bandwidth ('median' or float)

    Returns:
        embedded: (T, n_components) array
        info: dict with metadata
    """
    T = z.shape[0]

    # Subsample if too large (diffusion maps is O(n²))
    max_points = 2000
    if T > max_points:
        indices = np.linspace(0, T - 1, max_points, dtype=int)
        z_sub = z[indices]
    else:
        indices = np.arange(T)
        z_sub = z

    # Compute pairwise distances
    distances = squareform(pdist(z_sub, metric='euclidean'))

    # Determine epsilon (kernel bandwidth)
    if epsilon == "median":
        # Use median of k-nearest neighbor distances
        knn_dists = np.sort(distances, axis=1)[:, 1:k+1]
        eps = np.median(knn_dists)
    else:
        eps = float(epsilon)

    # Gaussian kernel
    K = np.exp(-distances**2 / (2 * eps**2))

    # Normalize to get transition matrix
    D_inv = np.diag(1.0 / K.sum(axis=1))
    P = D_inv @ K

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(P)

    # Sort by eigenvalue magnitude (skip first which is trivial)
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx].real
    eigenvectors = eigenvectors[:, idx].real

    # Take components 1 to n_components+1 (skip trivial)
    embedded_sub = eigenvectors[:, 1:n_components+1]

    # If subsampled, interpolate back (simple nearest neighbor)
    if T > max_points:
        from scipy.interpolate import interp1d
        f = interp1d(indices, embedded_sub, axis=0, kind='linear', fill_value='extrapolate')
        embedded = f(np.arange(T))
    else:
        embedded = embedded_sub

    return embedded, {
        "method": f"Diffusion Maps (k={k}, ε={eps:.3f})",
        "epsilon": eps,
        "k": k,
        "eigenvalues": eigenvalues[1:n_components+2].tolist(),
    }


def umap_embedding(
    z: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 30,
    min_dist: float = 0.2,
) -> tuple[np.ndarray, dict]:
    """
    UMAP embedding (visualization only - explicitly qualified).

    Use for global manifold shape visualization, NOT for flow vectors
    or quantitative metrics.

    Args:
        z: (T, D) latent trajectory
        n_components: output dimensions
        n_neighbors: UMAP parameter
        min_dist: UMAP parameter

    Returns:
        embedded: (T, n_components) array
        info: dict with metadata
    """
    if not HAS_UMAP:
        # Fallback to PCA
        print("  WARNING: UMAP not available, falling back to PCA")
        return pca_embedding(z, n_components)

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='euclidean',
        random_state=42,
    )
    embedded = reducer.fit_transform(z)

    return embedded, {
        "method": f"UMAP (nn={n_neighbors}, md={min_dist})",
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "note": "Visualization only - do not use for flow quantification",
    }


def delay_embedding(
    z: np.ndarray,
    tau: int = 5,
    dim: int = 3,
    n_components: int = 2,
) -> tuple[np.ndarray, dict]:
    """
    Takens-style delay embedding followed by PCA.

    Creates [z(t), z(t+τ), z(t+2τ), ...] then reduces with PCA.
    Theoretically grounded, makes recurrence structure explicit.

    Args:
        z: (T, D) latent trajectory
        tau: delay in samples
        dim: embedding dimension (number of delays)
        n_components: output dimensions after PCA

    Returns:
        embedded: (T - (dim-1)*tau, n_components) array
        info: dict with metadata
    """
    T, D = z.shape
    n_valid = T - (dim - 1) * tau

    if n_valid < 100:
        raise ValueError(f"Not enough points for delay embedding: {n_valid}")

    # Create delay vectors
    delayed = np.zeros((n_valid, D * dim))
    for i in range(dim):
        delayed[:, i*D:(i+1)*D] = z[i*tau:i*tau + n_valid]

    # PCA on delayed vectors
    pca = PCA(n_components=n_components)
    embedded = pca.fit_transform(delayed)

    return embedded, {
        "method": f"Delay (τ={tau}, d={dim})",
        "tau": tau,
        "dim": dim,
        "var_explained": pca.explained_variance_ratio_.tolist(),
    }


# =============================================================================
# FLOW METRICS
# =============================================================================

def compute_instantaneous_speed(embedded: np.ndarray) -> np.ndarray:
    """Compute speed in embedding space."""
    diff = np.diff(embedded, axis=0)
    return np.linalg.norm(diff, axis=1)


def detect_dwell_episodes(
    speed: np.ndarray,
    threshold_percentile: float = 20,
    min_duration: int = 10,
) -> list[tuple[int, int]]:
    """Detect contiguous low-speed runs."""
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
                episodes.append((start, i))

    if in_episode and len(is_slow) - start >= min_duration:
        episodes.append((start, len(is_slow)))

    return episodes


def compute_occupancy_entropy(embedded: np.ndarray, bins: int = 20) -> float:
    """
    Compute entropy of occupancy distribution.

    Higher entropy = more uniform exploration.
    Lower entropy = concentrated in few regions.
    """
    H, _, _ = np.histogram2d(embedded[:, 0], embedded[:, 1], bins=bins)
    H = H.flatten()
    H = H[H > 0]  # Remove empty bins
    p = H / H.sum()
    return entropy(p)


def compute_flow_metrics(embedded: np.ndarray) -> FlowMetrics:
    """Compute comprehensive flow geometry metrics."""
    speed = compute_instantaneous_speed(embedded)
    episodes = detect_dwell_episodes(speed)

    # Path metrics
    path_length = speed.sum()
    displacement = np.linalg.norm(embedded[-1] - embedded[0])
    tortuosity = path_length / displacement if displacement > 0 else np.inf

    # Explored variance
    explored_variance = np.var(embedded, axis=0).sum()

    # Occupancy entropy
    occ_entropy = compute_occupancy_entropy(embedded)

    return FlowMetrics(
        mean_speed=speed.mean(),
        speed_std=speed.std(),
        speed_cv=speed.std() / speed.mean() if speed.mean() > 0 else 0,
        n_dwell_episodes=len(episodes),
        total_dwell_time=sum(e[1] - e[0] for e in episodes),
        mean_dwell_duration=np.mean([e[1] - e[0] for e in episodes]) if episodes else 0,
        occupancy_entropy=occ_entropy,
        path_tortuosity=tortuosity,
        explored_variance=explored_variance,
    )


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_single_embedding(
    embedded: np.ndarray,
    info: dict,
    ax: plt.Axes,
    color_by: str = "time",
    speed: np.ndarray = None,
):
    """Plot trajectory in a single embedding."""
    T = len(embedded)

    if color_by == "time":
        colors = np.linspace(0, 1, T)
        cmap = 'viridis'
        label = 'Time'
    elif color_by == "speed" and speed is not None:
        colors = speed
        colors = np.concatenate([colors, [colors[-1]]])  # Pad to match T
        cmap = 'coolwarm_r'
        label = 'Speed'
    else:
        colors = np.linspace(0, 1, T)
        cmap = 'viridis'
        label = 'Time'

    # Line collection for colored trajectory
    points = embedded.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = Normalize(vmin=colors.min(), vmax=colors.max())
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=0.5, alpha=0.7)
    lc.set_array(colors[:-1])
    ax.add_collection(lc)

    ax.autoscale()
    ax.set_title(info["method"], fontsize=10, fontweight='bold')

    # Variance info if available (only for PCA-based methods)
    if "total_var" in info:
        var_str = f"Var: {info['total_var']*100:.1f}%"
        ax.text(0.02, 0.98, var_str, transform=ax.transAxes, fontsize=8,
                va='top', ha='left', color='gray')


def plot_density_overlay(embedded: np.ndarray, ax: plt.Axes, bins: int = 30):
    """Add density heatmap to axis."""
    H, xedges, yedges = np.histogram2d(embedded[:, 0], embedded[:, 1], bins=bins)
    H = gaussian_filter(H.T, sigma=1)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax.imshow(H, origin='lower', extent=extent, cmap='hot', aspect='auto', alpha=0.6)


def plot_multi_embedding_comparison(
    latent: np.ndarray,
    output_dir: Path,
    subject_id: str,
    group_name: str,
    show_plot: bool = True,
):
    """
    Generate comprehensive multi-embedding comparison plot.
    """
    print(f"\nComputing embeddings for {subject_id} ({group_name})...")

    # Compute all embeddings
    embeddings = {}

    print("  PCA...")
    embeddings["PCA"] = pca_embedding(latent)

    print("  Time-lagged PCA (τ=5)...")
    embeddings["tPCA"] = time_lagged_pca(latent, tau=5)

    print("  Diffusion Maps...")
    try:
        embeddings["Diffusion"] = diffusion_maps(latent, k=10)
    except Exception as e:
        print(f"    Failed: {e}")
        embeddings["Diffusion"] = pca_embedding(latent)  # Fallback

    print("  Delay Embedding (τ=5, d=3)...")
    try:
        embeddings["Delay"] = delay_embedding(latent, tau=5, dim=3)
    except Exception as e:
        print(f"    Failed: {e}")
        embeddings["Delay"] = pca_embedding(latent)

    if HAS_UMAP:
        print("  UMAP (visualization only)...")
        embeddings["UMAP"] = umap_embedding(latent)

    # Compute metrics for each
    metrics = {}
    for name, (emb, info) in embeddings.items():
        metrics[name] = compute_flow_metrics(emb)

    # Create figure
    n_emb = len(embeddings)
    fig, axes = plt.subplots(2, n_emb, figsize=(5 * n_emb, 10))

    for col, (name, (emb, info)) in enumerate(embeddings.items()):
        speed = compute_instantaneous_speed(emb)

        # Top row: time-colored trajectory
        plot_single_embedding(emb, info, axes[0, col], color_by="time")

        # Bottom row: density + trajectory
        plot_density_overlay(emb, axes[1, col])
        plot_single_embedding(emb, info, axes[1, col], color_by="speed", speed=speed)
        axes[1, col].set_title(f"{name}: Density + Speed", fontsize=10)

    plt.suptitle(
        f"Multi-Embedding Comparison: {subject_id} ({group_name})\n"
        "Top: Time-colored | Bottom: Density + Speed-colored",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()

    save_path = output_dir / f"multi_embedding_{subject_id}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    # Print metrics comparison
    print("\n" + "=" * 80)
    print("FLOW METRICS COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<20} " + " ".join([f"{name:>12}" for name in embeddings.keys()]))
    print("-" * 80)

    metric_names = [
        ("mean_speed", "Mean Speed"),
        ("speed_cv", "Speed CV"),
        ("n_dwell_episodes", "Dwell Episodes"),
        ("occupancy_entropy", "Occ. Entropy"),
        ("path_tortuosity", "Tortuosity"),
        ("explored_variance", "Explored Var"),
    ]

    for attr, label in metric_names:
        values = [f"{getattr(metrics[name], attr):>12.3f}" for name in embeddings.keys()]
        print(f"{label:<20} " + " ".join(values))

    # Highlight occupancy entropy consistency across embeddings
    occ_vals = [metrics[name].occupancy_entropy for name in embeddings.keys()]
    occ_cv = np.std(occ_vals) / np.mean(occ_vals) if np.mean(occ_vals) > 0 else 0

    print("\n" + "-" * 80)
    print("OCCUPANCY ENTROPY CONSISTENCY (across embeddings)")
    print(f"  Mean: {np.mean(occ_vals):.3f}, Std: {np.std(occ_vals):.3f}, CV: {occ_cv:.3f}")
    if occ_cv < 0.15:
        print("  → HIGH consistency: Occupancy entropy robust across methods")
    elif occ_cv < 0.30:
        print("  → MODERATE consistency: Some method-dependent variation")
    else:
        print("  → LOW consistency: Occupancy entropy varies by embedding method")

    return embeddings, metrics


def compute_cross_embedding_consistency(
    embeddings: dict,
) -> dict:
    """
    Quantify consistency of flow metrics across embeddings.

    If patterns persist across embeddings → robust dynamical signature.
    """
    embedding_names = list(embeddings.keys())
    n_emb = len(embedding_names)

    # Compute speed for each
    speeds = {}
    for name, (emb, _) in embeddings.items():
        speeds[name] = compute_instantaneous_speed(emb)

    # Compute rank correlations of speed
    speed_correlations = np.zeros((n_emb, n_emb))
    for i, name1 in enumerate(embedding_names):
        for j, name2 in enumerate(embedding_names):
            # Align lengths (they may differ due to lag)
            len1, len2 = len(speeds[name1]), len(speeds[name2])
            min_len = min(len1, len2)
            corr, _ = spearmanr(speeds[name1][:min_len], speeds[name2][:min_len])
            speed_correlations[i, j] = corr

    return {
        "speed_correlation_matrix": speed_correlations,
        "embedding_names": embedding_names,
        "mean_cross_correlation": np.mean(speed_correlations[np.triu_indices(n_emb, k=1)]),
    }


def plot_consistency_analysis(
    embeddings: dict,
    output_dir: Path,
    subject_id: str,
    show_plot: bool = True,
):
    """Plot cross-embedding consistency analysis."""
    consistency = compute_cross_embedding_consistency(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Correlation matrix
    ax = axes[0]
    im = ax.imshow(consistency["speed_correlation_matrix"], cmap='RdYlGn', vmin=-1, vmax=1)
    ax.set_xticks(range(len(consistency["embedding_names"])))
    ax.set_yticks(range(len(consistency["embedding_names"])))
    ax.set_xticklabels(consistency["embedding_names"], rotation=45, ha='right')
    ax.set_yticklabels(consistency["embedding_names"])
    ax.set_title("Speed Rank Correlation Across Embeddings", fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Spearman ρ')

    # Add correlation values as text
    for i in range(len(consistency["embedding_names"])):
        for j in range(len(consistency["embedding_names"])):
            ax.text(j, i, f'{consistency["speed_correlation_matrix"][i, j]:.2f}',
                    ha='center', va='center', fontsize=9)

    # Right: Interpretation
    ax = axes[1]
    ax.axis('off')

    mean_corr = consistency["mean_cross_correlation"]

    interpretation = f"""
CROSS-EMBEDDING CONSISTENCY ANALYSIS
=====================================

Subject: {subject_id}
Mean Cross-Correlation: {mean_corr:.3f}

INTERPRETATION:
---------------
"""
    if mean_corr > 0.7:
        interpretation += """
HIGH CONSISTENCY (ρ > 0.7)
✓ Flow patterns are ROBUST across embedding methods
✓ Dynamical features are not artifacts of projection
✓ Strong evidence for intrinsic structure
"""
    elif mean_corr > 0.4:
        interpretation += """
MODERATE CONSISTENCY (0.4 < ρ < 0.7)
~ Some patterns persist across embeddings
~ Partial support for intrinsic structure
~ Consider which embeddings agree/disagree
"""
    else:
        interpretation += """
LOW CONSISTENCY (ρ < 0.4)
⚠ Flow patterns differ across embeddings
⚠ Results may be projection-dependent
⚠ Interpret with caution
"""

    interpretation += """

NOTE: High consistency is necessary (not sufficient)
for claiming robust dynamical signatures.
"""

    ax.text(0.1, 0.9, interpretation, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle(f"Consistency Analysis: {subject_id}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = output_dir / f"embedding_consistency_{subject_id}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return consistency


def compare_groups(
    model,
    model_info: dict,
    groups: dict,
    n_subjects: int,
    n_chunks: int,
    output_dir: Path,
    show_plot: bool = True,
):
    """Compare flow metrics between HC and MCI across embeddings."""
    print("\n" + "=" * 80)
    print("GROUP COMPARISON: HC vs MCI")
    print("=" * 80)

    group_metrics = {"HC": [], "MCI": []}

    for group_key in ["hc", "mci"]:
        subjects = groups.get(group_key, [])
        if not subjects:
            continue

        group_name = "HC" if group_key == "hc" else "MCI"
        print(f"\nProcessing {group_name} subjects...")

        subjects_processed = 0
        for fif_path, label, condition, subject_id in subjects:
            if subjects_processed >= n_subjects:
                break

            print(f"  {subject_id}...", end=" ")

            # Load data
            data = load_and_preprocess_fif(
                fif_path, FILTER_LOW, FILTER_HIGH, CHUNK_DURATION,
                include_amplitude=model_info["include_amplitude"],
                verbose=False,
            )

            if len(data["chunks"]) == 0:
                print("no chunks, skip")
                continue

            # Extract latent
            chunks_to_use = min(n_chunks, len(data["chunks"]))
            latents = []
            for cidx in range(chunks_to_use):
                latent = compute_latent_trajectory(model, data["chunks"][cidx], DEVICE)
                latents.append(latent)
            latent = np.concatenate(latents, axis=0)

            # Compute PCA embedding and metrics (use PCA for group comparison - most stable)
            emb, _ = pca_embedding(latent)
            metrics = compute_flow_metrics(emb)
            group_metrics[group_name].append(metrics)

            print(f"speed_cv={metrics.speed_cv:.3f}")
            subjects_processed += 1

    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    metric_pairs = [
        ("mean_speed", "Mean Speed"),
        ("speed_cv", "Speed CV"),
        ("n_dwell_episodes", "Dwell Episodes"),
        ("occupancy_entropy", "Occupancy Entropy"),
        ("path_tortuosity", "Tortuosity"),
        ("explored_variance", "Explored Variance"),
    ]

    for idx, (attr, label) in enumerate(metric_pairs):
        ax = axes.flatten()[idx]

        hc_vals = [getattr(m, attr) for m in group_metrics["HC"]]
        mci_vals = [getattr(m, attr) for m in group_metrics["MCI"]]

        # Violin plot
        parts = ax.violinplot([hc_vals, mci_vals], positions=[0, 1], showmeans=True)
        parts['bodies'][0].set_facecolor('blue')
        parts['bodies'][0].set_alpha(0.7)
        parts['bodies'][1].set_facecolor('orange')
        parts['bodies'][1].set_alpha(0.7)

        # Individual points
        ax.scatter(np.zeros(len(hc_vals)) + np.random.uniform(-0.1, 0.1, len(hc_vals)),
                   hc_vals, c='blue', alpha=0.6, s=30, edgecolors='black', linewidths=0.5)
        ax.scatter(np.ones(len(mci_vals)) + np.random.uniform(-0.1, 0.1, len(mci_vals)),
                   mci_vals, c='orange', alpha=0.6, s=30, edgecolors='black', linewidths=0.5)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['HC', 'MCI'])
        ax.set_ylabel(label)
        ax.set_title(label, fontweight='bold')

        # Add means
        hc_mean, mci_mean = np.mean(hc_vals), np.mean(mci_vals)
        ax.axhline(hc_mean, color='blue', linestyle='--', alpha=0.5, xmin=0.1, xmax=0.4)
        ax.axhline(mci_mean, color='orange', linestyle='--', alpha=0.5, xmin=0.6, xmax=0.9)

    plt.suptitle(
        "Flow Geometry Metrics: HC vs MCI\n"
        "(PCA embedding, multiple subjects)",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()

    save_path = output_dir / "group_comparison_flow_metrics.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    # Print summary
    print("\n" + "=" * 80)
    print("GROUP SUMMARY (Mean ± Std)")
    print("=" * 80)
    print(f"{'Metric':<25} {'HC':>20} {'MCI':>20}")
    print("-" * 80)

    for attr, label in metric_pairs:
        hc_vals = [getattr(m, attr) for m in group_metrics["HC"]]
        mci_vals = [getattr(m, attr) for m in group_metrics["MCI"]]
        print(f"{label:<25} {np.mean(hc_vals):>8.3f} ± {np.std(hc_vals):<8.3f} "
              f"{np.mean(mci_vals):>8.3f} ± {np.std(mci_vals):<8.3f}")

    # Highlight occupancy entropy interpretation (key metric per critic agent)
    hc_occ = [m.occupancy_entropy for m in group_metrics["HC"]]
    mci_occ = [m.occupancy_entropy for m in group_metrics["MCI"]]
    occ_diff = np.mean(hc_occ) - np.mean(mci_occ)

    print("\n" + "=" * 80)
    print("KEY METRIC: OCCUPANCY ENTROPY")
    print("=" * 80)
    print(f"  HC mean:  {np.mean(hc_occ):.3f} (higher = more uniform exploration)")
    print(f"  MCI mean: {np.mean(mci_occ):.3f}")
    print(f"  Δ (HC - MCI): {occ_diff:+.3f}")
    print()
    if occ_diff > 0:
        print("  → HC explores latent space more uniformly")
        print("  → MCI shows concentrated/trapped dynamics (supercriticality)")
    else:
        print("  → Unexpected: MCI shows more uniform exploration")
        print("  → May indicate model not capturing expected dynamics")

    return group_metrics


def extract_continuous_latent(
    model,
    fif_path: Path,
    model_info: dict,
    n_chunks: int,
) -> np.ndarray:
    """Extract continuous latent trajectory from consecutive chunks."""
    data = load_and_preprocess_fif(
        fif_path, FILTER_LOW, FILTER_HIGH, CHUNK_DURATION,
        include_amplitude=model_info["include_amplitude"],
        verbose=False,
    )

    if len(data["chunks"]) == 0:
        return None

    chunks_to_use = min(n_chunks, len(data["chunks"]))
    latents = []

    for cidx in range(chunks_to_use):
        latent = compute_latent_trajectory(model, data["chunks"][cidx], DEVICE)
        latents.append(latent)

    return np.concatenate(latents, axis=0)


# =============================================================================
# COMPARE-ALL MODE: Pooled embedding with per-class density aggregation
# =============================================================================

class PooledEmbedder:
    """
    Fits embedding on pooled data, then transforms individual trajectories.
    Ensures all subjects are in the same 2D space for density comparison.
    """

    def __init__(self, method: str = "pca", tau: int = 5, delay_dim: int = 3):
        self.method = method
        self.tau = tau
        self.delay_dim = delay_dim
        self.transformer = None
        self.bounds = None  # (xmin, xmax, ymin, ymax) for shared grid

    def _preprocess_for_method(self, z: np.ndarray) -> np.ndarray:
        """Preprocess trajectory based on embedding method."""
        if self.method == "tpca":
            # Time-lagged: [z(t), z(t+τ)]
            return np.hstack([z[:-self.tau], z[self.tau:]])
        elif self.method == "delay":
            # Takens delay embedding
            T, D = z.shape
            n_valid = T - (self.delay_dim - 1) * self.tau
            if n_valid < 10:
                return z  # Fallback
            delayed = np.zeros((n_valid, D * self.delay_dim))
            for d in range(self.delay_dim):
                start = d * self.tau
                end = start + n_valid
                delayed[:, d * D:(d + 1) * D] = z[start:end]
            return delayed
        else:
            return z

    def fit(self, trajectories: list[np.ndarray], n_samples_per_traj: int = 500):
        """
        Fit embedding on pooled data sampled equally from each trajectory.

        Args:
            trajectories: List of (T_i, D) arrays
            n_samples_per_traj: Samples to take from each trajectory
        """
        # Sample equally from each trajectory
        pooled = []
        for traj in trajectories:
            processed = self._preprocess_for_method(traj)
            if len(processed) <= n_samples_per_traj:
                pooled.append(processed)
            else:
                indices = np.linspace(0, len(processed) - 1, n_samples_per_traj, dtype=int)
                pooled.append(processed[indices])

        pooled_data = np.vstack(pooled)
        print(f"  Pooled {len(trajectories)} trajectories → {pooled_data.shape[0]} points")

        # Fit transformer
        if self.method in ["pca", "tpca", "delay"]:
            self.transformer = PCA(n_components=2)
            embedded = self.transformer.fit_transform(pooled_data)
        elif self.method == "diffusion":
            # Diffusion maps: compute on pooled, store eigenvectors
            self._fit_diffusion_maps(pooled_data)
            embedded = self._transform_diffusion(pooled_data)
        elif self.method == "umap":
            if HAS_UMAP:
                self.transformer = umap.UMAP(
                    n_components=2,
                    n_neighbors=30,
                    min_dist=0.2,
                    metric='euclidean',
                    random_state=42,  # Fixed seed
                )
                embedded = self.transformer.fit_transform(pooled_data)
            else:
                self.transformer = PCA(n_components=2)
                embedded = self.transformer.fit_transform(pooled_data)

        # Store bounds for shared grid
        margin = 0.05
        x_range = embedded[:, 0].max() - embedded[:, 0].min()
        y_range = embedded[:, 1].max() - embedded[:, 1].min()
        self.bounds = (
            embedded[:, 0].min() - margin * x_range,
            embedded[:, 0].max() + margin * x_range,
            embedded[:, 1].min() - margin * y_range,
            embedded[:, 1].max() + margin * y_range,
        )

    def _fit_diffusion_maps(self, data: np.ndarray, k: int = 10):
        """Fit diffusion maps on pooled data."""
        # Subsample if needed
        max_points = 2000
        if len(data) > max_points:
            indices = np.linspace(0, len(data) - 1, max_points, dtype=int)
            data_sub = data[indices]
        else:
            data_sub = data

        distances = squareform(pdist(data_sub, metric='euclidean'))
        knn_dists = np.sort(distances, axis=1)[:, 1:k+1]
        self.dm_epsilon = np.median(knn_dists)

        K = np.exp(-distances**2 / (2 * self.dm_epsilon**2))
        D_inv = np.diag(1.0 / K.sum(axis=1))
        P = D_inv @ K

        eigenvalues, eigenvectors = np.linalg.eig(P)
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        self.dm_eigenvalues = eigenvalues[idx].real
        self.dm_eigenvectors = eigenvectors[:, idx].real
        self.dm_fit_data = data_sub

    def _transform_diffusion(self, data: np.ndarray) -> np.ndarray:
        """Transform new data using fitted diffusion maps (Nyström extension)."""
        # Compute kernel between new points and fit points
        from scipy.spatial.distance import cdist
        distances = cdist(data, self.dm_fit_data, metric='euclidean')
        K = np.exp(-distances**2 / (2 * self.dm_epsilon**2))
        K_normalized = K / K.sum(axis=1, keepdims=True)

        # Project onto eigenvectors
        embedded = K_normalized @ self.dm_eigenvectors[:, 1:3]
        return embedded

    def transform(self, trajectory: np.ndarray) -> np.ndarray:
        """Transform a single trajectory into the shared 2D space."""
        processed = self._preprocess_for_method(trajectory)

        if self.method in ["pca", "tpca", "delay"]:
            return self.transformer.transform(processed)
        elif self.method == "diffusion":
            return self._transform_diffusion(processed)
        elif self.method == "umap":
            if HAS_UMAP and hasattr(self.transformer, 'transform'):
                return self.transformer.transform(processed)
            else:
                return self.transformer.transform(processed)

    def get_method_name(self) -> str:
        """Get display name for the method."""
        names = {
            "pca": "PCA",
            "tpca": f"tPCA (τ={self.tau})",
            "diffusion": "Diffusion Maps",
            "delay": f"Delay (τ={self.tau}, d={self.delay_dim})",
            "umap": "UMAP",
        }
        return names.get(self.method, self.method.upper())


def compute_density_on_grid(
    embedded: np.ndarray,
    bounds: tuple,
    bins: int = 50,
    sigma: float = 1.5,
) -> np.ndarray:
    """
    Compute normalized 2D density on a shared grid.

    Args:
        embedded: (T, 2) trajectory in shared space
        bounds: (xmin, xmax, ymin, ymax)
        bins: Grid resolution
        sigma: Gaussian smoothing sigma

    Returns:
        (bins, bins) density array, normalized to sum=1
    """
    xmin, xmax, ymin, ymax = bounds
    H, _, _ = np.histogram2d(
        embedded[:, 0], embedded[:, 1],
        bins=bins,
        range=[[xmin, xmax], [ymin, ymax]]
    )
    H = gaussian_filter(H.T, sigma=sigma)

    # Normalize to probability distribution
    if H.sum() > 0:
        H = H / H.sum()

    return H


def compare_all_groups(
    model,
    model_info: dict,
    groups: dict,
    n_subjects_per_group: int,
    n_chunks: int,
    output_dir: Path,
    show_plot: bool = True,
    tau: int = 5,
    delay_dim: int = 3,
):
    """
    Compare phase-space density/flow across all groups using shared embeddings.

    For each embedding method:
    1. Pool trajectories from all subjects, fit embedding once
    2. Transform each subject's trajectory into shared 2D space
    3. Compute per-subject density on shared grid
    4. Average densities within each class
    5. Output density maps and difference maps
    """
    print("\n" + "=" * 80)
    print("COMPARE-ALL MODE: Pooled Embedding with Per-Class Density Aggregation")
    print("=" * 80)

    # Collect trajectories by group
    group_trajectories = {}  # group_key -> [(subject_id, trajectory), ...]
    all_trajectories = []    # For pooled fitting

    for group_key in ["hc", "mci", "ad"]:
        subjects = groups.get(group_key, [])
        if not subjects:
            continue

        group_name = GROUP_NAMES.get(subjects[0][1], group_key.upper())
        group_trajectories[group_key] = []

        print(f"\nExtracting {group_name} trajectories...")
        subjects_processed = 0

        for fif_path, label, condition, subject_id in subjects:
            if subjects_processed >= n_subjects_per_group:
                break

            latent = extract_continuous_latent(model, fif_path, model_info, n_chunks)
            if latent is None or len(latent) < 100:
                print(f"  {subject_id}: skipped (insufficient data)")
                continue

            group_trajectories[group_key].append((subject_id, latent))
            all_trajectories.append(latent)
            print(f"  {subject_id}: {latent.shape[0]} timepoints")
            subjects_processed += 1

        print(f"  → {subjects_processed} subjects extracted")

    if not all_trajectories:
        print("No trajectories extracted!")
        return

    # Embedding methods to compare
    methods = ["pca", "tpca", "diffusion", "delay"]
    if HAS_UMAP:
        methods.append("umap")

    # Store results for summary
    all_metrics = {method: {} for method in methods}  # method -> group -> [FlowMetrics]

    for method in methods:
        print(f"\n{'='*60}")
        print(f"EMBEDDING: {method.upper()}")
        print(f"{'='*60}")

        # Create and fit pooled embedder
        embedder = PooledEmbedder(method=method, tau=tau, delay_dim=delay_dim)
        print(f"Fitting {embedder.get_method_name()} on pooled data...")
        embedder.fit(all_trajectories)

        # Transform each subject and compute density
        group_densities = {}   # group_key -> [density arrays]
        group_metrics_list = {}  # group_key -> [FlowMetrics]

        bins = 50
        for group_key, traj_list in group_trajectories.items():
            if not traj_list:
                continue

            group_name = GROUP_NAMES.get(
                groups[group_key][0][1] if groups.get(group_key) else 0,
                group_key.upper()
            )
            print(f"\nProcessing {group_name}...")

            group_densities[group_key] = []
            group_metrics_list[group_key] = []

            for subject_id, trajectory in traj_list:
                # Transform to shared space
                embedded = embedder.transform(trajectory)

                # Compute density
                density = compute_density_on_grid(embedded, embedder.bounds, bins=bins)
                group_densities[group_key].append(density)

                # Compute flow metrics
                metrics = compute_flow_metrics(embedded)
                group_metrics_list[group_key].append(metrics)

                print(f"  {subject_id}: occ_entropy={metrics.occupancy_entropy:.3f}")

            all_metrics[method][group_key] = group_metrics_list[group_key]

        # Compute mean densities per class
        mean_densities = {}
        for group_key, densities in group_densities.items():
            if densities:
                mean_densities[group_key] = np.mean(densities, axis=0)

        # Create figure: density maps + difference maps
        active_groups = [k for k in ["hc", "mci", "ad"] if k in mean_densities]
        n_groups = len(active_groups)
        n_diffs = n_groups - 1 if "hc" in active_groups else 0

        fig, axes = plt.subplots(2, max(n_groups, n_diffs + 1), figsize=(5 * max(n_groups, 3), 10))

        extent = [embedder.bounds[0], embedder.bounds[1],
                  embedder.bounds[2], embedder.bounds[3]]

        # Row 1: Per-class densities
        for col, group_key in enumerate(active_groups):
            ax = axes[0, col] if n_groups > 1 else axes[0]
            density = mean_densities[group_key]
            group_name = GROUP_NAMES.get(
                groups[group_key][0][1] if groups.get(group_key) else 0,
                group_key.upper()
            )

            im = ax.imshow(density, origin='lower', extent=extent,
                          cmap='hot', aspect='auto')
            ax.set_title(f"{group_name} Mean Density\n(n={len(group_densities[group_key])})",
                        fontweight='bold')
            ax.set_xlabel("PC1" if "pca" in method else "Dim 1")
            ax.set_ylabel("PC2" if "pca" in method else "Dim 2")
            plt.colorbar(im, ax=ax, label='Probability')

        # Hide unused axes in row 1
        for col in range(n_groups, axes.shape[1]):
            axes[0, col].axis('off')

        # Row 2: Difference maps (MCI-HC, AD-HC)
        diff_idx = 0
        if "hc" in mean_densities:
            hc_density = mean_densities["hc"]

            for group_key in ["mci", "ad"]:
                if group_key not in mean_densities:
                    continue

                ax = axes[1, diff_idx] if axes.ndim > 1 else axes[1]
                diff = mean_densities[group_key] - hc_density
                group_name = GROUP_NAMES.get(
                    groups[group_key][0][1] if groups.get(group_key) else 0,
                    group_key.upper()
                )

                # Symmetric colormap for differences
                vmax = np.abs(diff).max()
                im = ax.imshow(diff, origin='lower', extent=extent,
                              cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
                ax.set_title(f"{group_name} − HC\n(Difference Map)", fontweight='bold')
                ax.set_xlabel("PC1" if "pca" in method else "Dim 1")
                ax.set_ylabel("PC2" if "pca" in method else "Dim 2")
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Δ Probability')

                diff_idx += 1

        # Hide unused axes in row 2
        for col in range(diff_idx, axes.shape[1]):
            axes[1, col].axis('off')

        plt.suptitle(f"{embedder.get_method_name()}: Class-Averaged Densities",
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = output_dir / f"compare_all_{method}_density.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    # Output summary table of flow metrics by class
    print("\n" + "=" * 80)
    print("FLOW METRICS SUMMARY BY CLASS (Mean ± Std)")
    print("=" * 80)

    metric_attrs = [
        ("mean_speed", "Mean Speed"),
        ("speed_cv", "Speed CV"),
        ("n_dwell_episodes", "Dwell Episodes"),
        ("occupancy_entropy", "Occ. Entropy"),
        ("path_tortuosity", "Tortuosity"),
        ("explored_variance", "Explored Var"),
    ]

    for method in methods:
        print(f"\n--- {method.upper()} ---")
        print(f"{'Metric':<18} ", end="")

        active_groups = [k for k in ["hc", "mci", "ad"] if k in all_metrics[method]]
        for gk in active_groups:
            gname = GROUP_NAMES.get(groups[gk][0][1], gk.upper())
            print(f"{gname:>18} ", end="")
        print()
        print("-" * (18 + 19 * len(active_groups)))

        for attr, label in metric_attrs:
            print(f"{label:<18} ", end="")
            for gk in active_groups:
                vals = [getattr(m, attr) for m in all_metrics[method][gk]]
                if vals:
                    print(f"{np.mean(vals):>7.3f}±{np.std(vals):<7.3f} ", end="")
                else:
                    print(f"{'N/A':>18} ", end="")
            print()

    # Create combined flow metrics comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, (attr, label) in enumerate(metric_attrs):
        ax = axes.flatten()[idx]

        # Collect data for each group across all methods
        x_positions = np.arange(len(methods))
        width = 0.25

        for gi, gk in enumerate(["hc", "mci", "ad"]):
            if gk not in group_trajectories:
                continue

            means = []
            stds = []
            for method in methods:
                if gk in all_metrics[method]:
                    vals = [getattr(m, attr) for m in all_metrics[method][gk]]
                    means.append(np.mean(vals) if vals else 0)
                    stds.append(np.std(vals) if vals else 0)
                else:
                    means.append(0)
                    stds.append(0)

            color = GROUP_COLORS.get(groups[gk][0][1] if groups.get(gk) else gi, 'gray')
            gname = GROUP_NAMES.get(groups[gk][0][1] if groups.get(gk) else gi, gk.upper())
            ax.bar(x_positions + gi * width, means, width, yerr=stds,
                   label=gname, color=color, alpha=0.7, capsize=3)

        ax.set_xlabel("Embedding Method")
        ax.set_ylabel(label)
        ax.set_title(label, fontweight='bold')
        ax.set_xticks(x_positions + width)
        ax.set_xticklabels([m.upper() for m in methods], rotation=45, ha='right')
        if idx == 0:
            ax.legend()

    plt.suptitle("Flow Metrics Across Embeddings and Groups", fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = output_dir / "compare_all_flow_metrics.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return all_metrics


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


def main():
    parser = argparse.ArgumentParser(
        description="Multi-embedding trajectory analysis (Rabinovich-aligned)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python multi_embedding.py                           # First HC subject, all embeddings
  python multi_embedding.py --subject S001            # Specific subject
  python multi_embedding.py --n-chunks 10             # Longer trajectory
  python multi_embedding.py --compare-groups          # HC vs MCI comparison
  python multi_embedding.py --compare-all             # Pooled density comparison
  python multi_embedding.py --embedding pca           # PCA only
  python multi_embedding.py --embedding diffusion     # Diffusion maps only
  python multi_embedding.py --conditions MCI          # Use MCI subjects
  python multi_embedding.py --list-subjects           # List available subjects

Compare-all mode:
  python multi_embedding.py --compare-all --conditions HID MCI AD --n-subjects 5 --n-chunks 10

  Fits each embedding ONCE on pooled data from all subjects, transforms each
  subject into shared 2D space, computes per-subject density, and averages
  within class. Outputs density maps + difference maps (MCI-HC, AD-HC).

Embedding methods:
  pca        - Standard PCA (baseline)
  tpca       - Time-lagged PCA (temporal structure)
  diffusion  - Diffusion maps (metastability-aware)
  delay      - Takens delay embedding + PCA
  umap       - UMAP (visualization only)
  all        - All of the above (default)
        """
    )
    parser.add_argument("--subject", type=str, default=None,
                        help="Specific subject ID")
    parser.add_argument("--n-chunks", type=int, default=5,
                        help="Number of consecutive chunks (default: 5)")
    parser.add_argument("--n-subjects", type=int, default=5,
                        help="Subjects per group for --compare-groups/--compare-all (default: 5)")
    parser.add_argument("--conditions", type=str, nargs="+", default=["HID", "MCI"],
                        help="Conditions to include (default: HID MCI)")
    parser.add_argument("--embedding", type=str, default="all",
                        choices=["pca", "tpca", "diffusion", "delay", "umap", "all"],
                        help="Embedding method (default: all)")
    parser.add_argument("--tau", type=int, default=5,
                        help="Time lag for tPCA and delay embedding (default: 5)")
    parser.add_argument("--delay-dim", type=int, default=3,
                        help="Embedding dimension for delay embedding (default: 3)")
    parser.add_argument("--compare-groups", action="store_true",
                        help="Compare HC vs MCI flow metrics")
    parser.add_argument("--compare-all", action="store_true",
                        help="Pooled embedding with per-class density aggregation")
    parser.add_argument("--list-subjects", action="store_true",
                        help="List available subjects")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't display plots interactively")
    args = parser.parse_args()

    output_dir = ensure_output_dir()

    # Get subjects
    fif_files = get_fif_files(args.conditions)
    groups = get_subjects_by_group(fif_files)

    group_counts = []
    for key in ["hc", "mci", "ad"]:
        if groups.get(key):
            group_counts.append(f"{len(groups[key])} {key.upper()}")
    print(f"Found {', '.join(group_counts)} subjects")

    if args.list_subjects:
        list_subjects(groups)
        return 0

    # Load model
    print("\nLoading model...")
    model_info = load_model_from_checkpoint(CHECKPOINT_PATH, DEVICE)

    # Get n_channels
    all_subjects = []
    for key in ["hc", "mci", "ad"]:
        all_subjects.extend(groups.get(key, []))

    if not all_subjects:
        print("No subjects found!")
        return 1

    first_data = load_and_preprocess_fif(
        all_subjects[0][0], FILTER_LOW, FILTER_HIGH, CHUNK_DURATION,
        include_amplitude=model_info["include_amplitude"],
        verbose=False,
    )
    model = create_model(first_data["n_channels"], model_info, DEVICE)

    # Handle --compare-all (pooled embedding with density aggregation)
    if args.compare_all:
        compare_all_groups(
            model, model_info, groups,
            args.n_subjects, args.n_chunks,
            output_dir, not args.no_show,
            tau=args.tau,
            delay_dim=args.delay_dim,
        )
        return 0

    # Handle --compare-groups
    if args.compare_groups:
        compare_groups(
            model, model_info, groups,
            args.n_subjects, args.n_chunks,
            output_dir, not args.no_show
        )
        return 0

    # Select subject
    if args.subject:
        selected = None
        for s in all_subjects:
            if args.subject in s[3]:
                selected = s
                break
        if not selected:
            print(f"Subject {args.subject} not found!")
            return 1
    else:
        selected = all_subjects[0]

    fif_path, label, condition, subject_id = selected
    group_name = GROUP_NAMES.get(label, "Unknown")

    print(f"\nSelected: {subject_id} ({condition}, {group_name})")

    # Extract latent
    latent = extract_continuous_latent(model, fif_path, model_info, args.n_chunks)
    if latent is None:
        print("No data!")
        return 1

    print(f"Latent shape: {latent.shape}")

    # Run analysis
    embeddings, metrics = plot_multi_embedding_comparison(
        latent, output_dir, subject_id, group_name, not args.no_show
    )

    # Consistency analysis
    plot_consistency_analysis(embeddings, output_dir, subject_id, not args.no_show)

    print(f"\nAll plots saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
