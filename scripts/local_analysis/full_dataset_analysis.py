#!/usr/bin/env python3
"""
Full-Dataset Statistical Analysis for Systems Neuroscience Paper

Quantifies reorganization of metastable brain dynamics across cognitive decline
using embedding-invariant state-space flow statistics with bootstrap confidence.

This is NOT about classification or biomarkers.
It IS about:
- Robust flow geometry changes across groups
- Statistical confidence via subject-level bootstrapping
- Cross-embedding consistency validation

Usage:
    python full_dataset_analysis.py                    # Fast analysis (pca, tpca, delay)
    python full_dataset_analysis.py --no-show          # Run without displaying plots
    python full_dataset_analysis.py --embedding all    # Include slow methods (diffusion, umap)
    python full_dataset_analysis.py --n-bootstrap 500  # Custom bootstrap iterations
    python full_dataset_analysis.py --quick            # Quick test (100 bootstrap, 5 subjects)
    python full_dataset_analysis.py --embedding pca    # Single embedding method

Key outputs:
    - Bootstrap confidence intervals for all flow metrics
    - Density difference maps with statistical masking
    - Radial density and speed profiles
    - Effect sizes (Cohen's d) with CIs
    - Cross-embedding robustness metrics
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import json
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, entropy
from sklearn.decomposition import PCA
from tqdm import tqdm

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

GROUP_COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#d62728"}  # Blue, Orange, Red
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


# =============================================================================
# DATA STRUCTURES
# =============================================================================

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


@dataclass
class BootstrapResult:
    """Bootstrap confidence interval result."""
    mean: float
    std: float
    ci_low: float
    ci_high: float
    samples: np.ndarray = field(repr=False)

    @classmethod
    def from_samples(cls, samples: np.ndarray, ci: float = 0.95):
        alpha = (1 - ci) / 2
        return cls(
            mean=np.mean(samples),
            std=np.std(samples),
            ci_low=np.percentile(samples, alpha * 100),
            ci_high=np.percentile(samples, (1 - alpha) * 100),
            samples=samples,
        )


@dataclass
class SubjectData:
    """Data for a single subject."""
    subject_id: str
    group: str  # "hc", "mci", "ad"
    label: int
    trajectory: np.ndarray  # (T, D) latent trajectory


# =============================================================================
# EMBEDDING METHODS (from multi_embedding.py)
# =============================================================================

def pca_embedding(z: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, dict]:
    """Standard PCA embedding."""
    pca = PCA(n_components=n_components)
    embedded = pca.fit_transform(z)
    return embedded, {
        "method": "PCA",
        "var_explained": pca.explained_variance_ratio_.tolist(),
        "total_var": sum(pca.explained_variance_ratio_),
    }


def time_lagged_pca(z: np.ndarray, tau: int = 5, n_components: int = 2) -> tuple[np.ndarray, dict]:
    """Time-lagged PCA: PCA on [z(t), z(t+τ)] pairs."""
    T, D = z.shape
    z_lagged = np.hstack([z[:-tau], z[tau:]])
    pca = PCA(n_components=n_components)
    embedded = pca.fit_transform(z_lagged)
    return embedded, {
        "method": f"tPCA (τ={tau})",
        "tau": tau,
        "var_explained": pca.explained_variance_ratio_.tolist(),
        "total_var": sum(pca.explained_variance_ratio_),
    }


def diffusion_maps(z: np.ndarray, n_components: int = 2, k: int = 10) -> tuple[np.ndarray, dict]:
    """Diffusion maps embedding."""
    T = z.shape[0]
    max_points = 2000
    if T > max_points:
        indices = np.linspace(0, T - 1, max_points, dtype=int)
        z_sub = z[indices]
    else:
        indices = np.arange(T)
        z_sub = z

    distances = squareform(pdist(z_sub, metric='euclidean'))
    knn_dists = np.sort(distances, axis=1)[:, 1:k+1]
    eps = np.median(knn_dists)
    K = np.exp(-distances**2 / (2 * eps**2))
    D_inv = np.diag(1.0 / K.sum(axis=1))
    P = D_inv @ K
    eigenvalues, eigenvectors = np.linalg.eig(P)
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx].real
    eigenvectors = eigenvectors[:, idx].real
    embedded_sub = eigenvectors[:, 1:n_components+1]

    if T > max_points:
        from scipy.interpolate import interp1d
        f = interp1d(indices, embedded_sub, axis=0, kind='linear', fill_value='extrapolate')
        embedded = f(np.arange(T))
    else:
        embedded = embedded_sub

    return embedded, {"method": f"Diffusion Maps (k={k})", "epsilon": eps, "k": k}


def delay_embedding(z: np.ndarray, tau: int = 5, dim: int = 3, n_components: int = 2) -> tuple[np.ndarray, dict]:
    """Takens-style delay embedding followed by PCA."""
    T, D = z.shape
    n_valid = T - (dim - 1) * tau
    if n_valid < 100:
        return pca_embedding(z, n_components)  # Fallback

    delayed = np.zeros((n_valid, D * dim))
    for i in range(dim):
        delayed[:, i*D:(i+1)*D] = z[i*tau:i*tau + n_valid]

    pca = PCA(n_components=n_components)
    embedded = pca.fit_transform(delayed)
    return embedded, {"method": f"Delay (τ={tau}, d={dim})", "tau": tau, "dim": dim}


# =============================================================================
# FLOW METRICS
# =============================================================================

def compute_instantaneous_speed(embedded: np.ndarray) -> np.ndarray:
    """Compute speed in embedding space."""
    diff = np.diff(embedded, axis=0)
    return np.linalg.norm(diff, axis=1)


def detect_dwell_episodes(speed: np.ndarray, threshold_percentile: float = 20, min_duration: int = 10) -> list:
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
    """Compute entropy of occupancy distribution."""
    H, _, _ = np.histogram2d(embedded[:, 0], embedded[:, 1], bins=bins)
    H = H.flatten()
    H = H[H > 0]
    p = H / H.sum()
    return entropy(p)


def compute_flow_metrics(embedded: np.ndarray) -> FlowMetrics:
    """Compute comprehensive flow geometry metrics."""
    speed = compute_instantaneous_speed(embedded)
    episodes = detect_dwell_episodes(speed)
    path_length = speed.sum()
    displacement = np.linalg.norm(embedded[-1] - embedded[0])
    tortuosity = path_length / displacement if displacement > 0 else np.inf
    explored_variance = np.var(embedded, axis=0).sum()
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
# POOLED EMBEDDER (for shared coordinate space)
# =============================================================================

class PooledEmbedder:
    """Fits embedding on pooled data, transforms individual trajectories."""

    def __init__(self, method: str = "pca", tau: int = 5, delay_dim: int = 3):
        self.method = method
        self.tau = tau
        self.delay_dim = delay_dim
        self.transformer = None
        self.bounds = None
        self.centroid = None

    def _preprocess(self, z: np.ndarray) -> np.ndarray:
        if self.method == "tpca":
            return np.hstack([z[:-self.tau], z[self.tau:]])
        elif self.method == "delay":
            T, D = z.shape
            n_valid = T - (self.delay_dim - 1) * self.tau
            if n_valid < 10:
                return z
            delayed = np.zeros((n_valid, D * self.delay_dim))
            for d in range(self.delay_dim):
                start = d * self.tau
                end = start + n_valid
                delayed[:, d * D:(d + 1) * D] = z[start:end]
            return delayed
        return z

    def fit(self, trajectories: list[np.ndarray], n_samples_per_traj: int = 500):
        """Fit embedding on pooled data."""
        pooled = []
        for traj in trajectories:
            processed = self._preprocess(traj)
            if len(processed) <= n_samples_per_traj:
                pooled.append(processed)
            else:
                indices = np.linspace(0, len(processed) - 1, n_samples_per_traj, dtype=int)
                pooled.append(processed[indices])

        pooled_data = np.vstack(pooled)

        if self.method in ["pca", "tpca", "delay"]:
            self.transformer = PCA(n_components=2)
            embedded = self.transformer.fit_transform(pooled_data)
        elif self.method == "diffusion":
            self._fit_diffusion_maps(pooled_data)
            embedded = self._transform_diffusion(pooled_data)
        elif self.method == "umap" and HAS_UMAP:
            self.transformer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.2, random_state=42)
            embedded = self.transformer.fit_transform(pooled_data)
        else:
            self.transformer = PCA(n_components=2)
            embedded = self.transformer.fit_transform(pooled_data)

        # Store bounds and centroid
        margin = 0.05
        x_range = embedded[:, 0].max() - embedded[:, 0].min()
        y_range = embedded[:, 1].max() - embedded[:, 1].min()
        self.bounds = (
            embedded[:, 0].min() - margin * x_range,
            embedded[:, 0].max() + margin * x_range,
            embedded[:, 1].min() - margin * y_range,
            embedded[:, 1].max() + margin * y_range,
        )
        self.centroid = embedded.mean(axis=0)

    def _fit_diffusion_maps(self, data: np.ndarray, k: int = 10):
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
        from scipy.spatial.distance import cdist
        distances = cdist(data, self.dm_fit_data, metric='euclidean')
        K = np.exp(-distances**2 / (2 * self.dm_epsilon**2))
        K_normalized = K / K.sum(axis=1, keepdims=True)
        return K_normalized @ self.dm_eigenvectors[:, 1:3]

    def transform(self, trajectory: np.ndarray) -> np.ndarray:
        processed = self._preprocess(trajectory)
        if self.method in ["pca", "tpca", "delay"]:
            return self.transformer.transform(processed)
        elif self.method == "diffusion":
            return self._transform_diffusion(processed)
        elif self.method == "umap" and HAS_UMAP:
            return self.transformer.transform(processed)
        return self.transformer.transform(processed)

    def get_method_name(self) -> str:
        names = {
            "pca": "PCA",
            "tpca": f"tPCA (τ={self.tau})",
            "diffusion": "Diffusion Maps",
            "delay": f"Delay (τ={self.tau}, d={self.delay_dim})",
            "umap": "UMAP",
        }
        return names.get(self.method, self.method.upper())


# =============================================================================
# DENSITY AND RADIAL COMPUTATIONS
# =============================================================================

def compute_density_on_grid(embedded: np.ndarray, bounds: tuple, bins: int = 50, sigma: float = 1.5) -> np.ndarray:
    """Compute normalized 2D density on a shared grid."""
    xmin, xmax, ymin, ymax = bounds
    H, _, _ = np.histogram2d(embedded[:, 0], embedded[:, 1], bins=bins, range=[[xmin, xmax], [ymin, ymax]])
    H = gaussian_filter(H.T, sigma=sigma)
    if H.sum() > 0:
        H = H / H.sum()
    return H


def compute_radial_profile(embedded: np.ndarray, centroid: np.ndarray, n_bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Compute radial density profile from centroid."""
    radii = np.linalg.norm(embedded - centroid, axis=1)
    max_r = np.percentile(radii, 99)
    bins = np.linspace(0, max_r, n_bins + 1)
    counts, _ = np.histogram(radii, bins=bins)
    # Normalize by ring area
    ring_areas = np.pi * (bins[1:]**2 - bins[:-1]**2)
    density = counts / (ring_areas * len(radii))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    return bin_centers, density


def compute_radial_speed_profile(embedded: np.ndarray, centroid: np.ndarray, n_bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean speed as a function of radius."""
    speed = compute_instantaneous_speed(embedded)
    radii = np.linalg.norm(embedded[:-1] - centroid, axis=1)  # Match speed length
    max_r = np.percentile(radii, 99)
    bins = np.linspace(0, max_r, n_bins + 1)
    bin_indices = np.digitize(radii, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    mean_speeds = np.zeros(n_bins)
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            mean_speeds[i] = speed[mask].mean()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    return bin_centers, mean_speeds


# =============================================================================
# EFFECT SIZE COMPUTATION
# =============================================================================

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


# =============================================================================
# BOOTSTRAP ANALYSIS
# =============================================================================

def bootstrap_flow_metrics(
    subjects: list[SubjectData],
    embedder: PooledEmbedder,
    n_bootstrap: int = 500,
    random_state: int = 42,
) -> dict[str, BootstrapResult]:
    """Bootstrap flow metrics at the subject level."""
    rng = np.random.RandomState(random_state)
    n_subjects = len(subjects)

    metric_names = ["mean_speed", "speed_cv", "n_dwell_episodes", "occupancy_entropy", "path_tortuosity", "explored_variance"]
    samples = {name: [] for name in metric_names}

    for _ in range(n_bootstrap):
        # Sample subjects with replacement
        indices = rng.choice(n_subjects, size=n_subjects, replace=True)

        # Compute metrics for bootstrap sample
        bootstrap_metrics = []
        for idx in indices:
            embedded = embedder.transform(subjects[idx].trajectory)
            metrics = compute_flow_metrics(embedded)
            bootstrap_metrics.append(metrics)

        # Store mean of each metric
        for name in metric_names:
            vals = [getattr(m, name) for m in bootstrap_metrics]
            samples[name].append(np.mean(vals))

    # Convert to BootstrapResult
    results = {}
    for name in metric_names:
        results[name] = BootstrapResult.from_samples(np.array(samples[name]))

    return results


def bootstrap_density_difference(
    hc_subjects: list[SubjectData],
    disease_subjects: list[SubjectData],
    embedder: PooledEmbedder,
    n_bootstrap: int = 500,
    bins: int = 50,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bootstrap density difference maps.

    Returns:
        mean_diff: Mean difference map
        ci_low: Lower 95% CI
        ci_high: Upper 95% CI
    """
    rng = np.random.RandomState(random_state)
    n_hc, n_disease = len(hc_subjects), len(disease_subjects)

    diff_samples = []

    for _ in range(n_bootstrap):
        # Sample subjects with replacement
        hc_indices = rng.choice(n_hc, size=n_hc, replace=True)
        disease_indices = rng.choice(n_disease, size=n_disease, replace=True)

        # Compute densities
        hc_densities = []
        for idx in hc_indices:
            embedded = embedder.transform(hc_subjects[idx].trajectory)
            density = compute_density_on_grid(embedded, embedder.bounds, bins=bins)
            hc_densities.append(density)

        disease_densities = []
        for idx in disease_indices:
            embedded = embedder.transform(disease_subjects[idx].trajectory)
            density = compute_density_on_grid(embedded, embedder.bounds, bins=bins)
            disease_densities.append(density)

        # Compute difference
        hc_mean = np.mean(hc_densities, axis=0)
        disease_mean = np.mean(disease_densities, axis=0)
        diff_samples.append(disease_mean - hc_mean)

    diff_samples = np.array(diff_samples)
    mean_diff = np.mean(diff_samples, axis=0)
    ci_low = np.percentile(diff_samples, 2.5, axis=0)
    ci_high = np.percentile(diff_samples, 97.5, axis=0)

    return mean_diff, ci_low, ci_high


def bootstrap_radial_profiles(
    subjects: list[SubjectData],
    embedder: PooledEmbedder,
    n_bootstrap: int = 500,
    n_bins: int = 20,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Bootstrap radial density and speed profiles.

    Returns:
        bin_centers, density_ci (mean, low, high), speed_ci (mean, low, high)
    """
    rng = np.random.RandomState(random_state)
    n_subjects = len(subjects)

    density_samples = []
    speed_samples = []
    bin_centers = None

    for _ in range(n_bootstrap):
        indices = rng.choice(n_subjects, size=n_subjects, replace=True)

        all_densities = []
        all_speeds = []

        for idx in indices:
            embedded = embedder.transform(subjects[idx].trajectory)
            bc, density = compute_radial_profile(embedded, embedder.centroid, n_bins)
            _, speed = compute_radial_speed_profile(embedded, embedder.centroid, n_bins)
            all_densities.append(density)
            all_speeds.append(speed)
            if bin_centers is None:
                bin_centers = bc

        density_samples.append(np.mean(all_densities, axis=0))
        speed_samples.append(np.mean(all_speeds, axis=0))

    density_samples = np.array(density_samples)
    speed_samples = np.array(speed_samples)

    density_ci = (
        np.mean(density_samples, axis=0),
        np.percentile(density_samples, 2.5, axis=0),
        np.percentile(density_samples, 97.5, axis=0),
    )

    speed_ci = (
        np.mean(speed_samples, axis=0),
        np.percentile(speed_samples, 2.5, axis=0),
        np.percentile(speed_samples, 97.5, axis=0),
    )

    return bin_centers, density_ci, speed_ci


def bootstrap_effect_size(
    group1_values: np.ndarray,
    group2_values: np.ndarray,
    n_bootstrap: int = 500,
    random_state: int = 42,
) -> BootstrapResult:
    """Bootstrap Cohen's d effect size."""
    rng = np.random.RandomState(random_state)
    n1, n2 = len(group1_values), len(group2_values)

    d_samples = []
    for _ in range(n_bootstrap):
        idx1 = rng.choice(n1, size=n1, replace=True)
        idx2 = rng.choice(n2, size=n2, replace=True)
        d = cohens_d(group1_values[idx1], group2_values[idx2])
        d_samples.append(d)

    return BootstrapResult.from_samples(np.array(d_samples))


# =============================================================================
# CROSS-EMBEDDING ROBUSTNESS
# =============================================================================

def compute_cross_embedding_robustness(
    subjects: list[SubjectData],
    methods: list[str],
    tau: int = 5,
    delay_dim: int = 3,
) -> dict:
    """Quantify similarity of flow metrics across embeddings."""
    # Get all trajectories
    trajectories = [s.trajectory for s in subjects]

    # Fit embedders for each method
    embedders = {}
    for method in methods:
        embedder = PooledEmbedder(method=method, tau=tau, delay_dim=delay_dim)
        embedder.fit(trajectories)
        embedders[method] = embedder

    # Compute metrics for each subject under each embedding
    all_metrics = {method: [] for method in methods}
    for subject in subjects:
        for method in methods:
            embedded = embedders[method].transform(subject.trajectory)
            metrics = compute_flow_metrics(embedded)
            all_metrics[method].append(metrics)

    # Compute cross-embedding correlations for each metric
    metric_names = ["mean_speed", "speed_cv", "occupancy_entropy", "path_tortuosity", "explored_variance"]
    correlations = {}

    for metric_name in metric_names:
        method_values = {}
        for method in methods:
            method_values[method] = np.array([getattr(m, metric_name) for m in all_metrics[method]])

        # Pairwise correlations
        corr_matrix = np.zeros((len(methods), len(methods)))
        for i, m1 in enumerate(methods):
            for j, m2 in enumerate(methods):
                corr, _ = spearmanr(method_values[m1], method_values[m2])
                corr_matrix[i, j] = corr

        correlations[metric_name] = {
            "matrix": corr_matrix,
            "mean_off_diagonal": np.mean(corr_matrix[np.triu_indices(len(methods), k=1)]),
        }

    return {
        "correlations": correlations,
        "methods": methods,
        "n_subjects": len(subjects),
    }


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_bootstrap_metrics_comparison(
    group_results: dict[str, dict[str, BootstrapResult]],
    output_dir: Path,
    embedding_name: str,
    show_plot: bool = True,
):
    """Plot bootstrap flow metrics comparison across groups."""
    metric_names = ["mean_speed", "speed_cv", "n_dwell_episodes", "occupancy_entropy", "path_tortuosity", "explored_variance"]
    metric_labels = ["Mean Speed", "Speed CV", "Dwell Episodes", "Occupancy Entropy", "Tortuosity", "Explored Variance"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    groups = list(group_results.keys())
    colors = [GROUP_COLORS.get({"HC": 0, "MCI": 1, "AD": 2}.get(g, 0), "gray") for g in groups]

    for idx, (metric, label) in enumerate(zip(metric_names, metric_labels)):
        ax = axes.flatten()[idx]

        x_pos = np.arange(len(groups))
        means = [group_results[g][metric].mean for g in groups]
        ci_lows = [group_results[g][metric].mean - group_results[g][metric].ci_low for g in groups]
        ci_highs = [group_results[g][metric].ci_high - group_results[g][metric].mean for g in groups]

        bars = ax.bar(x_pos, means, color=colors, alpha=0.7, edgecolor='black')
        ax.errorbar(x_pos, means, yerr=[ci_lows, ci_highs], fmt='none', color='black', capsize=5, linewidth=2)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(groups)
        ax.set_ylabel(label)
        ax.set_title(label, fontweight='bold')

        # Add CI text
        for i, g in enumerate(groups):
            ci_text = f"[{group_results[g][metric].ci_low:.3f}, {group_results[g][metric].ci_high:.3f}]"
            ax.text(i, means[i] + ci_highs[i] + 0.01 * max(means), ci_text, ha='center', fontsize=7)

    plt.suptitle(f"Flow Metrics with 95% Bootstrap CIs ({embedding_name})", fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = output_dir / f"bootstrap_metrics_{embedding_name.lower().replace(' ', '_')}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_density_difference_with_ci(
    mean_diff: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    bounds: tuple,
    group_name: str,
    output_dir: Path,
    embedding_name: str,
    show_plot: bool = True,
):
    """Plot density difference with statistical masking."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    extent = list(bounds)

    # Mean difference
    vmax = np.abs(mean_diff).max()
    im1 = axes[0].imshow(mean_diff, origin='lower', extent=extent, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
    axes[0].set_title(f"{group_name} − HC\nMean Difference", fontweight='bold')
    plt.colorbar(im1, ax=axes[0], label='Δ Probability', shrink=0.8)

    # Statistically significant regions (CI doesn't include 0)
    significant = (ci_low > 0) | (ci_high < 0)
    masked_diff = np.where(significant, mean_diff, np.nan)
    im2 = axes[1].imshow(masked_diff, origin='lower', extent=extent, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
    axes[1].set_title(f"Significant Regions\n(95% CI excludes 0)", fontweight='bold')
    plt.colorbar(im2, ax=axes[1], label='Δ Probability', shrink=0.8)

    # CI width (uncertainty)
    ci_width = ci_high - ci_low
    im3 = axes[2].imshow(ci_width, origin='lower', extent=extent, cmap='viridis', aspect='equal')
    axes[2].set_title("Uncertainty\n(CI Width)", fontweight='bold')
    plt.colorbar(im3, ax=axes[2], label='CI Width', shrink=0.8)

    for ax in axes:
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")

    plt.suptitle(f"Density Difference: {group_name} vs HC ({embedding_name})", fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = output_dir / f"density_diff_ci_{group_name.lower()}_{embedding_name.lower().replace(' ', '_')}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_radial_profiles(
    group_profiles: dict[str, tuple],  # group -> (bin_centers, density_ci, speed_ci)
    output_dir: Path,
    embedding_name: str,
    show_plot: bool = True,
):
    """Plot radial density and speed profiles with CIs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for group, (bin_centers, density_ci, speed_ci) in group_profiles.items():
        color = GROUP_COLORS.get({"HC": 0, "MCI": 1, "AD": 2}.get(group, 0), "gray")

        # Density profile
        axes[0].plot(bin_centers, density_ci[0], color=color, label=group, linewidth=2)
        axes[0].fill_between(bin_centers, density_ci[1], density_ci[2], color=color, alpha=0.2)

        # Speed profile
        axes[1].plot(bin_centers, speed_ci[0], color=color, label=group, linewidth=2)
        axes[1].fill_between(bin_centers, speed_ci[1], speed_ci[2], color=color, alpha=0.2)

    axes[0].set_xlabel("Radial Distance from Centroid")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Radial Density Profile", fontweight='bold')
    axes[0].legend()

    axes[1].set_xlabel("Radial Distance from Centroid")
    axes[1].set_ylabel("Mean Speed")
    axes[1].set_title("Radial Speed Profile", fontweight='bold')
    axes[1].legend()

    plt.suptitle(f"Radial Profiles with 95% Bootstrap CIs ({embedding_name})", fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = output_dir / f"radial_profiles_{embedding_name.lower().replace(' ', '_')}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_effect_sizes(
    effect_sizes: dict[str, dict[str, BootstrapResult]],
    output_dir: Path,
    show_plot: bool = True,
):
    """Plot Cohen's d effect sizes with CIs."""
    comparisons = list(effect_sizes.keys())
    metrics = list(effect_sizes[comparisons[0]].keys())

    n_comparisons = len(comparisons)
    n_metrics = len(metrics)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(n_metrics)
    width = 0.8 / n_comparisons

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, comparison in enumerate(comparisons):
        means = [effect_sizes[comparison][m].mean for m in metrics]
        ci_lows = [effect_sizes[comparison][m].mean - effect_sizes[comparison][m].ci_low for m in metrics]
        ci_highs = [effect_sizes[comparison][m].ci_high - effect_sizes[comparison][m].mean for m in metrics]

        offset = (i - n_comparisons / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, label=comparison, color=colors[i % len(colors)], alpha=0.7)
        ax.errorbar(x + offset, means, yerr=[ci_lows, ci_highs], fmt='none', color='black', capsize=3)

    # Add reference lines
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(0.2, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axhline(-0.2, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axhline(-0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axhline(0.8, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axhline(-0.8, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.set_ylabel("Cohen's d")
    ax.set_title("Effect Sizes with 95% Bootstrap CIs\n(|d|>0.2: small, |d|>0.5: medium, |d|>0.8: large)", fontweight='bold')
    ax.legend()

    plt.tight_layout()

    save_path = output_dir / "effect_sizes.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_cross_embedding_robustness(
    robustness: dict,
    output_dir: Path,
    show_plot: bool = True,
):
    """Plot cross-embedding consistency heatmaps."""
    metrics = list(robustness["correlations"].keys())
    methods = robustness["methods"]
    n_metrics = len(metrics)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, metric in enumerate(metrics):
        ax = axes.flatten()[idx]
        corr_matrix = robustness["correlations"][metric]["matrix"]
        mean_corr = robustness["correlations"][metric]["mean_off_diagonal"]

        im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
        ax.set_xticks(range(len(methods)))
        ax.set_yticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_yticklabels(methods)
        ax.set_title(f"{metric.replace('_', ' ').title()}\nMean ρ = {mean_corr:.2f}", fontweight='bold')

        # Add correlation values
        for i in range(len(methods)):
            for j in range(len(methods)):
                ax.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center', fontsize=8)

        plt.colorbar(im, ax=ax, label='Spearman ρ')

    # Hide unused axes
    for idx in range(n_metrics, 6):
        axes.flatten()[idx].axis('off')

    plt.suptitle(f"Cross-Embedding Robustness (n={robustness['n_subjects']} subjects)", fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = output_dir / "cross_embedding_robustness.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# GROUP FLOW FIELD ANALYSIS (Rabinovich-style)
# =============================================================================

def compute_group_flow_field(
    subjects: list[SubjectData],
    embedder: PooledEmbedder,
    grid_size: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute aggregate flow field for a group of subjects.

    Returns:
        X, Y: Grid coordinates
        flow_x, flow_y: Mean flow vectors
        counts: Number of samples per bin
    """
    # Get bounds from embedder
    xmin, xmax, ymin, ymax = embedder.bounds

    x_edges = np.linspace(xmin, xmax, grid_size + 1)
    y_edges = np.linspace(ymin, ymax, grid_size + 1)

    # Accumulate flow across all subjects
    flow_x_sum = np.zeros((grid_size, grid_size))
    flow_y_sum = np.zeros((grid_size, grid_size))
    counts = np.zeros((grid_size, grid_size))

    for subject in subjects:
        embedded = embedder.transform(subject.trajectory)

        # Compute displacements
        dx = np.diff(embedded[:, 0])
        dy = np.diff(embedded[:, 1])

        # Bin displacements
        for i in range(len(dx)):
            x_bin = np.searchsorted(x_edges[:-1], embedded[i, 0]) - 1
            y_bin = np.searchsorted(y_edges[:-1], embedded[i, 1]) - 1

            x_bin = np.clip(x_bin, 0, grid_size - 1)
            y_bin = np.clip(y_bin, 0, grid_size - 1)

            flow_x_sum[y_bin, x_bin] += dx[i]
            flow_y_sum[y_bin, x_bin] += dy[i]
            counts[y_bin, x_bin] += 1

    # Average
    flow_x = np.zeros_like(flow_x_sum)
    flow_y = np.zeros_like(flow_y_sum)
    mask = counts > 0
    flow_x[mask] = flow_x_sum[mask] / counts[mask]
    flow_y[mask] = flow_y_sum[mask] / counts[mask]

    # Grid centers
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers)

    return X, Y, flow_x, flow_y, counts


def compute_flow_divergence(flow_x: np.ndarray, flow_y: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """Compute divergence of flow field (positive = sources, negative = sinks)."""
    # Use central differences
    dfx_dx = np.gradient(flow_x, dx, axis=1)
    dfy_dy = np.gradient(flow_y, dx, axis=0)
    return dfx_dx + dfy_dy


def compute_flow_curl(flow_x: np.ndarray, flow_y: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """Compute curl (vorticity) of 2D flow field."""
    dfx_dy = np.gradient(flow_x, dx, axis=0)
    dfy_dx = np.gradient(flow_y, dx, axis=1)
    return dfy_dx - dfx_dy


def plot_group_flow_fields(
    subject_data: dict[str, list[SubjectData]],
    embedder: PooledEmbedder,
    output_dir: Path,
    embedding_name: str,
    grid_size: int = 20,
    show_plot: bool = True,
):
    """
    Plot Rabinovich-style flow fields for each group side by side.

    Shows:
    - Flow vectors (quiver plot)
    - Flow magnitude
    - Trajectory density overlay
    """
    groups_with_data = [g for g in ["HC", "MCI", "AD"] if len(subject_data.get(g, [])) >= 3]
    n_groups = len(groups_with_data)

    if n_groups < 2:
        print("  Not enough groups for flow field comparison")
        return

    fig, axes = plt.subplots(2, n_groups, figsize=(6 * n_groups, 12))
    if n_groups == 1:
        axes = axes.reshape(-1, 1)

    extent = list(embedder.bounds)
    flow_data = {}

    for idx, group in enumerate(groups_with_data):
        subjects = subject_data[group]

        # Compute flow field
        X, Y, flow_x, flow_y, counts = compute_group_flow_field(subjects, embedder, grid_size)
        flow_data[group] = (X, Y, flow_x, flow_y, counts)

        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        color = GROUP_COLORS.get({"HC": 0, "MCI": 1, "AD": 2}.get(group, 0), "gray")

        # Top row: Flow field with trajectory overlay
        ax = axes[0, idx]

        # Get all embedded trajectories for density
        all_embedded = []
        for s in subjects:
            emb = embedder.transform(s.trajectory)
            all_embedded.append(emb)
        all_embedded = np.vstack(all_embedded)

        # Plot trajectory density as background
        H, _, _ = np.histogram2d(all_embedded[:, 0], all_embedded[:, 1], bins=50,
                                  range=[[extent[0], extent[1]], [extent[2], extent[3]]])
        H = gaussian_filter(H.T, sigma=1.5)
        ax.imshow(H, origin='lower', extent=extent, cmap='Greys', alpha=0.4, aspect='equal')

        # Plot flow vectors
        ax.quiver(X, Y, flow_x, flow_y, magnitude, cmap='plasma', alpha=0.9, scale=None)

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect('equal')
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_title(f"{group} (n={len(subjects)})\nFlow Field + Density", fontweight='bold', color=color)

        # Bottom row: Flow magnitude heatmap
        ax = axes[1, idx]
        im = ax.imshow(magnitude, origin='lower', extent=extent, cmap='viridis', aspect='equal')
        plt.colorbar(im, ax=ax, label='Flow Magnitude', shrink=0.8)

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect('equal')
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_title(f"{group} Flow Magnitude", fontweight='bold')

    plt.suptitle(f"Group Flow Fields - Rabinovich-style ({embedding_name})", fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = output_dir / f"group_flow_fields_{embedding_name.lower().replace(' ', '_')}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return flow_data


def plot_flow_difference(
    subject_data: dict[str, list[SubjectData]],
    embedder: PooledEmbedder,
    output_dir: Path,
    embedding_name: str,
    grid_size: int = 20,
    show_plot: bool = True,
):
    """
    Plot flow field differences between groups.

    Shows:
    - Vector difference (MCI - HC, AD - HC)
    - Magnitude difference
    - Divergence/curl differences
    """
    hc_subjects = subject_data.get("HC", [])
    if len(hc_subjects) < 3:
        print("  Not enough HC subjects for flow difference")
        return

    # Compute HC flow
    X, Y, hc_flow_x, hc_flow_y, hc_counts = compute_group_flow_field(hc_subjects, embedder, grid_size)
    hc_mag = np.sqrt(hc_flow_x**2 + hc_flow_y**2)

    disease_groups = [g for g in ["MCI", "AD"] if len(subject_data.get(g, [])) >= 3]
    n_disease = len(disease_groups)

    if n_disease == 0:
        print("  Not enough disease subjects for flow difference")
        return

    fig, axes = plt.subplots(3, n_disease, figsize=(7 * n_disease, 15))
    if n_disease == 1:
        axes = axes.reshape(-1, 1)

    extent = list(embedder.bounds)

    for idx, disease_group in enumerate(disease_groups):
        disease_subjects = subject_data[disease_group]

        # Compute disease flow
        _, _, disease_flow_x, disease_flow_y, disease_counts = compute_group_flow_field(
            disease_subjects, embedder, grid_size
        )
        disease_mag = np.sqrt(disease_flow_x**2 + disease_flow_y**2)

        # Differences
        diff_flow_x = disease_flow_x - hc_flow_x
        diff_flow_y = disease_flow_y - hc_flow_y
        diff_mag = disease_mag - hc_mag
        diff_vector_mag = np.sqrt(diff_flow_x**2 + diff_flow_y**2)

        color = GROUP_COLORS.get({"HC": 0, "MCI": 1, "AD": 2}.get(disease_group, 1))

        # Row 1: Vector difference (quiver)
        ax = axes[0, idx]
        ax.quiver(X, Y, diff_flow_x, diff_flow_y, diff_vector_mag, cmap='coolwarm', alpha=0.9)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect('equal')
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_title(f"{disease_group} − HC\nFlow Vector Difference", fontweight='bold', color=color)

        # Row 2: Magnitude difference (heatmap)
        ax = axes[1, idx]
        vmax = np.abs(diff_mag).max()
        im = ax.imshow(diff_mag, origin='lower', extent=extent, cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, aspect='equal')
        plt.colorbar(im, ax=ax, label='Δ Magnitude', shrink=0.8)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect('equal')
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_title(f"{disease_group} − HC\nMagnitude Difference", fontweight='bold')

        # Row 3: Divergence difference
        ax = axes[2, idx]
        hc_div = compute_flow_divergence(hc_flow_x, hc_flow_y)
        disease_div = compute_flow_divergence(disease_flow_x, disease_flow_y)
        diff_div = disease_div - hc_div
        vmax = np.abs(diff_div).max()
        im = ax.imshow(diff_div, origin='lower', extent=extent, cmap='PuOr',
                       vmin=-vmax, vmax=vmax, aspect='equal')
        plt.colorbar(im, ax=ax, label='Δ Divergence', shrink=0.8)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect('equal')
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_title(f"{disease_group} − HC\nDivergence Difference\n(+sources, −sinks)", fontweight='bold')

    plt.suptitle(f"Flow Field Differences ({embedding_name})", fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = output_dir / f"flow_difference_{embedding_name.lower().replace(' ', '_')}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def compute_flow_statistics(
    X: np.ndarray, Y: np.ndarray,
    flow_x: np.ndarray, flow_y: np.ndarray,
    counts: np.ndarray
) -> dict:
    """Compute summary statistics for a flow field."""
    magnitude = np.sqrt(flow_x**2 + flow_y**2)

    # Weight by counts (how well sampled each bin is)
    valid = counts > 10

    if valid.sum() == 0:
        return {"mean_magnitude": 0, "std_magnitude": 0, "mean_divergence": 0}

    weighted_mag = magnitude[valid]

    # Divergence and curl
    div = compute_flow_divergence(flow_x, flow_y)
    curl = compute_flow_curl(flow_x, flow_y)

    return {
        "mean_magnitude": weighted_mag.mean(),
        "std_magnitude": weighted_mag.std(),
        "max_magnitude": weighted_mag.max(),
        "mean_divergence": div[valid].mean(),
        "std_divergence": div[valid].std(),
        "mean_curl": curl[valid].mean(),
        "std_curl": curl[valid].std(),
        "coverage": valid.sum() / valid.size,  # Fraction of grid with data
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def load_all_subjects(
    model,
    model_info: dict,
    groups: dict,
    n_subjects_per_group: Optional[int],
    n_chunks: int,
) -> dict[str, list[SubjectData]]:
    """Load all subjects' latent trajectories."""
    subject_data = {"HC": [], "MCI": [], "AD": []}

    for group_key in ["hc", "mci", "ad"]:
        subjects = groups.get(group_key, [])
        if not subjects:
            continue

        group_name = GROUP_NAMES.get(subjects[0][1], group_key.upper())
        max_subjects = n_subjects_per_group if n_subjects_per_group else len(subjects)

        print(f"\nLoading {group_name} subjects (max {max_subjects})...")
        subjects_processed = 0

        for fif_path, label, condition, subject_id in tqdm(subjects, desc=group_name):
            if subjects_processed >= max_subjects:
                break

            # Load and extract latent
            data = load_and_preprocess_fif(
                fif_path, FILTER_LOW, FILTER_HIGH, CHUNK_DURATION,
                include_amplitude=model_info["include_amplitude"],
                verbose=False,
            )

            if len(data["chunks"]) == 0:
                continue

            chunks_to_use = min(n_chunks, len(data["chunks"]))
            latents = []
            for cidx in range(chunks_to_use):
                latent = compute_latent_trajectory(model, data["chunks"][cidx], DEVICE)
                latents.append(latent)

            trajectory = np.concatenate(latents, axis=0)

            subject_data[group_name].append(SubjectData(
                subject_id=subject_id,
                group=group_key,
                label=label,
                trajectory=trajectory,
            ))
            subjects_processed += 1

        print(f"  Loaded {subjects_processed} {group_name} subjects")

    return subject_data


def run_full_analysis(
    subject_data: dict[str, list[SubjectData]],
    output_dir: Path,
    methods: list[str],
    n_bootstrap: int,
    tau: int,
    delay_dim: int,
    show_plot: bool,
):
    """Run full statistical analysis."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "n_bootstrap": n_bootstrap,
        "methods": methods,
        "n_subjects": {g: len(s) for g, s in subject_data.items()},
    }

    # Get all trajectories for pooled fitting
    all_subjects = []
    for group_subjects in subject_data.values():
        all_subjects.extend(group_subjects)
    all_trajectories = [s.trajectory for s in all_subjects]

    for method in methods:
        print(f"\n{'='*80}")
        print(f"ANALYZING: {method.upper()}")
        print(f"{'='*80}")

        # Fit pooled embedder
        print(f"\nFitting {method} embedder on pooled data...")
        embedder = PooledEmbedder(method=method, tau=tau, delay_dim=delay_dim)
        embedder.fit(all_trajectories)
        embedding_name = embedder.get_method_name()

        # 1. Bootstrap flow metrics for each group
        print(f"\nBootstrapping flow metrics ({n_bootstrap} iterations)...")
        group_bootstrap_results = {}
        for group, subjects in subject_data.items():
            if len(subjects) < 3:
                print(f"  Skipping {group} (only {len(subjects)} subjects)")
                continue
            print(f"  {group}...", end=" ", flush=True)
            group_bootstrap_results[group] = bootstrap_flow_metrics(subjects, embedder, n_bootstrap)
            print("done")

        # Plot bootstrap metrics comparison
        if len(group_bootstrap_results) > 1:
            plot_bootstrap_metrics_comparison(group_bootstrap_results, output_dir, embedding_name, show_plot)

        # 2. Density difference with CI
        print(f"\nBootstrapping density differences...")
        hc_subjects = subject_data.get("HC", [])
        for disease_group in ["MCI", "AD"]:
            disease_subjects = subject_data.get(disease_group, [])
            if len(hc_subjects) < 3 or len(disease_subjects) < 3:
                continue

            print(f"  {disease_group} - HC...", end=" ", flush=True)
            mean_diff, ci_low, ci_high = bootstrap_density_difference(
                hc_subjects, disease_subjects, embedder, n_bootstrap
            )
            print("done")

            plot_density_difference_with_ci(
                mean_diff, ci_low, ci_high, embedder.bounds, disease_group, output_dir, embedding_name, show_plot
            )

        # 3. Radial profiles
        print(f"\nBootstrapping radial profiles...")
        group_profiles = {}
        for group, subjects in subject_data.items():
            if len(subjects) < 3:
                continue
            print(f"  {group}...", end=" ", flush=True)
            bin_centers, density_ci, speed_ci = bootstrap_radial_profiles(subjects, embedder, n_bootstrap)
            group_profiles[group] = (bin_centers, density_ci, speed_ci)
            print("done")

        if len(group_profiles) > 1:
            plot_radial_profiles(group_profiles, output_dir, embedding_name, show_plot)

        # 4. Group flow fields (Rabinovich-style)
        print(f"\nComputing group flow fields...")
        flow_data = plot_group_flow_fields(subject_data, embedder, output_dir, embedding_name, show_plot=show_plot)

        # 5. Flow field differences
        print(f"\nComputing flow field differences...")
        plot_flow_difference(subject_data, embedder, output_dir, embedding_name, show_plot=show_plot)

        # Compute flow statistics for results
        flow_stats = {}
        if flow_data:
            for group, (X, Y, flow_x, flow_y, counts) in flow_data.items():
                flow_stats[group] = compute_flow_statistics(X, Y, flow_x, flow_y, counts)

        # Store results
        results[method] = {
            "bootstrap_metrics": {
                g: {m: {"mean": r.mean, "ci_low": r.ci_low, "ci_high": r.ci_high}
                    for m, r in metrics.items()}
                for g, metrics in group_bootstrap_results.items()
            },
            "flow_statistics": flow_stats,
        }

    # 6. Effect sizes (aggregate across embeddings using PCA)
    print(f"\n{'='*80}")
    print("COMPUTING EFFECT SIZES")
    print(f"{'='*80}")

    # Use PCA for effect size computation
    pca_embedder = PooledEmbedder(method="pca", tau=tau, delay_dim=delay_dim)
    pca_embedder.fit(all_trajectories)

    # Get metrics for each subject
    group_metric_values = {g: {} for g in subject_data.keys()}
    metric_names = ["mean_speed", "speed_cv", "n_dwell_episodes", "occupancy_entropy", "path_tortuosity", "explored_variance"]

    for group, subjects in subject_data.items():
        for metric in metric_names:
            group_metric_values[group][metric] = []
        for subject in subjects:
            embedded = pca_embedder.transform(subject.trajectory)
            metrics = compute_flow_metrics(embedded)
            for metric in metric_names:
                group_metric_values[group][metric].append(getattr(metrics, metric))

    # Compute effect sizes
    effect_sizes = {}
    hc_values = group_metric_values.get("HC", {})

    for disease_group in ["MCI", "AD"]:
        disease_values = group_metric_values.get(disease_group, {})
        if not disease_values or not hc_values:
            continue

        comparison = f"HC vs {disease_group}"
        effect_sizes[comparison] = {}

        for metric in metric_names:
            if metric not in hc_values or metric not in disease_values:
                continue
            if len(hc_values[metric]) < 3 or len(disease_values[metric]) < 3:
                continue

            print(f"  {comparison} - {metric}...", end=" ", flush=True)
            effect_sizes[comparison][metric] = bootstrap_effect_size(
                np.array(hc_values[metric]),
                np.array(disease_values[metric]),
                n_bootstrap
            )
            print(f"d = {effect_sizes[comparison][metric].mean:.3f}")

    if effect_sizes:
        plot_effect_sizes(effect_sizes, output_dir, show_plot)

    # 5. Cross-embedding robustness
    print(f"\n{'='*80}")
    print("COMPUTING CROSS-EMBEDDING ROBUSTNESS")
    print(f"{'='*80}")

    robustness = compute_cross_embedding_robustness(all_subjects, methods, tau, delay_dim)
    plot_cross_embedding_robustness(robustness, output_dir, show_plot)

    # Store effect sizes
    results["effect_sizes"] = {
        comparison: {m: {"mean": r.mean, "ci_low": r.ci_low, "ci_high": r.ci_high}
                     for m, r in metrics.items()}
        for comparison, metrics in effect_sizes.items()
    }

    # Store robustness
    results["cross_embedding_robustness"] = {
        metric: {"mean_correlation": data["mean_off_diagonal"]}
        for metric, data in robustness["correlations"].items()
    }

    # Save results to JSON
    results_path = output_dir / "full_analysis_results.json"

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    import json
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            return convert_numpy(obj)

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to: {results_path}")

    return results


def print_summary_table(results: dict, output_dir: Path):
    """Print and save summary table."""
    print("\n" + "=" * 100)
    print("SUMMARY TABLE: Flow Metrics with Bootstrap 95% CIs")
    print("=" * 100)

    lines = []
    lines.append("=" * 100)
    lines.append("SUMMARY TABLE: Flow Metrics with Bootstrap 95% CIs")
    lines.append("=" * 100)

    for method in results.get("methods", []):
        if method not in results:
            continue

        lines.append(f"\n--- {method.upper()} ---")
        print(f"\n--- {method.upper()} ---")

        bootstrap = results[method].get("bootstrap_metrics", {})
        if not bootstrap:
            continue

        groups = list(bootstrap.keys())
        metrics = list(bootstrap[groups[0]].keys()) if groups else []

        header = f"{'Metric':<25}" + "".join([f"{g:>30}" for g in groups])
        lines.append(header)
        print(header)

        lines.append("-" * (25 + 30 * len(groups)))
        print("-" * (25 + 30 * len(groups)))

        for metric in metrics:
            row = f"{metric.replace('_', ' ').title():<25}"
            for g in groups:
                m = bootstrap[g][metric]
                row += f"{m['mean']:>10.3f} [{m['ci_low']:.3f}, {m['ci_high']:.3f}]"
            lines.append(row)
            print(row)

    # Effect sizes
    effect_sizes = results.get("effect_sizes", {})
    if effect_sizes:
        lines.append("\n" + "=" * 100)
        lines.append("EFFECT SIZES (Cohen's d) with 95% CIs")
        lines.append("=" * 100)
        print("\n" + "=" * 100)
        print("EFFECT SIZES (Cohen's d) with 95% CIs")
        print("=" * 100)

        for comparison, metrics in effect_sizes.items():
            lines.append(f"\n{comparison}:")
            print(f"\n{comparison}:")
            for metric, data in metrics.items():
                line = f"  {metric.replace('_', ' ').title():<25}: d = {data['mean']:>6.3f} [{data['ci_low']:.3f}, {data['ci_high']:.3f}]"
                # Interpret
                d = abs(data['mean'])
                if d >= 0.8:
                    interp = "(large)"
                elif d >= 0.5:
                    interp = "(medium)"
                elif d >= 0.2:
                    interp = "(small)"
                else:
                    interp = "(negligible)"
                line += f" {interp}"
                lines.append(line)
                print(line)

    # Save to file
    table_path = output_dir / "summary_table.txt"
    with open(table_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nSummary table saved to: {table_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Full-dataset statistical analysis for systems neuroscience paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n-subjects", type=int, default=None,
                        help="Max subjects per group (default: all)")
    parser.add_argument("--n-chunks", type=int, default=10,
                        help="Chunks per subject (default: 10)")
    parser.add_argument("--n-bootstrap", type=int, default=500,
                        help="Bootstrap iterations (default: 500)")
    parser.add_argument("--conditions", type=str, nargs="+", default=["HID", "MCI", "AD"],
                        help="Conditions to include (default: HID MCI AD)")
    parser.add_argument("--embedding", type=str, default="fast",
                        choices=["pca", "tpca", "diffusion", "delay", "umap", "fast", "all"],
                        help="Embedding method: 'fast' (pca+tpca+delay, default), 'all' (includes slow diffusion/umap), or single method")
    parser.add_argument("--tau", type=int, default=5,
                        help="Time lag for tPCA and delay embedding (default: 5)")
    parser.add_argument("--delay-dim", type=int, default=3,
                        help="Delay embedding dimension (default: 3)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test mode (100 bootstrap, 5 subjects)")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't display plots interactively")
    args = parser.parse_args()

    # Set matplotlib backend to non-interactive if --no-show
    if args.no_show:
        import matplotlib
        matplotlib.use('Agg')
        print("Non-interactive mode: plots will be saved but not displayed")

    # Quick mode overrides
    if args.quick:
        args.n_bootstrap = 100
        args.n_subjects = 5
        print("QUICK MODE: 100 bootstrap iterations, 5 subjects per group")

    # Create timestamped output directory
    base_output_dir = ensure_output_dir()
    output_dir = create_timestamped_output_dir(base_output_dir, "full_dataset_analysis")
    print(f"Output directory: {output_dir}")

    # Get subjects
    fif_files = get_fif_files(args.conditions)
    groups = get_subjects_by_group(fif_files)

    print(f"\nDataset overview:")
    for key in ["hc", "mci", "ad"]:
        if groups.get(key):
            print(f"  {key.upper()}: {len(groups[key])} subjects")

    # Load model
    print("\nLoading model...")
    model_info = load_model_from_checkpoint(CHECKPOINT_PATH, DEVICE)

    # Get n_channels
    all_subjects_list = []
    for key in ["hc", "mci", "ad"]:
        all_subjects_list.extend(groups.get(key, []))

    first_data = load_and_preprocess_fif(
        all_subjects_list[0][0], FILTER_LOW, FILTER_HIGH, CHUNK_DURATION,
        include_amplitude=model_info["include_amplitude"],
        verbose=False,
    )
    model = create_model(first_data["n_channels"], model_info, DEVICE)

    # Load all subject data
    subject_data = load_all_subjects(
        model, model_info, groups,
        args.n_subjects, args.n_chunks
    )

    # Determine methods
    if args.embedding == "fast":
        methods = ["pca", "tpca", "delay"]  # Fast methods only (no diffusion/umap)
    elif args.embedding == "all":
        methods = ["pca", "tpca", "delay", "diffusion"]
        if HAS_UMAP:
            methods.append("umap")
    else:
        methods = [args.embedding]

    print(f"Using embedding methods: {methods}")

    # Save parameters for reproducibility
    save_parameters(output_dir, {
        "n_subjects": args.n_subjects,
        "n_chunks": args.n_chunks,
        "n_bootstrap": args.n_bootstrap,
        "conditions": args.conditions,
        "embedding": args.embedding,
        "methods": methods,
        "tau": args.tau,
        "delay_dim": args.delay_dim,
        "quick_mode": args.quick,
        "filter_low": FILTER_LOW,
        "filter_high": FILTER_HIGH,
        "chunk_duration": CHUNK_DURATION,
        "sfreq": SFREQ,
        "checkpoint_path": CHECKPOINT_PATH,
        "data_dir": DATA_DIR,
        "device": DEVICE,
    })

    # Run analysis
    results = run_full_analysis(
        subject_data, output_dir, methods,
        args.n_bootstrap, args.tau, args.delay_dim,
        not args.no_show
    )

    # Print summary
    print_summary_table(results, output_dir)

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"All outputs saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
