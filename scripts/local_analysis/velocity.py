"""
Velocity Estimation Utilities for Latent Trajectory Analysis

This module provides configurable velocity estimation methods to address
reviewer concerns about noise sensitivity when using Δt = 1 sample.

Methods:
    - finite_diff: Finite differences with configurable Δt (default: 1)
    - savgol: Savitzky-Golay derivative for noise-robust estimation

Usage:
    from local_analysis.velocity import compute_velocity, compute_speed, VelocityConfig

    # Default behavior (Δt=1, backwards compatible)
    speed = compute_speed(latent)

    # With larger Δt for noise robustness
    speed = compute_speed(latent, delta_t=3)

    # With Savitzky-Golay derivative
    speed = compute_speed(latent, method="savgol", savgol_window=5)

    # Using config object
    config = VelocityConfig(method="savgol", savgol_window=7, savgol_poly=2)
    speed = compute_speed(latent, config=config)
"""

from dataclasses import dataclass
from typing import Literal
import numpy as np

try:
    from scipy.signal import savgol_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class VelocityConfig:
    """Configuration for velocity estimation.

    Attributes:
        method: 'finite_diff' or 'savgol'
        delta_t: Time step for finite differences (samples). Default=1.
        savgol_window: Window length for Savitzky-Golay filter (must be odd). Default=5.
        savgol_poly: Polynomial order for Savitzky-Golay. Default=2.
        dt_seconds: Physical time step in seconds (for normalization). Default=1.0.
    """
    method: Literal["finite_diff", "savgol"] = "finite_diff"
    delta_t: int = 1
    savgol_window: int = 5
    savgol_poly: int = 2
    dt_seconds: float = 1.0  # For physical units; 1.0 = dimensionless

    def __post_init__(self):
        """Validate configuration."""
        if self.method not in ("finite_diff", "savgol"):
            raise ValueError(f"method must be 'finite_diff' or 'savgol', got {self.method}")
        if self.delta_t < 1:
            raise ValueError(f"delta_t must be >= 1, got {self.delta_t}")
        if self.savgol_window < 3:
            raise ValueError(f"savgol_window must be >= 3, got {self.savgol_window}")
        if self.savgol_window % 2 == 0:
            raise ValueError(f"savgol_window must be odd, got {self.savgol_window}")
        if self.savgol_poly >= self.savgol_window:
            raise ValueError(
                f"savgol_poly ({self.savgol_poly}) must be < savgol_window ({self.savgol_window})"
            )


# Default config (backwards compatible with Δt=1)
DEFAULT_CONFIG = VelocityConfig()


def compute_velocity(
    trajectory: np.ndarray,
    method: Literal["finite_diff", "savgol"] = "finite_diff",
    delta_t: int = 1,
    savgol_window: int = 5,
    savgol_poly: int = 2,
    dt_seconds: float = 1.0,
    config: VelocityConfig | None = None,
) -> np.ndarray:
    """
    Compute velocity vectors along a trajectory.

    Args:
        trajectory: (T, D) array of trajectory points
        method: 'finite_diff' for finite differences, 'savgol' for Savitzky-Golay
        delta_t: Step size for finite differences (default=1)
        savgol_window: Window for Savitzky-Golay (must be odd, default=5)
        savgol_poly: Polynomial order for Savitzky-Golay (default=2)
        dt_seconds: Time step in seconds for normalization (default=1.0)
        config: Optional VelocityConfig to use instead of individual parameters

    Returns:
        velocity: (T', D) array of velocity vectors
            - For finite_diff with delta_t=k: T' = T - k
            - For savgol: T' = T (same length, boundary effects at edges)

    Notes:
        - Finite differences: v(t) = (x(t + Δt) - x(t)) / (Δt * dt_seconds)
        - Savitzky-Golay: Polynomial fit derivative, more robust to noise
    """
    if config is not None:
        method = config.method
        delta_t = config.delta_t
        savgol_window = config.savgol_window
        savgol_poly = config.savgol_poly
        dt_seconds = config.dt_seconds

    trajectory = np.asarray(trajectory)
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)

    T, D = trajectory.shape

    if method == "finite_diff":
        # Forward difference: v(t) = (x(t+delta_t) - x(t)) / delta_t
        if T <= delta_t:
            raise ValueError(f"Trajectory length {T} must be > delta_t {delta_t}")

        velocity = (trajectory[delta_t:] - trajectory[:-delta_t]) / (delta_t * dt_seconds)
        return velocity

    elif method == "savgol":
        if not HAS_SCIPY:
            raise ImportError("scipy is required for Savitzky-Golay method")

        # Check window size vs trajectory length
        if savgol_window > T:
            # Fall back to shorter window if trajectory too short
            savgol_window = T if T % 2 == 1 else T - 1
            if savgol_window < 3:
                # Too short, fall back to finite diff
                return compute_velocity(
                    trajectory, method="finite_diff", delta_t=1, dt_seconds=dt_seconds
                )

        # Apply Savitzky-Golay derivative to each dimension
        # deriv=1 computes first derivative
        velocity = np.zeros((T, D))
        for d in range(D):
            velocity[:, d] = savgol_filter(
                trajectory[:, d],
                window_length=savgol_window,
                polyorder=savgol_poly,
                deriv=1,  # First derivative
                delta=dt_seconds,  # Time step for derivative scaling
            )
        return velocity

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_speed(
    trajectory: np.ndarray,
    method: Literal["finite_diff", "savgol"] = "finite_diff",
    delta_t: int = 1,
    savgol_window: int = 5,
    savgol_poly: int = 2,
    dt_seconds: float = 1.0,
    config: VelocityConfig | None = None,
) -> np.ndarray:
    """
    Compute speed (magnitude of velocity) along a trajectory.

    Args:
        trajectory: (T, D) array of trajectory points
        method: 'finite_diff' or 'savgol'
        delta_t: Step size for finite differences
        savgol_window: Window for Savitzky-Golay
        savgol_poly: Polynomial order for Savitzky-Golay
        dt_seconds: Time step in seconds
        config: Optional VelocityConfig

    Returns:
        speed: (T',) array of speeds (L2 norm of velocity)
    """
    velocity = compute_velocity(
        trajectory,
        method=method,
        delta_t=delta_t,
        savgol_window=savgol_window,
        savgol_poly=savgol_poly,
        dt_seconds=dt_seconds,
        config=config,
    )
    return np.linalg.norm(velocity, axis=1)


def compute_displacement(
    trajectory: np.ndarray,
    delta_t: int = 1,
) -> np.ndarray:
    """
    Compute displacement vectors (for flow field computation).

    This is specifically for flow field estimation where we want
    unnormalized displacements (not velocity).

    Args:
        trajectory: (T, D) array of trajectory points
        delta_t: Step size for differences

    Returns:
        displacement: (T - delta_t, D) array of displacement vectors
    """
    trajectory = np.asarray(trajectory)
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)

    T = trajectory.shape[0]
    if T <= delta_t:
        raise ValueError(f"Trajectory length {T} must be > delta_t {delta_t}")

    return trajectory[delta_t:] - trajectory[:-delta_t]


# =============================================================================
# BACKWARDS COMPATIBLE WRAPPERS
# =============================================================================

def compute_instantaneous_speed(latent: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    Compute instantaneous speed in latent space.

    DEPRECATED: Use compute_speed() instead. This function is kept for
    backwards compatibility with existing code.

    Args:
        latent: (T, D) trajectory
        dt: time step (default 1 sample) - NOTE: this is dt_seconds, not delta_t

    Returns:
        speed: (T-1,) array of speeds
    """
    return compute_speed(latent, method="finite_diff", delta_t=1, dt_seconds=dt)


def compute_latent_speed(latent: np.ndarray) -> np.ndarray:
    """
    Compute ||h(t+1) - h(t)||.

    DEPRECATED: Use compute_speed() instead. This function is kept for
    backwards compatibility with existing code.

    Args:
        latent: (T, D) trajectory

    Returns:
        speed: (T-1,) array of speeds (unnormalized)
    """
    return compute_speed(latent, method="finite_diff", delta_t=1, dt_seconds=1.0)


# =============================================================================
# ROBUSTNESS TESTING UTILITIES
# =============================================================================

def compute_speed_robustness(
    trajectory: np.ndarray,
    delta_t_values: list[int] = [1, 2, 3, 5],
    methods: list[str] = ["finite_diff", "savgol"],
    savgol_windows: list[int] = [5, 7, 9],
) -> dict:
    """
    Compute speed statistics across multiple configurations for robustness testing.

    Args:
        trajectory: (T, D) array
        delta_t_values: List of delta_t values to test
        methods: List of methods to test
        savgol_windows: List of Savitzky-Golay window sizes

    Returns:
        Dictionary with results for each configuration:
        {
            'finite_diff_dt1': {'mean': ..., 'std': ..., 'median': ...},
            'finite_diff_dt2': {...},
            'savgol_w5': {...},
            ...
        }
    """
    results = {}

    # Test finite differences with various delta_t
    for dt in delta_t_values:
        try:
            speed = compute_speed(trajectory, method="finite_diff", delta_t=dt)
            results[f"finite_diff_dt{dt}"] = {
                "mean": float(np.mean(speed)),
                "std": float(np.std(speed)),
                "median": float(np.median(speed)),
                "n_samples": len(speed),
            }
        except ValueError:
            pass  # Trajectory too short for this delta_t

    # Test Savitzky-Golay with various windows
    if "savgol" in methods and HAS_SCIPY:
        for window in savgol_windows:
            try:
                speed = compute_speed(trajectory, method="savgol", savgol_window=window)
                results[f"savgol_w{window}"] = {
                    "mean": float(np.mean(speed)),
                    "std": float(np.std(speed)),
                    "median": float(np.median(speed)),
                    "n_samples": len(speed),
                }
            except (ValueError, ImportError):
                pass

    return results


def compare_speed_configurations(
    trajectories: list[np.ndarray],
    group_labels: list[int],
    configurations: list[dict],
) -> dict:
    """
    Compare group differences across velocity configurations.

    This is useful for testing whether group effects are robust to
    different velocity estimation methods.

    Args:
        trajectories: List of (T_i, D) trajectories
        group_labels: Group label for each trajectory (e.g., 0=HC, 1=MCI)
        configurations: List of config dicts, e.g.:
            [
                {"method": "finite_diff", "delta_t": 1},
                {"method": "finite_diff", "delta_t": 3},
                {"method": "savgol", "savgol_window": 5},
            ]

    Returns:
        Dictionary with group mean speeds for each configuration
    """
    results = {}

    for config_dict in configurations:
        config_name = _config_to_name(config_dict)
        config = VelocityConfig(**config_dict)

        # Compute speeds for all trajectories
        speeds_by_group = {}
        for traj, label in zip(trajectories, group_labels):
            try:
                speed = compute_speed(traj, config=config)
                mean_speed = float(np.mean(speed))

                if label not in speeds_by_group:
                    speeds_by_group[label] = []
                speeds_by_group[label].append(mean_speed)
            except (ValueError, ImportError):
                pass

        # Compute group statistics
        group_stats = {}
        for label, speeds in speeds_by_group.items():
            group_stats[label] = {
                "mean": float(np.mean(speeds)),
                "std": float(np.std(speeds)),
                "n": len(speeds),
            }

        results[config_name] = group_stats

    return results


def _config_to_name(config_dict: dict) -> str:
    """Convert config dict to readable name."""
    method = config_dict.get("method", "finite_diff")
    if method == "finite_diff":
        dt = config_dict.get("delta_t", 1)
        return f"finite_diff_dt{dt}"
    elif method == "savgol":
        window = config_dict.get("savgol_window", 5)
        return f"savgol_w{window}"
    return str(config_dict)


# =============================================================================
# INTRINSIC LATENT METRICS (Coordinate-Invariant)
# =============================================================================

def compute_whitening_transform(trajectories: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute whitening transform from pooled trajectories.

    This computes the global covariance and its inverse square root,
    which can be used to compute Mahalanobis distances that are
    invariant to linear rescaling of latent dimensions.

    Args:
        trajectories: List of (T_i, D) trajectory arrays

    Returns:
        mean: (D,) mean vector
        whitening_matrix: (D, D) matrix W such that W @ (h - mean) is whitened
    """
    # Pool all points
    pooled = np.vstack(trajectories)
    mean = pooled.mean(axis=0)

    # Compute covariance
    centered = pooled - mean
    cov = (centered.T @ centered) / (len(pooled) - 1)

    # Compute whitening matrix: W = Σ^{-1/2}
    # Use eigendecomposition for numerical stability
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Clip small eigenvalues to avoid numerical issues
    eigenvalues = np.maximum(eigenvalues, 1e-10)

    # W = V @ diag(1/sqrt(λ)) @ V.T
    whitening_matrix = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T

    return mean, whitening_matrix


def compute_whitened_speed(
    trajectory: np.ndarray,
    whitening_matrix: np.ndarray,
    method: Literal["finite_diff", "savgol"] = "savgol",
    delta_t: int = 1,
    savgol_window: int = 5,
    savgol_poly: int = 2,
    config: VelocityConfig | None = None,
) -> np.ndarray:
    """
    Compute speed using Mahalanobis metric (whitened coordinates).

    This makes speed invariant to linear rescaling of latent dimensions,
    addressing the "arbitrary units" critique.

    Args:
        trajectory: (T, D) array of trajectory points in original latent space
        whitening_matrix: (D, D) whitening transform from compute_whitening_transform()
        method: Velocity estimation method
        delta_t: Step size for finite differences
        savgol_window: Window for Savitzky-Golay
        savgol_poly: Polynomial order for Savitzky-Golay
        config: Optional VelocityConfig

    Returns:
        speed: (T',) array of whitened speeds
    """
    # Compute velocity in original space
    velocity = compute_velocity(
        trajectory,
        method=method,
        delta_t=delta_t,
        savgol_window=savgol_window,
        savgol_poly=savgol_poly,
        config=config,
    )

    # Transform velocity to whitened space and compute magnitude
    # ||W @ v|| = sqrt(v.T @ W.T @ W @ v) = sqrt(v.T @ Σ^{-1} @ v) = Mahalanobis
    whitened_velocity = velocity @ whitening_matrix.T
    return np.linalg.norm(whitened_velocity, axis=1)


def compute_zscored_speed(
    trajectory: np.ndarray,
    dim_stds: np.ndarray,
    method: Literal["finite_diff", "savgol"] = "savgol",
    delta_t: int = 1,
    savgol_window: int = 5,
    savgol_poly: int = 2,
    config: VelocityConfig | None = None,
) -> np.ndarray:
    """
    Compute speed after z-scoring each latent dimension.

    Simpler alternative to full Mahalanobis: just normalize each dimension
    by its standard deviation. This removes the "arbitrary scale" critique
    while being more robust than full covariance inversion.

    Args:
        trajectory: (T, D) array of trajectory points
        dim_stds: (D,) standard deviation of each dimension (from pooled data)
        method: Velocity estimation method
        delta_t: Step size for finite differences
        savgol_window: Window for Savitzky-Golay
        savgol_poly: Polynomial order for Savitzky-Golay
        config: Optional VelocityConfig

    Returns:
        speed: (T',) array of z-scored speeds
    """
    # Compute velocity in original space
    velocity = compute_velocity(
        trajectory,
        method=method,
        delta_t=delta_t,
        savgol_window=savgol_window,
        savgol_poly=savgol_poly,
        config=config,
    )

    # Normalize each dimension by its std
    dim_stds = np.maximum(dim_stds, 1e-10)  # Avoid division by zero
    normalized_velocity = velocity / dim_stds

    return np.linalg.norm(normalized_velocity, axis=1)


@dataclass
class IntrinsicMetrics:
    """Intrinsic (coordinate-invariant) metrics computed in full latent h(t).

    These metrics do not depend on a specific 2D projection and are
    invariant to linear rescaling of latent dimensions when using
    whitened/z-scored distances.
    """
    mean_speed: float           # Mean speed (Euclidean in latent)
    mean_speed_whitened: float  # Mean speed (Mahalanobis / whitened)
    mean_speed_zscored: float   # Mean speed (z-scored dimensions)
    speed_std: float            # Speed variability
    speed_cv: float             # Coefficient of variation
    path_length: float          # Total path length
    displacement: float         # End-to-end displacement
    tortuosity: float           # path_length / displacement
    explored_variance: float    # trace(Cov(h)) = sum of per-dim variances
    latent_dim: int             # Dimensionality of latent space


def compute_intrinsic_metrics(
    trajectory: np.ndarray,
    whitening_matrix: np.ndarray | None = None,
    dim_stds: np.ndarray | None = None,
    velocity_config: VelocityConfig | None = None,
) -> IntrinsicMetrics:
    """
    Compute intrinsic metrics in full latent space h(t).

    These are the "coordinate-free" metrics that don't depend on
    a specific 2D projection. Speed is computed using:
    - Euclidean (raw)
    - Mahalanobis (whitened) - invariant to linear rescaling
    - Z-scored - simpler alternative

    Args:
        trajectory: (T, D) latent trajectory in full h(t) space
        whitening_matrix: (D, D) from compute_whitening_transform()
        dim_stds: (D,) per-dimension stds for z-scoring
        velocity_config: Optional velocity estimation config

    Returns:
        IntrinsicMetrics dataclass with all computed metrics
    """
    if velocity_config is None:
        velocity_config = VelocityConfig(method="savgol", savgol_window=5, savgol_poly=2)

    trajectory = np.asarray(trajectory)
    T, D = trajectory.shape

    # Euclidean speed
    speed_euclidean = compute_speed(trajectory, config=velocity_config)
    mean_speed = float(np.mean(speed_euclidean))
    speed_std = float(np.std(speed_euclidean))
    speed_cv = speed_std / mean_speed if mean_speed > 0 else 0.0

    # Whitened speed (Mahalanobis)
    if whitening_matrix is not None:
        speed_whitened = compute_whitened_speed(
            trajectory, whitening_matrix, config=velocity_config
        )
        mean_speed_whitened = float(np.mean(speed_whitened))
    else:
        mean_speed_whitened = mean_speed  # Fallback to Euclidean

    # Z-scored speed
    if dim_stds is not None:
        speed_zscored = compute_zscored_speed(
            trajectory, dim_stds, config=velocity_config
        )
        mean_speed_zscored = float(np.mean(speed_zscored))
    else:
        mean_speed_zscored = mean_speed  # Fallback to Euclidean

    # Path geometry
    path_length = float(np.sum(speed_euclidean))
    displacement = float(np.linalg.norm(trajectory[-1] - trajectory[0]))
    tortuosity = path_length / displacement if displacement > 0 else float('inf')

    # Explored variance: trace of covariance = sum of per-dim variances
    explored_variance = float(np.var(trajectory, axis=0).sum())

    return IntrinsicMetrics(
        mean_speed=mean_speed,
        mean_speed_whitened=mean_speed_whitened,
        mean_speed_zscored=mean_speed_zscored,
        speed_std=speed_std,
        speed_cv=speed_cv,
        path_length=path_length,
        displacement=displacement,
        tortuosity=tortuosity,
        explored_variance=explored_variance,
        latent_dim=D,
    )


def compute_pooled_normalization(trajectories: list[np.ndarray]) -> dict:
    """
    Compute normalization parameters from pooled trajectories.

    Returns both whitening matrix and per-dimension stds for
    different normalization strategies.

    Args:
        trajectories: List of (T_i, D) trajectory arrays

    Returns:
        Dict with:
            - 'mean': (D,) mean vector
            - 'whitening_matrix': (D, D) for Mahalanobis
            - 'dim_stds': (D,) per-dimension stds for z-scoring
    """
    pooled = np.vstack(trajectories)
    mean = pooled.mean(axis=0)
    dim_stds = pooled.std(axis=0)

    # Compute whitening matrix
    _, whitening_matrix = compute_whitening_transform(trajectories)

    return {
        'mean': mean,
        'whitening_matrix': whitening_matrix,
        'dim_stds': dim_stds,
    }
