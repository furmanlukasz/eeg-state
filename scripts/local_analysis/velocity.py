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
