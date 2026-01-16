"""Recurrence Quantification Analysis (RQA) feature computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numba import jit


@dataclass
class RQAFeatures:
    """Container for RQA features."""

    RR: float      # Recurrence Rate
    DET: float     # Determinism
    L: float       # Average diagonal line length
    Lmax: float    # Maximum diagonal line length
    DIV: float     # Divergence (1/Lmax)
    ENTR: float    # Entropy of diagonal line distribution
    LAM: float     # Laminarity
    TT: float      # Trapping Time
    Vmax: float    # Maximum vertical line length
    Ventr: float   # Entropy of vertical line distribution

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "RR": self.RR,
            "DET": self.DET,
            "L": self.L,
            "Lmax": self.Lmax,
            "DIV": self.DIV,
            "ENTR": self.ENTR,
            "LAM": self.LAM,
            "TT": self.TT,
            "Vmax": self.Vmax,
            "Ventr": self.Ventr,
        }

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array(list(self.to_dict().values()))

    @classmethod
    def feature_names(cls) -> list[str]:
        """Get feature names in order."""
        return ["RR", "DET", "L", "Lmax", "DIV", "ENTR", "LAM", "TT", "Vmax", "Ventr"]


@jit(nopython=True)
def _compute_diagonal_lines(R: np.ndarray, min_length: int = 2) -> np.ndarray:
    """
    Compute histogram of diagonal line lengths.

    Args:
        R: Binary recurrence matrix (N, N)
        min_length: Minimum line length to count

    Returns:
        Histogram of line lengths (index = length)
    """
    N = R.shape[0]
    max_length = N
    histogram = np.zeros(max_length + 1, dtype=np.int64)

    # Scan all diagonals
    for offset in range(-(N - min_length), N - min_length + 1):
        if offset >= 0:
            diag = np.diag(R, offset)
        else:
            diag = np.diag(R, offset)

        # Find runs of 1s
        current_length = 0
        for val in diag:
            if val > 0:
                current_length += 1
            else:
                if current_length >= min_length:
                    histogram[current_length] += 1
                current_length = 0

        # Handle run at end
        if current_length >= min_length:
            histogram[current_length] += 1

    return histogram


@jit(nopython=True)
def _compute_vertical_lines(R: np.ndarray, min_length: int = 2) -> np.ndarray:
    """
    Compute histogram of vertical line lengths.

    Args:
        R: Binary recurrence matrix (N, N)
        min_length: Minimum line length to count

    Returns:
        Histogram of line lengths (index = length)
    """
    N = R.shape[0]
    max_length = N
    histogram = np.zeros(max_length + 1, dtype=np.int64)

    # Scan all columns
    for col in range(N):
        current_length = 0
        for row in range(N):
            if R[row, col] > 0:
                current_length += 1
            else:
                if current_length >= min_length:
                    histogram[current_length] += 1
                current_length = 0

        # Handle run at end
        if current_length >= min_length:
            histogram[current_length] += 1

    return histogram


def compute_rqa_features(
    recurrence_matrix: np.ndarray,
    min_diagonal_length: int = 2,
    min_vertical_length: int = 2,
) -> RQAFeatures:
    """
    Compute RQA features from a binary recurrence matrix.

    Args:
        recurrence_matrix: Binary recurrence matrix (N, N)
        min_diagonal_length: Minimum diagonal line length
        min_vertical_length: Minimum vertical line length

    Returns:
        RQAFeatures object with computed metrics
    """
    R = recurrence_matrix.astype(np.float64)
    N = R.shape[0]

    # Basic recurrence rate (excluding main diagonal)
    mask = ~np.eye(N, dtype=bool)
    total_points = mask.sum()
    recurrence_points = R[mask].sum()
    RR = recurrence_points / total_points if total_points > 0 else 0.0

    # Diagonal line analysis
    diag_hist = _compute_diagonal_lines(R, min_diagonal_length)
    lengths = np.arange(len(diag_hist))

    total_diag_points = (diag_hist * lengths).sum()
    num_lines = diag_hist.sum()

    if num_lines > 0:
        DET = total_diag_points / recurrence_points if recurrence_points > 0 else 0.0
        L = total_diag_points / num_lines
        Lmax = np.max(np.where(diag_hist > 0)[0]) if np.any(diag_hist > 0) else 0

        # Entropy of diagonal distribution
        p = diag_hist[diag_hist > 0] / num_lines
        ENTR = -np.sum(p * np.log(p))
    else:
        DET = 0.0
        L = 0.0
        Lmax = 0
        ENTR = 0.0

    DIV = 1.0 / Lmax if Lmax > 0 else 0.0

    # Vertical line analysis
    vert_hist = _compute_vertical_lines(R, min_vertical_length)
    lengths_v = np.arange(len(vert_hist))

    total_vert_points = (vert_hist * lengths_v).sum()
    num_vert_lines = vert_hist.sum()

    if num_vert_lines > 0:
        LAM = total_vert_points / recurrence_points if recurrence_points > 0 else 0.0
        TT = total_vert_points / num_vert_lines
        Vmax = np.max(np.where(vert_hist > 0)[0]) if np.any(vert_hist > 0) else 0

        # Entropy of vertical distribution
        p_v = vert_hist[vert_hist > 0] / num_vert_lines
        Ventr = -np.sum(p_v * np.log(p_v))
    else:
        LAM = 0.0
        TT = 0.0
        Vmax = 0
        Ventr = 0.0

    return RQAFeatures(
        RR=RR,
        DET=DET,
        L=L,
        Lmax=float(Lmax),
        DIV=DIV,
        ENTR=ENTR,
        LAM=LAM,
        TT=TT,
        Vmax=float(Vmax),
        Ventr=Ventr,
    )


def apply_theiler_window(R: np.ndarray, theiler_window: int) -> np.ndarray:
    """
    Apply Theiler window to recurrence matrix (exclude near-diagonal recurrences).

    The Theiler window removes "trivial" recurrences caused by temporal autocorrelation.
    Points within |i - j| < theiler_window are set to 0.

    This is CRITICAL for meaningful RQA analysis because:
    - Adjacent time points in smooth signals are always similar
    - Without Theiler window, high DET/LAM may just reflect signal smoothness
    - With Theiler window, RQA measures longer-timescale regime returns

    Args:
        R: Binary recurrence matrix (N, N)
        theiler_window: Number of time steps to exclude around diagonal

    Returns:
        Recurrence matrix with near-diagonal band zeroed out
    """
    N = R.shape[0]
    R_theiler = R.copy()

    # Zero out band around diagonal
    for i in range(N):
        for j in range(max(0, i - theiler_window + 1), min(N, i + theiler_window)):
            R_theiler[i, j] = 0

    return R_theiler


def compute_rqa_from_distance_matrix(
    distance_matrix: np.ndarray,
    target_rr: float = 0.02,
    min_diagonal_length: int = 2,
    min_vertical_length: int = 2,
    theiler_window: int = 0,
) -> tuple[RQAFeatures, float]:
    """
    Compute RQA features from a distance matrix using RR-controlled thresholding.

    Args:
        distance_matrix: Distance matrix (angular or Euclidean)
        target_rr: Target recurrence rate (e.g., 0.02 for 2%)
        min_diagonal_length: Minimum diagonal line length
        min_vertical_length: Minimum vertical line length
        theiler_window: Exclude |i-j| < theiler_window from recurrence.
                       Recommended: 25-100 samples (0.1-0.4s at 250Hz).
                       Set to 0 to disable (default, for backwards compatibility).

    Returns:
        Tuple of (RQAFeatures, epsilon threshold used)

    Note on Theiler window:
        High DET/LAM at low RR (e.g., DET~0.85 at RR=2%) without Theiler window
        likely reflects trivial temporal autocorrelation, not meaningful dynamics.
        Apply theiler_window >= 25 to verify structure is real.
    """
    N = distance_matrix.shape[0]

    # Create mask excluding diagonal AND Theiler window for threshold computation
    if theiler_window > 0:
        mask = np.ones((N, N), dtype=bool)
        for i in range(N):
            for j in range(max(0, i - theiler_window + 1), min(N, i + theiler_window)):
                mask[i, j] = False
    else:
        mask = ~np.eye(N, dtype=bool)

    off_diag = distance_matrix[mask]

    if len(off_diag) == 0:
        # Theiler window too large for matrix size
        return RQAFeatures(0, 0, 0, 0, 0, 0, 0, 0, 0, 0), 0.0

    # Find threshold at target RR percentile
    epsilon = np.percentile(off_diag, target_rr * 100)

    # Create binary recurrence matrix
    R = (distance_matrix <= epsilon).astype(np.float64)

    # Apply Theiler window if specified
    if theiler_window > 0:
        R = apply_theiler_window(R, theiler_window)

    # Compute features
    features = compute_rqa_features(R, min_diagonal_length, min_vertical_length)

    return features, epsilon


def compute_rqa_time_shuffle_null(
    latent_trajectory: np.ndarray,
    target_rr: float = 0.02,
    theiler_window: int = 0,
    n_shuffles: int = 100,
    random_state: int = 42,
) -> dict:
    """
    Compute time-shuffle null distribution for RQA features.

    This tests whether RQA features capture meaningful temporal structure.
    If RQA features barely change after time-shuffling, they're measuring
    something trivial (like variance) rather than dynamics.

    Args:
        latent_trajectory: Latent states over time (T, hidden_size)
        target_rr: Target recurrence rate
        theiler_window: Theiler window for RQA computation
        n_shuffles: Number of shuffle iterations
        random_state: Random seed for reproducibility

    Returns:
        Dict with:
        - real_features: RQAFeatures from original data
        - null_mean: Mean of shuffled features
        - null_std: Std of shuffled features
        - z_scores: Z-score of real vs null for each feature
        - interpretation: Whether structure is meaningful
    """
    rng = np.random.RandomState(random_state)

    # Compute angular distance matrix
    norms = np.linalg.norm(latent_trajectory, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = latent_trajectory / norms
    cos_sim = np.dot(normalized, normalized.T)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    distance_matrix = np.arccos(cos_sim)

    # Real RQA
    real_features, _ = compute_rqa_from_distance_matrix(
        distance_matrix, target_rr=target_rr, theiler_window=theiler_window
    )

    # Shuffle null
    null_features = []
    T = len(latent_trajectory)

    for _ in range(n_shuffles):
        # Shuffle time order
        shuffle_idx = rng.permutation(T)
        shuffled = latent_trajectory[shuffle_idx]

        # Recompute distance matrix
        norms_s = np.linalg.norm(shuffled, axis=1, keepdims=True)
        norms_s = np.maximum(norms_s, 1e-8)
        normalized_s = shuffled / norms_s
        cos_sim_s = np.dot(normalized_s, normalized_s.T)
        cos_sim_s = np.clip(cos_sim_s, -1.0, 1.0)
        dist_s = np.arccos(cos_sim_s)

        # RQA on shuffled
        feat_s, _ = compute_rqa_from_distance_matrix(
            dist_s, target_rr=target_rr, theiler_window=theiler_window
        )
        null_features.append(feat_s.to_array())

    null_array = np.array(null_features)
    null_mean = null_array.mean(axis=0)
    null_std = null_array.std(axis=0) + 1e-8

    real_array = real_features.to_array()
    z_scores = (real_array - null_mean) / null_std

    # Interpretation
    # DET, LAM, ENTR are the key features to check
    det_idx = RQAFeatures.feature_names().index("DET")
    lam_idx = RQAFeatures.feature_names().index("LAM")
    entr_idx = RQAFeatures.feature_names().index("ENTR")

    key_z_scores = [abs(z_scores[det_idx]), abs(z_scores[lam_idx]), abs(z_scores[entr_idx])]
    avg_key_z = np.mean(key_z_scores)

    if avg_key_z < 1.5:
        interpretation = "WEAK: RQA features not significantly different from time-shuffled null. Structure may be trivial."
    elif avg_key_z < 3.0:
        interpretation = "MODERATE: Some temporal structure detected. Interpret with caution."
    else:
        interpretation = "STRONG: RQA features significantly different from null. Meaningful temporal dynamics present."

    return {
        "real_features": real_features,
        "null_mean": dict(zip(RQAFeatures.feature_names(), null_mean)),
        "null_std": dict(zip(RQAFeatures.feature_names(), null_std)),
        "z_scores": dict(zip(RQAFeatures.feature_names(), z_scores)),
        "avg_key_z_score": avg_key_z,
        "interpretation": interpretation,
    }
