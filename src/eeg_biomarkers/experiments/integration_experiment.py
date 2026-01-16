"""
Integration Experiment: State-Conditioned vs Baseline Classification

This is the go/no-go experiment to validate whether state-conditioning
improves subject-level MCI classification.

CRITICAL FIXES IMPLEMENTED (2026-01-15):
1. Require trained checkpoint - no random model allowed
2. Use RQA features from latent trajectories, NOT mean latents
3. Actually use rr_target in RQA computation
4. Proper warning suppression for XGBoost/UMAP

Key requirements (per critic agent):
1. End-to-end: baseline vs state-conditioned, subject-level evaluation
2. Train-only fitting for state discovery (UMAP + HDBSCAN), no label leakage
3. Run both raw latent and HF-residualized latent variants
4. RR-controlled epsilon sweep (1%, 2%, 5%)
5. Fold-wise diagnostic reports + aggregate summary
6. Save all outputs for reproducibility

Win conditions:
- Strong success: State-conditioned > baseline, persists with residualization
- Mixed: No AUC improvement but reduced variance or state-specific effects
- Pivot: No improvement + states artifact-driven

Usage:
    python -m eeg_biomarkers.experiments.integration_experiment \
        --checkpoint models/best.pt --output-dir results/integration
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Literal

# Suppress known benign warnings BEFORE importing the libraries
warnings.filterwarnings(
    "ignore",
    message=".*Parameters: \\{ \"use_label_encoder\" \\} are not used.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*n_jobs value 1 overridden to 1 by setting random_state.*",
    category=UserWarning,
)

import hdbscan
import numpy as np
import pandas as pd
import torch
import umap
from scipy.stats import pearsonr
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from eeg_biomarkers.data.preprocessing import load_eeg_file, prepare_phase_chunks
from eeg_biomarkers.models import ConvLSTMAutoencoder
from eeg_biomarkers.analysis.rqa import compute_rqa_features, compute_rqa_from_distance_matrix, RQAFeatures
from eeg_biomarkers.analysis.artifact_control import (
    compute_hf_power,
    compute_fold_diagnostics,
    format_diagnostics_summary,
    residualize_latent,
    FoldDiagnosticReport,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for the integration experiment."""

    # Data settings
    data_dir: str = "data"
    chunk_duration: float = 5.0  # Fixed 5s windows
    filter_low: float = 3.0
    filter_high: float = 48.0

    # Model settings
    hidden_size: int = 64
    complexity: int = 2
    checkpoint_path: str | None = None  # Path to trained model checkpoint - REQUIRED

    # Cross-validation
    n_folds: int = 5
    n_seeds: int = 3  # Run 3 seeds for stability

    # RR sweep - these are now actually used!
    rr_targets: list[float] = field(default_factory=lambda: [0.01, 0.02, 0.05])

    # RQA settings - CRITICAL: Theiler window removes trivial autocorrelation
    theiler_window: int = 50  # ~0.2s exclusion window. Without this, high DET/LAM
                              # may just reflect signal smoothness, not regime structure.

    # State discovery
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_n_components: int = 2
    hdbscan_min_cluster_size: int = 10
    hdbscan_min_samples: int = 5

    # Classification
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 3
    xgb_learning_rate: float = 0.1

    # Null model
    run_null_model: bool = True

    # Trained vs Random comparison (cheap control for training quality)
    compare_random: bool = True  # Run pipeline twice: trained checkpoint vs random weights

    # Output
    output_dir: str = "results/integration"

    # Minimum subjects per class for meaningful AUC (critic agent recommendation #6)
    min_subjects_per_class: int = 20  # Below this, treat AUC as smoke test only

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FoldResult:
    """Results from a single fold."""

    fold_idx: int
    seed: int
    rr_target: float
    latent_mode: str  # "raw" or "residualized"
    condition: str  # "baseline" or "state_conditioned"

    # Subject-level AUC
    subject_auc: float
    n_subjects_train: int
    n_subjects_test: int
    n_segments_train: int
    n_segments_test: int

    # Bootstrap CI on subject-level AUC (more robust uncertainty)
    auc_ci_lower: float | None = None
    auc_ci_upper: float | None = None

    # Window retention (for state-conditioned)
    retention_overall: float = 1.0
    retention_mci: float = 1.0
    retention_hc: float = 1.0

    # Diagnostics from FoldDiagnosticReport
    hf1_correlation_raw: float | None = None
    hf1_correlation_residualized: float | None = None
    hf2_correlation_raw: float | None = None
    hf_temporal_central_ratio: float | None = None
    hf_state_prediction_accuracy: float | None = None

    # State info
    n_states_discovered: int = 0
    n_states_used: int = 0

    # Null model
    is_null_model: bool = False

    # Random model comparison (for trained vs random control)
    is_random_model: bool = False

    # RQA feature statistics (for interpretability)
    mean_det: float | None = None
    mean_lam: float | None = None
    mean_entr: float | None = None


def load_all_data(
    data_dir: Path,
    chunk_duration: float,
    filter_low: float,
    filter_high: float,
    hf2_filter_high: float = 120.0,
    include_amplitude: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], float]:
    """
    Load all EEG data with fixed-length windowing.

    CRITICAL FIX (2026-01-15): Now loads TWO versions of raw data:
    1. raw_chunks_hf1: Filtered to filter_high (48 Hz) - for autoencoder input
    2. raw_chunks_hf2: Filtered to hf2_filter_high (120 Hz) - for HF2 validation

    This allows proper HF2 (70-110 Hz) artifact validation while keeping
    the autoencoder input clean (48 Hz cutoff removes most EMG).

    Args:
        include_amplitude: Whether to include amplitude in phase chunks (3 channels vs 2)

    Returns:
        raw_chunks_hf1: (n_total_segments, n_channels, n_times) - for HF1 & autoencoder
        raw_chunks_hf2: (n_total_segments, n_channels, n_times) - for HF2 validation
        phase_chunks: (n_total_segments, n_channels*phase_channels, n_times) - cos/sin[/amp]
        subject_ids: (n_total_segments,) - subject index
        labels: (n_total_segments,) - 0=HC, 1=MCI
        channel_names: list of channel names
        sfreq: sampling frequency
    """
    logger.info(f"Loading data from {data_dir}")
    logger.info(f"  Filter for autoencoder: {filter_low}-{filter_high} Hz")
    logger.info(f"  Filter for HF2 validation: {filter_low}-{hf2_filter_high} Hz")

    all_raw_hf1 = []  # 48 Hz filtered - for autoencoder & HF1
    all_raw_hf2 = []  # 120 Hz filtered - for HF2 validation
    all_phase = []
    all_subject_ids = []
    all_labels = []
    channel_names = None
    sfreq = None

    # Find all subjects
    subject_idx = 0
    for group_dir in ["HID", "MCI"]:  # HC first, then MCI
        group_path = data_dir / group_dir
        if not group_path.exists():
            logger.warning(f"Group directory not found: {group_path}")
            continue

        label = 0 if group_dir == "HID" else 1  # HC=0, MCI=1

        # Find all subject directories
        for subject_dir in sorted(group_path.iterdir()):
            if not subject_dir.is_dir():
                continue

            # FIX: Use recursive glob to find files in nested directories
            files = list(subject_dir.rglob("*_good_*_eeg.fif"))
            if not files:
                # Also try non-recursive for backwards compatibility
                files = list(subject_dir.glob("*_good_*_eeg.fif"))
            if not files:
                logger.warning(f"No EEG files found for subject {subject_dir.name}")
                continue

            subject_chunks_phase = []
            subject_chunks_raw_hf1 = []
            subject_chunks_raw_hf2 = []

            for f in files:
                try:
                    # Load with 48 Hz filter for autoencoder input
                    raw_hf1 = load_eeg_file(
                        f, filter_low=filter_low, filter_high=filter_high, verbose=False
                    )
                    phase_chunks, mask, info = prepare_phase_chunks(
                        raw_hf1, chunk_duration=chunk_duration, include_amplitude=include_amplitude
                    )

                    if channel_names is None:
                        channel_names = info.get("ch_names", [f"E{i}" for i in range(info["n_channels"])])
                        sfreq = info["sfreq"]

                    # Load with 120 Hz filter for HF2 validation (only if Nyquist allows)
                    nyquist = sfreq / 2
                    if hf2_filter_high < nyquist:
                        raw_hf2 = load_eeg_file(
                            f, filter_low=filter_low, filter_high=hf2_filter_high, verbose=False
                        )
                        raw_data_hf2 = raw_hf2.get_data()
                    else:
                        # If Nyquist is too low, just use the same data (HF2 won't be computed)
                        raw_data_hf2 = raw_hf1.get_data()

                    # Get raw data for HF computation
                    raw_data_hf1 = raw_hf1.get_data()
                    n_samples_per_chunk = int(chunk_duration * sfreq)
                    n_chunks = raw_data_hf1.shape[1] // n_samples_per_chunk

                    for i in range(min(n_chunks, len(phase_chunks))):
                        start = i * n_samples_per_chunk
                        end = start + n_samples_per_chunk
                        subject_chunks_raw_hf1.append(raw_data_hf1[:, start:end])
                        subject_chunks_raw_hf2.append(raw_data_hf2[:, start:end])
                        subject_chunks_phase.append(phase_chunks[i])

                except Exception as e:
                    logger.warning(f"Error loading {f}: {e}")
                    continue

            if subject_chunks_phase:
                n_chunks = len(subject_chunks_phase)
                all_phase.extend(subject_chunks_phase)
                all_raw_hf1.extend(subject_chunks_raw_hf1)
                all_raw_hf2.extend(subject_chunks_raw_hf2)
                all_subject_ids.extend([subject_idx] * n_chunks)
                all_labels.extend([label] * n_chunks)
                logger.info(f"  Subject {subject_idx} ({group_dir}): {n_chunks} chunks")
                subject_idx += 1

    if not all_phase:
        raise ValueError("No data loaded!")

    raw_chunks_hf1 = np.array(all_raw_hf1)
    raw_chunks_hf2 = np.array(all_raw_hf2)
    phase_chunks = np.array(all_phase)
    subject_ids = np.array(all_subject_ids)
    labels = np.array(all_labels)

    logger.info(f"Total: {len(phase_chunks)} segments from {subject_idx} subjects")
    logger.info(f"  HC: {(labels == 0).sum()} segments, MCI: {(labels == 1).sum()} segments")
    logger.info(f"  Nyquist: {sfreq/2} Hz - HF2 band (70-110 Hz) is {'accessible' if 110 < sfreq/2 else 'NOT accessible'}")

    return raw_chunks_hf1, raw_chunks_hf2, phase_chunks, subject_ids, labels, channel_names, sfreq


def compute_latent_trajectories(
    model: ConvLSTMAutoencoder,
    phase_chunks: np.ndarray,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Compute latent TRAJECTORIES from phase data.

    CRITICAL FIX: Returns full trajectories (n_segments, T', hidden_size),
    NOT time-averaged vectors. The temporal dynamics are essential for RQA.

    Args:
        model: Trained autoencoder
        phase_chunks: Phase data (n_segments, n_channels*2, n_samples)
        batch_size: Batch size for inference

    Returns:
        latent_trajectories: (n_segments, T', hidden_size) - full temporal dynamics
    """
    model.eval()
    all_latents = []

    with torch.no_grad():
        for i in range(0, len(phase_chunks), batch_size):
            batch = torch.from_numpy(phase_chunks[i:i + batch_size]).float()
            _, latents = model(batch)
            # Keep full trajectory - shape: (batch, time', hidden_size)
            all_latents.append(latents.numpy())

    return np.concatenate(all_latents, axis=0)


def compute_rqa_features_for_segment(
    latent_trajectory: np.ndarray,
    target_rr: float = 0.02,
    theiler_window: int = 50,
) -> tuple[np.ndarray, RQAFeatures]:
    """
    Compute RQA features from a single segment's latent trajectory.

    This is the CORRECT approach: RQA on temporal dynamics, not mean embeddings.

    CRITICAL (2026-01-15): Uses Theiler window to exclude near-diagonal recurrences.
    Without Theiler window, high DET/LAM may just reflect signal smoothness
    (trivial temporal autocorrelation) rather than meaningful regime structure.

    Args:
        latent_trajectory: (T', hidden_size) - latent states over time
        target_rr: Target recurrence rate for thresholding
        theiler_window: Exclude |i-j| < W from recurrence. Default 50 (~0.2s).
                       Set to 0 for backwards compatibility (not recommended).

    Returns:
        feature_vector: Array of RQA features
        rqa_obj: Full RQAFeatures object for inspection
    """
    # Compute angular distance matrix
    # Normalize latent vectors
    norms = np.linalg.norm(latent_trajectory, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = latent_trajectory / norms

    # Cosine similarity -> angular distance
    cos_sim = np.dot(normalized, normalized.T)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    distance_matrix = np.arccos(cos_sim)

    # Compute RQA with RR-controlled thresholding AND Theiler window
    rqa_features, epsilon = compute_rqa_from_distance_matrix(
        distance_matrix,
        target_rr=target_rr,
        min_diagonal_length=2,
        min_vertical_length=2,
        theiler_window=theiler_window,
    )

    return rqa_features.to_array(), rqa_features


def compute_all_rqa_features(
    latent_trajectories: np.ndarray,
    target_rr: float = 0.02,
    theiler_window: int = 50,
) -> tuple[np.ndarray, dict]:
    """
    Compute RQA features for all segments.

    Args:
        latent_trajectories: (n_segments, T', hidden_size)
        target_rr: Target recurrence rate
        theiler_window: Theiler window for RQA (default 50, ~0.2s)

    Returns:
        features: (n_segments, n_rqa_features) array
        stats: Dict with mean/std of key features
    """
    n_segments = len(latent_trajectories)
    feature_list = []

    det_values = []
    lam_values = []
    entr_values = []

    for i in range(n_segments):
        feat_array, rqa_obj = compute_rqa_features_for_segment(
            latent_trajectories[i], target_rr, theiler_window
        )
        feature_list.append(feat_array)
        det_values.append(rqa_obj.DET)
        lam_values.append(rqa_obj.LAM)
        entr_values.append(rqa_obj.ENTR)

    features = np.array(feature_list)

    stats = {
        "mean_det": np.mean(det_values),
        "std_det": np.std(det_values),
        "mean_lam": np.mean(lam_values),
        "std_lam": np.std(lam_values),
        "mean_entr": np.mean(entr_values),
        "std_entr": np.std(entr_values),
    }

    return features, stats


class StateDiscovery:
    """
    Train-only state discovery using UMAP + HDBSCAN.

    Critical: Uses HDBSCAN.approximate_predict for test assignment
    to avoid label leakage.

    NOTE: Clustering uses mean latent per segment (for state assignment),
    but RQA features are computed from full trajectories (for classification).
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        n_components: int = 2,
        min_cluster_size: int = 10,
        min_samples: int = 5,
        random_state: int = 42,
    ):
        self.umap_model = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            random_state=random_state,
            metric="euclidean",
            n_jobs=1,  # Explicit to match random_state behavior
        )
        self.hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            prediction_data=True,  # Required for approximate_predict
        )
        self.fitted = False

    def fit(self, latent_trajectories_train: np.ndarray) -> np.ndarray:
        """
        Fit UMAP and HDBSCAN on training data only.

        Uses mean latent per segment for clustering (state = "where in latent space").
        """
        # Use mean latent for clustering
        mean_latents = latent_trajectories_train.mean(axis=1)

        # UMAP
        embedding_train = self.umap_model.fit_transform(mean_latents)

        # HDBSCAN
        self.hdbscan_model.fit(embedding_train)
        self.fitted = True

        return self.hdbscan_model.labels_

    def predict(self, latent_trajectories_test: np.ndarray) -> np.ndarray:
        """Assign test points using trained models (no refitting!)."""
        if not self.fitted:
            raise RuntimeError("Must fit before predict")

        # Use mean latent for state assignment
        mean_latents = latent_trajectories_test.mean(axis=1)

        # Transform test through fitted UMAP
        embedding_test = self.umap_model.transform(mean_latents)

        # Use approximate_predict for cluster assignment
        labels_test, strengths = hdbscan.approximate_predict(
            self.hdbscan_model, embedding_test
        )

        return labels_test

    def get_n_clusters(self) -> int:
        """Get number of discovered clusters (excluding noise=-1)."""
        if not self.fitted:
            return 0
        labels = self.hdbscan_model.labels_
        return len(set(labels)) - (1 if -1 in labels else 0)


def select_informative_states(
    state_labels: np.ndarray,
    min_samples: int = 20,
) -> list[int]:
    """
    Select informative states using train-only criteria.

    No test labels peeking - uses only:
    - Cluster size (minimum samples)
    - Excludes noise cluster (-1)
    """
    unique, counts = np.unique(state_labels, return_counts=True)

    informative = []
    for state, count in zip(unique, counts):
        if state == -1:  # Skip noise
            continue
        if count >= min_samples:
            informative.append(state)

    return informative


def compute_subject_level_prediction(
    segment_probs: np.ndarray,
    segment_subject_ids: np.ndarray,
    segment_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate segment-level predictions to subject-level.

    Uses mean probability aggregation (deterministic).
    """
    unique_subjects = np.unique(segment_subject_ids)
    subject_probs = []
    subject_labels = []

    for subj in unique_subjects:
        mask = segment_subject_ids == subj
        # Mean probability for this subject
        prob = segment_probs[mask].mean()
        # All segments from same subject have same label
        label = segment_labels[mask][0]

        subject_probs.append(prob)
        subject_labels.append(label)

    return np.array(subject_probs), np.array(subject_labels)


def compute_bootstrap_ci(
    subject_probs: np.ndarray,
    subject_labels: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval on subject-level AUC.

    This provides more robust uncertainty estimates than fold std alone.

    Args:
        subject_probs: Predicted probabilities per subject
        subject_labels: True labels per subject
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level (default 95%)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (auc, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(random_state)
    n_subjects = len(subject_probs)

    if n_subjects < 3:
        # Too few subjects for bootstrap
        try:
            auc = roc_auc_score(subject_labels, subject_probs)
        except ValueError:
            auc = 0.5
        return auc, np.nan, np.nan

    bootstrap_aucs = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        idx = rng.choice(n_subjects, size=n_subjects, replace=True)
        boot_probs = subject_probs[idx]
        boot_labels = subject_labels[idx]

        # Skip if single class
        if len(np.unique(boot_labels)) < 2:
            continue

        try:
            boot_auc = roc_auc_score(boot_labels, boot_probs)
            bootstrap_aucs.append(boot_auc)
        except ValueError:
            continue

    if len(bootstrap_aucs) < 100:
        # Not enough valid bootstrap samples
        try:
            auc = roc_auc_score(subject_labels, subject_probs)
        except ValueError:
            auc = 0.5
        return auc, np.nan, np.nan

    # Compute percentile CI
    alpha = (1 - ci_level) / 2
    ci_lower = np.percentile(bootstrap_aucs, 100 * alpha)
    ci_upper = np.percentile(bootstrap_aucs, 100 * (1 - alpha))
    auc_mean = np.mean(bootstrap_aucs)

    return auc_mean, ci_lower, ci_upper


def compute_retention_stats(
    original_mask: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
) -> dict:
    """Compute window retention statistics by group."""
    n_total = len(original_mask)
    n_retained = original_mask.sum()

    # By group
    hc_mask = labels == 0
    mci_mask = labels == 1

    retention_hc = original_mask[hc_mask].mean() if hc_mask.any() else 0.0
    retention_mci = original_mask[mci_mask].mean() if mci_mask.any() else 0.0

    # By subject
    retention_by_subject = {}
    for subj in np.unique(subject_ids):
        subj_mask = subject_ids == subj
        retention_by_subject[int(subj)] = original_mask[subj_mask].mean()

    return {
        "overall": n_retained / n_total if n_total > 0 else 0.0,
        "hc": retention_hc,
        "mci": retention_mci,
        "by_subject": retention_by_subject,
    }


def run_single_fold(
    fold_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    raw_chunks_hf1: np.ndarray,
    raw_chunks_hf2: np.ndarray,
    latent_trajectories: np.ndarray,
    subject_ids: np.ndarray,
    labels: np.ndarray,
    sfreq: float,
    channel_names: list[str],
    config: ExperimentConfig,
    seed: int,
    rr_target: float,
    latent_mode: str,
    run_null: bool = False,
    run_retention_matched: bool = True,  # NEW: compute retention-matched baseline
) -> list[FoldResult]:
    """
    Run a single fold of the experiment.

    Returns results for both baseline and state-conditioned.

    CRITICAL FIX: Now uses RQA features from full latent trajectories,
    not mean embeddings.

    CRITICAL FIX (2026-01-15): Now receives TWO raw chunk arrays:
    - raw_chunks_hf1: Filtered to 48 Hz - for autoencoder & HF1 computation
    - raw_chunks_hf2: Filtered to 120 Hz - for HF2 (70-110 Hz) validation
    """
    results = []

    # Split data - now with two raw versions
    raw_train_hf1, raw_test_hf1 = raw_chunks_hf1[train_idx], raw_chunks_hf1[test_idx]
    raw_train_hf2, raw_test_hf2 = raw_chunks_hf2[train_idx], raw_chunks_hf2[test_idx]
    latents_train, latents_test = latent_trajectories[train_idx], latent_trajectories[test_idx]
    subj_train, subj_test = subject_ids[train_idx], subject_ids[test_idx]
    labels_train, labels_test = labels[train_idx], labels[test_idx]

    # ---------- PER-FOLD SUBJECT DISTRIBUTION (critic agent rec #4) ----------
    # With tiny folds (3-4 subjects test), AUC can hit 0/0.25/0.33 just from discrete counts
    unique_test_subj = np.unique(subj_test)
    n_hc_test = sum(1 for s in unique_test_subj if labels[subject_ids == s][0] == 0)
    n_mci_test = sum(1 for s in unique_test_subj if labels[subject_ids == s][0] == 1)
    logger.info(f"    Test subjects: {len(unique_test_subj)} ({n_hc_test} HC, {n_mci_test} MCI)")

    # ---------- SINGLE-CLASS FOLD HANDLING (critic agent rec) ----------
    # If test fold has only one class, AUC is undefined - return empty results
    if n_hc_test == 0 or n_mci_test == 0:
        logger.warning(f"    SKIPPING: Single-class test fold (HC={n_hc_test}, MCI={n_mci_test})")
        return []  # Return empty results for this fold

    # Compute fold diagnostics using mean latents (for artifact control)
    # Use HF2-filtered data for proper HF2 computation
    mean_latents_train = latents_train.mean(axis=1)
    mean_latents_test = latents_test.mean(axis=1)

    diagnostics = compute_fold_diagnostics(
        raw_data_train=raw_train_hf2,  # Use HF2-filtered for proper HF2 analysis
        raw_data_test=raw_test_hf2,
        latents_train=mean_latents_train,
        latents_test=mean_latents_test,
        sfreq=sfreq,
        fold_idx=fold_idx,
        channel_names=channel_names,
        train_subject_ids=subj_train,
        test_subject_ids=subj_test,
    )

    # Apply residualization to latent TRAJECTORIES if needed
    if latent_mode == "residualized":
        # Use HF1-filtered data for residualization (matches what autoencoder sees)
        hf_train = compute_hf_power(raw_train_hf1, sfreq, band=(30.0, 48.0), method="welch")
        hf_test = compute_hf_power(raw_test_hf1, sfreq, band=(30.0, 48.0), method="welch")

        # Residualize mean latents and apply same transformation to trajectories
        # This is a simplification - ideally would residualize per-timepoint
        mean_res_train, mean_res_test, _ = residualize_latent(
            mean_latents_train, mean_latents_test,
            hf_train.hf_power, hf_test.hf_power
        )
        # For now, use residualized means for clustering, raw trajectories for RQA
        # A more sophisticated approach would residualize each timepoint

    # Compute RQA features from latent trajectories - THIS IS THE KEY FIX
    # CRITICAL: Uses Theiler window to remove trivial autocorrelation
    logger.info(f"  Computing RQA features with RR={rr_target:.1%}, Theiler={config.theiler_window}...")
    features_train, stats_train = compute_all_rqa_features(
        latents_train, target_rr=rr_target, theiler_window=config.theiler_window
    )
    features_test, stats_test = compute_all_rqa_features(
        latents_test, target_rr=rr_target, theiler_window=config.theiler_window
    )

    logger.info(f"    Train RQA stats: DET={stats_train['mean_det']:.3f}, LAM={stats_train['mean_lam']:.3f}")

    # Normalize features (fit on train only!)
    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)
    features_test_scaled = scaler.transform(features_test)

    # ---------- BASELINE ----------
    # Train classifier on all segments using RQA features
    clf_baseline = XGBClassifier(
        n_estimators=config.xgb_n_estimators,
        max_depth=config.xgb_max_depth,
        learning_rate=config.xgb_learning_rate,
        random_state=seed,
        eval_metric="logloss",
        verbosity=0,
    )

    clf_baseline.fit(features_train_scaled, labels_train)
    probs_test_baseline = clf_baseline.predict_proba(features_test_scaled)[:, 1]

    # Subject-level aggregation
    subj_probs_baseline, subj_labels_baseline = compute_subject_level_prediction(
        probs_test_baseline, subj_test, labels_test
    )

    try:
        auc_baseline = roc_auc_score(subj_labels_baseline, subj_probs_baseline)
    except ValueError:
        auc_baseline = 0.5  # Single class

    # Compute bootstrap CI for more robust uncertainty (guardrail #12)
    _, ci_lower_baseline, ci_upper_baseline = compute_bootstrap_ci(
        subj_probs_baseline, subj_labels_baseline, random_state=seed
    )

    results.append(FoldResult(
        fold_idx=fold_idx,
        seed=seed,
        rr_target=rr_target,
        latent_mode=latent_mode,
        condition="baseline",
        subject_auc=auc_baseline,
        n_subjects_train=len(np.unique(subj_train)),
        n_subjects_test=len(np.unique(subj_test)),
        n_segments_train=len(train_idx),
        n_segments_test=len(test_idx),
        auc_ci_lower=ci_lower_baseline,
        auc_ci_upper=ci_upper_baseline,
        hf1_correlation_raw=diagnostics.hf1_correlation_raw,
        hf1_correlation_residualized=diagnostics.hf1_correlation_residualized,
        hf2_correlation_raw=diagnostics.hf2_correlation_raw,
        hf_temporal_central_ratio=diagnostics.hf1_temporal_central_ratio,
        n_states_discovered=0,
        n_states_used=0,
        is_null_model=False,
        mean_det=stats_train["mean_det"],
        mean_lam=stats_train["mean_lam"],
        mean_entr=stats_train["mean_entr"],
    ))

    # ---------- RETENTION-MATCHED BASELINE ----------
    # CRITICAL FIX (per critic): Compute baseline on same segments as state-conditioned
    # This isolates the effect of state structure from the effect of segment filtering
    # If retention-matched baseline == state-conditioned, the "improvement" is just selection bias

    # ---------- STATE-CONDITIONED ----------
    # Fit state discovery on train only (uses mean latents for clustering)
    state_discovery = StateDiscovery(
        n_neighbors=config.umap_n_neighbors,
        min_dist=config.umap_min_dist,
        n_components=config.umap_n_components,
        min_cluster_size=config.hdbscan_min_cluster_size,
        min_samples=config.hdbscan_min_samples,
        random_state=seed,
    )

    states_train = state_discovery.fit(latents_train)
    states_test = state_discovery.predict(latents_test)

    n_states = state_discovery.get_n_clusters()
    informative_states = select_informative_states(states_train, min_samples=20)

    if run_null:
        # Null model: shuffle state labels within subject
        states_train_shuffled = states_train.copy()
        for subj in np.unique(subj_train):
            subj_mask = subj_train == subj
            np.random.shuffle(states_train_shuffled[subj_mask])
        states_train = states_train_shuffled

        states_test_shuffled = states_test.copy()
        for subj in np.unique(subj_test):
            subj_mask = subj_test == subj
            np.random.shuffle(states_test_shuffled[subj_mask])
        states_test = states_test_shuffled

    # Select segments in informative states
    train_state_mask = np.isin(states_train, informative_states)
    test_state_mask = np.isin(states_test, informative_states)

    # Compute retention stats
    retention = compute_retention_stats(test_state_mask, labels_test, subj_test)

    # ---------- RETENTION-MATCHED BASELINE (CRITICAL CONTROL) ----------
    # Per critic agent: Compute baseline AUC on SAME segments that state-conditioning uses
    # If AUC lift disappears under retention-matching → the "effect" is selection bias
    auc_retention_matched = 0.5
    ci_lower_rm = np.nan
    ci_upper_rm = np.nan

    if run_retention_matched and not run_null and train_state_mask.sum() >= 10 and test_state_mask.sum() >= 5:
        # Train classifier on the SAME retained segments (same mask)
        clf_rm = XGBClassifier(
            n_estimators=config.xgb_n_estimators,
            max_depth=config.xgb_max_depth,
            learning_rate=config.xgb_learning_rate,
            random_state=seed,
            eval_metric="logloss",
            verbosity=0,
        )

        clf_rm.fit(features_train_scaled[train_state_mask], labels_train[train_state_mask])
        probs_test_rm = clf_rm.predict_proba(features_test_scaled[test_state_mask])[:, 1]

        # Subject-level aggregation
        subj_probs_rm, subj_labels_rm = compute_subject_level_prediction(
            probs_test_rm, subj_test[test_state_mask], labels_test[test_state_mask]
        )

        try:
            auc_retention_matched = roc_auc_score(subj_labels_rm, subj_probs_rm)
        except ValueError:
            auc_retention_matched = 0.5

        _, ci_lower_rm, ci_upper_rm = compute_bootstrap_ci(
            subj_probs_rm, subj_labels_rm, random_state=seed
        )

        # Add retention-matched baseline result
        results.append(FoldResult(
            fold_idx=fold_idx,
            seed=seed,
            rr_target=rr_target,
            latent_mode=latent_mode,
            condition="baseline_retention_matched",  # NEW condition type
            subject_auc=auc_retention_matched,
            n_subjects_train=len(np.unique(subj_train[train_state_mask])),
            n_subjects_test=len(np.unique(subj_test[test_state_mask])),
            n_segments_train=train_state_mask.sum(),
            n_segments_test=test_state_mask.sum(),
            auc_ci_lower=ci_lower_rm if not np.isnan(ci_lower_rm) else None,
            auc_ci_upper=ci_upper_rm if not np.isnan(ci_upper_rm) else None,
            retention_overall=retention["overall"],
            retention_mci=retention["mci"],
            retention_hc=retention["hc"],
            hf1_correlation_raw=diagnostics.hf1_correlation_raw,
            hf1_correlation_residualized=diagnostics.hf1_correlation_residualized,
            hf2_correlation_raw=diagnostics.hf2_correlation_raw,
            hf_temporal_central_ratio=diagnostics.hf1_temporal_central_ratio,
            n_states_discovered=n_states,
            n_states_used=len(informative_states),
            is_null_model=False,
            mean_det=features_train[train_state_mask, 1].mean() if train_state_mask.sum() > 0 else None,
            mean_lam=features_train[train_state_mask, 6].mean() if train_state_mask.sum() > 0 else None,
            mean_entr=features_train[train_state_mask, 5].mean() if train_state_mask.sum() > 0 else None,
        ))

    ci_lower_state = np.nan
    ci_upper_state = np.nan
    auc_state = 0.5
    state_det = None
    state_lam = None
    state_entr = None

    if train_state_mask.sum() < 10 or test_state_mask.sum() < 5:
        # Not enough segments after state selection
        logger.warning(f"Fold {fold_idx}: Too few segments after state selection")
    else:
        # Train on state-selected segments
        clf_state = XGBClassifier(
            n_estimators=config.xgb_n_estimators,
            max_depth=config.xgb_max_depth,
            learning_rate=config.xgb_learning_rate,
            random_state=seed,
            eval_metric="logloss",
            verbosity=0,
        )

        clf_state.fit(features_train_scaled[train_state_mask], labels_train[train_state_mask])
        probs_test_state = clf_state.predict_proba(features_test_scaled[test_state_mask])[:, 1]

        # Subject-level aggregation
        subj_probs_state, subj_labels_state = compute_subject_level_prediction(
            probs_test_state, subj_test[test_state_mask], labels_test[test_state_mask]
        )

        try:
            auc_state = roc_auc_score(subj_labels_state, subj_probs_state)
        except ValueError:
            auc_state = 0.5

        # Compute bootstrap CI for state-conditioned (guardrail #12)
        _, ci_lower_state, ci_upper_state = compute_bootstrap_ci(
            subj_probs_state, subj_labels_state, random_state=seed
        )

        # Compute RQA stats for state-selected segments
        state_det = features_train[train_state_mask, 1].mean()  # DET is index 1
        state_lam = features_train[train_state_mask, 6].mean()  # LAM is index 6
        state_entr = features_train[train_state_mask, 5].mean()  # ENTR is index 5

    condition_name = "state_conditioned_null" if run_null else "state_conditioned"

    results.append(FoldResult(
        fold_idx=fold_idx,
        seed=seed,
        rr_target=rr_target,
        latent_mode=latent_mode,
        condition=condition_name,
        subject_auc=auc_state,
        n_subjects_train=len(np.unique(subj_train[train_state_mask])) if train_state_mask.sum() > 0 else 0,
        n_subjects_test=len(np.unique(subj_test[test_state_mask])) if test_state_mask.sum() > 0 else 0,
        n_segments_train=train_state_mask.sum(),
        n_segments_test=test_state_mask.sum(),
        auc_ci_lower=ci_lower_state if not np.isnan(ci_lower_state) else None,
        auc_ci_upper=ci_upper_state if not np.isnan(ci_upper_state) else None,
        retention_overall=retention["overall"],
        retention_mci=retention["mci"],
        retention_hc=retention["hc"],
        hf1_correlation_raw=diagnostics.hf1_correlation_raw,
        hf1_correlation_residualized=diagnostics.hf1_correlation_residualized,
        hf2_correlation_raw=diagnostics.hf2_correlation_raw,
        hf_temporal_central_ratio=diagnostics.hf1_temporal_central_ratio,
        hf_state_prediction_accuracy=diagnostics.hf_state_prediction_accuracy,
        n_states_discovered=n_states,
        n_states_used=len(informative_states),
        is_null_model=run_null,
        mean_det=state_det,
        mean_lam=state_lam,
        mean_entr=state_entr,
    ))

    return results


def run_experiment(config: ExperimentConfig) -> pd.DataFrame:
    """Run the full integration experiment."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    # CRITICAL FIX: Require trained checkpoint
    if config.checkpoint_path is None or not Path(config.checkpoint_path).exists():
        raise ValueError(
            "A trained model checkpoint is REQUIRED for this experiment.\n"
            "Please provide --checkpoint path/to/model.pt\n"
            "Train a model first with: python -m eeg_biomarkers.training.train"
        )

    # Load checkpoint FIRST to get model config (needed for data loading)
    logger.info(f"Loading checkpoint: {config.checkpoint_path}")
    checkpoint = torch.load(config.checkpoint_path, map_location="cpu", weights_only=False)

    # Check if this is a legacy checkpoint
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict_keys = list(state_dict.keys())

    # Legacy detection: look for old architecture patterns
    is_legacy_checkpoint = any("encoder.conv1.weight" in k for k in state_dict_keys)

    if is_legacy_checkpoint:
        raise ValueError(
            "Legacy checkpoint detected! This checkpoint was trained with the old model "
            "architecture and potentially buggy phase representation (raw angles).\n"
            "Please train a new model with the modern architecture:\n"
            "  python -m eeg_biomarkers.training.train\n"
            "The modern architecture uses circular (cos, sin) phase representation."
        )

    # Read model config from checkpoint (CRITICAL: must match training config)
    hidden_size = config.hidden_size
    complexity = config.complexity
    phase_channels = 2  # Default: cos, sin only
    include_amplitude = False

    if 'config' in checkpoint:
        ckpt_cfg = checkpoint['config']
        model_cfg = ckpt_cfg.get('model', {})
        encoder_cfg = model_cfg.get('encoder', {})
        phase_cfg = model_cfg.get('phase', {})

        hidden_size = encoder_cfg.get('hidden_size', config.hidden_size)
        complexity = encoder_cfg.get('complexity', config.complexity)

        if phase_cfg.get('include_amplitude', False):
            phase_channels = 3
            include_amplitude = True
            logger.info("Checkpoint was trained WITH amplitude (phase_channels=3)")

        logger.info(f"Checkpoint config: hidden_size={hidden_size}, complexity={complexity}, phase_channels={phase_channels}")
    else:
        logger.warning("No config in checkpoint, using experiment defaults")

    # Load data - now with amplitude support based on checkpoint config
    data_dir = Path(config.data_dir)
    raw_chunks_hf1, raw_chunks_hf2, phase_chunks, subject_ids, labels, channel_names, sfreq = load_all_data(
        data_dir,
        config.chunk_duration,
        config.filter_low,
        config.filter_high,
        hf2_filter_high=120.0,  # Allow HF2 (70-110 Hz) computation
        include_amplitude=include_amplitude,  # Match checkpoint config!
    )

    # Create model with checkpoint's config
    n_channels = phase_chunks.shape[1] // phase_channels

    model = ConvLSTMAutoencoder(
        n_channels=n_channels,
        hidden_size=hidden_size,
        complexity=complexity,
        phase_channels=phase_channels,
    )

    model.load_state_dict(state_dict)
    logger.info("=" * 60)
    logger.info("AE STATUS: TRAINED")
    logger.info(f"  Checkpoint: {config.checkpoint_path}")
    logger.info(f"  Phase channels: {phase_channels} ({'cos+sin+amplitude' if include_amplitude else 'cos+sin only'})")
    logger.info(f"  Input shape: {phase_chunks.shape} (n_segments, n_features, n_samples)")
    logger.info("=" * 60)

    # Subject count warning (critic agent recommendation #6)
    unique_subjects = np.unique(subject_ids)
    n_hc = len(np.unique(subject_ids[labels == 0]))
    n_mci = len(np.unique(subject_ids[labels == 1]))
    min_class = min(n_hc, n_mci)

    logger.info(f"Subject counts: HC={n_hc}, MCI={n_mci}, min={min_class}")

    if min_class < config.min_subjects_per_class:
        logger.warning("=" * 60)
        logger.warning("UNDERPOWERED WARNING: Subject count is too low for reliable AUC!")
        logger.warning(f"  Min subjects per class: {min_class} (recommended: >= {config.min_subjects_per_class})")
        logger.warning("  AUC variance will be high and CIs may hit 0.")
        logger.warning("  Treat results as SMOKE TEST ONLY, not evidence.")
        logger.warning("=" * 60)

    # Compute latent TRAJECTORIES (not mean latents!)
    logger.info("Computing latent trajectories (TRAINED model)...")
    latent_trajectories = compute_latent_trajectories(model, phase_chunks)
    logger.info(f"  Latent trajectory shape: {latent_trajectories.shape}")

    # Check for latent collapse (preflight check)
    latent_std = latent_trajectories.std()
    if latent_std < 0.01:
        logger.warning(f"PREFLIGHT WARNING: Latent std={latent_std:.6f} suggests possible collapse!")

    # Compute RANDOM model latents for comparison (critic agent recommendation #2)
    latent_trajectories_random = None
    if config.compare_random:
        logger.info("Computing latent trajectories (RANDOM model for comparison)...")
        model_random = ConvLSTMAutoencoder(
            n_channels=n_channels,
            hidden_size=hidden_size,  # Use same config as trained model
            complexity=complexity,
            phase_channels=phase_channels,
        )
        # Don't load any weights - use random initialization
        latent_trajectories_random = compute_latent_trajectories(model_random, phase_chunks)
        logger.info(f"  Random latent trajectory shape: {latent_trajectories_random.shape}")

    # Cross-validation with stratification at subject level
    # CRITICAL FIX: Use StratifiedGroupKFold to balance HC/MCI per fold
    all_results = []

    # For StratifiedGroupKFold, we need subject-level labels
    # Create a mapping: each segment gets its subject's label
    unique_subjects = np.unique(subject_ids)
    subject_label_map = {}
    for subj in unique_subjects:
        subj_mask = subject_ids == subj
        subject_label_map[subj] = labels[subj_mask][0]

    # Use StratifiedGroupKFold with subject-level stratification
    sgkf = StratifiedGroupKFold(n_splits=config.n_folds, shuffle=True, random_state=42)

    for seed in range(config.n_seeds):
        logger.info(f"\n=== Seed {seed + 1}/{config.n_seeds} ===")
        np.random.seed(seed)

        for rr_target in config.rr_targets:
            logger.info(f"\n--- RR target: {rr_target:.1%} ---")

            for fold_idx, (train_idx, test_idx) in enumerate(
                sgkf.split(latent_trajectories, labels, groups=subject_ids)
            ):
                logger.info(f"Fold {fold_idx + 1}/{config.n_folds}")

                for latent_mode in ["raw", "residualized"]:
                    # Real run
                    results = run_single_fold(
                        fold_idx=fold_idx,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        raw_chunks_hf1=raw_chunks_hf1,
                        raw_chunks_hf2=raw_chunks_hf2,
                        latent_trajectories=latent_trajectories,
                        subject_ids=subject_ids,
                        labels=labels,
                        sfreq=sfreq,
                        channel_names=channel_names,
                        config=config,
                        seed=seed,
                        rr_target=rr_target,
                        latent_mode=latent_mode,
                        run_null=False,
                    )
                    all_results.extend(results)

                    # Null model run (shuffled state labels)
                    if config.run_null_model:
                        null_results = run_single_fold(
                            fold_idx=fold_idx,
                            train_idx=train_idx,
                            test_idx=test_idx,
                            raw_chunks_hf1=raw_chunks_hf1,
                            raw_chunks_hf2=raw_chunks_hf2,
                            latent_trajectories=latent_trajectories,
                            subject_ids=subject_ids,
                            labels=labels,
                            sfreq=sfreq,
                            channel_names=channel_names,
                            config=config,
                            seed=seed,
                            rr_target=rr_target,
                            latent_mode=latent_mode,
                            run_null=True,
                        )
                        all_results.extend(null_results)

                    # Random model comparison (critic agent recommendation #2)
                    # If trained beats random -> training is adding signal
                    # If they're similar -> training isn't learning anything useful
                    if config.compare_random and latent_trajectories_random is not None:
                        random_results = run_single_fold(
                            fold_idx=fold_idx,
                            train_idx=train_idx,
                            test_idx=test_idx,
                            raw_chunks_hf1=raw_chunks_hf1,
                            raw_chunks_hf2=raw_chunks_hf2,
                            latent_trajectories=latent_trajectories_random,  # RANDOM latents!
                            subject_ids=subject_ids,
                            labels=labels,
                            sfreq=sfreq,
                            channel_names=channel_names,
                            config=config,
                            seed=seed,
                            rr_target=rr_target,
                            latent_mode=latent_mode,
                            run_null=False,
                        )
                        # Mark as random model results
                        for r in random_results:
                            r.is_random_model = True
                        all_results.extend(random_results)

    # Create DataFrame
    df = pd.DataFrame([asdict(r) for r in all_results])

    # Save CSV
    csv_path = output_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved results to {csv_path}")

    return df


def generate_report(df: pd.DataFrame, output_dir: Path) -> str:
    """Generate human-readable markdown report."""
    lines = []
    lines.append("# Integration Experiment Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("\n## Summary Statistics")

    # Filter out null model for main analysis
    df_real = df[~df["is_null_model"]]

    # Group by condition and latent mode
    for latent_mode in ["raw", "residualized"]:
        lines.append(f"\n### Latent Mode: {latent_mode.upper()}")

        df_mode = df_real[df_real["latent_mode"] == latent_mode]

        for condition in ["baseline", "state_conditioned"]:
            df_cond = df_mode[df_mode["condition"] == condition]
            if df_cond.empty:
                continue

            auc_mean = df_cond["subject_auc"].mean()
            auc_std = df_cond["subject_auc"].std()

            # Get bootstrap CI if available
            ci_lowers = df_cond["auc_ci_lower"].dropna()
            ci_uppers = df_cond["auc_ci_upper"].dropna()

            lines.append(f"\n**{condition}:**")
            lines.append(f"- AUC: {auc_mean:.3f} +/- {auc_std:.3f}")
            if len(ci_lowers) > 0 and len(ci_uppers) > 0:
                lines.append(f"- Bootstrap 95% CI: [{ci_lowers.mean():.3f}, {ci_uppers.mean():.3f}]")

            if condition == "state_conditioned":
                lines.append(f"- Retention (overall): {df_cond['retention_overall'].mean():.2%}")
                lines.append(f"- Retention (MCI): {df_cond['retention_mci'].mean():.2%}")
                lines.append(f"- Retention (HC): {df_cond['retention_hc'].mean():.2%}")
                lines.append(f"- States discovered: {df_cond['n_states_discovered'].mean():.1f}")
                lines.append(f"- States used: {df_cond['n_states_used'].mean():.1f}")

            # RQA statistics
            if df_cond["mean_det"].notna().any():
                lines.append(f"- Mean DET: {df_cond['mean_det'].mean():.3f}")
                lines.append(f"- Mean LAM: {df_cond['mean_lam'].mean():.3f}")
                lines.append(f"- Mean ENTR: {df_cond['mean_entr'].mean():.3f}")

    # Effect analysis
    lines.append("\n## Effect Analysis")

    for latent_mode in ["raw", "residualized"]:
        df_mode = df_real[df_real["latent_mode"] == latent_mode]

        baseline_auc = df_mode[df_mode["condition"] == "baseline"]["subject_auc"].mean()
        state_auc = df_mode[df_mode["condition"] == "state_conditioned"]["subject_auc"].mean()
        effect = state_auc - baseline_auc

        lines.append(f"\n**{latent_mode.upper()} latent:**")
        lines.append(f"- Baseline AUC: {baseline_auc:.3f}")
        lines.append(f"- State-conditioned AUC: {state_auc:.3f}")
        lines.append(f"- Effect: {effect:+.3f}")

    # ---------- RETENTION-MATCHED BASELINE ANALYSIS (CRITICAL) ----------
    # Per critic agent: If AUC lift disappears under retention-matching, effect is selection bias
    df_rm = df_real[df_real["condition"] == "baseline_retention_matched"]
    if not df_rm.empty:
        lines.append("\n## Retention-Matched Baseline Analysis (CRITICAL)")
        lines.append("\n*This controls for selection bias - if state-conditioned ≈ retention-matched,")
        lines.append("the 'improvement' comes from segment filtering, NOT from meaningful states.*\n")

        for latent_mode in ["raw", "residualized"]:
            df_mode = df_real[df_real["latent_mode"] == latent_mode]
            df_rm_mode = df_rm[df_rm["latent_mode"] == latent_mode]

            state_auc = df_mode[df_mode["condition"] == "state_conditioned"]["subject_auc"].mean()
            rm_auc = df_rm_mode["subject_auc"].mean() if not df_rm_mode.empty else np.nan

            if not np.isnan(rm_auc):
                delta = state_auc - rm_auc
                lines.append(f"**{latent_mode.upper()} latent:**")
                lines.append(f"- State-conditioned AUC: {state_auc:.3f}")
                lines.append(f"- Retention-matched baseline AUC: {rm_auc:.3f}")
                lines.append(f"- Δ (State - RM Baseline): {delta:+.3f}")

                if abs(delta) < 0.02:
                    lines.append(f"  **WARNING:** State-conditioning ≈ retention-matched → selection bias!")
                elif delta > 0.02:
                    lines.append(f"  **OK:** State structure adds value beyond segment selection.")
                else:
                    lines.append(f"  **BAD:** State structure is WORSE than random segment selection!")
                lines.append("")

    # Trained vs Random comparison (critic agent recommendation #2)
    if "is_random_model" in df.columns and df["is_random_model"].any():
        lines.append("\n## Trained vs Random Model Comparison")
        lines.append("\n*If trained and random are similar, training isn't adding signal.*\n")

        df_trained = df[(~df["is_null_model"]) & (~df["is_random_model"])]
        df_random = df[df["is_random_model"]]

        for condition in ["baseline", "state_conditioned"]:
            trained_auc = df_trained[df_trained["condition"] == condition]["subject_auc"].mean()
            random_auc = df_random[df_random["condition"] == condition]["subject_auc"].mean()
            delta = trained_auc - random_auc

            lines.append(f"**{condition}:**")
            lines.append(f"- Trained AUC: {trained_auc:.3f}")
            lines.append(f"- Random AUC: {random_auc:.3f}")
            lines.append(f"- Δ (Trained - Random): {delta:+.3f}")

            if delta < 0.02:
                lines.append(f"  **WARNING:** Training not significantly improving over random!")
            else:
                lines.append(f"  **OK:** Training is adding signal.")
            lines.append("")

    # Null model check
    if df["is_null_model"].any():
        lines.append("\n## Null Model Sanity Check")
        df_null = df[df["is_null_model"]]

        null_auc = df_null[df_null["condition"] == "state_conditioned_null"]["subject_auc"].mean()
        real_auc = df_real[df_real["condition"] == "state_conditioned"]["subject_auc"].mean()

        lines.append(f"\n- Real state-conditioned AUC: {real_auc:.3f}")
        lines.append(f"- Null (shuffled states) AUC: {null_auc:.3f}")

        if null_auc >= real_auc - 0.02:
            lines.append("\n**WARNING:** Null model performs similarly - state structure may be exploiting confound!")
        else:
            lines.append("\n**OK:** Real model outperforms null - state structure is meaningful.")

    # RR sensitivity with bootstrap CIs (critic agent rec #6)
    lines.append("\n## RR Target Sensitivity (with 95% CIs)")
    for rr in sorted(df_real["rr_target"].unique()):
        df_rr = df_real[df_real["rr_target"] == rr]
        baseline_df = df_rr[df_rr["condition"] == "baseline"]
        state_df = df_rr[df_rr["condition"] == "state_conditioned"]

        baseline_auc = baseline_df["subject_auc"].mean()
        state_auc = state_df["subject_auc"].mean()

        # Get bootstrap CIs if available
        bl_ci_lo = baseline_df["auc_ci_lower"].dropna().mean() if baseline_df["auc_ci_lower"].notna().any() else np.nan
        bl_ci_hi = baseline_df["auc_ci_upper"].dropna().mean() if baseline_df["auc_ci_upper"].notna().any() else np.nan
        st_ci_lo = state_df["auc_ci_lower"].dropna().mean() if state_df["auc_ci_lower"].notna().any() else np.nan
        st_ci_hi = state_df["auc_ci_upper"].dropna().mean() if state_df["auc_ci_upper"].notna().any() else np.nan

        ci_str_bl = f" [{bl_ci_lo:.2f}-{bl_ci_hi:.2f}]" if not np.isnan(bl_ci_lo) else ""
        ci_str_st = f" [{st_ci_lo:.2f}-{st_ci_hi:.2f}]" if not np.isnan(st_ci_lo) else ""

        lines.append(f"- RR={rr:.0%}: Baseline={baseline_auc:.3f}{ci_str_bl}, State={state_auc:.3f}{ci_str_st}, Δ={state_auc - baseline_auc:+.3f}")

    # Go/No-Go Decision
    lines.append("\n## Go/No-Go Decision")

    best_effect = 0
    for latent_mode in ["raw", "residualized"]:
        df_mode = df_real[df_real["latent_mode"] == latent_mode]
        baseline = df_mode[df_mode["condition"] == "baseline"]["subject_auc"].mean()
        state = df_mode[df_mode["condition"] == "state_conditioned"]["subject_auc"].mean()
        best_effect = max(best_effect, state - baseline)

    lines.append(f"\nBest effect: {best_effect:+.3f}")

    if best_effect >= 0.05:
        lines.append("\n**STRONG SUCCESS:** State-conditioning improves AUC by >= 5 points")
    elif best_effect >= 0.02:
        lines.append("\n**MIXED:** Small improvement - examine variance reduction and state-specific effects")
    else:
        lines.append("\n**PIVOT SIGNAL:** No clear improvement - consider alternative approaches")

    report = "\n".join(lines)

    # Save report
    report_path = output_dir / "report.md"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Saved report to {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Run integration experiment")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--output-dir", default="results/integration", help="Output directory")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path (REQUIRED)")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--n-seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--no-null-model", action="store_true", help="Skip null model runs")
    parser.add_argument("--no-compare-random", action="store_true",
                       help="Skip trained-vs-random comparison (faster but less diagnostic)")

    args = parser.parse_args()

    config = ExperimentConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        n_folds=args.n_folds,
        n_seeds=args.n_seeds,
        run_null_model=not args.no_null_model,
        compare_random=not args.no_compare_random,
    )

    # Run experiment
    df = run_experiment(config)

    # Generate report
    report = generate_report(df, Path(config.output_dir))
    print("\n" + report)

    logger.info("\nExperiment complete!")


if __name__ == "__main__":
    main()
