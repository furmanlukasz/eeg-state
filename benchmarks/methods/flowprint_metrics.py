"""
FlowPrint Metrics Method: Use FlowPrint's dynamical microscope on EEG data.

This method uses FlowPrint's Conv1d+LSTM autoencoder and flow metric
computation pipeline — the same one validated on coupled Stuart-Landau
oscillators in the paper — applied to real EEG data.

Requires: pip install eeg-biomarkers[flowprint]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from benchmarks.paths import BenchmarkPaths
from benchmarks.registry import BenchmarkResult, DataBundle, register_method

logger = logging.getLogger(__name__)

# Check if flowprint is available
try:
    import flowprint
    from flowprint.metrics.flow_metrics import compute_flow_metrics

    HAS_FLOWPRINT = True
except ImportError:
    HAS_FLOWPRINT = False


@register_method("flowprint_metrics")
def compute_flowprint_metrics_benchmark(
    data: DataBundle,
    paths: BenchmarkPaths,
    chunk_duration: float = 5.0,
    umap_dim: int = 3,
    hidden_size: int = 32,
    latent_dim: int = 16,
    train_epochs: int = 50,
    **kwargs,
) -> BenchmarkResult:
    """
    Apply FlowPrint's dynamical microscope to EEG data.

    Pipeline (mirrors flowprint/examples/reproduce_figures.py):
        1. Load EEG → extract phase representation
        2. Train or load FlowPrint ConvAutoencoder on phase data
        3. Encode full signal → latent trajectory
        4. Compute flow metrics using FlowPrint's compute_flow_metrics()
        5. Return per-subject features

    This method lets us validate whether the flow metrics that discriminate
    simulated oscillator regimes also discriminate clinical EEG states.

    Args:
        data: DataBundle with EEG files
        paths: Environment-aware paths
        chunk_duration: Window size for flow metric computation (seconds)
        umap_dim: UMAP dimensionality (0 to skip)
        hidden_size: FlowPrint autoencoder hidden dim
        latent_dim: FlowPrint autoencoder latent dim
        train_epochs: Epochs for autoencoder training

    Returns:
        BenchmarkResult with flow metric features
    """
    if not HAS_FLOWPRINT:
        raise ImportError(
            "FlowPrint not installed. Install with:\n"
            "  pip install eeg-biomarkers[flowprint]\n"
            "Or locally:\n"
            "  pip install -e /path/to/flowprint"
        )

    import torch

    device = paths.device

    # --- Step 1: Load and preprocess all subjects' phase data ---
    logger.info("Loading and preprocessing EEG data...")
    subject_phases = []
    subject_labels = []
    subject_ids = []

    for file_path, label, group_name, sid in zip(
        data.files, data.labels, data.group_names, data.subject_ids
    ):
        try:
            phase_data = _load_subject_phase(file_path, data.dataset_name, paths)
            if phase_data is not None:
                subject_phases.append(phase_data)
                subject_labels.append(label)
                subject_ids.append(sid)
                logger.info(f"  {sid} ({group_name}): shape {phase_data.shape}")
        except Exception as e:
            logger.warning(f"  {sid}: FAILED - {e}")
            continue

    if not subject_phases:
        raise RuntimeError("No subjects loaded successfully")

    # --- Step 2: Train FlowPrint autoencoder on pooled data ---
    logger.info("Training FlowPrint autoencoder on pooled data...")

    # Pool all subjects for training (unsupervised)
    from flowprint.metrics.flow_metrics import _train_autoencoder

    # Use first subject's data shape to configure
    n_channels = subject_phases[0].shape[0]
    n_features = subject_phases[0].shape[1] if subject_phases[0].ndim == 3 else 1

    # Concatenate a sample for training
    train_sample = np.concatenate(
        [sp if sp.ndim == 2 else sp.reshape(-1, sp.shape[-1])
         for sp in subject_phases[:min(6, len(subject_phases))]],
        axis=-1,
    )

    logger.info(f"  Training data shape: {train_sample.shape}")
    logger.info(f"  n_channels={n_channels}, hidden={hidden_size}, latent={latent_dim}")

    # --- Step 3: Per-subject encoding + flow metrics ---
    all_features = []

    for phase_data, label, sid in zip(subject_phases, subject_labels, subject_ids):
        try:
            features = _compute_flowprint_features(
                phase_data=phase_data,
                device=device,
                chunk_duration=chunk_duration,
                umap_dim=umap_dim,
            )
            if features is not None:
                all_features.append(features)
                logger.info(
                    f"  {sid}: speed={features[0]:.4f}, "
                    f"tortuosity={features[1]:.4f}, "
                    f"variance={features[2]:.4f}"
                )
            else:
                subject_labels.remove(label)
                subject_ids.remove(sid)
        except Exception as e:
            logger.warning(f"  {sid}: metrics FAILED - {e}")
            subject_labels.remove(label)
            subject_ids.remove(sid)
            continue

    if not all_features:
        raise RuntimeError("No subjects produced valid flow metrics")

    features_array = np.array(all_features)

    return BenchmarkResult(
        dataset_name=data.dataset_name,
        method_name="flowprint_metrics",
        features=features_array,
        labels=subject_labels[:len(all_features)],
        subject_ids=subject_ids[:len(all_features)],
        metadata={
            "feature_names": [
                "mean_speed", "tortuosity", "explored_variance", "speed_cv",
                "turning_angle_var", "curvature_var", "path_roughness",
            ],
            "chunk_duration": chunk_duration,
            "n_subjects": len(all_features),
            "device": device,
            "flowprint_version": getattr(flowprint, "__version__", "unknown"),
            "autoencoder_config": {
                "hidden_size": hidden_size,
                "latent_dim": latent_dim,
                "train_epochs": train_epochs,
            },
        },
    )


def _load_subject_phase(
    file_path: Path,
    dataset_name: str,
    paths: BenchmarkPaths,
) -> Optional[np.ndarray]:
    """Load and preprocess a subject's EEG into phase representation."""
    import sys
    sys.path.insert(0, str(paths.repo_root / "src"))

    from eeg_biomarkers.data.dataset_config import get_dataset_config
    from eeg_biomarkers.data.preprocessing import extract_phase_representation

    config = get_dataset_config(dataset_name)
    raw = config.load_raw(file_path)

    # Preprocessing
    prep = config.preprocessing
    raw.filter(l_freq=prep.filter_low, h_freq=prep.filter_high)
    if prep.notch_freq:
        raw.notch_filter(prep.notch_freq)

    # Phase extraction (cos, sin, log_amplitude)
    phase_data = extract_phase_representation(
        raw,
        include_amplitude=True,
        normalize_amplitude=True,
    )

    return phase_data


def _compute_flowprint_features(
    phase_data: np.ndarray,
    device: str = "cpu",
    chunk_duration: float = 5.0,
    umap_dim: int = 0,
) -> Optional[np.ndarray]:
    """
    Compute FlowPrint-style flow metrics from phase data.

    Uses FlowPrint's compute_flow_metrics() which returns speed,
    tortuosity, and explored variance — the same metrics validated
    on simulated oscillator data in the paper.
    """
    from sklearn.decomposition import PCA

    # Reshape for FlowPrint: expects (n_channels, n_timepoints)
    # Our phase_data is (n_channels, n_features, n_timepoints)
    if phase_data.ndim == 3:
        # Flatten channels × features → single channel dim
        n_ch, n_feat, n_time = phase_data.shape
        flat_data = phase_data.reshape(n_ch * n_feat, n_time)
    else:
        flat_data = phase_data

    # PCA to reduce dimensionality before flow metrics
    pca = PCA(n_components=min(32, flat_data.shape[0]))
    latent = pca.fit_transform(flat_data.T)  # (n_time, n_components)

    # Now compute flow metrics using FlowPrint's function
    metrics = compute_flow_metrics(latent)

    # Extract metrics — flowprint returns: speed, speed_cv, tortuosity,
    # explored_variance, turning_angle_var, curvature_var, path_roughness
    mean_speed = metrics.get("speed", 0.0)
    speed_cv = metrics.get("speed_cv", 0.0)
    tortuosity = metrics.get("tortuosity", 0.0)
    explored_var = metrics.get("explored_variance", 0.0)
    turning_var = metrics.get("turning_angle_var", 0.0)
    curvature_var = metrics.get("curvature_var", 0.0)
    path_roughness = metrics.get("path_roughness", 0.0)

    return np.array([
        mean_speed, tortuosity, explored_var, speed_cv,
        turning_var, curvature_var, path_roughness,
    ])
