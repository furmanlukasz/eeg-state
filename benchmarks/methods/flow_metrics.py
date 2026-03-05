"""
Flow Metrics Method: Encode EEG -> latent trajectories -> flow metrics.

This method loads a trained transformer autoencoder, encodes EEG data
into latent trajectories, then computes flow metrics (speed, tortuosity,
explored variance) per chunk. These metrics can then be compared across
groups using ANOVA/classification.

Shared conceptually with FlowPrint's analysis pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from benchmarks.registry import register_method, DataBundle, BenchmarkResult
from benchmarks.paths import BenchmarkPaths

logger = logging.getLogger(__name__)


@register_method("flow_metrics")
def compute_flow_metrics_benchmark(
    data: DataBundle,
    paths: BenchmarkPaths,
    checkpoint: Optional[str] = None,
    chunk_duration: float = 5.0,
    umap_dim: int = 3,
    **kwargs,
) -> BenchmarkResult:
    """
    Compute flow metrics for each subject in the dataset.

    Pipeline:
        1. Load trained model checkpoint
        2. For each subject: preprocess -> encode -> latent trajectory
        3. Optionally UMAP reduce
        4. Compute per-chunk flow metrics: speed, tortuosity, explored_variance
        5. Return as BenchmarkResult

    Args:
        data: DataBundle with files, labels, subject_ids
        paths: Environment-aware paths (for checkpoint resolution)
        checkpoint: Override checkpoint path
        chunk_duration: Duration of flow metric windows (seconds)
        umap_dim: UMAP dimensionality (0 to skip UMAP)

    Returns:
        BenchmarkResult with features array (n_subjects, n_metrics)
    """
    import sys
    sys.path.insert(0, str(paths.repo_root / "src"))

    # Resolve checkpoint
    cp_path = Path(checkpoint) if checkpoint else paths.checkpoints.get(data.dataset_name)
    if cp_path is None or not cp_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found for {data.dataset_name}. "
            f"Searched: {cp_path}\n"
            f"Available checkpoints: {paths.checkpoints}"
        )

    logger.info(f"Loading checkpoint: {cp_path}")

    # Import model loading utilities
    try:
        from eeg_biomarkers.models.transformer_autoencoder import TransformerAutoencoder
        import torch

        device = paths.device
        checkpoint_data = torch.load(cp_path, map_location=device, weights_only=False)

        # Reconstruct model from checkpoint config
        model_config = checkpoint_data.get("model_config", checkpoint_data.get("config", {}))
        model = _build_model_from_config(model_config)
        model.load_state_dict(checkpoint_data["model_state_dict"])
        model.to(device)

        # Set to inference mode
        model.train(False)

        logger.info(f"Model loaded on {device}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Process each subject
    all_features = []
    all_labels = []
    all_sids = []

    for file_path, label, group_name, sid in zip(
        data.files, data.labels, data.group_names, data.subject_ids
    ):
        try:
            features = _process_subject(
                file_path=file_path,
                model=model,
                device=device,
                dataset_name=data.dataset_name,
                chunk_duration=chunk_duration,
                paths=paths,
            )
            if features is not None:
                all_features.append(features)
                all_labels.append(label)
                all_sids.append(sid)
                logger.info(
                    f"  {sid} ({group_name}): "
                    f"speed={features[0]:.4f}, "
                    f"tortuosity={features[1]:.4f}, "
                    f"variance={features[2]:.4f}"
                )
        except Exception as e:
            logger.warning(f"  {sid}: FAILED - {e}")
            continue

    if not all_features:
        raise RuntimeError("No subjects processed successfully")

    features_array = np.array(all_features)

    return BenchmarkResult(
        dataset_name=data.dataset_name,
        method_name="flow_metrics",
        features=features_array,
        labels=all_labels,
        subject_ids=all_sids,
        metadata={
            "checkpoint": str(cp_path),
            "feature_names": [
                "mean_speed", "tortuosity", "explored_variance",
                "speed_std", "speed_cv",
            ],
            "chunk_duration": chunk_duration,
            "n_subjects": len(all_sids),
            "device": device,
        },
    )


def _build_model_from_config(config: dict):
    """Reconstruct TransformerAutoencoder from saved config."""
    from eeg_biomarkers.models.transformer_autoencoder import TransformerAutoencoder

    # Handle nested config formats
    encoder_cfg = config.get("encoder", config)
    phase_cfg = config.get("phase", {})

    n_channels = encoder_cfg.get("n_channels", 64)
    n_features = 3 if phase_cfg.get("include_amplitude", True) else 2

    model = TransformerAutoencoder(
        n_channels=n_channels,
        n_features=n_features,
        hidden_size=encoder_cfg.get("hidden_size", 64),
        n_heads=encoder_cfg.get("n_heads", 4),
        n_transformer_layers=encoder_cfg.get("n_transformer_layers", 2),
        dim_feedforward=encoder_cfg.get("dim_feedforward", 256),
        dropout=encoder_cfg.get("dropout", 0.1),
        complexity=encoder_cfg.get("complexity", 2),
        constrain_output=config.get("decoder", {}).get("constrain_output", True),
    )
    return model


def _process_subject(
    file_path: Path,
    model,
    device: str,
    dataset_name: str,
    chunk_duration: float,
    paths,
) -> Optional[np.ndarray]:
    """
    Process a single subject: load -> preprocess -> encode -> flow metrics.

    Returns:
        Feature vector [mean_speed, tortuosity, explored_variance, speed_std, speed_cv]
        or None if processing fails.
    """
    import torch
    from eeg_biomarkers.data.dataset_config import get_dataset_config

    config = get_dataset_config(dataset_name)

    # Load raw
    raw = config.load_raw(file_path)

    # Basic preprocessing
    prep = config.preprocessing
    raw.filter(l_freq=prep.filter_low, h_freq=prep.filter_high)
    if prep.notch_freq:
        raw.notch_filter(prep.notch_freq)

    # Extract phase representation
    from eeg_biomarkers.data.preprocessing import extract_phase_representation
    phase_data = extract_phase_representation(
        raw,
        include_amplitude=True,
        normalize_amplitude=True,
    )

    # Chunk and encode
    sfreq = raw.info["sfreq"]
    chunk_samples = int(prep.chunk_duration * sfreq)
    n_chunks = phase_data.shape[-1] // chunk_samples

    if n_chunks < 2:
        logger.warning(f"Too few chunks ({n_chunks}) for {file_path.name}")
        return None

    latent_list = []
    with torch.no_grad():
        for i in range(n_chunks):
            chunk = phase_data[:, :, i * chunk_samples:(i + 1) * chunk_samples]
            chunk_tensor = torch.FloatTensor(chunk).unsqueeze(0).to(device)

            try:
                encoded = model.encode(chunk_tensor)
                # encoded shape: (1, latent_dim, time_steps)
                latent_list.append(encoded.squeeze(0).cpu().numpy())
            except Exception:
                continue

    if len(latent_list) < 2:
        return None

    # Concatenate latent trajectory
    latent = np.concatenate(latent_list, axis=-1)  # (latent_dim, total_time)
    latent = latent.T  # (total_time, latent_dim)

    # Compute flow metrics on latent trajectory
    return _compute_metrics(latent, sfreq=sfreq)


def _compute_metrics(latent: np.ndarray, sfreq: float = 250.0) -> np.ndarray:
    """
    Compute flow metrics from latent trajectory.

    Args:
        latent: (n_timepoints, n_dims) array
        sfreq: sampling frequency

    Returns:
        [mean_speed, tortuosity, explored_variance, speed_std, speed_cv]
    """
    from sklearn.decomposition import PCA

    # Velocity (finite differences)
    velocity = np.diff(latent, axis=0) * sfreq
    speed = np.linalg.norm(velocity, axis=1)

    # Mean speed
    mean_speed = np.mean(speed)
    speed_std = np.std(speed)
    speed_cv = speed_std / (mean_speed + 1e-10)  # coefficient of variation

    # Tortuosity: total path length / displacement
    total_distance = np.sum(speed / sfreq)
    displacement = np.linalg.norm(latent[-1] - latent[0])
    tortuosity = total_distance / (displacement + 1e-10)

    # Explored variance: fraction of variance in top 3 PCs
    if latent.shape[0] > latent.shape[1]:
        pca = PCA(n_components=min(3, latent.shape[1]))
        pca.fit(latent)
        explored_variance = np.sum(pca.explained_variance_ratio_)
    else:
        explored_variance = 1.0

    return np.array([
        mean_speed,
        tortuosity,
        explored_variance,
        speed_std,
        speed_cv,
    ])
