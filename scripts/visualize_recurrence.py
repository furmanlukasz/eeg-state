#!/usr/bin/env python
"""
Visualize Recurrence Matrices from Trained Autoencoder

This script generates visualizations of:
1. Angular distance matrices from latent trajectories
2. Binary recurrence matrices at different RR thresholds
3. RQA feature comparisons between HC and MCI

Usage:
    # With a trained checkpoint
    python scripts/visualize_recurrence.py \
        --checkpoint outputs/eeg_mci_biomarkers/<timestamp>/best.pt \
        --data-dir data \
        --output-dir results/recurrence_plots \
        --n-samples 5

    # On RunPod with cached data
    python scripts/visualize_recurrence.py \
        --checkpoint models/best.pt \
        --data-dir /workspace/data \
        --output-dir results/recurrence_plots \
        --use-cache \
        --n-samples 10

    # Quick sanity check (random model)
    python scripts/visualize_recurrence.py \
        --random-model \
        --output-dir results/recurrence_sanity
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

# Suppress MNE verbose output
import mne
mne.set_log_level("ERROR")

from eeg_biomarkers.models import ConvLSTMAutoencoder, TransformerAutoencoder
from eeg_biomarkers.analysis.rqa import compute_rqa_features, compute_rqa_from_distance_matrix, RQAFeatures
from eeg_biomarkers.data.preprocessing import load_eeg_file, prepare_phase_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_angular_distance_matrix(latent: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute angular distance matrix from latent trajectory.

    Args:
        latent: Latent trajectory (time, hidden_size)
        eps: Small constant for numerical stability

    Returns:
        Angular distance matrix (time, time)
    """
    # Normalize latent vectors
    norms = np.linalg.norm(latent, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    normalized = latent / norms

    # Cosine similarity matrix
    cos_sim = np.dot(normalized, normalized.T)

    # Clamp and compute angular distance
    cos_sim = np.clip(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7)
    angular_dist = np.arccos(cos_sim)

    return angular_dist


def plot_distance_and_recurrence(
    distance_matrix: np.ndarray,
    rr_targets: list[float] = [0.01, 0.02, 0.05],
    title: str = "Recurrence Analysis",
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Plot distance matrix and recurrence matrices at different thresholds.

    Args:
        distance_matrix: Angular distance matrix (time, time)
        rr_targets: List of target recurrence rates
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    n_plots = len(rr_targets) + 1
    fig = plt.figure(figsize=(5 * n_plots, 5))
    gs = gridspec.GridSpec(1, n_plots)

    # Plot distance matrix
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(distance_matrix, cmap='viridis', origin='lower', aspect='auto')
    ax0.set_title('Angular Distance Matrix')
    ax0.set_xlabel('Time')
    ax0.set_ylabel('Time')
    plt.colorbar(im0, ax=ax0, label='Angular Distance (rad)')

    # Plot recurrence matrices at each threshold
    for i, rr in enumerate(rr_targets):
        ax = fig.add_subplot(gs[0, i + 1])

        # Compute threshold for target RR
        N = distance_matrix.shape[0]
        mask = ~np.eye(N, dtype=bool)
        off_diag = distance_matrix[mask]
        epsilon = np.percentile(off_diag, rr * 100)

        # Create recurrence matrix
        R = (distance_matrix <= epsilon).astype(np.float32)

        # Compute RQA features
        features = compute_rqa_features(R)

        ax.imshow(R, cmap='binary', origin='lower', aspect='auto')
        ax.set_title(f'RR={rr*100:.1f}% (ε={epsilon:.3f})\nDET={features.DET:.2f}, LAM={features.LAM:.2f}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Time')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")

    return fig


def plot_hc_vs_mci_comparison(
    hc_latents: list[np.ndarray],
    mci_latents: list[np.ndarray],
    rr_target: float = 0.02,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Compare recurrence patterns between HC and MCI subjects.

    Args:
        hc_latents: List of HC latent trajectories
        mci_latents: List of MCI latent trajectories
        rr_target: Target recurrence rate
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    n_hc = min(len(hc_latents), 3)
    n_mci = min(len(mci_latents), 3)

    fig, axes = plt.subplots(2, max(n_hc, n_mci), figsize=(5 * max(n_hc, n_mci), 10))

    # Plot HC examples
    for i in range(n_hc):
        ax = axes[0, i] if max(n_hc, n_mci) > 1 else axes[0]
        dist_mat = compute_angular_distance_matrix(hc_latents[i])
        features, epsilon = compute_rqa_from_distance_matrix(dist_mat, target_rr=rr_target)
        R = (dist_mat <= epsilon).astype(np.float32)

        ax.imshow(R, cmap='Blues', origin='lower', aspect='auto')
        ax.set_title(f'HC #{i+1}\nDET={features.DET:.2f}, LAM={features.LAM:.2f}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Time')

    # Hide empty HC subplots
    for i in range(n_hc, max(n_hc, n_mci)):
        if max(n_hc, n_mci) > 1:
            axes[0, i].axis('off')

    # Plot MCI examples
    for i in range(n_mci):
        ax = axes[1, i] if max(n_hc, n_mci) > 1 else axes[1]
        dist_mat = compute_angular_distance_matrix(mci_latents[i])
        features, epsilon = compute_rqa_from_distance_matrix(dist_mat, target_rr=rr_target)
        R = (dist_mat <= epsilon).astype(np.float32)

        ax.imshow(R, cmap='Reds', origin='lower', aspect='auto')
        ax.set_title(f'MCI #{i+1}\nDET={features.DET:.2f}, LAM={features.LAM:.2f}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Time')

    # Hide empty MCI subplots
    for i in range(n_mci, max(n_hc, n_mci)):
        if max(n_hc, n_mci) > 1:
            axes[1, i].axis('off')

    fig.suptitle(f'Recurrence Comparison: HC vs MCI (RR={rr_target*100:.0f}%)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")

    return fig


def plot_rqa_feature_distributions(
    hc_features: list[RQAFeatures],
    mci_features: list[RQAFeatures],
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Plot distributions of RQA features for HC vs MCI.

    Args:
        hc_features: List of RQAFeatures for HC subjects
        mci_features: List of RQAFeatures for MCI subjects
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    feature_names = ['DET', 'LAM', 'L', 'TT', 'ENTR', 'DIV']

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i, name in enumerate(feature_names):
        ax = axes[i]

        hc_vals = [getattr(f, name) for f in hc_features]
        mci_vals = [getattr(f, name) for f in mci_features]

        # Box plot comparison
        bp = ax.boxplot([hc_vals, mci_vals], labels=['HC', 'MCI'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')

        ax.set_title(name)
        ax.set_ylabel('Value')

        # Add individual points
        ax.scatter(np.ones(len(hc_vals)) + np.random.randn(len(hc_vals)) * 0.05,
                   hc_vals, alpha=0.5, c='blue', s=20)
        ax.scatter(np.ones(len(mci_vals)) * 2 + np.random.randn(len(mci_vals)) * 0.05,
                   mci_vals, alpha=0.5, c='red', s=20)

    fig.suptitle('RQA Feature Distributions: HC vs MCI', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")

    return fig


def load_model_from_checkpoint(checkpoint_path: str) -> tuple[torch.nn.Module, dict]:
    """Load model from checkpoint, detecting type automatically."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    state_dict_keys = list(state_dict.keys())

    # Extract config
    config = checkpoint.get('config', {})
    model_cfg = config.get('model', {})
    encoder_cfg = model_cfg.get('encoder', {})
    phase_cfg = model_cfg.get('phase', {})

    hidden_size = encoder_cfg.get('hidden_size', 64)
    complexity = encoder_cfg.get('complexity', 2)
    phase_channels = 3 if phase_cfg.get('include_amplitude', False) else 2

    # Detect model type
    model_name = model_cfg.get('name', 'convlstm_autoencoder')
    is_transformer = 'transformer' in model_name.lower() or any('transformer_encoder' in k for k in state_dict_keys)

    logger.info(f"Detected model type: {'Transformer' if is_transformer else 'ConvLSTM'}")
    logger.info(f"Config: hidden_size={hidden_size}, complexity={complexity}, phase_channels={phase_channels}")

    return state_dict, {
        'is_transformer': is_transformer,
        'hidden_size': hidden_size,
        'complexity': complexity,
        'phase_channels': phase_channels,
        'include_amplitude': phase_cfg.get('include_amplitude', False),
        'n_heads': encoder_cfg.get('n_heads', 4),
        'n_transformer_layers': encoder_cfg.get('n_transformer_layers', 2),
        'dim_feedforward': encoder_cfg.get('dim_feedforward', 256),
    }


def create_model(n_channels: int, config: dict, load_weights: bool = True, state_dict: dict = None) -> torch.nn.Module:
    """Create model based on config."""
    if config['is_transformer']:
        model = TransformerAutoencoder(
            n_channels=n_channels,
            hidden_size=config['hidden_size'],
            complexity=config['complexity'],
            phase_channels=config['phase_channels'],
            n_heads=config['n_heads'],
            n_transformer_layers=config['n_transformer_layers'],
            dim_feedforward=config['dim_feedforward'],
        )
    else:
        model = ConvLSTMAutoencoder(
            n_channels=n_channels,
            hidden_size=config['hidden_size'],
            complexity=config['complexity'],
            phase_channels=config['phase_channels'],
        )

    if load_weights and state_dict is not None:
        model.load_state_dict(state_dict)

    return model


def find_eeg_files(data_dir: Path) -> tuple[list[Path], list[Path]]:
    """Find HC and MCI EEG files."""
    hc_files = list(data_dir.glob("HID/**/*raw*.fif")) + list(data_dir.glob("AD/**/*raw*.fif"))
    mci_files = list(data_dir.glob("MCI/**/*raw*.fif"))

    logger.info(f"Found {len(hc_files)} HC files, {len(mci_files)} MCI files")

    return hc_files, mci_files


def main():
    parser = argparse.ArgumentParser(description="Visualize recurrence matrices from trained model")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to EEG data directory")
    parser.add_argument("--output-dir", type=str, default="results/recurrence_plots", help="Output directory")
    parser.add_argument("--n-samples", type=int, default=5, help="Number of samples per group to visualize")
    parser.add_argument("--rr-targets", type=float, nargs="+", default=[0.01, 0.02, 0.05], help="Target recurrence rates")
    parser.add_argument("--use-cache", action="store_true", help="Use cached phase data")
    parser.add_argument("--cache-dir", type=str, default=".cache/phase", help="Cache directory")
    parser.add_argument("--random-model", action="store_true", help="Use random (untrained) model for sanity check")
    parser.add_argument("--chunk-duration", type=float, default=5.0, help="Chunk duration in seconds")
    parser.add_argument("--filter-low", type=float, default=3.0, help="Low-pass filter frequency")
    parser.add_argument("--filter-high", type=float, default=48.0, help="High-pass filter frequency")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)

    # Load checkpoint or create random model
    if args.random_model:
        logger.info("Using RANDOM (untrained) model for sanity check")
        model_config = {
            'is_transformer': True,
            'hidden_size': 64,
            'complexity': 2,
            'phase_channels': 3,
            'include_amplitude': True,
            'n_heads': 4,
            'n_transformer_layers': 2,
            'dim_feedforward': 256,
        }
        state_dict = None
    else:
        if not args.checkpoint:
            parser.error("--checkpoint is required unless --random-model is specified")
        state_dict, model_config = load_model_from_checkpoint(args.checkpoint)

    # Find EEG files
    hc_files, mci_files = find_eeg_files(data_dir)

    if len(hc_files) == 0 or len(mci_files) == 0:
        logger.error("No EEG files found. Check --data-dir path.")
        return

    # Process samples
    hc_latents = []
    mci_latents = []
    hc_features_list = []
    mci_features_list = []

    model = None

    for group_name, files, latents_list, features_list in [
        ("HC", hc_files[:args.n_samples], hc_latents, hc_features_list),
        ("MCI", mci_files[:args.n_samples], mci_latents, mci_features_list),
    ]:
        logger.info(f"Processing {group_name} samples...")

        for i, fif_file in enumerate(files):
            try:
                logger.info(f"  Loading: {fif_file.name}")

                # Load and preprocess
                raw = load_eeg_file(
                    fif_file,
                    filter_low=args.filter_low,
                    filter_high=args.filter_high,
                    verbose=False,
                )

                # Prepare chunks
                chunks, mask, info = prepare_phase_chunks(
                    raw,
                    chunk_duration=args.chunk_duration,
                    include_amplitude=model_config['include_amplitude'],
                )

                n_channels = info['n_channels']

                # Create model if not exists
                if model is None:
                    model = create_model(n_channels, model_config,
                                         load_weights=(state_dict is not None),
                                         state_dict=state_dict)
                    model.eval()
                    logger.info(f"Model created: n_channels={n_channels}")

                # Convert to tensor
                chunks_tensor = torch.from_numpy(chunks).float()

                # Compute latents (use first chunk only for visualization)
                with torch.no_grad():
                    _, latent = model(chunks_tensor[:1])  # (1, time', hidden_size)

                latent_np = latent[0].cpu().numpy()  # (time', hidden_size)
                latents_list.append(latent_np)

                # Compute RQA features
                dist_mat = compute_angular_distance_matrix(latent_np)
                features, _ = compute_rqa_from_distance_matrix(dist_mat, target_rr=args.rr_targets[1])
                features_list.append(features)

                logger.info(f"    Latent shape: {latent_np.shape}, DET={features.DET:.3f}, LAM={features.LAM:.3f}")

            except Exception as e:
                logger.warning(f"  Failed to process {fif_file.name}: {e}")
                continue

    if len(hc_latents) == 0 and len(mci_latents) == 0:
        logger.error("No samples could be processed!")
        return

    # Generate plots
    logger.info("\nGenerating plots...")

    # 1. Individual recurrence analysis for first sample
    if len(hc_latents) > 0:
        dist_mat = compute_angular_distance_matrix(hc_latents[0])
        plot_distance_and_recurrence(
            dist_mat,
            rr_targets=args.rr_targets,
            title="HC Example: Distance & Recurrence Matrices",
            save_path=output_dir / "hc_example_recurrence.png",
        )
        plt.close()

    if len(mci_latents) > 0:
        dist_mat = compute_angular_distance_matrix(mci_latents[0])
        plot_distance_and_recurrence(
            dist_mat,
            rr_targets=args.rr_targets,
            title="MCI Example: Distance & Recurrence Matrices",
            save_path=output_dir / "mci_example_recurrence.png",
        )
        plt.close()

    # 2. HC vs MCI comparison
    if len(hc_latents) > 0 and len(mci_latents) > 0:
        plot_hc_vs_mci_comparison(
            hc_latents,
            mci_latents,
            rr_target=args.rr_targets[1],
            save_path=output_dir / "hc_vs_mci_comparison.png",
        )
        plt.close()

    # 3. RQA feature distributions
    if len(hc_features_list) > 1 and len(mci_features_list) > 1:
        plot_rqa_feature_distributions(
            hc_features_list,
            mci_features_list,
            save_path=output_dir / "rqa_feature_distributions.png",
        )
        plt.close()

    # 4. Summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("RQA FEATURE SUMMARY")
    logger.info("=" * 60)

    for name, features_list in [("HC", hc_features_list), ("MCI", mci_features_list)]:
        if len(features_list) > 0:
            det_vals = [f.DET for f in features_list]
            lam_vals = [f.LAM for f in features_list]
            logger.info(f"{name} (n={len(features_list)}):")
            logger.info(f"  DET: {np.mean(det_vals):.3f} +/- {np.std(det_vals):.3f}")
            logger.info(f"  LAM: {np.mean(lam_vals):.3f} +/- {np.std(lam_vals):.3f}")

    logger.info("=" * 60)
    logger.info(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
