#!/usr/bin/env python3
"""
Recurrence Matrix Visualization Script

Generates publication-quality plots of recurrence matrices from trained autoencoder latents.
Works on RunPod (headless) by using Agg backend.

Usage:
    # Basic usage with checkpoint
    python scripts/plot_recurrence_matrices.py \
        --checkpoint outputs/eeg_mci_biomarkers/<timestamp>/best.pt \
        --data-dir data \
        --output-dir results/recurrence_plots

    # With specific subjects and RR targets
    python scripts/plot_recurrence_matrices.py \
        --checkpoint best.pt \
        --data-dir data \
        --output-dir results/recurrence_plots \
        --n-subjects 5 \
        --rr-targets 0.01 0.02 0.05 \
        --theiler-window 25

    # Compare HC vs MCI
    python scripts/plot_recurrence_matrices.py \
        --checkpoint best.pt \
        --data-dir data \
        --output-dir results/recurrence_plots \
        --compare-groups
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Use Agg backend for headless environments (RunPod)
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add src to path if running as script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eeg_biomarkers.models import ConvLSTMAutoencoder, TransformerAutoencoder
from eeg_biomarkers.analysis.rqa import (
    compute_rqa_features,
    compute_rqa_from_distance_matrix,
    apply_theiler_window,
    RQAFeatures,
)
from eeg_biomarkers.data.dataset import CachedFileDataset, get_cache_key

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_angular_distance_matrix(latent_trajectory: np.ndarray) -> np.ndarray:
    """
    Compute angular distance matrix from latent trajectory.

    Args:
        latent_trajectory: (T, hidden_size) latent states over time

    Returns:
        (T, T) angular distance matrix
    """
    # Normalize to unit sphere
    norms = np.linalg.norm(latent_trajectory, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = latent_trajectory / norms

    # Cosine similarity -> angular distance
    cos_sim = np.dot(normalized, normalized.T)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    distance_matrix = np.arccos(cos_sim)

    return distance_matrix


def compute_recurrence_matrix(
    distance_matrix: np.ndarray,
    target_rr: float = 0.02,
    theiler_window: int = 0,
) -> tuple[np.ndarray, float]:
    """
    Compute binary recurrence matrix with RR-controlled threshold.

    Args:
        distance_matrix: (T, T) distance matrix
        target_rr: Target recurrence rate (e.g., 0.02 for 2%)
        theiler_window: Exclude |i-j| < theiler_window

    Returns:
        Binary recurrence matrix and threshold used
    """
    N = distance_matrix.shape[0]

    # Create mask for threshold computation
    if theiler_window > 0:
        mask = np.ones((N, N), dtype=bool)
        for i in range(N):
            for j in range(max(0, i - theiler_window + 1), min(N, i + theiler_window)):
                mask[i, j] = False
    else:
        mask = ~np.eye(N, dtype=bool)

    off_diag = distance_matrix[mask]

    # Find threshold at target RR percentile
    epsilon = np.percentile(off_diag, target_rr * 100)

    # Create binary recurrence matrix
    R = (distance_matrix <= epsilon).astype(np.float64)

    # Apply Theiler window
    if theiler_window > 0:
        R = apply_theiler_window(R, theiler_window)

    return R, epsilon


def plot_single_recurrence_matrix(
    R: np.ndarray,
    title: str = "Recurrence Matrix",
    rqa_features: RQAFeatures | None = None,
    sfreq: float = 250.0,
    figsize: tuple = (10, 10),
    cmap: str = "binary",
) -> plt.Figure:
    """
    Plot a single recurrence matrix with optional RQA stats.

    Args:
        R: Binary recurrence matrix (T, T)
        title: Plot title
        rqa_features: Optional RQA features to display
        sfreq: Sampling frequency for time axis
        figsize: Figure size
        cmap: Colormap

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    T = R.shape[0]
    time_extent = T / sfreq  # Convert to seconds

    im = ax.imshow(
        R,
        cmap=cmap,
        origin="lower",
        extent=[0, time_extent, 0, time_extent],
        aspect="equal",
    )

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Time (s)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add RQA stats as text box
    if rqa_features is not None:
        stats_text = (
            f"RR: {rqa_features.RR:.3f}\n"
            f"DET: {rqa_features.DET:.3f}\n"
            f"LAM: {rqa_features.LAM:.3f}\n"
            f"L: {rqa_features.L:.1f}\n"
            f"ENTR: {rqa_features.ENTR:.2f}"
        )
        props = dict(boxstyle="round", facecolor="white", alpha=0.8)
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=props,
        )

    plt.tight_layout()
    return fig


def plot_distance_and_recurrence(
    distance_matrix: np.ndarray,
    R: np.ndarray,
    title: str = "Distance and Recurrence",
    rqa_features: RQAFeatures | None = None,
    sfreq: float = 250.0,
    figsize: tuple = (16, 7),
) -> plt.Figure:
    """
    Plot distance matrix and recurrence matrix side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    T = R.shape[0]
    time_extent = T / sfreq

    # Distance matrix
    im1 = axes[0].imshow(
        distance_matrix,
        cmap="viridis",
        origin="lower",
        extent=[0, time_extent, 0, time_extent],
        aspect="equal",
    )
    axes[0].set_xlabel("Time (s)", fontsize=12)
    axes[0].set_ylabel("Time (s)", fontsize=12)
    axes[0].set_title("Angular Distance Matrix", fontsize=12)
    plt.colorbar(im1, ax=axes[0], label="Distance (rad)")

    # Recurrence matrix
    im2 = axes[1].imshow(
        R,
        cmap="binary",
        origin="lower",
        extent=[0, time_extent, 0, time_extent],
        aspect="equal",
    )
    axes[1].set_xlabel("Time (s)", fontsize=12)
    axes[1].set_ylabel("Time (s)", fontsize=12)
    axes[1].set_title("Recurrence Matrix", fontsize=12)

    # Add RQA stats
    if rqa_features is not None:
        stats_text = (
            f"RR: {rqa_features.RR:.3f}  DET: {rqa_features.DET:.3f}  "
            f"LAM: {rqa_features.LAM:.3f}  ENTR: {rqa_features.ENTR:.2f}"
        )
        fig.suptitle(f"{title}\n{stats_text}", fontsize=14, fontweight="bold")
    else:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_rr_comparison(
    distance_matrix: np.ndarray,
    rr_targets: list[float],
    title: str = "RR Comparison",
    theiler_window: int = 0,
    sfreq: float = 250.0,
    figsize: tuple = (18, 5),
) -> plt.Figure:
    """
    Plot recurrence matrices at different RR targets side by side.
    """
    n_targets = len(rr_targets)
    fig, axes = plt.subplots(1, n_targets, figsize=figsize)

    if n_targets == 1:
        axes = [axes]

    T = distance_matrix.shape[0]
    time_extent = T / sfreq

    for i, rr_target in enumerate(rr_targets):
        R, epsilon = compute_recurrence_matrix(
            distance_matrix, target_rr=rr_target, theiler_window=theiler_window
        )
        rqa_features = compute_rqa_features(R)

        axes[i].imshow(
            R,
            cmap="binary",
            origin="lower",
            extent=[0, time_extent, 0, time_extent],
            aspect="equal",
        )
        axes[i].set_xlabel("Time (s)", fontsize=11)
        axes[i].set_ylabel("Time (s)", fontsize=11)
        axes[i].set_title(
            f"RR={rr_target*100:.0f}%\n"
            f"DET={rqa_features.DET:.2f}, LAM={rqa_features.LAM:.2f}",
            fontsize=11,
        )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_group_comparison(
    latents_hc: list[np.ndarray],
    latents_mci: list[np.ndarray],
    target_rr: float = 0.02,
    theiler_window: int = 0,
    sfreq: float = 250.0,
    figsize: tuple = (16, 8),
) -> plt.Figure:
    """
    Plot HC vs MCI recurrence matrices comparison.
    """
    n_examples = min(3, len(latents_hc), len(latents_mci))
    fig, axes = plt.subplots(2, n_examples, figsize=figsize)

    rqa_hc_list = []
    rqa_mci_list = []

    for i in range(n_examples):
        # HC
        dist_hc = compute_angular_distance_matrix(latents_hc[i])
        R_hc, _ = compute_recurrence_matrix(dist_hc, target_rr, theiler_window)
        rqa_hc = compute_rqa_features(R_hc)
        rqa_hc_list.append(rqa_hc)

        T = R_hc.shape[0]
        time_extent = T / sfreq

        axes[0, i].imshow(R_hc, cmap="binary", origin="lower",
                         extent=[0, time_extent, 0, time_extent])
        axes[0, i].set_title(f"HC #{i+1}\nDET={rqa_hc.DET:.2f}, LAM={rqa_hc.LAM:.2f}")
        axes[0, i].set_xlabel("Time (s)")
        if i == 0:
            axes[0, i].set_ylabel("Time (s)")

        # MCI
        dist_mci = compute_angular_distance_matrix(latents_mci[i])
        R_mci, _ = compute_recurrence_matrix(dist_mci, target_rr, theiler_window)
        rqa_mci = compute_rqa_features(R_mci)
        rqa_mci_list.append(rqa_mci)

        T = R_mci.shape[0]
        time_extent = T / sfreq

        axes[1, i].imshow(R_mci, cmap="binary", origin="lower",
                         extent=[0, time_extent, 0, time_extent])
        axes[1, i].set_title(f"MCI #{i+1}\nDET={rqa_mci.DET:.2f}, LAM={rqa_mci.LAM:.2f}")
        axes[1, i].set_xlabel("Time (s)")
        if i == 0:
            axes[1, i].set_ylabel("Time (s)")

    # Compute group averages
    avg_det_hc = np.mean([r.DET for r in rqa_hc_list])
    avg_det_mci = np.mean([r.DET for r in rqa_mci_list])
    avg_lam_hc = np.mean([r.LAM for r in rqa_hc_list])
    avg_lam_mci = np.mean([r.LAM for r in rqa_mci_list])

    fig.suptitle(
        f"HC vs MCI Recurrence Matrices (RR={target_rr*100:.0f}%)\n"
        f"HC avg: DET={avg_det_hc:.2f}, LAM={avg_lam_hc:.2f} | "
        f"MCI avg: DET={avg_det_mci:.2f}, LAM={avg_lam_mci:.2f}",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    return fig


def load_model_from_checkpoint(checkpoint_path: Path) -> tuple[torch.nn.Module, dict]:
    """Load model from checkpoint, auto-detecting model type."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    config = checkpoint.get("config", {})

    model_cfg = config.get("model", {})
    encoder_cfg = model_cfg.get("encoder", {})
    phase_cfg = model_cfg.get("phase", {})

    hidden_size = encoder_cfg.get("hidden_size", 64)
    complexity = encoder_cfg.get("complexity", 1)
    phase_channels = 3 if phase_cfg.get("include_amplitude", False) else 2

    # Detect model type
    model_name = model_cfg.get("name", "convlstm_autoencoder")
    is_transformer = "transformer" in model_name.lower() or any(
        "transformer_encoder" in k for k in state_dict.keys()
    )

    # We need n_channels from data, will be set later
    model_info = {
        "hidden_size": hidden_size,
        "complexity": complexity,
        "phase_channels": phase_channels,
        "is_transformer": is_transformer,
        "n_heads": encoder_cfg.get("n_heads", 4),
        "n_transformer_layers": encoder_cfg.get("n_transformer_layers", 2),
        "dim_feedforward": encoder_cfg.get("dim_feedforward", 256),
        "state_dict": state_dict,
        "include_amplitude": phase_cfg.get("include_amplitude", False),
    }

    return None, model_info  # Model created later when n_channels known


def create_model(n_channels: int, model_info: dict) -> torch.nn.Module:
    """Create model with known n_channels."""
    if model_info["is_transformer"]:
        model = TransformerAutoencoder(
            n_channels=n_channels,
            hidden_size=model_info["hidden_size"],
            complexity=model_info["complexity"],
            phase_channels=model_info["phase_channels"],
            n_heads=model_info["n_heads"],
            n_transformer_layers=model_info["n_transformer_layers"],
            dim_feedforward=model_info["dim_feedforward"],
        )
    else:
        model = ConvLSTMAutoencoder(
            n_channels=n_channels,
            hidden_size=model_info["hidden_size"],
            complexity=model_info["complexity"],
            phase_channels=model_info["phase_channels"],
        )

    model.load_state_dict(model_info["state_dict"])
    model.eval()
    return model


def compute_latent_trajectory(
    model: torch.nn.Module,
    phase_data: np.ndarray,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Compute latent trajectory from phase data.

    Args:
        model: Trained autoencoder
        phase_data: (n_features, n_samples) phase data for one segment
        batch_size: Batch size for inference

    Returns:
        (n_samples, hidden_size) latent trajectory
    """
    model.eval()
    device = next(model.parameters()).device

    # Add batch dimension: (1, n_features, n_samples)
    x = torch.from_numpy(phase_data).float().unsqueeze(0).to(device)

    with torch.no_grad():
        _, latent = model(x)  # latent: (1, hidden_size, T)

    # Return as (T, hidden_size)
    return latent.squeeze(0).permute(1, 0).cpu().numpy()


def load_cached_subjects(
    data_dir: Path,
    cache_dir: Path,
    include_amplitude: bool,
    max_subjects: int | None = None,
) -> tuple[list[np.ndarray], list[str], list[int]]:
    """
    Load cached phase data for subjects.

    Supports two cache formats:
    1. New format: cache_dir/{cache_key}.pt with "chunks" key
    2. Preprocessed format: cache_dir/{condition}/FILT/{subject}/*.pt with "phase_data" key

    Returns:
        phase_data_list, subject_ids, labels
    """
    phase_data_list = []
    subject_ids = []
    labels = []

    # Try preprocessed_data format first (recursive .pt search)
    pt_files = list(cache_dir.rglob("*.pt"))

    if pt_files:
        logger.info(f"  Found {len(pt_files)} .pt files in {cache_dir}")

        for pt_file in sorted(pt_files):
            # Determine label from path (HC/HID = 0, MCI/AD = 1)
            path_str = str(pt_file).upper()
            if "/HID/" in path_str or "/HC/" in path_str or "\\HID\\" in path_str or "\\HC\\" in path_str:
                label = 0
            elif "/MCI/" in path_str or "/AD/" in path_str or "\\MCI\\" in path_str or "\\AD\\" in path_str:
                label = 1
            else:
                logger.warning(f"  Cannot determine label for {pt_file}, skipping")
                continue

            # Extract subject ID from path or filename
            # Format: .../S017 20140124 0857/S017 20140124 0857_good_1_eeg_xxx.pt
            subject_id = pt_file.parent.name.split()[0] if pt_file.parent.name.startswith("S") else pt_file.stem.split("_")[0]

            try:
                cached = torch.load(pt_file, weights_only=False, map_location="cpu")

                # Handle different cache formats
                if "chunks" in cached:
                    # New format: list of chunks
                    chunks = cached["chunks"]
                    for chunk in chunks:
                        data = chunk.numpy() if isinstance(chunk, torch.Tensor) else chunk
                        phase_data_list.append(data)
                        subject_ids.append(subject_id)
                        labels.append(label)
                elif "phase_data" in cached:
                    # Preprocessed format: single phase_data array
                    data = cached["phase_data"]
                    if isinstance(data, torch.Tensor):
                        data = data.numpy()
                    phase_data_list.append(data)
                    subject_ids.append(subject_id)
                    labels.append(label)
                elif isinstance(cached, torch.Tensor):
                    # Raw tensor format
                    phase_data_list.append(cached.numpy())
                    subject_ids.append(subject_id)
                    labels.append(label)
                elif isinstance(cached, np.ndarray):
                    # Raw numpy format
                    phase_data_list.append(cached)
                    subject_ids.append(subject_id)
                    labels.append(label)
                else:
                    # Try to find any tensor/array in the dict
                    for key, value in cached.items():
                        if isinstance(value, (torch.Tensor, np.ndarray)):
                            data = value.numpy() if isinstance(value, torch.Tensor) else value
                            if data.ndim == 2:  # (n_features, n_samples)
                                phase_data_list.append(data)
                                subject_ids.append(subject_id)
                                labels.append(label)
                                logger.info(f"  Loaded {pt_file.name} using key '{key}'")
                                break

            except Exception as e:
                logger.warning(f"  Failed to load {pt_file}: {e}")
                continue

            if max_subjects and len(set(subject_ids)) >= max_subjects:
                break

        if phase_data_list:
            return phase_data_list, subject_ids, labels

    # Fallback: Try original format with .fif files
    for condition_dir in ["HC", "HID", "MCI", "AD"]:
        condition_path = data_dir / condition_dir
        if not condition_path.exists():
            continue

        label = 0 if condition_dir in ["HC", "HID"] else 1

        for fif_file in sorted(condition_path.glob("*.fif")):
            subject_id = fif_file.stem
            cache_key = get_cache_key(
                fif_file,
                chunk_duration=5.0,
                filter_low=1.0,
                filter_high=30.0,
                include_amplitude=include_amplitude,
            )
            cache_path = cache_dir / f"{cache_key}.pt"

            if cache_path.exists():
                cached = torch.load(cache_path, weights_only=False)
                chunks = cached["chunks"]

                for chunk in chunks:
                    phase_data_list.append(chunk.numpy() if isinstance(chunk, torch.Tensor) else chunk)
                    subject_ids.append(subject_id)
                    labels.append(label)

                if max_subjects and len(set(subject_ids)) >= max_subjects:
                    break

        if max_subjects and len(set(subject_ids)) >= max_subjects:
            break

    return phase_data_list, subject_ids, labels


def main():
    parser = argparse.ArgumentParser(
        description="Generate recurrence matrix visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Path to data directory"
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Path to cache directory (default: data_dir/.cache)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/recurrence_plots",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--n-subjects", type=int, default=5,
        help="Number of subjects to plot"
    )
    parser.add_argument(
        "--rr-targets", type=float, nargs="+", default=[0.01, 0.02, 0.05],
        help="Target recurrence rates"
    )
    parser.add_argument(
        "--theiler-window", type=int, default=0,
        help="Theiler window (samples to exclude around diagonal)"
    )
    parser.add_argument(
        "--compare-groups", action="store_true",
        help="Generate HC vs MCI comparison plot"
    )
    parser.add_argument(
        "--sfreq", type=float, default=250.0,
        help="Sampling frequency"
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="DPI for saved figures"
    )

    args = parser.parse_args()

    # Setup paths
    checkpoint_path = Path(args.checkpoint)
    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else data_dir / ".cache"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Cache dir: {cache_dir}")
    logger.info(f"Output dir: {output_dir}")

    # Load model info
    logger.info("Loading checkpoint...")
    _, model_info = load_model_from_checkpoint(checkpoint_path)
    logger.info(f"  Model type: {'Transformer' if model_info['is_transformer'] else 'ConvLSTM'}")
    logger.info(f"  Hidden size: {model_info['hidden_size']}")
    logger.info(f"  Phase channels: {model_info['phase_channels']}")

    # Load cached data
    logger.info("Loading cached data...")
    phase_data_list, subject_ids, labels = load_cached_subjects(
        data_dir, cache_dir,
        include_amplitude=model_info["include_amplitude"],
        max_subjects=args.n_subjects * 2,  # Load more to have both HC and MCI
    )

    if len(phase_data_list) == 0:
        logger.error("No cached data found! Run training first to generate cache.")
        return 1

    logger.info(f"  Loaded {len(phase_data_list)} segments from {len(set(subject_ids))} subjects")

    # Determine n_channels from data
    n_features = phase_data_list[0].shape[0]
    n_channels = n_features // model_info["phase_channels"]
    logger.info(f"  n_channels: {n_channels}, n_features: {n_features}")

    # Create model
    model = create_model(n_channels, model_info)
    logger.info("Model loaded successfully")

    # Compute latents and generate plots
    latents_by_label = {0: [], 1: []}  # HC=0, MCI=1

    unique_subjects = list(dict.fromkeys(subject_ids))[:args.n_subjects]

    for i, subject_id in enumerate(unique_subjects):
        # Find first segment for this subject
        idx = subject_ids.index(subject_id)
        phase_data = phase_data_list[idx]
        label = labels[idx]
        group = "HC" if label == 0 else "MCI"

        logger.info(f"Processing subject {i+1}/{len(unique_subjects)}: {subject_id} ({group})")
        logger.info(f"  Input phase_data shape: {phase_data.shape} (n_features, n_samples)")
        logger.info(f"  Input duration: {phase_data.shape[1] / args.sfreq:.2f} seconds")

        # Compute latent trajectory
        latent = compute_latent_trajectory(model, phase_data)
        latents_by_label[label].append(latent)

        logger.info(f"  Latent trajectory shape: {latent.shape} (n_time, hidden_size)")
        logger.info(f"  Latent duration: {latent.shape[0] / args.sfreq:.2f} seconds")

        # Compute distance matrix
        distance_matrix = compute_angular_distance_matrix(latent)

        # Plot 1: Distance + Recurrence side by side
        R, epsilon = compute_recurrence_matrix(
            distance_matrix,
            target_rr=args.rr_targets[1] if len(args.rr_targets) > 1 else args.rr_targets[0],
            theiler_window=args.theiler_window,
        )
        rqa_features = compute_rqa_features(R)

        fig = plot_distance_and_recurrence(
            distance_matrix, R,
            title=f"Subject: {subject_id} ({group})",
            rqa_features=rqa_features,
            sfreq=args.sfreq,
        )
        save_path = output_dir / f"{subject_id}_distance_recurrence.png"
        fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved: {save_path}")

        # Plot 2: RR comparison
        if len(args.rr_targets) > 1:
            fig = plot_rr_comparison(
                distance_matrix,
                args.rr_targets,
                title=f"Subject: {subject_id} ({group}) - RR Comparison",
                theiler_window=args.theiler_window,
                sfreq=args.sfreq,
            )
            save_path = output_dir / f"{subject_id}_rr_comparison.png"
            fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"  Saved: {save_path}")

    # Group comparison plot
    if args.compare_groups and len(latents_by_label[0]) > 0 and len(latents_by_label[1]) > 0:
        logger.info("Generating HC vs MCI comparison...")
        fig = plot_group_comparison(
            latents_by_label[0],
            latents_by_label[1],
            target_rr=args.rr_targets[1] if len(args.rr_targets) > 1 else args.rr_targets[0],
            theiler_window=args.theiler_window,
            sfreq=args.sfreq,
        )
        save_path = output_dir / "group_comparison_hc_vs_mci.png"
        fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {save_path}")

    logger.info(f"\nAll plots saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
