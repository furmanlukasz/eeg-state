"""
Model Loading Utilities for Local Analysis

Handles loading both ConvLSTM and Transformer autoencoders.
"""

from pathlib import Path
import torch
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from eeg_biomarkers.models import ConvLSTMAutoencoder, TransformerAutoencoder


def load_model_from_checkpoint(checkpoint_path: Path, device: str = "mps"):
    """
    Load autoencoder model from checkpoint.

    Automatically detects model type (ConvLSTM vs Transformer).

    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: Device to load model on ("mps", "cuda", "cpu")

    Returns:
        Tuple of (model, model_info_dict)
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    config = checkpoint.get("config", {})

    model_cfg = config.get("model", {})
    encoder_cfg = model_cfg.get("encoder", {})
    phase_cfg = model_cfg.get("phase", {})

    # Extract model parameters
    hidden_size = encoder_cfg.get("hidden_size", 64)
    complexity = encoder_cfg.get("complexity", 2)
    phase_channels = 3 if phase_cfg.get("include_amplitude", False) else 2
    include_amplitude = phase_cfg.get("include_amplitude", False)

    # Detect model type
    model_name = model_cfg.get("name", "convlstm_autoencoder")
    is_transformer = "transformer" in model_name.lower() or any(
        "transformer_encoder" in k for k in state_dict.keys()
    )

    # Transformer-specific params
    n_heads = encoder_cfg.get("n_heads", 4)
    n_transformer_layers = encoder_cfg.get("n_transformer_layers", 2)
    dim_feedforward = encoder_cfg.get("dim_feedforward", 256)

    model_info = {
        "hidden_size": hidden_size,
        "complexity": complexity,
        "phase_channels": phase_channels,
        "include_amplitude": include_amplitude,
        "is_transformer": is_transformer,
        "n_heads": n_heads,
        "n_transformer_layers": n_transformer_layers,
        "dim_feedforward": dim_feedforward,
        "state_dict": state_dict,
        "val_loss": checkpoint.get("val_loss", None),
        "epoch": checkpoint.get("epoch", None),
    }

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  Model type: {'Transformer' if is_transformer else 'ConvLSTM'}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Phase channels: {phase_channels} ({'with amplitude' if include_amplitude else 'cos+sin only'})")
    if model_info["val_loss"]:
        print(f"  Val loss: {model_info['val_loss']:.4f}")

    return model_info


def create_model(n_channels: int, model_info: dict, device: str = "mps", load_weights: bool = True):
    """
    Create model instance with correct architecture.

    Args:
        n_channels: Number of EEG channels (e.g., 256)
        model_info: Dict from load_model_from_checkpoint
        device: Device to put model on
        load_weights: If True, load trained weights. If False, use random initialization.

    Returns:
        Loaded model ready for inference
    """
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

    if load_weights:
        model.load_state_dict(model_info["state_dict"])
        print(f"Model loaded with trained weights on {device}")
    else:
        print(f"Model initialized with RANDOM weights on {device}")

    model.to(device)
    model.eval()

    return model


def compute_latent_trajectory(model, phase_data, device: str = "mps"):
    """
    Compute latent trajectory from phase data.

    Args:
        model: Trained autoencoder
        phase_data: (n_features, n_samples) numpy array
        device: Device for inference

    Returns:
        (n_samples, hidden_size) numpy array - latent trajectory over time
    """
    import numpy as np

    model.eval()

    # Add batch dimension: (1, n_features, n_samples)
    x = torch.from_numpy(phase_data).float().unsqueeze(0).to(device)

    with torch.no_grad():
        _, latent = model(x)

    # Handle different output formats
    latent = latent.squeeze(0)  # Remove batch dim

    # If shape is (hidden_size, time) where hidden_size < time, transpose
    if latent.shape[0] < latent.shape[1]:
        latent = latent.permute(1, 0)

    return latent.cpu().numpy()


if __name__ == "__main__":
    # Quick test
    from config import CHECKPOINT_PATH, DEVICE

    model_info = load_model_from_checkpoint(CHECKPOINT_PATH, DEVICE)
    print("\nModel info loaded successfully!")
