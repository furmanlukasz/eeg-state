#!/usr/bin/env python3
"""
Generate synthetic EEG from trained autoencoder.

This script explores the latent space of the trained autoencoder by:
1. Decoding random latent vectors to generate novel EEG
2. Interpolating between real EEG samples in latent space
3. Reconstructing real EEG through the autoencoder

Usage:
    python scripts/generate_eeg.py --checkpoint models/best2.pt --mode random
    python scripts/generate_eeg.py --checkpoint models/best2.pt --mode interpolate --input-file /path/to/eeg.fif
    python scripts/generate_eeg.py --checkpoint models/best2.pt --mode reconstruct --input-file /path/to/eeg.fif
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eeg_biomarkers.models import ConvLSTMAutoencoder, TransformerAutoencoder


def load_model(checkpoint_path: Path, device: str = "cpu"):
    """Load trained autoencoder from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get model config from checkpoint
    if "config" in checkpoint:
        config = checkpoint["config"]
        model_name = config.get("model", {}).get("name", "convlstm_autoencoder")
        n_channels = config.get("n_channels", 256)
        include_amplitude = config.get("model", {}).get("phase", {}).get("include_amplitude", False)
    else:
        # Fallback: try to infer from state dict
        model_name = "convlstm_autoencoder"
        include_amplitude = False
        # Try to get n_channels from first conv layer
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        for key in state_dict:
            if "conv" in key and "weight" in key:
                in_channels = state_dict[key].shape[1]
                # Could be n_channels * 2 (cos/sin) or n_channels * 3 (cos/sin/amp)
                if in_channels % 3 == 0:
                    n_channels = in_channels // 3
                    include_amplitude = True
                else:
                    n_channels = in_channels // 2
                    include_amplitude = False
                break
        else:
            n_channels = 256

    # Calculate phase channels
    phase_channels = 3 if include_amplitude else 2

    print(f"Loading {model_name} with {n_channels} EEG channels")
    print(f"  include_amplitude: {include_amplitude} (phase_channels={phase_channels})")

    # Create model
    if model_name == "transformer_autoencoder":
        from omegaconf import OmegaConf
        model_cfg = OmegaConf.create(config.get("model", {}))
        model = TransformerAutoencoder.from_config(model_cfg, n_channels)
    else:
        from omegaconf import OmegaConf
        model_cfg = OmegaConf.create(config.get("model", {}))
        model = ConvLSTMAutoencoder.from_config(model_cfg, n_channels)

    # Load weights
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, n_channels, include_amplitude


def generate_from_random(model, n_channels: int, seq_len: int = 1250,
                         n_samples: int = 5, device: str = "cpu",
                         include_amplitude: bool = False):
    """
    Generate EEG by decoding random latent vectors.

    The latent space is typically learned to be roughly Gaussian,
    so we sample from N(0, 1) and decode.
    """
    print(f"\nGenerating {n_samples} samples from random latent vectors...")

    # Get latent dimension from model
    # We need to figure out the shape by doing a forward pass

    # Calculate input channels: n_channels * (3 if amplitude else 2)
    phase_channels = 3 if include_amplitude else 2
    input_channels = n_channels * phase_channels

    # Create a dummy input to get latent shape
    dummy_input = torch.randn(1, input_channels, seq_len).to(device)

    with torch.no_grad():
        # Use model's encode method if available (handles tuple returns)
        if hasattr(model, 'encode'):
            latent = model.encode(dummy_input)
        else:
            # Fallback for ConvLSTM which returns tensor directly
            encoder_out = model.encoder(dummy_input)
            latent = encoder_out[0] if isinstance(encoder_out, tuple) else encoder_out

        latent_shape = latent.shape[1:]  # Remove batch dim
        print(f"Latent shape: {latent_shape}")

        # Generate random latents
        generated = []
        for i in range(n_samples):
            # Sample from standard normal, scale down slightly for stability
            z = torch.randn(1, *latent_shape).to(device) * 0.5

            # Decode
            output = model.decoder(z)
            generated.append(output.cpu().numpy())

    return np.concatenate(generated, axis=0), phase_channels


def generate_interpolation(model, input_data: np.ndarray, n_steps: int = 10,
                           device: str = "cpu"):
    """
    Interpolate between two EEG samples in latent space.

    Takes first and last chunk from input_data and interpolates between them.
    """
    print(f"\nInterpolating between samples with {n_steps} steps...")

    # Use first and last samples
    x1 = torch.from_numpy(input_data[0:1]).float().to(device)
    x2 = torch.from_numpy(input_data[-1:]).float().to(device)

    with torch.no_grad():
        # Encode both - use encode method if available (handles tuple returns)
        if hasattr(model, 'encode'):
            z1 = model.encode(x1)
            z2 = model.encode(x2)
        else:
            enc1 = model.encoder(x1)
            enc2 = model.encoder(x2)
            z1 = enc1[0] if isinstance(enc1, tuple) else enc1
            z2 = enc2[0] if isinstance(enc2, tuple) else enc2

        # Interpolate
        generated = []
        for alpha in np.linspace(0, 1, n_steps):
            z_interp = (1 - alpha) * z1 + alpha * z2
            output = model.decoder(z_interp)
            generated.append(output.cpu().numpy())

    return np.concatenate(generated, axis=0)


def reconstruct(model, input_data: np.ndarray, device: str = "cpu"):
    """
    Reconstruct EEG through the autoencoder.

    Useful for seeing what the model learns to preserve/discard.
    """
    print(f"\nReconstructing {len(input_data)} samples...")

    x = torch.from_numpy(input_data).float().to(device)

    with torch.no_grad():
        output = model(x)
        # Handle tuple return (reconstruction, latent) from Transformer
        reconstructed = output[0] if isinstance(output, tuple) else output

    return reconstructed.cpu().numpy()


def phase_to_signal(phase_data: np.ndarray, phase_channels: int = 2) -> np.ndarray:
    """
    Convert (cos, sin[, amp]) phase representation back to approximate signal.

    Args:
        phase_data: Shape (batch, n_channels*phase_channels, time)
        phase_channels: 2 for (cos, sin) or 3 for (cos, sin, amp)

    Returns:
        Approximate signal (batch, n_channels, time)
    """
    batch, features, time = phase_data.shape
    n_channels = features // phase_channels

    # Reshape to separate components
    phase_data = phase_data.reshape(batch, n_channels, phase_channels, time)
    cos_phase = phase_data[:, :, 0, :]
    sin_phase = phase_data[:, :, 1, :]

    # Reconstruct phase angle
    phase = np.arctan2(sin_phase, cos_phase)

    if phase_channels == 3:
        # We have amplitude - use it to scale the signal
        amplitude = phase_data[:, :, 2, :]
        # amplitude is log-normalized, so exp to get back to linear scale
        # For visualization, we can use a simplified reconstruction
        signal = amplitude * np.cos(phase)
        return signal
    else:
        # No amplitude - just return phase for visualization
        return phase


def plot_generated(generated: np.ndarray, sfreq: float = 250.0,
                   n_channels_to_plot: int = 8, output_path: Path = None,
                   phase_channels: int = 2):
    """Plot generated EEG samples."""
    n_samples = min(len(generated), 4)
    n_channels = generated.shape[1] // phase_channels

    # Convert phase to signal-like representation
    signals = phase_to_signal(generated[:n_samples], phase_channels=phase_channels)

    fig, axes = plt.subplots(n_samples, 1, figsize=(14, 3 * n_samples))
    if n_samples == 1:
        axes = [axes]

    time = np.arange(signals.shape[2]) / sfreq

    for i, (ax, signal) in enumerate(zip(axes, signals)):
        # Plot a subset of channels
        channels_to_plot = np.linspace(0, n_channels - 1, n_channels_to_plot, dtype=int)

        for j, ch in enumerate(channels_to_plot):
            offset = j * 3  # Vertical offset for visibility
            ax.plot(time, signal[ch] + offset, linewidth=0.5, alpha=0.8)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Channel (offset)")
        ax.set_title(f"Generated Sample {i + 1}")
        ax.set_xlim(0, time[-1])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def plot_reconstruction_comparison(original: np.ndarray, reconstructed: np.ndarray,
                                    sfreq: float = 250.0, output_path: Path = None,
                                    phase_channels: int = 2):
    """Plot original vs reconstructed EEG."""
    # Convert to signals
    orig_signal = phase_to_signal(original[:1], phase_channels=phase_channels)[0]
    recon_signal = phase_to_signal(reconstructed[:1], phase_channels=phase_channels)[0]

    n_channels = orig_signal.shape[0]
    time = np.arange(orig_signal.shape[1]) / sfreq

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # Plot 8 channels
    channels_to_plot = np.linspace(0, n_channels - 1, 8, dtype=int)

    for j, ch in enumerate(channels_to_plot):
        offset = j * 3
        axes[0].plot(time, orig_signal[ch] + offset, linewidth=0.5, alpha=0.8)
        axes[1].plot(time, recon_signal[ch] + offset, linewidth=0.5, alpha=0.8)

    axes[0].set_title("Original EEG (phase)")
    axes[1].set_title("Reconstructed EEG (phase)")
    axes[1].set_xlabel("Time (s)")

    for ax in axes:
        ax.set_ylabel("Channel (offset)")
        ax.set_xlim(0, time[-1])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def load_real_eeg(file_path: Path, chunk_duration: float = 5.0,
                  include_amplitude: bool = False):
    """Load and preprocess real EEG for comparison."""
    import mne
    from eeg_biomarkers.data.preprocessing import preprocess_raw, extract_phase_circular

    print(f"Loading EEG from {file_path}...")

    # Detect file type and load
    if file_path.suffix == ".fif":
        raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
    elif file_path.suffix == ".bdf":
        raw = mne.io.read_raw_bdf(file_path, preload=True, verbose=False)
        # Select only EEG channels
        eeg_picks = mne.pick_types(raw.info, eeg=True)
        raw = raw.pick(eeg_picks)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    # Preprocess
    raw = preprocess_raw(raw, filter_low=1.0, filter_high=48.0, reference="average")

    # Get data and extract phase
    data = raw.get_data()
    sfreq = raw.info["sfreq"]

    # Extract circular phase (optionally with amplitude)
    phase_data = extract_phase_circular(data, include_amplitude=include_amplitude)
    # Shape: (n_channels, phase_channels, n_samples) -> reshape to (n_channels*phase_channels, n_samples)
    n_channels = phase_data.shape[0]
    phase_channels = phase_data.shape[1]
    phase_data = phase_data.reshape(n_channels * phase_channels, -1)

    # Chunk into segments
    chunk_samples = int(chunk_duration * sfreq)
    n_chunks = phase_data.shape[1] // chunk_samples

    chunks = []
    for i in range(n_chunks):
        start = i * chunk_samples
        end = start + chunk_samples
        chunks.append(phase_data[:, start:end])

    return np.array(chunks), sfreq, phase_channels


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic EEG from autoencoder")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--mode", choices=["random", "interpolate", "reconstruct"],
                        default="random", help="Generation mode")
    parser.add_argument("--input-file", type=Path,
                        help="Input EEG file (required for interpolate/reconstruct)")
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Number of samples to generate (random mode)")
    parser.add_argument("--n-steps", type=int, default=10,
                        help="Number of interpolation steps")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path for plot (default: show)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu, cuda, mps)")
    parser.add_argument("--seq-len", type=int, default=1250,
                        help="Sequence length (samples) for random generation")

    args = parser.parse_args()

    # Load model
    model, n_channels, include_amplitude = load_model(args.checkpoint, args.device)
    phase_channels = 3 if include_amplitude else 2
    print(f"Model loaded: {n_channels} EEG channels, phase_channels={phase_channels}")

    if args.mode == "random":
        # Generate from random latent vectors
        generated, phase_channels = generate_from_random(
            model, n_channels,
            seq_len=args.seq_len,
            n_samples=args.n_samples,
            device=args.device,
            include_amplitude=include_amplitude
        )
        plot_generated(generated, output_path=args.output, phase_channels=phase_channels)

    elif args.mode == "interpolate":
        if args.input_file is None:
            parser.error("--input-file required for interpolate mode")

        # Load real EEG (must match model's amplitude setting)
        real_data, sfreq, phase_channels = load_real_eeg(
            args.input_file, include_amplitude=include_amplitude
        )

        # Interpolate
        generated = generate_interpolation(
            model, real_data,
            n_steps=args.n_steps,
            device=args.device
        )
        plot_generated(generated, sfreq=sfreq, output_path=args.output,
                      phase_channels=phase_channels)

    elif args.mode == "reconstruct":
        if args.input_file is None:
            parser.error("--input-file required for reconstruct mode")

        # Load real EEG (must match model's amplitude setting)
        real_data, sfreq, phase_channels = load_real_eeg(
            args.input_file, include_amplitude=include_amplitude
        )

        # Reconstruct
        reconstructed = reconstruct(model, real_data, device=args.device)
        plot_reconstruction_comparison(
            real_data, reconstructed,
            sfreq=sfreq, output_path=args.output,
            phase_channels=phase_channels
        )


if __name__ == "__main__":
    main()
