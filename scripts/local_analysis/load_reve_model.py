"""
REVE (foundation EEG model) Loading Utilities for Comparison Analysis

REVE is a pretrained masked autoencoder for EEG that uses:
- Temporal patching (each channel is split into time patches)
- 4D spatio-temporal positional encoding (x, y, z, t coordinates)
- Transformer encoder

Key shape difference from our trained model:
- Our model: (batch, features, time) -> latent (batch, time, hidden)
- REVE: (batch, channels, time) -> tokens (batch, channels, patches, hidden)

To get comparable trajectories, we pool across channels:
    trajectory = mean(tokens, axis=channels) -> (patches, hidden)

Reference: https://brain-bzh.github.io/reve/
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np
import torch
import warnings

# Suppress HuggingFace warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class REVEEncoder:
    """
    Wrapper for REVE model to extract latent trajectories.

    Usage:
        encoder = REVEEncoder(device="mps")
        trajectory = encoder.compute_trajectory(eeg_data, channel_names)
        # trajectory shape: (n_patches, hidden_dim)
    """

    # REVE expects 200 Hz sampling rate
    EXPECTED_SFREQ = 200.0

    # Standard 10-20 electrode names that REVE recognizes
    STANDARD_10_20 = [
        "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
        "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
        "A1", "A2", "T7", "T8", "P7", "P8", "FC1", "FC2", "CP1", "CP2",
        "FC5", "FC6", "CP5", "CP6", "TP9", "TP10", "POz", "Oz", "Iz",
        "AF3", "AF4", "F1", "F2", "FC3", "FC4", "C1", "C2", "CP3", "CP4",
        "P1", "P2", "PO3", "PO4", "PO7", "PO8", "O9", "O10",
    ]

    def __init__(
        self,
        device: str = "mps",
        model_name: str = "brain-bzh/reve-base",
        positions_name: str = "brain-bzh/reve-positions",
    ):
        """
        Initialize REVE encoder.

        Args:
            device: Device to run on ("mps", "cuda", "cpu")
            model_name: HuggingFace model name
            positions_name: HuggingFace positions bank name
        """
        self.device = device
        self.model_name = model_name
        self.positions_name = positions_name

        self.model = None
        self.pos_bank = None
        self._loaded = False

        # Cached positions for channel sets
        self._positions_cache = {}

    def _load_models(self):
        """Lazy load REVE model and positions bank."""
        if self._loaded:
            return

        from transformers import AutoModel

        # Get token for gated model access
        token = get_hf_token()
        if token:
            print(f"Using HuggingFace token for authentication")

        print(f"Loading REVE model: {self.model_name}")
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            token=token,
        )
        self.model.to(self.device)
        self.model.eval()

        print(f"Loading REVE positions: {self.positions_name}")
        self.pos_bank = AutoModel.from_pretrained(
            self.positions_name,
            trust_remote_code=True,
            token=token,
        )

        # Get hidden dimension from model config (REVE uses embed_dim, not hidden_size)
        self.hidden_dim = getattr(self.model.config, 'embed_dim', 512)
        print(f"  Hidden dimension: {self.hidden_dim}")
        print(f"  Available positions: {len(self.pos_bank.get_all_positions())} electrodes")

        self._loaded = True

    def _get_positions(self, channel_names: list) -> torch.Tensor:
        """
        Get 3D positions for channel names.

        Args:
            channel_names: List of electrode names

        Returns:
            Tensor of shape (n_channels, 3) with (x, y, z) positions
        """
        self._load_models()

        # Check cache
        key = tuple(channel_names)
        if key in self._positions_cache:
            return self._positions_cache[key]

        # Get available positions
        available = set(self.pos_bank.get_all_positions())

        # Map channel names to standard names
        mapped_names = []
        valid_indices = []

        for i, name in enumerate(channel_names):
            # Try exact match first
            if name in available:
                mapped_names.append(name)
                valid_indices.append(i)
            # Try common variations
            elif name.upper() in available:
                mapped_names.append(name.upper())
                valid_indices.append(i)
            elif name.replace("-", "") in available:
                mapped_names.append(name.replace("-", ""))
                valid_indices.append(i)
            # Try stripping numbers (e.g., "EEG 001" -> check if maps to standard name)
            else:
                # Skip unrecognized channels
                pass

        if len(mapped_names) == 0:
            raise ValueError(
                f"No channel names could be mapped to REVE positions. "
                f"First 10 channels: {channel_names[:10]}. "
                f"Available positions: {sorted(list(available))[:20]}"
            )

        positions = self.pos_bank(mapped_names)  # (n_valid, 3)

        self._positions_cache[key] = (positions, valid_indices)
        return positions, valid_indices

    def _resample(self, data: np.ndarray, source_sfreq: float) -> np.ndarray:
        """
        Resample data to REVE's expected 200 Hz.

        Args:
            data: (n_channels, n_samples) array
            source_sfreq: Source sampling frequency

        Returns:
            Resampled data at 200 Hz
        """
        if abs(source_sfreq - self.EXPECTED_SFREQ) < 1.0:
            return data

        from scipy.signal import resample

        n_channels, n_samples = data.shape
        target_samples = int(n_samples * self.EXPECTED_SFREQ / source_sfreq)

        resampled = resample(data, target_samples, axis=1)
        return resampled

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data as REVE expects (session-level z-score).

        Args:
            data: (n_channels, n_samples) array

        Returns:
            Normalized data
        """
        # Z-score per channel (session-level normalization)
        mean = data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, keepdims=True)
        std = np.where(std == 0, 1, std)  # Avoid division by zero

        normalized = (data - mean) / std

        # Clip extreme values (as per REVE preprocessing)
        normalized = np.clip(normalized, -10, 10)

        return normalized

    def compute_trajectory(
        self,
        eeg_data: np.ndarray,
        channel_names: list,
        sfreq: float = 250.0,
        pooling: str = "mean",
    ) -> np.ndarray:
        """
        Compute latent trajectory from raw EEG data.

        This is the main interface matching compute_latent_trajectory() signature.

        Args:
            eeg_data: (n_channels, n_samples) raw EEG data
            channel_names: List of channel names
            sfreq: Sampling frequency of input data
            pooling: How to pool across channels ("mean", "max", "flatten")

        Returns:
            trajectory: (n_patches, hidden_dim) latent trajectory over time patches
        """
        self._load_models()

        # Get valid channels and their positions
        positions, valid_indices = self._get_positions(channel_names)
        n_valid = len(valid_indices)

        # Select only valid channels
        valid_data = eeg_data[valid_indices, :]

        # Preprocess: resample to 200 Hz
        resampled = self._resample(valid_data, sfreq)

        # Normalize
        normalized = self._normalize(resampled)

        # Convert to tensor: (1, n_channels, n_samples)
        x = torch.from_numpy(normalized).float().unsqueeze(0).to(self.device)

        # Expand positions for batch: (1, n_channels, 3)
        pos = positions.unsqueeze(0).to(self.device)

        # Forward pass through REVE
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda" if self.device == "cuda" else "cpu", dtype=torch.float16):
                # REVE output shape: (batch, channels, patches, hidden)
                output = self.model(x, pos)

        # output shape: [1, C, P, D]
        output = output.squeeze(0)  # [C, P, D]

        # Pool across channels to get trajectory
        if pooling == "mean":
            trajectory = output.mean(dim=0)  # [P, D]
        elif pooling == "max":
            trajectory = output.max(dim=0)[0]  # [P, D]
        elif pooling == "flatten":
            # Flatten channel and patch dims, then apply PCA later
            trajectory = output.reshape(-1, output.shape[-1])  # [C*P, D]
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")

        return trajectory.cpu().float().numpy()

    def get_model_info(self) -> dict:
        """Get model info dict matching our model interface."""
        self._load_models()

        return {
            "hidden_size": self.hidden_dim,
            "is_transformer": True,
            "is_reve": True,
            "model_name": self.model_name,
            "include_amplitude": False,  # REVE uses raw signal, not phase
            "phase_channels": 0,  # N/A for REVE
        }


def create_reve_trajectory_fn(device: str = "mps"):
    """
    Create a trajectory function compatible with full_dataset_analysis.

    Returns a function with signature:
        compute_trajectory(model, phase_data, device) -> trajectory

    For REVE, we ignore 'model' and 'phase_data' and use raw EEG instead.
    """
    encoder = REVEEncoder(device=device)

    def compute_trajectory(
        raw_data: np.ndarray,
        channel_names: list,
        sfreq: float = 250.0,
    ) -> np.ndarray:
        """
        Compute REVE trajectory from raw EEG.

        Note: This has a DIFFERENT interface than compute_latent_trajectory()
        because REVE needs raw data, not phase.
        """
        return encoder.compute_trajectory(raw_data, channel_names, sfreq)

    return compute_trajectory, encoder


def get_hf_token() -> Optional[str]:
    """Get HuggingFace token from environment or cache."""
    import os

    # Try environment variable first
    token = os.environ.get("HF_ACCESS_TOKEN") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token

    # Try huggingface-cli cached token
    try:
        from huggingface_hub import HfFolder
        return HfFolder.get_token()
    except Exception:
        return None


def check_reve_access() -> bool:
    """
    Check if user has access to REVE model on HuggingFace.

    Returns True if access is granted, False otherwise with helpful message.
    """
    try:
        from huggingface_hub import HfApi

        token = get_hf_token()
        api = HfApi(token=token)

        # Try to get model info - this will fail if no access
        api.model_info("brain-bzh/reve-base")
        return True

    except Exception as e:
        error_str = str(e).lower()

        if "gated" in error_str or "403" in error_str:
            print("=" * 70)
            print("REVE ACCESS REQUIRED")
            print("=" * 70)
            print()
            print("REVE is a gated model that requires approval from the authors.")
            print()
            print("Steps to get access:")
            print("1. Go to: https://huggingface.co/brain-bzh/reve-base")
            print("2. Click 'Request access' and fill out the form")
            print("3. Wait for approval (usually quick)")
            print("4. Set HF_ACCESS_TOKEN env var or run: huggingface-cli login")
            print("5. Re-run this script")
            print()
            print("=" * 70)
            return False

        elif "login" in error_str or "token" in error_str:
            print("=" * 70)
            print("HUGGINGFACE LOGIN REQUIRED")
            print("=" * 70)
            print()
            print("Please authenticate with HuggingFace:")
            print("  export HF_ACCESS_TOKEN=your_token_here")
            print("  # or")
            print("  huggingface-cli login")
            print()
            print("=" * 70)
            return False

        else:
            print(f"Unknown error checking REVE access: {e}")
            return False


if __name__ == "__main__":
    # Quick test
    import sys

    print("Testing REVE encoder loading...")
    print()

    # First check access
    if not check_reve_access():
        sys.exit(1)

    try:
        encoder = REVEEncoder(device="mps")

        # Test with synthetic data matching 10-20 montage
        test_channels = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
        test_data = np.random.randn(len(test_channels), 1000)  # 5 seconds at 200 Hz

        print(f"\nTest data shape: {test_data.shape}")
        print(f"Test channels: {test_channels}")

        trajectory = encoder.compute_trajectory(test_data, test_channels, sfreq=200.0)

        print(f"\nSUCCESS!")
        print(f"Trajectory shape: {trajectory.shape}")
        print(f"  (patches, hidden_dim)")

        info = encoder.get_model_info()
        print(f"\nModel info: {info}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
