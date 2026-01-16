"""ConvLSTM Autoencoder for EEG phase representation learning."""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from omegaconf import DictConfig

from eeg_biomarkers.models.encoder import ConvLSTMEncoder
from eeg_biomarkers.models.decoder import ConvLSTMDecoder


class ConvLSTMAutoencoder(nn.Module):
    """
    Autoencoder for learning compressed representations of EEG phase data.

    Key features:
    - Uses circular (cos, sin) representation for phase (not raw angles)
    - Computes angular distance matrices in latent space
    - Supports RR-controlled thresholding for recurrence matrices

    Args:
        n_channels: Number of EEG channels
        hidden_size: Latent dimension
        complexity: Model depth (0-3)
        dropout: Dropout probability
        phase_channels: 2 for (cos, sin), 3 if including amplitude
    """

    def __init__(
        self,
        n_channels: int,
        hidden_size: int = 64,
        complexity: int = 2,
        dropout: float = 0.1,
        phase_channels: int = 2,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.hidden_size = hidden_size
        self.phase_channels = phase_channels

        self.encoder = ConvLSTMEncoder(
            n_channels=n_channels,
            hidden_size=hidden_size,
            complexity=complexity,
            dropout=dropout,
            phase_channels=phase_channels,
        )

        self.decoder = ConvLSTMDecoder(
            n_channels=n_channels,
            hidden_size=hidden_size,
            complexity=complexity,
            dropout=dropout,
            phase_channels=phase_channels,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode then decode.

        Args:
            x: Input (batch, n_channels * phase_channels, time)

        Returns:
            reconstruction: Reconstructed signal (same shape as input)
            latent: Latent representation (batch, time', hidden_size)
        """
        latent, hidden = self.encoder(x)
        reconstruction = self.decoder(latent, hidden)
        return reconstruction, latent

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.

        Args:
            x: Input (batch, n_channels * phase_channels, time)

        Returns:
            latent: Latent trajectory (batch, time', hidden_size)
        """
        latent, _ = self.encoder(x)
        return latent

    def compute_angular_distance_matrix(
        self, latent: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Compute angular distance matrix from latent trajectory.

        Uses cosine similarity converted to angular distance, which is
        more appropriate for the circular nature of phase data.

        Args:
            latent: Latent trajectory (time', hidden_size) for single sample
            eps: Small constant for numerical stability

        Returns:
            distance_matrix: Angular distances (time', time')
        """
        # Normalize latent vectors
        norms = torch.norm(latent, dim=1, keepdim=True)
        norms = torch.clamp(norms, min=eps)
        normalized = latent / norms

        # Cosine similarity matrix
        cos_sim = torch.mm(normalized, normalized.t())

        # Clamp to [-1+eps, 1-eps] for numerical stability
        # torch.acos can segfault on some hardware even with valid inputs
        # Use numpy for safety (this is not in training loop anyway)
        cos_sim_np = cos_sim.detach().cpu().numpy()
        cos_sim_np = np.clip(cos_sim_np, -1.0 + 1e-7, 1.0 - 1e-7)
        angular_dist_np = np.arccos(cos_sim_np)
        angular_dist = torch.from_numpy(angular_dist_np).to(latent.device)

        return angular_dist

    def compute_recurrence_matrix(
        self,
        latent: torch.Tensor,
        threshold_method: str = "rr_controlled",
        target_rr: float = 0.02,
        fixed_epsilon: float | None = None,
    ) -> tuple[torch.Tensor, float]:
        """
        Compute recurrence matrix from latent trajectory.

        Args:
            latent: Latent trajectory (time', hidden_size)
            threshold_method: "rr_controlled" or "fixed"
            target_rr: Target recurrence rate (for rr_controlled)
            fixed_epsilon: Fixed threshold (for fixed method)

        Returns:
            recurrence_matrix: Binary recurrence matrix (time', time')
            epsilon: Threshold used
        """
        # Compute angular distance matrix
        dist_matrix = self.compute_angular_distance_matrix(latent)

        # Determine threshold
        if threshold_method == "rr_controlled":
            epsilon = self._get_rr_controlled_threshold(dist_matrix, target_rr)
        elif threshold_method == "fixed":
            if fixed_epsilon is None:
                raise ValueError("fixed_epsilon required for fixed threshold method")
            epsilon = fixed_epsilon
        else:
            raise ValueError(f"Unknown threshold method: {threshold_method}")

        # Create binary recurrence matrix
        recurrence = (dist_matrix <= epsilon).float()

        return recurrence, epsilon

    def _get_rr_controlled_threshold(
        self, distance_matrix: torch.Tensor, target_rr: float
    ) -> float:
        """
        Find threshold that yields target recurrence rate.

        Args:
            distance_matrix: Angular distance matrix (T, T)
            target_rr: Target recurrence rate (e.g., 0.02 for 2%)

        Returns:
            epsilon: Threshold value
        """
        n = distance_matrix.shape[0]

        # Exclude diagonal (self-recurrence)
        mask = ~torch.eye(n, dtype=torch.bool, device=distance_matrix.device)
        off_diag = distance_matrix[mask]

        # Find threshold at target percentile
        percentile = target_rr * 100
        epsilon = torch.quantile(off_diag, target_rr).item()

        return epsilon

    @classmethod
    def from_config(cls, cfg: DictConfig, n_channels: int) -> ConvLSTMAutoencoder:
        """Create autoencoder from Hydra config."""
        return cls(
            n_channels=n_channels,
            hidden_size=cfg.encoder.hidden_size,
            complexity=cfg.encoder.complexity,
            dropout=cfg.encoder.dropout,
            phase_channels=3 if cfg.phase.include_amplitude else 2,
        )

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path: str, n_channels: int, **kwargs
    ) -> ConvLSTMAutoencoder:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to .pth checkpoint file
            n_channels: Number of EEG channels
            **kwargs: Additional model arguments

        Returns:
            Loaded model
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Extract hyperparameters from checkpoint if available
        if "hyperparameters" in checkpoint:
            hp = checkpoint["hyperparameters"]
            model = cls(
                n_channels=n_channels,
                hidden_size=hp.get("hidden_size", kwargs.get("hidden_size", 64)),
                complexity=hp.get("complexity", kwargs.get("complexity", 2)),
            )
        else:
            model = cls(n_channels=n_channels, **kwargs)

        # Load state dict
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)

        return model
