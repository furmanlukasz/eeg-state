"""Temporal Transformer Autoencoder for EEG phase representation learning."""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from omegaconf import DictConfig

from eeg_biomarkers.models.transformer_encoder import TemporalTransformerEncoder
from eeg_biomarkers.models.transformer_decoder import TemporalTransformerDecoder


class TransformerAutoencoder(nn.Module):
    """
    Transformer-based autoencoder for learning EEG phase representations.

    Key improvements over ConvLSTM version:
    - Self-attention captures long-range temporal dependencies
    - Better gradient flow (no vanishing gradients like LSTM)
    - Parallel processing of sequence (faster training)

    Supports:
    - Circular (cos, sin) phase representation
    - Optional amplitude channel
    - Angular distance matrices for RQA

    Args:
        n_channels: Number of EEG channels
        hidden_size: Latent/transformer dimension
        complexity: Model depth (0-3, controls conv layers)
        dropout: Dropout probability
        phase_channels: 2 for (cos, sin), 3 if including amplitude
        n_heads: Number of attention heads
        n_transformer_layers: Number of transformer layers
        dim_feedforward: Feedforward dimension in transformer
    """

    def __init__(
        self,
        n_channels: int,
        hidden_size: int = 64,
        complexity: int = 2,
        dropout: float = 0.1,
        phase_channels: int = 2,
        n_heads: int = 4,
        n_transformer_layers: int = 2,
        dim_feedforward: int = 256,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.hidden_size = hidden_size
        self.phase_channels = phase_channels
        self.n_heads = n_heads
        self.n_transformer_layers = n_transformer_layers

        self.encoder = TemporalTransformerEncoder(
            n_channels=n_channels,
            hidden_size=hidden_size,
            complexity=complexity,
            dropout=dropout,
            phase_channels=phase_channels,
            n_heads=n_heads,
            n_transformer_layers=n_transformer_layers,
            dim_feedforward=dim_feedforward,
        )

        self.decoder = TemporalTransformerDecoder(
            n_channels=n_channels,
            hidden_size=hidden_size,
            complexity=complexity,
            dropout=dropout,
            phase_channels=phase_channels,
            n_heads=n_heads,
            n_transformer_layers=n_transformer_layers,
            dim_feedforward=dim_feedforward,
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
        latent, _ = self.encoder(x)
        reconstruction = self.decoder(latent)
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

        Uses cosine similarity converted to angular distance.

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

        # Clamp and compute angular distance using numpy for safety
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
        dist_matrix = self.compute_angular_distance_matrix(latent)

        if threshold_method == "rr_controlled":
            epsilon = self._get_rr_controlled_threshold(dist_matrix, target_rr)
        elif threshold_method == "fixed":
            if fixed_epsilon is None:
                raise ValueError("fixed_epsilon required for fixed threshold method")
            epsilon = fixed_epsilon
        else:
            raise ValueError(f"Unknown threshold method: {threshold_method}")

        recurrence = (dist_matrix <= epsilon).float()
        return recurrence, epsilon

    def _get_rr_controlled_threshold(
        self, distance_matrix: torch.Tensor, target_rr: float
    ) -> float:
        """Find threshold that yields target recurrence rate."""
        n = distance_matrix.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=distance_matrix.device)
        off_diag = distance_matrix[mask]
        epsilon = torch.quantile(off_diag, target_rr).item()
        return epsilon

    @classmethod
    def from_config(cls, cfg: DictConfig, n_channels: int) -> "TransformerAutoencoder":
        """Create autoencoder from Hydra config."""
        return cls(
            n_channels=n_channels,
            hidden_size=cfg.encoder.hidden_size,
            complexity=cfg.encoder.complexity,
            dropout=cfg.encoder.dropout,
            phase_channels=3 if cfg.phase.include_amplitude else 2,
            n_heads=getattr(cfg.encoder, 'n_heads', 4),
            n_transformer_layers=getattr(cfg.encoder, 'n_transformer_layers', 2),
            dim_feedforward=getattr(cfg.encoder, 'dim_feedforward', 256),
        )

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path: str, n_channels: int, **kwargs
    ) -> "TransformerAutoencoder":
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint file
            n_channels: Number of EEG channels
            **kwargs: Additional model arguments

        Returns:
            Loaded model
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Extract hyperparameters from checkpoint if available
        if "config" in checkpoint:
            cfg = checkpoint["config"]
            model_cfg = cfg.get("model", {})
            encoder_cfg = model_cfg.get("encoder", {})
            phase_cfg = model_cfg.get("phase", {})

            model = cls(
                n_channels=n_channels,
                hidden_size=encoder_cfg.get("hidden_size", kwargs.get("hidden_size", 64)),
                complexity=encoder_cfg.get("complexity", kwargs.get("complexity", 2)),
                dropout=encoder_cfg.get("dropout", kwargs.get("dropout", 0.1)),
                phase_channels=3 if phase_cfg.get("include_amplitude", False) else 2,
                n_heads=encoder_cfg.get("n_heads", kwargs.get("n_heads", 4)),
                n_transformer_layers=encoder_cfg.get("n_transformer_layers", kwargs.get("n_transformer_layers", 2)),
                dim_feedforward=encoder_cfg.get("dim_feedforward", kwargs.get("dim_feedforward", 256)),
            )
        else:
            model = cls(n_channels=n_channels, **kwargs)

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)

        return model
