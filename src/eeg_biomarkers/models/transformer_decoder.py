"""Temporal Transformer Decoder for EEG phase reconstruction."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from omegaconf import DictConfig


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (same as encoder)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        # Handle both even and odd d_model
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        # Scale down to prevent large values dominating input
        pe = pe * 0.1
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalTransformerDecoder(nn.Module):
    """
    Temporal Transformer decoder for EEG phase reconstruction.

    Mirrors the encoder architecture with transposed convolutions.

    Output constraints:
    - cos/sin channels: tanh activation to keep in [-1, 1]
    - amplitude channel (if present): linear (z-scored amplitude can be any value)

    Args:
        n_channels: Number of EEG channels
        hidden_size: Transformer dimension (must match encoder)
        complexity: Network depth (0-3, must match encoder)
        dropout: Dropout probability
        phase_channels: 2 for cos/sin, 3 if including amplitude
        n_heads: Number of attention heads
        n_transformer_layers: Number of transformer decoder layers
        dim_feedforward: Feedforward dimension in transformer
        constrain_output: Whether to apply tanh to cos/sin outputs
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
        constrain_output: bool = True,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.hidden_size = hidden_size
        self.complexity = complexity
        self.phase_channels = phase_channels
        self.constrain_output = constrain_output

        # Output dimension: n_channels * phase_channels
        out_channels = n_channels * phase_channels

        # Determine conv input size to match encoder's conv output
        if complexity == 0:
            conv_input = 32
        elif complexity == 1:
            conv_input = 64
        elif complexity == 2:
            conv_input = 128
        else:
            conv_input = 512

        self.conv_input = conv_input

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size, max_len=5000, dropout=dropout)

        # Transformer decoder (using encoder architecture - no cross-attention needed)
        # We use TransformerEncoder here because we're doing autoencoding, not seq2seq
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer_decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=n_transformer_layers,
        )

        # Project from transformer dimension to conv input
        self.output_projection = nn.Linear(hidden_size, conv_input)

        # Build transposed convolutional layers (reverse of encoder)
        deconv_layers = []
        if complexity == 0:
            deconv_layers.append(self._deconv_block(32, out_channels, dropout, final=True))
        elif complexity == 1:
            deconv_layers.append(self._deconv_block(64, 32, dropout))
            deconv_layers.append(self._deconv_block(32, out_channels, dropout, final=True))
        elif complexity == 2:
            deconv_layers.append(self._deconv_block(128, 64, dropout))
            deconv_layers.append(self._deconv_block(64, 32, dropout))
            deconv_layers.append(self._deconv_block(32, out_channels, dropout, final=True))
        else:  # complexity >= 3
            deconv_layers.append(self._deconv_block(512, 256, dropout))
            deconv_layers.append(self._deconv_block(256, 128, dropout))
            deconv_layers.append(self._deconv_block(128, 64, dropout))
            deconv_layers.append(self._deconv_block(64, out_channels, dropout, final=True))

        self.deconv_layers = nn.Sequential(*deconv_layers)

        # Pre-transformer layer norm for stability
        self.pre_transformer_norm = nn.LayerNorm(hidden_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier/Kaiming for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _deconv_block(
        self, in_channels: int, out_channels: int, dropout: float, final: bool = False
    ) -> nn.Sequential:
        """Create a transposed convolutional block."""
        layers = [
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, padding=1),
        ]
        if not final:
            layers.extend([
                nn.BatchNorm1d(out_channels),
                nn.GELU(),  # GELU to match encoder
                nn.Dropout(dropout),
            ])
        return nn.Sequential(*layers)

    def forward(
        self,
        latent: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through decoder.

        Args:
            latent: Latent representation (batch, time', hidden_size)
            hidden: Not used (for API compatibility)

        Returns:
            reconstruction: Reconstructed signal (batch, channels, time)
        """
        # Pre-transformer layer norm for stability
        x = self.pre_transformer_norm(latent)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer decoding
        x = self.transformer_decoder(x)

        # Project to conv input dimension
        x = self.output_projection(x)

        # Transpose for convolutions: (batch, features, time)
        x = x.transpose(1, 2)

        # Transposed convolutions for reconstruction
        reconstruction = self.deconv_layers(x)

        # Apply output constraints if enabled
        if self.constrain_output:
            batch_size, total_channels, time = reconstruction.shape

            # Reshape to (batch, n_channels, phase_channels, time)
            reconstruction = reconstruction.view(batch_size, self.n_channels, self.phase_channels, time)

            # Apply tanh to cos/sin channels (indices 0 and 1)
            reconstruction[:, :, 0, :] = torch.tanh(reconstruction[:, :, 0, :])  # cos
            reconstruction[:, :, 1, :] = torch.tanh(reconstruction[:, :, 1, :])  # sin
            # Leave amplitude (index 2, if present) unconstrained

            # Reshape back to (batch, n_channels * phase_channels, time)
            reconstruction = reconstruction.view(batch_size, total_channels, time)

        return reconstruction

    @classmethod
    def from_config(cls, cfg: DictConfig, n_channels: int) -> "TemporalTransformerDecoder":
        """Create decoder from Hydra config."""
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
