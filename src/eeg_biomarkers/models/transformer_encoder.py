"""Temporal Transformer Encoder for EEG phase data."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from omegaconf import DictConfig


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer.

    Adds positional information to input embeddings so the transformer
    can learn temporal relationships.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalTransformerEncoder(nn.Module):
    """
    Temporal Transformer encoder for EEG phase representations.

    Architecture:
    1. Conv1D layers for initial feature extraction (same as ConvLSTM)
    2. Transformer encoder with self-attention
    3. Learns long-range temporal dependencies better than LSTM

    Args:
        n_channels: Number of EEG channels
        hidden_size: Transformer/output dimension
        complexity: Network depth (0-3, controls conv layers)
        dropout: Dropout probability
        phase_channels: 2 for cos/sin, 3 if including amplitude
        n_heads: Number of attention heads
        n_transformer_layers: Number of transformer encoder layers
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
        self.complexity = complexity
        self.phase_channels = phase_channels
        self.n_heads = n_heads

        # Input dimension: n_channels * phase_channels
        in_channels = n_channels * phase_channels

        # Build convolutional layers based on complexity (same as ConvLSTM)
        conv_layers = []
        if complexity == 0:
            conv_layers.append(self._conv_block(in_channels, 32, dropout))
            conv_output = 32
        elif complexity == 1:
            conv_layers.append(self._conv_block(in_channels, 32, dropout))
            conv_layers.append(self._conv_block(32, 64, dropout))
            conv_output = 64
        elif complexity == 2:
            conv_layers.append(self._conv_block(in_channels, 32, dropout))
            conv_layers.append(self._conv_block(32, 64, dropout))
            conv_layers.append(self._conv_block(64, 128, dropout))
            conv_output = 128
        else:  # complexity >= 3
            conv_layers.append(self._conv_block(in_channels, 64, dropout))
            conv_layers.append(self._conv_block(64, 128, dropout))
            conv_layers.append(self._conv_block(128, 256, dropout))
            conv_layers.append(self._conv_block(256, 512, dropout))
            conv_output = 512

        self.conv_layers = nn.Sequential(*conv_layers)
        self.conv_output = conv_output

        # Project conv output to transformer dimension
        self.input_projection = nn.Linear(conv_output, hidden_size)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size, max_len=5000, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',  # GELU often works better than ReLU for transformers
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_transformer_layers,
        )

        # Optional: Layer norm at output
        self.output_norm = nn.LayerNorm(hidden_size)

    def _conv_block(
        self, in_channels: int, out_channels: int, dropout: float
    ) -> nn.Sequential:
        """Create a convolutional block with BatchNorm, GELU, and Dropout."""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),  # GELU instead of ReLU for smoother gradients
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        """
        Forward pass through encoder.

        Args:
            x: Input tensor (batch, channels, time)
               where channels = n_channels * phase_channels

        Returns:
            latent: Latent representation (batch, time', hidden_size)
            hidden: None (for API compatibility with LSTM encoder)
        """
        # Convolutional feature extraction
        # (batch, in_channels, time) -> (batch, conv_out, time)
        conv_out = self.conv_layers(x)

        # Transpose for transformer: (batch, time, features)
        conv_out = conv_out.transpose(1, 2)

        # Project to transformer dimension
        x = self.input_projection(conv_out)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding with self-attention
        latent = self.transformer_encoder(x)

        # Output normalization
        latent = self.output_norm(latent)

        return latent, None  # None for hidden state (LSTM compatibility)

    @classmethod
    def from_config(cls, cfg: DictConfig, n_channels: int) -> "TemporalTransformerEncoder":
        """Create encoder from Hydra config."""
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
