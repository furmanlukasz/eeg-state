"""ConvLSTM Encoder for EEG phase data."""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


class ConvLSTMEncoder(nn.Module):
    """
    Convolutional LSTM encoder for EEG phase representations.

    Architecture scales with complexity parameter:
    - complexity=0: 1 Conv1d (32) → LSTM
    - complexity=1: 2 Conv1d (32, 64) → LSTM
    - complexity=2: 3 Conv1d (32, 64, 128) → LSTM
    - complexity=3: 4 Conv1d (64, 128, 256, 512) → 2 LSTMs → FC

    Args:
        n_channels: Number of EEG channels (input dimension per time step)
        hidden_size: LSTM hidden dimension
        complexity: Network depth (0-3)
        dropout: Dropout probability
        phase_channels: Number of channels per EEG channel (2 for cos/sin, 3 if amplitude)
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
        self.complexity = complexity
        self.phase_channels = phase_channels

        # Input dimension: n_channels * phase_channels (e.g., 256 * 2 for cos/sin)
        in_channels = n_channels * phase_channels

        # Build convolutional layers based on complexity
        conv_layers = []
        if complexity == 0:
            conv_layers.append(self._conv_block(in_channels, 32, dropout))
            lstm_input = 32
        elif complexity == 1:
            conv_layers.append(self._conv_block(in_channels, 32, dropout))
            conv_layers.append(self._conv_block(32, 64, dropout))
            lstm_input = 64
        elif complexity == 2:
            conv_layers.append(self._conv_block(in_channels, 32, dropout))
            conv_layers.append(self._conv_block(32, 64, dropout))
            conv_layers.append(self._conv_block(64, 128, dropout))
            lstm_input = 128
        else:  # complexity >= 3
            conv_layers.append(self._conv_block(in_channels, 64, dropout))
            conv_layers.append(self._conv_block(64, 128, dropout))
            conv_layers.append(self._conv_block(128, 256, dropout))
            conv_layers.append(self._conv_block(256, 512, dropout))
            lstm_input = 512

        self.conv_layers = nn.Sequential(*conv_layers)

        # LSTM layers
        n_lstm_layers = 2 if complexity >= 3 else 1
        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=hidden_size,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )

        # Optional projection layer for complex architectures
        self.projection = None
        if complexity >= 3:
            self.projection = nn.Linear(hidden_size, hidden_size)

    def _conv_block(
        self, in_channels: int, out_channels: int, dropout: float
    ) -> nn.Sequential:
        """Create a convolutional block with BatchNorm, ReLU, and Dropout."""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.

        Args:
            x: Input tensor of shape (batch, channels, time)
               where channels = n_channels * phase_channels

        Returns:
            latent: Latent representation (batch, time', hidden_size)
            hidden: Final LSTM hidden state for decoder initialization
        """
        # Convolutional feature extraction
        # (batch, in_channels, time) -> (batch, conv_out, time)
        conv_out = self.conv_layers(x)

        # Transpose for LSTM: (batch, time, features)
        conv_out = conv_out.transpose(1, 2)

        # LSTM encoding
        latent, (h_n, c_n) = self.lstm(conv_out)

        # Optional projection
        if self.projection is not None:
            latent = self.projection(latent)

        return latent, (h_n, c_n)

    @classmethod
    def from_config(cls, cfg: DictConfig, n_channels: int) -> ConvLSTMEncoder:
        """Create encoder from Hydra config."""
        return cls(
            n_channels=n_channels,
            hidden_size=cfg.encoder.hidden_size,
            complexity=cfg.encoder.complexity,
            dropout=cfg.encoder.dropout,
            phase_channels=3 if cfg.phase.include_amplitude else 2,
        )
