"""ConvLSTM Decoder for EEG phase reconstruction."""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


class ConvLSTMDecoder(nn.Module):
    """
    Convolutional LSTM decoder for EEG phase reconstruction.

    Mirrors the encoder architecture with transposed convolutions.

    Output constraints:
    - cos/sin channels: tanh activation to keep in [-1, 1]
    - amplitude channel (if present): linear (z-scored amplitude can be any value)

    Args:
        n_channels: Number of EEG channels (output dimension per time step)
        hidden_size: LSTM hidden dimension (must match encoder)
        complexity: Network depth (0-3, must match encoder)
        dropout: Dropout probability
        phase_channels: Number of channels per EEG channel (2 for cos/sin, 3 if amplitude)
        constrain_output: Whether to apply tanh to cos/sin outputs (recommended)
    """

    def __init__(
        self,
        n_channels: int,
        hidden_size: int = 64,
        complexity: int = 2,
        dropout: float = 0.1,
        phase_channels: int = 2,
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

        # LSTM layers (mirror encoder)
        n_lstm_layers = 2 if complexity >= 3 else 1

        # Determine input size to match encoder's conv output
        if complexity == 0:
            conv_input = 32
        elif complexity == 1:
            conv_input = 64
        elif complexity == 2:
            conv_input = 128
        else:
            conv_input = 512

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=conv_input,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )

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
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
        # No activation on final layer - outputs cos/sin in [-1, 1] naturally
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
            hidden: Not used (kept for API compatibility). Decoder LSTM
                    has different hidden_size than encoder.

        Returns:
            reconstruction: Reconstructed signal (batch, channels, time)
                           where channels = n_channels * phase_channels
        """
        # LSTM decoding - don't use encoder hidden state since sizes differ
        lstm_out, _ = self.lstm(latent)

        # Transpose for convolutions: (batch, features, time)
        lstm_out = lstm_out.transpose(1, 2)

        # Transposed convolutions for reconstruction
        reconstruction = self.deconv_layers(lstm_out)

        # Apply output constraints if enabled
        if self.constrain_output:
            # Shape: (batch, n_channels * phase_channels, time)
            # cos/sin are in positions 0,1 for each channel
            # amplitude (if present) is in position 2
            batch_size, total_channels, time = reconstruction.shape

            # Reshape to (batch, n_channels, phase_channels, time)
            reconstruction = reconstruction.view(batch_size, self.n_channels, self.phase_channels, time)

            # Apply tanh to cos/sin channels (indices 0 and 1)
            reconstruction[:, :, 0, :] = torch.tanh(reconstruction[:, :, 0, :])  # cos
            reconstruction[:, :, 1, :] = torch.tanh(reconstruction[:, :, 1, :])  # sin
            # Leave amplitude (index 2, if present) unconstrained (z-scored, can be any value)

            # Reshape back to (batch, n_channels * phase_channels, time)
            reconstruction = reconstruction.view(batch_size, total_channels, time)

        return reconstruction

    @classmethod
    def from_config(cls, cfg: DictConfig, n_channels: int) -> ConvLSTMDecoder:
        """Create decoder from Hydra config."""
        return cls(
            n_channels=n_channels,
            hidden_size=cfg.encoder.hidden_size,  # Must match encoder
            complexity=cfg.encoder.complexity,    # Must match encoder
            dropout=cfg.encoder.dropout,
            phase_channels=3 if cfg.phase.include_amplitude else 2,
        )
