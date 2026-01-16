"""Tests for model architectures."""

import numpy as np
import torch
import pytest

from eeg_biomarkers.models import ConvLSTMAutoencoder, ConvLSTMEncoder, ConvLSTMDecoder


class TestConvLSTMEncoder:
    """Tests for the encoder module."""

    @pytest.mark.parametrize("complexity", [0, 1, 2, 3])
    def test_forward_shapes(self, complexity):
        """Test output shapes for different complexity levels."""
        n_channels = 64
        hidden_size = 32
        batch_size = 4
        time_steps = 100
        phase_channels = 2

        encoder = ConvLSTMEncoder(
            n_channels=n_channels,
            hidden_size=hidden_size,
            complexity=complexity,
            phase_channels=phase_channels,
        )

        x = torch.randn(batch_size, n_channels * phase_channels, time_steps)
        latent, (h_n, c_n) = encoder(x)

        # Latent should be (batch, time', hidden_size)
        assert latent.shape[0] == batch_size
        assert latent.shape[2] == hidden_size


class TestConvLSTMAutoencoder:
    """Tests for the full autoencoder."""

    def test_reconstruction_shape(self):
        """Reconstruction should match input shape."""
        n_channels = 64
        hidden_size = 32
        batch_size = 4
        time_steps = 100
        phase_channels = 2

        model = ConvLSTMAutoencoder(
            n_channels=n_channels,
            hidden_size=hidden_size,
            complexity=2,
            phase_channels=phase_channels,
        )

        x = torch.randn(batch_size, n_channels * phase_channels, time_steps)
        reconstruction, latent = model(x)

        assert reconstruction.shape == x.shape

    def test_encode_only(self):
        """Test encoding without decoding."""
        model = ConvLSTMAutoencoder(n_channels=64, hidden_size=32)

        x = torch.randn(4, 128, 100)  # 64 channels * 2 phase_channels
        latent = model.encode(x)

        assert latent.shape[0] == 4  # Batch
        assert latent.shape[2] == 32  # Hidden size

    def test_angular_distance_matrix(self):
        """Test angular distance matrix computation."""
        model = ConvLSTMAutoencoder(n_channels=64, hidden_size=32)

        # Create a simple latent trajectory
        latent = torch.randn(10, 32)  # 10 time steps, 32 dims

        dist_matrix = model.compute_angular_distance_matrix(latent)

        # Should be square
        assert dist_matrix.shape == (10, 10)

        # Diagonal should be approximately zero (self-distance)
        # Use relaxed tolerance for floating point precision
        torch.testing.assert_close(
            torch.diag(dist_matrix),
            torch.zeros(10),
            atol=1e-3,  # Relaxed due to floating point precision
            rtol=1e-3,
        )

        # Should be symmetric
        torch.testing.assert_close(dist_matrix, dist_matrix.T, atol=1e-5, rtol=1e-5)

        # Values should be in [0, π]
        assert torch.all(dist_matrix >= 0)
        assert torch.all(dist_matrix <= np.pi + 1e-5)

    def test_recurrence_matrix_rr_controlled(self):
        """Test RR-controlled recurrence matrix."""
        model = ConvLSTMAutoencoder(n_channels=64, hidden_size=32)

        latent = torch.randn(50, 32)
        target_rr = 0.05  # 5%

        R, epsilon = model.compute_recurrence_matrix(
            latent, threshold_method="rr_controlled", target_rr=target_rr
        )

        # Should be binary
        assert torch.all((R == 0) | (R == 1))

        # Diagonal should be 1 (self-recurrence)
        assert torch.all(torch.diag(R) == 1)

        # Check approximate RR (off-diagonal)
        n = R.shape[0]
        off_diag_mask = ~torch.eye(n, dtype=torch.bool)
        actual_rr = R[off_diag_mask].mean().item()

        # Should be close to target (within reasonable tolerance)
        assert abs(actual_rr - target_rr) < 0.02
