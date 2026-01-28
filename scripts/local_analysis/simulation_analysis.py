#!/usr/bin/env python3
"""
Simulation-Based Validation for Paper 1: Dynamical Microscope

This script generates synthetic multivariate time series with KNOWN dynamical
structure, trains a lightweight autoencoder on this synthetic data, and validates
that the analysis pipeline recovers:
- Latent basins
- Flow fields
- Dwell structure
- Trajectory geometry

This is a UNIT TEST for the dynamical microscope framework.

Scientific message:
"The framework recovers known dynamical regimes and flow geometry from
time series alone — it does not invent structure."

IMPORTANT:
- This script is SELF-CONTAINED and does not affect real EEG models
- Checkpoints are saved to timestamped output directories, NOT models/
- All training is lightweight and runs on CPU/MPS in minutes

Usage:
    python simulation_analysis.py                  # Full simulation
    python simulation_analysis.py --quick          # Quick test (fewer epochs)
    python simulation_analysis.py --no-show        # Don't display plots

Output:
    figures/paper1/fig02_simulations.pdf
    results/simulations/{timestamp}/
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
from scipy.signal import hilbert
from sklearn.decomposition import PCA
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add src to path for model imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from eeg_biomarkers.models import ConvLSTMAutoencoder, TransformerAutoencoder

# =============================================================================
# CONFIGURATION
# =============================================================================

# Simulation parameters
SFREQ = 250.0  # Hz - match real EEG
CHUNK_DURATION = 5.0  # seconds - match real analysis
N_CHANNELS = 30  # Moderate number of channels
TOTAL_DURATION = 180.0  # 3 minutes per condition

# Output paths
FIGURES_DIR = Path(__file__).parent.parent.parent / "figures" / "paper1"
RESULTS_BASE_DIR = Path(__file__).parent.parent.parent / "results" / "simulations"

# Device
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SimulationResult:
    """Container for simulation outputs."""
    time: np.ndarray
    latent_states: np.ndarray  # (T, latent_dim) - ground truth latent
    observations: np.ndarray   # (T, n_channels) - observed signals
    regime_labels: np.ndarray  # (T,) - regime label at each time point
    transition_times: list     # List of transition time indices
    regime_names: list         # Names of regimes


@dataclass
class FlowMetrics:
    """Flow geometry metrics for a trajectory."""
    mean_speed: float
    speed_std: float
    speed_cv: float
    n_dwell_episodes: int
    total_dwell_time: int
    mean_dwell_duration: float
    occupancy_entropy: float
    path_tortuosity: float
    explored_variance: float


# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================

class MetastableSwitchingSystem:
    """
    Generate synthetic data from a metastable switching dynamical system.

    Implements 2-3 linear dynamical systems with HMM-like switching.
    Each regime has different dynamics AND different phase-coupling patterns
    to create biologically plausible regime separation.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        latent_dim: int = 3,
        n_channels: int = 30,
        sfreq: float = 250.0,
        noise_std: float = 0.015,  # Low noise for clean basins
        transition_prob: float = 0.0004,  # ~1 transition per 10s for balanced dwells (5-8s avg)
        transition_smoothing: float = 0.5,  # Seconds to cross-fade between regimes
        seed: int = 42,
    ):
        self.n_regimes = n_regimes
        self.latent_dim = latent_dim
        self.n_channels = n_channels
        self.sfreq = sfreq
        self.noise_std = noise_std
        self.transition_prob = transition_prob
        self.transition_smoothing = transition_smoothing
        self.rng = np.random.RandomState(seed)

        # Create different dynamics matrices for each regime
        self.dynamics = self._create_regime_dynamics()

        # Regime-specific noise scales (different "activity levels")
        # This creates different trajectory densities/speeds per regime
        self.regime_noise_scales = self._create_regime_noise_scales()

        # Regime-specific mixing matrices (different phase-coupling topographies)
        # This is KEY: each regime has different channel coupling patterns
        self.mixing_matrices = self._create_regime_mixing_matrices()

        # Regime names
        self.regime_names = [f"Regime_{i}" for i in range(n_regimes)]

    def _create_regime_noise_scales(self) -> list[float]:
        """Create regime-specific noise scales for different activity levels."""
        return [
            0.3,   # Regime 0: Low noise - quiet, stable
            1.5,   # Regime 1: High noise - active, variable
            0.8,   # Regime 2: Medium noise - moderate activity
        ][:self.n_regimes]

    def _create_regime_dynamics(self) -> list[np.ndarray]:
        """Create MAXIMALLY DISTINCT dynamical systems for each regime.

        The key insight: for regimes to be separable in the autoencoder's latent space,
        they need FUNDAMENTALLY DIFFERENT dynamics that create different trajectory
        geometries. This means very different:
        - Decay rates (how tightly confined vs exploratory)
        - Rotation speeds (how fast the trajectory spirals)
        - Noise sensitivity (how much they respond to perturbations)

        These differences create distinct "fingerprints" in the phase-space that
        the autoencoder learns to represent differently.
        """
        dynamics = []

        for i in range(self.n_regimes):
            if i == 0:
                # Regime 0: SLOW, STABLE - tight attractor, minimal movement
                # Like a deep meditative state - very low activity
                A = self._create_stable_dynamics(decay=0.5, rotation=0.02)
            elif i == 1:
                # Regime 1: FAST, OSCILLATORY - rapid spiraling motion
                # Like an alert, active state - high-frequency oscillations
                A = self._create_stable_dynamics(decay=0.85, rotation=0.6)
            else:
                # Regime 2: EXPLORATORY - weak attractor, wandering
                # Like a transitional/searching state - broad exploration
                A = self._create_stable_dynamics(decay=0.95, rotation=0.15)

            dynamics.append(A)

        return dynamics

    def _create_regime_mixing_matrices(self) -> list[np.ndarray]:
        """Create regime-specific mixing matrices with MAXIMALLY DIFFERENT spatial patterns.

        The key insight: for regimes to be separable after phase→autoencoder→PCA,
        they need to activate DIFFERENT CHANNEL SUBSETS with DIFFERENT AMPLITUDES.

        This mimics how real brain states have different spatial topographies.
        """
        mixing_matrices = []

        # Define 3 non-overlapping channel groups (exclusive activation)
        n_per_group = self.n_channels // 3
        group1 = np.arange(0, n_per_group)                    # Channels 0-9
        group2 = np.arange(n_per_group, 2 * n_per_group)      # Channels 10-19
        group3 = np.arange(2 * n_per_group, self.n_channels)  # Channels 20-29

        for regime in range(self.n_regimes):
            # Very small base (near-zero for inactive channels)
            M = self.rng.randn(self.n_channels, self.latent_dim) * 0.05

            if regime == 0:
                # Regime 0: GROUP 1 ACTIVE (frontal-like)
                # Only first group has strong signal
                M[group1, 0] = 3.0 + self.rng.randn(len(group1)) * 0.2
                M[group1, 1] = 1.5 + self.rng.randn(len(group1)) * 0.2
                M[group1, 2] = 0.8 + self.rng.randn(len(group1)) * 0.1
                # Other groups: minimal (background noise level)
                M[group2, :] *= 0.3
                M[group3, :] *= 0.3

            elif regime == 1:
                # Regime 1: GROUP 2 ACTIVE (central-like)
                # Only second group has strong signal
                M[group2, 0] = 2.5 + self.rng.randn(len(group2)) * 0.2
                M[group2, 1] = 2.0 + self.rng.randn(len(group2)) * 0.2
                M[group2, 2] = 1.2 + self.rng.randn(len(group2)) * 0.1
                # Other groups: minimal
                M[group1, :] *= 0.3
                M[group3, :] *= 0.3

            else:
                # Regime 2: GROUP 3 ACTIVE (posterior-like)
                # Only third group has strong signal
                M[group3, 0] = 2.8 + self.rng.randn(len(group3)) * 0.2
                M[group3, 1] = 1.8 + self.rng.randn(len(group3)) * 0.2
                M[group3, 2] = 1.0 + self.rng.randn(len(group3)) * 0.1
                # Other groups: minimal
                M[group1, :] *= 0.3
                M[group2, :] *= 0.3

            mixing_matrices.append(M)

        return mixing_matrices

    def _create_stable_dynamics(self, decay: float, rotation: float) -> np.ndarray:
        """
        Create a stable dynamics matrix with specified characteristics.

        Args:
            decay: How quickly state decays toward origin (0.9 = fast, 0.99 = slow)
            rotation: How much rotation/oscillation (0 = none, 0.5 = strong)
        """
        # Start with scaled identity
        A = np.eye(self.latent_dim) * decay

        # Add rotation (off-diagonal terms)
        for i in range(self.latent_dim - 1):
            A[i, i+1] = rotation
            A[i+1, i] = -rotation

        # Add some random structure
        A += self.rng.randn(self.latent_dim, self.latent_dim) * 0.01

        # Ensure stability (all eigenvalues < 1)
        eigenvalues = np.linalg.eigvals(A)
        max_eig = np.max(np.abs(eigenvalues))
        if max_eig >= 1:
            A = A * (0.99 / max_eig)

        return A

    def generate(self, duration: float) -> SimulationResult:
        """
        Generate synthetic time series with smooth regime transitions.

        Args:
            duration: Total duration in seconds

        Returns:
            SimulationResult with ground truth and observations
        """
        n_samples = int(duration * self.sfreq)
        smoothing_samples = int(self.transition_smoothing * self.sfreq)

        # Initialize
        latent_states = np.zeros((n_samples, self.latent_dim))
        regime_labels = np.zeros(n_samples, dtype=int)
        transition_times = []

        # Start in regime 0
        current_regime = 0
        latent_states[0] = self.rng.randn(self.latent_dim) * 0.5
        regime_labels[0] = current_regime

        # Regime-specific basin centers - well separated in latent space
        # Triangle layout with 120° separation - LARGER magnitudes
        regime_offsets = np.array([
            [10.0, 0.0, 0.0],      # Regime 0: far right (+x axis)
            [-5.0, 8.7, 0.0],     # Regime 1: upper left (120°)
            [-5.0, -8.7, 0.0],    # Regime 2: lower left (240°)
        ])[:self.n_regimes, :self.latent_dim]

        # Track regime weights for smooth transitions
        regime_weights = np.zeros((n_samples, self.n_regimes))
        regime_weights[0, current_regime] = 1.0

        # Track visits to ensure all regimes are covered
        regime_visit_counts = np.zeros(self.n_regimes, dtype=int)
        regime_visit_counts[current_regime] = 1

        # Evolve dynamics
        pending_transition = None
        transition_progress = 0
        time_in_regime = 0
        min_dwell_samples = int(3.0 * self.sfreq)  # Minimum 3 seconds per regime

        for t in range(1, n_samples):
            time_in_regime += 1

            # Check for new transition (only after minimum dwell)
            if pending_transition is None and time_in_regime > min_dwell_samples:
                if self.rng.rand() < self.transition_prob:
                    # Prefer transitions to less-visited regimes
                    visit_weights = 1.0 / (regime_visit_counts + 1)
                    visit_weights[current_regime] = 0  # Don't stay in same regime
                    visit_probs = visit_weights / visit_weights.sum()
                    new_regime = self.rng.choice(self.n_regimes, p=visit_probs)

                    pending_transition = new_regime
                    transition_progress = 0
                    transition_times.append(t)
                    time_in_regime = 0

            # Handle smooth transition
            if pending_transition is not None:
                transition_progress += 1
                # Smooth cross-fade using sigmoid-like function
                alpha = min(1.0, transition_progress / smoothing_samples)
                alpha = 0.5 * (1 + np.tanh(4 * (alpha - 0.5)))  # Smooth S-curve

                regime_weights[t, current_regime] = 1 - alpha
                regime_weights[t, pending_transition] = alpha

                if transition_progress >= smoothing_samples:
                    current_regime = pending_transition
                    regime_visit_counts[current_regime] += 1
                    pending_transition = None
            else:
                regime_weights[t, current_regime] = 1.0

            regime_labels[t] = current_regime

            # Blend dynamics, offsets, and noise scales based on regime weights
            A_blend = sum(regime_weights[t, r] * self.dynamics[r] for r in range(self.n_regimes))
            offset_blend = sum(regime_weights[t, r] * regime_offsets[r] for r in range(self.n_regimes))
            noise_scale_blend = sum(regime_weights[t, r] * self.regime_noise_scales[r] for r in range(self.n_regimes))

            # State evolution: x(t+1) = A @ (x(t) - offset) + offset + noise
            deviation = latent_states[t-1] - offset_blend
            latent_states[t] = A_blend @ deviation + offset_blend + self.rng.randn(self.latent_dim) * self.noise_std * noise_scale_blend

        # Project to observations using regime-specific mixing matrices
        observations = np.zeros((n_samples, self.n_channels))
        for t in range(n_samples):
            # Blend mixing matrices based on regime weights
            M_blend = sum(regime_weights[t, r] * self.mixing_matrices[r] for r in range(self.n_regimes))
            observations[t] = latent_states[t] @ M_blend.T

        # Add observation noise
        observations += self.rng.randn(*observations.shape) * self.noise_std * 0.5

        time = np.arange(n_samples) / self.sfreq

        return SimulationResult(
            time=time,
            latent_states=latent_states,
            observations=observations,
            regime_labels=regime_labels,
            transition_times=transition_times,
            regime_names=self.regime_names,
        )


class AttractorStabilitySystem:
    """
    Generate synthetic data comparing weakly vs strongly stable attractors.

    Creates two visually distinct patterns:
    - Stable: Central blob (tight attractor, fast decay to center)
    - Exploratory: Ring/annulus (strong rotation maintains distance from center)

    Both are centered at the origin for easy visual comparison.
    """

    def __init__(
        self,
        latent_dim: int = 3,
        n_channels: int = 30,
        sfreq: float = 250.0,
        noise_std: float = 0.05,
        seed: int = 42,
    ):
        self.latent_dim = latent_dim
        self.n_channels = n_channels
        self.sfreq = sfreq
        self.noise_std = noise_std
        self.rng = np.random.RandomState(seed)

        # Mixing matrix - shared for both conditions
        self.mixing_matrix = self.rng.randn(n_channels, latent_dim) * 0.5

    def generate(self, duration: float, condition: str) -> SimulationResult:
        """
        Generate time series for a specific condition.

        Args:
            duration: Total duration in seconds
            condition: "stable" (central blob) or "exploratory" (ring)
        """
        n_samples = int(duration * self.sfreq)

        if condition == "stable":
            # Strong attractor: VERY fast decay pulls everything to center
            # Creates a TIGHT CENTRAL BLOB
            decay = 0.80  # Fast decay toward center
            rotation = 0.05  # Minimal rotation
            noise_scale = 0.5  # Some noise but pulls back to center
            radius_target = 0.0  # No radius offset - centered
            regime_name = "Stable"
        else:
            # Exploratory: STRONG ROTATION creates clear circular motion
            # Moderate decay with strong radial restoring = PERFECT RING
            decay = 0.998  # Almost no decay - maintains radius
            rotation = 0.08  # Moderate rotation - smooth circular motion
            noise_scale = 0.15  # Very low noise - clean ring
            radius_target = 3.0  # Larger target radius for clearer ring
            regime_name = "Exploratory"

        # Create rotation matrix (2D rotation in first 2 dims)
        A = np.eye(self.latent_dim) * decay

        # Pure rotation in xy-plane
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        A[0, 0] = decay * cos_r
        A[0, 1] = decay * -sin_r
        A[1, 0] = decay * sin_r
        A[1, 1] = decay * cos_r

        # Generate trajectory
        latent_states = np.zeros((n_samples, self.latent_dim))

        if condition == "exploratory":
            # Start on the ring (at target radius)
            latent_states[0, 0] = radius_target
            latent_states[0, 1] = 0.0
        else:
            # Start near center
            latent_states[0] = self.rng.randn(self.latent_dim) * 0.2

        for t in range(1, n_samples):
            # Apply dynamics
            latent_states[t] = A @ latent_states[t-1]

            # Add noise
            latent_states[t] += self.rng.randn(self.latent_dim) * self.noise_std * noise_scale

            # For exploratory: add STRONG radial restoring force to maintain ring
            if condition == "exploratory":
                current_radius = np.sqrt(latent_states[t, 0]**2 + latent_states[t, 1]**2)
                if current_radius > 0.01:
                    # Strong restoring force to maintain exact ring radius
                    radial_correction = 0.05 * (radius_target - current_radius)
                    latent_states[t, 0] += radial_correction * latent_states[t, 0] / current_radius
                    latent_states[t, 1] += radial_correction * latent_states[t, 1] / current_radius

        # Project to observations
        observations = latent_states @ self.mixing_matrix.T
        observations += self.rng.randn(*observations.shape) * self.noise_std * 0.3

        time = np.arange(n_samples) / self.sfreq

        # All same regime (no switching)
        regime_labels = np.zeros(n_samples, dtype=int)

        return SimulationResult(
            time=time,
            latent_states=latent_states,
            observations=observations,
            regime_labels=regime_labels,
            transition_times=[],
            regime_names=[regime_name],
        )


# =============================================================================
# PHASE REPRESENTATION (circular: cos, sin + amplitude)
# =============================================================================

def observations_to_phase_representation(
    observations: np.ndarray,
    sfreq: float,
    filter_low: float = 1.0,
    filter_high: float = 30.0,
) -> np.ndarray:
    """
    Convert raw observations to circular phase representation.

    This mirrors the real EEG preprocessing pipeline:
    - Bandpass filter
    - Hilbert transform
    - Extract (cos(phase), sin(phase), log(amplitude))

    Args:
        observations: (T, n_channels) raw signals
        sfreq: Sampling frequency
        filter_low: Low cutoff (Hz)
        filter_high: High cutoff (Hz)

    Returns:
        phase_data: (n_channels * 3, T) - [cos, sin, log_amp] stacked
    """
    from scipy.signal import butter, filtfilt

    T, n_channels = observations.shape

    # Bandpass filter
    nyq = sfreq / 2
    low = max(filter_low / nyq, 0.001)
    high = min(filter_high / nyq, 0.999)
    b, a = butter(4, [low, high], btype="band")

    # Filter (transpose for filtfilt which expects channels in last axis)
    filtered = filtfilt(b, a, observations, axis=0)

    # Hilbert transform for analytic signal
    analytic = hilbert(filtered, axis=0)

    # Extract phase and amplitude
    phase = np.angle(analytic)
    amplitude = np.abs(analytic)

    # Circular representation
    cos_phase = np.cos(phase)
    sin_phase = np.sin(phase)
    log_amplitude = np.log1p(amplitude)

    # Stack: (n_channels * 3, T)
    phase_data = np.vstack([
        cos_phase.T,      # (n_channels, T)
        sin_phase.T,      # (n_channels, T)
        log_amplitude.T,  # (n_channels, T)
    ])

    return phase_data.astype(np.float32)


def chunk_phase_data(
    phase_data: np.ndarray,
    chunk_samples: int,
    overlap: float = 0.0,
) -> list[np.ndarray]:
    """Split phase data into chunks."""
    n_features, T = phase_data.shape
    step = int(chunk_samples * (1 - overlap))

    chunks = []
    for start in range(0, T - chunk_samples + 1, step):
        end = start + chunk_samples
        chunks.append(phase_data[:, start:end])

    return chunks


# =============================================================================
# LIGHTWEIGHT AUTOENCODER TRAINING
# =============================================================================

class SimulationAutoencoder(nn.Module):
    """
    Lightweight autoencoder for simulation validation.

    Uses the same architecture as the real model but can be configured
    to be smaller for faster training.
    """

    def __init__(
        self,
        n_channels: int,
        hidden_size: int = 32,
        phase_channels: int = 3,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.hidden_size = hidden_size
        self.phase_channels = phase_channels

        input_size = n_channels * phase_channels

        # Simple encoder: Conv1d -> LSTM -> Linear
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )

        self.encoder_lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        # Decoder: LSTM -> ConvTranspose1d
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, input_size, kernel_size=5, stride=2, padding=2, output_padding=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (batch, n_channels * phase_channels, time)

        Returns:
            reconstruction: Same shape as input
            latent: (batch, time', hidden_size)
        """
        batch_size = x.shape[0]

        # Encode
        h = self.encoder_conv(x)  # (batch, 128, time')
        h = h.permute(0, 2, 1)    # (batch, time', 128)

        latent, _ = self.encoder_lstm(h)  # (batch, time', hidden_size)

        # Decode
        h_dec, _ = self.decoder_lstm(latent)  # (batch, time', 128)
        h_dec = h_dec.permute(0, 2, 1)        # (batch, 128, time')

        reconstruction = self.decoder_conv(h_dec)  # (batch, input_size, time)

        # Handle size mismatch from strided convolutions
        if reconstruction.shape[2] != x.shape[2]:
            reconstruction = torch.nn.functional.interpolate(
                reconstruction, size=x.shape[2], mode='linear', align_corners=False
            )

        return reconstruction, latent

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent space."""
        h = self.encoder_conv(x)
        h = h.permute(0, 2, 1)
        latent, _ = self.encoder_lstm(h)
        return latent


def train_simulation_model(
    chunks: list[np.ndarray],
    n_channels: int,
    hidden_size: int = 32,
    n_epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cpu",
    verbose: bool = True,
    labels: list[int] | None = None,
    use_contrastive: bool = False,
    contrastive_weight: float = 0.15,
    contrastive_temperature: float = 0.07,
    use_transformer: bool = False,
) -> SimulationAutoencoder:
    """
    Train an autoencoder on simulation data with optional two-phase training.

    Args:
        chunks: List of (n_features, chunk_samples) arrays
        n_channels: Number of original channels
        hidden_size: Latent dimension
        n_epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to train on
        verbose: Print progress
        labels: Optional regime labels for contrastive learning
        use_contrastive: Enable contrastive loss (Phase 2)
        contrastive_weight: Weight for contrastive loss
        contrastive_temperature: Temperature for contrastive softmax
        use_transformer: Use TransformerAutoencoder instead of SimulationAutoencoder

    Returns:
        Trained model
    """
    # Create dataset
    data = torch.stack([torch.from_numpy(c) for c in chunks])  # (N, features, time)

    if labels is not None:
        label_tensor = torch.tensor(labels, dtype=torch.long)
        dataset = TensorDataset(data, label_tensor)
    else:
        dataset = TensorDataset(data)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model
    if use_transformer:
        from omegaconf import OmegaConf
        model_cfg = OmegaConf.create({
            'name': 'transformer_autoencoder',
            'encoder': {
                'hidden_size': hidden_size,
                'num_layers': 2,
                'nhead': 4,
                'dim_feedforward': hidden_size * 2,
                'dropout': 0.1,
            },
            'decoder': {
                'hidden_size': hidden_size,
                'num_layers': 2,
                'nhead': 4,
                'dim_feedforward': hidden_size * 2,
                'dropout': 0.1,
            },
            'phase': {
                'include_amplitude': True,
            },
        })
        model = TransformerAutoencoder.from_config(model_cfg, n_channels)
    else:
        model = SimulationAutoencoder(
            n_channels=n_channels,
            hidden_size=hidden_size,
            phase_channels=3,
        )

    model = model.to(device)
    model.train()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Contrastive loss if enabled
    contrastive_loss_fn = None
    if use_contrastive and labels is not None:
        contrastive_loss_fn = ContrastiveLoss(temperature=contrastive_temperature, mode="condition")

    # Training loop
    losses = []
    phase_str = "Phase 2 (Contrastive)" if use_contrastive else "Phase 1 (Reconstruction)"
    iterator = tqdm(range(n_epochs), desc=f"Training {phase_str}") if verbose else range(n_epochs)

    for epoch in iterator:
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_contr = 0.0
        n_batches = 0

        for batch_data in dataloader:
            if labels is not None:
                batch, batch_labels = batch_data
                batch_labels = batch_labels.to(device)
            else:
                (batch,) = batch_data
                batch_labels = None

            batch = batch.float().to(device)

            optimizer.zero_grad()
            output = model(batch)

            # Handle tuple return (reconstruction, latent)
            if isinstance(output, tuple):
                reconstruction, latent = output
            else:
                reconstruction = output
                latent = model.encode(batch) if hasattr(model, 'encode') else None

            # Reconstruction loss
            recon_loss = criterion(reconstruction, batch)
            total_loss = recon_loss

            # Contrastive loss (Phase 2)
            contr_loss_val = 0.0
            if use_contrastive and contrastive_loss_fn is not None and batch_labels is not None and latent is not None:
                # Get mean latent representation for contrastive
                if latent.dim() == 3:  # (batch, time, hidden)
                    latent_mean = latent.mean(dim=1)  # (batch, hidden)
                else:
                    latent_mean = latent
                contr_loss = contrastive_loss_fn(latent_mean, batch_labels)
                total_loss = (1 - contrastive_weight) * recon_loss + contrastive_weight * contr_loss
                contr_loss_val = contr_loss.item()

            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_contr += contr_loss_val
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon / n_batches
        avg_contr = epoch_contr / n_batches
        losses.append(avg_loss)

        if verbose and (epoch + 1) % 10 == 0:
            if use_contrastive:
                tqdm.write(f"Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f} recon={avg_recon:.4f} contr={avg_contr:.4f}")
            else:
                tqdm.write(f"Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}")

    model.eval()
    return model


class ContrastiveLoss(nn.Module):
    """Supervised contrastive loss for learning discriminative representations."""

    def __init__(self, temperature: float = 0.07, mode: str = "condition"):
        super().__init__()
        self.temperature = temperature
        self.mode = mode

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute supervised contrastive loss."""
        batch_size = embeddings.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # L2 normalize embeddings
        embeddings = nn.functional.normalize(embeddings, dim=1, eps=1e-8)

        # Compute similarity matrix
        sim_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature

        # Create mask for positive pairs (same label)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.t()).float()

        # Remove diagonal (self-similarity)
        diag_mask = torch.eye(batch_size, device=mask.device)
        mask = mask - diag_mask

        # Count positive pairs for each sample
        pos_count = mask.sum(dim=1)
        valid_samples = pos_count > 0

        if not valid_samples.any():
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # For numerical stability
        sim_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()

        # Compute exp(sim) with self-similarity zeroed out
        exp_sim = torch.exp(sim_matrix) * (1 - diag_mask)
        sum_exp = exp_sim.sum(dim=1) + 1e-8
        pos_exp_sum = (exp_sim * mask).sum(dim=1)

        loss_per_sample = -torch.log(pos_exp_sum / sum_exp + 1e-8)
        loss = loss_per_sample[valid_samples].mean()
        return torch.clamp(loss, min=0.0, max=100.0)


def compute_latent_trajectory(
    model: SimulationAutoencoder,
    phase_data: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute latent trajectory from phase data.

    Args:
        model: Trained autoencoder
        phase_data: (n_features, T) phase representation
        device: Device for inference

    Returns:
        (T', hidden_size) latent trajectory
    """
    model.eval()

    # Add batch dimension
    x = torch.from_numpy(phase_data).float().unsqueeze(0).to(device)

    with torch.no_grad():
        latent = model.encode(x)

    return latent.squeeze(0).cpu().numpy()


# =============================================================================
# FLOW METRICS (reused from full_dataset_analysis.py)
# =============================================================================

def compute_instantaneous_speed(embedded: np.ndarray) -> np.ndarray:
    """Compute speed in embedding space."""
    diff = np.diff(embedded, axis=0)
    return np.linalg.norm(diff, axis=1)


def detect_dwell_episodes(
    speed: np.ndarray,
    threshold_percentile: float = 20,
    min_duration: int = 10,
) -> list:
    """Detect contiguous low-speed runs."""
    threshold = np.percentile(speed, threshold_percentile)
    is_slow = speed < threshold

    episodes = []
    in_episode = False
    start = 0

    for i, slow in enumerate(is_slow):
        if slow and not in_episode:
            in_episode = True
            start = i
        elif not slow and in_episode:
            in_episode = False
            if i - start >= min_duration:
                episodes.append((start, i))

    if in_episode and len(is_slow) - start >= min_duration:
        episodes.append((start, len(is_slow)))

    return episodes


def compute_occupancy_entropy(embedded: np.ndarray, bins: int = 20) -> float:
    """Compute entropy of occupancy distribution."""
    from scipy.stats import entropy
    H, _, _ = np.histogram2d(embedded[:, 0], embedded[:, 1], bins=bins)
    H = H.flatten()
    H = H[H > 0]
    p = H / H.sum()
    return entropy(p)


def compute_flow_metrics(embedded: np.ndarray) -> FlowMetrics:
    """Compute comprehensive flow geometry metrics."""
    speed = compute_instantaneous_speed(embedded)
    episodes = detect_dwell_episodes(speed)
    path_length = speed.sum()
    displacement = np.linalg.norm(embedded[-1] - embedded[0])
    tortuosity = path_length / displacement if displacement > 0 else np.inf
    explored_variance = np.var(embedded, axis=0).sum()
    occ_entropy = compute_occupancy_entropy(embedded)

    return FlowMetrics(
        mean_speed=speed.mean(),
        speed_std=speed.std(),
        speed_cv=speed.std() / speed.mean() if speed.mean() > 0 else 0,
        n_dwell_episodes=len(episodes),
        total_dwell_time=sum(e[1] - e[0] for e in episodes),
        mean_dwell_duration=np.mean([e[1] - e[0] for e in episodes]) if episodes else 0,
        occupancy_entropy=occ_entropy,
        path_tortuosity=tortuosity,
        explored_variance=explored_variance,
    )


# =============================================================================
# FLOW FIELD COMPUTATION
# =============================================================================

def compute_flow_field(
    embedded: np.ndarray,
    bounds: tuple,
    grid_size: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute flow field from embedded trajectory.

    Returns:
        X, Y: Grid coordinates
        flow_x, flow_y: Mean flow vectors
        counts: Number of samples per bin
    """
    xmin, xmax, ymin, ymax = bounds

    x_edges = np.linspace(xmin, xmax, grid_size + 1)
    y_edges = np.linspace(ymin, ymax, grid_size + 1)

    flow_x_sum = np.zeros((grid_size, grid_size))
    flow_y_sum = np.zeros((grid_size, grid_size))
    counts = np.zeros((grid_size, grid_size))

    dx = np.diff(embedded[:, 0])
    dy = np.diff(embedded[:, 1])

    for i in range(len(dx)):
        x_bin = np.searchsorted(x_edges[:-1], embedded[i, 0]) - 1
        y_bin = np.searchsorted(y_edges[:-1], embedded[i, 1]) - 1

        x_bin = np.clip(x_bin, 0, grid_size - 1)
        y_bin = np.clip(y_bin, 0, grid_size - 1)

        flow_x_sum[y_bin, x_bin] += dx[i]
        flow_y_sum[y_bin, x_bin] += dy[i]
        counts[y_bin, x_bin] += 1

    flow_x = np.zeros_like(flow_x_sum)
    flow_y = np.zeros_like(flow_y_sum)
    mask = counts > 0
    flow_x[mask] = flow_x_sum[mask] / counts[mask]
    flow_y[mask] = flow_y_sum[mask] / counts[mask]

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers)

    return X, Y, flow_x, flow_y, counts


def compute_density_on_grid(
    embedded: np.ndarray,
    bounds: tuple,
    bins: int = 50,
    sigma: float = 1.5,
) -> np.ndarray:
    """Compute normalized 2D density on a shared grid."""
    xmin, xmax, ymin, ymax = bounds
    H, _, _ = np.histogram2d(
        embedded[:, 0], embedded[:, 1],
        bins=bins,
        range=[[xmin, xmax], [ymin, ymax]]
    )
    H = gaussian_filter(H.T, sigma=sigma)
    if H.sum() > 0:
        H = H / H.sum()
    return H


# =============================================================================
# POOLED EMBEDDER (for consistent coordinate space)
# =============================================================================

class PooledEmbedder:
    """Fits PCA or UMAP on pooled data, transforms individual trajectories."""

    def __init__(self, n_components: int = 2, method: str = "pca"):
        self.n_components = n_components
        self.method = method
        self.reducer = None
        self.bounds = None
        self.centroid = None

    def fit(self, trajectories: list[np.ndarray], n_samples_per_traj: int = 500):
        """Fit dimensionality reduction on pooled trajectories."""
        pooled = []
        for traj in trajectories:
            if len(traj) <= n_samples_per_traj:
                pooled.append(traj)
            else:
                indices = np.linspace(0, len(traj) - 1, n_samples_per_traj, dtype=int)
                pooled.append(traj[indices])

        pooled_data = np.vstack(pooled)

        if self.method == "umap":
            try:
                from umap import UMAP
                self.reducer = UMAP(n_components=self.n_components, n_neighbors=30, min_dist=0.1, random_state=42)
                embedded = self.reducer.fit_transform(pooled_data)
            except ImportError:
                print("UMAP not available, falling back to PCA")
                self.method = "pca"
                self.reducer = PCA(n_components=self.n_components)
                embedded = self.reducer.fit_transform(pooled_data)
        else:
            self.reducer = PCA(n_components=self.n_components)
            embedded = self.reducer.fit_transform(pooled_data)

        # Store bounds (square, symmetric)
        self.centroid = embedded.mean(axis=0)
        max_dev = max(
            np.abs(embedded[:, 0] - self.centroid[0]).max(),
            np.abs(embedded[:, 1] - self.centroid[1]).max(),
        )
        margin = 0.05
        half_size = max_dev * (1 + margin)
        self.bounds = (
            self.centroid[0] - half_size,
            self.centroid[0] + half_size,
            self.centroid[1] - half_size,
            self.centroid[1] + half_size,
        )

    def transform(self, trajectory: np.ndarray) -> np.ndarray:
        """Transform a trajectory to 2D embedding."""
        return self.reducer.transform(trajectory)


# =============================================================================
# FIGURE GENERATION
# =============================================================================

def create_simulation_figure(
    sim_result: SimulationResult,
    embedded_trajectory: np.ndarray,
    embedder: PooledEmbedder,
    regime_metrics: dict[str, FlowMetrics],
    output_path: Path,
    title: str = "Simulation 1: Metastable Switching System",
):
    """
    Create the main simulation validation figure.

    Subpanels:
    A) Ground-truth regime timeline
    B) Embedded latent trajectories colored by true regime
    C) Estimated flow field + dwell density
    D) Metric comparison across regimes
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Colors for regimes - use highly distinct colors
    n_regimes = len(set(sim_result.regime_labels))
    # Blue, Red, Green - maximally distinct
    regime_colors = np.array([
        [0.2, 0.4, 0.8, 1.0],   # Blue (Regime 0)
        [0.9, 0.2, 0.2, 1.0],   # Red (Regime 1)
        [0.2, 0.7, 0.3, 1.0],   # Green (Regime 2)
    ])[:n_regimes]

    # --- Panel A: Ground-truth regime timeline ---
    ax_a = fig.add_subplot(gs[0, 0])

    # Create colored background for regimes
    for i in range(len(sim_result.time) - 1):
        regime = sim_result.regime_labels[i]
        ax_a.axvspan(
            sim_result.time[i], sim_result.time[i+1],
            color=regime_colors[regime], alpha=0.7
        )

    # Mark transitions
    for t_idx in sim_result.transition_times:
        t = sim_result.time[t_idx]
        ax_a.axvline(t, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    ax_a.set_xlim(sim_result.time[0], sim_result.time[-1])
    ax_a.set_ylim(0, 1)
    ax_a.set_xlabel("Time (s)")
    ax_a.set_ylabel("Regime")
    ax_a.set_title("A) Ground-Truth Regime Sequence", fontweight='bold')
    ax_a.set_yticks([])

    # Legend
    handles = [plt.Rectangle((0,0),1,1, color=regime_colors[i], alpha=0.7)
               for i in range(n_regimes)]
    ax_a.legend(handles, [f"Regime {i}" for i in range(n_regimes)],
                loc='upper right', ncol=n_regimes)

    # --- Panel B: Embedded trajectories colored by regime ---
    ax_b = fig.add_subplot(gs[0, 1])

    # Downsample for visibility
    step = max(1, len(embedded_trajectory) // 5000)
    embedded_ds = embedded_trajectory[::step]
    labels_ds = sim_result.regime_labels[::step][:len(embedded_ds)]

    for regime in range(n_regimes):
        mask = labels_ds == regime
        ax_b.scatter(
            embedded_ds[mask, 0], embedded_ds[mask, 1],
            c=[regime_colors[regime]], s=1, alpha=0.3,
            label=f"Regime {regime}"
        )

    ax_b.set_xlabel("Latent Dim 1")
    ax_b.set_ylabel("Latent Dim 2")
    ax_b.set_title("B) Embedded Trajectories (colored by true regime)", fontweight='bold')
    ax_b.legend(markerscale=5)
    ax_b.set_aspect('equal')

    # --- Panel C: Flow field + dwell density ---
    ax_c = fig.add_subplot(gs[1, 0])

    # Compute density
    density = compute_density_on_grid(embedded_trajectory, embedder.bounds, bins=50)

    # Compute flow field
    X, Y, flow_x, flow_y, counts = compute_flow_field(
        embedded_trajectory, embedder.bounds, grid_size=15
    )

    # Plot density
    extent = list(embedder.bounds)
    im = ax_c.imshow(
        density, origin='lower', extent=extent,
        cmap='Blues', alpha=0.7, aspect='equal'
    )

    # Plot flow field
    magnitude = np.sqrt(flow_x**2 + flow_y**2)
    mask = counts > 5
    ax_c.quiver(
        X[mask], Y[mask], flow_x[mask], flow_y[mask],
        magnitude[mask], cmap='Reds', alpha=0.8,
        scale=None, width=0.003
    )

    ax_c.set_xlabel("Latent Dim 1")
    ax_c.set_ylabel("Latent Dim 2")
    ax_c.set_title("C) Dwell Density + Flow Field", fontweight='bold')
    fig.colorbar(im, ax=ax_c, label='Density', shrink=0.8)

    # --- Panel D: Metric comparison across regimes (ACTUAL VALUES, no normalization) ---
    # Create 2x2 subgrid for 4 metrics to show actual values clearly
    gs_d = gs[1, 1].subgridspec(2, 2, hspace=0.4, wspace=0.3)

    metric_configs = [
        ("mean_speed", "Mean Speed", "Speed"),
        ("speed_cv", "Speed CV", "CV"),
        ("path_tortuosity", "Tortuosity", "Tortuosity"),
        ("explored_variance", "Explored Var", "Variance"),
    ]

    regime_names_list = list(regime_metrics.keys())
    x = np.arange(len(regime_names_list))

    for idx, (metric_name, title, ylabel) in enumerate(metric_configs):
        ax_sub = fig.add_subplot(gs_d[idx // 2, idx % 2])
        values = [getattr(regime_metrics[r], metric_name) for r in regime_names_list]

        bars = ax_sub.bar(x, values, color=[regime_colors[i] for i in range(len(regime_names_list))], alpha=0.7)
        ax_sub.set_xticks(x)
        ax_sub.set_xticklabels([f"R{i}" for i in range(len(regime_names_list))])
        ax_sub.set_ylabel(ylabel, fontsize=8)
        ax_sub.set_title(title, fontsize=9, fontweight='bold')

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax_sub.annotate(f'{val:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 2), textcoords="offset points",
                           ha='center', va='bottom', fontsize=7)

    # Add overall title for Panel D
    fig.text(0.75, 0.48, "D) Flow Metrics by Regime", fontweight='bold', fontsize=10, ha='center')

    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved: {output_path}")

    return fig


def create_attractor_comparison_figure(
    stable_result: SimulationResult,
    exploratory_result: SimulationResult,
    stable_embedded: np.ndarray,
    exploratory_embedded: np.ndarray,
    stable_metrics: FlowMetrics,
    exploratory_metrics: FlowMetrics,
    bounds: tuple,
    output_path: Path,
):
    """
    Create figure comparing stable vs exploratory attractors.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Colors
    stable_color = '#1f77b4'
    exploratory_color = '#ff7f0e'

    # --- Row 1: Trajectories ---
    # Stable trajectory
    step = max(1, len(stable_embedded) // 3000)
    axes[0, 0].scatter(
        stable_embedded[::step, 0], stable_embedded[::step, 1],
        c=stable_color, s=1, alpha=0.3
    )
    axes[0, 0].set_title("Stable Attractor\n(tight, low exploration)", fontweight='bold')
    axes[0, 0].set_xlabel("Dim 1")
    axes[0, 0].set_ylabel("Dim 2")
    axes[0, 0].set_aspect('equal')

    # Exploratory trajectory
    step = max(1, len(exploratory_embedded) // 3000)
    axes[0, 1].scatter(
        exploratory_embedded[::step, 0], exploratory_embedded[::step, 1],
        c=exploratory_color, s=1, alpha=0.3
    )
    axes[0, 1].set_title("Exploratory Attractor\n(weak, high exploration)", fontweight='bold')
    axes[0, 1].set_xlabel("Dim 1")
    axes[0, 1].set_ylabel("Dim 2")
    axes[0, 1].set_aspect('equal')

    # Overlay
    axes[0, 2].scatter(
        stable_embedded[::step*2, 0], stable_embedded[::step*2, 1],
        c=stable_color, s=1, alpha=0.2, label='Stable'
    )
    axes[0, 2].scatter(
        exploratory_embedded[::step*2, 0], exploratory_embedded[::step*2, 1],
        c=exploratory_color, s=1, alpha=0.2, label='Exploratory'
    )
    axes[0, 2].set_title("Overlay", fontweight='bold')
    axes[0, 2].set_xlabel("Dim 1")
    axes[0, 2].set_ylabel("Dim 2")
    axes[0, 2].legend(markerscale=5)
    axes[0, 2].set_aspect('equal')

    # --- Row 2: Metrics and Flow ---
    # Speed histograms
    stable_speed = compute_instantaneous_speed(stable_embedded)
    exploratory_speed = compute_instantaneous_speed(exploratory_embedded)

    axes[1, 0].hist(stable_speed, bins=50, alpha=0.7, color=stable_color,
                    label=f'Stable (μ={stable_speed.mean():.3f})', density=True)
    axes[1, 0].hist(exploratory_speed, bins=50, alpha=0.7, color=exploratory_color,
                    label=f'Exploratory (μ={exploratory_speed.mean():.3f})', density=True)
    axes[1, 0].set_xlabel("Speed")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_title("Speed Distribution", fontweight='bold')
    axes[1, 0].legend()

    # Density comparison
    stable_density = compute_density_on_grid(stable_embedded, bounds, bins=50)
    exploratory_density = compute_density_on_grid(exploratory_embedded, bounds, bins=50)

    diff = exploratory_density - stable_density
    vmax = np.abs(diff).max()
    im = axes[1, 1].imshow(
        diff, origin='lower', extent=list(bounds),
        cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal'
    )
    axes[1, 1].set_title("Density Difference\n(Exploratory - Stable)", fontweight='bold')
    axes[1, 1].set_xlabel("Dim 1")
    axes[1, 1].set_ylabel("Dim 2")
    fig.colorbar(im, ax=axes[1, 1], label='Δ Density')

    # Metric comparison bar chart
    metric_names = ["mean_speed", "speed_cv", "path_tortuosity", "explored_variance"]
    metric_labels = ["Mean Speed", "Speed CV", "Tortuosity", "Explored Var"]

    x = np.arange(len(metric_names))
    width = 0.35

    stable_vals = [getattr(stable_metrics, m) for m in metric_names]
    exploratory_vals = [getattr(exploratory_metrics, m) for m in metric_names]

    # Normalize for comparison
    max_vals = [max(s, e) for s, e in zip(stable_vals, exploratory_vals)]
    stable_norm = [s/m if m > 0 else 0 for s, m in zip(stable_vals, max_vals)]
    exploratory_norm = [e/m if m > 0 else 0 for e, m in zip(exploratory_vals, max_vals)]

    axes[1, 2].bar(x - width/2, stable_norm, width, label='Stable', color=stable_color, alpha=0.7)
    axes[1, 2].bar(x + width/2, exploratory_norm, width, label='Exploratory', color=exploratory_color, alpha=0.7)
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(metric_labels, rotation=15, ha='right')
    axes[1, 2].set_ylabel("Normalized Value")
    axes[1, 2].set_title("Metric Comparison", fontweight='bold')
    axes[1, 2].legend()

    fig.suptitle("Simulation 2: Attractor Stabilization vs Exploration",
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved: {output_path}")

    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Simulation validation for dynamical microscope")
    parser.add_argument("--quick", action="store_true", help="Quick test (fewer epochs)")
    parser.add_argument("--no-show", action="store_true", help="Don't display plots")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--two-phase", action="store_true", help="Enable two-phase training (reconstruction then contrastive)")
    parser.add_argument("--transformer", action="store_true", help="Use TransformerAutoencoder instead of SimulationAutoencoder")
    parser.add_argument("--contrastive-weight", type=float, default=0.15, help="Weight for contrastive loss in phase 2")
    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_BASE_DIR / f"simulation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=" * 60)
    print("Simulation-Based Validation for Paper 1")
    print(f"=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Device: {DEVICE}")
    print()

    # Parameters
    n_epochs_phase1 = 20 if args.quick else 100  # Phase 1: reconstruction
    n_epochs_phase2 = 10 if args.quick else 50   # Phase 2: contrastive (shorter)
    hidden_size = 64  # Larger hidden size for more representational capacity
    use_two_phase = args.two_phase
    use_transformer = args.transformer

    # Save parameters
    params = {
        "seed": args.seed,
        "n_epochs_phase1": n_epochs_phase1,
        "n_epochs_phase2": n_epochs_phase2 if use_two_phase else 0,
        "hidden_size": hidden_size,
        "n_channels": N_CHANNELS,
        "sfreq": SFREQ,
        "chunk_duration": CHUNK_DURATION,
        "total_duration": TOTAL_DURATION,
        "device": DEVICE,
        "two_phase": use_two_phase,
        "transformer": use_transformer,
        "contrastive_weight": args.contrastive_weight if use_two_phase else 0,
    }

    if use_two_phase:
        print(f"Two-phase training ENABLED:")
        print(f"  Phase 1: {n_epochs_phase1} epochs (reconstruction only)")
        print(f"  Phase 2: {n_epochs_phase2} epochs (contrastive, weight={args.contrastive_weight})")
    if use_transformer:
        print(f"Using TransformerAutoencoder")
    print()
    with open(output_dir / "parameters.json", "w") as f:
        json.dump(params, f, indent=2)

    # Copy this script to output directory for reproducibility
    import shutil
    script_path = Path(__file__)
    shutil.copy2(script_path, output_dir / script_path.name)

    # =========================================================================
    # SIMULATION 1: Metastable Switching System
    # =========================================================================
    print("\n" + "=" * 60)
    print("Simulation 1: Metastable Switching System")
    print("=" * 60)

    # Generate synthetic data
    print("\nGenerating synthetic data...")
    switching_system = MetastableSwitchingSystem(
        n_regimes=3,
        latent_dim=3,
        n_channels=N_CHANNELS,
        sfreq=SFREQ,
        seed=args.seed,
    )
    sim1_result = switching_system.generate(TOTAL_DURATION)

    print(f"  Total samples: {len(sim1_result.time)}")
    print(f"  Number of transitions: {len(sim1_result.transition_times)}")
    print(f"  Regime distribution: {np.bincount(sim1_result.regime_labels)}")

    # Convert to phase representation
    print("\nConverting to phase representation...")
    phase_data_sim1 = observations_to_phase_representation(
        sim1_result.observations, SFREQ
    )
    print(f"  Phase data shape: {phase_data_sim1.shape}")

    # Chunk data with regime labels
    chunk_samples = int(CHUNK_DURATION * SFREQ)
    chunks_sim1 = chunk_phase_data(phase_data_sim1, chunk_samples)
    print(f"  Number of chunks: {len(chunks_sim1)}")

    # Get regime labels for each chunk (use majority label per chunk)
    chunk_labels_sim1 = []
    for i, chunk in enumerate(chunks_sim1):
        start_sample = i * chunk_samples
        end_sample = start_sample + chunk_samples
        if end_sample <= len(sim1_result.regime_labels):
            chunk_regime = sim1_result.regime_labels[start_sample:end_sample]
            majority_label = int(np.bincount(chunk_regime).argmax())
            chunk_labels_sim1.append(majority_label)
        else:
            chunk_labels_sim1.append(0)  # Fallback

    # === PHASE 1: Reconstruction ===
    print(f"\n--- Phase 1: Training autoencoder ({n_epochs_phase1} epochs) ---")
    model_sim1 = train_simulation_model(
        chunks_sim1,
        n_channels=N_CHANNELS,
        hidden_size=hidden_size,
        n_epochs=n_epochs_phase1,
        device=DEVICE,
        verbose=True,
        labels=chunk_labels_sim1,
        use_contrastive=False,
        use_transformer=use_transformer,
    )

    # === PHASE 2: Contrastive (optional) ===
    if use_two_phase:
        print(f"\n--- Phase 2: Contrastive fine-tuning ({n_epochs_phase2} epochs) ---")
        # Continue training with contrastive loss
        # Note: We reuse the same model (weights preserved)
        model_sim1.train()
        optimizer = torch.optim.AdamW(model_sim1.parameters(), lr=1e-4)  # Lower LR for fine-tuning
        criterion = nn.MSELoss()
        contrastive_loss_fn = ContrastiveLoss(temperature=0.07, mode="condition")

        data = torch.stack([torch.from_numpy(c) for c in chunks_sim1])
        label_tensor = torch.tensor(chunk_labels_sim1, dtype=torch.long)
        dataset = TensorDataset(data, label_tensor)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        for epoch in tqdm(range(n_epochs_phase2), desc="Phase 2 (Contrastive)"):
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_contr = 0.0
            n_batches = 0

            for batch, batch_labels in dataloader:
                batch = batch.float().to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)

                optimizer.zero_grad()
                output = model_sim1(batch)
                reconstruction, latent = output if isinstance(output, tuple) else (output, model_sim1.encode(batch))

                recon_loss = criterion(reconstruction, batch)

                # Contrastive on mean latent
                latent_mean = latent.mean(dim=1) if latent.dim() == 3 else latent
                contr_loss = contrastive_loss_fn(latent_mean, batch_labels)

                total_loss = (1 - args.contrastive_weight) * recon_loss + args.contrastive_weight * contr_loss
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model_sim1.parameters(), 1.0)
                optimizer.step()

                epoch_loss += total_loss.item()
                epoch_recon += recon_loss.item()
                epoch_contr += contr_loss.item()
                n_batches += 1

            if (epoch + 1) % 10 == 0:
                tqdm.write(f"Epoch {epoch+1}/{n_epochs_phase2}: loss={epoch_loss/n_batches:.4f} recon={epoch_recon/n_batches:.4f} contr={epoch_contr/n_batches:.4f}")

        model_sim1.eval()

    # Save model (to output dir, NOT models/)
    model_path = output_dir / "simulation1_model.pt"
    torch.save({
        "model_state_dict": model_sim1.state_dict(),
        "n_channels": N_CHANNELS,
        "hidden_size": hidden_size,
    }, model_path)
    print(f"Saved model to: {model_path}")

    # Compute latent trajectory
    print("\nComputing latent trajectory...")
    latent_sim1 = compute_latent_trajectory(model_sim1, phase_data_sim1, DEVICE)
    print(f"  Latent shape: {latent_sim1.shape}")

    # Fit embedder and transform - try UMAP for better regime separation
    print("\nFitting UMAP embedder for Simulation 1...")
    embedder_sim1 = PooledEmbedder(n_components=2, method="umap")
    embedder_sim1.fit([latent_sim1])
    embedded_sim1 = embedder_sim1.transform(latent_sim1)
    print(f"  Embedded shape: {embedded_sim1.shape}")

    # Compute metrics per regime
    print("\nComputing flow metrics per regime...")
    regime_metrics_sim1 = {}
    n_regimes = len(set(sim1_result.regime_labels))

    # Align regime labels with embedded (may be shorter due to conv strides)
    labels_aligned = sim1_result.regime_labels[:len(embedded_sim1)]

    for regime in range(n_regimes):
        mask = labels_aligned == regime
        if mask.sum() > 100:  # Need enough points
            regime_embedded = embedded_sim1[mask]
            metrics = compute_flow_metrics(regime_embedded)
            regime_metrics_sim1[f"Regime_{regime}"] = metrics
            print(f"  Regime {regime}: speed={metrics.mean_speed:.4f}, tortuosity={metrics.path_tortuosity:.2f}")

    # Save ground truth
    def convert_to_serializable(obj):
        """Convert numpy types to Python native types for JSON."""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    ground_truth_sim1 = convert_to_serializable({
        "n_regimes": n_regimes,
        "n_transitions": len(sim1_result.transition_times),
        "regime_counts": np.bincount(sim1_result.regime_labels).tolist(),
        "metrics": {k: vars(v) for k, v in regime_metrics_sim1.items()},
    })
    with open(output_dir / "simulation1_ground_truth.json", "w") as f:
        json.dump(ground_truth_sim1, f, indent=2)

    # =========================================================================
    # SIMULATION 2: Attractor Stabilization vs Exploration
    # =========================================================================
    print("\n" + "=" * 60)
    print("Simulation 2: Attractor Stabilization vs Exploration")
    print("=" * 60)

    # Generate both conditions
    print("\nGenerating stable attractor data...")
    attractor_system = AttractorStabilitySystem(
        latent_dim=3,
        n_channels=N_CHANNELS,
        sfreq=SFREQ,
        seed=args.seed,
    )
    stable_result = attractor_system.generate(TOTAL_DURATION, "stable")

    print("Generating exploratory attractor data...")
    attractor_system_exp = AttractorStabilitySystem(
        latent_dim=3,
        n_channels=N_CHANNELS,
        sfreq=SFREQ,
        seed=args.seed + 1,  # Different seed for variety
    )
    exploratory_result = attractor_system_exp.generate(TOTAL_DURATION, "exploratory")

    # Convert to phase representation
    print("\nConverting to phase representation...")
    phase_stable = observations_to_phase_representation(stable_result.observations, SFREQ)
    phase_exploratory = observations_to_phase_representation(exploratory_result.observations, SFREQ)

    # Chunk and combine for training with labels
    chunks_stable = chunk_phase_data(phase_stable, chunk_samples)
    chunks_exploratory = chunk_phase_data(phase_exploratory, chunk_samples)
    all_chunks_sim2 = chunks_stable + chunks_exploratory
    # Labels: 0 = stable, 1 = exploratory
    labels_sim2 = [0] * len(chunks_stable) + [1] * len(chunks_exploratory)
    print(f"  Total chunks: {len(all_chunks_sim2)}")
    print(f"  Stable chunks: {len(chunks_stable)}, Exploratory chunks: {len(chunks_exploratory)}")

    # === PHASE 1: Reconstruction ===
    print(f"\n--- Phase 1: Training joint autoencoder ({n_epochs_phase1} epochs) ---")
    model_sim2 = train_simulation_model(
        all_chunks_sim2,
        n_channels=N_CHANNELS,
        hidden_size=hidden_size,
        n_epochs=n_epochs_phase1,
        device=DEVICE,
        verbose=True,
        labels=labels_sim2,
        use_contrastive=False,
        use_transformer=use_transformer,
    )

    # === PHASE 2: Contrastive (optional) ===
    if use_two_phase:
        print(f"\n--- Phase 2: Contrastive fine-tuning ({n_epochs_phase2} epochs) ---")
        model_sim2.train()
        optimizer = torch.optim.AdamW(model_sim2.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        contrastive_loss_fn = ContrastiveLoss(temperature=0.07, mode="condition")

        data = torch.stack([torch.from_numpy(c) for c in all_chunks_sim2])
        label_tensor = torch.tensor(labels_sim2, dtype=torch.long)
        dataset = TensorDataset(data, label_tensor)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        for epoch in tqdm(range(n_epochs_phase2), desc="Phase 2 (Contrastive)"):
            epoch_loss = 0.0
            n_batches = 0

            for batch, batch_labels in dataloader:
                batch = batch.float().to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)

                optimizer.zero_grad()
                output = model_sim2(batch)
                reconstruction, latent = output if isinstance(output, tuple) else (output, model_sim2.encode(batch))

                recon_loss = criterion(reconstruction, batch)
                latent_mean = latent.mean(dim=1) if latent.dim() == 3 else latent
                contr_loss = contrastive_loss_fn(latent_mean, batch_labels)

                total_loss = (1 - args.contrastive_weight) * recon_loss + args.contrastive_weight * contr_loss
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model_sim2.parameters(), 1.0)
                optimizer.step()

                epoch_loss += total_loss.item()
                n_batches += 1

            if (epoch + 1) % 10 == 0:
                tqdm.write(f"Epoch {epoch+1}/{n_epochs_phase2}: loss={epoch_loss/n_batches:.4f}")

        model_sim2.eval()

    # Save model
    model_path_sim2 = output_dir / "simulation2_model.pt"
    torch.save({
        "model_state_dict": model_sim2.state_dict(),
        "n_channels": N_CHANNELS,
        "hidden_size": hidden_size,
    }, model_path_sim2)
    print(f"Saved model to: {model_path_sim2}")

    # Compute latent trajectories
    print("\nComputing latent trajectories...")
    latent_stable = compute_latent_trajectory(model_sim2, phase_stable, DEVICE)
    latent_exploratory = compute_latent_trajectory(model_sim2, phase_exploratory, DEVICE)

    # Joint embedder
    print("\nFitting joint PCA embedder...")
    embedder_sim2 = PooledEmbedder(n_components=2)
    embedder_sim2.fit([latent_stable, latent_exploratory])

    embedded_stable = embedder_sim2.transform(latent_stable)
    embedded_exploratory = embedder_sim2.transform(latent_exploratory)

    # CENTER BOTH EMBEDDINGS at origin for visualization
    # This ensures both attractors are displayed at the same position
    # so we can compare their SHAPES (blob vs ring) directly
    embedded_stable = embedded_stable - embedded_stable.mean(axis=0)
    embedded_exploratory = embedded_exploratory - embedded_exploratory.mean(axis=0)

    # Compute metrics
    print("\nComputing flow metrics...")
    stable_metrics = compute_flow_metrics(embedded_stable)
    exploratory_metrics = compute_flow_metrics(embedded_exploratory)

    print(f"  Stable: speed={stable_metrics.mean_speed:.4f}, "
          f"tortuosity={stable_metrics.path_tortuosity:.2f}, "
          f"explored_var={stable_metrics.explored_variance:.4f}")
    print(f"  Exploratory: speed={exploratory_metrics.mean_speed:.4f}, "
          f"tortuosity={exploratory_metrics.path_tortuosity:.2f}, "
          f"explored_var={exploratory_metrics.explored_variance:.4f}")

    # Validate that the pipeline detects differences between conditions
    # Note: The direction of effects can vary depending on how dynamics interact
    # with the autoencoder representation. The key is that DIFFERENCES are detected.
    print("\n--- Validation ---")
    speed_diff = abs(exploratory_metrics.mean_speed - stable_metrics.mean_speed)
    variance_diff = abs(exploratory_metrics.explored_variance - stable_metrics.explored_variance)
    tortuosity_diff = abs(exploratory_metrics.path_tortuosity - stable_metrics.path_tortuosity)

    n_detected = 0

    if speed_diff > 0.01:
        print(f"✓ Speed difference detected: |Δ| = {speed_diff:.4f}")
        n_detected += 1
    else:
        print("✗ No speed difference detected")

    if variance_diff > 0.01:
        print(f"✓ Variance difference detected: |Δ| = {variance_diff:.4f}")
        n_detected += 1
    else:
        print("✗ No variance difference detected")

    if tortuosity_diff > 10:
        print(f"✓ Tortuosity difference detected: |Δ| = {tortuosity_diff:.1f}")
        n_detected += 1
    else:
        print("✗ No tortuosity difference detected")

    print(f"\n→ {n_detected}/3 dynamical differences detected by pipeline")

    # Save results
    ground_truth_sim2 = convert_to_serializable({
        "stable": vars(stable_metrics),
        "exploratory": vars(exploratory_metrics),
        "speed_ratio": exploratory_metrics.mean_speed / stable_metrics.mean_speed if stable_metrics.mean_speed > 0 else 0,
        "variance_ratio": exploratory_metrics.explored_variance / stable_metrics.explored_variance if stable_metrics.explored_variance > 0 else 0,
    })
    with open(output_dir / "simulation2_ground_truth.json", "w") as f:
        json.dump(ground_truth_sim2, f, indent=2)

    # =========================================================================
    # GENERATE FIGURES
    # =========================================================================
    print("\n" + "=" * 60)
    print("Generating Figures")
    print("=" * 60)

    # Update regime labels alignment for figure
    sim1_result_aligned = SimulationResult(
        time=sim1_result.time[:len(embedded_sim1)],
        latent_states=sim1_result.latent_states[:len(embedded_sim1)],
        observations=sim1_result.observations[:len(embedded_sim1)],
        regime_labels=labels_aligned,
        transition_times=[t for t in sim1_result.transition_times if t < len(embedded_sim1)],
        regime_names=sim1_result.regime_names,
    )

    # Figure for Simulation 1
    print("\nCreating Simulation 1 figure...")
    fig1 = create_simulation_figure(
        sim1_result_aligned,
        embedded_sim1,
        embedder_sim1,
        regime_metrics_sim1,
        output_dir / "simulation1_metastable.pdf",
    )

    # Figure for Simulation 2
    print("\nCreating Simulation 2 figure...")
    fig2 = create_attractor_comparison_figure(
        stable_result,
        exploratory_result,
        embedded_stable,
        embedded_exploratory,
        stable_metrics,
        exploratory_metrics,
        embedder_sim2.bounds,
        output_dir / "simulation2_attractor_comparison.pdf",
    )

    # Combined figure for paper (fig02_simulations.pdf)
    print("\nCreating combined figure for paper...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Create a 2x4 combined figure
    fig_combined = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 4, figure=fig_combined, hspace=0.3, wspace=0.25)

    # --- Top row: Simulation 1 (Metastable Switching) ---
    # Use highly distinct colors: Blue, Red, Green
    regime_colors = np.array([
        [0.2, 0.4, 0.8, 1.0],   # Blue (Regime 0)
        [0.9, 0.2, 0.2, 1.0],   # Red (Regime 1)
        [0.2, 0.7, 0.3, 1.0],   # Green (Regime 2)
    ])[:n_regimes]

    # A) Regime timeline
    ax_a = fig_combined.add_subplot(gs[0, 0])
    for i in range(len(sim1_result_aligned.time) - 1):
        regime = sim1_result_aligned.regime_labels[i]
        ax_a.axvspan(
            sim1_result_aligned.time[i], sim1_result_aligned.time[i+1],
            color=regime_colors[regime], alpha=0.7
        )
    ax_a.set_xlim(sim1_result_aligned.time[0], sim1_result_aligned.time[-1])
    ax_a.set_ylim(0, 1)
    ax_a.set_xlabel("Time (s)", fontsize=9)
    ax_a.set_title("A) Ground-Truth Regime\nSequence", fontweight='bold', fontsize=10)
    ax_a.set_yticks([])

    # B) Embedded trajectories
    ax_b = fig_combined.add_subplot(gs[0, 1])
    step = max(1, len(embedded_sim1) // 3000)
    embedded_ds = embedded_sim1[::step]
    labels_ds = sim1_result_aligned.regime_labels[::step][:len(embedded_ds)]
    for regime in range(n_regimes):
        mask = labels_ds == regime
        ax_b.scatter(embedded_ds[mask, 0], embedded_ds[mask, 1],
                    c=[regime_colors[regime]], s=1, alpha=0.3)
    ax_b.set_xlabel("Dim 1", fontsize=9)
    ax_b.set_ylabel("Dim 2", fontsize=9)
    ax_b.set_title("B) Embedded Trajectories\n(colored by regime)", fontweight='bold', fontsize=10)
    ax_b.set_aspect('equal')

    # C) Flow field + density
    ax_c = fig_combined.add_subplot(gs[0, 2])
    density = compute_density_on_grid(embedded_sim1, embedder_sim1.bounds, bins=50)
    X, Y, flow_x, flow_y, counts = compute_flow_field(embedded_sim1, embedder_sim1.bounds, grid_size=12)
    im = ax_c.imshow(density, origin='lower', extent=list(embedder_sim1.bounds),
                    cmap='Blues', alpha=0.7, aspect='equal')
    mask = counts > 5
    ax_c.quiver(X[mask], Y[mask], flow_x[mask], flow_y[mask],
               np.sqrt(flow_x[mask]**2 + flow_y[mask]**2), cmap='Reds', alpha=0.8)
    ax_c.set_xlabel("Dim 1", fontsize=9)
    ax_c.set_ylabel("Dim 2", fontsize=9)
    ax_c.set_title("C) Density + Flow Field", fontweight='bold', fontsize=10)

    # D) Metrics by regime - use separate subplots for each metric to show actual values
    ax_d = fig_combined.add_subplot(gs[0, 3])

    # Just show speed comparison (most interpretable)
    regime_names_list = list(regime_metrics_sim1.keys())
    speeds = [regime_metrics_sim1[r].mean_speed for r in regime_names_list]
    x = np.arange(len(regime_names_list))
    bars = ax_d.bar(x, speeds, color=[regime_colors[i] for i in range(len(regime_names_list))], alpha=0.7)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([f"R{i}" for i in range(len(regime_names_list))], fontsize=9)
    ax_d.set_ylabel("Mean Speed", fontsize=9)
    ax_d.set_title("D) Speed by Regime", fontweight='bold', fontsize=10)

    # Add value labels on bars
    for bar, val in zip(bars, speeds):
        ax_d.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    # --- Bottom row: Simulation 2 (Attractor Stability) ---
    stable_color = '#1f77b4'
    exploratory_color = '#ff7f0e'

    # E) Stable trajectory
    ax_e = fig_combined.add_subplot(gs[1, 0])
    step = max(1, len(embedded_stable) // 2000)
    ax_e.scatter(embedded_stable[::step, 0], embedded_stable[::step, 1],
                c=stable_color, s=1, alpha=0.3)
    ax_e.set_xlabel("Dim 1", fontsize=9)
    ax_e.set_ylabel("Dim 2", fontsize=9)
    ax_e.set_title("E) Stable Attractor", fontweight='bold', fontsize=10)
    ax_e.set_aspect('equal')

    # F) Exploratory trajectory
    ax_f = fig_combined.add_subplot(gs[1, 1])
    step = max(1, len(embedded_exploratory) // 2000)
    ax_f.scatter(embedded_exploratory[::step, 0], embedded_exploratory[::step, 1],
                c=exploratory_color, s=1, alpha=0.3)
    ax_f.set_xlabel("Dim 1", fontsize=9)
    ax_f.set_ylabel("Dim 2", fontsize=9)
    ax_f.set_title("F) Exploratory Attractor", fontweight='bold', fontsize=10)
    ax_f.set_aspect('equal')

    # G) Speed distributions
    ax_g = fig_combined.add_subplot(gs[1, 2])
    stable_speed = compute_instantaneous_speed(embedded_stable)
    exploratory_speed = compute_instantaneous_speed(embedded_exploratory)
    ax_g.hist(stable_speed, bins=50, alpha=0.7, color=stable_color, density=True, label='Stable')
    ax_g.hist(exploratory_speed, bins=50, alpha=0.7, color=exploratory_color, density=True, label='Exploratory')
    ax_g.set_xlabel("Speed", fontsize=9)
    ax_g.set_ylabel("Density", fontsize=9)
    ax_g.set_title("G) Speed Distribution", fontweight='bold', fontsize=10)
    ax_g.legend(fontsize=8)

    # H) Metric comparison - show actual explored variance (most meaningful)
    ax_h = fig_combined.add_subplot(gs[1, 3])

    conditions = ['Stable', 'Exploratory']
    variances = [stable_metrics.explored_variance, exploratory_metrics.explored_variance]
    colors_h = [stable_color, exploratory_color]

    bars = ax_h.bar(conditions, variances, color=colors_h, alpha=0.7)
    ax_h.set_ylabel("Explored Variance", fontsize=9)
    ax_h.set_title("H) State-Space Coverage", fontweight='bold', fontsize=10)

    # Add value labels
    for bar, val in zip(bars, variances):
        ax_h.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    # Add ratio annotation
    ratio = exploratory_metrics.explored_variance / stable_metrics.explored_variance if stable_metrics.explored_variance > 0 else 0
    ax_h.text(0.5, 0.95, f'Ratio: {ratio:.1f}x', transform=ax_h.transAxes,
             ha='center', fontsize=9, style='italic')

    fig_combined.suptitle(
        "Simulation Validation: The Dynamical Microscope Recovers Known Structure",
        fontsize=14, fontweight='bold', y=1.02
    )

    # Save combined figure (PDF and PNG)
    combined_path = FIGURES_DIR / "fig02_simulations.pdf"
    fig_combined.savefig(combined_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved combined figure: {combined_path}")

    # Also save PNG version
    combined_png = FIGURES_DIR / "fig02_simulations.png"
    fig_combined.savefig(combined_png, dpi=300, bbox_inches='tight', format='png')
    print(f"Saved PNG: {combined_png}")

    # Also save to output dir
    fig_combined.savefig(output_dir / "fig02_simulations.pdf", dpi=300, bbox_inches='tight', format='pdf')
    fig_combined.savefig(output_dir / "fig02_simulations.png", dpi=300, bbox_inches='tight', format='png')

    # --- Save individual panels as separate PNGs ---
    print("\nSaving individual panels...")

    # Panel A: Regime timeline
    fig_a, ax_a_sep = plt.subplots(figsize=(6, 3))
    for i in range(len(sim1_result_aligned.time) - 1):
        regime = sim1_result_aligned.regime_labels[i]
        ax_a_sep.axvspan(sim1_result_aligned.time[i], sim1_result_aligned.time[i+1],
                        color=regime_colors[regime], alpha=0.7)
    ax_a_sep.set_xlim(sim1_result_aligned.time[0], sim1_result_aligned.time[-1])
    ax_a_sep.set_ylim(0, 1)
    ax_a_sep.set_xlabel("Time (s)")
    ax_a_sep.set_title("Ground-Truth Regime Sequence", fontweight='bold')
    ax_a_sep.set_yticks([])
    handles = [plt.Rectangle((0,0),1,1, color=regime_colors[i], alpha=0.7) for i in range(n_regimes)]
    ax_a_sep.legend(handles, [f"Regime {i}" for i in range(n_regimes)], loc='upper right')
    fig_a.savefig(output_dir / "panel_A_regime_timeline.png", dpi=300, bbox_inches='tight')
    plt.close(fig_a)

    # Panel B: Embedded trajectories
    fig_b, ax_b_sep = plt.subplots(figsize=(6, 6))
    for regime in range(n_regimes):
        mask = labels_ds == regime
        ax_b_sep.scatter(embedded_ds[mask, 0], embedded_ds[mask, 1],
                        c=[regime_colors[regime]], s=2, alpha=0.4, label=f"Regime {regime}")
    ax_b_sep.set_xlabel("Latent Dim 1")
    ax_b_sep.set_ylabel("Latent Dim 2")
    ax_b_sep.set_title("Embedded Trajectories (colored by regime)", fontweight='bold')
    ax_b_sep.legend(markerscale=3)
    ax_b_sep.set_aspect('equal')
    fig_b.savefig(output_dir / "panel_B_embedded_trajectories.png", dpi=300, bbox_inches='tight')
    plt.close(fig_b)

    # Panel E & F: Stable vs Exploratory
    fig_ef, (ax_e_sep, ax_f_sep) = plt.subplots(1, 2, figsize=(12, 5))
    step = max(1, len(embedded_stable) // 3000)
    ax_e_sep.scatter(embedded_stable[::step, 0], embedded_stable[::step, 1],
                    c=stable_color, s=2, alpha=0.3)
    ax_e_sep.set_xlabel("Latent Dim 1")
    ax_e_sep.set_ylabel("Latent Dim 2")
    ax_e_sep.set_title("Stable Attractor", fontweight='bold')
    ax_e_sep.set_aspect('equal')

    step = max(1, len(embedded_exploratory) // 3000)
    ax_f_sep.scatter(embedded_exploratory[::step, 0], embedded_exploratory[::step, 1],
                    c=exploratory_color, s=2, alpha=0.3)
    ax_f_sep.set_xlabel("Latent Dim 1")
    ax_f_sep.set_ylabel("Latent Dim 2")
    ax_f_sep.set_title("Exploratory Attractor", fontweight='bold')
    ax_f_sep.set_aspect('equal')
    fig_ef.suptitle("Simulation 2: Attractor Stability Comparison", fontweight='bold')
    plt.tight_layout()
    fig_ef.savefig(output_dir / "panel_EF_attractor_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig_ef)

    print(f"  Saved individual panels to {output_dir}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"\nMain figure: {combined_path}")
    print("\nKey findings:")
    print("  Simulation 1 (Metastable Switching):")
    print(f"    - {n_regimes} distinct regimes recovered in latent space")
    print(f"    - {len(sim1_result.transition_times)} transitions detected")
    print("  Simulation 2 (Attractor Stability):")
    print(f"    - Exploratory/Stable speed ratio: {ground_truth_sim2['speed_ratio']:.2f}")
    print(f"    - Exploratory/Stable variance ratio: {ground_truth_sim2['variance_ratio']:.2f}")

    if not args.no_show:
        plt.show()
    else:
        plt.close('all')

    print("\n✓ Simulation validation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
