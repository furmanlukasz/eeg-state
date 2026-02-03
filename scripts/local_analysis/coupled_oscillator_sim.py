"""
coupled_oscillator_sim.py

Coupled Stuart–Landau (Hopf normal form) oscillator network simulator with
switching coupling topology, designed for regime-switching validation studies.

Key design goals:
- Multivariate oscillatory dynamics with explicit phase–amplitude structure.
- Regime switching implemented via coupling topology (adjacency/Laplacian).
- Ground-truth regime labels and transition times returned.
- Simple linear observation model (mixing to "EEG channels").
- Euler–Maruyama integration (SDE with additive Gaussian noise).

This module is intentionally lightweight and self-contained so it can be dropped
into existing pipelines (Hilbert features, autoencoder, flow metrics).

Integration with simulation_analysis.py:
- Use `to_legacy_format()` to convert SimulationResult for compatibility
- Use `observations_to_phase_representation()` from simulation_analysis.py
- Train with `train_simulation_model()` from simulation_analysis.py

Usage:
    from coupled_oscillator_sim import CoupledStuartLandauNetwork, to_legacy_format

    net = CoupledStuartLandauNetwork(n_oscillators=30, n_channels=30)
    net.default_topologies()
    result = net.generate(total_duration_s=180.0)

    # Convert for use with existing pipeline
    legacy_result = to_legacy_format(result)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, NamedTuple

import numpy as np
from scipy.signal import butter, lfilter


# -------------------------
# Results container
# -------------------------

@dataclass
class SimulationResult:
    """Container for simulated data and ground truth."""
    y: np.ndarray                 # (n_channels, n_samples) real-valued observations
    z: np.ndarray                 # (n_oscillators, n_samples) complex oscillator states
    t: np.ndarray                 # (n_samples,) time vector in seconds
    regime_names: List[str]       # unique regime names in order of appearance
    regime_id: np.ndarray         # (n_samples,) integer regime index per sample
    switch_times: List[float]     # switch start times in seconds
    params: Dict[str, object]     # dict of simulation parameters


# -------------------------
# Coupling topology helpers
# -------------------------

def _normalize_adjacency(A: np.ndarray, mode: str = "mean_degree") -> np.ndarray:
    """
    Normalize adjacency to keep coupling stable across topologies.

    IMPORTANT: "max_eig" normalization equalizes different topologies, erasing
    the dynamical impact of topology structure. Use "mean_degree" instead to
    preserve topology-dependent dynamics while keeping coupling stable.

    mode:
      - "mean_degree": divide by mean degree (RECOMMENDED - preserves topology structure)
      - "max_eig": divide by largest eigenvalue magnitude (equalizes topologies - NOT recommended)
      - "row": row-stochastic (rows sum to 1 where possible).
      - "none": no normalization.
    """
    A = np.asarray(A, dtype=float)
    if mode == "none":
        return A
    if mode == "row":
        row_sums = A.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return A / row_sums
    if mode == "mean_degree":
        # Normalize by mean degree - preserves topology structure differences
        mean_degree = A.sum() / A.shape[0]
        if mean_degree == 0:
            mean_degree = 1.0
        return A / mean_degree
    if mode == "max_eig":
        # WARNING: This equalizes different topologies, erasing dynamical differences
        w = np.linalg.eigvals(A)
        lam = np.max(np.abs(w)) if w.size else 1.0
        if lam == 0:
            lam = 1.0
        return A / lam
    raise ValueError(f"Unknown normalization mode: {mode}")


def adjacency_global(n: int, self_loops: bool = False) -> np.ndarray:
    """All-to-all adjacency."""
    A = np.ones((n, n), dtype=float)
    if not self_loops:
        np.fill_diagonal(A, 0.0)
    return A


def adjacency_clusters(n: int, n_clusters: int = 3, p_in: float = 1.0, p_out: float = 0.01,
                       seed: Optional[int] = None) -> np.ndarray:
    """
    Block-structured adjacency: dense within clusters, very sparse between clusters.

    For strong modularity contrast (to produce distinct multi-cluster synchrony):
    - p_in: probability of within-cluster connection (default 1.0 = fully connected)
    - p_out: probability of between-cluster connection (default 0.01 = very sparse)

    This creates clusters that synchronize internally but weakly interact externally.
    """
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n), dtype=float)
    # Assign nodes to clusters as evenly as possible
    labels = np.repeat(np.arange(n_clusters), np.ceil(n / n_clusters).astype(int))[:n]
    rng.shuffle(labels)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if labels[i] == labels[j]:
                if rng.random() < p_in:
                    A[i, j] = 1.0
            else:
                if rng.random() < p_out:
                    A[i, j] = 1.0
    return A


def adjacency_sparse(n: int, density: float = 0.05, directed: bool = False,
                     seed: Optional[int] = None) -> np.ndarray:
    """Random sparse adjacency (Erdős–Rényi)."""
    rng = np.random.default_rng(seed)
    A = (rng.random((n, n)) < density).astype(float)
    np.fill_diagonal(A, 0.0)
    if not directed:
        A = np.maximum(A, A.T)
    return A


def adjacency_ring(n: int, k_neighbors: int = 2, directed: bool = True) -> np.ndarray:
    """
    Ring lattice: each node connects to k_neighbors on each side.

    For traveling wave dynamics, use directed=True (default) which creates
    asymmetric coupling that promotes directional wave propagation.

    Args:
        n: Number of nodes
        k_neighbors: Number of neighbors on each side
        directed: If True (default), creates asymmetric ring for traveling waves
    """
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        for d in range(1, k_neighbors + 1):
            j1 = (i + d) % n
            j2 = (i - d) % n
            A[i, j1] = 1.0
            A[i, j2] = 1.0
    if not directed:
        A = np.maximum(A, A.T)
    return A


def laplacian_from_adjacency(A: np.ndarray) -> np.ndarray:
    """Graph Laplacian L = D - A (for diffusive coupling)."""
    A = np.asarray(A, dtype=float)
    deg = np.sum(A, axis=1)
    return np.diag(deg) - A


def compute_laplacian_spectrum(L: np.ndarray) -> Dict[str, float]:
    """
    Compute spectral properties of the graph Laplacian.

    Key properties:
    - lambda_2 (algebraic connectivity / Fiedler value):
      Larger = more connected/synchronizable
    - lambda_max: Largest eigenvalue, affects stability
    - spectral_gap: lambda_2 / lambda_max, measures synchronization efficiency
    - spectral_width: std of eigenvalues, measures spectral spread

    For the 4 topologies:
    - Global: High lambda_2, small spectral gap (all eigenvalues similar)
    - Cluster: Moderate lambda_2 with gap structure (clusters appear as near-zero eigenvalues)
    - Sparse: Low lambda_2 (poor connectivity)
    - Ring: Low lambda_2 but structured spectrum (traveling wave modes)

    Args:
        L: Graph Laplacian matrix (n x n)

    Returns:
        Dict with spectral properties
    """
    # Compute eigenvalues (Laplacian is symmetric for undirected graphs)
    # For directed graphs, we use the symmetric part for spectral analysis
    L_sym = (L + L.T) / 2
    eigenvalues = np.linalg.eigvalsh(L_sym)
    eigenvalues = np.sort(np.real(eigenvalues))

    # Remove numerical noise around zero
    eigenvalues = np.where(np.abs(eigenvalues) < 1e-10, 0, eigenvalues)

    # Lambda_2: algebraic connectivity (second smallest eigenvalue)
    # First eigenvalue is always 0 for connected graphs
    lambda_2 = eigenvalues[1] if len(eigenvalues) > 1 else 0.0

    # Lambda_max
    lambda_max = eigenvalues[-1] if len(eigenvalues) > 0 else 1.0

    # Spectral gap ratio
    spectral_gap = lambda_2 / lambda_max if lambda_max > 0 else 0.0

    # Spectral width (std of non-zero eigenvalues)
    nonzero_eigs = eigenvalues[eigenvalues > 1e-10]
    spectral_width = float(np.std(nonzero_eigs)) if len(nonzero_eigs) > 0 else 0.0

    # Number of near-zero eigenvalues (indicates number of connected components)
    n_components = int(np.sum(eigenvalues < 1e-10))

    return {
        "lambda_2": float(lambda_2),
        "lambda_max": float(lambda_max),
        "spectral_gap": float(spectral_gap),
        "spectral_width": float(spectral_width),
        "n_components": n_components,
        "eigenvalues": eigenvalues.tolist(),
    }


def analyze_topology_spectra(topologies: Dict[str, np.ndarray],
                             laplacians: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    Analyze spectral properties of all registered topologies.

    Args:
        topologies: Dict mapping name -> adjacency matrix
        laplacians: Dict mapping name -> Laplacian matrix

    Returns:
        Dict mapping name -> spectral properties
    """
    spectra = {}
    for name, L in laplacians.items():
        spectra[name] = compute_laplacian_spectrum(L)

        # Also add basic adjacency stats
        A = topologies.get(name)
        if A is not None:
            spectra[name]["mean_degree"] = float(np.mean(np.sum(A, axis=1)))
            spectra[name]["density"] = float(np.sum(A) / (A.shape[0] * (A.shape[0] - 1)))

    return spectra


# -------------------------
# Stuart–Landau dynamics
# -------------------------

def euler_maruyama_step(
    z: np.ndarray,
    mu: np.ndarray,
    omega: np.ndarray,
    dt: float,
    noise_std: float,
    L: Optional[np.ndarray] = None,
    coupling_strength: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    One Euler–Maruyama step for coupled Stuart–Landau oscillators in complex form.

    dz = [(mu + i*omega) z - |z|^2 z - coupling_strength * (L z)] dt + noise_std * sqrt(dt) dW

    We use Laplacian coupling: -L z = A z - D z, equivalent to sum_k A_jk (z_k - z_j).

    Args:
        z: (n,) complex
        mu: (n,) float
        omega: (n,) float
        dt: timestep (s)
        noise_std: std of additive complex noise (applied to real and imag independently)
        L: (n,n) Laplacian matrix, or None for no coupling
        coupling_strength: global scaling of coupling
        rng: numpy Generator

    Returns:
        z_next: (n,) complex
    """
    if rng is None:
        rng = np.random.default_rng()

    # Intrinsic drift
    drift = (mu + 1j * omega) * z - (np.abs(z) ** 2) * z

    # Coupling drift
    if L is not None and coupling_strength != 0.0:
        drift = drift - coupling_strength * (L @ z)

    # Additive complex noise (real + imag)
    if noise_std > 0:
        dW = (rng.normal(size=z.shape) + 1j * rng.normal(size=z.shape)) * np.sqrt(dt)
        z = z + drift * dt + noise_std * dW
    else:
        z = z + drift * dt

    return z


# -------------------------
# Main simulator class
# -------------------------

class CoupledStuartLandauNetwork:
    """
    Coupled Stuart–Landau oscillator network with switching coupling topology.

    Regime switching is implemented by switching the coupling Laplacian L(t),
    optionally with smooth transitions (linear interpolation between L matrices).

    Notes on interpretation:
    - With mu > 0, each oscillator tends toward a limit cycle with radius ~sqrt(mu).
    - Coupling can synchronize phases/amplitudes, reduce exploration, and change flow geometry.
    """

    def __init__(
        self,
        n_oscillators: int = 30,
        n_channels: int = 30,
        sfreq: float = 250.0,
        seed: Optional[int] = 0,
        mixing: str = "random",
        mixing_scale: float = 1.0,
    ) -> None:
        self.n_osc = int(n_oscillators)
        self.n_ch = int(n_channels)
        self.sfreq = float(sfreq)
        self.dt = 1.0 / self.sfreq
        self.rng = np.random.default_rng(seed)

        # Mixing matrix W: (n_channels, n_oscillators)
        if mixing == "identity":
            if self.n_ch != self.n_osc:
                raise ValueError("mixing='identity' requires n_channels == n_oscillators")
            W = np.eye(self.n_ch, dtype=float)
        elif mixing == "random":
            W = self.rng.normal(size=(self.n_ch, self.n_osc))
            # Column-normalize to avoid a few oscillators dominating
            col_norm = np.linalg.norm(W, axis=0, keepdims=True)
            col_norm[col_norm == 0] = 1.0
            W = W / col_norm
        else:
            raise ValueError("mixing must be 'random' or 'identity'")
        self.W = mixing_scale * W

        self._topologies: Dict[str, np.ndarray] = {}   # adjacency
        self._laplacians: Dict[str, np.ndarray] = {}   # laplacian

    def set_topologies(self, topologies: Dict[str, np.ndarray], normalize: str = "mean_degree") -> None:
        """
        Register coupling topologies by name.

        Args:
            topologies: dict mapping name -> adjacency matrix A (n_osc x n_osc)
            normalize: adjacency normalization mode ("mean_degree" recommended, "max_eig", "row", "none")
        """
        self._topologies = {}
        self._laplacians = {}
        for name, A in topologies.items():
            A = np.asarray(A, dtype=float)
            if A.shape != (self.n_osc, self.n_osc):
                raise ValueError(f"Topology '{name}' has shape {A.shape}, expected {(self.n_osc, self.n_osc)}")
            A = _normalize_adjacency(A, mode=normalize)
            L = laplacian_from_adjacency(A)
            self._topologies[name] = A
            self._laplacians[name] = L

    def default_topologies(self, seed: Optional[int] = None) -> None:
        """
        Create a standard set of four named topologies with HIGH CONTRAST.

        - global: all-to-all (promotes full synchronization)
        - cluster: strongly modular (3 clusters, tight within, sparse between)
        - sparse: very sparse random (promotes desynchronization)
        - ring: directed ring (promotes traveling waves)

        Uses mean_degree normalization to preserve topology structure differences.
        """
        if seed is None:
            seed = int(self.rng.integers(0, 10_000_000))
        tops = {
            "global": adjacency_global(self.n_osc, self_loops=False),
            "cluster": adjacency_clusters(self.n_osc, n_clusters=3, p_in=1.0, p_out=0.01, seed=seed),
            "sparse": adjacency_sparse(self.n_osc, density=0.03, directed=False, seed=seed + 1),
            "ring": adjacency_ring(self.n_osc, k_neighbors=2, directed=True),  # Directed for waves
        }
        self.set_topologies(tops, normalize="mean_degree")

    def generate(
        self,
        total_duration_s: float = 180.0,
        regime_schedule: Optional[List[Tuple[str, float]]] = None,
        mu_mean: float = 1.0,
        mu_std: float = 0.2,
        omega_mean_hz: float = 10.0,
        omega_std_hz: float = 2.0,
        omega_gradient_hz: float = 2.0,
        coupling_strength: float = 0.5,
        noise_std: float = 0.1,
        obs_noise_std: float = 0.0,
        obs_noise_color: float = 0.0,
        transition_s: float = 0.0,
        z0: Optional[np.ndarray] = None,
    ) -> SimulationResult:
        """
        Generate time series with scheduled regime switches.

        Args:
            total_duration_s: total duration in seconds
            regime_schedule: list of (regime_name, duration_s). If None, cycles through
                all registered topologies with equal duration.
            mu_mean, mu_std: oscillator mu distribution (controls limit cycle radius)
            omega_mean_hz, omega_std_hz: oscillator frequency distribution in Hz
            omega_gradient_hz: frequency gradient range for ring topology (creates traveling waves)
                The ring regime will have frequencies from omega_mean - gradient/2 to omega_mean + gradient/2
                arranged spatially around the ring to promote directional wave propagation.
            coupling_strength: scaling for Laplacian coupling
            noise_std: additive complex noise std (per sqrt(second))
            obs_noise_std: additive observation noise on y (white noise)
            obs_noise_color: if >0, use 1/f^alpha colored noise instead of white
                (1.0 = pink noise, 2.0 = brown noise, typical EEG ~1.0-1.5)
            transition_s: if >0, smoothly interpolate Laplacians over this duration at each switch
            z0: optional initial complex state (n_osc,)

        Returns:
            SimulationResult
        """
        if not self._laplacians:
            raise RuntimeError("No topologies set. Call default_topologies() or set_topologies().")

        n_steps = int(np.round(total_duration_s * self.sfreq))
        t = np.arange(n_steps) / self.sfreq

        # Regime schedule
        if regime_schedule is None:
            names = list(self._laplacians.keys())
            per = total_duration_s / len(names)
            regime_schedule = [(nm, per) for nm in names]

        # Expand schedule into per-sample regime id, keeping exact duration in samples
        regime_names: List[str] = []
        regime_id = np.zeros(n_steps, dtype=int)
        switch_times: List[float] = [0.0]
        cursor = 0
        for nm, dur_s in regime_schedule:
            if nm not in self._laplacians:
                raise ValueError(f"Regime '{nm}' not found in topologies.")
            n = int(np.round(dur_s * self.sfreq))
            if n <= 0:
                continue
            end = min(n_steps, cursor + n)
            if cursor >= n_steps:
                break
            if (not regime_names) or (regime_names[-1] != nm):
                regime_names.append(nm)
            rid = len(regime_names) - 1
            regime_id[cursor:end] = rid
            cursor = end
            if cursor < n_steps:
                switch_times.append(cursor / self.sfreq)

        # If schedule shorter than total, pad with last regime
        if cursor < n_steps:
            regime_id[cursor:] = regime_id[cursor - 1] if cursor > 0 else 0

        # Per-oscillator parameters
        mu = self.rng.normal(loc=mu_mean, scale=mu_std, size=self.n_osc)
        # Keep mu positive by default (sustained oscillations), but allow small negative if desired.
        # If you want strictly positive, uncomment:
        # mu = np.clip(mu, 0.05, None)

        # Base frequencies (random around mean)
        omega_base = 2 * np.pi * self.rng.normal(loc=omega_mean_hz, scale=omega_std_hz, size=self.n_osc)  # rad/s

        # Create frequency gradient for ring topology (promotes traveling waves)
        # Frequencies arranged spatially: low -> high around the ring
        omega_gradient = 2 * np.pi * np.linspace(-omega_gradient_hz/2, omega_gradient_hz/2, self.n_osc)

        # Create per-regime omega arrays
        # For ring: use gradient to promote traveling waves
        # For others: use random base frequencies
        omega_per_regime = {}
        for name in regime_names:
            if name == "ring":
                # Use gradient arrangement for traveling waves
                omega_per_regime[name] = 2 * np.pi * omega_mean_hz + omega_gradient
            else:
                # Use random frequencies for other regimes
                omega_per_regime[name] = omega_base

        # Default omega (will be switched per regime)
        omega = omega_base

        # Initialize states near limit cycle
        if z0 is None:
            r0 = np.sqrt(np.maximum(mu, 0.05))
            phase0 = self.rng.uniform(0, 2*np.pi, size=self.n_osc)
            z = r0 * np.exp(1j * phase0)
        else:
            z = np.asarray(z0, dtype=complex).copy()
            if z.shape != (self.n_osc,):
                raise ValueError(f"z0 must have shape {(self.n_osc,)}, got {z.shape}")

        Z = np.zeros((self.n_osc, n_steps), dtype=np.complex128)

        # Precompute Laplacians in the schedule order for interpolation
        unique_L = [self._laplacians[nm] for nm in regime_names]

        trans_steps = int(np.round(transition_s * self.sfreq)) if transition_s > 0 else 0

        # Track transition state for smooth interpolation
        in_transition = False
        transition_counter = 0
        prev_L = None
        target_L = None

        for i in range(n_steps):
            rid = regime_id[i]
            regime_name = regime_names[rid]
            L_current = unique_L[rid]

            # Get regime-specific omega (for ring: uses gradient for waves)
            omega_current = omega_per_regime[regime_name]

            # Smooth transitions by interpolating between previous and current Laplacian
            if trans_steps > 0 and i > 0:
                prev_rid = regime_id[i - 1]
                if prev_rid != rid and not in_transition:
                    # Start of a new transition
                    in_transition = True
                    transition_counter = 0
                    prev_L = unique_L[prev_rid]
                    target_L = L_current
                    prev_omega = omega_per_regime[regime_names[prev_rid]]
                    target_omega = omega_current

                if in_transition:
                    transition_counter += 1
                    # Smooth S-curve interpolation (sigmoid-like)
                    alpha = min(1.0, transition_counter / trans_steps)
                    alpha = 0.5 * (1 + np.tanh(4 * (alpha - 0.5)))  # Smooth S-curve
                    L = (1 - alpha) * prev_L + alpha * target_L
                    omega = (1 - alpha) * prev_omega + alpha * target_omega

                    if transition_counter >= trans_steps:
                        in_transition = False
                else:
                    L = L_current
                    omega = omega_current
            else:
                L = L_current
                omega = omega_current

            Z[:, i] = z
            z = euler_maruyama_step(
                z=z,
                mu=mu,
                omega=omega,
                dt=self.dt,
                noise_std=noise_std,
                L=L,
                coupling_strength=coupling_strength,
                rng=self.rng,
            )

        # Observation model: y = W @ Re(z) + noise
        y = self.W @ np.real(Z)
        if obs_noise_std > 0:
            if obs_noise_color > 0:
                # Use colored (1/f^alpha) noise for more realistic EEG-like background
                colored = generate_colored_noise(
                    n_samples=n_steps,
                    n_channels=self.n_ch,
                    sfreq=self.sfreq,
                    alpha=obs_noise_color,
                    scale=obs_noise_std,
                    seed=int(self.rng.integers(0, 10_000_000)),
                )
                y = y + colored
            else:
                # White noise
                y = y + self.rng.normal(scale=obs_noise_std, size=y.shape)

        params = dict(
            n_oscillators=self.n_osc,
            n_channels=self.n_ch,
            sfreq=self.sfreq,
            dt=self.dt,
            mu_mean=mu_mean,
            mu_std=mu_std,
            omega_mean_hz=omega_mean_hz,
            omega_std_hz=omega_std_hz,
            coupling_strength=coupling_strength,
            noise_std=noise_std,
            obs_noise_std=obs_noise_std,
            obs_noise_color=obs_noise_color,
            transition_s=transition_s,
            topologies=list(self._laplacians.keys()),
            regime_schedule=regime_schedule,
        )

        return SimulationResult(
            y=y,
            z=Z,
            t=t,
            regime_names=regime_names,
            regime_id=regime_id,
            switch_times=switch_times,
            params=params,
        )


# -------------------------
# Simple visualization helper (optional)
# -------------------------

def plot_electrode_timeseries(
    result: SimulationResult,
    channels: List[int] = [0, 5, 15, 20, 28],
    time_window: Tuple[float, float] = (0.0, 30.0),
    nfft: int = 2048,
) -> "plt.Figure":
    """
    Multi-panel visualization:
      A) Raw time series (selected channels) with regime segments indicated
      B) Power spectral density (simple periodogram)
      C) Hilbert amplitude envelope (per-channel) for one representative channel
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import hilbert

    fs = result.params["sfreq"]
    t = result.t
    y = result.y

    t0, t1 = time_window
    i0 = int(max(0, np.floor(t0 * fs)))
    i1 = int(min(y.shape[1], np.ceil(t1 * fs)))

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1.2, 1.2], hspace=0.35)

    # Panel A
    ax1 = fig.add_subplot(gs[0, 0])
    offset = 0.0
    for ch in channels:
        sig = y[ch, i0:i1]
        ax1.plot(t[i0:i1], sig + offset, lw=1.0)
        ax1.text(t0, offset, f"Ch{ch}", va="bottom", fontsize=9)
        offset += 2.5 * np.std(sig) + 1e-6

    # Regime boundaries
    for st in result.switch_times:
        if t0 <= st <= t1:
            ax1.axvline(st, linestyle="--", linewidth=1)

    ax1.set_title("Raw time series (selected channels) with regime switches")
    ax1.set_xlabel("Time (s)")
    ax1.set_yticks([])

    # Panel B: PSD (periodogram) averaged across channels
    ax2 = fig.add_subplot(gs[1, 0])
    seg = y[:, i0:i1]
    freqs = np.fft.rfftfreq(nfft, d=1/fs)
    Y = np.fft.rfft(seg - seg.mean(axis=1, keepdims=True), n=nfft, axis=1)
    psd = (np.abs(Y) ** 2).mean(axis=0)
    ax2.plot(freqs, psd)
    ax2.set_xlim(0, 60)
    ax2.set_title("Power spectral density (simple periodogram, mean across channels)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Power (a.u.)")

    # Panel C: Hilbert amplitude for one channel
    ax3 = fig.add_subplot(gs[2, 0])
    ch0 = channels[0]
    analytic = hilbert(y[ch0, i0:i1])
    amp = np.abs(analytic)
    ax3.plot(t[i0:i1], amp, lw=1.0)
    for st in result.switch_times:
        if t0 <= st <= t1:
            ax3.axvline(st, linestyle="--", linewidth=1)
    ax3.set_title(f"Hilbert amplitude envelope (channel {ch0})")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Amplitude (a.u.)")

    return fig


# -------------------------
# Legacy format compatibility
# -------------------------

@dataclass
class LegacySimulationResult:
    """
    Container matching simulation_analysis.py SimulationResult format.

    This allows the coupled oscillator output to be used with existing
    phase representation and autoencoder training pipelines.
    """
    time: np.ndarray              # (n_samples,) time in seconds
    latent_states: np.ndarray     # (n_samples, latent_dim) - ground truth latent
    observations: np.ndarray      # (n_samples, n_channels) - observed signals
    regime_labels: np.ndarray     # (n_samples,) - regime label at each time point
    transition_times: List[int]   # List of transition time INDICES (not seconds)
    regime_names: List[str]       # Names of regimes


def to_legacy_format(result: SimulationResult) -> LegacySimulationResult:
    """
    Convert CoupledStuartLandauNetwork output to simulation_analysis.py format.

    Key conversions:
    - y: (n_channels, n_samples) -> observations: (n_samples, n_channels)
    - z: (n_oscillators, n_samples) complex -> latent_states: (n_samples, latent_dim) real
    - switch_times: seconds -> transition_times: sample indices
    """
    # Transpose observations: (n_ch, T) -> (T, n_ch)
    observations = result.y.T

    # Convert complex oscillator states to real latent representation
    # Use [Re(z), Im(z)] flattened, or just Re(z) for simplicity
    # For flow field analysis, Re(z) captures the essential dynamics
    latent_states = np.real(result.z).T  # (T, n_oscillators)

    # Convert switch times from seconds to sample indices
    sfreq = result.params["sfreq"]
    transition_times = [int(st * sfreq) for st in result.switch_times if st > 0]

    return LegacySimulationResult(
        time=result.t,
        latent_states=latent_states,
        observations=observations,
        regime_labels=result.regime_id,
        transition_times=transition_times,
        regime_names=result.regime_names,
    )


# -------------------------
# Colored noise generator
# -------------------------

def generate_colored_noise(
    n_samples: int,
    n_channels: int,
    sfreq: float,
    alpha: float = 1.0,
    scale: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate 1/f^alpha colored noise for more realistic EEG-like background.

    Args:
        n_samples: Number of time samples
        n_channels: Number of channels
        sfreq: Sampling frequency (Hz)
        alpha: Spectral exponent (1.0 = pink noise, 2.0 = brown noise)
        scale: Output amplitude scaling
        seed: Random seed

    Returns:
        noise: (n_channels, n_samples) colored noise
    """
    rng = np.random.default_rng(seed)

    # Generate white noise
    white = rng.normal(size=(n_channels, n_samples))

    # FFT and shape spectrum
    freqs = np.fft.rfftfreq(n_samples, d=1/sfreq)
    freqs[0] = 1e-10  # Avoid division by zero

    # 1/f^alpha spectrum
    spectrum_filter = 1.0 / (freqs ** (alpha / 2))
    spectrum_filter[0] = 0  # Remove DC

    # Apply filter in frequency domain
    white_fft = np.fft.rfft(white, axis=1)
    colored_fft = white_fft * spectrum_filter
    colored = np.fft.irfft(colored_fft, n=n_samples, axis=1)

    # Normalize and scale
    colored = colored / np.std(colored) * scale

    return colored


# -------------------------
# Kuramoto order parameter
# -------------------------

def kuramoto_order_parameter(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Kuramoto order parameter R(t) for synchronization analysis.

    R = |1/N * sum_j exp(i * theta_j)|

    R ≈ 1: perfect synchrony
    R ≈ 0: complete desynchrony

    Args:
        z: (n_oscillators, n_samples) complex oscillator states

    Returns:
        R: (n_samples,) order parameter magnitude
        psi: (n_samples,) mean phase
    """
    # Extract phases
    phases = np.angle(z)  # (n_osc, T)

    # Mean field
    mean_field = np.mean(np.exp(1j * phases), axis=0)  # (T,)

    R = np.abs(mean_field)
    psi = np.angle(mean_field)

    return R, psi


def compute_regime_synchrony(result: SimulationResult) -> Dict[str, Dict[str, float]]:
    """
    Compute synchrony statistics per regime.

    Returns:
        Dict mapping regime_name -> {mean_R, std_R, min_R, max_R}
    """
    R, _ = kuramoto_order_parameter(result.z)

    stats = {}
    for i, name in enumerate(result.regime_names):
        mask = result.regime_id == i
        R_regime = R[mask]
        if len(R_regime) > 0:
            stats[name] = {
                "mean_R": float(np.mean(R_regime)),
                "std_R": float(np.std(R_regime)),
                "min_R": float(np.min(R_regime)),
                "max_R": float(np.max(R_regime)),
            }

    return stats


# -------------------------
# Demo / quick test
# -------------------------

def demo_simulation(
    total_duration_s: float = 60.0,
    show_plot: bool = True,
    save_path: Optional[str] = None,
) -> SimulationResult:
    """
    Run a quick demo simulation with default parameters.

    This demonstrates:
    - 4 coupling topologies cycling through
    - Kuramoto order parameter differences between regimes
    - Electrode time series visualization
    """
    print("Creating coupled Stuart-Landau network...")
    net = CoupledStuartLandauNetwork(
        n_oscillators=30,
        n_channels=30,
        sfreq=250.0,
        seed=42,
    )
    net.default_topologies()

    # Regime schedule: 15s per regime
    schedule = [
        ("global", 15.0),   # High synchrony
        ("cluster", 15.0),  # Moderate synchrony (within-cluster)
        ("sparse", 15.0),   # Low synchrony
        ("ring", 15.0),     # Wave-like patterns
    ]

    print("Generating simulation...")
    result = net.generate(
        total_duration_s=total_duration_s,
        regime_schedule=schedule,
        mu_mean=1.0,
        mu_std=0.2,
        omega_mean_hz=10.0,
        omega_std_hz=2.0,
        coupling_strength=0.5,
        noise_std=0.1,
        obs_noise_std=0.05,
        obs_noise_color=1.0,  # Pink noise for EEG-like 1/f background
        transition_s=0.3,     # Smooth 300ms transitions between regimes
    )

    # Compute synchrony per regime
    print("\nSynchrony (Kuramoto R) per regime:")
    sync_stats = compute_regime_synchrony(result)
    for name, stats in sync_stats.items():
        print(f"  {name}: R = {stats['mean_R']:.3f} ± {stats['std_R']:.3f}")

    if show_plot:
        fig = plot_electrode_timeseries(
            result,
            channels=[0, 5, 10, 15, 25],
            time_window=(0, total_duration_s),
        )
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"\nFigure saved to: {save_path}")
        import matplotlib.pyplot as plt
        plt.show()

    return result


def run_full_analysis(
    output_dir: Optional[str] = None,
    total_duration_s: float = 180.0,
    coupling_strength: float = 5.0,
    noise_std: float = 0.1,
    obs_noise_std: float = 0.05,
    obs_noise_color: float = 1.0,
    transition_s: float = 0.3,
    n_epochs: int = 50,
    hidden_size: int = 32,
    embedding_method: str = "umap",
    seed: int = 42,
    show_plots: bool = False,
    quick: bool = False,
    n_cycles: int = 1,
    regime_duration_s: Optional[float] = None,
) -> Dict:
    """
    Run full analysis pipeline with coupled Stuart-Landau simulation.

    This integrates with simulation_analysis.py to:
    1. Generate coupled oscillator data
    2. Convert to phase representation
    3. Train autoencoder
    4. Embed latent trajectories
    5. Compute flow metrics per regime
    6. Save all results and figures

    Args:
        output_dir: Output directory (auto-generated if None)
        total_duration_s: Total simulation duration
        coupling_strength: Coupling strength parameter
        noise_std: Oscillator noise std
        obs_noise_std: Observation noise std
        obs_noise_color: Colored noise exponent (0=white, 1=pink, 2=brown)
        transition_s: Smooth transition duration between regimes
        n_epochs: Training epochs
        hidden_size: Autoencoder latent dimension
        embedding_method: "pca" or "umap"
        seed: Random seed
        show_plots: Whether to display plots
        quick: Quick mode (fewer epochs)
        n_cycles: Number of full regime cycles (default 1 = each regime once)
            Use n_cycles > 1 to test transition tracking (e.g., n_cycles=4 with
            regime_duration_s=10 gives 10s×4regimes×4cycles = 160s)
        regime_duration_s: Duration per regime in seconds (default: total_duration_s / 4 / n_cycles)

    Returns:
        Dict with all results and statistics
    """
    import json
    import sys
    from pathlib import Path
    from datetime import datetime

    # Import from simulation_analysis
    sys.path.insert(0, str(Path(__file__).parent))
    from simulation_analysis import (
        observations_to_phase_representation,
        chunk_phase_data,
        train_simulation_model,
        compute_latent_trajectory,
        PooledEmbedder,
        compute_flow_metrics,
        compute_flow_field,
        compute_density_on_grid,
        FlowMetrics,
        SFREQ,
        CHUNK_DURATION,
        DEVICE,
    )

    import torch
    import matplotlib
    if not show_plots:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent.parent.parent / "results" / "simulations" / f"coupled_sl_{timestamp}"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Coupled Stuart-Landau Network - Full Analysis Pipeline")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Device: {DEVICE}")

    if quick:
        n_epochs = min(n_epochs, 20)

    # Save parameters
    params = {
        "seed": seed,
        "total_duration_s": total_duration_s,
        "coupling_strength": coupling_strength,
        "noise_std": noise_std,
        "obs_noise_std": obs_noise_std,
        "obs_noise_color": obs_noise_color,
        "transition_s": transition_s,
        "n_epochs": n_epochs,
        "hidden_size": hidden_size,
        "embedding_method": embedding_method,
        "sfreq": SFREQ,
        "chunk_duration": CHUNK_DURATION,
        "device": DEVICE,
    }
    with open(output_dir / "parameters.json", "w") as f:
        json.dump(params, f, indent=2)

    # =========================================================================
    # STEP 1: Generate Simulation
    # =========================================================================
    print("\n" + "-" * 50)
    print("Step 1: Generating Coupled Stuart-Landau Simulation")
    print("-" * 50)

    net = CoupledStuartLandauNetwork(
        n_oscillators=30,
        n_channels=30,
        sfreq=SFREQ,
        seed=seed,
    )
    net.default_topologies(seed=seed)

    # Regime schedule: support multiple cycles for transition tracking
    # Default: 10s per regime (4 cycles for 160s total with 180s recording)
    regime_names_order = ["global", "cluster", "sparse", "ring"]
    if regime_duration_s is not None:
        per_regime = regime_duration_s
    else:
        # Default to 10s per regime
        per_regime = 10.0
        # Calculate how many cycles fit in the total duration
        n_cycles = int(total_duration_s / (4 * per_regime))

    schedule = []
    for _ in range(n_cycles):
        for name in regime_names_order:
            schedule.append((name, per_regime))

    # Adjust total duration to match schedule
    actual_duration = per_regime * 4 * n_cycles
    if abs(actual_duration - total_duration_s) > 0.1:
        print(f"  Note: Adjusted duration from {total_duration_s}s to {actual_duration}s for {n_cycles} cycle(s)")
        total_duration_s = actual_duration

    result = net.generate(
        total_duration_s=total_duration_s,
        regime_schedule=schedule,
        mu_mean=1.0,
        mu_std=0.2,
        omega_mean_hz=10.0,
        omega_std_hz=2.0,
        coupling_strength=coupling_strength,
        noise_std=noise_std,
        obs_noise_std=obs_noise_std,
        obs_noise_color=obs_noise_color,
        transition_s=transition_s,
    )

    # Compute synchrony
    sync_stats = compute_regime_synchrony(result)
    print(f"  Total samples: {result.y.shape[1]}")
    print(f"  Regimes: {result.regime_names}")
    print(f"  Switch times: {result.switch_times}")
    print("\n  Synchrony (Kuramoto R) per regime:")
    for name, stats in sync_stats.items():
        print(f"    {name}: R = {stats['mean_R']:.3f} ± {stats['std_R']:.3f}")

    # Compute Laplacian spectral analysis for topology verification
    print("\n  Laplacian Spectral Analysis (topology verification):")
    topology_spectra = analyze_topology_spectra(net._topologies, net._laplacians)
    for name, spec in topology_spectra.items():
        print(f"    {name}: λ₂={spec['lambda_2']:.4f}, λ_max={spec['lambda_max']:.3f}, "
              f"gap={spec['spectral_gap']:.4f}, density={spec.get('density', 0):.3f}")

    # Convert to legacy format
    legacy = to_legacy_format(result)

    # =========================================================================
    # STEP 2: Phase Representation
    # =========================================================================
    print("\n" + "-" * 50)
    print("Step 2: Converting to Phase Representation")
    print("-" * 50)

    phase_data = observations_to_phase_representation(legacy.observations, SFREQ)
    print(f"  Phase data shape: {phase_data.shape}")

    # Chunk data
    chunk_samples = int(CHUNK_DURATION * SFREQ)
    chunks = chunk_phase_data(phase_data, chunk_samples)
    print(f"  Number of chunks: {len(chunks)}")

    # =========================================================================
    # STEP 3: Train Autoencoder
    # =========================================================================
    print("\n" + "-" * 50)
    print(f"Step 3: Training Autoencoder ({n_epochs} epochs)")
    print("-" * 50)

    model = train_simulation_model(
        chunks=chunks,
        n_channels=30,
        hidden_size=hidden_size,
        n_epochs=n_epochs,
        device=DEVICE,
        verbose=True,
    )

    # Save model
    model_path = output_dir / "autoencoder.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_channels": 30,
        "hidden_size": hidden_size,
    }, model_path)
    print(f"  Saved model: {model_path}")

    # =========================================================================
    # STEP 4: Compute Latent Trajectory
    # =========================================================================
    print("\n" + "-" * 50)
    print("Step 4: Computing Latent Trajectory")
    print("-" * 50)

    latent = compute_latent_trajectory(model, phase_data, DEVICE)
    print(f"  Latent shape: {latent.shape}")

    # =========================================================================
    # STEP 5: Standardize Latent, Embed, and Compute Flow Metrics
    # =========================================================================
    print("\n" + "-" * 50)
    print(f"Step 5: Standardize Latent, Embed ({embedding_method.upper()}), Flow Metrics")
    print("-" * 50)

    # Standardize latent before embedding (prevents PCA collapse)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    latent_scaled = scaler.fit_transform(latent)
    print(f"  Latent variance before scaling: {latent.var(axis=0).mean():.6f}")
    print(f"  Latent variance after scaling: {latent_scaled.var(axis=0).mean():.4f}")

    # Check for outliers and clip (use percentile-based clipping)
    p99 = np.percentile(np.abs(latent_scaled), 99)
    clip_threshold = max(3.0, p99)  # At least 3 std
    latent_clipped = np.clip(latent_scaled, -clip_threshold, clip_threshold)
    n_clipped = (np.abs(latent_scaled) > clip_threshold).sum()
    if n_clipped > 0:
        print(f"  Clipped {n_clipped} extreme values (>{clip_threshold:.1f} std)")

    # Embed standardized latent
    embedder = PooledEmbedder(n_components=2, method=embedding_method)
    embedder.fit([latent_clipped])
    embedded = embedder.transform(latent_clipped)
    print(f"  Embedded shape: {embedded.shape}")

    # Align regime labels with embedded (downsample to match compression from conv strides)
    # The latent trajectory is compressed ~4x due to strided convolutions
    n_original = len(legacy.regime_labels)
    n_embedded = len(embedded)
    compression_ratio = n_original / n_embedded
    print(f"  Compression ratio: {compression_ratio:.2f}x")

    # Downsample labels by taking the label at the corresponding original index
    labels_aligned = np.array([
        legacy.regime_labels[min(int(i * compression_ratio), n_original - 1)]
        for i in range(n_embedded)
    ])
    print(f"  Label distribution: {dict(zip(*np.unique(labels_aligned, return_counts=True)))}")

    # Compute flow metrics on FULL LATENT (not 2D projection) for reliable statistics
    # Use 2D embedding only for visualization
    # Helper: compute hardened flow metrics on full latent
    def compute_robust_flow_metrics(trajectory: np.ndarray, smooth_window: int = 5) -> dict:
        """
        Compute flow metrics with hardened tortuosity (smoothing + epsilon floor).

        Args:
            trajectory: (n_samples, n_dims) array
            smooth_window: window for Savitzky-Golay smoothing (must be odd)
        """
        from scipy.signal import savgol_filter
        from scipy.stats import entropy as scipy_entropy

        # Smooth trajectory for tortuosity (removes high-freq jitter)
        if smooth_window > 1 and len(trajectory) > smooth_window:
            smooth_window = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
            trajectory_smooth = savgol_filter(trajectory, smooth_window, polyorder=2, axis=0)
        else:
            trajectory_smooth = trajectory

        # Speed on smoothed trajectory
        velocity = np.diff(trajectory_smooth, axis=0)
        speed = np.linalg.norm(velocity, axis=1)

        # Tortuosity with epsilon floor
        path_length = speed.sum()
        displacement = np.linalg.norm(trajectory_smooth[-1] - trajectory_smooth[0])
        epsilon = 1e-6 * path_length  # Scale epsilon to path length
        tortuosity = path_length / (displacement + epsilon)
        # Also compute median tortuosity over segments for robustness
        segment_len = max(10, len(trajectory) // 20)
        segment_tortuosities = []
        for start in range(0, len(trajectory_smooth) - segment_len, segment_len):
            seg = trajectory_smooth[start:start + segment_len]
            seg_path = np.linalg.norm(np.diff(seg, axis=0), axis=1).sum()
            seg_disp = np.linalg.norm(seg[-1] - seg[0])
            segment_tortuosities.append(seg_path / (seg_disp + epsilon))
        median_tortuosity = np.median(segment_tortuosities) if segment_tortuosities else tortuosity

        # Speed statistics
        mean_speed = float(np.mean(speed))
        std_speed = float(np.std(speed))
        cv_speed = std_speed / mean_speed if mean_speed > 0 else 0

        # Explored variance (on original, not smoothed)
        explored_variance = float(np.var(trajectory, axis=0).sum())

        # Occupancy entropy (simplified: on first 2 PCs if high-dim)
        if trajectory.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            traj_2d = pca.fit_transform(trajectory)
        else:
            traj_2d = trajectory
        hist, _, _ = np.histogram2d(traj_2d[:, 0], traj_2d[:, 1], bins=20)
        hist_flat = hist.flatten()
        hist_flat = hist_flat[hist_flat > 0]
        occ_entropy = float(scipy_entropy(hist_flat / hist_flat.sum()))

        return {
            "mean_speed": mean_speed,
            "speed_std": std_speed,
            "speed_cv": cv_speed,
            "path_tortuosity": float(tortuosity),
            "median_tortuosity": float(median_tortuosity),
            "explored_variance": explored_variance,
            "occupancy_entropy": occ_entropy,
        }

    # Helper: compute field-level metrics (divergence, curl/circulation) on 2D embedded space
    def compute_field_metrics(embedded_2d: np.ndarray, bounds: tuple, grid_size: int = 15) -> dict:
        """
        Compute field-level metrics from the flow field on 2D embedded space.

        Divergence: ∂vx/∂x + ∂vy/∂y
          - Positive: sources/expansion (exploratory dynamics)
          - Negative: sinks/contraction (attractor dynamics)

        Curl (2D): ∂vy/∂x - ∂vx/∂y
          - Non-zero: rotational/circular flow patterns
          - Useful for detecting oscillatory/cyclic dynamics in latent space

        Args:
            embedded_2d: (n_samples, 2) embedded trajectory
            bounds: (x_min, x_max, y_min, y_max) from embedder
            grid_size: number of grid cells per dimension

        Returns:
            Dict with divergence and curl statistics
        """
        x_min, x_max, y_min, y_max = bounds

        # Create grid
        x_edges = np.linspace(x_min, x_max, grid_size + 1)
        y_edges = np.linspace(y_min, y_max, grid_size + 1)
        dx = (x_max - x_min) / grid_size
        dy = (y_max - y_min) / grid_size

        # Compute velocity at each point
        velocity = np.diff(embedded_2d, axis=0)  # (n-1, 2)
        positions = embedded_2d[:-1]  # (n-1, 2)

        # Bin velocities into grid cells
        flow_x = np.zeros((grid_size, grid_size))
        flow_y = np.zeros((grid_size, grid_size))
        counts = np.zeros((grid_size, grid_size))

        x_idx = np.clip(np.digitize(positions[:, 0], x_edges) - 1, 0, grid_size - 1)
        y_idx = np.clip(np.digitize(positions[:, 1], y_edges) - 1, 0, grid_size - 1)

        for i in range(len(positions)):
            xi, yi = x_idx[i], y_idx[i]
            flow_x[yi, xi] += velocity[i, 0]
            flow_y[yi, xi] += velocity[i, 1]
            counts[yi, xi] += 1

        # Average velocities
        mask = counts > 0
        flow_x[mask] /= counts[mask]
        flow_y[mask] /= counts[mask]

        # Compute divergence using finite differences (central)
        # div = ∂vx/∂x + ∂vy/∂y
        dvx_dx = np.zeros_like(flow_x)
        dvy_dy = np.zeros_like(flow_y)

        # Central differences for interior points
        dvx_dx[:, 1:-1] = (flow_x[:, 2:] - flow_x[:, :-2]) / (2 * dx)
        dvy_dy[1:-1, :] = (flow_y[2:, :] - flow_y[:-2, :]) / (2 * dy)

        divergence = dvx_dx + dvy_dy

        # Compute curl (2D): ∂vy/∂x - ∂vx/∂y
        dvy_dx = np.zeros_like(flow_y)
        dvx_dy = np.zeros_like(flow_x)

        dvy_dx[:, 1:-1] = (flow_y[:, 2:] - flow_y[:, :-2]) / (2 * dx)
        dvx_dy[1:-1, :] = (flow_x[2:, :] - flow_x[:-2, :]) / (2 * dy)

        curl = dvy_dx - dvx_dy

        # Only consider cells with sufficient samples
        min_samples = 3
        valid_mask = counts >= min_samples

        if valid_mask.sum() > 0:
            div_valid = divergence[valid_mask]
            curl_valid = curl[valid_mask]

            return {
                "mean_divergence": float(np.mean(div_valid)),
                "std_divergence": float(np.std(div_valid)),
                "mean_abs_divergence": float(np.mean(np.abs(div_valid))),
                "mean_curl": float(np.mean(curl_valid)),
                "std_curl": float(np.std(curl_valid)),
                "mean_abs_curl": float(np.mean(np.abs(curl_valid))),
                "curl_circulation": float(np.sum(curl_valid) * dx * dy),  # Total circulation
                "n_valid_cells": int(valid_mask.sum()),
                "divergence_grid": divergence,
                "curl_grid": curl,
                "counts_grid": counts,
            }
        else:
            return {
                "mean_divergence": 0.0,
                "std_divergence": 0.0,
                "mean_abs_divergence": 0.0,
                "mean_curl": 0.0,
                "std_curl": 0.0,
                "mean_abs_curl": 0.0,
                "curl_circulation": 0.0,
                "n_valid_cells": 0,
                "divergence_grid": divergence,
                "curl_grid": curl,
                "counts_grid": counts,
            }

    # Get unique regime names (handles multiple cycles)
    unique_regime_names = list(dict.fromkeys(result.regime_names))  # Preserves order, removes duplicates

    # Create mapping from regime_id to regime_name
    # regime_id values map to indices in result.regime_names
    regime_id_to_name = {i: result.regime_names[i] for i in range(len(result.regime_names))}

    regime_metrics = {}
    print("\n  Flow metrics per regime (on full latent, hardened tortuosity):")
    for name in unique_regime_names:
        # Find all regime_ids that correspond to this regime name
        matching_ids = [i for i, n in enumerate(result.regime_names) if n == name]
        # Combine masks for all occurrences of this regime
        mask = np.isin(labels_aligned, matching_ids)
        if mask.sum() > 100:
            # Compute on FULL LATENT (not 2D projection) for reliable statistics
            regime_latent = latent_clipped[mask]
            metrics = compute_robust_flow_metrics(regime_latent, smooth_window=5)
            metrics["n_samples"] = int(mask.sum())
            regime_metrics[name] = metrics
            print(f"    {name}: speed={metrics['mean_speed']:.4f}, CV={metrics['speed_cv']:.3f}, "
                  f"tortuosity={metrics['median_tortuosity']:.2f}, variance={metrics['explored_variance']:.3f}")

    # =========================================================================
    # STEP 5b: Compute Per-Window Metrics for Discriminability Analysis
    # =========================================================================
    print("\n  Computing per-window metrics for discriminability analysis...")

    # Compute metrics on sliding windows to get distributions per regime
    window_size = 50  # ~0.2s at 250Hz after 4x compression
    window_metrics = {name: {"speed": [], "variance": [], "tortuosity": []} for name in unique_regime_names}

    for name in unique_regime_names:
        matching_ids = [i for i, n in enumerate(result.regime_names) if n == name]
        mask = np.isin(labels_aligned, matching_ids)
        regime_latent = latent_clipped[mask]

        # Compute metrics on non-overlapping windows
        n_windows = len(regime_latent) // window_size
        for w in range(n_windows):
            window = regime_latent[w * window_size : (w + 1) * window_size]
            if len(window) < window_size:
                continue

            # Speed: mean step size
            velocity = np.diff(window, axis=0)
            speeds = np.linalg.norm(velocity, axis=1)
            window_metrics[name]["speed"].append(float(np.mean(speeds)))

            # Variance: total variance in window
            window_metrics[name]["variance"].append(float(np.var(window, axis=0).sum()))

            # Tortuosity: path_length / displacement
            path_len = speeds.sum()
            disp = np.linalg.norm(window[-1] - window[0])
            tort = path_len / (disp + 1e-8)
            window_metrics[name]["tortuosity"].append(float(tort))

    # Convert to arrays
    for name in unique_regime_names:
        for metric in window_metrics[name]:
            window_metrics[name][metric] = np.array(window_metrics[name][metric])

    # Compute discriminability statistics
    from scipy.stats import f_oneway, kruskal

    discriminability = {}
    print("\n  Discriminability analysis (ANOVA + effect size):")
    for metric in ["speed", "variance", "tortuosity"]:
        groups = [window_metrics[name][metric] for name in unique_regime_names]
        # Filter out empty groups
        groups = [g for g in groups if len(g) > 0]

        if len(groups) >= 2 and all(len(g) > 1 for g in groups):
            # ANOVA F-statistic
            f_stat, p_val = f_oneway(*groups)

            # Eta-squared (effect size): SS_between / SS_total
            all_data = np.concatenate(groups)
            grand_mean = np.mean(all_data)
            ss_total = np.sum((all_data - grand_mean) ** 2)
            ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
            eta_sq = ss_between / ss_total if ss_total > 0 else 0

            discriminability[metric] = {
                "f_statistic": float(f_stat),
                "p_value": float(p_val),
                "eta_squared": float(eta_sq),
                "n_windows": [len(g) for g in groups],
            }

            # Interpretation of eta-squared: 0.01=small, 0.06=medium, 0.14=large
            effect_label = "large" if eta_sq > 0.14 else "medium" if eta_sq > 0.06 else "small"
            print(f"    {metric}: F={f_stat:.1f}, p={p_val:.2e}, η²={eta_sq:.3f} ({effect_label})")

    # =========================================================================
    # STEP 5c: Compute Field-Level Metrics (Divergence, Curl) per Regime
    # =========================================================================
    print("\n  Computing field-level metrics (divergence, curl) per regime...")

    field_metrics = {}
    regime_flow_data = {}  # Store flow field data for each regime for plotting

    for name in unique_regime_names:
        matching_ids = [i for i, n in enumerate(result.regime_names) if n == name]
        mask = np.isin(labels_aligned, matching_ids)
        regime_embedded = embedded[mask]

        if len(regime_embedded) > 100:
            fmetrics = compute_field_metrics(regime_embedded, embedder.bounds, grid_size=15)

            # Store scalar metrics (exclude grids for JSON serialization)
            field_metrics[name] = {k: v for k, v in fmetrics.items()
                                   if not k.endswith('_grid')}

            # Store flow data for plotting
            X, Y, flow_x, flow_y, counts = compute_flow_field(regime_embedded, embedder.bounds, grid_size=15)
            regime_flow_data[name] = {
                "X": X, "Y": Y, "flow_x": flow_x, "flow_y": flow_y,
                "counts": counts, "embedded": regime_embedded,
                "divergence": fmetrics["divergence_grid"],
                "curl": fmetrics["curl_grid"],
            }

            print(f"    {name}: div={fmetrics['mean_divergence']:.4f}±{fmetrics['std_divergence']:.4f}, "
                  f"curl={fmetrics['mean_abs_curl']:.4f}, circulation={fmetrics['curl_circulation']:.4f}")

    # =========================================================================
    # STEP 6: Generate Figures
    # =========================================================================
    print("\n" + "-" * 50)
    print("Step 6: Generating Figures")
    print("-" * 50)

    # Figure 1: Electrode time series
    fig1 = plot_electrode_timeseries(
        result,
        channels=[0, 5, 10, 20, 28],
        time_window=(0, min(60, total_duration_s)),
    )
    fig1.savefig(output_dir / "fig_electrode_timeseries.png", dpi=150, bbox_inches="tight")
    fig1.savefig(output_dir / "fig_electrode_timeseries.pdf", dpi=300, bbox_inches="tight")
    print(f"  Saved: fig_electrode_timeseries.png/pdf")

    # Figure 2: Main analysis figure (4 panels)
    regime_colors = {
        "global": "#1f77b4",
        "cluster": "#ff7f0e",
        "sparse": "#2ca02c",
        "ring": "#d62728",
    }

    fig2 = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig2, hspace=0.3, wspace=0.25)

    # Panel A: Ground-truth regime timeline
    # Use ONLY switch_times and regime_names - NOT per-sample regime_id
    # This ensures alignment with results.json and avoids plotting artifacts
    ax_a = fig2.add_subplot(gs[0, 0])

    # Build intervals from switch_times: each regime runs from switch_times[i] to switch_times[i+1]
    # Note: result.regime_names may have duplicates if n_cycles > 1, so we iterate over schedule
    switch_times = result.switch_times  # [0.0, 45.0, 90.0, 135.0, ...]
    regime_name_sequence = result.regime_names  # ['global', 'cluster', 'sparse', 'ring', ...]

    for i, regime_name in enumerate(regime_name_sequence):
        start_time = switch_times[i]
        # End time is next switch or total duration
        end_time = switch_times[i + 1] if i + 1 < len(switch_times) else total_duration_s
        color = regime_colors.get(regime_name, "#888888")
        ax_a.axvspan(start_time, end_time, color=color, alpha=0.7)

    ax_a.set_xlim(0, total_duration_s)
    ax_a.set_ylim(0, 1)
    ax_a.set_xlabel("Time (s)")
    ax_a.set_title("A) Ground-Truth Regime Sequence", fontweight='bold')
    ax_a.set_yticks([])  # No y-axis needed for timeline
    # No legend - regime colors are self-explanatory from adjacent panels

    # Panel B: Embedded trajectories
    ax_b = fig2.add_subplot(gs[0, 1])
    step = max(1, len(embedded) // 5000)
    embedded_ds = embedded[::step]
    labels_ds = labels_aligned[::step][:len(embedded_ds)]
    # Map labels to unique regime names for coloring
    for name in unique_regime_names:
        matching_ids = [i for i, n in enumerate(result.regime_names) if n == name]
        mask = np.isin(labels_ds, matching_ids)
        color = regime_colors.get(name, "#888888")
        ax_b.scatter(embedded_ds[mask, 0], embedded_ds[mask, 1], c=color, s=2, alpha=0.4, label=name)
    ax_b.set_xlabel("Dim 1")
    ax_b.set_ylabel("Dim 2")
    ax_b.set_title("B) Embedded Trajectories (colored by regime)", fontweight='bold')
    ax_b.legend(markerscale=3)
    ax_b.set_aspect('equal')

    # Panel C: Density + Flow field
    ax_c = fig2.add_subplot(gs[1, 0])
    density = compute_density_on_grid(embedded, embedder.bounds, bins=50)
    X, Y, flow_x, flow_y, counts = compute_flow_field(embedded, embedder.bounds, grid_size=15)
    im = ax_c.imshow(density, origin='lower', extent=list(embedder.bounds), cmap='Blues', alpha=0.7, aspect='equal')
    mask = counts > 5
    if mask.any():
        mag = np.sqrt(flow_x[mask]**2 + flow_y[mask]**2)
        norm_fx = np.where(mag > 0, flow_x[mask] / mag, 0)
        norm_fy = np.where(mag > 0, flow_y[mask] / mag, 0)
        ax_c.quiver(X[mask], Y[mask], norm_fx, norm_fy, mag, cmap='inferno', alpha=0.85,
                   scale=25, width=0.004, headwidth=4, headlength=5)
    ax_c.set_xlabel("Dim 1")
    ax_c.set_ylabel("Dim 2")
    ax_c.set_title("C) Density + Flow Field", fontweight='bold')

    # Panel D: Metric comparison (use unique regime names)
    ax_d = fig2.add_subplot(gs[1, 1])
    metric_names = ["mean_speed", "speed_cv", "median_tortuosity", "explored_variance"]
    metric_labels = ["Speed", "Speed CV", "Tortuosity", "Variance"]
    x = np.arange(len(unique_regime_names))
    width = 0.18
    for j, (metric, label) in enumerate(zip(metric_names, metric_labels)):
        values = [regime_metrics.get(name, {}).get(metric, 0) for name in unique_regime_names]
        # Normalize for comparison
        max_val = max(values) if values else 1
        norm_values = [v / max_val if max_val > 0 else 0 for v in values]
        offset = (j - 1.5) * width
        bars = ax_d.bar(x + offset, norm_values, width, label=label, alpha=0.8)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(unique_regime_names)
    ax_d.set_ylabel("Normalized Value")
    ax_d.set_title("D) Flow Metrics by Regime", fontweight='bold')
    ax_d.legend(loc='upper right')

    fig2.suptitle("Coupled Stuart-Landau Network: Dynamical Microscope Analysis", fontsize=14, fontweight='bold')
    fig2.savefig(output_dir / "fig_analysis_main.png", dpi=150, bbox_inches="tight")
    fig2.savefig(output_dir / "fig_analysis_main.pdf", dpi=300, bbox_inches="tight")
    print(f"  Saved: fig_analysis_main.png/pdf")

    # Figure 3: Discriminability analysis (violin plots + effect sizes)
    fig3, axes = plt.subplots(1, 3, figsize=(14, 5))
    metric_titles = {
        "speed": "Speed (latent units/step)",
        "variance": "Explored Variance",
        "tortuosity": "Path Tortuosity",
    }

    for ax, metric in zip(axes, ["speed", "variance", "tortuosity"]):
        # Prepare data for violin plot
        data_for_plot = []
        positions = []
        colors_for_plot = []
        for i, name in enumerate(unique_regime_names):
            vals = window_metrics[name][metric]
            if len(vals) > 0:
                data_for_plot.append(vals)
                positions.append(i)
                colors_for_plot.append(regime_colors.get(name, "#888888"))

        if data_for_plot:
            # Violin plot
            parts = ax.violinplot(data_for_plot, positions=positions, showmeans=True, showmedians=True)

            # Color the violins
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors_for_plot[i])
                pc.set_alpha(0.7)

            # Style the lines
            for partname in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
                if partname in parts:
                    parts[partname].set_color('black')
                    parts[partname].set_linewidth(1)

        ax.set_xticks(range(len(unique_regime_names)))
        ax.set_xticklabels(unique_regime_names)
        ax.set_title(metric_titles[metric], fontweight='bold')
        ax.set_ylabel("Value")

        # Add effect size annotation
        if metric in discriminability:
            eta_sq = discriminability[metric]["eta_squared"]
            f_stat = discriminability[metric]["f_statistic"]
            p_val = discriminability[metric]["p_value"]
            effect_label = "large" if eta_sq > 0.14 else "medium" if eta_sq > 0.06 else "small"
            sig_str = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            ax.text(0.02, 0.98, f"η²={eta_sq:.3f} ({effect_label})\nF={f_stat:.1f} {sig_str}",
                   transform=ax.transAxes, va='top', ha='left', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig3.suptitle("Regime Discriminability: Per-Window Metric Distributions", fontsize=14, fontweight='bold')
    fig3.tight_layout()
    fig3.savefig(output_dir / "fig_discriminability.png", dpi=150, bbox_inches="tight")
    fig3.savefig(output_dir / "fig_discriminability.pdf", dpi=300, bbox_inches="tight")
    print(f"  Saved: fig_discriminability.png/pdf")

    # Figure 4: Regime-specific flow fields (2x2 small multiples)
    fig4, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for idx, name in enumerate(unique_regime_names[:4]):  # Limit to 4 regimes
        ax = axes[idx]

        if name in regime_flow_data:
            data = regime_flow_data[name]
            X, Y = data["X"], data["Y"]
            flow_x, flow_y = data["flow_x"], data["flow_y"]
            counts = data["counts"]
            regime_embedded = data["embedded"]

            # Plot density background
            regime_density = compute_density_on_grid(regime_embedded, embedder.bounds, bins=50)
            ax.imshow(regime_density, origin='lower', extent=list(embedder.bounds),
                     cmap='Blues', alpha=0.6, aspect='equal')

            # Plot flow field where we have sufficient samples
            mask = counts > 3
            if mask.any():
                mag = np.sqrt(flow_x[mask]**2 + flow_y[mask]**2)
                norm_fx = np.where(mag > 0, flow_x[mask] / mag, 0)
                norm_fy = np.where(mag > 0, flow_y[mask] / mag, 0)
                ax.quiver(X[mask], Y[mask], norm_fx, norm_fy, mag, cmap='inferno', alpha=0.85,
                         scale=25, width=0.005, headwidth=4, headlength=5)

            # Add field metrics annotation
            if name in field_metrics:
                fm = field_metrics[name]
                ax.text(0.02, 0.98,
                       f"div: {fm['mean_divergence']:.3f}\ncurl: {fm['mean_abs_curl']:.3f}",
                       transform=ax.transAxes, va='top', ha='left', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_title(f"{name.capitalize()}", fontweight='bold', color=regime_colors.get(name, 'black'))
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_aspect('equal')

    fig4.suptitle("Regime-Specific Flow Fields (Density + Velocity)", fontsize=14, fontweight='bold')
    fig4.tight_layout()
    fig4.savefig(output_dir / "fig_flow_fields.png", dpi=150, bbox_inches="tight")
    fig4.savefig(output_dir / "fig_flow_fields.pdf", dpi=300, bbox_inches="tight")
    print(f"  Saved: fig_flow_fields.png/pdf")

    # Figure 5: Laplacian eigenvalue spectra comparison
    fig5, axes5 = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Eigenvalue spectra
    ax5a = axes5[0]
    for name in unique_regime_names:
        if name in topology_spectra:
            eigs = topology_spectra[name]["eigenvalues"]
            ax5a.plot(range(len(eigs)), eigs, 'o-', label=name, color=regime_colors.get(name, '#888888'),
                     markersize=4, alpha=0.8)
    ax5a.set_xlabel("Eigenvalue Index")
    ax5a.set_ylabel("Eigenvalue (λ)")
    ax5a.set_title("A) Laplacian Eigenvalue Spectra", fontweight='bold')
    ax5a.legend()
    ax5a.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Panel B: Summary metrics comparison
    ax5b = axes5[1]
    metrics_to_plot = ["lambda_2", "spectral_gap", "density"]
    metric_labels = ["λ₂ (Algebraic\nConnectivity)", "Spectral Gap\n(λ₂/λ_max)", "Edge Density"]
    x = np.arange(len(unique_regime_names))
    width = 0.25

    for j, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        values = [topology_spectra.get(name, {}).get(metric, 0) for name in unique_regime_names]
        # Normalize for visualization
        max_val = max(values) if values and max(values) > 0 else 1
        norm_values = [v / max_val for v in values]
        offset = (j - 1) * width
        ax5b.bar(x + offset, norm_values, width, label=label, alpha=0.8)

    ax5b.set_xticks(x)
    ax5b.set_xticklabels(unique_regime_names)
    ax5b.set_ylabel("Normalized Value")
    ax5b.set_title("B) Topology Spectral Properties", fontweight='bold')
    ax5b.legend(loc='upper right')

    fig5.suptitle("Laplacian Spectral Analysis: Topology Verification", fontsize=14, fontweight='bold')
    fig5.tight_layout()
    fig5.savefig(output_dir / "fig_laplacian_spectra.png", dpi=150, bbox_inches="tight")
    fig5.savefig(output_dir / "fig_laplacian_spectra.pdf", dpi=300, bbox_inches="tight")
    print(f"  Saved: fig_laplacian_spectra.png/pdf")

    if not show_plots:
        plt.close('all')
    else:
        plt.show()

    # =========================================================================
    # STEP 7: Save Results
    # =========================================================================
    print("\n" + "-" * 50)
    print("Step 7: Saving Results")
    print("-" * 50)

    # Prepare topology spectra for JSON (remove eigenvalue arrays for cleaner output)
    topology_spectra_summary = {}
    for name, spec in topology_spectra.items():
        topology_spectra_summary[name] = {k: v for k, v in spec.items() if k != "eigenvalues"}

    results = {
        "parameters": params,
        "synchrony_stats": sync_stats,
        "regime_metrics": regime_metrics,
        "field_metrics": field_metrics,
        "topology_spectra": topology_spectra_summary,
        "discriminability": discriminability,
        "n_samples": int(result.y.shape[1]),
        "n_regimes": len(result.regime_names),
        "regime_names": result.regime_names,
        "switch_times": result.switch_times,
        "embedded_shape": list(embedded.shape),
        "latent_shape": list(latent.shape),
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: results.json")

    # Save numpy arrays
    np.savez_compressed(
        output_dir / "trajectories.npz",
        embedded=embedded,
        latent=latent,
        regime_labels=labels_aligned,
        time=legacy.time[:len(embedded)],
    )
    print(f"  Saved: trajectories.npz")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nKey files:")
    print(f"  - parameters.json: Simulation parameters")
    print(f"  - results.json: Flow metrics and statistics")
    print(f"  - trajectories.npz: Embedded/latent trajectories")
    print(f"  - autoencoder.pt: Trained model")
    print(f"  - fig_*.png/pdf: Analysis figures")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Coupled Stuart-Landau oscillator simulation")
    parser.add_argument("--duration", type=float, default=180.0, help="Total duration (s)")
    parser.add_argument("--coupling", type=float, default=5.0, help="Coupling strength")
    parser.add_argument("--noise", type=float, default=0.1, help="Oscillator noise std")
    parser.add_argument("--obs-noise", type=float, default=0.05, help="Observation noise std")
    parser.add_argument("--obs-noise-color", type=float, default=1.0, help="Colored noise exponent")
    parser.add_argument("--transition", type=float, default=0.3, help="Transition duration (s)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--hidden", type=int, default=32, help="Hidden size")
    parser.add_argument("--embedding", type=str, default="umap", choices=["pca", "umap"], help="Embedding method")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer epochs)")
    parser.add_argument("--show", action="store_true", help="Show plots")
    parser.add_argument("--cycles", type=int, default=1, help="Number of regime cycles (default 1)")
    parser.add_argument("--regime-duration", type=float, default=None, help="Duration per regime (s)")

    # Legacy demo mode
    parser.add_argument("--demo", action="store_true", help="Run quick demo only")
    parser.add_argument("--no-show", action="store_true", help="[Demo] Don't show plot")
    parser.add_argument("--save", type=str, default=None, help="[Demo] Save figure path")

    args = parser.parse_args()

    if args.demo:
        # Legacy demo mode
        demo_simulation(
            total_duration_s=args.duration,
            show_plot=not args.no_show,
            save_path=args.save,
        )
    else:
        # Full analysis mode
        run_full_analysis(
            output_dir=args.output,
            total_duration_s=args.duration,
            coupling_strength=args.coupling,
            noise_std=args.noise,
            obs_noise_std=args.obs_noise,
            obs_noise_color=args.obs_noise_color,
            transition_s=args.transition,
            n_epochs=args.epochs,
            hidden_size=args.hidden,
            embedding_method=args.embedding,
            seed=args.seed,
            show_plots=args.show,
            quick=args.quick,
            n_cycles=args.cycles,
            regime_duration_s=args.regime_duration,
        )
