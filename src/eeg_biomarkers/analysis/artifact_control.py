"""
Artifact control and HF-proxy analysis for latent space validation.

This module addresses the critic agent's concerns about high-frequency
artifact contamination in the latent space. It provides:

1. Proper HF power computation (not crude signal variance)
2. Topographic analysis (EMG-prone vs central electrodes)
3. Latent residualization (regress out HF-proxy influence)
4. State-HF relationship testing

Key insight from critic:
- r=-0.708 correlation with HF proxy is a yellow flag, not red
- Could be EMG artifact OR legitimate aperiodic/broadband neural changes
- Need to test and control, not panic
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.stats import pearsonr, spearmanr, kruskal
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from dataclasses import dataclass
from typing import Literal
import warnings


@dataclass
class HFProxyResult:
    """Results from HF-proxy computation."""

    hf_power: np.ndarray  # (n_segments,) or (n_segments, n_channels)
    band: tuple[float, float]  # Frequency band used
    channels_used: list[str] | None  # Channel names if available
    method: str  # Computation method

    # Topographic breakdown (if computed)
    hf_power_temporal: np.ndarray | None = None  # EMG-prone
    hf_power_central: np.ndarray | None = None   # More neural
    hf_power_occipital: np.ndarray | None = None # Most neural


@dataclass
class DualBandHFResult:
    """Results from dual-band HF analysis (critic agent recommendation).

    HF1 (30-48 Hz): Mixed neural beta/gamma + potential EMG
    HF2 (70-110 Hz): More EMG-prone, useful as "muscle-likeness" indicator

    Interpretation:
    - If HF2 correlation >> HF1 correlation AND topographically temporal → likely EMG
    - If HF1 correlation >= HF2 AND centrally/occipitally distributed → likely neural
    """

    hf1_result: HFProxyResult  # 30-48 Hz (or available upper band)
    hf2_result: HFProxyResult | None  # 70-110 Hz (if sfreq allows)

    # Correlation with latent for each band
    hf1_latent_correlation: float
    hf2_latent_correlation: float | None

    # Topographic ratios (temporal / central power)
    hf1_temporal_central_ratio: float | None
    hf2_temporal_central_ratio: float | None

    # Interpretation
    emg_signature_detected: bool
    interpretation: str


@dataclass
class ArtifactAnalysisResult:
    """Results from artifact/HF-proxy analysis."""

    # Correlation with latent
    correlation_with_latent_norm: float
    correlation_with_latent_mean: float
    correlation_pvalue: float

    # Per-dimension correlations
    per_dim_correlations: np.ndarray  # (latent_dim,)
    max_dim_correlation: float
    max_dim_index: int

    # Topographic analysis
    temporal_correlation: float | None = None
    central_correlation: float | None = None
    occipital_correlation: float | None = None

    # Interpretation
    likely_artifact: bool = False
    interpretation: str = ""


def compute_hf_power(
    data: np.ndarray,
    sfreq: float,
    band: tuple[float, float] = (40.0, 100.0),
    method: Literal["welch", "bandpower", "rms"] = "welch",
    channel_names: list[str] | None = None,
) -> HFProxyResult:
    """
    Compute high-frequency power proxy for artifact detection.

    Args:
        data: EEG data (n_segments, n_channels, n_times) or (n_channels, n_times)
        sfreq: Sampling frequency in Hz
        band: Frequency band for HF power (default 40-100 Hz for EMG detection)
        method: Power computation method
        channel_names: Optional channel names for topographic analysis

    Returns:
        HFProxyResult with HF power values and metadata

    Notes:
        - For EMG detection, 70-110 Hz is most specific but requires high sfreq
        - 40-100 Hz is a good compromise for typical EEG (250 Hz sampling)
        - Compare temporal (EMG-prone) vs occipital (more neural) for interpretation
    """
    # Handle single segment vs batch
    if data.ndim == 2:
        data = data[np.newaxis, ...]  # Add batch dim

    n_segments, n_channels, n_times = data.shape

    # Validate band against Nyquist
    nyquist = sfreq / 2
    if band[1] > nyquist:
        warnings.warn(f"HF band {band} exceeds Nyquist ({nyquist}). Adjusting to {(band[0], nyquist-1)}")
        band = (band[0], nyquist - 1)

    hf_power_all = np.zeros((n_segments, n_channels))

    for seg_idx in range(n_segments):
        seg_data = data[seg_idx]  # (n_channels, n_times)

        if method == "welch":
            # Welch PSD - most robust
            freqs, psd = signal.welch(seg_data, fs=sfreq, nperseg=min(256, n_times))
            band_mask = (freqs >= band[0]) & (freqs <= band[1])
            hf_power_all[seg_idx] = psd[:, band_mask].mean(axis=1)

        elif method == "bandpower":
            # Bandpass then compute variance (simpler)
            sos = signal.butter(4, band, btype='band', fs=sfreq, output='sos')
            filtered = signal.sosfiltfilt(sos, seg_data, axis=1)
            hf_power_all[seg_idx] = np.var(filtered, axis=1)

        elif method == "rms":
            # RMS in band
            sos = signal.butter(4, band, btype='band', fs=sfreq, output='sos')
            filtered = signal.sosfiltfilt(sos, seg_data, axis=1)
            hf_power_all[seg_idx] = np.sqrt(np.mean(filtered**2, axis=1))

    # Average across channels for scalar proxy
    hf_power_scalar = hf_power_all.mean(axis=1)

    # Topographic analysis if channel names provided
    hf_temporal = hf_central = hf_occipital = None

    if channel_names is not None:
        temporal_idx, central_idx, occipital_idx = _get_topographic_indices(channel_names)

        if temporal_idx:
            hf_temporal = hf_power_all[:, temporal_idx].mean(axis=1)
        if central_idx:
            hf_central = hf_power_all[:, central_idx].mean(axis=1)
        if occipital_idx:
            hf_occipital = hf_power_all[:, occipital_idx].mean(axis=1)

    return HFProxyResult(
        hf_power=hf_power_scalar,
        band=band,
        channels_used=channel_names,
        method=method,
        hf_power_temporal=hf_temporal,
        hf_power_central=hf_central,
        hf_power_occipital=hf_occipital,
    )


def _get_topographic_indices(channel_names: list[str]) -> tuple[list[int], list[int], list[int]]:
    """
    Get channel indices for topographic regions.

    Optimized for 256-channel EGI Geodesic Sensor Net (GSN).

    EMG contamination is strongest in:
    - Temporal electrodes: Near ears, jaw muscles (masseter, temporalis)
    - Frontal periphery: Eye blinks, frontalis muscle

    Neural HF activity is more reliable in:
    - Central electrodes: Vertex area, less muscle contamination
    - Occipital electrodes: Visual cortex, furthest from facial muscles

    For EGI 256-channel system, electrodes are numbered E1-E256.
    Key landmark mappings (approximate 10-20 equivalents):
    - Temporal: E40-E49 (left), E115-E124 (right) - near ears
    - Frontal periphery: E1-E18, E25-E32 (forehead/eye area)
    - Central: E6, E7, E31, E55, E80, E106, E129 (Cz region)
    - Occipital: E70-E75, E82-E83 (O1/O2/Oz region)

    Also works with standard 10-20 naming conventions.
    """
    # Patterns for standard 10-20 names
    temporal_patterns = [
        'T7', 'T8', 'T3', 'T4', 'T5', 'T6',  # Temporal
        'TP7', 'TP8', 'TP9', 'TP10',  # Temporo-parietal
        'FT7', 'FT8', 'FT9', 'FT10',  # Fronto-temporal
        'F7', 'F8',  # Frontal lateral
        'Fp1', 'Fp2', 'Fpz',  # Frontal pole (eye artifacts)
        'AF3', 'AF4', 'AF7', 'AF8',  # Anterior frontal
    ]
    central_patterns = [
        'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6',  # Central
        'CPz', 'CP1', 'CP2', 'CP3', 'CP4',  # Centro-parietal
        'FCz', 'FC1', 'FC2', 'FC3', 'FC4',  # Fronto-central
        'Fz', 'F1', 'F2',  # Frontal midline
    ]
    occipital_patterns = [
        'O1', 'O2', 'Oz',  # Occipital
        'PO3', 'PO4', 'PO7', 'PO8', 'POz',  # Parieto-occipital
        'I1', 'I2', 'Iz',  # Inion
        'P1', 'P2', 'Pz', 'P3', 'P4',  # Parietal
    ]

    # EGI 256 electrode numbers for regions (approximate)
    # These are based on EGI GSN 256 layout
    egi_temporal = list(range(40, 50)) + list(range(115, 125)) + \
                   list(range(1, 19)) + list(range(25, 33)) + \
                   [226, 227, 228, 229, 230, 231, 232, 233]  # Peripheral
    egi_central = [6, 7, 31, 55, 80, 106, 129, 30, 37, 54, 79, 87, 105]
    egi_occipital = list(range(70, 84)) + [66, 67, 68, 69, 84, 85, 86]

    temporal_idx = []
    central_idx = []
    occipital_idx = []

    for i, name in enumerate(channel_names):
        name_upper = name.upper().strip()

        # Check for EGI electrode numbers (E1, E2, etc.)
        if name_upper.startswith('E') and name_upper[1:].isdigit():
            electrode_num = int(name_upper[1:])
            if electrode_num in egi_temporal:
                temporal_idx.append(i)
            elif electrode_num in egi_central:
                central_idx.append(i)
            elif electrode_num in egi_occipital:
                occipital_idx.append(i)
        # Check for standard 10-20 names
        elif any(p.upper() == name_upper or name_upper.startswith(p.upper())
                 for p in temporal_patterns):
            temporal_idx.append(i)
        elif any(p.upper() == name_upper or name_upper.startswith(p.upper())
                 for p in central_patterns):
            central_idx.append(i)
        elif any(p.upper() == name_upper or name_upper.startswith(p.upper())
                 for p in occipital_patterns):
            occipital_idx.append(i)

    return temporal_idx, central_idx, occipital_idx


def compute_dual_band_hf(
    data: np.ndarray,
    sfreq: float,
    latents: np.ndarray,
    channel_names: list[str] | None = None,
    hf1_band: tuple[float, float] = (30.0, 48.0),
    hf2_band: tuple[float, float] = (70.0, 110.0),
) -> DualBandHFResult:
    """
    Compute dual-band HF analysis as recommended by critic agent.

    HF1 (30-48 Hz): Mixed neural beta/gamma + potential EMG
    HF2 (70-110 Hz): More EMG-prone, useful as "muscle-likeness" indicator

    Args:
        data: EEG data (n_segments, n_channels, n_times)
        sfreq: Sampling frequency in Hz
        latents: Latent representations (n_segments, latent_dim)
        channel_names: Optional channel names for topographic analysis
        hf1_band: Lower HF band (default 30-48 Hz)
        hf2_band: Upper HF band (default 70-110 Hz, requires sfreq > 220 Hz)

    Returns:
        DualBandHFResult with analysis for both bands
    """
    nyquist = sfreq / 2

    # Compute HF1 (always available)
    hf1_result = compute_hf_power(
        data, sfreq, band=hf1_band, method="welch", channel_names=channel_names
    )

    # Compute latent norms for correlation
    latent_norms = np.linalg.norm(latents, axis=1)
    hf1_corr, _ = pearsonr(hf1_result.hf_power, latent_norms)

    # Compute HF1 topographic ratio
    hf1_ratio = None
    if hf1_result.hf_power_temporal is not None and hf1_result.hf_power_central is not None:
        hf1_ratio = np.mean(hf1_result.hf_power_temporal) / (np.mean(hf1_result.hf_power_central) + 1e-10)

    # Compute HF2 if sampling rate allows
    hf2_result = None
    hf2_corr = None
    hf2_ratio = None

    if hf2_band[1] < nyquist:
        hf2_result = compute_hf_power(
            data, sfreq, band=hf2_band, method="welch", channel_names=channel_names
        )
        hf2_corr, _ = pearsonr(hf2_result.hf_power, latent_norms)

        if hf2_result.hf_power_temporal is not None and hf2_result.hf_power_central is not None:
            hf2_ratio = np.mean(hf2_result.hf_power_temporal) / (np.mean(hf2_result.hf_power_central) + 1e-10)

    # Interpretation
    emg_signature = False
    interpretation_parts = []

    interpretation_parts.append(f"HF1 ({hf1_band[0]}-{hf1_band[1]} Hz): r={hf1_corr:.3f}")

    if hf2_corr is not None:
        interpretation_parts.append(f"HF2 ({hf2_band[0]}-{hf2_band[1]} Hz): r={hf2_corr:.3f}")

        # Check for EMG signature: HF2 >> HF1 AND temporally dominant
        if abs(hf2_corr) > abs(hf1_corr) + 0.15:
            if hf2_ratio is not None and hf2_ratio > 1.5:
                emg_signature = True
                interpretation_parts.append("EMG SIGNATURE: HF2 correlation stronger AND temporally dominant")
            else:
                interpretation_parts.append("HF2 stronger but not temporally dominant - unclear source")
        elif abs(hf1_corr) >= abs(hf2_corr):
            interpretation_parts.append("HF1 >= HF2 - suggests neural rather than EMG source")
    else:
        interpretation_parts.append(f"HF2 not computed (sfreq={sfreq} Hz < required for {hf2_band} Hz band)")

    if hf1_ratio is not None:
        interpretation_parts.append(f"HF1 temporal/central ratio: {hf1_ratio:.2f}")
    if hf2_ratio is not None:
        interpretation_parts.append(f"HF2 temporal/central ratio: {hf2_ratio:.2f}")

    return DualBandHFResult(
        hf1_result=hf1_result,
        hf2_result=hf2_result,
        hf1_latent_correlation=hf1_corr,
        hf2_latent_correlation=hf2_corr,
        hf1_temporal_central_ratio=hf1_ratio,
        hf2_temporal_central_ratio=hf2_ratio,
        emg_signature_detected=emg_signature,
        interpretation=" | ".join(interpretation_parts),
    )


def analyze_latent_hf_correlation(
    latents: np.ndarray,
    hf_proxy: HFProxyResult | np.ndarray,
) -> ArtifactAnalysisResult:
    """
    Analyze correlation between latent space and HF-proxy.

    Args:
        latents: Latent representations (n_segments, latent_dim)
        hf_proxy: HF power proxy (HFProxyResult or array of shape (n_segments,))

    Returns:
        ArtifactAnalysisResult with correlation analysis and interpretation
    """
    if isinstance(hf_proxy, HFProxyResult):
        hf_power = hf_proxy.hf_power
        hf_temporal = hf_proxy.hf_power_temporal
        hf_central = hf_proxy.hf_power_central
        hf_occipital = hf_proxy.hf_power_occipital
    else:
        hf_power = hf_proxy
        hf_temporal = hf_central = hf_occipital = None

    # Global correlations
    latent_norms = np.linalg.norm(latents, axis=1)
    latent_means = latents.mean(axis=1)

    corr_norm, p_norm = pearsonr(hf_power, latent_norms)
    corr_mean, _ = pearsonr(hf_power, latent_means)

    # Per-dimension correlations
    per_dim_corr = np.array([
        pearsonr(hf_power, latents[:, d])[0]
        for d in range(latents.shape[1])
    ])
    max_dim_idx = np.argmax(np.abs(per_dim_corr))
    max_dim_corr = per_dim_corr[max_dim_idx]

    # Topographic correlations
    corr_temporal = corr_central = corr_occipital = None

    if hf_temporal is not None:
        corr_temporal, _ = pearsonr(hf_temporal, latent_norms)
    if hf_central is not None:
        corr_central, _ = pearsonr(hf_central, latent_norms)
    if hf_occipital is not None:
        corr_occipital, _ = pearsonr(hf_occipital, latent_norms)

    # Interpretation
    likely_artifact = False
    interpretation = []

    if abs(corr_norm) > 0.5:
        interpretation.append(f"Strong overall correlation (r={corr_norm:.3f})")

        # Check topographic pattern for EMG signature
        if corr_temporal is not None and corr_central is not None:
            if abs(corr_temporal) > abs(corr_central) + 0.2:
                likely_artifact = True
                interpretation.append(
                    f"EMG pattern: temporal ({corr_temporal:.3f}) > central ({corr_central:.3f})"
                )
            elif abs(corr_central) >= abs(corr_temporal):
                interpretation.append(
                    f"Neural pattern: central ({corr_central:.3f}) >= temporal ({corr_temporal:.3f})"
                )
    else:
        interpretation.append(f"Weak overall correlation (r={corr_norm:.3f}) - likely OK")

    return ArtifactAnalysisResult(
        correlation_with_latent_norm=corr_norm,
        correlation_with_latent_mean=corr_mean,
        correlation_pvalue=p_norm,
        per_dim_correlations=per_dim_corr,
        max_dim_correlation=max_dim_corr,
        max_dim_index=max_dim_idx,
        temporal_correlation=corr_temporal,
        central_correlation=corr_central,
        occipital_correlation=corr_occipital,
        likely_artifact=likely_artifact,
        interpretation=" | ".join(interpretation),
    )


def residualize_latent(
    latents_train: np.ndarray,
    latents_test: np.ndarray | None,
    hf_proxy_train: np.ndarray,
    hf_proxy_test: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, LinearRegression]:
    """
    Remove HF-proxy influence from latent space via linear regression.

    This is the key mitigation recommended by the critic agent:
    - Fit regression of latent features on HF proxy (TRAIN ONLY)
    - Subtract predicted component from both train and test
    - Cluster/classify on residual latent

    Args:
        latents_train: Training latents (n_train, latent_dim)
        latents_test: Test latents (n_test, latent_dim) or None
        hf_proxy_train: HF power for training (n_train,)
        hf_proxy_test: HF power for test (n_test,) or None

    Returns:
        Tuple of (residual_train, residual_test, fitted_regressor)
    """
    # Fit on train only
    reg = LinearRegression()
    reg.fit(hf_proxy_train.reshape(-1, 1), latents_train)

    # Predict and subtract
    predicted_train = reg.predict(hf_proxy_train.reshape(-1, 1))
    residual_train = latents_train - predicted_train

    residual_test = None
    if latents_test is not None and hf_proxy_test is not None:
        predicted_test = reg.predict(hf_proxy_test.reshape(-1, 1))
        residual_test = latents_test - predicted_test

    return residual_train, residual_test, reg


def check_states_vs_hf_proxy(
    state_labels: np.ndarray,
    hf_proxy: np.ndarray,
) -> dict:
    """
    Check if discovered states are just HF-proxy bins.

    If HF proxy alone predicts state extremely well, states are likely
    "artifact regimes" unless explicitly framed that way.

    Args:
        state_labels: Cluster/state assignments (n_segments,)
        hf_proxy: HF power values (n_segments,)

    Returns:
        Dict with test results:
        - kruskal_statistic: Kruskal-Wallis H statistic
        - kruskal_pvalue: p-value for HF differences across states
        - prediction_accuracy: How well HF proxy predicts state
        - interpretation: Human-readable interpretation
    """
    unique_states = np.unique(state_labels)
    n_states = len(unique_states)

    # Kruskal-Wallis test: are HF distributions different across states?
    groups = [hf_proxy[state_labels == s] for s in unique_states]

    if n_states < 2:
        return {
            "kruskal_statistic": np.nan,
            "kruskal_pvalue": np.nan,
            "prediction_accuracy": np.nan,
            "interpretation": "Only one state found - cannot test",
        }

    stat, pval = kruskal(*groups)

    # Can we predict state from HF proxy alone?
    clf = LogisticRegression(max_iter=1000, random_state=42)
    try:
        scores = cross_val_score(
            clf,
            hf_proxy.reshape(-1, 1),
            state_labels,
            cv=min(5, n_states)
        )
        pred_acc = scores.mean()
    except Exception:
        pred_acc = np.nan

    # Interpretation
    if pval < 0.001 and pred_acc > 0.7:
        interpretation = (
            f"WARNING: States strongly related to HF proxy "
            f"(H={stat:.1f}, p={pval:.2e}, pred_acc={pred_acc:.2f}). "
            f"Consider residualization before clustering."
        )
    elif pval < 0.05:
        interpretation = (
            f"Moderate HF-state relationship (H={stat:.1f}, p={pval:.3f}). "
            f"Monitor but not necessarily problematic."
        )
    else:
        interpretation = (
            f"States appear independent of HF proxy (p={pval:.3f}). Good."
        )

    return {
        "kruskal_statistic": stat,
        "kruskal_pvalue": pval,
        "prediction_accuracy": pred_acc,
        "n_states": n_states,
        "hf_per_state_mean": {s: hf_proxy[state_labels == s].mean() for s in unique_states},
        "interpretation": interpretation,
    }


def run_artifact_control_analysis(
    latents: np.ndarray,
    raw_data: np.ndarray,
    sfreq: float,
    channel_names: list[str] | None = None,
    state_labels: np.ndarray | None = None,
    hf_band: tuple[float, float] = (40.0, 100.0),
) -> dict:
    """
    Run complete artifact control analysis pipeline.

    This is the comprehensive check recommended by the critic agent.

    Args:
        latents: Latent representations (n_segments, latent_dim)
        raw_data: Original EEG data (n_segments, n_channels, n_times)
        sfreq: Sampling frequency
        channel_names: Optional channel names for topographic analysis
        state_labels: Optional state/cluster labels for state-HF testing
        hf_band: Frequency band for HF proxy

    Returns:
        Dict with complete analysis results
    """
    results = {}

    # Step 1: Compute HF proxy properly
    hf_result = compute_hf_power(
        raw_data, sfreq, band=hf_band,
        method="welch", channel_names=channel_names
    )
    results["hf_proxy"] = hf_result

    # Step 2: Analyze correlation with latent
    artifact_analysis = analyze_latent_hf_correlation(latents, hf_result)
    results["artifact_analysis"] = artifact_analysis

    # Step 3: Test states vs HF proxy (if states provided)
    if state_labels is not None:
        state_hf_test = check_states_vs_hf_proxy(state_labels, hf_result.hf_power)
        results["state_hf_test"] = state_hf_test

    # Step 4: Compute residualized latent for comparison
    residual_latent, _, reg = residualize_latent(
        latents, None, hf_result.hf_power, None
    )
    results["residual_latent"] = residual_latent
    results["residualization_model"] = reg

    # Verify residualization worked
    post_resid_corr, _ = pearsonr(
        hf_result.hf_power,
        np.linalg.norm(residual_latent, axis=1)
    )
    results["post_residualization_correlation"] = post_resid_corr

    # Summary
    results["summary"] = {
        "original_correlation": artifact_analysis.correlation_with_latent_norm,
        "post_residualization_correlation": post_resid_corr,
        "likely_artifact": artifact_analysis.likely_artifact,
        "recommendation": _get_recommendation(artifact_analysis, post_resid_corr),
    }

    return results


def _get_recommendation(analysis: ArtifactAnalysisResult, post_resid_corr: float) -> str:
    """Generate recommendation based on analysis."""

    if abs(analysis.correlation_with_latent_norm) < 0.3:
        return "Low HF correlation - proceed without modification"

    if analysis.likely_artifact:
        return (
            "EMG-like topographic pattern detected. "
            "STRONGLY recommend using residualized latent for clustering/classification."
        )

    if abs(post_resid_corr) < 0.1:
        return (
            "Residualization effective. Run analyses with both original and "
            "residualized latents, report if results differ."
        )

    return (
        "Moderate HF correlation with unclear source. "
        "Compare results with/without residualization to ensure robustness."
    )


@dataclass
class FoldDiagnosticReport:
    """Diagnostic report for a single fold (per critic agent recommendation).

    This logs all metrics needed to verify artifact control without
    optimizing on them - just record for later analysis.
    """

    fold_idx: int

    # HF correlations with latent (raw)
    hf1_correlation_raw: float  # 30-48 Hz
    hf2_correlation_raw: float | None  # 70-110 Hz (if available)

    # HF correlations with latent (after residualization)
    hf1_correlation_residualized: float
    hf2_correlation_residualized: float | None

    # Topographic analysis
    hf1_temporal_central_ratio: float | None
    hf2_temporal_central_ratio: float | None

    # State predictability from HF alone
    hf_state_prediction_accuracy: float | None
    hf_state_kruskal_pvalue: float | None

    # Classification performance
    baseline_auc: float | None  # Without state conditioning
    state_conditioned_auc: float | None  # With state conditioning
    raw_latent_auc: float | None  # Using original latent
    residualized_latent_auc: float | None  # Using residualized latent

    # Sample counts
    n_train_segments: int
    n_test_segments: int
    n_train_subjects: int
    n_test_subjects: int

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "fold_idx": self.fold_idx,
            "hf1_correlation_raw": self.hf1_correlation_raw,
            "hf2_correlation_raw": self.hf2_correlation_raw,
            "hf1_correlation_residualized": self.hf1_correlation_residualized,
            "hf2_correlation_residualized": self.hf2_correlation_residualized,
            "hf1_temporal_central_ratio": self.hf1_temporal_central_ratio,
            "hf2_temporal_central_ratio": self.hf2_temporal_central_ratio,
            "hf_state_prediction_accuracy": self.hf_state_prediction_accuracy,
            "hf_state_kruskal_pvalue": self.hf_state_kruskal_pvalue,
            "baseline_auc": self.baseline_auc,
            "state_conditioned_auc": self.state_conditioned_auc,
            "raw_latent_auc": self.raw_latent_auc,
            "residualized_latent_auc": self.residualized_latent_auc,
            "n_train_segments": self.n_train_segments,
            "n_test_segments": self.n_test_segments,
            "n_train_subjects": self.n_train_subjects,
            "n_test_subjects": self.n_test_subjects,
        }


def compute_fold_diagnostics(
    raw_data_train: np.ndarray,
    raw_data_test: np.ndarray,
    latents_train: np.ndarray,
    latents_test: np.ndarray,
    sfreq: float,
    fold_idx: int,
    channel_names: list[str] | None = None,
    state_labels_train: np.ndarray | None = None,
    train_subject_ids: np.ndarray | None = None,
    test_subject_ids: np.ndarray | None = None,
    hf1_band: tuple[float, float] = (30.0, 48.0),
    hf2_band: tuple[float, float] = (70.0, 110.0),
) -> FoldDiagnosticReport:
    """
    Compute diagnostic metrics for a single cross-validation fold.

    This implements the critic agent's recommended logging:
    - HF1 and HF2 correlation with latent (raw + residualized)
    - Temporal / central HF power ratio
    - HF-only predictability of state labels

    Classification metrics (baseline_auc, state_conditioned_auc, etc.)
    are set to None and should be filled in by the calling code
    after running classification.

    Args:
        raw_data_train: Training EEG data (n_train, n_channels, n_times)
        raw_data_test: Test EEG data (n_test, n_channels, n_times)
        latents_train: Training latents (n_train, latent_dim)
        latents_test: Test latents (n_test, latent_dim)
        sfreq: Sampling frequency
        fold_idx: Fold index for logging
        channel_names: Optional channel names for topographic analysis
        state_labels_train: Optional state labels for HF-state testing
        train_subject_ids: Subject IDs for training samples
        test_subject_ids: Subject IDs for test samples
        hf1_band: Lower HF band (default 30-48 Hz)
        hf2_band: Upper HF band (default 70-110 Hz)

    Returns:
        FoldDiagnosticReport with all diagnostic metrics
    """
    nyquist = sfreq / 2

    # Compute HF1 for train and test
    hf1_train = compute_hf_power(
        raw_data_train, sfreq, band=hf1_band,
        method="welch", channel_names=channel_names
    )
    hf1_test = compute_hf_power(
        raw_data_test, sfreq, band=hf1_band,
        method="welch", channel_names=channel_names
    )

    # Raw correlations with latent
    latent_norms_train = np.linalg.norm(latents_train, axis=1)
    latent_norms_test = np.linalg.norm(latents_test, axis=1)

    hf1_corr_raw, _ = pearsonr(hf1_train.hf_power, latent_norms_train)

    # Topographic ratio for HF1
    hf1_ratio = None
    if hf1_train.hf_power_temporal is not None and hf1_train.hf_power_central is not None:
        hf1_ratio = np.mean(hf1_train.hf_power_temporal) / (np.mean(hf1_train.hf_power_central) + 1e-10)

    # Compute HF2 if sampling rate allows
    hf2_corr_raw = None
    hf2_ratio = None
    hf2_train_power = None
    hf2_test_power = None

    if hf2_band[1] < nyquist:
        hf2_train = compute_hf_power(
            raw_data_train, sfreq, band=hf2_band,
            method="welch", channel_names=channel_names
        )
        hf2_test = compute_hf_power(
            raw_data_test, sfreq, band=hf2_band,
            method="welch", channel_names=channel_names
        )
        hf2_train_power = hf2_train.hf_power
        hf2_test_power = hf2_test.hf_power

        hf2_corr_raw, _ = pearsonr(hf2_train.hf_power, latent_norms_train)

        if hf2_train.hf_power_temporal is not None and hf2_train.hf_power_central is not None:
            hf2_ratio = np.mean(hf2_train.hf_power_temporal) / (np.mean(hf2_train.hf_power_central) + 1e-10)

    # Residualization
    resid_train, resid_test, _ = residualize_latent(
        latents_train, latents_test,
        hf1_train.hf_power, hf1_test.hf_power
    )

    resid_norms_train = np.linalg.norm(resid_train, axis=1)
    hf1_corr_resid, _ = pearsonr(hf1_train.hf_power, resid_norms_train)

    hf2_corr_resid = None
    if hf2_train_power is not None:
        hf2_corr_resid, _ = pearsonr(hf2_train_power, resid_norms_train)

    # State-HF analysis (if state labels provided)
    hf_state_pred_acc = None
    hf_state_kruskal_p = None

    if state_labels_train is not None:
        state_hf_result = check_states_vs_hf_proxy(state_labels_train, hf1_train.hf_power)
        hf_state_pred_acc = state_hf_result.get("prediction_accuracy")
        hf_state_kruskal_p = state_hf_result.get("kruskal_pvalue")

    # Count unique subjects
    n_train_subjects = len(np.unique(train_subject_ids)) if train_subject_ids is not None else 0
    n_test_subjects = len(np.unique(test_subject_ids)) if test_subject_ids is not None else 0

    return FoldDiagnosticReport(
        fold_idx=fold_idx,
        hf1_correlation_raw=hf1_corr_raw,
        hf2_correlation_raw=hf2_corr_raw,
        hf1_correlation_residualized=hf1_corr_resid,
        hf2_correlation_residualized=hf2_corr_resid,
        hf1_temporal_central_ratio=hf1_ratio,
        hf2_temporal_central_ratio=hf2_ratio,
        hf_state_prediction_accuracy=hf_state_pred_acc,
        hf_state_kruskal_pvalue=hf_state_kruskal_p,
        baseline_auc=None,  # To be filled by classification
        state_conditioned_auc=None,
        raw_latent_auc=None,
        residualized_latent_auc=None,
        n_train_segments=len(latents_train),
        n_test_segments=len(latents_test),
        n_train_subjects=n_train_subjects,
        n_test_subjects=n_test_subjects,
    )


def format_diagnostics_summary(reports: list[FoldDiagnosticReport]) -> str:
    """
    Format diagnostic reports as a human-readable summary.

    Args:
        reports: List of FoldDiagnosticReport from each fold

    Returns:
        Formatted string summary
    """
    lines = []
    lines.append("=" * 60)
    lines.append("PRE-INTEGRATION DIAGNOSTIC SUMMARY")
    lines.append("=" * 60)

    # Aggregate HF correlations
    hf1_raw = [r.hf1_correlation_raw for r in reports]
    hf1_resid = [r.hf1_correlation_residualized for r in reports]

    lines.append(f"\nHF1 (30-48 Hz) Correlation with Latent:")
    lines.append(f"  Raw:          {np.mean(hf1_raw):.3f} ± {np.std(hf1_raw):.3f}")
    lines.append(f"  Residualized: {np.mean(hf1_resid):.3f} ± {np.std(hf1_resid):.3f}")

    hf2_raw = [r.hf2_correlation_raw for r in reports if r.hf2_correlation_raw is not None]
    hf2_resid = [r.hf2_correlation_residualized for r in reports if r.hf2_correlation_residualized is not None]

    if hf2_raw:
        lines.append(f"\nHF2 (70-110 Hz) Correlation with Latent:")
        lines.append(f"  Raw:          {np.mean(hf2_raw):.3f} ± {np.std(hf2_raw):.3f}")
        lines.append(f"  Residualized: {np.mean(hf2_resid):.3f} ± {np.std(hf2_resid):.3f}")

    # Topographic ratios
    hf1_ratios = [r.hf1_temporal_central_ratio for r in reports if r.hf1_temporal_central_ratio is not None]
    if hf1_ratios:
        lines.append(f"\nTopographic Analysis (temporal/central ratio):")
        lines.append(f"  HF1: {np.mean(hf1_ratios):.2f} ± {np.std(hf1_ratios):.2f}")
        if np.mean(hf1_ratios) > 1.5:
            lines.append("  WARNING: High temporal/central ratio suggests EMG contribution")
        else:
            lines.append("  OK: Ratio suggests neural source")

    # State-HF predictability
    state_pred = [r.hf_state_prediction_accuracy for r in reports if r.hf_state_prediction_accuracy is not None]
    if state_pred:
        lines.append(f"\nHF-only State Predictability:")
        lines.append(f"  Accuracy: {np.mean(state_pred):.3f} ± {np.std(state_pred):.3f}")
        if np.mean(state_pred) > 0.7:
            lines.append("  WARNING: States may be HF-driven - check interpretation")
        else:
            lines.append("  OK: States not easily predicted from HF alone")

    # Classification results (if available)
    baseline = [r.baseline_auc for r in reports if r.baseline_auc is not None]
    state_cond = [r.state_conditioned_auc for r in reports if r.state_conditioned_auc is not None]
    raw_auc = [r.raw_latent_auc for r in reports if r.raw_latent_auc is not None]
    resid_auc = [r.residualized_latent_auc for r in reports if r.residualized_latent_auc is not None]

    if baseline or state_cond:
        lines.append(f"\nClassification Performance (AUC):")
        if baseline:
            lines.append(f"  Baseline:         {np.mean(baseline):.3f} ± {np.std(baseline):.3f}")
        if state_cond:
            lines.append(f"  State-conditioned: {np.mean(state_cond):.3f} ± {np.std(state_cond):.3f}")
        if raw_auc:
            lines.append(f"  Raw latent:       {np.mean(raw_auc):.3f} ± {np.std(raw_auc):.3f}")
        if resid_auc:
            lines.append(f"  Residualized:     {np.mean(resid_auc):.3f} ± {np.std(resid_auc):.3f}")

        if baseline and state_cond:
            improvement = np.mean(state_cond) - np.mean(baseline)
            lines.append(f"\n  State conditioning effect: {improvement:+.3f}")
        if raw_auc and resid_auc:
            resid_effect = np.mean(resid_auc) - np.mean(raw_auc)
            lines.append(f"  Residualization effect:   {resid_effect:+.3f}")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)
