"""Analysis tools for biomarker discovery."""

from eeg_biomarkers.analysis.rqa import compute_rqa_features, RQAFeatures
from eeg_biomarkers.analysis.state_discovery import StateDiscovery
from eeg_biomarkers.analysis.classification import ClassificationPipeline
from eeg_biomarkers.analysis.artifact_control import (
    compute_hf_power,
    compute_dual_band_hf,
    analyze_latent_hf_correlation,
    residualize_latent,
    check_states_vs_hf_proxy,
    run_artifact_control_analysis,
    compute_fold_diagnostics,
    format_diagnostics_summary,
    HFProxyResult,
    DualBandHFResult,
    ArtifactAnalysisResult,
    FoldDiagnosticReport,
)

__all__ = [
    "compute_rqa_features",
    "RQAFeatures",
    "StateDiscovery",
    "ClassificationPipeline",
    # Artifact control
    "compute_hf_power",
    "compute_dual_band_hf",
    "analyze_latent_hf_correlation",
    "residualize_latent",
    "check_states_vs_hf_proxy",
    "run_artifact_control_analysis",
    "compute_fold_diagnostics",
    "format_diagnostics_summary",
    "HFProxyResult",
    "DualBandHFResult",
    "ArtifactAnalysisResult",
    "FoldDiagnosticReport",
]
