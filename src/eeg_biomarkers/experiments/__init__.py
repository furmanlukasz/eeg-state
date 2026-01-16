"""Experiment modules for EEG biomarker discovery."""

from eeg_biomarkers.experiments.integration_experiment import (
    ExperimentConfig,
    FoldResult,
    run_experiment,
    generate_report,
)

__all__ = [
    "ExperimentConfig",
    "FoldResult",
    "run_experiment",
    "generate_report",
]
