"""
Benchmark system for cross-dataset evaluation of EEG analysis methods.

Provides a standardized interface for comparing methods across datasets
with environment-aware paths (local Mac vs RunPod GPU).

Usage:
    # Quick test (2-3 subjects per group):
    python -m benchmarks --dataset greek_resting --method flow_metrics --quick

    # Full benchmark:
    python -m benchmarks --dataset meditation_bids --method flow_metrics

    # All datasets × all methods:
    python -m benchmarks --all

    # List available datasets and methods:
    python -m benchmarks --list
"""

# Import methods to trigger registration when benchmarks is imported
import benchmarks.methods  # noqa: F401
