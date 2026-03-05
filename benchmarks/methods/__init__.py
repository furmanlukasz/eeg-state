"""
Benchmark methods — each transforms a DataBundle into a BenchmarkResult.

Available methods:
    - flow_metrics: Encode → latent trajectories → speed/tortuosity/variance
    - flowprint_metrics: FlowPrint dynamical microscope on EEG data (requires flowprint)
    - rqa_features: Encode → latent → RQA features
    - (future) hmm_baseline: Raw → HMM state decomposition
    - (future) transformer_classify: End-to-end transformer features → classify
"""

# Import methods to trigger registration
from benchmarks.methods import flow_metrics  # noqa: F401

# Conditionally register flowprint method if available
try:
    from benchmarks.methods import flowprint_metrics  # noqa: F401
except ImportError:
    pass  # flowprint not installed — method not available
