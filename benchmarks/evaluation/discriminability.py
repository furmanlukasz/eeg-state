"""
Discriminability evaluation: ANOVA with eta-squared effect sizes.

Computes per-feature discriminability between groups, matching
the FlowPrint paper's approach.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy import stats

from benchmarks.registry import BenchmarkResult

logger = logging.getLogger(__name__)


def compute_discriminability(
    result: BenchmarkResult,
    feature_names: Optional[list[str]] = None,
) -> dict:
    """
    Compute ANOVA discriminability for each feature.

    Args:
        result: BenchmarkResult with features and labels
        feature_names: Override feature names (defaults to metadata)

    Returns:
        Dict with per-feature metrics:
            {feature_name: {F, p, eta_squared, effect_size_label}}
        Plus summary metrics.
    """
    features = np.array(result.features)
    labels = np.array(result.labels)
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        raise ValueError(f"Need at least 2 groups, got {len(unique_labels)}")

    names = feature_names or result.metadata.get("feature_names", [])
    if not names:
        names = [f"feature_{i}" for i in range(features.shape[1])]

    results = {}
    for i, name in enumerate(names):
        feat = features[:, i]

        # Group values
        groups = [feat[labels == lab] for lab in unique_labels]

        # One-way ANOVA
        if all(len(g) >= 2 for g in groups):
            f_stat, p_value = stats.f_oneway(*groups)

            # Eta-squared: SS_between / SS_total
            grand_mean = np.mean(feat)
            ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
            ss_total = np.sum((feat - grand_mean) ** 2)
            eta_sq = ss_between / (ss_total + 1e-10)

            # Effect size label
            if eta_sq >= 0.14:
                effect_label = "large"
            elif eta_sq >= 0.06:
                effect_label = "medium"
            else:
                effect_label = "small"

            results[name] = {
                "F": float(f_stat),
                "p": float(p_value),
                "eta_squared": float(eta_sq),
                "effect_size": effect_label,
                "group_means": {
                    str(lab): float(np.mean(g))
                    for lab, g in zip(unique_labels, groups)
                },
                "group_stds": {
                    str(lab): float(np.std(g))
                    for lab, g in zip(unique_labels, groups)
                },
            }
        else:
            results[name] = {
                "F": float("nan"),
                "p": float("nan"),
                "eta_squared": float("nan"),
                "effect_size": "insufficient_data",
            }

    # Summary
    valid_etas = [r["eta_squared"] for r in results.values() if not np.isnan(r["eta_squared"])]
    significant = [
        name for name, r in results.items()
        if r["p"] < 0.05 and not np.isnan(r["p"])
    ]

    summary = {
        "n_features": len(names),
        "n_significant_p05": len(significant),
        "significant_features": significant,
        "mean_eta_squared": float(np.mean(valid_etas)) if valid_etas else 0.0,
        "max_eta_squared": float(np.max(valid_etas)) if valid_etas else 0.0,
        "best_feature": max(results, key=lambda k: results[k].get("eta_squared", 0)),
    }

    return {"features": results, "summary": summary}
