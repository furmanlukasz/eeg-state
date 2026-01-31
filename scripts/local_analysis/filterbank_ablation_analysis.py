#!/usr/bin/env python3
"""
Filterbank Ablation Analysis

Compares narrowband filterbank models with broadband models to address:
1. "Broadband Hilbert phase is not interpretable" critique
2. "Speed = amplitude proxy" critique (when using phase-only filterbank)

Analyzes:
- Ablation A: Filterbank phase-only (narrowband cos/sin, no amplitude)
- Ablation B: Filterbank full (narrowband cos/sin + log_amp)
- Comparison with broadband phase-only and full models

Usage:
    # After training filterbank models on RunPod:
    EEG_DATASET=meditation_bids python scripts/local_analysis/filterbank_ablation_analysis.py \
        --filterbank-phase-only-checkpoint outputs/ablation_filterbank_phase_only/.../best.pt \
        --filterbank-full-checkpoint outputs/ablation_filterbank_full/.../best.pt \
        --broadband-phase-only-checkpoint outputs/ablation_phase_only/.../best.pt \
        --broadband-full-checkpoint outputs/ablation_no_contrastive/.../best.pt

Author: Claude + Łukasz Furman
Date: January 2026
"""

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from config import DATASET, OUTPUT_DIR


@dataclass
class ModelResult:
    """Results for a single model."""
    name: str
    representation: str  # "broadband" or "filterbank"
    include_amplitude: bool

    # Group statistics
    expert_mean_speed: float
    expert_std_speed: float
    novice_mean_speed: float
    novice_std_speed: float

    # Effect size
    cohens_d: float
    ci_lower: float
    ci_upper: float

    # Sample sizes
    n_expert: int
    n_novice: int


@dataclass
class FilterbankAblationResult:
    """Complete filterbank ablation analysis results."""
    # Individual model results
    filterbank_phase_only: ModelResult | None
    filterbank_full: ModelResult | None
    broadband_phase_only: ModelResult | None
    broadband_full: ModelResult | None

    # Comparisons
    filterbank_vs_broadband_phase_only: dict | None
    filterbank_vs_broadband_full: dict | None
    phase_only_vs_full_filterbank: dict | None

    # Summary
    all_effects_consistent: bool
    narrowband_validates_broadband: bool


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0.0


def bootstrap_ci(group1: np.ndarray, group2: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    """Compute bootstrap confidence interval for Cohen's d."""
    d_samples = []
    for _ in range(n_bootstrap):
        idx1 = np.random.randint(0, len(group1), len(group1))
        idx2 = np.random.randint(0, len(group2), len(group2))
        d_samples.append(compute_cohens_d(group1[idx1], group2[idx2]))

    alpha = (1 - ci) / 2
    return np.percentile(d_samples, alpha * 100), np.percentile(d_samples, (1 - alpha) * 100)


def load_model_and_analyze(
    checkpoint_path: Path,
    name: str,
    representation: str,
    include_amplitude: bool,
    device: str = "mps",
) -> ModelResult | None:
    """Load a model and compute speed statistics."""

    if not checkpoint_path.exists():
        print(f"  Checkpoint not found: {checkpoint_path}")
        return None

    print(f"\nLoading {name} from: {checkpoint_path}")

    # Import here to avoid circular imports
    from load_data import load_model_and_compute_trajectories

    try:
        trajectories_by_group = load_model_and_compute_trajectories(
            checkpoint_path=checkpoint_path,
            device=device,
            return_amplitudes=False,
        )
    except Exception as e:
        print(f"  Error loading model: {e}")
        return None

    # Compute mean speed per subject
    expert_speeds = []
    novice_speeds = []

    for group_name, trajectories in trajectories_by_group.items():
        for traj in trajectories:
            # Compute speed: ||h(t+1) - h(t)||
            if len(traj) > 1:
                velocity = np.diff(traj, axis=0)
                speed = np.linalg.norm(velocity, axis=1)
                mean_speed = np.mean(speed)

                if "expert" in group_name.lower() or "meditator" in group_name.lower():
                    expert_speeds.append(mean_speed)
                else:
                    novice_speeds.append(mean_speed)

    expert_speeds = np.array(expert_speeds)
    novice_speeds = np.array(novice_speeds)

    # Compute effect size
    d = compute_cohens_d(expert_speeds, novice_speeds)
    ci_low, ci_high = bootstrap_ci(expert_speeds, novice_speeds)

    print(f"  Expert: {np.mean(expert_speeds):.3f} ± {np.std(expert_speeds):.3f} (n={len(expert_speeds)})")
    print(f"  Novice: {np.mean(novice_speeds):.3f} ± {np.std(novice_speeds):.3f} (n={len(novice_speeds)})")
    print(f"  Cohen's d: {d:.3f} [{ci_low:.3f}, {ci_high:.3f}]")

    return ModelResult(
        name=name,
        representation=representation,
        include_amplitude=include_amplitude,
        expert_mean_speed=float(np.mean(expert_speeds)),
        expert_std_speed=float(np.std(expert_speeds)),
        novice_mean_speed=float(np.mean(novice_speeds)),
        novice_std_speed=float(np.std(novice_speeds)),
        cohens_d=float(d),
        ci_lower=float(ci_low),
        ci_upper=float(ci_high),
        n_expert=len(expert_speeds),
        n_novice=len(novice_speeds),
    )


def compare_models(model1: ModelResult, model2: ModelResult) -> dict:
    """Compare two models' effect sizes."""
    return {
        "model1_name": model1.name,
        "model2_name": model2.name,
        "model1_d": model1.cohens_d,
        "model2_d": model2.cohens_d,
        "same_direction": np.sign(model1.cohens_d) == np.sign(model2.cohens_d),
        "magnitude_ratio": abs(model1.cohens_d / model2.cohens_d) if model2.cohens_d != 0 else float('inf'),
    }


def create_comparison_figure(results: FilterbankAblationResult, output_path: Path):
    """Create visualization of filterbank ablation results."""

    # Collect available models
    models = []
    if results.broadband_full:
        models.append(("Broadband\n(phase+amp)", results.broadband_full))
    if results.broadband_phase_only:
        models.append(("Broadband\n(phase only)", results.broadband_phase_only))
    if results.filterbank_full:
        models.append(("Filterbank\n(phase+amp)", results.filterbank_full))
    if results.filterbank_phase_only:
        models.append(("Filterbank\n(phase only)", results.filterbank_phase_only))

    if len(models) < 2:
        print("Not enough models to create comparison figure")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Effect sizes with CIs
    ax1 = axes[0]
    names = [m[0] for m in models]
    d_values = [m[1].cohens_d for m in models]
    ci_lows = [m[1].ci_lower for m in models]
    ci_highs = [m[1].ci_upper for m in models]

    colors = ['steelblue' if 'Broadband' in n else 'darkorange' for n in names]

    x = np.arange(len(models))
    ax1.bar(x, d_values, color=colors, alpha=0.7, edgecolor='black')
    ax1.errorbar(x, d_values, yerr=[np.array(d_values) - np.array(ci_lows), np.array(ci_highs) - np.array(d_values)],
                 fmt='none', color='black', capsize=5)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9)
    ax1.set_ylabel("Cohen's d (Expert - Novice)")
    ax1.set_title("Effect Size Comparison\n(negative = experts slower)")

    # Right: Summary table
    ax2 = axes[1]
    ax2.axis('off')

    table_data = []
    for name, model in models:
        ci_str = f"[{model.ci_lower:.2f}, {model.ci_upper:.2f}]"
        table_data.append([
            name.replace('\n', ' '),
            f"{model.cohens_d:.3f}",
            ci_str,
            "Yes" if model.ci_upper < 0 else "No"
        ])

    table = ax2.table(
        cellText=table_data,
        colLabels=["Model", "Cohen's d", "95% CI", "CI excludes 0"],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax2.set_title("Summary Statistics", fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved figure: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Filterbank ablation analysis")
    parser.add_argument("--filterbank-phase-only-checkpoint", type=Path, default=None,
                        help="Path to filterbank phase-only model checkpoint")
    parser.add_argument("--filterbank-full-checkpoint", type=Path, default=None,
                        help="Path to filterbank full model checkpoint")
    parser.add_argument("--broadband-phase-only-checkpoint", type=Path, default=None,
                        help="Path to broadband phase-only model checkpoint")
    parser.add_argument("--broadband-full-checkpoint", type=Path, default=None,
                        help="Path to broadband full model checkpoint")
    parser.add_argument("--device", type=str, default="mps",
                        help="Device for inference")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for results")
    args = parser.parse_args()

    # Setup output
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = OUTPUT_DIR / f"filterbank_ablation_{DATASET}_{timestamp}"
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FILTERBANK ABLATION ANALYSIS")
    print("=" * 70)
    print(f"Dataset: {DATASET}")
    print(f"Output: {output_dir}")

    # Load and analyze each model
    fb_phase_only = None
    fb_full = None
    bb_phase_only = None
    bb_full = None

    if args.filterbank_phase_only_checkpoint:
        fb_phase_only = load_model_and_analyze(
            args.filterbank_phase_only_checkpoint,
            "Filterbank Phase-Only",
            "filterbank",
            include_amplitude=False,
            device=args.device,
        )

    if args.filterbank_full_checkpoint:
        fb_full = load_model_and_analyze(
            args.filterbank_full_checkpoint,
            "Filterbank Full",
            "filterbank",
            include_amplitude=True,
            device=args.device,
        )

    if args.broadband_phase_only_checkpoint:
        bb_phase_only = load_model_and_analyze(
            args.broadband_phase_only_checkpoint,
            "Broadband Phase-Only",
            "broadband",
            include_amplitude=False,
            device=args.device,
        )

    if args.broadband_full_checkpoint:
        bb_full = load_model_and_analyze(
            args.broadband_full_checkpoint,
            "Broadband Full",
            "broadband",
            include_amplitude=True,
            device=args.device,
        )

    # Compute comparisons
    comparisons = {}

    if fb_phase_only and bb_phase_only:
        comparisons["filterbank_vs_broadband_phase_only"] = compare_models(fb_phase_only, bb_phase_only)

    if fb_full and bb_full:
        comparisons["filterbank_vs_broadband_full"] = compare_models(fb_full, bb_full)

    if fb_phase_only and fb_full:
        comparisons["phase_only_vs_full_filterbank"] = compare_models(fb_phase_only, fb_full)

    # Determine if effects are consistent
    available_models = [m for m in [fb_phase_only, fb_full, bb_phase_only, bb_full] if m is not None]
    all_consistent = all(m.cohens_d < 0 for m in available_models) if available_models else False

    # Narrowband validates broadband if both show same direction
    narrowband_validates = False
    if fb_phase_only and bb_phase_only:
        narrowband_validates = np.sign(fb_phase_only.cohens_d) == np.sign(bb_phase_only.cohens_d)

    # Create results object
    results = FilterbankAblationResult(
        filterbank_phase_only=fb_phase_only,
        filterbank_full=fb_full,
        broadband_phase_only=bb_phase_only,
        broadband_full=bb_full,
        filterbank_vs_broadband_phase_only=comparisons.get("filterbank_vs_broadband_phase_only"),
        filterbank_vs_broadband_full=comparisons.get("filterbank_vs_broadband_full"),
        phase_only_vs_full_filterbank=comparisons.get("phase_only_vs_full_filterbank"),
        all_effects_consistent=all_consistent,
        narrowband_validates_broadband=narrowband_validates,
    )

    # Create figure
    create_comparison_figure(results, output_dir / "filterbank_ablation_results.png")

    # Save results
    results_dict = {
        "filterbank_phase_only": asdict(fb_phase_only) if fb_phase_only else None,
        "filterbank_full": asdict(fb_full) if fb_full else None,
        "broadband_phase_only": asdict(bb_phase_only) if bb_phase_only else None,
        "broadband_full": asdict(bb_full) if bb_full else None,
        "comparisons": comparisons,
        "all_effects_consistent": all_consistent,
        "narrowband_validates_broadband": narrowband_validates,
    }

    with open(output_dir / "filterbank_ablation_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY: FILTERBANK ABLATION")
    print("=" * 70)

    if all_consistent:
        print("\n✓ All models show CONSISTENT effect direction (experts slower)")
    else:
        print("\n✗ Effect direction NOT consistent across models")

    if narrowband_validates:
        print("✓ Narrowband filterbank VALIDATES broadband result")
        print("  → 'Broadband Hilbert interpretability' critique addressed")

    if fb_phase_only and fb_phase_only.ci_upper < 0:
        print("✓ Filterbank phase-only effect significant (CI excludes 0)")
        print("  → Both 'broadband Hilbert' AND 'amplitude proxy' critiques addressed")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
