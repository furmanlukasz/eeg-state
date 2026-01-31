#!/usr/bin/env python3
"""
Amplitude Ablation Analysis Script

Implements three controls to prove speed effect is not just amplitude proxy:

1. AMPLITUDE-MATCHED SPEED COMPARISON
   - Bin time points by amplitude deciles
   - Compare expert vs novice speed within each bin
   - If effect persists → amplitude cannot be sole driver

2. PHASE-ONLY CONTROL
   - Use model trained without amplitude features
   - Compare speed effect direction/magnitude
   - If preserved → effect reflects phase coordination dynamics

3. PREDICTIVE REGRESSION
   - Predict group from amplitude alone vs amplitude + speed
   - If speed adds explanatory power → independent dynamical descriptor

Usage:
    EEG_DATASET=meditation_bids python scripts/local_analysis/amplitude_ablation_analysis.py

    # With phase-only checkpoint:
    EEG_DATASET=meditation_bids python scripts/local_analysis/amplitude_ablation_analysis.py \
        --phase-only-checkpoint outputs/ablation_phase_only/.../checkpoints/best.pt

Author: PhD EEG Analysis
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# Add local paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    CHECKPOINT_PATH,
    DATA_PATHS,
    DATASET,
    OUTPUT_DIR,
    ensure_output_dir,
)


@dataclass
class AmplitudeMatchedResult:
    """Results from amplitude-matched speed comparison."""
    n_bins: int
    bin_edges: list
    expert_speeds_by_bin: list  # Mean speed per bin
    novice_speeds_by_bin: list
    effect_per_bin: list  # Cohen's d per bin
    pooled_effect: float  # Average effect across bins
    pooled_ci: tuple  # 95% CI for pooled effect
    n_expert_samples_by_bin: list
    n_novice_samples_by_bin: list


@dataclass
class PhaseOnlyResult:
    """Results from phase-only control."""
    has_phase_only_model: bool
    full_model_effect: float  # Cohen's d with amplitude
    full_model_ci: tuple
    phase_only_effect: Optional[float]  # Cohen's d phase-only
    phase_only_ci: Optional[tuple]
    direction_preserved: Optional[bool]  # Same sign?
    magnitude_ratio: Optional[float]  # phase_only / full


@dataclass
class RegressionResult:
    """Results from predictive regression."""
    amplitude_only_accuracy: float
    speed_only_accuracy: float
    combined_accuracy: float
    amplitude_only_auc: float
    speed_only_auc: float
    combined_auc: float
    speed_adds_value: bool  # Does speed improve over amplitude?
    improvement_pct: float  # % improvement in AUC


def compute_amplitude_matched_comparison(
    trajectories_by_group: dict,
    amplitudes_by_group: dict,
    n_bins: int = 10,
    n_bootstrap: int = 1000,
) -> AmplitudeMatchedResult:
    """
    Compare speed within amplitude-matched bins.

    This is the cleanest kill for the amplitude-proxy critique.
    If experts are still slower than novices at the SAME amplitude,
    amplitude cannot be the sole driver.
    """
    from velocity import compute_speed, VelocityConfig

    velocity_config = VelocityConfig(method="savgol", savgol_window=5, savgol_poly=2)

    # Collect all amplitude values to define global bins
    all_amplitudes = []
    for group_amps in amplitudes_by_group.values():
        for subject_amps in group_amps:
            all_amplitudes.extend(subject_amps.flatten())

    # Define amplitude bins (deciles)
    bin_edges = np.percentile(all_amplitudes, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-6  # Ensure max value is included

    # Collect speed and amplitude for each group
    group_data = {}
    for group_name in trajectories_by_group.keys():
        speeds = []
        amps = []

        trajectories = trajectories_by_group[group_name]
        amplitudes = amplitudes_by_group[group_name]

        for traj, amp in zip(trajectories, amplitudes):
            # Compute speed
            speed = compute_speed(traj, config=velocity_config)

            # Align lengths (speed is 1 shorter due to derivatives)
            min_len = min(len(speed), len(amp))
            speeds.extend(speed[:min_len])
            amps.extend(amp[:min_len])

        group_data[group_name] = {
            'speeds': np.array(speeds),
            'amplitudes': np.array(amps),
        }

    # Get group names (expect 'expert' and 'novice' for meditation)
    group_names = list(group_data.keys())
    expert_key = [g for g in group_names if 'expert' in g.lower()][0]
    novice_key = [g for g in group_names if 'novice' in g.lower()][0]

    expert_speeds = group_data[expert_key]['speeds']
    expert_amps = group_data[expert_key]['amplitudes']
    novice_speeds = group_data[novice_key]['speeds']
    novice_amps = group_data[novice_key]['amplitudes']

    # Compute effect within each amplitude bin
    expert_speeds_by_bin = []
    novice_speeds_by_bin = []
    effect_per_bin = []
    n_expert_by_bin = []
    n_novice_by_bin = []

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]

        # Select samples in this amplitude bin
        expert_mask = (expert_amps >= low) & (expert_amps < high)
        novice_mask = (novice_amps >= low) & (novice_amps < high)

        expert_in_bin = expert_speeds[expert_mask]
        novice_in_bin = novice_speeds[novice_mask]

        n_expert_by_bin.append(len(expert_in_bin))
        n_novice_by_bin.append(len(novice_in_bin))

        if len(expert_in_bin) > 10 and len(novice_in_bin) > 10:
            expert_speeds_by_bin.append(float(np.mean(expert_in_bin)))
            novice_speeds_by_bin.append(float(np.mean(novice_in_bin)))

            # Cohen's d for this bin
            pooled_std = np.sqrt(
                (np.var(expert_in_bin) + np.var(novice_in_bin)) / 2
            )
            d = (np.mean(expert_in_bin) - np.mean(novice_in_bin)) / pooled_std
            effect_per_bin.append(float(d))
        else:
            expert_speeds_by_bin.append(np.nan)
            novice_speeds_by_bin.append(np.nan)
            effect_per_bin.append(np.nan)

    # Pooled effect: average across bins (weighted by sample size)
    valid_effects = [(e, n_e + n_n) for e, n_e, n_n in
                     zip(effect_per_bin, n_expert_by_bin, n_novice_by_bin)
                     if not np.isnan(e)]

    if valid_effects:
        effects, weights = zip(*valid_effects)
        pooled_effect = float(np.average(effects, weights=weights))

        # Bootstrap CI for pooled effect
        bootstrap_effects = []
        for _ in range(n_bootstrap):
            # Resample bins
            indices = np.random.choice(len(effects), size=len(effects), replace=True)
            sampled_effects = [effects[i] for i in indices]
            sampled_weights = [weights[i] for i in indices]
            bootstrap_effects.append(np.average(sampled_effects, weights=sampled_weights))

        pooled_ci = (
            float(np.percentile(bootstrap_effects, 2.5)),
            float(np.percentile(bootstrap_effects, 97.5))
        )
    else:
        pooled_effect = np.nan
        pooled_ci = (np.nan, np.nan)

    return AmplitudeMatchedResult(
        n_bins=n_bins,
        bin_edges=bin_edges.tolist(),
        expert_speeds_by_bin=expert_speeds_by_bin,
        novice_speeds_by_bin=novice_speeds_by_bin,
        effect_per_bin=effect_per_bin,
        pooled_effect=pooled_effect,
        pooled_ci=pooled_ci,
        n_expert_samples_by_bin=n_expert_by_bin,
        n_novice_samples_by_bin=n_novice_by_bin,
    )


def compute_phase_only_comparison(
    trajectories_full: dict,
    trajectories_phase_only: Optional[dict],
    n_bootstrap: int = 1000,
) -> PhaseOnlyResult:
    """
    Compare speed effects between full model and phase-only model.

    If effect direction is preserved in phase-only model,
    amplitude cannot be the sole driver.
    """
    from velocity import compute_speed, VelocityConfig

    velocity_config = VelocityConfig(method="savgol", savgol_window=5, savgol_poly=2)

    def compute_group_effect(trajectories_by_group):
        """Compute Cohen's d for expert vs novice speed."""
        group_names = list(trajectories_by_group.keys())
        expert_key = [g for g in group_names if 'expert' in g.lower()][0]
        novice_key = [g for g in group_names if 'novice' in g.lower()][0]

        expert_speeds = []
        for traj in trajectories_by_group[expert_key]:
            speed = compute_speed(traj, config=velocity_config)
            expert_speeds.append(float(np.mean(speed)))

        novice_speeds = []
        for traj in trajectories_by_group[novice_key]:
            speed = compute_speed(traj, config=velocity_config)
            novice_speeds.append(float(np.mean(speed)))

        expert_speeds = np.array(expert_speeds)
        novice_speeds = np.array(novice_speeds)

        pooled_std = np.sqrt(
            (np.var(expert_speeds) + np.var(novice_speeds)) / 2
        )
        d = (np.mean(expert_speeds) - np.mean(novice_speeds)) / pooled_std

        # Bootstrap CI
        bootstrap_ds = []
        for _ in range(n_bootstrap):
            exp_sample = np.random.choice(expert_speeds, size=len(expert_speeds), replace=True)
            nov_sample = np.random.choice(novice_speeds, size=len(novice_speeds), replace=True)
            pooled = np.sqrt((np.var(exp_sample) + np.var(nov_sample)) / 2)
            bootstrap_ds.append((np.mean(exp_sample) - np.mean(nov_sample)) / pooled)

        ci = (
            float(np.percentile(bootstrap_ds, 2.5)),
            float(np.percentile(bootstrap_ds, 97.5))
        )

        return float(d), ci

    # Full model effect
    full_d, full_ci = compute_group_effect(trajectories_full)

    # Phase-only effect (if available)
    if trajectories_phase_only is not None:
        phase_d, phase_ci = compute_group_effect(trajectories_phase_only)
        direction_preserved = (full_d * phase_d) > 0
        magnitude_ratio = abs(phase_d) / abs(full_d) if full_d != 0 else None
    else:
        phase_d = None
        phase_ci = None
        direction_preserved = None
        magnitude_ratio = None

    return PhaseOnlyResult(
        has_phase_only_model=trajectories_phase_only is not None,
        full_model_effect=full_d,
        full_model_ci=full_ci,
        phase_only_effect=phase_d,
        phase_only_ci=phase_ci,
        direction_preserved=direction_preserved,
        magnitude_ratio=magnitude_ratio,
    )


def compute_predictive_regression(
    trajectories_by_group: dict,
    amplitudes_by_group: dict,
) -> RegressionResult:
    """
    Predict group from amplitude alone vs amplitude + speed.

    If speed adds explanatory power beyond amplitude,
    it is an independent dynamical descriptor.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from velocity import compute_speed, VelocityConfig

    velocity_config = VelocityConfig(method="savgol", savgol_window=5, savgol_poly=2)

    # Prepare subject-level features
    features = []
    labels = []

    group_names = list(trajectories_by_group.keys())
    expert_key = [g for g in group_names if 'expert' in g.lower()][0]
    novice_key = [g for g in group_names if 'novice' in g.lower()][0]

    for group_name, label in [(expert_key, 0), (novice_key, 1)]:
        trajectories = trajectories_by_group[group_name]
        amplitudes = amplitudes_by_group[group_name]

        for traj, amp in zip(trajectories, amplitudes):
            speed = compute_speed(traj, config=velocity_config)

            # Subject-level features
            mean_speed = np.mean(speed)
            mean_amplitude = np.mean(amp)

            features.append({
                'mean_amplitude': mean_amplitude,
                'mean_speed': mean_speed,
            })
            labels.append(label)

    df = pd.DataFrame(features)
    X_amp = df[['mean_amplitude']].values
    X_speed = df[['mean_speed']].values
    X_combined = df[['mean_amplitude', 'mean_speed']].values
    y = np.array(labels)

    # Standardize
    scaler = StandardScaler()
    X_amp_scaled = scaler.fit_transform(X_amp)
    X_speed_scaled = scaler.fit_transform(X_speed)
    X_combined_scaled = scaler.fit_transform(X_combined)

    # Cross-validated predictions
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Amplitude only
    clf_amp = LogisticRegression(random_state=42, max_iter=1000)
    pred_amp = cross_val_predict(clf_amp, X_amp_scaled, y, cv=cv, method='predict_proba')[:, 1]
    acc_amp = accuracy_score(y, pred_amp > 0.5)
    auc_amp = roc_auc_score(y, pred_amp)

    # Speed only
    clf_speed = LogisticRegression(random_state=42, max_iter=1000)
    pred_speed = cross_val_predict(clf_speed, X_speed_scaled, y, cv=cv, method='predict_proba')[:, 1]
    acc_speed = accuracy_score(y, pred_speed > 0.5)
    auc_speed = roc_auc_score(y, pred_speed)

    # Combined
    clf_combined = LogisticRegression(random_state=42, max_iter=1000)
    pred_combined = cross_val_predict(clf_combined, X_combined_scaled, y, cv=cv, method='predict_proba')[:, 1]
    acc_combined = accuracy_score(y, pred_combined > 0.5)
    auc_combined = roc_auc_score(y, pred_combined)

    # Does speed add value?
    speed_adds_value = auc_combined > auc_amp
    improvement_pct = ((auc_combined - auc_amp) / auc_amp) * 100

    return RegressionResult(
        amplitude_only_accuracy=float(acc_amp),
        speed_only_accuracy=float(acc_speed),
        combined_accuracy=float(acc_combined),
        amplitude_only_auc=float(auc_amp),
        speed_only_auc=float(auc_speed),
        combined_auc=float(auc_combined),
        speed_adds_value=speed_adds_value,
        improvement_pct=float(improvement_pct),
    )


def plot_amplitude_ablation_results(
    amp_matched: AmplitudeMatchedResult,
    phase_only: PhaseOnlyResult,
    regression: RegressionResult,
    output_dir: Path,
):
    """Create visualization of amplitude ablation results."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Amplitude-matched speed by bin
    ax1 = axes[0, 0]
    bins = np.arange(amp_matched.n_bins)
    width = 0.35

    ax1.bar(bins - width/2, amp_matched.expert_speeds_by_bin, width,
            label='Expert', color='steelblue', alpha=0.8)
    ax1.bar(bins + width/2, amp_matched.novice_speeds_by_bin, width,
            label='Novice', color='coral', alpha=0.8)
    ax1.set_xlabel('Amplitude Decile')
    ax1.set_ylabel('Mean Speed')
    ax1.set_title('Speed by Amplitude Bin\n(controlling for amplitude)')
    ax1.legend()
    ax1.set_xticks(bins)
    ax1.set_xticklabels([f'{i+1}' for i in bins])

    # 2. Effect size per bin
    ax2 = axes[0, 1]
    valid_bins = [i for i, e in enumerate(amp_matched.effect_per_bin) if not np.isnan(e)]
    valid_effects = [amp_matched.effect_per_bin[i] for i in valid_bins]

    colors = ['steelblue' if e < 0 else 'coral' for e in valid_effects]
    ax2.bar(valid_bins, valid_effects, color=colors, alpha=0.8)
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.axhline(amp_matched.pooled_effect, color='red', linestyle='--',
                linewidth=2, label=f'Pooled d = {amp_matched.pooled_effect:.2f}')
    ax2.fill_between([min(valid_bins)-0.5, max(valid_bins)+0.5],
                     amp_matched.pooled_ci[0], amp_matched.pooled_ci[1],
                     color='red', alpha=0.2, label='95% CI')
    ax2.set_xlabel('Amplitude Decile')
    ax2.set_ylabel("Cohen's d (Expert - Novice)")
    ax2.set_title('Effect Size by Amplitude Bin\n(negative = experts slower)')
    ax2.legend()
    ax2.set_xticks(bins)
    ax2.set_xticklabels([f'{i+1}' for i in bins])

    # 3. Full vs Phase-only comparison
    ax3 = axes[1, 0]
    if phase_only.has_phase_only_model and phase_only.phase_only_effect is not None:
        effects = [phase_only.full_model_effect, phase_only.phase_only_effect]
        cis = [phase_only.full_model_ci, phase_only.phase_only_ci]
        labels = ['Full Model\n(with amplitude)', 'Phase Only\n(no amplitude)']

        x = [0, 1]
        colors = ['steelblue' if e < 0 else 'coral' for e in effects]
        yerr = [[e - ci[0] for e, ci in zip(effects, cis)],
                [ci[1] - e for e, ci in zip(effects, cis)]]

        ax3.bar(x, effects, color=colors, alpha=0.8, yerr=yerr, capsize=5)
        ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels)
        ax3.set_ylabel("Cohen's d (Expert - Novice)")

        preserved_str = "✓ Direction preserved" if phase_only.direction_preserved else "✗ Direction flipped"
        ratio_str = f"Magnitude ratio: {phase_only.magnitude_ratio:.2f}" if phase_only.magnitude_ratio else ""
        ax3.set_title(f'Phase-Only Control\n{preserved_str}\n{ratio_str}')
    else:
        ax3.text(0.5, 0.5, 'Phase-only model\nnot available',
                ha='center', va='center', transform=ax3.transAxes,
                fontsize=14)
        ax3.set_title('Phase-Only Control\n(requires training phase-only model)')

    # 4. Predictive regression comparison
    ax4 = axes[1, 1]
    models = ['Amplitude\nOnly', 'Speed\nOnly', 'Combined']
    aucs = [regression.amplitude_only_auc, regression.speed_only_auc, regression.combined_auc]

    colors = ['gray', 'steelblue', 'green']
    ax4.bar(models, aucs, color=colors, alpha=0.8)
    ax4.axhline(0.5, color='red', linestyle='--', label='Chance')
    ax4.set_ylabel('ROC-AUC')
    ax4.set_ylim(0.4, 1.0)

    improvement_str = f"+{regression.improvement_pct:.1f}%" if regression.speed_adds_value else f"{regression.improvement_pct:.1f}%"
    ax4.set_title(f'Predictive Regression\nSpeed adds value: {"✓" if regression.speed_adds_value else "✗"} ({improvement_str})')
    ax4.legend()

    plt.tight_layout()

    output_path = output_dir / 'amplitude_ablation_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Amplitude Ablation Analysis")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Full model checkpoint (default: from config)")
    parser.add_argument("--phase-only-checkpoint", type=Path, default=None,
                        help="Phase-only model checkpoint (optional)")
    parser.add_argument("--device", type=str, default="mps",
                        help="Device for inference")
    parser.add_argument("--n-bins", type=int, default=10,
                        help="Number of amplitude bins")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Bootstrap iterations")
    args = parser.parse_args()

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ensure_output_dir() / f"amplitude_ablation_{DATASET}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("AMPLITUDE ABLATION ANALYSIS")
    print("=" * 70)
    print(f"Dataset: {DATASET}")
    print(f"Output: {output_dir}")

    # Load model
    checkpoint_path = args.checkpoint or CHECKPOINT_PATH
    print(f"\nLoading model from: {checkpoint_path}")

    import torch
    from load_data import load_model_and_compute_trajectories

    # Get trajectories and amplitudes
    print("\nComputing trajectories and amplitudes...")
    trajectories_by_group, amplitudes_by_group = load_model_and_compute_trajectories(
        checkpoint_path=checkpoint_path,
        device=args.device,
        return_amplitudes=True,
    )

    print(f"\nLoaded groups:")
    for group, trajs in trajectories_by_group.items():
        print(f"  {group}: {len(trajs)} subjects")

    # 1. Amplitude-matched comparison
    print("\n" + "=" * 50)
    print("1. AMPLITUDE-MATCHED SPEED COMPARISON")
    print("=" * 50)

    amp_matched = compute_amplitude_matched_comparison(
        trajectories_by_group,
        amplitudes_by_group,
        n_bins=args.n_bins,
        n_bootstrap=args.n_bootstrap,
    )

    print(f"\nPooled effect (across amplitude bins): d = {amp_matched.pooled_effect:.3f}")
    print(f"95% CI: [{amp_matched.pooled_ci[0]:.3f}, {amp_matched.pooled_ci[1]:.3f}]")
    print(f"\nEffect by amplitude decile:")
    for i, (e_spd, n_spd, eff) in enumerate(zip(
        amp_matched.expert_speeds_by_bin,
        amp_matched.novice_speeds_by_bin,
        amp_matched.effect_per_bin
    )):
        if not np.isnan(eff):
            print(f"  Decile {i+1}: Expert={e_spd:.3f}, Novice={n_spd:.3f}, d={eff:.3f}")

    # Check if effect persists
    ci_excludes_zero = (amp_matched.pooled_ci[0] > 0) or (amp_matched.pooled_ci[1] < 0)
    print(f"\n✓ Effect persists when amplitude is matched: {ci_excludes_zero}")

    # 2. Phase-only control
    print("\n" + "=" * 50)
    print("2. PHASE-ONLY CONTROL")
    print("=" * 50)

    if args.phase_only_checkpoint and args.phase_only_checkpoint.exists():
        print(f"Loading phase-only model from: {args.phase_only_checkpoint}")
        trajectories_phase_only, _ = load_model_and_compute_trajectories(
            checkpoint_path=args.phase_only_checkpoint,
            device=args.device,
            return_amplitudes=True,
        )
    else:
        print("Phase-only checkpoint not provided or not found.")
        print("To run phase-only control, train with:")
        print("  python -m eeg_biomarkers.training.train --config-name=experiment/ablation_phase_only")
        trajectories_phase_only = None

    phase_only = compute_phase_only_comparison(
        trajectories_by_group,
        trajectories_phase_only,
        n_bootstrap=args.n_bootstrap,
    )

    print(f"\nFull model effect: d = {phase_only.full_model_effect:.3f} "
          f"CI [{phase_only.full_model_ci[0]:.3f}, {phase_only.full_model_ci[1]:.3f}]")

    if phase_only.has_phase_only_model:
        print(f"Phase-only effect: d = {phase_only.phase_only_effect:.3f} "
              f"CI [{phase_only.phase_only_ci[0]:.3f}, {phase_only.phase_only_ci[1]:.3f}]")
        print(f"Direction preserved: {phase_only.direction_preserved}")
        if phase_only.magnitude_ratio:
            print(f"Magnitude ratio: {phase_only.magnitude_ratio:.2f}")

    # 3. Predictive regression
    print("\n" + "=" * 50)
    print("3. PREDICTIVE REGRESSION")
    print("=" * 50)

    regression = compute_predictive_regression(
        trajectories_by_group,
        amplitudes_by_group,
    )

    print(f"\nAmplitude only: AUC = {regression.amplitude_only_auc:.3f}")
    print(f"Speed only: AUC = {regression.speed_only_auc:.3f}")
    print(f"Combined: AUC = {regression.combined_auc:.3f}")
    print(f"\nSpeed adds value: {regression.speed_adds_value}")
    print(f"Improvement: {regression.improvement_pct:+.1f}%")

    # Plot results
    print("\n" + "=" * 50)
    print("GENERATING PLOTS")
    print("=" * 50)

    plot_amplitude_ablation_results(
        amp_matched, phase_only, regression, output_dir
    )

    # Save results
    results = {
        "timestamp": timestamp,
        "dataset": DATASET,
        "checkpoint": str(checkpoint_path),
        "phase_only_checkpoint": str(args.phase_only_checkpoint) if args.phase_only_checkpoint else None,
        "amplitude_matched": asdict(amp_matched),
        "phase_only": asdict(phase_only),
        "regression": asdict(regression),
    }

    results_path = output_dir / "amplitude_ablation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {results_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: AMPLITUDE-PROXY CRITIQUE REBUTTAL")
    print("=" * 70)

    print("\n1. AMPLITUDE-MATCHED COMPARISON:")
    if ci_excludes_zero:
        print("   ✓ Effect PERSISTS when amplitude is controlled")
        print(f"   → Experts slower at SAME amplitude (d = {amp_matched.pooled_effect:.2f})")
    else:
        print("   ~ Effect attenuated when amplitude matched")
        print(f"   → Pooled d = {amp_matched.pooled_effect:.2f} (CI includes zero)")

    print("\n2. PHASE-ONLY CONTROL:")
    if phase_only.has_phase_only_model:
        if phase_only.direction_preserved:
            print("   ✓ Effect direction PRESERVED without amplitude")
            print(f"   → Phase coordination alone shows same pattern")
        else:
            print("   ✗ Effect direction REVERSED without amplitude")
    else:
        print("   - Not tested (train phase-only model first)")

    print("\n3. PREDICTIVE REGRESSION:")
    if regression.speed_adds_value:
        print("   ✓ Speed ADDS explanatory power beyond amplitude")
        print(f"   → Combined AUC ({regression.combined_auc:.2f}) > Amplitude AUC ({regression.amplitude_only_auc:.2f})")
    else:
        print("   ~ Speed does not add beyond amplitude")

    # Overall verdict
    checks_passed = sum([
        ci_excludes_zero,
        phase_only.direction_preserved if phase_only.has_phase_only_model else False,
        regression.speed_adds_value,
    ])
    checks_total = 2 + (1 if phase_only.has_phase_only_model else 0)

    print(f"\n{'=' * 70}")
    print(f"VERDICT: {checks_passed}/{checks_total} controls support speed ≠ amplitude proxy")
    print("=" * 70)


if __name__ == "__main__":
    main()
