#!/usr/bin/env python3
"""
Ablation Study Evaluation: Compare trained variants

This script runs the downstream analysis for each ablation variant and
generates comparison figures and tables.

Key metrics computed:
- Mean speed (expert vs novice) with bootstrap CI
- Effect size d for mean speed
- Reconstruction MSE (val)
- Training epochs to best checkpoint

Usage:
    python scripts/run_ablation_evaluation.py --ablation_dir outputs/ablation --data_dir /path/to/ds001787

Output:
    outputs/ablation/comparison/
        - ablation_comparison_figure.png
        - ablation_summary_table.csv
        - ablation_results.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def find_checkpoint(variant_dir: Path) -> Optional[Path]:
    """Find best checkpoint in variant output directory."""
    # Check common locations
    candidates = [
        variant_dir / "checkpoints" / "best.pt",
        variant_dir / "best.pt",
    ]

    # Also search subdirectories (Hydra creates timestamped dirs)
    for subdir in variant_dir.iterdir():
        if subdir.is_dir():
            candidates.append(subdir / "checkpoints" / "best.pt")

    for path in candidates:
        if path.exists():
            return path

    return None


def load_checkpoint_info(checkpoint_path: Path) -> dict:
    """Load training info from checkpoint."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    info = {
        "checkpoint_path": str(checkpoint_path),
        "epoch": checkpoint.get("epoch", None),
        "best_val_loss": checkpoint.get("best_val_loss", None),
    }

    # Extract config if available
    if "config" in checkpoint:
        cfg = checkpoint["config"]
        if isinstance(cfg, dict):
            training = cfg.get("training", {})
            if isinstance(training, str):
                # Config stored as string, parse it
                import ast
                try:
                    training = ast.literal_eval(training)
                except:
                    training = {}

            info["lambda_contrastive"] = training.get("lambda_contrastive", None)
            info["contrastive_shuffle_labels"] = training.get("contrastive_shuffle_labels", False)

    return info


def compute_latent_speed_analysis(
    checkpoint_path: Path,
    data_dir: Path,
    n_bootstrap: int = 1000,
) -> dict:
    """
    Compute mean latent speed for expert vs novice groups.

    Returns dict with speed statistics and bootstrap CIs.
    """
    from scripts.local_analysis.load_model import (
        load_model_from_checkpoint,
        create_model,
        compute_latent_trajectory,
    )
    from scripts.local_analysis.load_data import load_eeg_from_file, extract_phase_circular
    from scripts.local_analysis import config as cfg

    # Setup config for meditation
    cfg.DATASET = "meditation_bids"
    cfg.DATA_DIR = data_dir

    # Load model
    device = "cpu"  # Use CPU for evaluation
    model_info = load_model_from_checkpoint(checkpoint_path, device)

    # Infer n_channels from checkpoint
    import torch
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    for key in state_dict:
        if "conv" in key and "weight" in key and state_dict[key].dim() == 3:
            in_features = state_dict[key].shape[1]
            if in_features % 3 == 0:
                n_channels = in_features // 3
                break

    model = create_model(n_channels, model_info, device)

    # Get data files
    data_files = cfg.get_data_files_via_config()
    groups = cfg.get_subjects_by_group_unified(data_files)

    # Process subjects
    expert_speeds = []
    novice_speeds = []

    for group_key, subjects in groups.items():
        for file_path, label, group_name, subject_id in tqdm(
            subjects, desc=f"Processing {group_name}", leave=False
        ):
            try:
                # Load and process
                raw_data, sfreq, channel_names = load_eeg_from_file(file_path, verbose=False)

                # Chunk
                chunk_samples = int(5.0 * sfreq)
                n_chunks = raw_data.shape[1] // chunk_samples

                subject_speeds = []
                for i in range(n_chunks):
                    start = i * chunk_samples
                    end = start + chunk_samples
                    chunk_data = raw_data[:, start:end]

                    # Extract phase
                    phase_data = extract_phase_circular(
                        chunk_data, sfreq,
                        include_amplitude=model_info["include_amplitude"],
                        skip_filter=True  # Already filtered
                    )

                    # Get latent trajectory
                    latent = compute_latent_trajectory(model, phase_data, device)

                    # Compute speed
                    diff = np.diff(latent, axis=0)
                    speed = np.linalg.norm(diff, axis=1)
                    subject_speeds.append(np.mean(speed))

                # Store subject mean speed
                mean_speed = np.mean(subject_speeds)
                if group_name.lower() == "expert":
                    expert_speeds.append(mean_speed)
                else:
                    novice_speeds.append(mean_speed)

            except Exception as e:
                print(f"  Error processing {subject_id}: {e}")
                continue

    expert_speeds = np.array(expert_speeds)
    novice_speeds = np.array(novice_speeds)

    # Compute statistics
    expert_mean = np.mean(expert_speeds)
    novice_mean = np.mean(novice_speeds)
    diff = novice_mean - expert_mean

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        ((len(expert_speeds) - 1) * np.var(expert_speeds) +
         (len(novice_speeds) - 1) * np.var(novice_speeds)) /
        (len(expert_speeds) + len(novice_speeds) - 2)
    )
    cohens_d = diff / pooled_std if pooled_std > 0 else 0

    # Bootstrap CI for difference
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        expert_boot = np.random.choice(expert_speeds, len(expert_speeds), replace=True)
        novice_boot = np.random.choice(novice_speeds, len(novice_speeds), replace=True)
        bootstrap_diffs.append(np.mean(novice_boot) - np.mean(expert_boot))

    bootstrap_diffs = np.array(bootstrap_diffs)
    ci_low = np.percentile(bootstrap_diffs, 2.5)
    ci_high = np.percentile(bootstrap_diffs, 97.5)

    # Bootstrap CI for Cohen's d
    bootstrap_ds = []
    for _ in range(n_bootstrap):
        expert_boot = np.random.choice(expert_speeds, len(expert_speeds), replace=True)
        novice_boot = np.random.choice(novice_speeds, len(novice_speeds), replace=True)
        diff_boot = np.mean(novice_boot) - np.mean(expert_boot)
        pooled_std_boot = np.sqrt(
            ((len(expert_boot) - 1) * np.var(expert_boot) +
             (len(novice_boot) - 1) * np.var(novice_boot)) /
            (len(expert_boot) + len(novice_boot) - 2)
        )
        d_boot = diff_boot / pooled_std_boot if pooled_std_boot > 0 else 0
        bootstrap_ds.append(d_boot)

    bootstrap_ds = np.array(bootstrap_ds)
    d_ci_low = np.percentile(bootstrap_ds, 2.5)
    d_ci_high = np.percentile(bootstrap_ds, 97.5)

    return {
        "n_expert": len(expert_speeds),
        "n_novice": len(novice_speeds),
        "expert_mean_speed": float(expert_mean),
        "novice_mean_speed": float(novice_mean),
        "speed_difference": float(diff),
        "speed_diff_ci_low": float(ci_low),
        "speed_diff_ci_high": float(ci_high),
        "cohens_d": float(cohens_d),
        "cohens_d_ci_low": float(d_ci_low),
        "cohens_d_ci_high": float(d_ci_high),
        "expert_speeds": expert_speeds.tolist(),
        "novice_speeds": novice_speeds.tolist(),
    }


def plot_ablation_comparison(results: dict, output_path: Path):
    """Create comparison figure across ablation variants."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    variants = list(results.keys())
    colors = {"full": "#1f77b4", "no_contrastive": "#ff7f0e", "shuffled_contrastive": "#2ca02c"}

    # Panel 1: Speed difference with CI
    ax = axes[0]
    x = np.arange(len(variants))
    for i, variant in enumerate(variants):
        r = results[variant]
        diff = r["speed_difference"]
        ci_low = r["speed_diff_ci_low"]
        ci_high = r["speed_diff_ci_high"]

        ax.bar(i, diff, color=colors.get(variant, "gray"), alpha=0.7, width=0.6)
        ax.errorbar(i, diff, yerr=[[diff - ci_low], [ci_high - diff]],
                   fmt='none', color='black', capsize=5, capthick=2)

    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace("_", "\n") for v in variants], fontsize=9)
    ax.set_ylabel("Speed Difference (novice - expert)")
    ax.set_title("A. Group Speed Difference")

    # Panel 2: Effect size (Cohen's d) with CI
    ax = axes[1]
    for i, variant in enumerate(variants):
        r = results[variant]
        d = r["cohens_d"]
        ci_low = r["cohens_d_ci_low"]
        ci_high = r["cohens_d_ci_high"]

        ax.bar(i, d, color=colors.get(variant, "gray"), alpha=0.7, width=0.6)
        ax.errorbar(i, d, yerr=[[d - ci_low], [ci_high - d]],
                   fmt='none', color='black', capsize=5, capthick=2)

    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace("_", "\n") for v in variants], fontsize=9)
    ax.set_ylabel("Cohen's d")
    ax.set_title("B. Effect Size")

    # Panel 3: Reconstruction loss comparison
    ax = axes[2]
    val_losses = []
    for variant in variants:
        r = results[variant]
        val_losses.append(r.get("best_val_loss", np.nan))

    ax.bar(x, val_losses, color=[colors.get(v, "gray") for v in variants], alpha=0.7, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace("_", "\n") for v in variants], fontsize=9)
    ax.set_ylabel("Best Validation Loss (MSE)")
    ax.set_title("C. Reconstruction Quality")

    plt.suptitle("Ablation Study: Contrastive Loss Analysis", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ablation study variants and generate comparison"
    )
    parser.add_argument(
        "--ablation_dir",
        type=Path,
        required=True,
        help="Directory containing ablation variant outputs",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Path to meditation BIDS dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for comparison results (default: ablation_dir/comparison)",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap iterations",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["full", "no_contrastive", "shuffled_contrastive"],
        help="Variants to evaluate",
    )

    args = parser.parse_args()

    # Setup output directory
    if args.output_dir is None:
        output_dir = args.ablation_dir / "comparison"
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ABLATION STUDY EVALUATION")
    print("=" * 70)
    print(f"Ablation directory: {args.ablation_dir}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Variants: {args.variants}")

    results = {}

    for variant in args.variants:
        print(f"\n{'='*50}")
        print(f"Evaluating: {variant.upper()}")
        print(f"{'='*50}")

        variant_dir = args.ablation_dir / variant

        # Find checkpoint
        checkpoint_path = find_checkpoint(variant_dir)
        if checkpoint_path is None:
            print(f"  WARNING: No checkpoint found for {variant}, skipping")
            continue

        print(f"  Checkpoint: {checkpoint_path}")

        # Load checkpoint info
        ckpt_info = load_checkpoint_info(checkpoint_path)
        print(f"  Best epoch: {ckpt_info['epoch']}")
        print(f"  Best val loss: {ckpt_info['best_val_loss']:.4f}")

        # Compute speed analysis
        print(f"  Computing latent speed analysis...")
        speed_results = compute_latent_speed_analysis(
            checkpoint_path=checkpoint_path,
            data_dir=args.data_dir,
            n_bootstrap=args.n_bootstrap,
        )

        # Combine results
        results[variant] = {**ckpt_info, **speed_results}

        print(f"  Expert mean speed: {speed_results['expert_mean_speed']:.4f}")
        print(f"  Novice mean speed: {speed_results['novice_mean_speed']:.4f}")
        print(f"  Speed difference: {speed_results['speed_difference']:.4f} "
              f"[{speed_results['speed_diff_ci_low']:.4f}, {speed_results['speed_diff_ci_high']:.4f}]")
        print(f"  Cohen's d: {speed_results['cohens_d']:.3f} "
              f"[{speed_results['cohens_d_ci_low']:.3f}, {speed_results['cohens_d_ci_high']:.3f}]")

    if len(results) == 0:
        print("\nERROR: No variants could be evaluated")
        sys.exit(1)

    # Generate outputs
    print(f"\n{'='*70}")
    print("GENERATING OUTPUTS")
    print(f"{'='*70}")

    # Save full results JSON
    json_path = output_dir / "ablation_results.json"
    # Remove numpy arrays for JSON serialization
    results_for_json = {
        k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list) or kk not in ["expert_speeds", "novice_speeds"]}
        for k, v in results.items()
    }
    with open(json_path, "w") as f:
        json.dump(results_for_json, f, indent=2)
    print(f"Saved: {json_path}")

    # Create summary table
    rows = []
    for variant, r in results.items():
        rows.append({
            "variant": variant,
            "epoch": r.get("epoch"),
            "val_loss": r.get("best_val_loss"),
            "lambda_contrastive": r.get("lambda_contrastive"),
            "shuffle_labels": r.get("contrastive_shuffle_labels"),
            "n_expert": r.get("n_expert"),
            "n_novice": r.get("n_novice"),
            "expert_mean_speed": r.get("expert_mean_speed"),
            "novice_mean_speed": r.get("novice_mean_speed"),
            "speed_difference": r.get("speed_difference"),
            "speed_diff_ci_low": r.get("speed_diff_ci_low"),
            "speed_diff_ci_high": r.get("speed_diff_ci_high"),
            "cohens_d": r.get("cohens_d"),
            "cohens_d_ci_low": r.get("cohens_d_ci_low"),
            "cohens_d_ci_high": r.get("cohens_d_ci_high"),
        })

    df = pd.DataFrame(rows)
    csv_path = output_dir / "ablation_summary_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Generate comparison figure
    fig_path = output_dir / "ablation_comparison_figure.png"
    plot_ablation_comparison(results, fig_path)

    # Print summary table
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY TABLE")
    print("=" * 70)
    print(df.to_string(index=False))

    # Print interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if "full" in results and "no_contrastive" in results:
        full_d = results["full"]["cohens_d"]
        no_contr_d = results["no_contrastive"]["cohens_d"]

        if no_contr_d > 0 and abs(no_contr_d) > abs(full_d) * 0.5:
            print("✓ NO_CONTRASTIVE shows effect in same direction")
            print("  → Group effect is NOT solely induced by contrastive loss")
        else:
            print("⚠ NO_CONTRASTIVE shows weak/reversed effect")
            print("  → Group effect may be partially induced by contrastive loss")

    if "full" in results and "shuffled_contrastive" in results:
        full_d = results["full"]["cohens_d"]
        shuf_d = results["shuffled_contrastive"]["cohens_d"]

        if abs(shuf_d) < abs(full_d) * 0.7:
            print("✓ SHUFFLED_CONTRASTIVE shows weaker effect than FULL")
            print("  → Correct group structure in contrastive improves discrimination")
        else:
            print("⚠ SHUFFLED_CONTRASTIVE shows similar effect to FULL")
            print("  → Group structure in contrastive may not matter much")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
