#!/usr/bin/env python3
"""
Ablation Study Runner for Contrastive Loss Analysis

This script runs the ablation study to address reviewer concern that
expert/novice effects might be induced by the contrastive objective.

Variants:
- FULL: Baseline with contrastive loss (lambda_contrastive=0.1)
- NO_CONTRASTIVE: Same as FULL but lambda_contrastive=0.0
- SHUFFLED_CONTRASTIVE: Same as FULL but with shuffled labels for contrastive pairs

Usage (RunPod):
    # Run all variants
    python scripts/run_ablation_study.py --data_dir /workspace/ds001787 --seed 42

    # Run specific variants
    python scripts/run_ablation_study.py --variants full no_contrastive --data_dir /workspace/ds001787

    # Dry run (print commands only)
    python scripts/run_ablation_study.py --dry_run --data_dir /workspace/ds001787

Output:
    outputs/ablation/<variant>/checkpoints/best.pt
    outputs/ablation/<variant>/training_summary.json
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


VARIANTS = {
    "full": {
        "config": "experiment/ablation_full",
        "description": "Full model with contrastive loss (baseline)",
    },
    "no_contrastive": {
        "config": "experiment/ablation_no_contrastive",
        "description": "No contrastive loss (lambda_contrastive=0.0)",
    },
    "shuffled_contrastive": {
        "config": "experiment/ablation_shuffled_contrastive",
        "description": "Contrastive with shuffled labels",
    },
}


def build_training_command(
    variant: str,
    data_dir: str,
    output_dir: str,
    seed: int,
    wandb_enabled: bool = True,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build the training command for a variant."""
    config = VARIANTS[variant]["config"]

    cmd = [
        sys.executable, "-m", "eeg_biomarkers.training.train",
        f"--config-name={config}",
        # Use + prefix to add/override keys that may not exist in struct
        f"+paths.data_dir={data_dir}",
        f"experiment.seed={seed}",
        f"logging.wandb.enabled={str(wandb_enabled).lower()}",
        # Override output directory to organize by variant
        f"hydra.run.dir={output_dir}/{variant}",
    ]

    if extra_args:
        cmd.extend(extra_args)

    return cmd


def run_variant(
    variant: str,
    data_dir: str,
    output_dir: str,
    seed: int,
    wandb_enabled: bool,
    dry_run: bool,
    extra_args: list[str] | None = None,
) -> dict:
    """Run training for a single variant."""
    print(f"\n{'='*70}")
    print(f"VARIANT: {variant.upper()}")
    print(f"Description: {VARIANTS[variant]['description']}")
    print(f"{'='*70}")

    cmd = build_training_command(
        variant=variant,
        data_dir=data_dir,
        output_dir=output_dir,
        seed=seed,
        wandb_enabled=wandb_enabled,
        extra_args=extra_args,
    )

    print(f"\nCommand:\n  {' '.join(cmd)}\n")

    if dry_run:
        print("[DRY RUN] Skipping execution")
        return {"variant": variant, "status": "dry_run", "command": " ".join(cmd)}

    # Run training
    start_time = datetime.now()
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Let output stream to console
        )
        status = "success"
        error = None
    except subprocess.CalledProcessError as e:
        status = "failed"
        error = str(e)
        print(f"\nERROR: Training failed for {variant}: {error}")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    return {
        "variant": variant,
        "status": status,
        "duration_seconds": duration,
        "command": " ".join(cmd),
        "error": error,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation study for contrastive loss analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all variants on RunPod
    python scripts/run_ablation_study.py --data_dir /workspace/ds001787

    # Run specific variants
    python scripts/run_ablation_study.py --variants full no_contrastive --data_dir /workspace/ds001787

    # Dry run to see commands
    python scripts/run_ablation_study.py --dry_run --data_dir /workspace/ds001787
        """,
    )

    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANTS.keys()),
        choices=list(VARIANTS.keys()),
        help=f"Variants to run (default: all). Choices: {list(VARIANTS.keys())}",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to meditation BIDS dataset (ds001787)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/ablation",
        help="Base output directory for all variants",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (same for all variants)",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--extra_args",
        nargs="*",
        help="Extra Hydra overrides to pass to all variants",
    )

    args = parser.parse_args()

    # Verify data directory
    data_dir = Path(args.data_dir)
    if not args.dry_run and not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ABLATION STUDY: Contrastive Loss Analysis")
    print("=" * 70)
    print(f"\nVariants to run: {args.variants}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Seed: {args.seed}")
    print(f"W&B enabled: {not args.no_wandb}")
    print(f"Dry run: {args.dry_run}")

    # Run each variant
    results = []
    for variant in args.variants:
        result = run_variant(
            variant=variant,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            seed=args.seed,
            wandb_enabled=not args.no_wandb,
            dry_run=args.dry_run,
            extra_args=args.extra_args,
        )
        results.append(result)

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "seed": args.seed,
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "results": results,
    }

    summary_path = output_dir / "ablation_run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    # Print final summary
    print("\n" + "=" * 70)
    print("ABLATION STUDY SUMMARY")
    print("=" * 70)
    for r in results:
        status_icon = "✓" if r["status"] == "success" else "✗" if r["status"] == "failed" else "○"
        duration = f"{r.get('duration_seconds', 0):.0f}s" if "duration_seconds" in r else "N/A"
        print(f"  {status_icon} {r['variant']:25s} [{r['status']:8s}] {duration}")

    # Check for failures
    failed = [r for r in results if r["status"] == "failed"]
    if failed:
        print(f"\n{len(failed)} variant(s) failed. Check logs for details.")
        sys.exit(1)

    print("\nNext steps:")
    print("  1. Run downstream analysis for each variant:")
    print(f"     python scripts/run_ablation_evaluation.py --ablation_dir {args.output_dir}")
    print("  2. Compare results across variants")


if __name__ == "__main__":
    main()
