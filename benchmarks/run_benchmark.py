#!/usr/bin/env python3
"""
Benchmark Runner: Evaluate methods across datasets.

Usage:
    # Quick test on a single dataset:
    python -m benchmarks.run_benchmark --dataset meditation_bids --method flow_metrics --quick

    # Full run:
    python -m benchmarks.run_benchmark --dataset greek_resting --method flow_metrics

    # Show available datasets and methods:
    python -m benchmarks.run_benchmark --list

    # All combinations:
    python -m benchmarks.run_benchmark --all --quick

Output is saved to results/benchmarks/{dataset}/{method}/{timestamp}/
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="EEG Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", type=str, help="Dataset name (e.g., greek_resting)")
    parser.add_argument("--method", type=str, help="Method name (e.g., flow_metrics)")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 3 subjects per group")
    parser.add_argument("--list", action="store_true", help="List available datasets and methods")
    parser.add_argument("--all", action="store_true", help="Run all dataset x method combinations")
    parser.add_argument("--checkpoint", type=str, help="Override checkpoint path")
    parser.add_argument("--data-root", type=str, help="Override data root directory")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("benchmark")

    # Import after path setup
    from benchmarks.paths import get_paths
    from benchmarks.registry import load_dataset, list_methods, run_method
    import benchmarks.methods  # noqa: F401 — trigger registration
    from benchmarks.evaluation import compute_discriminability

    # Resolve paths
    paths = get_paths(data_root=args.data_root)
    logger.info(f"Environment: {paths.environment}")
    logger.info(f"Data root: {paths.data_root}")
    logger.info(f"Device: {paths.device}")

    # Validate
    warnings = paths.validate()
    for w in warnings:
        logger.warning(w)

    # List mode
    if args.list:
        print("\n=== Available Datasets ===")
        for ds in paths.available_datasets():
            print(f"  {ds}")

        print("\n=== Available Methods ===")
        for m in list_methods():
            print(f"  {m}")

        print(f"\n=== Environment: {paths.environment} ===")
        print(f"  Data root: {paths.data_root}")
        print(f"  Device: {paths.device}")
        return

    # Determine what to run
    if args.all:
        combinations = [
            (ds, method)
            for ds in paths.available_datasets()
            for method in list_methods()
        ]
    elif args.dataset and args.method:
        combinations = [(args.dataset, args.method)]
    else:
        parser.error("Specify --dataset + --method, or --all, or --list")
        return

    # Run benchmarks
    for dataset_name, method_name in combinations:
        logger.info(f"\n{'='*60}")
        logger.info(f"BENCHMARK: {dataset_name} x {method_name}")
        logger.info(f"{'='*60}")

        try:
            # Load data
            data = load_dataset(dataset_name, paths=paths, quick=args.quick)
            logger.info(data.summary())

            # Run method
            kwargs = {}
            if args.checkpoint:
                kwargs["checkpoint"] = args.checkpoint

            result = run_method(method_name, data, paths=paths, **kwargs)

            # Evaluate
            disc = compute_discriminability(result)

            # Report
            _print_report(result, disc, logger)

            # Save
            output_dir = _get_output_dir(paths, dataset_name, method_name, args.output_dir)
            _save_results(output_dir, result, disc, args)

            logger.info(f"Results saved to: {output_dir}")

        except Exception as e:
            logger.error(f"FAILED: {dataset_name} x {method_name}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            continue


def _print_report(result, disc, logger):
    """Print human-readable benchmark report."""
    logger.info(f"\n--- {result.dataset_name} x {result.method_name} ---")
    logger.info(f"Subjects: {result.metadata.get('n_subjects', '?')}")

    summary = disc["summary"]
    logger.info(f"Significant features (p<0.05): {summary['n_significant_p05']}/{summary['n_features']}")
    logger.info(f"Best feature: {summary['best_feature']} (eta2={summary['max_eta_squared']:.4f})")

    for name, metrics in disc["features"].items():
        eta = metrics["eta_squared"]
        p = metrics["p"]
        effect = metrics["effect_size"]
        star = "*" if p < 0.05 else " "
        logger.info(f"  {star} {name:25s}: eta2={eta:.4f}  F={metrics['F']:.2f}  p={p:.4f}  [{effect}]")


def _get_output_dir(paths, dataset_name, method_name, override=None):
    """Create timestamped output directory."""
    if override:
        output_dir = Path(override)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = paths.output_root / dataset_name / method_name / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def _save_results(output_dir, result, disc, args):
    """Save benchmark results to disk."""
    # Parameters
    params = {
        "dataset": result.dataset_name,
        "method": result.method_name,
        "quick": args.quick,
        "timestamp": datetime.now().isoformat(),
        "metadata": result.metadata,
    }
    with open(output_dir / "parameters.json", "w") as f:
        json.dump(params, f, indent=2, cls=_NumpyEncoder)

    # Discriminability results
    with open(output_dir / "discriminability.json", "w") as f:
        json.dump(disc, f, indent=2, cls=_NumpyEncoder)

    # Raw features
    np.savez(
        output_dir / "features.npz",
        features=result.features,
        labels=np.array(result.labels),
        subject_ids=np.array(result.subject_ids),
    )

    # Summary table (CSV)
    try:
        import pandas as pd

        rows = []
        for name, metrics in disc["features"].items():
            rows.append({
                "feature": name,
                "F": metrics["F"],
                "p": metrics["p"],
                "eta_squared": metrics["eta_squared"],
                "effect_size": metrics["effect_size"],
            })
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / "summary.csv", index=False)
    except ImportError:
        pass  # pandas optional for CSV output


if __name__ == "__main__":
    main()
