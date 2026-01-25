#!/usr/bin/env python3
"""
REVE Foundation Model Comparison Analysis

Uses the pretrained REVE EEG foundation model as a comparison encoder to generate
latent trajectories, then applies the same flow/density analysis as full_dataset_analysis.py.

This tests whether dynamical signatures (flow geometry, density differences) are:
- Representation-dependent (specific to our trained phase+amplitude model)
- Representation-invariant (also appear in a pretrained foundation model's embeddings)

If REVE-based trajectories reproduce group-level dynamical signatures, that's a big
credibility boost for the dynamical microscope framework.

Usage:
    python reve_analysis.py                    # Use PCA embedding (default)
    python reve_analysis.py --no-show          # Run without displaying plots
    python reve_analysis.py --n-subjects 20    # Limit subjects per group
    python reve_analysis.py --quick            # Quick test mode

Key differences from full_dataset_analysis.py:
- Uses raw EEG instead of phase extraction
- REVE outputs patch-level embeddings (coarser time resolution)
- Trajectory shape: (n_patches, hidden_dim) instead of (n_timepoints, hidden_dim)

Reference: https://brain-bzh.github.io/reve/
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json
from typing import Optional

import numpy as np
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_DIR, OUTPUT_DIR, DEVICE, DATASET,
    FILTER_LOW, FILTER_HIGH, CHUNK_DURATION, SFREQ,
    ensure_output_dir, get_data_files_via_config, get_subjects_by_group_unified,
)
from load_data import load_eeg_from_file
from load_reve_model import REVEEncoder, check_reve_access

# Import analysis functions from full_dataset_analysis
from full_dataset_analysis import (
    SubjectData, FlowMetrics, BootstrapResult,
    PooledEmbedder,
    compute_flow_metrics,
    bootstrap_flow_metrics,
    bootstrap_density_difference,
    bootstrap_radial_profiles,
    bootstrap_effect_size,
    compute_cross_embedding_robustness,
    compute_group_flow_field,
    compute_flow_statistics,
    plot_bootstrap_metrics_comparison,
    plot_density_difference_with_ci,
    plot_radial_profiles,
    plot_effect_sizes,
    plot_cross_embedding_robustness,
    plot_group_flow_fields,
    plot_flow_difference,
    get_group_config,
    get_reference_group,
    get_comparison_groups,
    get_all_groups,
    create_timestamped_output_dir,
    save_parameters,
    print_summary_table,
)

GROUP_CONFIG = get_group_config()


def load_all_subjects_reve(
    encoder: REVEEncoder,
    groups: dict,
    n_subjects_per_group: Optional[int],
    chunk_duration: float = 5.0,
    n_chunks: int = 10,
) -> dict[str, list[SubjectData]]:
    """
    Load all subjects' latent trajectories using REVE encoder.

    Key difference from original: Uses raw EEG, not phase extraction.
    REVE handles its own preprocessing (resampling, normalization).

    Args:
        encoder: REVEEncoder instance
        groups: Dict from get_subjects_by_group_unified()
        n_subjects_per_group: Max subjects per group (None = all)
        chunk_duration: Duration of each chunk in seconds
        n_chunks: Max chunks per subject to use

    Returns:
        Dict mapping group name -> list of SubjectData
    """
    group_keys = GROUP_CONFIG["keys"]
    subject_data = {}

    for group_key in group_keys:
        subjects = groups.get(group_key, [])
        if not subjects:
            continue

        # Get display name
        group_name = subjects[0][2] if len(subjects[0]) > 2 else group_key.upper()
        display_name = group_name.upper() if DATASET != "meditation_bids" else group_name.capitalize()

        subject_data[display_name] = []

        max_subjects = n_subjects_per_group if n_subjects_per_group else len(subjects)

        print(f"\nLoading {display_name} subjects with REVE (max {max_subjects})...")
        subjects_processed = 0

        for entry in tqdm(subjects, desc=display_name):
            if subjects_processed >= max_subjects:
                break

            file_path, label, condition, subject_id = entry

            try:
                # Load raw EEG (not phase!)
                raw_data, sfreq, channel_names = load_eeg_from_file(file_path, verbose=False)

                # Compute chunk samples
                chunk_samples = int(chunk_duration * sfreq)
                n_samples = raw_data.shape[1]

                # Split into chunks and compute REVE trajectories
                trajectories = []
                for i in range(min(n_chunks, n_samples // chunk_samples)):
                    start = i * chunk_samples
                    end = start + chunk_samples
                    chunk_data = raw_data[:, start:end]

                    try:
                        # Get REVE trajectory for this chunk
                        traj = encoder.compute_trajectory(chunk_data, channel_names, sfreq)
                        trajectories.append(traj)
                    except Exception as e:
                        # Skip chunks that fail (e.g., no valid channels)
                        if i == 0:
                            raise  # Re-raise if first chunk fails
                        continue

                if len(trajectories) == 0:
                    print(f"  Warning: No trajectories for {subject_id}")
                    continue

                # Concatenate chunk trajectories
                trajectory = np.concatenate(trajectories, axis=0)

                subject_data[display_name].append(SubjectData(
                    subject_id=subject_id,
                    group=group_key,
                    label=label,
                    trajectory=trajectory,
                ))
                subjects_processed += 1

            except Exception as e:
                print(f"  Warning: Failed to process {subject_id}: {e}")
                continue

        print(f"  Loaded {subjects_processed} {display_name} subjects")

    return subject_data


def run_reve_analysis(
    subject_data: dict[str, list[SubjectData]],
    output_dir: Path,
    methods: list[str],
    n_bootstrap: int,
    tau: int,
    delay_dim: int,
    show_plot: bool,
):
    """
    Run full statistical analysis using REVE trajectories.

    This is a simplified version of run_full_analysis that focuses on PCA
    since REVE trajectories have different structure than our model's.
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "encoder": "REVE",
        "n_bootstrap": n_bootstrap,
        "methods": methods,
        "n_subjects": {g: len(s) for g, s in subject_data.items()},
    }

    # Get all trajectories for pooled fitting
    all_subjects = []
    for group_subjects in subject_data.values():
        all_subjects.extend(group_subjects)
    all_trajectories = [s.trajectory for s in all_subjects]

    # Print trajectory stats
    traj_lengths = [t.shape[0] for t in all_trajectories]
    print(f"\nTrajectory statistics:")
    print(f"  Mean length: {np.mean(traj_lengths):.1f} patches")
    print(f"  Min/Max: {min(traj_lengths)}/{max(traj_lengths)}")
    print(f"  Hidden dim: {all_trajectories[0].shape[1]}")

    for method in methods:
        print(f"\n{'='*80}")
        print(f"ANALYZING: {method.upper()} (REVE embeddings)")
        print(f"{'='*80}")

        # Fit pooled embedder
        print(f"\nFitting {method} embedder on pooled REVE trajectories...")
        embedder = PooledEmbedder(method=method, tau=tau, delay_dim=delay_dim)
        embedder.fit(all_trajectories)
        embedding_name = f"REVE_{embedder.get_method_name()}"

        # 1. Bootstrap flow metrics for each group
        print(f"\nBootstrapping flow metrics ({n_bootstrap} iterations)...")
        group_bootstrap_results = {}
        for group, subjects in subject_data.items():
            if len(subjects) < 3:
                print(f"  Skipping {group} (only {len(subjects)} subjects)")
                continue
            print(f"  {group}...", end=" ", flush=True)
            group_bootstrap_results[group] = bootstrap_flow_metrics(subjects, embedder, n_bootstrap)
            print("done")

        # Plot bootstrap metrics comparison
        if len(group_bootstrap_results) > 1:
            plot_bootstrap_metrics_comparison(group_bootstrap_results, output_dir, embedding_name, show_plot)

        # 2. Density difference with CI
        print(f"\nBootstrapping density differences...")
        ref_group = get_reference_group()
        ref_subjects = subject_data.get(ref_group, [])
        for comp_group in get_comparison_groups():
            comp_subjects = subject_data.get(comp_group, [])
            if len(ref_subjects) < 3 or len(comp_subjects) < 3:
                continue

            print(f"  {comp_group} - {ref_group}...", end=" ", flush=True)
            mean_diff, ci_low, ci_high = bootstrap_density_difference(
                ref_subjects, comp_subjects, embedder, n_bootstrap
            )
            print("done")

            plot_density_difference_with_ci(
                mean_diff, ci_low, ci_high, embedder.bounds, comp_group, output_dir, embedding_name, show_plot
            )

        # 3. Radial profiles
        print(f"\nBootstrapping radial profiles...")
        group_profiles = {}
        for group, subjects in subject_data.items():
            if len(subjects) < 3:
                continue
            print(f"  {group}...", end=" ", flush=True)
            bin_centers, density_ci, speed_ci = bootstrap_radial_profiles(subjects, embedder, n_bootstrap)
            group_profiles[group] = (bin_centers, density_ci, speed_ci)
            print("done")

        if len(group_profiles) > 1:
            plot_radial_profiles(group_profiles, output_dir, embedding_name, show_plot)

        # 4. Group flow fields
        print(f"\nComputing group flow fields...")
        flow_data = plot_group_flow_fields(subject_data, embedder, output_dir, embedding_name, show_plot=show_plot)

        # 5. Flow field differences
        print(f"\nComputing flow field differences...")
        plot_flow_difference(subject_data, embedder, output_dir, embedding_name, show_plot=show_plot)

        # Compute flow statistics for results
        flow_stats = {}
        if flow_data:
            for group, (X, Y, flow_x, flow_y, counts) in flow_data.items():
                flow_stats[group] = compute_flow_statistics(X, Y, flow_x, flow_y, counts)

        # Store results
        results[method] = {
            "bootstrap_metrics": {
                g: {m: {"mean": r.mean, "ci_low": r.ci_low, "ci_high": r.ci_high}
                    for m, r in metrics.items()}
                for g, metrics in group_bootstrap_results.items()
            },
            "flow_statistics": flow_stats,
        }

    # 6. Effect sizes (using PCA)
    print(f"\n{'='*80}")
    print("COMPUTING EFFECT SIZES")
    print(f"{'='*80}")

    pca_embedder = PooledEmbedder(method="pca", tau=tau, delay_dim=delay_dim)
    pca_embedder.fit(all_trajectories)

    # Get metrics for each subject
    group_metric_values = {g: {} for g in subject_data.keys()}
    metric_names = ["mean_speed", "speed_cv", "n_dwell_episodes", "occupancy_entropy", "path_tortuosity", "explored_variance"]

    for group, subjects in subject_data.items():
        for metric in metric_names:
            group_metric_values[group][metric] = []
        for subject in subjects:
            embedded = pca_embedder.transform(subject.trajectory)
            metrics = compute_flow_metrics(embedded)
            for metric in metric_names:
                group_metric_values[group][metric].append(getattr(metrics, metric))

    # Compute effect sizes
    effect_sizes = {}
    ref_group = get_reference_group()
    ref_values = group_metric_values.get(ref_group, {})

    for comp_group in get_comparison_groups():
        comp_values = group_metric_values.get(comp_group, {})
        if not comp_values or not ref_values:
            continue

        comparison = f"{ref_group} vs {comp_group}"
        effect_sizes[comparison] = {}

        for metric in metric_names:
            if metric not in ref_values or metric not in comp_values:
                continue
            if len(ref_values[metric]) < 3 or len(comp_values[metric]) < 3:
                continue

            print(f"  {comparison} - {metric}...", end=" ", flush=True)
            effect_sizes[comparison][metric] = bootstrap_effect_size(
                np.array(ref_values[metric]),
                np.array(comp_values[metric]),
                n_bootstrap
            )
            print(f"d = {effect_sizes[comparison][metric].mean:.3f}")

    if effect_sizes:
        plot_effect_sizes(effect_sizes, output_dir, show_plot)

    # 7. Cross-embedding robustness (if multiple methods)
    if len(methods) > 1:
        print(f"\n{'='*80}")
        print("COMPUTING CROSS-EMBEDDING ROBUSTNESS")
        print(f"{'='*80}")

        robustness = compute_cross_embedding_robustness(all_subjects, methods, tau, delay_dim)
        plot_cross_embedding_robustness(robustness, output_dir, show_plot)

        results["cross_embedding_robustness"] = {
            metric: {"mean_correlation": data["mean_off_diagonal"]}
            for metric, data in robustness["correlations"].items()
        }

    # Store effect sizes
    results["effect_sizes"] = {
        comparison: {m: {"mean": r.mean, "ci_low": r.ci_low, "ci_high": r.ci_high}
                     for m, r in metrics.items()}
        for comparison, metrics in effect_sizes.items()
    }

    # Save results to JSON
    results_path = output_dir / "reve_analysis_results.json"

    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            return convert_numpy(obj)

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="REVE foundation model comparison analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n-subjects", type=int, default=None,
                        help="Max subjects per group (default: all)")
    parser.add_argument("--n-chunks", type=int, default=10,
                        help="Chunks per subject (default: 10)")
    parser.add_argument("--n-bootstrap", type=int, default=500,
                        help="Bootstrap iterations (default: 500)")
    parser.add_argument("--embedding", type=str, default="pca",
                        choices=["pca", "tpca", "delay", "fast"],
                        help="Embedding method (default: pca)")
    parser.add_argument("--tau", type=int, default=5,
                        help="Time lag for tPCA and delay embedding (default: 5)")
    parser.add_argument("--delay-dim", type=int, default=3,
                        help="Delay embedding dimension (default: 3)")
    parser.add_argument("--chunk-duration", type=float, default=5.0,
                        help="Chunk duration in seconds (default: 5.0)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test mode (100 bootstrap, 5 subjects)")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't display plots interactively")
    args = parser.parse_args()

    # Set matplotlib backend
    if args.no_show:
        import matplotlib
        matplotlib.use('Agg')
        print("Non-interactive mode: plots will be saved but not displayed")

    # Quick mode overrides
    if args.quick:
        args.n_bootstrap = 100
        args.n_subjects = 5
        print("QUICK MODE: 100 bootstrap iterations, 5 subjects per group")

    print(f"\n{'='*80}")
    print(f"REVE FOUNDATION MODEL COMPARISON ANALYSIS")
    print(f"{'='*80}")
    print(f"Dataset: {DATASET}")
    print(f"Using pretrained REVE encoder (brain-bzh/reve-base)")

    # Check REVE access first
    print("\nChecking HuggingFace access to REVE model...")
    if not check_reve_access():
        return 1
    print("Access confirmed!")

    # Create timestamped output directory
    base_output_dir = ensure_output_dir()
    output_dir = create_timestamped_output_dir(base_output_dir, f"reve_analysis_{DATASET}")
    print(f"Output directory: {output_dir}")

    # Initialize REVE encoder
    print("\nInitializing REVE encoder...")
    try:
        encoder = REVEEncoder(device=DEVICE)
    except Exception as e:
        print(f"ERROR: Failed to initialize REVE encoder: {e}")
        print("\nNote: REVE requires HuggingFace authentication.")
        print("Run: huggingface-cli login")
        return 1

    # Get subjects
    data_files = get_data_files_via_config(None)
    groups = get_subjects_by_group_unified(data_files)

    print(f"\nDataset overview ({DATASET}):")
    for key in GROUP_CONFIG["keys"]:
        if groups.get(key):
            print(f"  {key}: {len(groups[key])} subjects")

    if not any(groups.values()):
        print("ERROR: No data files found. Check your data paths and dataset selection.")
        return 1

    # Load all subject data using REVE
    subject_data = load_all_subjects_reve(
        encoder, groups,
        args.n_subjects, args.chunk_duration, args.n_chunks
    )

    # Determine methods
    if args.embedding == "fast":
        methods = ["pca", "tpca", "delay"]
    else:
        methods = [args.embedding]

    print(f"\nUsing embedding methods: {methods}")

    # Save parameters
    save_parameters(output_dir, {
        "dataset": DATASET,
        "encoder": "REVE (brain-bzh/reve-base)",
        "n_subjects": args.n_subjects,
        "n_chunks": args.n_chunks,
        "n_bootstrap": args.n_bootstrap,
        "embedding": args.embedding,
        "methods": methods,
        "tau": args.tau,
        "delay_dim": args.delay_dim,
        "chunk_duration": args.chunk_duration,
        "quick_mode": args.quick,
        "device": DEVICE,
    })

    # Run analysis
    results = run_reve_analysis(
        subject_data, output_dir, methods,
        args.n_bootstrap, args.tau, args.delay_dim,
        not args.no_show
    )

    # Print summary
    print_summary_table(results, output_dir)

    print(f"\n{'='*80}")
    print("REVE ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"All outputs saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
