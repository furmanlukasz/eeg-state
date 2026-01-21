#!/usr/bin/env python3
"""
Compare Groups in Latent Dynamics

Side-by-side comparison of recurrence matrices and RQA features
between different groups (HC, MCI, AD).

Usage:
    python compare_hc_mci.py                       # Default: HC vs MCI (3 subjects each)
    python compare_hc_mci.py --conditions HID MCI  # Compare HC vs MCI
    python compare_hc_mci.py --conditions HID AD   # Compare HC vs AD
    python compare_hc_mci.py --conditions HID MCI AD  # Compare all three groups
    python compare_hc_mci.py --n-subjects 5        # Compare 5 unique subjects per group
    python compare_hc_mci.py --n-chunks 3          # Use 3 chunks per subject (averaged)
    python compare_hc_mci.py --chunk 2             # Use specific chunk index
    python compare_hc_mci.py --list-subjects       # List available subjects
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CHECKPOINT_PATH, DATA_DIR, OUTPUT_DIR, DEVICE,
    FILTER_LOW, FILTER_HIGH, CHUNK_DURATION, SFREQ,
    RR_TARGETS, THEILER_WINDOW, ensure_output_dir, get_fif_files,
    get_subjects_by_group, get_subject_id, get_label_name
)
from load_model import load_model_from_checkpoint, create_model, compute_latent_trajectory
from load_data import load_and_preprocess_fif
from plot_recurrence import compute_angular_distance_matrix, compute_recurrence_matrix, compute_rqa_stats

# Color and name mapping for groups
GROUP_COLORS = {
    0: "blue",   # HC
    1: "orange", # MCI
    2: "red",    # AD
}

GROUP_NAMES = {
    0: "HC",
    1: "MCI",
    2: "AD",
}


def create_timestamped_output_dir(base_dir: Path, script_name: str) -> Path:
    """Create a timestamped output directory for versioned results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / f"{script_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_parameters(output_dir: Path, params: dict):
    """Save parameters to a JSON file for reproducibility."""
    params_path = output_dir / "parameters.json"

    # Convert Path objects to strings
    serializable_params = {}
    for k, v in params.items():
        if isinstance(v, Path):
            serializable_params[k] = str(v)
        else:
            serializable_params[k] = v

    serializable_params["timestamp"] = datetime.now().isoformat()

    with open(params_path, 'w') as f:
        json.dump(serializable_params, f, indent=2)
    print(f"Parameters saved to: {params_path}")


def analyze_subject(
    model,
    fif_path: Path,
    model_info: dict,
    rr_target: float = 0.02,
    theiler: int = 50,
    chunk_idx: int | None = None,
    n_chunks: int = 1,
):
    """
    Analyze a single subject and return metrics.

    Args:
        chunk_idx: Specific chunk to use (None = use first n_chunks and average)
        n_chunks: Number of chunks to average (if chunk_idx is None)
    """
    data = load_and_preprocess_fif(
        fif_path, FILTER_LOW, FILTER_HIGH, CHUNK_DURATION,
        include_amplitude=model_info["include_amplitude"],
        verbose=False,
    )

    available_chunks = len(data["chunks"])

    # Determine which chunks to process
    if chunk_idx is not None:
        if chunk_idx >= available_chunks:
            print(f"    Warning: chunk {chunk_idx} not available, using 0")
            chunk_idx = 0
        chunks_to_use = [chunk_idx]
    else:
        chunks_to_use = list(range(min(n_chunks, available_chunks)))

    # Process each chunk and collect RQA stats
    all_stats = []
    last_latent = None
    last_R = None

    for cidx in chunks_to_use:
        phase_data = data["chunks"][cidx]
        latent = compute_latent_trajectory(model, phase_data, DEVICE)
        distance_matrix = compute_angular_distance_matrix(latent)
        R, eps = compute_recurrence_matrix(distance_matrix, rr_target, theiler)
        stats = compute_rqa_stats(R)
        all_stats.append(stats)
        last_latent = latent
        last_R = R

    # Average stats across chunks (all RQA features)
    feature_keys = all_stats[0].keys() if all_stats else []
    avg_stats = {key: np.mean([s[key] for s in all_stats]) for key in feature_keys}

    return {
        "subject_id": data["subject_id"],
        "latent": last_latent,  # Keep last for visualization
        "recurrence_matrix": last_R,
        "rqa_stats": avg_stats,
        "n_chunks_used": len(chunks_to_use),
        "n_chunks_available": available_chunks,
    }


def plot_group_comparison(
    group_results: dict,
    output_dir: Path,
    rr_target: float,
    show_plot: bool = True,
):
    """Create comparison plot between groups."""
    # Determine layout
    n_groups = len(group_results)
    max_subjects = max(len(results) for results in group_results.values())

    fig, axes = plt.subplots(n_groups, max_subjects, figsize=(5 * max_subjects, 5 * n_groups))

    # Ensure axes is 2D
    if n_groups == 1:
        axes = axes.reshape(1, -1)
    if max_subjects == 1:
        axes = axes.reshape(-1, 1)

    # Get time extent from first result
    first_results = list(group_results.values())[0]
    time_extent = first_results[0]["latent"].shape[0] / SFREQ

    # Plot each group
    for row_idx, (group_key, results) in enumerate(group_results.items()):
        label = results[0]["rqa_stats"]  # Just to check structure
        group_name = GROUP_NAMES.get(int(group_key) if group_key.isdigit() else {"hc": 0, "mci": 1, "ad": 2}.get(group_key, 0), group_key.upper())

        for col_idx, res in enumerate(results):
            axes[row_idx, col_idx].imshow(
                res["recurrence_matrix"], cmap="binary", origin="lower",
                extent=[0, time_extent, 0, time_extent]
            )
            axes[row_idx, col_idx].set_title(
                f"{group_name}: {res['subject_id']}\nDET={res['rqa_stats']['DET']:.2f}",
                fontsize=11
            )
            axes[row_idx, col_idx].set_xlabel("Time (s)")
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel("Time (s)")

        # Hide unused axes
        for col_idx in range(len(results), max_subjects):
            axes[row_idx, col_idx].axis("off")

    # Compute group stats
    group_stats = {}
    for group_key, results in group_results.items():
        group_name = GROUP_NAMES.get(int(group_key) if group_key.isdigit() else {"hc": 0, "mci": 1, "ad": 2}.get(group_key, 0), group_key.upper())
        det_mean = np.mean([r["rqa_stats"]["DET"] for r in results])
        group_stats[group_name] = det_mean

    stats_str = " | ".join([f"{name} avg DET={det:.3f}" for name, det in group_stats.items()])
    group_names_str = " vs ".join(group_stats.keys())

    plt.suptitle(
        f"{group_names_str} Recurrence Matrices (RR={rr_target*100:.0f}%)\n{stats_str}",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    save_path = output_dir / f"group_comparison_rr{int(rr_target*100)}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_rqa_violin(group_results: dict, output_dir: Path, show_plot: bool = True):
    """
    Violin plots comparing all RQA features between groups.

    Expected MCI vs HC trends (brain criticality framework):
    =========================================================
    MCI shows shift toward SUPERCRITICALITY (more rigid dynamics):

    Feature   | MCI vs HC | Interpretation
    ----------|-----------|-------------------------------------------
    DET       | ↑ higher  | More deterministic/predictable dynamics
    LAM       | ↑ higher  | More laminar states (system gets "stuck")
    TT        | ↑ higher  | Longer trapping time in states
    L_mean    | ↑ higher  | Longer diagonal lines = more predictable
    L_max     | ↑ higher  | Longer max recurrence = attractor collapse
    ENTR      | ↓ lower   | Reduced complexity in dynamics
    DIV       | ↓ lower   | Lower divergence = less chaotic

    Healthy brains operate at "edge of chaos" (critical point).
    MCI/AD shifts toward overcoupled, rigid dynamics.
    """

    # Get all feature keys from first result
    first_results = list(group_results.values())[0]
    if not first_results:
        print("No results to plot")
        return

    all_features = list(first_results[0]["rqa_stats"].keys())

    # Features to plot (exclude RR as it's controlled)
    features_to_plot = ["DET", "LAM", "TT", "L_mean", "ENTR", "DIV"]
    features_to_plot = [f for f in features_to_plot if f in all_features]

    n_features = len(features_to_plot)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

    # Prepare data for each feature
    group_keys = list(group_results.keys())
    group_names = [GROUP_NAMES.get({"hc": 0, "mci": 1, "ad": 2}.get(k.lower(), 0), k.upper()) for k in group_keys]
    colors = [GROUP_COLORS.get({"hc": 0, "mci": 1, "ad": 2}.get(k.lower(), 0), "gray") for k in group_keys]

    for idx, feature in enumerate(features_to_plot):
        ax = axes[idx]

        # Collect data for violin plot
        data_per_group = []
        for group_key in group_keys:
            results = group_results[group_key]
            values = [r["rqa_stats"][feature] for r in results]
            data_per_group.append(values)

        # Create violin plot
        positions = np.arange(len(group_keys))
        parts = ax.violinplot(data_per_group, positions=positions, showmeans=True, showmedians=True)

        # Color the violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)

        # Style the lines
        for partname in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
            if partname in parts:
                parts[partname].set_edgecolor('black')
                parts[partname].set_linewidth(1)

        # Add individual points (jittered)
        for i, (group_key, values) in enumerate(zip(group_keys, data_per_group)):
            jitter = np.random.uniform(-0.1, 0.1, len(values))
            ax.scatter(positions[i] + jitter, values, c=colors[i], alpha=0.6, s=40, edgecolors='black', linewidths=0.5, zorder=5)

        ax.set_xticks(positions)
        ax.set_xticklabels(group_names)
        ax.set_ylabel(feature)
        ax.set_title(feature, fontweight="bold")

        # Add expected trend annotation
        expected_trends = {
            "DET": "↑ in MCI",
            "LAM": "↑ in MCI",
            "TT": "↑ in MCI",
            "L_mean": "↑ in MCI",
            "ENTR": "↓ in MCI",
            "DIV": "↓ in MCI",
        }
        if feature in expected_trends:
            ax.annotate(f"Expected: {expected_trends[feature]}", xy=(0.02, 0.98),
                       xycoords='axes fraction', fontsize=8, ha='left', va='top',
                       style='italic', color='gray')

    # Hide unused axes
    for idx in range(len(features_to_plot), len(axes)):
        axes[idx].axis("off")

    group_str = " vs ".join(group_names)
    plt.suptitle(f"RQA Features: {group_str}\n(Violin plots with individual subjects)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_path = output_dir / "rqa_violin_comparison.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def print_expected_trends():
    """Print expected MCI vs HC trends based on brain criticality framework."""
    print("\n" + "=" * 70)
    print("EXPECTED MCI vs HC TRENDS (Brain Criticality Framework)")
    print("=" * 70)
    print("""
MCI represents shift toward SUPERCRITICALITY (more rigid, less flexible):

Feature   | MCI vs HC | Interpretation
----------|-----------|------------------------------------------------
DET       | ↑ higher  | More deterministic dynamics (predictable)
LAM       | ↑ higher  | More laminar states (system gets "stuck")
TT        | ↑ higher  | Longer trapping time in attractor states
L_mean    | ↑ higher  | Longer diagonal lines = sustained patterns
L_max     | ↑ higher  | Max recurrence length = attractor collapse
ENTR      | ↓ lower   | Reduced complexity/flexibility
DIV       | ↓ lower   | Lower divergence = less chaotic

Healthy brains operate near criticality ("edge of chaos").
MCI/AD shows overcoupled, rigid dynamics = departure from optimal.
""")
    print("=" * 70)


def list_subjects(groups: dict):
    """List available subjects by group."""
    print("\n" + "=" * 60)
    print("AVAILABLE SUBJECTS")
    print("=" * 60)

    for group_key in ["hc", "mci", "ad"]:
        subjects = groups.get(group_key, [])
        if not subjects:
            continue

        group_name = GROUP_NAMES.get(subjects[0][1], group_key.upper())
        print(f"\n{group_name} subjects ({len(subjects)}):")
        for i, (fif_path, label, condition, subject_id) in enumerate(subjects):
            print(f"  {i+1:3d}. {subject_id:10s} ({condition})")

    total = sum(len(groups.get(k, [])) for k in ["hc", "mci", "ad"])
    print(f"\nTotal: {total} unique subjects")


def main():
    parser = argparse.ArgumentParser(
        description="Compare group latent dynamics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_hc_mci.py                       # Default: HC vs MCI, 3 subjects each
  python compare_hc_mci.py --conditions HID MCI  # HC vs MCI only
  python compare_hc_mci.py --conditions HID AD   # HC vs AD only
  python compare_hc_mci.py --conditions HID MCI AD  # All three groups
  python compare_hc_mci.py --n-subjects 5        # 5 subjects per group
  python compare_hc_mci.py --n-chunks 3          # Average RQA over 3 chunks per subject
  python compare_hc_mci.py --chunk 2             # Use only chunk 2 for each subject
  python compare_hc_mci.py --list-subjects       # Show available subjects
        """
    )
    parser.add_argument("--n-subjects", type=int, default=3,
                        help="Number of unique subjects per group (default: 3)")
    parser.add_argument("--n-chunks", type=int, default=1,
                        help="Number of chunks to average per subject (default: 1)")
    parser.add_argument("--chunk", type=int, default=None,
                        help="Specific chunk index to use (overrides --n-chunks)")
    parser.add_argument("--conditions", type=str, nargs="+", default=["HID", "MCI"],
                        help="Conditions to compare: HID, MCI, AD (default: HID MCI)")
    parser.add_argument("--rr-target", type=float, default=0.02,
                        help="Target recurrence rate (default: 0.02 = 2%%)")
    parser.add_argument("--theiler", type=int, default=THEILER_WINDOW,
                        help=f"Theiler window in samples (default: {THEILER_WINDOW})")
    parser.add_argument("--no-theiler", action="store_true",
                        help="Disable Theiler window")
    parser.add_argument("--list-subjects", action="store_true",
                        help="List available subjects and exit")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't display plots interactively")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Custom output directory (default: timestamped folder)")
    args = parser.parse_args()

    # Handle Theiler window
    theiler = 0 if args.no_theiler else args.theiler

    # Create timestamped output directory
    base_output_dir = ensure_output_dir()
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = create_timestamped_output_dir(base_output_dir, "compare_hc_mci")

    # Get subjects by group for selected conditions
    fif_files = get_fif_files(args.conditions)
    groups = get_subjects_by_group(fif_files)

    # Print summary
    group_counts = []
    for key in ["hc", "mci", "ad"]:
        if groups.get(key):
            group_counts.append(f"{len(groups[key])} {key.upper()}")
    print(f"Found {', '.join(group_counts)} subjects for conditions: {args.conditions}")

    # Handle --list-subjects
    if args.list_subjects:
        list_subjects(groups)
        return 0

    # Check we have at least two groups
    active_groups = {k: v for k, v in groups.items() if v}
    if len(active_groups) < 2:
        print("Need at least 2 groups with subjects for comparison!")
        print("Available groups:", list(active_groups.keys()))
        return 1

    # Save parameters
    params = {
        "n_subjects": args.n_subjects,
        "n_chunks": args.n_chunks,
        "chunk": args.chunk,
        "conditions": args.conditions,
        "rr_target": args.rr_target,
        "theiler": theiler,
        "checkpoint_path": str(CHECKPOINT_PATH),
        "filter_low": FILTER_LOW,
        "filter_high": FILTER_HIGH,
        "chunk_duration": CHUNK_DURATION,
        "sfreq": SFREQ,
        "groups_found": {k: len(v) for k, v in active_groups.items()},
    }
    save_parameters(output_dir, params)

    # Load model
    print("\nLoading model...")
    model_info = load_model_from_checkpoint(CHECKPOINT_PATH, DEVICE)

    # Create model
    first_subject = list(active_groups.values())[0][0]
    first_data = load_and_preprocess_fif(
        first_subject[0], FILTER_LOW, FILTER_HIGH, CHUNK_DURATION,
        include_amplitude=model_info["include_amplitude"],
        verbose=False,
    )
    model = create_model(first_data["n_channels"], model_info, DEVICE)

    # Analyze subjects from each group
    group_results = {}

    for group_key, subjects in active_groups.items():
        group_name = GROUP_NAMES.get(subjects[0][1], group_key.upper())

        print(f"\nAnalyzing {group_name} subjects (targeting {args.n_subjects})...")
        group_results[group_key] = []

        # Iterate through subjects until we have enough valid ones
        subjects_processed = 0
        for fif_path, label, condition, subject_id in subjects:
            if subjects_processed >= args.n_subjects:
                break

            print(f"  {subject_id} ({condition})...")
            res = analyze_subject(
                model, fif_path, model_info,
                rr_target=args.rr_target,
                theiler=theiler,
                chunk_idx=args.chunk,
                n_chunks=args.n_chunks,
            )

            # Skip subjects with no chunks
            if res['n_chunks_available'] == 0:
                print(f"    WARNING: No chunks available, skipping subject")
                continue

            print(f"    Used {res['n_chunks_used']}/{res['n_chunks_available']} chunks")
            group_results[group_key].append(res)
            subjects_processed += 1

        if subjects_processed < args.n_subjects:
            print(f"  NOTE: Only found {subjects_processed} valid subjects (requested {args.n_subjects})")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Settings: RR={args.rr_target*100:.0f}%, Theiler={theiler}, Chunks={args.n_chunks if args.chunk is None else f'#{args.chunk}'}")

    for group_key, results in group_results.items():
        group_name = GROUP_NAMES.get(results[0]["rqa_stats"] if isinstance(results[0]["rqa_stats"], int) else {"hc": 0, "mci": 1, "ad": 2}.get(group_key, 0), group_key.upper())
        print(f"\n{group_name} subjects (n={len(results)}):")
        for r in results:
            print(f"  {r['subject_id']}: DET={r['rqa_stats']['DET']:.3f}, RR={r['rqa_stats']['RR']:.4f}")
        mean_det = np.mean([r['rqa_stats']['DET'] for r in results])
        print(f"  Mean DET: {mean_det:.3f}")

    # Print expected trends before analysis summary
    print_expected_trends()

    # Generate plots
    print("\nGenerating plots...")
    plot_group_comparison(group_results, output_dir, args.rr_target, not args.no_show)
    plot_rqa_violin(group_results, output_dir, not args.no_show)

    print(f"\nAll plots saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
