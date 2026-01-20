#!/usr/bin/env python3
"""
3D UMAP Visualization of Latent Trajectories

Compare latent space structure between different groups (HC, MCI, AD).
Interactive 3D plots for local analysis on M1 Mac.

Usage:
    python plot_umap_3d.py                           # Default: HC vs MCI comparison
    python plot_umap_3d.py --conditions HID MCI      # Compare specific conditions
    python plot_umap_3d.py --conditions HID MCI AD   # Compare all three groups
    python plot_umap_3d.py --n-subjects 5            # 5 unique subjects per group
    python plot_umap_3d.py --n-chunks 10             # 10 chunks per subject
    python plot_umap_3d.py --mode mean               # Only mean latents plot
    python plot_umap_3d.py --mode subject            # Only per-subject coloring
    python plot_umap_3d.py --mode trajectory         # Trajectory plot (slow, many points!)
    python plot_umap_3d.py --mode all                # All plots including trajectory
    python plot_umap_3d.py --list-subjects           # List available subjects
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import umap

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CHECKPOINT_PATH, DATA_DIR, OUTPUT_DIR, DEVICE,
    FILTER_LOW, FILTER_HIGH, CHUNK_DURATION, SFREQ,
    UMAP_N_NEIGHBORS, UMAP_MIN_DIST, UMAP_N_COMPONENTS,
    ensure_output_dir, get_fif_files, get_subjects_by_group, get_label_name
)
from load_model import load_model_from_checkpoint, create_model, compute_latent_trajectory
from load_data import load_and_preprocess_fif

# Color mapping for groups
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


def collect_latents_for_groups(
    model,
    groups: dict,
    model_info: dict,
    n_subjects_per_group: int = 3,
    n_chunks_per_subject: int = 5,
):
    """
    Collect latent trajectories for multiple groups.

    Args:
        model: Trained autoencoder
        groups: Dict with 'hc', 'mci', 'ad' lists of (fif_path, label, condition, subject_id)
        model_info: Model configuration dict
        n_subjects_per_group: Number of unique subjects per group
        n_chunks_per_subject: Number of chunks to use per subject

    Returns:
        Dict with latents organized by group key
    """
    results = {}

    for group_key, subjects in groups.items():
        if not subjects:
            continue

        group_name = GROUP_NAMES.get(subjects[0][1], group_key.upper())
        print(f"\nProcessing {group_name} group (targeting {n_subjects_per_group} subjects)...")

        results[group_key] = {
            "latents": [],
            "subject_ids": [],
            "chunk_ids": [],
            "label": subjects[0][1] if subjects else 0,
        }

        # Iterate through subjects until we have enough valid ones
        subjects_processed = 0
        for fif_path, label, condition, subject_id in subjects:
            if subjects_processed >= n_subjects_per_group:
                break

            print(f"  Processing {subject_id} ({condition})...")

            data = load_and_preprocess_fif(
                fif_path, FILTER_LOW, FILTER_HIGH, CHUNK_DURATION,
                include_amplitude=model_info["include_amplitude"],
                verbose=False,
            )

            # Skip subjects with no chunks
            if len(data["chunks"]) == 0:
                print(f"    WARNING: No chunks available, skipping subject")
                continue

            n_chunks = min(n_chunks_per_subject, len(data["chunks"]))
            print(f"    Using {n_chunks}/{len(data['chunks'])} chunks")

            for chunk_idx in range(n_chunks):
                phase_data = data["chunks"][chunk_idx]
                latent = compute_latent_trajectory(model, phase_data, DEVICE)

                results[group_key]["latents"].append(latent)
                results[group_key]["subject_ids"].append(subject_id)
                results[group_key]["chunk_ids"].append(chunk_idx)

            subjects_processed += 1

        if subjects_processed < n_subjects_per_group:
            print(f"  NOTE: Only found {subjects_processed} valid subjects (requested {n_subjects_per_group})")

    return results


def plot_umap_mean_latents(results: dict, output_dir: Path, show_plot: bool = True):
    """
    Plot UMAP of mean latents (one point per chunk).

    Each chunk is reduced to its mean latent vector, then UMAP is applied.
    """
    print("\nComputing mean latents for UMAP...")

    # Compute mean latent per chunk
    all_means = []
    all_labels = []
    all_subject_ids = []

    for group_key, group_data in results.items():
        label = group_data["label"]
        for latent, subj_id in zip(group_data["latents"], group_data["subject_ids"]):
            mean_latent = latent.mean(axis=0)  # (hidden_size,)
            all_means.append(mean_latent)
            all_labels.append(label)
            all_subject_ids.append(subj_id)

    all_means = np.array(all_means)
    all_labels = np.array(all_labels)
    unique_labels = sorted(set(all_labels))
    print(f"  Shape: {all_means.shape} (n_chunks, hidden_size)")
    print(f"  Groups: {[GROUP_NAMES.get(l, str(l)) for l in unique_labels]}")

    # UMAP
    print("  Running UMAP...")
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=3,
        random_state=42,
    )
    embedding = reducer.fit_transform(all_means)

    # Plot 3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    for label_val in unique_labels:
        mask = all_labels == label_val
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            embedding[mask, 2],
            c=GROUP_COLORS.get(label_val, "gray"),
            label=GROUP_NAMES.get(label_val, f"Label {label_val}"),
            alpha=0.6,
            s=50,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")

    group_str = " vs ".join([GROUP_NAMES.get(l, str(l)) for l in unique_labels])
    ax.set_title(f"3D UMAP of Mean Latents (per chunk)\n{group_str}", fontsize=14)
    ax.legend()

    save_path = output_dir / "umap_mean_latents_3d.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return embedding, all_labels


def plot_umap_trajectories(results: dict, output_dir: Path, show_plot: bool = True):
    """
    Plot UMAP of all timepoints, showing temporal trajectories.

    All timepoints from all chunks are projected together.
    """
    print("\nComputing UMAP on all timepoints...")

    # Concatenate all latents
    all_latents = []
    all_labels = []
    all_time_indices = []
    trajectory_boundaries = []  # Track where each chunk starts/ends

    current_idx = 0
    for group_key, group_data in results.items():
        label = group_data["label"]
        for latent in group_data["latents"]:
            n_time = latent.shape[0]
            all_latents.append(latent)
            all_labels.extend([label] * n_time)
            all_time_indices.extend(range(n_time))
            trajectory_boundaries.append((current_idx, current_idx + n_time, label))
            current_idx += n_time

    all_latents = np.vstack(all_latents)
    all_labels = np.array(all_labels)
    all_time_indices = np.array(all_time_indices)
    unique_labels = sorted(set(all_labels))
    print(f"  Shape: {all_latents.shape} (total_timepoints, hidden_size)")

    # UMAP (subsample if too many points)
    max_points = 20000
    if len(all_latents) > max_points:
        print(f"  Subsampling from {len(all_latents)} to {max_points} points...")
        idx = np.random.choice(len(all_latents), max_points, replace=False)
        idx = np.sort(idx)
        all_latents_sub = all_latents[idx]
        all_labels_sub = all_labels[idx]
        all_time_sub = all_time_indices[idx]
    else:
        all_latents_sub = all_latents
        all_labels_sub = all_labels
        all_time_sub = all_time_indices

    print("  Running UMAP...")
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=3,
        random_state=42,
    )
    embedding = reducer.fit_transform(all_latents_sub)

    # Plot 3D with time coloring
    fig = plt.figure(figsize=(14, 6))

    # Left: Color by group
    ax1 = fig.add_subplot(121, projection="3d")
    for label_val in unique_labels:
        name = GROUP_NAMES.get(label_val, f"Label {label_val}")
        mask = all_labels_sub == label_val
        ax1.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            embedding[mask, 2],
            c=GROUP_COLORS.get(label_val, "gray"),
            label=name,
            alpha=0.3,
            s=5,
        )
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")
    ax1.set_zlabel("UMAP 3")
    group_str = " vs ".join([GROUP_NAMES.get(l, str(l)) for l in unique_labels])
    ax1.set_title(f"Colored by Group ({group_str})")
    ax1.legend()

    # Right: Color by time within chunk
    ax2 = fig.add_subplot(122, projection="3d")
    sc = ax2.scatter(
        embedding[:, 0],
        embedding[:, 1],
        embedding[:, 2],
        c=all_time_sub,
        cmap="viridis",
        alpha=0.3,
        s=5,
    )
    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")
    ax2.set_zlabel("UMAP 3")
    ax2.set_title("Colored by Time (within chunk)")
    plt.colorbar(sc, ax=ax2, label="Time index")

    plt.suptitle("3D UMAP of Latent Trajectories", fontsize=14)
    plt.tight_layout()

    save_path = output_dir / "umap_trajectories_3d.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return embedding, all_labels_sub


def plot_umap_per_subject(results: dict, output_dir: Path, show_plot: bool = True):
    """
    Plot UMAP where each subject has a different color.

    Shows how individual subjects cluster in latent space.
    """
    print("\nComputing UMAP colored by subject...")

    all_means = []
    all_labels = []
    all_subject_ids = []

    for group_key, group_data in results.items():
        label = group_data["label"]
        for latent, subj_id in zip(group_data["latents"], group_data["subject_ids"]):
            mean_latent = latent.mean(axis=0)
            all_means.append(mean_latent)
            all_labels.append(label)
            all_subject_ids.append(subj_id)

    all_means = np.array(all_means)
    all_labels = np.array(all_labels)
    unique_subjects = list(dict.fromkeys(all_subject_ids))  # Preserve order
    unique_labels = sorted(set(all_labels))

    print("  Running UMAP...")
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=3,
        random_state=42,
    )
    embedding = reducer.fit_transform(all_means)

    # Create color map for subjects
    n_subjects = len(unique_subjects)
    cmap = plt.cm.get_cmap("tab20", n_subjects)

    fig = plt.figure(figsize=(14, 6))

    # Left: Color by subject
    ax1 = fig.add_subplot(121, projection="3d")
    markers = {0: "o", 1: "^", 2: "s"}  # Circle for HC, triangle for MCI, square for AD
    for i, subj in enumerate(unique_subjects):
        mask = np.array([s == subj for s in all_subject_ids])
        group_label = all_labels[np.where(mask)[0][0]]
        marker = markers.get(group_label, "o")
        group_name = GROUP_NAMES.get(group_label, "?")
        ax1.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            embedding[mask, 2],
            c=[cmap(i)],
            label=f"{subj} ({group_name})",
            marker=marker,
            alpha=0.7,
            s=60,
        )
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")
    ax1.set_zlabel("UMAP 3")
    ax1.set_title("Colored by Subject")
    ax1.legend(loc="upper left", fontsize=8)

    # Right: Color by group with subject markers
    ax2 = fig.add_subplot(122, projection="3d")
    for label_val in unique_labels:
        name = GROUP_NAMES.get(label_val, f"Label {label_val}")
        mask = np.array(all_labels) == label_val
        ax2.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            embedding[mask, 2],
            c=GROUP_COLORS.get(label_val, "gray"),
            label=name,
            alpha=0.6,
            s=50,
        )
    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")
    ax2.set_zlabel("UMAP 3")
    ax2.set_title("Colored by Group")
    ax2.legend()

    group_str = " vs ".join([GROUP_NAMES.get(l, str(l)) for l in unique_labels])
    plt.suptitle(f"UMAP: Subject vs Group Clustering ({group_str})", fontsize=14)
    plt.tight_layout()

    save_path = output_dir / "umap_per_subject_3d.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return embedding, unique_subjects


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
        description="3D UMAP visualization of latent space",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_umap_3d.py                           # Default: HC vs MCI (no AD)
  python plot_umap_3d.py --conditions HID MCI      # HC vs MCI only
  python plot_umap_3d.py --conditions HID MCI AD   # All three groups
  python plot_umap_3d.py --conditions HID AD       # HC vs AD only
  python plot_umap_3d.py --n-subjects 5            # 5 unique subjects per group
  python plot_umap_3d.py --n-chunks 10             # 10 chunks per subject
  python plot_umap_3d.py --mode mean               # Only mean latent UMAP
  python plot_umap_3d.py --mode subject            # Only per-subject coloring
  python plot_umap_3d.py --mode trajectory         # Trajectory UMAP (slow, many points!)
  python plot_umap_3d.py --mode all                # All plots including trajectory
  python plot_umap_3d.py --list-subjects           # Show available subjects

Note: 'trajectory' mode projects all timepoints (1000s of points) and is slow.
      Default mode excludes trajectory for faster execution.
        """
    )
    parser.add_argument("--n-subjects", type=int, default=3,
                        help="Number of unique subjects per group (default: 3)")
    parser.add_argument("--n-chunks", type=int, default=5,
                        help="Number of chunks per subject (default: 5)")
    parser.add_argument("--conditions", type=str, nargs="+", default=["HID", "MCI"],
                        help="Conditions to include: HID, MCI, AD (default: HID MCI)")
    parser.add_argument("--mode", type=str, choices=["all", "default", "mean", "trajectory", "subject"],
                        default="default", help="Which plots: default (mean+subject), all, mean, trajectory, subject")
    parser.add_argument("--list-subjects", action="store_true",
                        help="List available subjects and exit")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't display plots interactively")
    args = parser.parse_args()

    output_dir = ensure_output_dir()

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

    # Check we have at least one group with subjects
    active_groups = {k: v for k, v in groups.items() if v}
    if len(active_groups) < 2:
        print("Need at least 2 groups with subjects for comparison!")
        print("Available groups:", list(active_groups.keys()))
        return 1

    # Load model
    print("\nLoading model...")
    model_info = load_model_from_checkpoint(CHECKPOINT_PATH, DEVICE)

    # Create model using first available subject
    first_subject = list(active_groups.values())[0][0]
    first_data = load_and_preprocess_fif(
        first_subject[0], FILTER_LOW, FILTER_HIGH, CHUNK_DURATION,
        include_amplitude=model_info["include_amplitude"],
        verbose=False,
    )
    model = create_model(first_data["n_channels"], model_info, DEVICE)

    # Collect latents
    print(f"\nCollecting latents for {args.n_subjects} subjects per group, {args.n_chunks} chunks each...")
    results = collect_latents_for_groups(
        model, active_groups, model_info,
        n_subjects_per_group=args.n_subjects,
        n_chunks_per_subject=args.n_chunks,
    )

    # Count unique subjects in results
    for group_key, group_data in results.items():
        n_unique = len(set(group_data['subject_ids']))
        n_chunks = len(group_data['latents'])
        group_name = GROUP_NAMES.get(group_data['label'], group_key.upper())
        print(f"  {group_name}: {n_chunks} chunks from {n_unique} subjects")

    # Generate plots
    # "default" mode = mean + subject (fast, excludes trajectory)
    # "all" mode = mean + trajectory + subject
    if args.mode in ["all", "default", "mean"]:
        plot_umap_mean_latents(results, output_dir, show_plot=not args.no_show)

    if args.mode in ["all", "trajectory"]:
        print("\nNote: Trajectory mode processes many points, this may take a while...")
        plot_umap_trajectories(results, output_dir, show_plot=not args.no_show)

    if args.mode in ["all", "default", "subject"]:
        plot_umap_per_subject(results, output_dir, show_plot=not args.no_show)

    print(f"\nAll plots saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
