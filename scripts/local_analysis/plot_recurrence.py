#!/usr/bin/env python3
"""
Plot Recurrence Matrices for Local Analysis

Visualize recurrence matrices from trained autoencoder latents.
Interactive version for local analysis on M1 Mac.

Usage:
    python plot_recurrence.py                       # Default: 2 subjects
    python plot_recurrence.py --n-subjects 5        # Plot 5 subjects
    python plot_recurrence.py --subject S001        # Plot specific subject
    python plot_recurrence.py --list-subjects       # List available subjects
    python plot_recurrence.py --list-chunks S001    # List chunks for subject
    python plot_recurrence.py --chunk 3             # Use chunk 3 (0-indexed)
    python plot_recurrence.py --chunk all           # Plot all chunks
    python plot_recurrence.py --theiler 0           # Disable Theiler window
    python plot_recurrence.py --no-theiler          # Same as --theiler 0
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CHECKPOINT_PATH, DATA_DIR, OUTPUT_DIR, DEVICE,
    FILTER_LOW, FILTER_HIGH, CHUNK_DURATION, SFREQ,
    RR_TARGETS, THEILER_WINDOW, ensure_output_dir, get_fif_files, get_subject_id,
    get_unique_subjects
)
from load_model import load_model_from_checkpoint, create_model, compute_latent_trajectory
from load_data import load_and_preprocess_fif


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


def compute_angular_distance_matrix(latent_trajectory: np.ndarray) -> np.ndarray:
    """Compute angular distance matrix from latent trajectory."""
    norms = np.linalg.norm(latent_trajectory, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = latent_trajectory / norms

    cos_sim = np.dot(normalized, normalized.T)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    distance_matrix = np.arccos(cos_sim)

    return distance_matrix


def compute_recurrence_matrix(
    distance_matrix: np.ndarray,
    target_rr: float = 0.02,
    theiler_window: int = 0,
) -> tuple[np.ndarray, float]:
    """Compute binary recurrence matrix with RR-controlled threshold."""
    N = distance_matrix.shape[0]

    # Create mask for threshold computation
    if theiler_window > 0:
        mask = np.ones((N, N), dtype=bool)
        for i in range(N):
            for j in range(max(0, i - theiler_window + 1), min(N, i + theiler_window)):
                mask[i, j] = False
    else:
        mask = ~np.eye(N, dtype=bool)

    off_diag = distance_matrix[mask]
    epsilon = np.percentile(off_diag, target_rr * 100)

    R = (distance_matrix <= epsilon).astype(np.float64)

    if theiler_window > 0:
        for i in range(N):
            for j in range(max(0, i - theiler_window + 1), min(N, i + theiler_window)):
                R[i, j] = 0

    return R, epsilon


def compute_rqa_stats(R: np.ndarray, l_min: int = 2, v_min: int = 2) -> dict:
    """
    Compute RQA statistics from recurrence matrix.

    Features computed:
    - RR: Recurrence Rate
    - DET: Determinism (diagonal lines)
    - L_mean: Mean diagonal line length
    - L_max: Max diagonal line length
    - LAM: Laminarity (vertical lines)
    - TT: Trapping Time (mean vertical line length)
    - V_max: Max vertical line length
    - DIV: Divergence (1/L_max)
    - ENTR: Shannon entropy of diagonal line distribution
    """
    N = R.shape[0]
    mask = ~np.eye(N, dtype=bool)

    total_points = mask.sum()
    recurrence_points = R[mask].sum()
    rr = recurrence_points / total_points if total_points > 0 else 0

    # Diagonal line analysis (for DET, L_mean, L_max, ENTR)
    diagonal_lengths = []
    for offset in range(l_min, N):
        diag = np.diag(R, offset)
        run_length = 0
        for val in diag:
            if val > 0:
                run_length += 1
            else:
                if run_length >= l_min:
                    diagonal_lengths.append(run_length)
                run_length = 0
        if run_length >= l_min:
            diagonal_lengths.append(run_length)

    # Also check negative diagonals (below main diagonal)
    for offset in range(l_min, N):
        diag = np.diag(R, -offset)
        run_length = 0
        for val in diag:
            if val > 0:
                run_length += 1
            else:
                if run_length >= l_min:
                    diagonal_lengths.append(run_length)
                run_length = 0
        if run_length >= l_min:
            diagonal_lengths.append(run_length)

    det_points = sum(diagonal_lengths)
    det = det_points / recurrence_points if recurrence_points > 0 else 0
    l_mean = np.mean(diagonal_lengths) if diagonal_lengths else 0
    l_max = max(diagonal_lengths) if diagonal_lengths else 0
    div = 1.0 / l_max if l_max > 0 else 0

    # Entropy of diagonal line length distribution
    if diagonal_lengths:
        hist, _ = np.histogram(diagonal_lengths, bins=range(l_min, max(diagonal_lengths) + 2))
        hist = hist[hist > 0]
        p = hist / hist.sum()
        entr = -np.sum(p * np.log(p))
    else:
        entr = 0

    # Vertical line analysis (for LAM, TT, V_max)
    vertical_lengths = []
    for col in range(N):
        run_length = 0
        for row in range(N):
            if row == col:  # Skip main diagonal
                if run_length >= v_min:
                    vertical_lengths.append(run_length)
                run_length = 0
                continue
            if R[row, col] > 0:
                run_length += 1
            else:
                if run_length >= v_min:
                    vertical_lengths.append(run_length)
                run_length = 0
        if run_length >= v_min:
            vertical_lengths.append(run_length)

    lam_points = sum(vertical_lengths)
    lam = lam_points / recurrence_points if recurrence_points > 0 else 0
    tt = np.mean(vertical_lengths) if vertical_lengths else 0
    v_max = max(vertical_lengths) if vertical_lengths else 0

    return {
        "RR": rr,
        "DET": det,
        "L_mean": l_mean,
        "L_max": l_max,
        "LAM": lam,
        "TT": tt,
        "V_max": v_max,
        "DIV": div,
        "ENTR": entr,
    }


def list_subjects(fif_files: list):
    """List all available unique subjects."""
    unique = get_unique_subjects(fif_files)

    print("\nAvailable unique subjects:")
    print("-" * 60)

    # Separate by group: HC (label=0), MCI (label=1), AD (label=2)
    groups = {"HC": [], "MCI": [], "AD": []}
    label_to_group = {0: "HC", 1: "MCI", 2: "AD"}

    for subject_id, (fif_path, label, condition) in unique.items():
        group_name = label_to_group.get(label, "Unknown")
        groups[group_name].append((subject_id, condition))

    for group_name in ["HC", "MCI", "AD"]:
        subjects = groups[group_name]
        if subjects:
            print(f"\n{group_name} subjects ({len(subjects)}):")
            for i, (subject_id, condition) in enumerate(sorted(subjects)):
                print(f"  {i+1:3d}. {subject_id:10s} ({condition})")

    total = sum(len(g) for g in groups.values())
    counts = [f"{len(groups[g])} {g}" for g in ["HC", "MCI", "AD"] if groups[g]]
    print(f"\nTotal: {' + '.join(counts)} = {total} unique subjects")


def list_chunks_for_subject(fif_path: Path, model_info: dict):
    """List available chunks for a subject."""
    data = load_and_preprocess_fif(
        fif_path,
        FILTER_LOW, FILTER_HIGH, CHUNK_DURATION,
        include_amplitude=model_info["include_amplitude"],
    )

    subject_id = data["subject_id"]
    n_chunks = len(data["chunks"])
    chunk_samples = data["chunks"][0].shape[1]
    chunk_duration = chunk_samples / data["sfreq"]

    print(f"\nSubject: {subject_id}")
    print(f"Total chunks: {n_chunks}")
    print(f"Chunk duration: {chunk_duration:.1f}s ({chunk_samples} samples)")
    print(f"Total duration: {n_chunks * chunk_duration:.1f}s")
    print(f"\nChunk indices: 0 to {n_chunks - 1}")
    print(f"Use --chunk <idx> or --chunk all")


def plot_single_subject(
    model,
    fif_path: Path,
    model_info: dict,
    output_dir: Path,
    rr_targets: list = [0.01, 0.02, 0.05],
    theiler_window: int = 50,
    chunk_idx: int | str = 0,
    show_plot: bool = True,
):
    """Plot recurrence matrices for a single subject.

    Args:
        chunk_idx: Integer for specific chunk, or "all" to plot all chunks
    """

    # Load and preprocess
    data = load_and_preprocess_fif(
        fif_path,
        FILTER_LOW, FILTER_HIGH, CHUNK_DURATION,
        include_amplitude=model_info["include_amplitude"],
    )

    subject_id = data["subject_id"]
    n_chunks = len(data["chunks"])

    # Determine which chunks to process
    if chunk_idx == "all":
        chunks_to_process = list(range(n_chunks))
    else:
        if chunk_idx >= n_chunks:
            print(f"Warning: chunk_idx {chunk_idx} >= {n_chunks}, using 0")
            chunk_idx = 0
        chunks_to_process = [chunk_idx]

    print(f"Subject {subject_id}: {n_chunks} chunks available, processing {len(chunks_to_process)}")

    results = []
    for cidx in chunks_to_process:
        phase_data = data["chunks"][cidx]

        # Compute latent trajectory
        print(f"  Computing latent trajectory for chunk {cidx}...")
        latent = compute_latent_trajectory(model, phase_data, DEVICE)
        print(f"    Latent shape: {latent.shape} (time, hidden_size)")

        # Compute distance matrix
        distance_matrix = compute_angular_distance_matrix(latent)

        # Create figure with distance + multiple RR comparisons
        n_rr = len(rr_targets)
        fig, axes = plt.subplots(1, n_rr + 1, figsize=(5 * (n_rr + 1), 5))

        time_extent = latent.shape[0] / SFREQ

        # Distance matrix
        im0 = axes[0].imshow(
            distance_matrix, cmap="viridis", origin="lower",
            extent=[0, time_extent, 0, time_extent]
        )
        axes[0].set_title("Angular Distance Matrix")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Time (s)")
        plt.colorbar(im0, ax=axes[0], label="Distance (rad)")

        # Recurrence matrices at different RR
        for i, rr in enumerate(rr_targets):
            R, eps = compute_recurrence_matrix(distance_matrix, rr, theiler_window)
            stats = compute_rqa_stats(R)

            axes[i + 1].imshow(
                R, cmap="binary", origin="lower",
                extent=[0, time_extent, 0, time_extent]
            )
            axes[i + 1].set_title(f"RR={rr*100:.0f}%\nDET={stats['DET']:.2f}")
            axes[i + 1].set_xlabel("Time (s)")
            if i == 0:
                axes[i + 1].set_ylabel("Time (s)")

        # Title with Theiler info
        theiler_str = f"Theiler={theiler_window}" if theiler_window > 0 else "No Theiler"
        chunk_str = f"chunk {cidx}" if len(chunks_to_process) > 1 or chunk_idx != 0 else ""
        title_parts = [f"Subject: {subject_id}", theiler_str]
        if chunk_str:
            title_parts.append(chunk_str)
        plt.suptitle(" | ".join(title_parts), fontsize=14, fontweight="bold")
        plt.tight_layout()

        # Save
        if len(chunks_to_process) > 1:
            save_path = output_dir / f"{subject_id}_chunk{cidx}_recurrence.png"
        else:
            save_path = output_dir / f"{subject_id}_recurrence_matrices.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"    Saved: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        results.append((latent, distance_matrix))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Plot recurrence matrices locally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_recurrence.py                       # Default: 2 subjects, chunk 0
  python plot_recurrence.py --n-subjects 5        # Plot 5 subjects
  python plot_recurrence.py --subject S001        # Specific subject
  python plot_recurrence.py --list-subjects       # List available subjects
  python plot_recurrence.py --list-chunks S001    # Show chunks for subject
  python plot_recurrence.py --chunk 3             # Use chunk 3
  python plot_recurrence.py --chunk all           # Plot all chunks
  python plot_recurrence.py --no-theiler          # Disable Theiler window
  python plot_recurrence.py --theiler 100         # Custom Theiler window
        """
    )
    parser.add_argument("--n-subjects", type=int, default=2, help="Number of subjects to plot (default: 2)")
    parser.add_argument("--subject", type=str, default=None, help="Specific subject ID to plot")
    parser.add_argument("--conditions", type=str, nargs="+", default=["MCI", "HID", "AD", "HC"],
                        help="Conditions to include (default: all)")
    parser.add_argument("--chunk", type=str, default="0",
                        help="Chunk index (0-based) or 'all' for all chunks (default: 0)")
    parser.add_argument("--theiler", type=int, default=THEILER_WINDOW,
                        help=f"Theiler window in samples (default: {THEILER_WINDOW}, ~{THEILER_WINDOW/SFREQ:.2f}s)")
    parser.add_argument("--no-theiler", action="store_true",
                        help="Disable Theiler window (same as --theiler 0)")
    parser.add_argument("--list-subjects", action="store_true", help="List available subjects and exit")
    parser.add_argument("--list-chunks", type=str, metavar="SUBJECT",
                        help="List chunks for a specific subject and exit")
    parser.add_argument("--no-show", action="store_true", help="Don't display plots interactively")
    args = parser.parse_args()

    # Handle Theiler window
    theiler_window = 0 if args.no_theiler else args.theiler

    # Parse chunk argument
    if args.chunk.lower() == "all":
        chunk_idx = "all"
    else:
        try:
            chunk_idx = int(args.chunk)
        except ValueError:
            print(f"Error: --chunk must be an integer or 'all', got '{args.chunk}'")
            return 1

    # Create timestamped output directory
    base_output_dir = ensure_output_dir()
    output_dir = create_timestamped_output_dir(base_output_dir, "plot_recurrence")
    print(f"Output directory: {output_dir}")

    # Save parameters for reproducibility
    save_parameters(output_dir, {
        "n_subjects": args.n_subjects,
        "subject": args.subject,
        "conditions": args.conditions,
        "chunk": args.chunk,
        "theiler": theiler_window,
        "rr_targets": RR_TARGETS,
        "filter_low": FILTER_LOW,
        "filter_high": FILTER_HIGH,
        "chunk_duration": CHUNK_DURATION,
        "sfreq": SFREQ,
        "checkpoint_path": CHECKPOINT_PATH,
        "data_dir": DATA_DIR,
        "device": DEVICE,
    })

    # Get files first (before loading model for list commands)
    fif_files = get_fif_files(args.conditions)

    # Get unique subjects
    unique_subjects = get_unique_subjects(fif_files)
    print(f"Found {len(unique_subjects)} unique subjects in conditions: {args.conditions}")

    if len(unique_subjects) == 0:
        print(f"No .fif files found in {DATA_DIR}")
        return 1

    # Handle --list-subjects
    if args.list_subjects:
        list_subjects(fif_files)
        return 0

    # Load model
    print("\nLoading model...")
    model_info = load_model_from_checkpoint(CHECKPOINT_PATH, DEVICE)

    # Handle --list-chunks
    if args.list_chunks:
        if args.list_chunks in unique_subjects:
            fif_path, _, _ = unique_subjects[args.list_chunks]
            list_chunks_for_subject(fif_path, model_info)
        else:
            # Try partial match
            matching = [sid for sid in unique_subjects if args.list_chunks in sid]
            if not matching:
                print(f"No files found for subject {args.list_chunks}")
                return 1
            fif_path, _, _ = unique_subjects[matching[0]]
            list_chunks_for_subject(fif_path, model_info)
        return 0

    # Create model (need n_channels from first file)
    first_subject = list(unique_subjects.values())[0]
    first_data = load_and_preprocess_fif(
        first_subject[0], FILTER_LOW, FILTER_HIGH, CHUNK_DURATION,
        include_amplitude=model_info["include_amplitude"],
    )
    n_channels = first_data["n_channels"]
    model = create_model(n_channels, model_info, DEVICE)

    # Filter by subject if specified
    if args.subject:
        if args.subject in unique_subjects:
            subjects_to_plot = {args.subject: unique_subjects[args.subject]}
        else:
            # Try partial match
            subjects_to_plot = {sid: data for sid, data in unique_subjects.items() if args.subject in sid}
        if not subjects_to_plot:
            print(f"No files found for subject {args.subject}")
            return 1
    else:
        subjects_to_plot = unique_subjects

    # Plot
    subjects_plotted = 0
    label_to_group = {0: "HC", 1: "MCI", 2: "AD"}
    for subject_id, (fif_path, label, condition) in list(subjects_to_plot.items())[:args.n_subjects]:
        group = label_to_group.get(label, "Unknown")
        print(f"\n{'='*60}")
        print(f"Processing: {subject_id} ({condition}, {group})")
        print(f"Theiler window: {theiler_window} samples ({theiler_window/SFREQ:.2f}s)" if theiler_window > 0 else "Theiler window: DISABLED")
        print(f"{'='*60}")

        plot_single_subject(
            model, fif_path, model_info, output_dir,
            rr_targets=RR_TARGETS,
            theiler_window=theiler_window,
            chunk_idx=chunk_idx,
            show_plot=not args.no_show,
        )
        subjects_plotted += 1

    print(f"\n{'='*60}")
    print(f"Done! Plotted {subjects_plotted} unique subject(s)")
    print(f"All plots saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
