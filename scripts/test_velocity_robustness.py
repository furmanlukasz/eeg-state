#!/usr/bin/env python3
"""
Velocity Robustness Test: Verify group differences are stable across Δt and methods

This script addresses reviewer concern about noise sensitivity when using Δt = 1 sample
for velocity estimation. It tests whether expert/novice (or HC/MCI) speed differences
are preserved across:
- Finite differences with Δt ∈ {1, 2, 3, 5}
- Savitzky-Golay derivative (noise-robust polynomial fit)
- Moving average smoothing before differentiation

Usage:
    # Test on meditation data with trained model
    python scripts/test_velocity_robustness.py \
        --checkpoint models/best_meditation.pt \
        --data_dir /path/to/ds001787 \
        --output_dir results/velocity_robustness

    # Test on Greek MCI data
    python scripts/test_velocity_robustness.py \
        --checkpoint models/best_MCI_AD_HC.pt \
        --data_dir /path/to/greek_data \
        --dataset greek_resting

    # Quick test (fewer subjects)
    python scripts/test_velocity_robustness.py --quick

Output:
    - velocity_robustness_table.csv (group means per config)
    - velocity_robustness_figure.png (bar plot with CIs)
    - velocity_robustness_results.json (full results)
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
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
sys.path.insert(0, str(SCRIPT_DIR / "local_analysis"))

from velocity import (
    compute_speed,
    compute_speed_robustness,
    compare_speed_configurations,
    VelocityConfig,
)


def load_model_and_config(checkpoint_path: Path, device: str = "cpu"):
    """Load model from checkpoint."""
    import torch
    import ast
    from eeg_biomarkers.models import TransformerAutoencoder

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract model info from checkpoint
    config = checkpoint.get("config", {})

    # Config values may be stored as strings (Hydra serialization quirk)
    model_config_raw = config.get("model", {})
    if isinstance(model_config_raw, str):
        model_config = ast.literal_eval(model_config_raw)
    else:
        model_config = model_config_raw

    encoder_config = model_config.get("encoder", {})
    phase_config = model_config.get("phase", {})

    # Determine model parameters
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Check if amplitude is included (phase_channels = 3 vs 2)
    include_amplitude = phase_config.get("include_amplitude", False)
    phase_channels = 3 if include_amplitude else 2

    # Infer n_channels from conv layer input dimension
    # Conv layer shape is (out_channels, in_channels, kernel_size)
    # where in_channels = n_channels * phase_channels
    for key in state_dict:
        if "conv_layers.0.0.weight" in key:
            input_dim = state_dict[key].shape[1]
            n_channels = input_dim // phase_channels
            break
    else:
        n_channels = 79  # Default for meditation dataset

    print(f"  Inferred: n_channels={n_channels}, phase_channels={phase_channels}, include_amplitude={include_amplitude}")

    # Create model
    model = TransformerAutoencoder(
        n_channels=n_channels,
        hidden_size=encoder_config.get("hidden_size", 64),
        n_heads=encoder_config.get("n_heads", 4),
        n_transformer_layers=encoder_config.get("n_transformer_layers", 2),
        dim_feedforward=encoder_config.get("dim_feedforward", 256),
        dropout=encoder_config.get("dropout", 0.1),
        phase_channels=phase_channels,
    )

    # Load weights
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    return model, {"n_channels": n_channels, "include_amplitude": include_amplitude, "phase_channels": phase_channels}


def get_latent_trajectories(
    model,
    data_files: list,
    model_info: dict,
    device: str = "cpu",
    max_subjects_per_group: Optional[int] = None,
    chunk_duration: float = 5.0,
    sfreq: float = 250.0,
) -> dict:
    """
    Load data and compute latent trajectories for all subjects.

    Args:
        max_subjects_per_group: Max subjects PER GROUP (not total)

    Returns:
        Dict mapping group_name -> list of (subject_id, trajectory) tuples
    """
    import torch
    from load_data import load_eeg_from_file, extract_phase_circular

    trajectories_by_group = {}
    subjects_per_group = {}  # Track count per group

    for file_path, label, group_name in tqdm(data_files, desc="Loading subjects"):
        # Check if we have enough subjects for this group
        if max_subjects_per_group:
            if subjects_per_group.get(group_name, 0) >= max_subjects_per_group:
                continue

        try:
            # Load EEG
            raw_data, sfreq_actual, _ = load_eeg_from_file(file_path, verbose=False)

            # Chunk and process
            chunk_samples = int(chunk_duration * sfreq_actual)
            n_chunks = raw_data.shape[1] // chunk_samples
            if n_chunks == 0:
                continue

            subject_latents = []
            for i in range(n_chunks):
                start = i * chunk_samples
                end = start + chunk_samples
                chunk_data = raw_data[:, start:end]

                # Extract phase
                phase_data = extract_phase_circular(
                    chunk_data, sfreq_actual,
                    include_amplitude=model_info["include_amplitude"],
                    skip_filter=True,
                )

                # Compute latent
                with torch.no_grad():
                    x = torch.tensor(phase_data, dtype=torch.float32, device=device)
                    x = x.unsqueeze(0)  # Add batch dimension
                    latent = model.encode(x).squeeze(0).cpu().numpy()
                    subject_latents.append(latent)

            if subject_latents:
                # Concatenate chunks into continuous trajectory
                trajectory = np.concatenate(subject_latents, axis=0)
                subject_id = file_path.stem[:10]

                if group_name not in trajectories_by_group:
                    trajectories_by_group[group_name] = []
                trajectories_by_group[group_name].append((subject_id, trajectory))
                subjects_per_group[group_name] = subjects_per_group.get(group_name, 0) + 1

        except Exception as e:
            print(f"  Error processing {file_path.name}: {e}")
            continue

    return trajectories_by_group


def test_velocity_robustness(
    trajectories_by_group: dict,
    configurations: list[dict],
) -> pd.DataFrame:
    """
    Test group differences across velocity configurations.

    Args:
        trajectories_by_group: Dict mapping group -> list of (subject_id, trajectory)
        configurations: List of velocity config dicts

    Returns:
        DataFrame with results for each configuration
    """
    results = []

    for config_dict in configurations:
        config = VelocityConfig(**config_dict)
        config_name = _config_to_name(config_dict)

        group_speeds = {}

        for group_name, subjects in trajectories_by_group.items():
            speeds = []
            for subject_id, trajectory in subjects:
                try:
                    speed = compute_speed(trajectory, config=config)
                    speeds.append(np.mean(speed))
                except (ValueError, ImportError):
                    pass

            if speeds:
                group_speeds[group_name] = {
                    "mean": np.mean(speeds),
                    "std": np.std(speeds),
                    "n": len(speeds),
                    "values": speeds,
                }

        # Compute effect size if we have two groups
        group_names = list(group_speeds.keys())
        if len(group_names) >= 2:
            g1, g2 = group_names[0], group_names[1]
            v1 = group_speeds[g1]["values"]
            v2 = group_speeds[g2]["values"]

            # Cohen's d
            pooled_std = np.sqrt(
                ((len(v1) - 1) * np.var(v1) + (len(v2) - 1) * np.var(v2)) /
                (len(v1) + len(v2) - 2)
            )
            cohens_d = (np.mean(v1) - np.mean(v2)) / pooled_std if pooled_std > 0 else 0

            # t-test
            t_stat, p_value = stats.ttest_ind(v1, v2)

            results.append({
                "config": config_name,
                "method": config.method,
                "delta_t": config.delta_t if config.method == "finite_diff" else None,
                "savgol_window": config.savgol_window if config.method == "savgol" else None,
                f"{g1}_mean": group_speeds[g1]["mean"],
                f"{g1}_std": group_speeds[g1]["std"],
                f"{g1}_n": group_speeds[g1]["n"],
                f"{g2}_mean": group_speeds[g2]["mean"],
                f"{g2}_std": group_speeds[g2]["std"],
                f"{g2}_n": group_speeds[g2]["n"],
                "cohens_d": cohens_d,
                "t_stat": t_stat,
                "p_value": p_value,
            })

    return pd.DataFrame(results)


def _config_to_name(config_dict: dict) -> str:
    """Convert config dict to readable name."""
    method = config_dict.get("method", "finite_diff")
    if method == "finite_diff":
        dt = config_dict.get("delta_t", 1)
        return f"FD(Δt={dt})"
    elif method == "savgol":
        window = config_dict.get("savgol_window", 5)
        return f"SG(w={window})"
    return str(config_dict)


def plot_robustness_results(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Velocity Estimation Robustness",
):
    """Create bar plot comparing group differences across configurations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Find group column names
    group_cols = [c for c in df.columns if c.endswith("_mean") and not c.startswith("config")]
    if len(group_cols) < 2:
        print("Warning: Need at least 2 groups for comparison plot")
        return

    g1_mean = group_cols[0]
    g2_mean = group_cols[1]
    g1_name = g1_mean.replace("_mean", "")
    g2_name = g2_mean.replace("_mean", "")

    # Left: Group means by config
    ax = axes[0]
    x = np.arange(len(df))
    width = 0.35

    ax.bar(x - width/2, df[g1_mean], width, label=g1_name, alpha=0.8)
    ax.bar(x + width/2, df[g2_mean], width, label=g2_name, alpha=0.8)

    ax.set_xlabel("Velocity Configuration")
    ax.set_ylabel("Mean Speed")
    ax.set_title(f"Mean Speed by Group and Configuration")
    ax.set_xticks(x)
    ax.set_xticklabels(df["config"], rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Right: Effect size (Cohen's d)
    ax = axes[1]
    colors = ["green" if d > 0 else "red" for d in df["cohens_d"]]
    bars = ax.bar(x, df["cohens_d"], color=colors, alpha=0.7)

    # Add significance markers
    for i, (d, p) in enumerate(zip(df["cohens_d"], df["p_value"])):
        marker = ""
        if p < 0.001:
            marker = "***"
        elif p < 0.01:
            marker = "**"
        elif p < 0.05:
            marker = "*"

        if marker:
            y_pos = d + 0.05 if d > 0 else d - 0.1
            ax.text(i, y_pos, marker, ha="center", fontsize=12)

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.axhline(y=0.2, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axhline(y=-0.2, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axhline(y=-0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axhline(y=0.8, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axhline(y=-0.8, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    ax.set_xlabel("Velocity Configuration")
    ax.set_ylabel("Cohen's d")
    ax.set_title(f"Effect Size ({g1_name} vs {g2_name})")
    ax.set_xticks(x)
    ax.set_xticklabels(df["config"], rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    # Add legend for significance
    ax.text(0.02, 0.98, "* p<0.05  ** p<0.01  *** p<0.001",
            transform=ax.transAxes, fontsize=9, verticalalignment="top")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test velocity estimation robustness across Δt and methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to data directory",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/velocity_robustness",
        help="Output directory for results",
    )
    parser.add_argument(
        "--dataset", type=str, default="meditation_bids",
        choices=["meditation_bids", "greek_resting"],
        help="Dataset type",
    )
    parser.add_argument(
        "--max_subjects", type=int, default=None,
        help="Max subjects per group (for quick testing)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test mode (5 subjects per group)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for inference (cpu, cuda, mps)",
    )

    args = parser.parse_args()

    if args.quick:
        args.max_subjects = 5

    # Setup paths
    checkpoint_path = Path(args.checkpoint)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("VELOCITY ROBUSTNESS TEST")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data dir: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Dataset: {args.dataset}")
    print(f"Max subjects: {args.max_subjects or 'all'}")

    # Define configurations to test
    configurations = [
        # Finite differences with various Δt
        {"method": "finite_diff", "delta_t": 1},
        {"method": "finite_diff", "delta_t": 2},
        {"method": "finite_diff", "delta_t": 3},
        {"method": "finite_diff", "delta_t": 5},
        # Savitzky-Golay with various windows
        {"method": "savgol", "savgol_window": 5, "savgol_poly": 2},
        {"method": "savgol", "savgol_window": 7, "savgol_poly": 2},
        {"method": "savgol", "savgol_window": 9, "savgol_poly": 2},
    ]

    # Load model
    print("\nLoading model...")
    model, model_info = load_model_and_config(checkpoint_path, args.device)
    print(f"  n_channels: {model_info['n_channels']}")
    print(f"  include_amplitude: {model_info['include_amplitude']}")

    # Get data files
    print("\nGetting data files...")
    import os
    os.environ["EEG_DATASET"] = args.dataset

    if args.dataset == "meditation_bids":
        from eeg_biomarkers.data.dataset_config import get_dataset_config
        config = get_dataset_config("meditation_bids")
        data_files = []
        for group in config.groups:
            group_files = config.get_files_for_group(data_dir, group)
            for f in group_files:
                data_files.append((f, group.label, group.name))
    else:
        from config import get_fif_files
        data_files = get_fif_files(["MCI", "HID"])

    print(f"  Found {len(data_files)} files")

    if len(data_files) == 0:
        print("\nERROR: No data files found!")
        print(f"  Check that {data_dir} contains valid EEG files")
        print("  For meditation_bids: needs participants.tsv and sub-*/eeg/*.bdf files")
        sys.exit(1)

    # Load trajectories
    print("\nComputing latent trajectories...")
    trajectories_by_group = get_latent_trajectories(
        model, data_files, model_info,
        device=args.device,
        max_subjects_per_group=args.max_subjects,
    )

    for group, subjects in trajectories_by_group.items():
        print(f"  {group}: {len(subjects)} subjects")

    # Test robustness
    print("\nTesting velocity robustness across configurations...")
    results_df = test_velocity_robustness(trajectories_by_group, configurations)

    # Save results
    csv_path = output_dir / f"velocity_robustness_table_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Plot results
    fig_path = output_dir / f"velocity_robustness_figure_{timestamp}.png"
    plot_robustness_results(results_df, fig_path)

    # Save full results as JSON
    json_path = output_dir / f"velocity_robustness_results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "checkpoint": str(checkpoint_path),
            "dataset": args.dataset,
            "configurations": configurations,
            "results": results_df.to_dict(orient="records"),
        }, f, indent=2)
    print(f"Saved: {json_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("ROBUSTNESS SUMMARY")
    print("=" * 70)

    if results_df.empty or "config" not in results_df.columns:
        print("\nNo results to display - check data availability.")
        print("Make sure the data directory contains valid EEG files.")
        return

    print(f"\n{'Config':<15} {'Effect (d)':<12} {'p-value':<12} {'Robust?':<10}")
    print("-" * 50)

    baseline_rows = results_df[results_df["config"] == "FD(Δt=1)"]
    if baseline_rows.empty:
        print("Warning: No baseline (FD Δt=1) results found")
        baseline_d = 0
    else:
        baseline_d = baseline_rows["cohens_d"].values[0]

    for _, row in results_df.iterrows():
        d = row["cohens_d"]
        p = row["p_value"]
        # "Robust" if direction matches baseline and magnitude is similar
        same_direction = (d * baseline_d) > 0
        similar_magnitude = abs(abs(d) - abs(baseline_d)) < 0.3
        robust = "✓" if (same_direction and similar_magnitude) else "~" if same_direction else "✗"

        print(f"{row['config']:<15} {d:>+.3f}       {p:<.4f}       {robust}")

    print("\n✓ = robust (same direction, similar magnitude)")
    print("~ = partially robust (same direction, different magnitude)")
    print("✗ = not robust (opposite direction)")


if __name__ == "__main__":
    main()
