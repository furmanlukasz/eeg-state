#!/usr/bin/env python3
"""
MCI Subtype Analysis - Dynamical Geometry Within MCI Strata

Analyzes whether MCI contains distinct dynamical regimes based on:
- Severity strata (MMSE-based)
- Functional impairment strata (FUCAS/FRSSD)
- Depression comorbidity

Also examines regional localization using EGI HydroCel-256 coordinate-based ROIs.

This is NOT about classification - it's about detecting whether MCI contains
distinct dynamical regimes and whether they localize to anterior/posterior regions.

Usage:
    python mci_subtype_analysis.py                    # Full analysis
    python mci_subtype_analysis.py --quick            # Quick test mode
    python mci_subtype_analysis.py --no-show          # Don't display plots
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CHECKPOINT_PATH, DATA_DIR, OUTPUT_DIR, DEVICE,
    FILTER_LOW, FILTER_HIGH, CHUNK_DURATION, SFREQ,
    ensure_output_dir, CONDITION_SUBDIRS,
)
from load_model import load_model_from_checkpoint, create_model, compute_latent_trajectory
from load_data import load_and_preprocess_file

# Import flow analysis functions from full_dataset_analysis
from full_dataset_analysis import (
    # Core data structures
    SubjectData, PooledEmbedder, FlowMetrics, BootstrapResult,
    # Metric computation
    compute_flow_metrics, compute_instantaneous_speed,
    compute_density_on_grid, compute_radial_profile, compute_radial_speed_profile,
    compute_group_flow_field, compute_flow_divergence, compute_flow_curl,
    compute_trajectory_curvature, compute_curvature_field,
    compute_speed_curvature_distribution, compute_dwell_time_field,
    compute_directional_entropy_field, compute_ftle_field,
    compute_temporal_heterogeneity_field, compute_flow_statistics,
    # Bootstrap functions
    bootstrap_flow_metrics, bootstrap_effect_size,
    bootstrap_density_difference, bootstrap_radial_profiles,
    # Plotting functions
    plot_bootstrap_metrics_comparison, plot_density_difference_with_ci,
    plot_radial_profiles, plot_effect_sizes,
    plot_group_flow_fields, plot_flow_difference,
    plot_curvature_analysis, plot_speed_curvature_phase,
    plot_dwell_time_fields, plot_directional_entropy,
    plot_ftle_analysis, plot_temporal_heterogeneity,
    plot_streamline_bundles,
    # Utilities
    create_timestamped_output_dir, create_square_subplots,
    get_group_color,
)

# =============================================================================
# PATHS
# =============================================================================

DEMOGRAPHICS_CSV = Path("/Volumes/Nvme_Data/GreekData/EEG_Demographics/Sheet1-Table 1.csv")
MCI_DATA_DIR = Path("/Volumes/Nvme_Data/GreekData/MCI-RAW/FILT")


# =============================================================================
# DEMOGRAPHICS LOADING AND CLEANING
# =============================================================================

def load_demographics(csv_path: Path = DEMOGRAPHICS_CSV) -> pd.DataFrame:
    """
    Load and clean the demographics CSV.

    Handles:
    - Semicolon delimiter
    - First row is junk (skip it)
    - Greek character variants in CONDC
    - Comma as decimal separator in some fields
    """
    # Read with proper settings
    df = pd.read_csv(csv_path, sep=";", skiprows=1, encoding="utf-8")

    # Drop fully empty columns (Unnamed:*)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df.dropna(how="all", axis=1)

    # Clean up column names
    df.columns = df.columns.str.strip()

    # Normalize CONDC: handle Greek 'Μ' (capital mu) vs Latin 'M'
    if "CONDC" in df.columns:
        df["CONDC_original"] = df["CONDC"]
        # Replace Greek Μ with Latin M, normalize whitespace
        df["CONDC"] = df["CONDC"].str.replace("Μ", "M", regex=False)
        df["CONDC"] = df["CONDC"].str.strip()

    # Parse numeric fields with comma as decimal separator
    numeric_cols = ["AGE", "EDU", "MMSE", "MoCA", "NPI", "FRSSD tot", "FUCAS tot",
                    "trailb", "RBMT1", "RBMT2", "rocft", "ROCFT2",
                    "ravlt1", "ravlt2", "ravlt3", "ravlt4", "ravlt5", "ravltt",
                    "ravlt delayed", "FAS", "PSS", "BDI", "GDS", "HAM"]

    for col in numeric_cols:
        if col in df.columns:
            # Handle comma decimal separator
            df[col] = df[col].astype(str).str.replace(",", ".").str.strip()
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean eeg code
    if "eeg code" in df.columns:
        df["eeg_code"] = df["eeg code"].str.strip().str.lower()

    return df


def filter_mci_subjects(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to MCI subjects only and add derived columns.

    Creates:
    - depression_comorbid: True if "depression" in CONDC
    - mci_group: Normalized MCI label
    """
    # MCI variants: "MCI", "ΜCI" (Greek M), "MCI + depression"
    mci_mask = df["CONDC"].str.upper().str.contains("MCI", na=False)
    mci_df = df[mci_mask].copy()

    # Add depression comorbidity flag
    mci_df["depression_comorbid"] = mci_df["CONDC"].str.lower().str.contains("depression", na=False)

    # Normalize group label
    mci_df["mci_group"] = "MCI"

    print(f"Found {len(mci_df)} MCI subjects")
    print(f"  With depression: {mci_df['depression_comorbid'].sum()}")
    print(f"  Without depression: {(~mci_df['depression_comorbid']).sum()}")

    return mci_df


def compute_missingness_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Compute missingness statistics for specified columns."""
    stats = []
    for col in columns:
        if col in df.columns:
            n_total = len(df)
            n_valid = df[col].notna().sum()
            n_missing = n_total - n_valid
            pct_missing = 100 * n_missing / n_total
            stats.append({
                "Variable": col,
                "N Valid": n_valid,
                "N Missing": n_missing,
                "% Missing": f"{pct_missing:.1f}%"
            })
    return pd.DataFrame(stats)


# =============================================================================
# MCI STRATIFICATION SCHEMES
# =============================================================================

@dataclass
class StratificationScheme:
    """Definition of a stratification scheme."""
    name: str
    variable: str
    bins: list[tuple[float, float, str]]  # (low, high, label)
    min_n_per_group: int = 5


def create_mmse_strata(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Create MMSE-based severity strata.

    Clinical bins:
    - Normal: MMSE >= 27
    - Mild: MMSE 24-26
    - Moderate: MMSE <= 23

    Falls back to tertiles if clinical bins create imbalanced groups.
    """
    mmse = df["MMSE"]
    valid = mmse.notna()
    n_valid = valid.sum()

    if n_valid < 10:
        print(f"  MMSE: Only {n_valid} valid values, skipping")
        return None

    strata = pd.Series(index=df.index, dtype=object)

    # Try clinical bins first
    strata[mmse >= 27] = "MMSE_High"
    strata[(mmse >= 24) & (mmse < 27)] = "MMSE_Mid"
    strata[mmse < 24] = "MMSE_Low"

    # Check group sizes
    counts = strata.value_counts()
    print(f"  MMSE strata (clinical bins): {dict(counts)}")

    # If any group is too small, use tertiles instead
    if counts.min() < 5:
        print("  Falling back to tertiles...")
        tertiles = pd.qcut(mmse[valid], q=3, labels=["MMSE_Low", "MMSE_Mid", "MMSE_High"])
        strata = pd.Series(index=df.index, dtype=object)
        strata.loc[valid] = tertiles
        counts = strata.value_counts()
        print(f"  MMSE strata (tertiles): {dict(counts)}")

    return strata


def create_functional_strata(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Create functional impairment strata based on FUCAS or FRSSD.

    Uses FUCAS tot preferentially (higher = more impaired),
    falls back to FRSSD tot if FUCAS has too much missing data.
    """
    # Try FUCAS first (higher = more impaired)
    for var, ascending in [("FUCAS tot", False), ("FRSSD tot", False)]:
        if var not in df.columns:
            continue

        values = df[var]
        valid = values.notna()
        n_valid = valid.sum()

        if n_valid < 10:
            print(f"  {var}: Only {n_valid} valid values, trying next...")
            continue

        # Use tertiles
        try:
            tertiles = pd.qcut(values[valid], q=3, labels=["Func_Low", "Func_Mid", "Func_High"])
            strata = pd.Series(index=df.index, dtype=object)
            strata.loc[valid] = tertiles
            counts = strata.value_counts()
            print(f"  Functional strata ({var}, tertiles): {dict(counts)}")
            return strata
        except ValueError as e:
            print(f"  {var}: Cannot create tertiles ({e})")
            continue

    print("  No suitable functional variable found")
    return None


def create_depression_strata(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Create depression comorbidity strata.

    Based on the explicit "MCI + depression" label or mood scales.
    """
    if "depression_comorbid" not in df.columns:
        return None

    strata = pd.Series(index=df.index, dtype=object)
    strata[df["depression_comorbid"]] = "MCI_Dep"
    strata[~df["depression_comorbid"]] = "MCI_NoDep"

    counts = strata.value_counts()
    print(f"  Depression strata: {dict(counts)}")

    # Check if either group is too small
    if counts.min() < 3:
        print("  Depression strata too imbalanced, skipping")
        return None

    return strata


def create_all_strata(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Create all available stratification schemes."""
    print("\nCreating MCI stratification schemes...")

    strata = {}

    # MMSE severity
    mmse_strata = create_mmse_strata(df)
    if mmse_strata is not None:
        strata["mmse_severity"] = mmse_strata

    # Functional impairment
    func_strata = create_functional_strata(df)
    if func_strata is not None:
        strata["functional_impairment"] = func_strata

    # Depression comorbidity
    dep_strata = create_depression_strata(df)
    if dep_strata is not None:
        strata["depression_comorbid"] = dep_strata

    return strata


# =============================================================================
# EGI-256 ROI REGIONALIZATION
# =============================================================================

def get_egi256_roi_channels() -> dict[str, list[str]]:
    """
    Define ROIs for EGI HydroCel-256 using coordinate-based regionalization.

    Uses MNE's standard montage to get 3D electrode positions,
    then divides by geometry (X/Y/Z coordinates in head space):
    - X: left (-) / right (+)
    - Y: posterior (-) / anterior (+)
    - Z: inferior (-) / superior (+)

    Returns dict mapping ROI name to list of channel names.
    """
    import mne

    # Get standard montage
    montage = mne.channels.make_standard_montage("GSN-HydroCel-256")

    # Get positions as dict
    positions = montage.get_positions()["ch_pos"]

    # Convert to DataFrame for easier manipulation
    data = []
    for ch, pos in positions.items():
        data.append({"channel": ch, "x": pos[0], "y": pos[1], "z": pos[2]})
    pos_df = pd.DataFrame(data)

    # Compute thresholds based on distribution
    x_mid = pos_df["x"].median()
    y_mid = pos_df["y"].median()
    z_mid = pos_df["z"].median()

    # Define ROIs by coordinate criteria
    rois = {}

    # Frontal: anterior (high Y), exclude very inferior (temporal)
    frontal_mask = (pos_df["y"] > pos_df["y"].quantile(0.6)) & (pos_df["z"] > pos_df["z"].quantile(0.3))
    rois["frontal"] = pos_df.loc[frontal_mask, "channel"].tolist()

    # Occipital: posterior (low Y), exclude temporal
    occipital_mask = (pos_df["y"] < pos_df["y"].quantile(0.25)) & (pos_df["z"] > pos_df["z"].quantile(0.3))
    rois["occipital"] = pos_df.loc[occipital_mask, "channel"].tolist()

    # Parietal: posterior-central, superior
    parietal_mask = (
        (pos_df["y"] > pos_df["y"].quantile(0.25)) &
        (pos_df["y"] < pos_df["y"].quantile(0.6)) &
        (pos_df["z"] > pos_df["z"].quantile(0.5))
    )
    rois["parietal"] = pos_df.loc[parietal_mask, "channel"].tolist()

    # Temporal Left: inferior, left hemisphere
    temporal_l_mask = (
        (pos_df["x"] < pos_df["x"].quantile(0.3)) &
        (pos_df["z"] < pos_df["z"].quantile(0.4)) &
        (pos_df["y"] > pos_df["y"].quantile(0.2)) &
        (pos_df["y"] < pos_df["y"].quantile(0.7))
    )
    rois["temporal_L"] = pos_df.loc[temporal_l_mask, "channel"].tolist()

    # Temporal Right: inferior, right hemisphere
    temporal_r_mask = (
        (pos_df["x"] > pos_df["x"].quantile(0.7)) &
        (pos_df["z"] < pos_df["z"].quantile(0.4)) &
        (pos_df["y"] > pos_df["y"].quantile(0.2)) &
        (pos_df["y"] < pos_df["y"].quantile(0.7))
    )
    rois["temporal_R"] = pos_df.loc[temporal_r_mask, "channel"].tolist()

    # Midline: central strip
    midline_mask = (pos_df["x"].abs() < pos_df["x"].std() * 0.5)
    rois["midline"] = pos_df.loc[midline_mask, "channel"].tolist()

    # Central: top of head, central
    central_mask = (
        (pos_df["z"] > pos_df["z"].quantile(0.6)) &
        (pos_df["y"] > pos_df["y"].quantile(0.3)) &
        (pos_df["y"] < pos_df["y"].quantile(0.7))
    )
    rois["central"] = pos_df.loc[central_mask, "channel"].tolist()

    # Print summary
    print("\nROI Channel Counts (EGI-256 coordinate-based):")
    for roi, channels in rois.items():
        print(f"  {roi}: {len(channels)} channels")

    return rois


def save_roi_channels(rois: dict[str, list[str]], output_path: Path):
    """Save ROI channel definitions to JSON."""
    with open(output_path, "w") as f:
        json.dump(rois, f, indent=2)
    print(f"Saved ROI definitions to: {output_path}")


def visualize_rois(rois: dict[str, list[str]], output_path: Path, show_plot: bool = True):
    """Visualize ROI electrode locations on a 2D head map."""
    import mne

    montage = mne.channels.make_standard_montage("GSN-HydroCel-256")
    positions = montage.get_positions()["ch_pos"]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Color map for ROIs
    colors = {
        "frontal": "#e41a1c",
        "parietal": "#377eb8",
        "occipital": "#4daf4a",
        "temporal_L": "#984ea3",
        "temporal_R": "#ff7f00",
        "central": "#ffff33",
        "midline": "#a65628",
    }

    # Plot each ROI
    for roi, channels in rois.items():
        x_coords = []
        y_coords = []
        for ch in channels:
            if ch in positions:
                pos = positions[ch]
                x_coords.append(pos[0])
                y_coords.append(pos[1])

        color = colors.get(roi, "#999999")
        ax.scatter(x_coords, y_coords, c=color, label=f"{roi} ({len(channels)})",
                   alpha=0.7, s=50, edgecolors="black", linewidth=0.5)

    # Draw head outline
    theta = np.linspace(0, 2 * np.pi, 100)
    head_radius = 0.095
    ax.plot(head_radius * np.cos(theta), head_radius * np.sin(theta),
            "k-", linewidth=2)

    # Nose marker
    ax.plot([0, 0], [head_radius, head_radius + 0.01], "k-", linewidth=2)

    ax.set_xlim(-0.12, 0.12)
    ax.set_ylim(-0.12, 0.12)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("EGI-256 ROI Regionalization\n(Coordinate-based)", fontsize=14)
    ax.axis("off")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved ROI visualization to: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# DATA LOADING FOR MCI SUBJECTS
# =============================================================================

def get_mci_eeg_files(mci_df: pd.DataFrame, data_dir: Path = MCI_DATA_DIR) -> list[tuple[Path, dict]]:
    """
    Find EEG files matching MCI subjects in demographics.

    Returns list of (file_path, demographics_row) tuples.
    Only returns ONE file per subject (the main *_eeg.fif file, not epoch files).
    """
    # Get only top-level *_eeg.fif files (not subdirectory epoch files)
    fif_files = list(data_dir.glob("*_eeg.fif"))
    print(f"Found {len(fif_files)} main EEG files in {data_dir}")

    # Build lookup by eeg_code
    eeg_codes = set(mci_df["eeg_code"].dropna())

    # Track which subjects we've already matched
    matched_subjects = set()
    matched = []

    for fif_path in sorted(fif_files):  # Sort for reproducibility
        # Extract subject ID from path (e.g., i105 from I105_20150917_1202_eeg.fif)
        file_id = fif_path.stem.split("_")[0].lower()

        # Only take first file per subject
        if file_id in eeg_codes and file_id not in matched_subjects:
            matched_subjects.add(file_id)
            # Get demographics row
            row = mci_df[mci_df["eeg_code"] == file_id].iloc[0].to_dict()
            matched.append((fif_path, row))

    print(f"Matched {len(matched)} unique subjects to MCI demographics")
    return matched


def load_mci_subjects(
    matched_files: list[tuple[Path, dict]],
    model,
    model_info: dict,
    max_subjects: Optional[int] = None,
    max_chunks: int = 10,
) -> list[tuple[SubjectData, dict]]:
    """
    Load EEG data and compute trajectories for MCI subjects.

    Returns list of (SubjectData, demographics_dict) tuples.
    """
    if max_subjects:
        matched_files = matched_files[:max_subjects]

    subjects = []
    for fif_path, demographics in tqdm(matched_files, desc="Loading MCI subjects"):
        try:
            # Load and preprocess
            data = load_and_preprocess_file(
                fif_path, FILTER_LOW, FILTER_HIGH, CHUNK_DURATION,
                include_amplitude=model_info["include_amplitude"],
                verbose=False,
            )

            # Limit chunks
            chunks = data["chunks"][:max_chunks]
            if len(chunks) < 3:
                continue

            # Compute trajectory for each chunk and concatenate
            latents = []
            for chunk in chunks:
                latent = compute_latent_trajectory(model, chunk, DEVICE)
                latents.append(latent)
            trajectory = np.concatenate(latents, axis=0)

            subject_data = SubjectData(
                subject_id=data["subject_id"],
                trajectory=trajectory,
                group="MCI",
                label=1,  # MCI label (arbitrary since we're only analyzing MCI)
            )

            subjects.append((subject_data, demographics))

        except Exception as e:
            print(f"  Error loading {fif_path.name}: {e}")
            continue

    print(f"Successfully loaded {len(subjects)} MCI subjects")
    return subjects


# =============================================================================
# SUBGROUP ANALYSIS
# =============================================================================

def run_subgroup_analysis(
    subjects: list[tuple[SubjectData, dict]],
    strata: dict[str, pd.Series],
    output_dir: Path,
    methods: list[str] = ["pca", "tpca", "delay"],
    n_bootstrap: int = 500,
    tau: int = 5,
    delay_dim: int = 3,
    show_plot: bool = True,
) -> dict:
    """
    Run comprehensive flow analysis comparing MCI subgroups within each stratification scheme.

    Includes:
    - Bootstrap flow metrics
    - Density differences with CIs
    - Flow field differences
    - Radial profiles
    - Curvature analysis
    - Dwell-time fields
    - Directional entropy
    - Streamline bundles
    - Temporal heterogeneity
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "n_subjects": len(subjects),
        "strata_names": list(strata.keys()),
    }

    # Build subject ID to demographics mapping
    subject_demographics = {}
    for subject_data, demo in subjects:
        eeg_code = demo.get("eeg_code", subject_data.subject_id.lower())
        subject_demographics[eeg_code] = demo

    # Get all trajectories for pooled fitting
    all_trajectories = [s.trajectory for s, _ in subjects]

    for scheme_name, scheme_strata in strata.items():
        print(f"\n{'='*80}")
        print(f"STRATIFICATION: {scheme_name.upper()}")
        print(f"{'='*80}")

        # Group subjects by stratum
        subgroups = {}
        for subject_data, demo in subjects:
            eeg_code = demo.get("eeg_code", subject_data.subject_id.lower())

            # Get stratum from demographics
            stratum = demo.get(f"stratum_{scheme_name}")
            if stratum is None or pd.isna(stratum):
                continue

            if stratum not in subgroups:
                subgroups[stratum] = []
            subgroups[stratum].append(subject_data)

        # Report subgroup sizes
        print(f"\nSubgroup sizes:")
        for stratum, group_subjects in subgroups.items():
            print(f"  {stratum}: {len(group_subjects)} subjects")

        # Skip if not enough subgroups
        if len(subgroups) < 2:
            print("  Not enough subgroups for comparison, skipping")
            continue

        # Skip if any subgroup is too small
        min_size = min(len(s) for s in subgroups.values())
        if min_size < 3:
            print(f"  Smallest subgroup has only {min_size} subjects, skipping")
            continue

        # Convert subgroups to format expected by full_dataset_analysis functions
        # They expect dict[str, list[SubjectData]]
        subject_data_by_group = subgroups

        # Run analysis for each embedding method
        scheme_results = {}
        for method in methods:
            print(f"\n--- {method.upper()} ---")

            # Fit embedder on all data
            embedder = PooledEmbedder(method=method, tau=tau, delay_dim=delay_dim)
            embedder.fit(all_trajectories)
            embedding_name = embedder.get_method_name()
            suffix = f"{embedding_name}_{scheme_name}".lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")

            # ================================================================
            # 1. Bootstrap flow metrics
            # ================================================================
            print(f"\n  [1/8] Bootstrap flow metrics...")
            bootstrap_results = {}
            for stratum, group_subjects in subgroups.items():
                if len(group_subjects) < 3:
                    continue
                print(f"    {stratum}...", end=" ", flush=True)
                bootstrap_results[stratum] = bootstrap_flow_metrics(
                    group_subjects, embedder, n_bootstrap
                )
                print("done")

            if len(bootstrap_results) >= 2:
                plot_bootstrap_metrics_comparison(
                    bootstrap_results, output_dir, suffix, show_plot
                )

            # ================================================================
            # 2. Density differences with bootstrap CIs
            # ================================================================
            print(f"\n  [2/8] Density differences...")
            strata_list = list(subgroups.keys())
            for i, stratum_a in enumerate(strata_list):
                for stratum_b in strata_list[i+1:]:
                    if len(subgroups[stratum_a]) < 3 or len(subgroups[stratum_b]) < 3:
                        continue
                    print(f"    {stratum_a} vs {stratum_b}...", end=" ", flush=True)
                    mean_diff, ci_low, ci_high = bootstrap_density_difference(
                        subgroups[stratum_a], subgroups[stratum_b], embedder, n_bootstrap
                    )
                    print("done")
                    comparison_name = f"{stratum_a}_vs_{stratum_b}".lower()
                    plot_density_difference_with_ci(
                        mean_diff, ci_low, ci_high, embedder.bounds,
                        comparison_name, output_dir, suffix, show_plot
                    )

            # ================================================================
            # 3. Radial profiles with CIs
            # ================================================================
            print(f"\n  [3/8] Radial profiles...")
            group_profiles = {}
            for stratum, group_subjects in subgroups.items():
                if len(group_subjects) < 3:
                    continue
                print(f"    {stratum}...", end=" ", flush=True)
                bin_centers, density_ci, speed_ci = bootstrap_radial_profiles(
                    group_subjects, embedder, n_bootstrap
                )
                group_profiles[stratum] = (bin_centers, density_ci, speed_ci)
                print("done")

            if len(group_profiles) >= 2:
                plot_radial_profiles(group_profiles, output_dir, suffix, show_plot)

            # ================================================================
            # 4. Flow field analysis and differences
            # ================================================================
            print(f"\n  [4/8] Flow fields and differences...")
            plot_mci_flow_fields(subject_data_by_group, embedder, output_dir, suffix, show_plot)
            plot_mci_flow_difference(subject_data_by_group, embedder, output_dir, suffix, show_plot)

            # ================================================================
            # 5. Curvature analysis
            # ================================================================
            print(f"\n  [5/8] Curvature analysis...")
            plot_mci_curvature_analysis(subject_data_by_group, embedder, output_dir, suffix, show_plot)
            plot_mci_speed_curvature_phase(subject_data_by_group, embedder, output_dir, suffix, show_plot)

            # ================================================================
            # 6. Dwell-time fields
            # ================================================================
            print(f"\n  [6/8] Dwell-time fields...")
            plot_mci_dwell_time_fields(subject_data_by_group, embedder, output_dir, suffix, show_plot)

            # ================================================================
            # 7. Directional entropy
            # ================================================================
            print(f"\n  [7/8] Directional entropy...")
            plot_mci_directional_entropy(subject_data_by_group, embedder, output_dir, suffix, show_plot)

            # ================================================================
            # 8. Streamline bundles and temporal heterogeneity
            # ================================================================
            print(f"\n  [8/8] Streamlines and temporal heterogeneity...")
            plot_mci_streamline_bundles(subject_data_by_group, embedder, output_dir, suffix, show_plot)
            plot_mci_temporal_heterogeneity(subject_data_by_group, embedder, output_dir, suffix, show_plot)

            # ================================================================
            # Compute effect sizes between all pairs
            # ================================================================
            print(f"\n  Computing effect sizes...")
            effect_sizes = {}
            for i, stratum_a in enumerate(strata_list):
                for stratum_b in strata_list[i+1:]:
                    if len(subgroups[stratum_a]) < 3 or len(subgroups[stratum_b]) < 3:
                        continue
                    comparison = f"{stratum_a}_vs_{stratum_b}"
                    effect_sizes[comparison] = compute_subgroup_effect_sizes(
                        subgroups[stratum_a], subgroups[stratum_b],
                        embedder, n_bootstrap
                    )

            # Store results
            scheme_results[method] = {
                "bootstrap": {
                    s: {m: {"mean": r.mean, "ci_low": r.ci_low, "ci_high": r.ci_high}
                        for m, r in metrics.items()}
                    for s, metrics in bootstrap_results.items()
                },
                "effect_sizes": effect_sizes,
            }

        results[scheme_name] = scheme_results

    return results


# =============================================================================
# MCI-SPECIFIC PLOTTING WRAPPERS
# These wrap the full_dataset_analysis functions for MCI strata
# =============================================================================

def plot_mci_flow_fields(
    subject_data: dict[str, list[SubjectData]],
    embedder: PooledEmbedder,
    output_dir: Path,
    suffix: str,
    show_plot: bool = True,
    grid_size: int = 25,
):
    """Plot flow fields for each MCI stratum."""
    from matplotlib.colors import Normalize

    groups = list(subject_data.keys())
    n_groups = len(groups)

    if n_groups < 2:
        return

    # Compute flow fields
    flow_data = {}
    for group in groups:
        subjects = subject_data[group]
        X, Y, flow_x, flow_y, counts = compute_group_flow_field(subjects, embedder, grid_size)
        flow_data[group] = (X, Y, flow_x, flow_y, counts)

    # Plot
    fig, axes = plt.subplots(1, n_groups, figsize=(5*n_groups, 5))
    if n_groups == 1:
        axes = [axes]

    for idx, group in enumerate(groups):
        ax = axes[idx]
        X, Y, flow_x, flow_y, counts = flow_data[group]

        # Mask low-count bins
        mask = counts > 10
        speed = np.sqrt(flow_x**2 + flow_y**2)
        speed_masked = np.where(mask, speed, np.nan)

        # Background: speed magnitude
        im = ax.imshow(speed_masked, origin='lower', extent=list(embedder.bounds),
                       cmap='viridis', aspect='equal', alpha=0.7)

        # Quiver plot
        skip = 2
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                  np.where(mask[::skip, ::skip], flow_x[::skip, ::skip], 0),
                  np.where(mask[::skip, ::skip], flow_y[::skip, ::skip], 0),
                  color='white', alpha=0.8, scale=None)

        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_title(f"{group}\nFlow Field", fontweight='bold')

    plt.colorbar(im, ax=axes[-1], label='Speed', shrink=0.8)
    fig.suptitle(f"MCI Strata Flow Fields", fontsize=14, fontweight='bold')

    save_path = output_dir / f"flow_fields_{suffix}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"    Saved: {save_path.name}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return flow_data


def plot_mci_flow_difference(
    subject_data: dict[str, list[SubjectData]],
    embedder: PooledEmbedder,
    output_dir: Path,
    suffix: str,
    show_plot: bool = True,
    grid_size: int = 25,
):
    """Plot flow field differences between MCI strata pairs."""
    groups = list(subject_data.keys())
    n_groups = len(groups)

    if n_groups < 2:
        return

    # Compute flow fields
    flow_data = {}
    for group in groups:
        subjects = subject_data[group]
        X, Y, flow_x, flow_y, counts = compute_group_flow_field(subjects, embedder, grid_size)
        flow_data[group] = (X, Y, flow_x, flow_y, counts)

    # Number of pairwise comparisons
    n_pairs = n_groups * (n_groups - 1) // 2
    fig, axes = plt.subplots(1, n_pairs, figsize=(5*n_pairs, 5))
    if n_pairs == 1:
        axes = [axes]

    pair_idx = 0
    for i, group_a in enumerate(groups):
        for group_b in groups[i+1:]:
            ax = axes[pair_idx]

            X, Y, flow_a_x, flow_a_y, counts_a = flow_data[group_a]
            _, _, flow_b_x, flow_b_y, counts_b = flow_data[group_b]

            # Difference
            diff_x = flow_a_x - flow_b_x
            diff_y = flow_a_y - flow_b_y
            diff_mag = np.sqrt(diff_x**2 + diff_y**2)

            # Mask low-count bins
            valid = (counts_a > 10) & (counts_b > 10)
            diff_mag_masked = np.where(valid, diff_mag, np.nan)

            # Plot
            im = ax.imshow(diff_mag_masked, origin='lower', extent=list(embedder.bounds),
                           cmap='hot', aspect='equal')

            skip = 2
            ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                      np.where(valid[::skip, ::skip], diff_x[::skip, ::skip], 0),
                      np.where(valid[::skip, ::skip], diff_y[::skip, ::skip], 0),
                      color='cyan', alpha=0.7, scale=None)

            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
            ax.set_title(f"{group_a} − {group_b}\nFlow Difference", fontweight='bold')
            plt.colorbar(im, ax=ax, label='|Δ Flow|', shrink=0.8)

            pair_idx += 1

    fig.suptitle(f"MCI Strata Flow Differences", fontsize=14, fontweight='bold')

    save_path = output_dir / f"flow_difference_{suffix}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"    Saved: {save_path.name}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_mci_curvature_analysis(
    subject_data: dict[str, list[SubjectData]],
    embedder: PooledEmbedder,
    output_dir: Path,
    suffix: str,
    show_plot: bool = True,
    grid_size: int = 30,
):
    """Plot curvature maps for MCI strata and differences."""
    groups = list(subject_data.keys())
    n_groups = len(groups)

    if n_groups < 2:
        return

    extent = list(embedder.bounds)

    # Compute curvature fields
    curvature_data = {}
    for group in groups:
        curv_mean, curv_std, counts = compute_curvature_field(
            subject_data[group], embedder, embedder.bounds, grid_size
        )
        curvature_data[group] = (curv_mean, curv_std, counts)

    # Determine vmax for consistent colorscale
    valid_curvatures = []
    for group in groups:
        curv_mean, _, counts = curvature_data[group]
        valid_mask = counts > 10
        if valid_mask.any():
            valid_curvatures.extend(curv_mean[valid_mask].flatten())

    if not valid_curvatures:
        return
    vmax = np.percentile(valid_curvatures, 95)

    # Plot: individual maps + difference
    n_cols = n_groups + 1  # +1 for difference
    fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))

    for idx, group in enumerate(groups):
        ax = axes[idx]
        curv_mean, curv_std, counts = curvature_data[group]
        masked_curv = np.where(counts > 10, curv_mean, np.nan)

        im = ax.imshow(masked_curv, origin='lower', extent=extent, cmap='hot',
                       vmin=0, vmax=vmax, aspect='equal')
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_title(f"{group}\nMean Curvature", fontweight='bold')

    plt.colorbar(im, ax=axes[n_groups-1], label='Curvature', shrink=0.8)

    # Difference (first vs second group)
    if n_groups >= 2:
        ax = axes[n_groups]
        group_a, group_b = groups[0], groups[1]
        curv_a, _, counts_a = curvature_data[group_a]
        curv_b, _, counts_b = curvature_data[group_b]

        valid = (counts_a > 10) & (counts_b > 10)
        diff_curv = np.where(valid, curv_a - curv_b, np.nan)

        vmax_diff = np.nanpercentile(np.abs(diff_curv), 95) if not np.all(np.isnan(diff_curv)) else 1.0
        im = ax.imshow(diff_curv, origin='lower', extent=extent, cmap='RdBu_r',
                       vmin=-vmax_diff, vmax=vmax_diff, aspect='equal')
        plt.colorbar(im, ax=ax, label='Δ Curvature', shrink=0.8)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_title(f"{group_a} − {group_b}\nCurvature Diff", fontweight='bold')

    fig.suptitle(f"Trajectory Curvature Analysis", fontsize=14, fontweight='bold')

    save_path = output_dir / f"curvature_analysis_{suffix}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"    Saved: {save_path.name}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_mci_speed_curvature_phase(
    subject_data: dict[str, list[SubjectData]],
    embedder: PooledEmbedder,
    output_dir: Path,
    suffix: str,
    show_plot: bool = True,
):
    """Plot speed-curvature phase distributions for MCI strata."""
    groups = list(subject_data.keys())
    n_groups = len(groups)

    if n_groups < 2:
        return

    fig, axes = plt.subplots(1, n_groups, figsize=(5*n_groups, 5))
    if n_groups == 1:
        axes = [axes]

    for idx, group in enumerate(groups):
        ax = axes[idx]

        hist, speed_edges, curv_edges = compute_speed_curvature_distribution(
            subject_data[group], embedder
        )

        im = ax.imshow(hist.T, origin='lower', aspect='auto',
                       extent=[speed_edges[0], speed_edges[-1], curv_edges[0], curv_edges[-1]],
                       cmap='viridis')
        ax.set_xlabel("Speed")
        ax.set_ylabel("Curvature")
        ax.set_title(f"{group}\nSpeed-Curvature Phase", fontweight='bold')

    plt.colorbar(im, ax=axes[-1], label='Density', shrink=0.8)
    fig.suptitle(f"Speed-Curvature Phase Plots", fontsize=14, fontweight='bold')

    save_path = output_dir / f"speed_curvature_phase_{suffix}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"    Saved: {save_path.name}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_mci_dwell_time_fields(
    subject_data: dict[str, list[SubjectData]],
    embedder: PooledEmbedder,
    output_dir: Path,
    suffix: str,
    show_plot: bool = True,
    grid_size: int = 30,
):
    """Plot dwell-time spatial fields for MCI strata."""
    groups = list(subject_data.keys())
    n_groups = len(groups)

    if n_groups < 2:
        return

    extent = list(embedder.bounds)

    fig, axes = plt.subplots(1, n_groups, figsize=(5*n_groups, 5))
    if n_groups == 1:
        axes = [axes]

    dwell_data = {}
    for idx, group in enumerate(groups):
        ax = axes[idx]

        dwell_mean, dwell_std, counts = compute_dwell_time_field(
            subject_data[group], embedder, embedder.bounds, grid_size
        )
        dwell_data[group] = (dwell_mean, dwell_std, counts)

        masked_dwell = np.where(counts > 10, dwell_mean, np.nan)
        im = ax.imshow(masked_dwell, origin='lower', extent=extent, cmap='plasma',
                       aspect='equal')
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_title(f"{group}\nMean Dwell Time", fontweight='bold')

    plt.colorbar(im, ax=axes[-1], label='Dwell Time', shrink=0.8)
    fig.suptitle(f"Dwell-Time Spatial Fields", fontsize=14, fontweight='bold')

    save_path = output_dir / f"dwell_time_fields_{suffix}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"    Saved: {save_path.name}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_mci_directional_entropy(
    subject_data: dict[str, list[SubjectData]],
    embedder: PooledEmbedder,
    output_dir: Path,
    suffix: str,
    show_plot: bool = True,
    grid_size: int = 30,
):
    """Plot directional entropy fields for MCI strata."""
    groups = list(subject_data.keys())
    n_groups = len(groups)

    if n_groups < 2:
        return

    extent = list(embedder.bounds)

    fig, axes = plt.subplots(1, n_groups, figsize=(5*n_groups, 5))
    if n_groups == 1:
        axes = [axes]

    for idx, group in enumerate(groups):
        ax = axes[idx]

        entropy, counts = compute_directional_entropy_field(
            subject_data[group], embedder, embedder.bounds, grid_size
        )

        masked_entropy = np.where(counts > 10, entropy, np.nan)
        im = ax.imshow(masked_entropy, origin='lower', extent=extent, cmap='coolwarm',
                       aspect='equal')
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_title(f"{group}\nDirectional Entropy", fontweight='bold')

    plt.colorbar(im, ax=axes[-1], label='Entropy (bits)', shrink=0.8)
    fig.suptitle(f"Directional Entropy Fields", fontsize=14, fontweight='bold')

    save_path = output_dir / f"directional_entropy_{suffix}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"    Saved: {save_path.name}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_mci_streamline_bundles(
    subject_data: dict[str, list[SubjectData]],
    embedder: PooledEmbedder,
    output_dir: Path,
    suffix: str,
    show_plot: bool = True,
    grid_size: int = 20,
    n_streamlines: int = 100,
):
    """Plot streamline bundles for MCI strata."""
    groups = list(subject_data.keys())
    n_groups = len(groups)

    if n_groups < 2:
        return

    fig, axes = plt.subplots(1, n_groups, figsize=(5*n_groups, 5))
    if n_groups == 1:
        axes = [axes]

    for idx, group in enumerate(groups):
        ax = axes[idx]

        # Get flow field
        subjects = subject_data[group]
        X, Y, flow_x, flow_y, counts = compute_group_flow_field(subjects, embedder, grid_size)

        # Mask
        mask = counts > 5
        speed = np.sqrt(flow_x**2 + flow_y**2)

        # Background
        ax.imshow(np.where(mask, speed, np.nan), origin='lower',
                  extent=list(embedder.bounds), cmap='Greys', alpha=0.3, aspect='equal')

        # Streamlines
        try:
            ax.streamplot(X, Y, flow_x, flow_y, density=1.5, color='steelblue',
                          linewidth=0.5, arrowsize=0.5)
        except:
            pass  # streamplot can fail if flow is too sparse

        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_title(f"{group}\nFlow Streamlines", fontweight='bold')
        ax.set_xlim(embedder.bounds[0], embedder.bounds[1])
        ax.set_ylim(embedder.bounds[2], embedder.bounds[3])

    fig.suptitle(f"Streamline Bundles", fontsize=14, fontweight='bold')

    save_path = output_dir / f"streamline_bundles_{suffix}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"    Saved: {save_path.name}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_mci_temporal_heterogeneity(
    subject_data: dict[str, list[SubjectData]],
    embedder: PooledEmbedder,
    output_dir: Path,
    suffix: str,
    show_plot: bool = True,
    grid_size: int = 30,
):
    """Plot temporal heterogeneity (burstiness) for MCI strata."""
    groups = list(subject_data.keys())
    n_groups = len(groups)

    if n_groups < 2:
        return

    extent = list(embedder.bounds)

    fig, axes = plt.subplots(1, n_groups, figsize=(5*n_groups, 5))
    if n_groups == 1:
        axes = [axes]

    for idx, group in enumerate(groups):
        ax = axes[idx]

        # Returns (speed_cv, iei_mean) - 2 values
        speed_cv, iei_mean = compute_temporal_heterogeneity_field(
            subject_data[group], embedder, embedder.bounds, grid_size
        )

        # Plot speed CV (burstiness) - mask bins with NaN (insufficient data)
        masked_cv = np.where(np.isfinite(speed_cv), speed_cv, np.nan)
        im = ax.imshow(masked_cv, origin='lower', extent=extent, cmap='YlOrRd',
                       aspect='equal')
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_title(f"{group}\nSpeed CV (Burstiness)", fontweight='bold')

    plt.colorbar(im, ax=axes[-1], label='CV', shrink=0.8)
    fig.suptitle(f"Temporal Heterogeneity Maps", fontsize=14, fontweight='bold')

    save_path = output_dir / f"temporal_heterogeneity_{suffix}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"    Saved: {save_path.name}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def compute_subgroup_effect_sizes(
    group_a: list[SubjectData],
    group_b: list[SubjectData],
    embedder: PooledEmbedder,
    n_bootstrap: int = 500,
) -> dict[str, dict]:
    """Compute effect sizes between two subgroups."""
    metric_names = ["mean_speed", "speed_cv", "n_dwell_episodes",
                    "occupancy_entropy", "path_tortuosity", "explored_variance"]

    # Get metrics for each subject
    values_a = {m: [] for m in metric_names}
    values_b = {m: [] for m in metric_names}

    for subject in group_a:
        embedded = embedder.transform(subject.trajectory)
        metrics = compute_flow_metrics(embedded)
        for m in metric_names:
            values_a[m].append(getattr(metrics, m))

    for subject in group_b:
        embedded = embedder.transform(subject.trajectory)
        metrics = compute_flow_metrics(embedded)
        for m in metric_names:
            values_b[m].append(getattr(metrics, m))

    # Bootstrap effect sizes
    effect_sizes = {}
    for m in metric_names:
        a = np.array(values_a[m])
        b = np.array(values_b[m])
        if len(a) >= 3 and len(b) >= 3:
            effect_sizes[m] = bootstrap_effect_size(a, b, n_bootstrap)

    return {
        m: {"mean": r.mean, "ci_low": r.ci_low, "ci_high": r.ci_high}
        for m, r in effect_sizes.items()
    }


# =============================================================================
# ROI-SPECIFIC ANALYSIS
# =============================================================================

def get_channel_indices(channel_names: list[str], roi_channels: list[str]) -> list[int]:
    """Get indices of ROI channels in the full channel list."""
    indices = []
    for i, ch in enumerate(channel_names):
        if ch in roi_channels:
            indices.append(i)
    return indices


def compute_roi_power(
    raw_data: np.ndarray,
    sfreq: float,
    channel_indices: list[int],
    freq_bands: dict[str, tuple[float, float]] = None,
) -> dict[str, float]:
    """
    Compute band power for a specific ROI.

    Args:
        raw_data: (n_channels, n_samples) EEG data
        sfreq: Sampling frequency
        channel_indices: Indices of channels in this ROI
        freq_bands: Dict of band name -> (low, high) Hz

    Returns:
        Dict of band name -> mean power across ROI channels
    """
    from scipy.signal import welch

    if freq_bands is None:
        freq_bands = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 45),
        }

    # Extract ROI channels
    roi_data = raw_data[channel_indices, :]

    # Compute PSD for each channel
    powers = {band: [] for band in freq_bands}

    for ch_data in roi_data:
        freqs, psd = welch(ch_data, fs=sfreq, nperseg=min(len(ch_data), int(sfreq * 2)))

        for band, (low, high) in freq_bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            if band_mask.any():
                band_power = np.mean(psd[band_mask])
                powers[band].append(band_power)

    # Average across channels
    return {band: np.mean(vals) if vals else np.nan for band, vals in powers.items()}


def compute_roi_dynamics_correlation(
    trajectory: np.ndarray,
    roi_power_timeseries: np.ndarray,
) -> dict[str, float]:
    """
    Correlate latent trajectory dynamics with ROI power.

    Args:
        trajectory: (T, D) latent trajectory
        roi_power_timeseries: (T,) power timeseries for ROI

    Returns:
        Dict with correlation metrics
    """
    # Compute trajectory speed
    speed = compute_instantaneous_speed(trajectory)

    # Align lengths
    min_len = min(len(speed), len(roi_power_timeseries))
    speed = speed[:min_len]
    power = roi_power_timeseries[:min_len]

    # Correlation
    if len(speed) > 10 and np.std(speed) > 0 and np.std(power) > 0:
        corr, pval = spearmanr(speed, power)
        return {"correlation": corr, "pvalue": pval}
    return {"correlation": np.nan, "pvalue": np.nan}


def run_roi_comparison_analysis(
    matched_files: list[tuple[Path, dict]],
    subjects_with_strata: list[tuple[SubjectData, dict]],
    rois: dict[str, list[str]],
    strata: dict[str, pd.Series],
    output_dir: Path,
    n_bootstrap: int = 500,
    show_plot: bool = True,
) -> dict:
    """
    Run ROI-specific analysis comparing frontal vs posterior dynamics across MMSE strata.

    This analysis:
    1. Computes band power per ROI for each subject
    2. Compares ROI power across MMSE strata
    3. Tests frontal vs posterior dissociation
    """
    print("\n" + "="*80)
    print("ROI-SPECIFIC ANALYSIS: Frontal vs Posterior Dissociation")
    print("="*80)

    # Build mapping from subject_id to file path
    subject_to_file = {}
    for fif_path, demo in matched_files:
        eeg_code = demo.get("eeg_code", "").lower()
        subject_to_file[eeg_code] = fif_path

    # Key ROIs for frontal/posterior comparison
    key_rois = ["frontal", "parietal", "occipital", "temporal_L", "temporal_R"]
    available_rois = [r for r in key_rois if r in rois]

    if len(available_rois) < 2:
        print("  Not enough ROIs available for comparison")
        return {}

    # Collect ROI power per subject per stratum
    results = {}

    for scheme_name in strata.keys():
        print(f"\n--- Stratification: {scheme_name} ---")

        # Group subjects by stratum
        stratum_data = {}  # stratum -> list of {subject_id, roi_powers}

        for subject_data, demo in subjects_with_strata:
            stratum = demo.get(f"stratum_{scheme_name}")
            if stratum is None or pd.isna(stratum):
                continue

            eeg_code = demo.get("eeg_code", subject_data.subject_id.lower())
            if eeg_code not in subject_to_file:
                continue

            if stratum not in stratum_data:
                stratum_data[stratum] = []

            # Load raw data for this subject
            try:
                fif_path = subject_to_file[eeg_code]
                raw_data, sfreq, channel_names = load_eeg_from_file(fif_path, verbose=False)

                # Compute power for each ROI
                roi_powers = {}
                for roi_name in available_rois:
                    ch_indices = get_channel_indices(channel_names, rois[roi_name])
                    if len(ch_indices) > 0:
                        roi_powers[roi_name] = compute_roi_power(raw_data, sfreq, ch_indices)

                stratum_data[stratum].append({
                    "subject_id": eeg_code,
                    "roi_powers": roi_powers,
                })
            except Exception as e:
                print(f"    Error processing {eeg_code}: {e}")
                continue

        # Report sample sizes
        for stratum, subjects in stratum_data.items():
            print(f"  {stratum}: {len(subjects)} subjects with ROI data")

        # Skip if not enough data
        if len(stratum_data) < 2:
            continue
        if any(len(s) < 3 for s in stratum_data.values()):
            print("  Some strata have < 3 subjects, skipping")
            continue

        # Compute ROI power statistics per stratum
        roi_stats = {}
        freq_bands = ["delta", "theta", "alpha", "beta", "gamma"]

        for roi_name in available_rois:
            roi_stats[roi_name] = {}
            for band in freq_bands:
                roi_stats[roi_name][band] = {}
                for stratum, subjects in stratum_data.items():
                    powers = [s["roi_powers"].get(roi_name, {}).get(band, np.nan)
                              for s in subjects]
                    powers = [p for p in powers if np.isfinite(p)]
                    if powers:
                        roi_stats[roi_name][band][stratum] = {
                            "mean": np.mean(powers),
                            "std": np.std(powers),
                            "n": len(powers),
                        }

        # Plot ROI power comparison
        plot_roi_power_comparison(roi_stats, available_rois, freq_bands,
                                  output_dir, scheme_name, show_plot)

        # Compute frontal vs posterior asymmetry
        if "frontal" in available_rois and ("parietal" in available_rois or "occipital" in available_rois):
            posterior_roi = "parietal" if "parietal" in available_rois else "occipital"
            plot_frontal_posterior_asymmetry(stratum_data, rois, "frontal", posterior_roi,
                                             output_dir, scheme_name, show_plot)

        results[scheme_name] = {
            "roi_stats": roi_stats,
            "n_per_stratum": {s: len(subj) for s, subj in stratum_data.items()},
        }

    return results


def plot_roi_power_comparison(
    roi_stats: dict,
    roi_names: list[str],
    freq_bands: list[str],
    output_dir: Path,
    scheme_name: str,
    show_plot: bool = True,
):
    """Plot ROI power comparison across strata."""
    n_rois = len(roi_names)
    n_bands = len(freq_bands)

    fig, axes = plt.subplots(n_rois, n_bands, figsize=(3*n_bands, 3*n_rois))
    if n_rois == 1:
        axes = axes.reshape(1, -1)
    if n_bands == 1:
        axes = axes.reshape(-1, 1)

    colors = plt.cm.Set2(np.linspace(0, 1, 8))

    for i, roi in enumerate(roi_names):
        for j, band in enumerate(freq_bands):
            ax = axes[i, j]

            if roi in roi_stats and band in roi_stats[roi]:
                strata = list(roi_stats[roi][band].keys())
                means = [roi_stats[roi][band][s]["mean"] for s in strata]
                stds = [roi_stats[roi][band][s]["std"] for s in strata]

                x = np.arange(len(strata))
                bars = ax.bar(x, means, yerr=stds, capsize=3,
                              color=[colors[k % len(colors)] for k in range(len(strata))],
                              alpha=0.7, edgecolor='black')

                ax.set_xticks(x)
                ax.set_xticklabels(strata, rotation=45, ha='right', fontsize=8)

            if i == 0:
                ax.set_title(band.capitalize(), fontweight='bold')
            if j == 0:
                ax.set_ylabel(roi.replace("_", " ").title(), fontweight='bold')

    fig.suptitle(f"ROI Band Power by {scheme_name.replace('_', ' ').title()}\n(Mean ± SD)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = output_dir / f"roi_power_comparison_{scheme_name}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path.name}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_frontal_posterior_asymmetry(
    stratum_data: dict,
    rois: dict,
    frontal_roi: str,
    posterior_roi: str,
    output_dir: Path,
    scheme_name: str,
    show_plot: bool = True,
):
    """
    Plot frontal vs posterior asymmetry across MMSE strata.

    Tests whether MMSE effects are driven more by posterior vs frontal regions.
    """
    freq_bands = ["theta", "alpha", "beta"]
    strata = list(stratum_data.keys())

    fig, axes = plt.subplots(1, len(freq_bands), figsize=(5*len(freq_bands), 5))
    if len(freq_bands) == 1:
        axes = [axes]

    colors = {"frontal": "#e41a1c", "posterior": "#377eb8"}

    for j, band in enumerate(freq_bands):
        ax = axes[j]

        frontal_means = []
        frontal_sems = []
        posterior_means = []
        posterior_sems = []

        for stratum in strata:
            # Frontal power
            frontal_powers = [s["roi_powers"].get(frontal_roi, {}).get(band, np.nan)
                              for s in stratum_data[stratum]]
            frontal_powers = [p for p in frontal_powers if np.isfinite(p)]

            # Posterior power
            posterior_powers = [s["roi_powers"].get(posterior_roi, {}).get(band, np.nan)
                                for s in stratum_data[stratum]]
            posterior_powers = [p for p in posterior_powers if np.isfinite(p)]

            if frontal_powers:
                frontal_means.append(np.mean(frontal_powers))
                frontal_sems.append(np.std(frontal_powers) / np.sqrt(len(frontal_powers)))
            else:
                frontal_means.append(np.nan)
                frontal_sems.append(np.nan)

            if posterior_powers:
                posterior_means.append(np.mean(posterior_powers))
                posterior_sems.append(np.std(posterior_powers) / np.sqrt(len(posterior_powers)))
            else:
                posterior_means.append(np.nan)
                posterior_sems.append(np.nan)

        x = np.arange(len(strata))
        width = 0.35

        ax.bar(x - width/2, frontal_means, width, yerr=frontal_sems, capsize=3,
               label=f"Frontal", color=colors["frontal"], alpha=0.7)
        ax.bar(x + width/2, posterior_means, width, yerr=posterior_sems, capsize=3,
               label=f"Posterior ({posterior_roi})", color=colors["posterior"], alpha=0.7)

        ax.set_xlabel("MMSE Stratum")
        ax.set_ylabel(f"{band.capitalize()} Power (μV²/Hz)")
        ax.set_title(f"{band.capitalize()} Band", fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(strata, rotation=45, ha='right')
        ax.legend()

    fig.suptitle(f"Frontal vs Posterior Power Dissociation\n({scheme_name.replace('_', ' ').title()})",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = output_dir / f"frontal_posterior_asymmetry_{scheme_name}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path.name}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_roi_effect_sizes(
    stratum_data: dict,
    rois: dict,
    roi_names: list[str],
    output_dir: Path,
    scheme_name: str,
    n_bootstrap: int = 500,
    show_plot: bool = True,
):
    """
    Compute and plot effect sizes (Cohen's d) for ROI power differences between strata.

    Tests which ROIs show the strongest MMSE-related differences.
    """
    freq_bands = ["theta", "alpha", "beta"]
    strata = list(stratum_data.keys())

    if len(strata) < 2:
        return

    # Compare first two strata (e.g., MMSE_Low vs MMSE_High)
    stratum_a, stratum_b = strata[0], strata[1]

    # Compute effect sizes
    effect_sizes = {}
    for roi in roi_names:
        effect_sizes[roi] = {}
        for band in freq_bands:
            powers_a = [s["roi_powers"].get(roi, {}).get(band, np.nan)
                        for s in stratum_data[stratum_a]]
            powers_a = np.array([p for p in powers_a if np.isfinite(p)])

            powers_b = [s["roi_powers"].get(roi, {}).get(band, np.nan)
                        for s in stratum_data[stratum_b]]
            powers_b = np.array([p for p in powers_b if np.isfinite(p)])

            if len(powers_a) >= 3 and len(powers_b) >= 3:
                result = bootstrap_effect_size(powers_a, powers_b, n_bootstrap)
                effect_sizes[roi][band] = {
                    "d": result.mean,
                    "ci_low": result.ci_low,
                    "ci_high": result.ci_high,
                }

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(roi_names))
    width = 0.25
    offsets = np.linspace(-width, width, len(freq_bands))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(freq_bands)))

    for i, band in enumerate(freq_bands):
        ds = []
        ci_lows = []
        ci_highs = []
        for roi in roi_names:
            if roi in effect_sizes and band in effect_sizes[roi]:
                ds.append(effect_sizes[roi][band]["d"])
                ci_lows.append(effect_sizes[roi][band]["d"] - effect_sizes[roi][band]["ci_low"])
                ci_highs.append(effect_sizes[roi][band]["ci_high"] - effect_sizes[roi][band]["d"])
            else:
                ds.append(0)
                ci_lows.append(0)
                ci_highs.append(0)

        ax.bar(x + offsets[i], ds, width, yerr=[ci_lows, ci_highs], capsize=3,
               label=band.capitalize(), color=colors[i], alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=0.2, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axhline(y=-0.2, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axhline(y=-0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axhline(y=0.8, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axhline(y=-0.8, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.set_xlabel("ROI")
    ax.set_ylabel("Cohen's d (95% CI)")
    ax.set_title(f"ROI Power Effect Sizes: {stratum_a} vs {stratum_b}\n(Dashed lines: small/medium/large effect thresholds)",
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", " ").title() for r in roi_names], rotation=45, ha='right')
    ax.legend(title="Frequency Band")

    plt.tight_layout()

    save_path = output_dir / f"roi_effect_sizes_{scheme_name}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path.name}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return effect_sizes


# Import load_eeg_from_file for ROI analysis
from load_data import load_eeg_from_file


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MCI Subtype Analysis - Dynamical Geometry Within MCI Strata"
    )
    parser.add_argument("--n-subjects", type=int, default=None,
                        help="Max subjects to load (default: all)")
    parser.add_argument("--n-chunks", type=int, default=10,
                        help="Chunks per subject (default: 10)")
    parser.add_argument("--n-bootstrap", type=int, default=500,
                        help="Bootstrap iterations (default: 500)")
    parser.add_argument("--embedding", type=str, default="fast",
                        choices=["pca", "tpca", "delay", "fast", "all"],
                        help="Embedding method(s) (default: fast = pca+tpca+delay)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test mode (100 bootstrap, 10 subjects)")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't display plots interactively")
    args = parser.parse_args()

    # Set matplotlib backend
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
        print("Non-interactive mode: plots will be saved but not displayed")

    # Quick mode overrides
    if args.quick:
        args.n_bootstrap = 100
        args.n_subjects = 10
        print("QUICK MODE: 100 bootstrap, 10 subjects")

    # Create output directory
    base_output = ensure_output_dir()
    output_dir = create_timestamped_output_dir(base_output, "mci_subtype_analysis")
    print(f"Output directory: {output_dir}")

    # Determine methods
    if args.embedding == "fast":
        methods = ["pca", "tpca", "delay"]
    elif args.embedding == "all":
        methods = ["pca", "tpca", "delay", "diffusion"]
    else:
        methods = [args.embedding]

    # ==========================================================================
    # 1. LOAD AND CLEAN DEMOGRAPHICS
    # ==========================================================================
    print("\n" + "="*80)
    print("LOADING DEMOGRAPHICS")
    print("="*80)

    df = load_demographics()
    print(f"Loaded {len(df)} total subjects")

    mci_df = filter_mci_subjects(df)

    # Report missingness
    key_vars = ["MMSE", "MoCA", "FUCAS tot", "FRSSD tot", "NPI", "BDI", "GDS", "HAM", "EDU", "AGE"]
    missingness = compute_missingness_table(mci_df, key_vars)
    print("\nMissingness in MCI subjects:")
    print(missingness.to_string(index=False))
    missingness.to_csv(output_dir / "missingness_table.csv", index=False)

    # ==========================================================================
    # 2. CREATE STRATIFICATION SCHEMES
    # ==========================================================================
    print("\n" + "="*80)
    print("CREATING STRATIFICATION SCHEMES")
    print("="*80)

    strata = create_all_strata(mci_df)

    # Add strata to dataframe for later use
    for scheme_name, scheme_strata in strata.items():
        mci_df[f"stratum_{scheme_name}"] = scheme_strata

    # Save stratification summary
    strata_summary = {
        name: {k: int(v) for k, v in strata[name].value_counts().dropna().items()}
        for name in strata.keys()
    }
    with open(output_dir / "stratification_summary.json", "w") as f:
        json.dump(strata_summary, f, indent=2)

    # ==========================================================================
    # 3. CREATE ROI DEFINITIONS
    # ==========================================================================
    print("\n" + "="*80)
    print("CREATING ROI DEFINITIONS")
    print("="*80)

    rois = get_egi256_roi_channels()
    save_roi_channels(rois, output_dir / "roi_channels.json")
    visualize_rois(rois, output_dir / "roi_visualization.png", show_plot=not args.no_show)

    # ==========================================================================
    # 4. LOAD EEG DATA
    # ==========================================================================
    print("\n" + "="*80)
    print("LOADING EEG DATA")
    print("="*80)

    # Find matching files
    matched_files = get_mci_eeg_files(mci_df)

    if not matched_files:
        print("ERROR: No matching EEG files found!")
        return 1

    # Load model
    print("\nLoading model...")
    model_info = load_model_from_checkpoint(CHECKPOINT_PATH, DEVICE)

    # Get n_channels from first file
    first_data = load_and_preprocess_file(
        matched_files[0][0], FILTER_LOW, FILTER_HIGH, CHUNK_DURATION,
        include_amplitude=model_info["include_amplitude"],
        verbose=False,
    )
    model = create_model(first_data["n_channels"], model_info, DEVICE)

    # Load all subjects
    subjects = load_mci_subjects(
        matched_files, model, model_info,
        max_subjects=args.n_subjects,
        max_chunks=args.n_chunks,
    )

    if len(subjects) < 10:
        print(f"WARNING: Only {len(subjects)} subjects loaded, analysis may be unreliable")

    # ==========================================================================
    # 5. RUN SUBGROUP ANALYSIS
    # ==========================================================================
    print("\n" + "="*80)
    print("RUNNING SUBGROUP ANALYSIS")
    print("="*80)

    # We need to re-index strata to match loaded subjects
    # Build mapping from eeg_code to strata values
    eeg_to_strata = {}
    for idx, row in mci_df.iterrows():
        eeg_code = row.get("eeg_code")
        if pd.notna(eeg_code):
            eeg_to_strata[eeg_code] = {
                scheme: row.get(f"stratum_{scheme}")
                for scheme in strata.keys()
            }

    # Add strata info to demographics
    subjects_with_strata = []
    for subject_data, demo in subjects:
        eeg_code = demo.get("eeg_code", subject_data.subject_id.lower())
        if eeg_code in eeg_to_strata:
            demo.update({f"stratum_{k}": v for k, v in eeg_to_strata[eeg_code].items()})
        subjects_with_strata.append((subject_data, demo))

    # Run analysis
    results = run_subgroup_analysis(
        subjects_with_strata,
        strata,
        output_dir,
        methods=methods,
        n_bootstrap=args.n_bootstrap,
        show_plot=not args.no_show,
    )

    # Save results
    results_path = output_dir / "mci_subtype_results.json"

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return super().default(obj)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to: {results_path}")

    # ==========================================================================
    # 6. ROI-SPECIFIC ANALYSIS (Frontal vs Posterior Dissociation)
    # ==========================================================================
    print("\n" + "="*80)
    print("ROI-SPECIFIC ANALYSIS")
    print("="*80)

    roi_results = run_roi_comparison_analysis(
        matched_files,
        subjects_with_strata,
        rois,
        strata,
        output_dir,
        n_bootstrap=args.n_bootstrap,
        show_plot=not args.no_show,
    )

    # Save ROI results
    if roi_results:
        roi_results_path = output_dir / "roi_comparison_results.json"
        with open(roi_results_path, "w") as f:
            json.dump(roi_results, f, indent=2, cls=NumpyEncoder)
        print(f"\nROI results saved to: {roi_results_path}")

    # ==========================================================================
    # 7. SUMMARY
    # ==========================================================================
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Files generated:")
    for f in sorted(output_dir.iterdir()):
        print(f"  - {f.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
