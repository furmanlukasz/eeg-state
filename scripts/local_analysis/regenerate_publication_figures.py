"""
regenerate_publication_figures.py

Regenerate all simulation figures from saved results with larger text
for two-column journal publication. Uses the same data/model/trajectories
as the original run — only font sizes change.

Generates 6 figures:
  1-5) Original coupled oscillator analysis figures
  6)   Kinetic energy analysis figure

Usage:
    python regenerate_publication_figures.py \
        --input-dir results/simulations/coupled_sl_20260203_204453 \
        --output-dir results/simulations/coupled_sl_20260203_204453_publication \
        --ke-input-dir results/kinetic_energy/analysis_20260204_225237
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Ensure sibling modules are importable
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from coupled_oscillator_sim import (
    CoupledStuartLandauNetwork,
    analyze_topology_spectra,
)
from simulation_analysis import (
    compute_flow_field,
    compute_density_on_grid,
    SFREQ,
)


# ---------------------------------------------------------------------------
# Publication rcParams
# ---------------------------------------------------------------------------
def set_publication_style():
    """Set matplotlib rcParams for publication-quality figures."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 14,
        "figure.titlesize": 20,
        "figure.dpi": 150,
    })


# ---------------------------------------------------------------------------
# Helper: reconstruct PooledEmbedder bounds from embedded data
# ---------------------------------------------------------------------------
def reconstruct_bounds(embedded: np.ndarray, margin: float = 0.05):
    """Reconstruct bounds using the same formula as PooledEmbedder."""
    centroid = embedded.mean(axis=0)
    max_dev = max(
        np.abs(embedded[:, 0] - centroid[0]).max(),
        np.abs(embedded[:, 1] - centroid[1]).max(),
    )
    half_size = max_dev * (1 + margin)
    return (
        centroid[0] - half_size,
        centroid[0] + half_size,
        centroid[1] - half_size,
        centroid[1] + half_size,
    )


# ---------------------------------------------------------------------------
# Helper: recompute per-window metrics for discriminability violin plots
# ---------------------------------------------------------------------------
def recompute_window_metrics(
    latent: np.ndarray,
    regime_labels: np.ndarray,
    regime_names_sequence: list[str],
    unique_regime_names: list[str],
    window_size: int = 50,
):
    """Recompute per-window speed/variance/tortuosity distributions."""
    window_metrics = {
        name: {"speed": [], "variance": [], "tortuosity": []}
        for name in unique_regime_names
    }

    for name in unique_regime_names:
        matching_ids = [i for i, n in enumerate(regime_names_sequence) if n == name]
        mask = np.isin(regime_labels, matching_ids)
        regime_latent = latent[mask]

        n_windows = len(regime_latent) // window_size
        for w in range(n_windows):
            window = regime_latent[w * window_size : (w + 1) * window_size]
            if len(window) < window_size:
                continue

            velocity = np.diff(window, axis=0)
            speeds = np.linalg.norm(velocity, axis=1)
            window_metrics[name]["speed"].append(float(np.mean(speeds)))
            window_metrics[name]["variance"].append(float(np.var(window, axis=0).sum()))

            path_len = speeds.sum()
            disp = np.linalg.norm(window[-1] - window[0])
            tort = path_len / (disp + 1e-8)
            window_metrics[name]["tortuosity"].append(float(tort))

    for name in unique_regime_names:
        for metric in window_metrics[name]:
            window_metrics[name][metric] = np.array(window_metrics[name][metric])

    return window_metrics


# ---------------------------------------------------------------------------
# Helper: recompute discriminability stats from window_metrics
# ---------------------------------------------------------------------------
def recompute_discriminability(window_metrics, unique_regime_names):
    from scipy.stats import f_oneway

    discriminability = {}
    for metric in ["speed", "variance", "tortuosity"]:
        groups = [window_metrics[name][metric] for name in unique_regime_names]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) >= 2 and all(len(g) > 1 for g in groups):
            f_stat, p_val = f_oneway(*groups)
            all_data = np.concatenate(groups)
            grand_mean = np.mean(all_data)
            ss_total = np.sum((all_data - grand_mean) ** 2)
            ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
            eta_sq = ss_between / ss_total if ss_total > 0 else 0

            discriminability[metric] = {
                "f_statistic": float(f_stat),
                "p_value": float(p_val),
                "eta_squared": float(eta_sq),
                "n_windows": [len(g) for g in groups],
            }

    return discriminability


# ---------------------------------------------------------------------------
# Helper: recompute field metrics (divergence, curl) on 2D embedded space
# ---------------------------------------------------------------------------
def compute_field_metrics(embedded_2d: np.ndarray, bounds: tuple, grid_size: int = 15) -> dict:
    x_min, x_max, y_min, y_max = bounds

    x_edges = np.linspace(x_min, x_max, grid_size + 1)
    y_edges = np.linspace(y_min, y_max, grid_size + 1)
    dx = (x_max - x_min) / grid_size
    dy = (y_max - y_min) / grid_size

    velocity = np.diff(embedded_2d, axis=0)
    positions = embedded_2d[:-1]

    flow_x = np.zeros((grid_size, grid_size))
    flow_y = np.zeros((grid_size, grid_size))
    counts = np.zeros((grid_size, grid_size))

    x_idx = np.clip(np.digitize(positions[:, 0], x_edges) - 1, 0, grid_size - 1)
    y_idx = np.clip(np.digitize(positions[:, 1], y_edges) - 1, 0, grid_size - 1)

    for i in range(len(positions)):
        xi, yi = x_idx[i], y_idx[i]
        flow_x[yi, xi] += velocity[i, 0]
        flow_y[yi, xi] += velocity[i, 1]
        counts[yi, xi] += 1

    mask = counts > 0
    flow_x[mask] /= counts[mask]
    flow_y[mask] /= counts[mask]

    dvx_dx = np.zeros_like(flow_x)
    dvy_dy = np.zeros_like(flow_y)
    dvx_dx[:, 1:-1] = (flow_x[:, 2:] - flow_x[:, :-2]) / (2 * dx)
    dvy_dy[1:-1, :] = (flow_y[2:, :] - flow_y[:-2, :]) / (2 * dy)
    divergence = dvx_dx + dvy_dy

    dvy_dx = np.zeros_like(flow_y)
    dvx_dy = np.zeros_like(flow_x)
    dvy_dx[:, 1:-1] = (flow_y[:, 2:] - flow_y[:, :-2]) / (2 * dx)
    dvx_dy[1:-1, :] = (flow_x[2:, :] - flow_x[:-2, :]) / (2 * dy)
    curl = dvy_dx - dvx_dy

    min_samples = 3
    valid_mask = counts >= min_samples

    if valid_mask.sum() > 0:
        div_valid = divergence[valid_mask]
        curl_valid = curl[valid_mask]
        return {
            "mean_divergence": float(np.mean(div_valid)),
            "mean_abs_curl": float(np.mean(np.abs(curl_valid))),
            "divergence_grid": divergence,
            "curl_grid": curl,
            "counts_grid": counts,
        }
    else:
        return {
            "mean_divergence": 0.0,
            "mean_abs_curl": 0.0,
            "divergence_grid": divergence,
            "curl_grid": curl,
            "counts_grid": counts,
        }


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Regenerate simulation figures with publication-quality font sizes."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent.parent
            / "results" / "simulations" / "coupled_sl_20260203_204453"
        ),
        help="Path to original results directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: <input-dir>_publication)",
    )
    parser.add_argument(
        "--ke-input-dir",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent.parent
            / "results" / "kinetic_energy" / "analysis_20260204_225237"
        ),
        help="Path to kinetic energy results directory",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if args.output_dir is None:
        output_dir = input_dir.parent / (input_dir.name + "_publication")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Regenerate Publication Figures")
    print("=" * 70)
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")

    # ------------------------------------------------------------------
    # Load saved data
    # ------------------------------------------------------------------
    with open(input_dir / "parameters.json") as f:
        params = json.load(f)
    with open(input_dir / "results.json") as f:
        results = json.load(f)

    traj = np.load(input_dir / "trajectories.npz")
    embedded = traj["embedded"]
    latent_raw = traj["latent"]
    regime_labels = traj["regime_labels"]

    seed = params["seed"]
    regime_names_sequence = results["regime_names"]       # with duplicates
    switch_times = results["switch_times"]
    unique_regime_names = list(dict.fromkeys(regime_names_sequence))
    regime_metrics = results["regime_metrics"]

    # Compute actual duration from the regime schedule (params may store the
    # unadjusted value, e.g. 180s, while the real schedule only covers 160s)
    per_regime = switch_times[1] - switch_times[0] if len(switch_times) > 1 else 10.0
    total_duration_s = switch_times[-1] + per_regime

    # Reproduce the same StandardScaler + clipping as the original pipeline
    # (the saved latent is the raw autoencoder output, NOT the clipped version)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    latent_scaled = scaler.fit_transform(latent_raw)
    p99 = np.percentile(np.abs(latent_scaled), 99)
    clip_threshold = max(3.0, p99)
    latent_clipped = np.clip(latent_scaled, -clip_threshold, clip_threshold)

    print(f"  Loaded embedded {embedded.shape}, latent {latent_raw.shape}")
    print(f"  Latent scaled & clipped (threshold={clip_threshold:.1f})")
    print(f"  Regimes: {unique_regime_names}, {len(regime_names_sequence)} segments")

    # ------------------------------------------------------------------
    # Set publication style BEFORE any figure creation
    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    set_publication_style()

    # ------------------------------------------------------------------
    # Reconstruct shared quantities
    # ------------------------------------------------------------------
    bounds = reconstruct_bounds(embedded)
    print(f"  Bounds: {bounds}")

    regime_colors = {
        "global": "#1f77b4",
        "cluster": "#ff7f0e",
        "sparse": "#2ca02c",
        "ring": "#d62728",
    }

    # ==================================================================
    # Figure 1: Electrode time series (needs simulation re-run)
    # ==================================================================
    print("\n--- Figure 1: Electrode time series ---")
    print("  Re-running simulation with seed={} ...".format(seed))

    net = CoupledStuartLandauNetwork(
        n_oscillators=30,
        n_channels=30,
        sfreq=SFREQ,
        seed=seed,
    )
    net.default_topologies(seed=seed)

    regime_names_order = ["global", "cluster", "sparse", "ring"]
    per_regime = 10.0
    n_cycles = int(total_duration_s / (4 * per_regime))
    schedule = []
    for _ in range(n_cycles):
        for name in regime_names_order:
            schedule.append((name, per_regime))
    actual_duration = per_regime * 4 * n_cycles

    sim_result = net.generate(
        total_duration_s=actual_duration,
        regime_schedule=schedule,
        mu_mean=1.0,
        mu_std=0.2,
        omega_mean_hz=10.0,
        omega_std_hz=2.0,
        coupling_strength=params["coupling_strength"],
        noise_std=params["noise_std"],
        obs_noise_std=params["obs_noise_std"],
        obs_noise_color=params["obs_noise_color"],
        transition_s=params["transition_s"],
    )
    print(f"  Simulation done: {sim_result.y.shape[1]} samples")

    # --- Plot fig1 (adapted from plot_electrode_timeseries) ---
    from scipy.signal import hilbert as scipy_hilbert

    channels = [0, 5, 10, 20, 28]
    time_window = (0, min(60, actual_duration))
    nfft = 2048
    fs = SFREQ
    t = sim_result.t
    y = sim_result.y

    t0, t1 = time_window
    i0 = int(max(0, np.floor(t0 * fs)))
    i1 = int(min(y.shape[1], np.ceil(t1 * fs)))

    fig1 = plt.figure(figsize=(12, 11))
    gs1 = fig1.add_gridspec(3, 1, height_ratios=[2.2, 1.2, 1.2], hspace=0.65)

    # Panel A
    ax1 = fig1.add_subplot(gs1[0, 0])
    offset = 0.0
    for ch in channels:
        sig = y[ch, i0:i1]
        ax1.plot(t[i0:i1], sig + offset, lw=1.0)
        ax1.text(t0, offset, f"Ch{ch}", va="bottom", fontsize=13)
        offset += 2.5 * np.std(sig) + 1e-6

    for st in sim_result.switch_times:
        if t0 <= st <= t1:
            ax1.axvline(st, linestyle="--", linewidth=1)

    ax1.set_title("Raw time series (selected channels) with regime switches")
    ax1.set_xlabel("Time (s)")
    ax1.set_yticks([])

    # Panel B: PSD
    ax2 = fig1.add_subplot(gs1[1, 0])
    seg = y[:, i0:i1]
    freqs = np.fft.rfftfreq(nfft, d=1 / fs)
    Y_fft = np.fft.rfft(seg - seg.mean(axis=1, keepdims=True), n=nfft, axis=1)
    psd = (np.abs(Y_fft) ** 2).mean(axis=0)
    ax2.plot(freqs, psd)
    ax2.set_xlim(0, 60)
    ax2.set_title("Power spectral density (simple periodogram, mean across channels)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Power (a.u.)")

    # Panel C: Hilbert amplitude
    ax3 = fig1.add_subplot(gs1[2, 0])
    ch0 = channels[0]
    analytic = scipy_hilbert(y[ch0, i0:i1])
    amp = np.abs(analytic)
    ax3.plot(t[i0:i1], amp, lw=1.0)
    for st in sim_result.switch_times:
        if t0 <= st <= t1:
            ax3.axvline(st, linestyle="--", linewidth=1)
    ax3.set_title(f"Hilbert amplitude envelope (channel {ch0})")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Amplitude (a.u.)")

    fig1.savefig(output_dir / "fig_electrode_timeseries.png", dpi=150, bbox_inches="tight")
    fig1.savefig(output_dir / "fig_electrode_timeseries.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print("  Saved fig_electrode_timeseries.png/pdf")

    # ==================================================================
    # Figure 2: Main analysis (4 panels)
    # ==================================================================
    print("\n--- Figure 2: Main analysis ---")

    fig2 = plt.figure(figsize=(16, 12))
    gs2 = GridSpec(2, 2, figure=fig2, hspace=0.3, wspace=0.25)

    # Panel A: Ground-truth regime timeline
    ax_a = fig2.add_subplot(gs2[0, 0])
    for i, rname in enumerate(regime_names_sequence):
        start_time = switch_times[i]
        end_time = switch_times[i + 1] if i + 1 < len(switch_times) else total_duration_s
        color = regime_colors.get(rname, "#888888")
        ax_a.axvspan(start_time, end_time, color=color, alpha=0.7)

    ax_a.set_xlim(0, total_duration_s)
    ax_a.set_ylim(0, 1)
    ax_a.set_xlabel("Time (s)")
    ax_a.set_title("A) Ground-Truth Regime Sequence", fontweight="bold")
    ax_a.set_yticks([])

    # Panel B: Embedded trajectories
    ax_b = fig2.add_subplot(gs2[0, 1])
    step = max(1, len(embedded) // 5000)
    embedded_ds = embedded[::step]
    labels_ds = regime_labels[::step][: len(embedded_ds)]
    for name in unique_regime_names:
        matching_ids = [i for i, n in enumerate(regime_names_sequence) if n == name]
        mask = np.isin(labels_ds, matching_ids)
        color = regime_colors.get(name, "#888888")
        ax_b.scatter(embedded_ds[mask, 0], embedded_ds[mask, 1], c=color, s=2, alpha=0.4, label=name)
    ax_b.set_xlabel("Dim 1")
    ax_b.set_ylabel("Dim 2")
    ax_b.set_title("B) Embedded Trajectories (colored by regime)", fontweight="bold")
    ax_b.legend(markerscale=3)
    ax_b.set_aspect("equal")

    # Panel C: Density + Flow field
    ax_c = fig2.add_subplot(gs2[1, 0])
    density = compute_density_on_grid(embedded, bounds, bins=50)
    X, Y_grid, flow_x, flow_y, counts = compute_flow_field(embedded, bounds, grid_size=15)
    ax_c.imshow(density, origin="lower", extent=list(bounds), cmap="Blues", alpha=0.7, aspect="equal")
    ff_mask = counts > 5
    if ff_mask.any():
        mag = np.sqrt(flow_x[ff_mask] ** 2 + flow_y[ff_mask] ** 2)
        norm_fx = np.where(mag > 0, flow_x[ff_mask] / mag, 0)
        norm_fy = np.where(mag > 0, flow_y[ff_mask] / mag, 0)
        ax_c.quiver(
            X[ff_mask], Y_grid[ff_mask], norm_fx, norm_fy, mag,
            cmap="inferno", alpha=0.85, scale=25, width=0.004, headwidth=4, headlength=5,
        )
    ax_c.set_xlabel("Dim 1")
    ax_c.set_ylabel("Dim 2")
    ax_c.set_title("C) Density + Flow Field", fontweight="bold")

    # Panel D: Metric comparison
    ax_d = fig2.add_subplot(gs2[1, 1])
    metric_names = ["mean_speed", "speed_cv", "median_tortuosity", "explored_variance"]
    metric_labels = ["Speed", "Speed CV", "Tortuosity", "Variance"]
    x = np.arange(len(unique_regime_names))
    width = 0.18
    for j, (metric, label) in enumerate(zip(metric_names, metric_labels)):
        values = [regime_metrics.get(name, {}).get(metric, 0) for name in unique_regime_names]
        max_val = max(values) if values else 1
        norm_values = [v / max_val if max_val > 0 else 0 for v in values]
        bar_offset = (j - 1.5) * width
        ax_d.bar(x + bar_offset, norm_values, width, label=label, alpha=0.8)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(unique_regime_names)
    ax_d.set_ylabel("Normalized Value")
    ax_d.set_title("D) Flow Metrics by Regime", fontweight="bold")
    ax_d.legend(loc="upper right")

    fig2.suptitle(
        "Coupled Stuart-Landau Network: Dynamical Microscope Analysis",
        fontsize=20, fontweight="bold",
    )
    fig2.savefig(output_dir / "fig_analysis_main.png", dpi=150, bbox_inches="tight")
    fig2.savefig(output_dir / "fig_analysis_main.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print("  Saved fig_analysis_main.png/pdf")

    # ==================================================================
    # Figure 3: Discriminability (violin plots + effect sizes)
    # ==================================================================
    print("\n--- Figure 3: Discriminability ---")

    # Recompute per-window metrics from latent_clipped (matches original pipeline)
    window_metrics = recompute_window_metrics(
        latent_clipped, regime_labels, regime_names_sequence, unique_regime_names,
    )
    discriminability = recompute_discriminability(window_metrics, unique_regime_names)

    fig3, axes3 = plt.subplots(1, 3, figsize=(14, 5))
    metric_titles = {
        "speed": "Speed (latent units/step)",
        "variance": "Explored Variance",
        "tortuosity": "Path Tortuosity",
    }

    for ax, metric in zip(axes3, ["speed", "variance", "tortuosity"]):
        data_for_plot = []
        positions = []
        colors_for_plot = []
        for i, name in enumerate(unique_regime_names):
            vals = window_metrics[name][metric]
            if len(vals) > 0:
                data_for_plot.append(vals)
                positions.append(i)
                colors_for_plot.append(regime_colors.get(name, "#888888"))

        if data_for_plot:
            parts = ax.violinplot(data_for_plot, positions=positions, showmeans=True, showmedians=True)
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(colors_for_plot[i])
                pc.set_alpha(0.7)
            for partname in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
                if partname in parts:
                    parts[partname].set_color("black")
                    parts[partname].set_linewidth(1)

        ax.set_xticks(range(len(unique_regime_names)))
        ax.set_xticklabels(unique_regime_names)
        ax.set_title(metric_titles[metric], fontweight="bold")
        ax.set_ylabel("Value")

        if metric in discriminability:
            eta_sq = discriminability[metric]["eta_squared"]
            f_stat = discriminability[metric]["f_statistic"]
            p_val = discriminability[metric]["p_value"]
            effect_label = "large" if eta_sq > 0.14 else "medium" if eta_sq > 0.06 else "small"
            sig_str = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            ax.text(
                0.02, 0.98,
                f"\u03b7\u00b2={eta_sq:.3f} ({effect_label})\nF={f_stat:.1f} {sig_str}",
                transform=ax.transAxes, va="top", ha="left", fontsize=13,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

    fig3.suptitle(
        "Regime Discriminability: Per-Window Metric Distributions",
        fontsize=20, fontweight="bold",
    )
    fig3.tight_layout()
    fig3.savefig(output_dir / "fig_discriminability.png", dpi=150, bbox_inches="tight")
    fig3.savefig(output_dir / "fig_discriminability.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig3)
    print("  Saved fig_discriminability.png/pdf")

    # ==================================================================
    # Figure 4: Regime-specific flow fields (2x2)
    # ==================================================================
    print("\n--- Figure 4: Regime-specific flow fields ---")

    fig4, axes4 = plt.subplots(2, 2, figsize=(12, 12))
    axes4_flat = axes4.flatten()

    for idx, name in enumerate(unique_regime_names[:4]):
        ax = axes4_flat[idx]

        matching_ids = [i for i, n in enumerate(regime_names_sequence) if n == name]
        mask = np.isin(regime_labels, matching_ids)
        regime_embedded = embedded[mask]

        if len(regime_embedded) > 100:
            regime_density = compute_density_on_grid(regime_embedded, bounds, bins=50)
            ax.imshow(
                regime_density, origin="lower", extent=list(bounds),
                cmap="Blues", alpha=0.6, aspect="equal",
            )

            X_r, Y_r, flow_x_r, flow_y_r, counts_r = compute_flow_field(
                regime_embedded, bounds, grid_size=15,
            )
            ff_mask_r = counts_r > 3
            if ff_mask_r.any():
                mag_r = np.sqrt(flow_x_r[ff_mask_r] ** 2 + flow_y_r[ff_mask_r] ** 2)
                norm_fx_r = np.where(mag_r > 0, flow_x_r[ff_mask_r] / mag_r, 0)
                norm_fy_r = np.where(mag_r > 0, flow_y_r[ff_mask_r] / mag_r, 0)
                ax.quiver(
                    X_r[ff_mask_r], Y_r[ff_mask_r], norm_fx_r, norm_fy_r, mag_r,
                    cmap="inferno", alpha=0.85, scale=25, width=0.005, headwidth=4, headlength=5,
                )

            fm = compute_field_metrics(regime_embedded, bounds, grid_size=15)
            ax.text(
                0.02, 0.98,
                f"div: {fm['mean_divergence']:.3f}\ncurl: {fm['mean_abs_curl']:.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=13,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        ax.set_title(
            f"{name.capitalize()}", fontweight="bold",
            color=regime_colors.get(name, "black"),
        )
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_aspect("equal")

    fig4.suptitle(
        "Regime-Specific Flow Fields (Density + Velocity)",
        fontsize=20, fontweight="bold",
    )
    fig4.tight_layout()
    fig4.savefig(output_dir / "fig_flow_fields.png", dpi=150, bbox_inches="tight")
    fig4.savefig(output_dir / "fig_flow_fields.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig4)
    print("  Saved fig_flow_fields.png/pdf")

    # ==================================================================
    # Figure 5: Laplacian eigenvalue spectra
    # ==================================================================
    print("\n--- Figure 5: Laplacian spectra ---")
    print("  Using normalized topology matrices from simulation network ...")

    # Reuse the net object (already created for fig1 with correct normalization)
    topology_spectra = analyze_topology_spectra(net._topologies, net._laplacians)

    fig5, axes5 = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Eigenvalue spectra
    ax5a = axes5[0]
    for name in unique_regime_names:
        if name in topology_spectra:
            eigs = topology_spectra[name]["eigenvalues"]
            ax5a.plot(
                range(len(eigs)), eigs, "o-", label=name,
                color=regime_colors.get(name, "#888888"),
                markersize=4, alpha=0.8,
            )
    ax5a.set_xlabel("Eigenvalue Index")
    ax5a.set_ylabel("Eigenvalue (\u03bb)")
    ax5a.set_title("A) Laplacian Eigenvalue Spectra", fontweight="bold")
    ax5a.legend()
    ax5a.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Panel B: Summary metrics comparison
    ax5b = axes5[1]
    metrics_to_plot = ["lambda_2", "spectral_gap", "density"]
    metric_labels_5 = [
        "\u03bb\u2082 (Algebraic\nConnectivity)",
        "Spectral Gap\n(\u03bb\u2082/\u03bb_max)",
        "Edge Density",
    ]
    x5 = np.arange(len(unique_regime_names))
    width5 = 0.25

    for j, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels_5)):
        values = [topology_spectra.get(name, {}).get(metric, 0) for name in unique_regime_names]
        max_val = max(values) if values and max(values) > 0 else 1
        norm_values = [v / max_val for v in values]
        bar_offset = (j - 1) * width5
        ax5b.bar(x5 + bar_offset, norm_values, width5, label=label, alpha=0.8)

    ax5b.set_xticks(x5)
    ax5b.set_xticklabels(unique_regime_names)
    ax5b.set_ylabel("Normalized Value")
    ax5b.set_title("B) Topology Spectral Properties", fontweight="bold")
    ax5b.legend(loc="upper right")

    fig5.suptitle(
        "Laplacian Spectral Analysis: Topology Verification",
        fontsize=20, fontweight="bold",
    )
    fig5.tight_layout()
    fig5.savefig(output_dir / "fig_laplacian_spectra.png", dpi=150, bbox_inches="tight")
    fig5.savefig(output_dir / "fig_laplacian_spectra.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig5)
    print("  Saved fig_laplacian_spectra.png/pdf")

    # ==================================================================
    # Figure 6: Kinetic Energy Analysis
    # ==================================================================
    ke_input_dir = Path(args.ke_input_dir)
    if ke_input_dir.exists():
        print("\n--- Figure 6: Kinetic energy analysis ---")

        ke_data = np.load(ke_input_dir / "energy_timeseries.npz")
        energy = ke_data["energy"]
        velocity_norm = ke_data["velocity_norm"]
        ke_regime_labels = ke_data["regime_labels"]

        with open(ke_input_dir / "kinetic_energy_results.json") as f:
            ke_results = json.load(f)

        ke_sfreq = ke_results["parameters"]["sfreq"]
        ke_savgol_window = ke_results["parameters"]["savgol_window"]

        # Trim adjustment: energy_timeseries was computed with savgol trim
        trim_n_ke = ke_savgol_window * 2
        embedded_for_ke = embedded[trim_n_ke:trim_n_ke + len(energy)]

        ke_time = np.arange(len(energy)) / ke_sfreq

        from scipy.ndimage import uniform_filter1d

        fig6 = plt.figure(figsize=(16, 12))
        gs6 = GridSpec(2, 2, figure=fig6, hspace=0.3, wspace=0.3)

        # Panel A: Energy time series with regime coloring
        ax6a = fig6.add_subplot(gs6[0, 0])
        for name in unique_regime_names:
            matching_ids = [i for i, n in enumerate(regime_names_sequence) if n == name]
            mask = np.isin(ke_regime_labels, matching_ids)
            color = regime_colors.get(name, "#888888")
            ax6a.scatter(ke_time[mask], energy[mask], c=color, s=1, alpha=0.3, label=name)

        smoothed = uniform_filter1d(energy, size=int(ke_sfreq * 0.5))
        ax6a.plot(ke_time, smoothed, "k-", lw=1.5, alpha=0.8, label="Smoothed")
        ax6a.set_xlabel("Time (s)")
        ax6a.set_ylabel(r"Kinetic Energy ($\|\mathbf{v}\|^2$)")
        ax6a.set_title("A) Kinetic Energy Time Series", fontweight="bold")
        ax6a.legend(markerscale=5)

        # Panel B: Energy distribution per regime (violin plots)
        ax6b = fig6.add_subplot(gs6[0, 1])
        data_for_violin = []
        violin_positions = []
        violin_colors = []
        for i, name in enumerate(unique_regime_names):
            matching_ids = [j for j, n in enumerate(regime_names_sequence) if n == name]
            mask = np.isin(ke_regime_labels, matching_ids)
            regime_energy = energy[mask]
            if len(regime_energy) > 0:
                data_for_violin.append(regime_energy)
                violin_positions.append(i)
                violin_colors.append(regime_colors.get(name, "#888888"))

        if data_for_violin:
            parts = ax6b.violinplot(
                data_for_violin, positions=violin_positions,
                showmeans=True, showmedians=True,
            )
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(violin_colors[i])
                pc.set_alpha(0.7)
            for partname in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
                if partname in parts:
                    parts[partname].set_color("black")

        ax6b.set_xticks(range(len(unique_regime_names)))
        ax6b.set_xticklabels(unique_regime_names)
        ax6b.set_ylabel(r"Kinetic Energy ($\|\mathbf{v}\|^2$)")
        ax6b.set_title("B) Energy Distribution by Regime", fontweight="bold")

        # Add discriminability stats
        ke_disc = ke_results.get("discriminability", {})
        if "eta_squared" in ke_disc:
            eta_sq_ke = ke_disc["eta_squared"]
            f_stat_ke = ke_disc["f_statistic"]
            p_val_ke = ke_disc["p_value"]
            effect_ke = ke_disc.get("effect_size", "")
            sig_str_ke = "***" if p_val_ke < 0.001 else "**" if p_val_ke < 0.01 else "*" if p_val_ke < 0.05 else "ns"
            ax6b.text(
                0.02, 0.98,
                f"\u03b7\u00b2={eta_sq_ke:.3f} ({effect_ke})\nF={f_stat_ke:.1f} {sig_str_ke}",
                transform=ax6b.transAxes, va="top", ha="left", fontsize=13,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

        # Panel C: Energy landscape on 2D embedding
        ax6c = fig6.add_subplot(gs6[1, 0])

        # Compute energy landscape from saved data
        ke_bounds = ke_results.get("landscape", {}).get("bounds", None)
        if ke_bounds is None:
            ke_bounds = list(reconstruct_bounds(embedded_for_ke))
        ke_grid_size = 20

        xmin_ke, xmax_ke, ymin_ke, ymax_ke = ke_bounds
        x_edges_ke = np.linspace(xmin_ke, xmax_ke, ke_grid_size + 1)
        y_edges_ke = np.linspace(ymin_ke, ymax_ke, ke_grid_size + 1)

        # Compute embedded-space velocity for energy landscape
        from scipy.signal import savgol_filter as sg_filter
        emb_vel_ke = sg_filter(embedded_for_ke, ke_savgol_window, 2, deriv=1, axis=0, mode="interp")
        emb_energy_ke = np.sum(emb_vel_ke ** 2, axis=1)

        mean_grid_ke = np.full((ke_grid_size, ke_grid_size), np.nan)
        x_idx_ke = np.clip(np.digitize(embedded_for_ke[:, 0], x_edges_ke) - 1, 0, ke_grid_size - 1)
        y_idx_ke = np.clip(np.digitize(embedded_for_ke[:, 1], y_edges_ke) - 1, 0, ke_grid_size - 1)

        for i in range(ke_grid_size):
            for j in range(ke_grid_size):
                mask = (x_idx_ke == i) & (y_idx_ke == j)
                if mask.sum() >= 5:
                    mean_grid_ke[j, i] = np.mean(emb_energy_ke[mask])

        im6c = ax6c.imshow(
            mean_grid_ke, origin="lower",
            extent=ke_bounds, cmap="YlOrRd", aspect="equal",
        )
        plt.colorbar(im6c, ax=ax6c, label="Mean Energy")

        step_ke = max(1, len(embedded_for_ke) // 2000)
        ax6c.scatter(
            embedded_for_ke[::step_ke, 0], embedded_for_ke[::step_ke, 1],
            c="blue", s=1, alpha=0.1,
        )
        ax6c.set_xlabel("Dim 1")
        ax6c.set_ylabel("Dim 2")
        ax6c.set_title("C) Energy Landscape on Manifold", fontweight="bold")

        # Panel D: Summary bar chart of metrics per regime
        ax6d = fig6.add_subplot(gs6[1, 1])
        ke_regime_metrics = ke_results.get("regime_metrics", {})

        x6 = np.arange(len(unique_regime_names))
        means_ke = [
            ke_regime_metrics.get(name, {}).get("mean_energy", 0)
            for name in unique_regime_names
        ]
        cvs_ke = [
            ke_regime_metrics.get(name, {}).get("cv_energy", 0)
            for name in unique_regime_names
        ]

        width6 = 0.35
        ax6d.bar(
            x6 - width6 / 2, means_ke, width6, label="Mean Energy",
            color=[regime_colors.get(n, "#888") for n in unique_regime_names], alpha=0.8,
        )

        ax6d2 = ax6d.twinx()
        ax6d2.bar(x6 + width6 / 2, cvs_ke, width6, label="CV", color="gray", alpha=0.5)

        ax6d.set_xticks(x6)
        ax6d.set_xticklabels(unique_regime_names)
        ax6d.set_ylabel(r"Mean Energy ($\|\mathbf{v}\|^2$)")
        ax6d2.set_ylabel("Coefficient of Variation")
        ax6d.set_title("D) Energy Metrics by Regime", fontweight="bold")
        ax6d.legend(loc="upper left")
        ax6d2.legend(loc="upper right")

        fig6.suptitle(
            "Kinetic Energy Analysis: Dynamical Activity Proxy",
            fontsize=20, fontweight="bold",
        )
        fig6.savefig(output_dir / "fig_kinetic_energy.png", dpi=150, bbox_inches="tight")
        fig6.savefig(output_dir / "fig_kinetic_energy.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig6)
        print("  Saved fig_kinetic_energy.png/pdf")
    else:
        print(f"\n--- Skipping Figure 6: kinetic energy dir not found ({ke_input_dir}) ---")

    # ------------------------------------------------------------------
    n_figures = 6 if ke_input_dir.exists() else 5
    print("\n" + "=" * 70)
    print(f"All {n_figures} figures saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
