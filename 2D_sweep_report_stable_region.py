"""
Aggregate detuning sweep results and plot:

  1) Raw scatter: contrast_rare_center vs coupling metric
     η = DeltaOmega_over_geff = ΔΩ / |g_eff|

  2) Raw scatter: contrast_rare_center vs scaled detuning
     x = δ_A / f1A = delta_Hz / f1A_Hz

  3) Raw scatter: |Δslope_center| vs coupling metric
     |Δslope_center| = |I_z_slope_on_center - I_z_slope_off_center|

  4) Raw scatter: |Δslope_center| vs scaled detuning

PLUS (NEW):
  5) Objective "stable region" analysis in x = δ_A / f1A:
     - bins points by x (rounded)
     - computes pass fraction p(x) = frac[ C<0 and |C|>=C_min ]
     - finds largest contiguous interval where p(x) >= p_min
     - prints a summary table and writes stable_region_stats.json
"""

import argparse
import json
import math
import os
import sys

import tkinter as tk
from tkinter import filedialog

import numpy as np

# Use a non-GUI backend so Tk / TkAgg never get involved
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize


POINT_ALPHA = 0.85
POINT_SIZE = 24
ZOOM_PERCENTILES = (1.0, 99.0)

# F1A values are expected to be in the range 5–50 kHz (steps of 2.5 kHz).
# We color by kHz for readability.
F1A_COLOR_VMIN_KHZ = 5.0
F1A_COLOR_VMAX_KHZ = 50.0
F1A_COLORBAR_TICKS_KHZ = np.arange(5.0, 50.0 + 0.001, 5.0)


def pick_root_dir_via_ui() -> str:
    """Open a folder picker dialog and return the selected directory.

    Returns an empty string if the user cancels.
    """
    ui = tk.Tk()
    ui.withdraw()
    try:
        ui.attributes("-topmost", True)
    except Exception:
        pass

    selected = filedialog.askdirectory(
        title="Select the detuning sweep results folder",
        initialdir=os.getcwd(),
        mustexist=True,
    )

    try:
        ui.destroy()
    except Exception:
        pass
    return selected or ""


def _apply_zoom_to_main_data(ax, x, y, percentiles=ZOOM_PERCENTILES):
    """Zoom axes limits to the main bulk of the data using robust percentiles."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(m) < 5:
        return

    x_f = x[m]
    y_f = y[m]

    lo, hi = percentiles
    x_lo, x_hi = np.percentile(x_f, [lo, hi])
    if math.isfinite(x_lo) and math.isfinite(x_hi) and x_hi > x_lo:
        ax.set_xlim(x_lo, x_hi)
        mx = (x_f >= x_lo) & (x_f <= x_hi)
        y_zoom = y_f[mx] if np.count_nonzero(mx) >= 5 else y_f
    else:
        y_zoom = y_f

    y_lo, y_hi = np.percentile(y_zoom, [lo, hi])
    if math.isfinite(y_lo) and math.isfinite(y_hi) and y_hi > y_lo:
        pad = 0.05 * (y_hi - y_lo)
        ax.set_ylim(y_lo - pad, y_hi + pad)


def _scatter_with_coloring(
    ax,
    x,
    y,
    c_metric,
    cbar_label,
    add_colorbar=True,
    vmin=None,
    vmax=None,
    cbar_ticks=None,
):
    """Scatter with continuous coloring (and gray for missing color values)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    c_metric = np.asarray(c_metric, dtype=float)

    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    c_metric = c_metric[finite]

    c_finite = np.isfinite(c_metric)
    if np.any(c_finite):
        norm = None
        if vmin is not None and vmax is not None and math.isfinite(vmin) and math.isfinite(vmax) and vmax > vmin:
            norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

        sc = ax.scatter(
            x[c_finite],
            y[c_finite],
            s=POINT_SIZE,
            c=c_metric[c_finite],
            alpha=POINT_ALPHA,
            norm=norm,
        )
        if add_colorbar:
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label(cbar_label)
            if cbar_ticks is not None:
                cbar.set_ticks(cbar_ticks)

    if np.any(~c_finite):
        ax.scatter(
            x[~c_finite],
            y[~c_finite],
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            color="0.5",
            label="color missing",
        )
        ax.legend(loc="best")


def _save_fig(fig, out_path_png, pdf):
    fig.tight_layout()
    pdf.savefig(fig)
    fig.savefig(out_path_png, dpi=300)
    plt.close(fig)


def find_sweep_summaries(root_dir: str):
    """Recursively find all summary.json files under root_dir."""
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "summary.json" in filenames:
            yield os.path.join(dirpath, "summary.json")


def load_data_from_summary(summary_path: str):
    """
    Load per-detuning metrics for one sweep.

    Returns list of dicts with:
      - coupling_metric (float)          [DeltaOmega_over_geff]
      - contrast (float)                 [contrast_rare_center]
      - f1A_Hz (float)                   [from global_params]
      - delta_Hz (float)                 [per-detuning]
      - abs_delta_slope_center (float)   [|s_on - s_off|], if available
    """
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    global_params = data.get("global_params", {})
    sweep_results = data.get("sweep_results", [])

    f1A_Hz = global_params.get("f1A_Hz", None)
    if f1A_Hz is None:
        return []

    points = []
    for row in sweep_results:
        coupling = row.get("DeltaOmega_over_geff", float("nan"))
        contrast = row.get("contrast_rare_center", float("nan"))
        delta_Hz = row.get("delta_Hz", float("nan"))

        slope_off = row.get("I_z_slope_off_center", None)
        slope_on = row.get("I_z_slope_on_center", None)
        abs_delta_slope = float("nan")

        if slope_off is not None and slope_on is not None:
            try:
                s_off_val = float(slope_off)
                s_on_val = float(slope_on)
                if math.isfinite(s_off_val) and math.isfinite(s_on_val):
                    abs_delta_slope = abs(s_on_val - s_off_val)
            except (TypeError, ValueError):
                abs_delta_slope = float("nan")

        try:
            coupling = float(coupling)
            contrast = float(contrast)
            delta_Hz = float(delta_Hz)
            f1A_val = float(f1A_Hz)
        except (TypeError, ValueError):
            continue

        if not (
            math.isfinite(coupling)
            and math.isfinite(contrast)
            and math.isfinite(delta_Hz)
            and math.isfinite(f1A_val)
            and f1A_val != 0.0
        ):
            continue

        points.append(
            {
                "coupling_metric": coupling,
                "contrast": contrast,
                "f1A_Hz": f1A_val,
                "delta_Hz": delta_Hz,
                "abs_delta_slope_center": abs_delta_slope,
            }
        )

    return points


def aggregate_points(root_dir: str):
    """Gather all points from all summary.json files under root_dir."""
    all_points = []
    for summary_path in find_sweep_summaries(root_dir):
        pts = load_data_from_summary(summary_path)
        all_points.extend(pts)
    return all_points


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def compute_stable_region(
    detuning_ratio: np.ndarray,
    contrast: np.ndarray,
    c_min: float,
    p_min: float,
    bin_decimals: int,
    require_negative: bool = True,
):
    """
    Bin by x=detuning_ratio (rounded), compute pass fraction per bin:
        pass = (C < 0) & (|C| >= c_min)   [if require_negative]
        pass = (C > 0) & (|C| >= c_min)   [if not require_negative]

    Then choose largest contiguous run of bins with p >= p_min.
    """
    x = np.asarray(detuning_ratio, dtype=float)
    c = np.asarray(contrast, dtype=float)
    m = np.isfinite(x) & np.isfinite(c)
    x = x[m]
    c = c[m]

    if x.size == 0:
        raise RuntimeError("No finite (x, contrast) points for stable-region analysis.")

    # Bin by rounding. Your sweep makes x values typically land on a neat grid.
    x_bin = np.round(x, decimals=bin_decimals)

    bins = {}
    for xb, cb in zip(x_bin, c):
        bins.setdefault(float(xb), []).append(float(cb))

    x_centers = np.array(sorted(bins.keys()), dtype=float)

    stats = []
    for xc in x_centers:
        vals = np.array(bins[float(xc)], dtype=float)
        N = int(vals.size)
        med = float(np.median(vals)) if N > 0 else float("nan")
        mad = _mad(vals)

        if require_negative:
            passed = (vals < 0.0) & (np.abs(vals) >= c_min)
        else:
            passed = (vals > 0.0) & (np.abs(vals) >= c_min)

        p = float(np.mean(passed)) if N > 0 else float("nan")

        stats.append(
            {
                "x": float(xc),
                "N": N,
                "p": p,
                "median_C": med,
                "mad_C": mad,
            }
        )

    # Determine which bins qualify
    qualify = np.array([(s["p"] >= p_min) for s in stats], dtype=bool)

    # Find contiguous runs of True
    runs = []
    i = 0
    while i < qualify.size:
        if not qualify[i]:
            i += 1
            continue
        j = i
        while j < qualify.size and qualify[j]:
            j += 1
        runs.append((i, j - 1))
        i = j

    best = None
    for (i0, i1) in runs:
        run_stats = stats[i0 : i1 + 1]
        run_N = sum(s["N"] for s in run_stats)
        run_len = (i1 - i0 + 1)

        run_vals = []
        for s in run_stats:
            run_vals.extend(bins[s["x"]])
        run_vals = np.asarray(run_vals, dtype=float)

        run_median = float(np.median(run_vals)) if run_vals.size else float("nan")

        # Primary: longest run; secondary: more points; tertiary: more negative median (if require_negative)
        key = (
            run_len,
            run_N,
            (-run_median if require_negative and math.isfinite(run_median) else 0.0),
        )
        if best is None or key > best["key"]:
            best = {
                "i0": i0,
                "i1": i1,
                "x_lo": float(x_centers[i0]),
                "x_hi": float(x_centers[i1]),
                "run_len": int(run_len),
                "run_N": int(run_N),
                "run_median_C": run_median,
                "key": key,
            }

    return stats, best


def make_plots_and_analyze(
    root_dir: str,
    pdf_path: str,
    c_min: float,
    p_min: float,
    bin_decimals: int,
    stable_json_path: str,
    add_stability_page: bool,
):
    all_points = aggregate_points(root_dir)
    if not all_points:
        raise RuntimeError(f"No valid data points found under {root_dir!r}")

    coupling = np.array([p["coupling_metric"] for p in all_points], dtype=float)
    contrast = np.array([p["contrast"] for p in all_points], dtype=float)
    f1A_Hz = np.array([p["f1A_Hz"] for p in all_points], dtype=float)
    delta_Hz = np.array([p["delta_Hz"] for p in all_points], dtype=float)
    abs_delta_slope = np.array([p["abs_delta_slope_center"] for p in all_points], dtype=float)

    base_mask = (
        np.isfinite(coupling)
        & np.isfinite(contrast)
        & np.isfinite(f1A_Hz)
        & np.isfinite(delta_Hz)
        & (f1A_Hz != 0.0)
    )

    coupling = coupling[base_mask]
    contrast = contrast[base_mask]
    f1A_Hz = f1A_Hz[base_mask]
    delta_Hz = delta_Hz[base_mask]
    abs_delta_slope = abs_delta_slope[base_mask]

    detuning_ratio = delta_Hz / f1A_Hz
    f1A_kHz = f1A_Hz / 1000.0

    # --- Stable-region analysis ---
    stats, best = compute_stable_region(
        detuning_ratio=detuning_ratio,
        contrast=contrast,
        c_min=c_min,
        p_min=p_min,
        bin_decimals=bin_decimals,
        require_negative=True,
    )

    # Print concise table
    print("\n=== Stable-region analysis in x = delta_A / f1A ===")
    print(f"Criterion: pass = (C < 0) and (|C| >= {c_min:g});  p_min = {p_min:g}")
    print(f"Binning: x rounded to {bin_decimals} decimals\n")
    print("   x        N     p(pass)   median(C)    MAD(C)")
    print("----------------------------------------------------")
    for s in stats:
        print(f"{s['x']:7.3f}  {s['N']:6d}   {s['p']:7.3f}   {s['median_C']:10.4f}  {s['mad_C']:9.4f}")

    if best is None:
        print("\nNo contiguous stable region found for the chosen thresholds.")
    else:
        print("\nBest stable region (largest contiguous run with p>=p_min):")
        print(f"  x in [{best['x_lo']:.3f}, {best['x_hi']:.3f}]")
        print(f"  bins = {best['run_len']}, points = {best['run_N']}, median(C) = {best['run_median_C']:.4f}")

    # Save JSON summary
    out = {
        "criteria": {
            "c_min": float(c_min),
            "p_min": float(p_min),
            "bin_decimals": int(bin_decimals),
            "require_negative": True,
        },
        "per_bin": stats,
        "best_region": best,
    }
    with open(stable_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote: {stable_json_path}")

    # Output folder for individual graphs
    graphs_dir = os.path.join(os.path.dirname(pdf_path), "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        # Page 1: contrast vs eta
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        _scatter_with_coloring(
            ax1,
            coupling,
            contrast,
            f1A_kHz,
            r"$f_{1A}$ (kHz)",
            vmin=F1A_COLOR_VMIN_KHZ,
            vmax=F1A_COLOR_VMAX_KHZ,
            cbar_ticks=F1A_COLORBAR_TICKS_KHZ,
        )
        ax1.set_xlabel(r"Coupling metric $\eta = \Delta\Omega / |g_{\mathrm{eff}}|$")
        ax1.set_ylabel(r"Contrast")
        ax1.set_title("Contrast vs coupling metric\n(all detuning points across all sweeps)")
        ax1.grid(True, alpha=0.3)
        _save_fig(fig1, os.path.join(graphs_dir, "01_contrast_vs_eta.png"), pdf)

        # Page 2: contrast vs scaled detuning
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        _scatter_with_coloring(
            ax2,
            detuning_ratio,
            contrast,
            f1A_kHz,
            r"$f_{1A}$ (kHz)",
            vmin=F1A_COLOR_VMIN_KHZ,
            vmax=F1A_COLOR_VMAX_KHZ,
            cbar_ticks=F1A_COLORBAR_TICKS_KHZ,
        )
        ax2.set_xlabel(r"Scaled detuning $x=\delta_A / f_{1A}$")
        ax2.set_ylabel(r"Contrast")
        ax2.set_title(r"Contrast vs $x=\delta_A / f_{1A}$" "\n(all detuning points across all sweeps)")
        ax2.grid(True, alpha=0.3)
        _save_fig(fig2, os.path.join(graphs_dir, "02_contrast_vs_scaled_detuning.png"), pdf)

        # Page 3/4: abs slope diff (if available)
        mask_slope = np.isfinite(abs_delta_slope)
        coupling_s = coupling[mask_slope]
        abs_delta_slope_s = abs_delta_slope[mask_slope]
        f1A_kHz_s = f1A_kHz[mask_slope]

        if coupling_s.size > 0:
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            _scatter_with_coloring(
                ax3,
                coupling_s,
                abs_delta_slope_s,
                f1A_kHz_s,
                r"$f_{1A}$ (kHz)",
                vmin=F1A_COLOR_VMIN_KHZ,
                vmax=F1A_COLOR_VMAX_KHZ,
                cbar_ticks=F1A_COLORBAR_TICKS_KHZ,
            )
            _apply_zoom_to_main_data(ax3, coupling_s, abs_delta_slope_s)
            ax3.set_xlabel(r"Coupling metric $\eta = \Delta\Omega / |g_{\mathrm{eff}}|$")
            ax3.set_ylabel(r"$| \Delta I^z_{\mathrm{slope,center}} |$")
            ax3.set_title(r"Absolute slope difference vs coupling metric" "\n(all detuning points across all sweeps)")
            ax3.grid(True, alpha=0.3)
            _save_fig(fig3, os.path.join(graphs_dir, "03_abs_slope_diff_vs_eta_zoom.png"), pdf)

            detuning_ratio_s = detuning_ratio[mask_slope]
            fig4, ax4 = plt.subplots(figsize=(8, 5))
            _scatter_with_coloring(
                ax4,
                detuning_ratio_s,
                abs_delta_slope_s,
                f1A_kHz_s,
                r"$f_{1A}$ (kHz)",
                vmin=F1A_COLOR_VMIN_KHZ,
                vmax=F1A_COLOR_VMAX_KHZ,
                cbar_ticks=F1A_COLORBAR_TICKS_KHZ,
            )
            _apply_zoom_to_main_data(ax4, detuning_ratio_s, abs_delta_slope_s)
            ax4.set_xlabel(r"Scaled detuning $x=\delta_A / f_{1A}$")
            ax4.set_ylabel(r"$| \Delta I^z_{\mathrm{slope,center}} |$")
            ax4.set_title(r"Absolute slope difference vs $x=\delta_A / f_{1A}$" "\n(all detuning points across all sweeps)")
            ax4.grid(True, alpha=0.3)
            _save_fig(fig4, os.path.join(graphs_dir, "04_abs_slope_diff_vs_scaled_detuning_zoom.png"), pdf)

        # Optional Page 5: stability summary plot
        if add_stability_page:
            xs = np.array([s["x"] for s in stats], dtype=float)
            ps = np.array([s["p"] for s in stats], dtype=float)
            meds = np.array([s["median_C"] for s in stats], dtype=float)

            fig5, ax5 = plt.subplots(figsize=(8, 5))
            ax5.plot(xs, ps, marker="o")
            ax5.axhline(p_min, linestyle="--")
            ax5.set_xlabel(r"Scaled detuning $x=\delta_A / f_{1A}$")
            ax5.set_ylabel(r"Pass fraction $p(x)$")
            title = f"Stable-region pass fraction (C<0 and |C|>={c_min:g})"
            if best is not None:
                ax5.axvspan(best["x_lo"], best["x_hi"], alpha=0.2)
                title += f"\nBest band: [{best['x_lo']:.3f}, {best['x_hi']:.3f}]"
            ax5.set_title(title)
            ax5.grid(True, alpha=0.3)
            _save_fig(fig5, os.path.join(graphs_dir, "05_pass_fraction_vs_scaled_detuning.png"), pdf)

    print(f"\nWrote summary PDF to: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Gather detuning sweep results and plot scatter figures.\n"
            "Also computes an objective 'stable region' in x=delta_A/f1A using a pass-fraction rule."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=None,
        help=(
            "Root directory containing sea_detuning_sweep_* subfolders with summary.json files. "
            "If omitted, a folder picker UI will open."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output PDF path (default: <root>/contrast_vs_coupling_summary.pdf)",
    )

    # --- New stability options ---
    parser.add_argument(
        "--c-min",
        type=float,
        default=0.2,
        help="Minimum contrast magnitude for a point to count as a 'pass' (default: 0.2).",
    )
    parser.add_argument(
        "--p-min",
        type=float,
        default=0.8,
        help="Minimum per-bin pass fraction to qualify as stable (default: 0.8).",
    )
    parser.add_argument(
        "--bin-decimals",
        type=int,
        default=3,
        help="Decimals used to bin x=delta_A/f1A by rounding (default: 3).",
    )
    parser.add_argument(
        "--stable-json",
        default=None,
        help="Path to write stable region JSON (default: <root>/stable_region_stats.json).",
    )
    parser.add_argument(
        "--add-stability-page",
        action="store_true",
        help="Add an extra PDF page plotting pass fraction vs scaled detuning.",
    )

    args = parser.parse_args()

    if args.root is None:
        selected = pick_root_dir_via_ui()
        if not selected:
            print("No folder selected. Exiting.")
            raise SystemExit(1)
        root_dir = os.path.abspath(selected)
    else:
        root_dir = os.path.abspath(args.root)

    if not os.path.isdir(root_dir):
        print(f"Root folder does not exist: {root_dir}")
        raise SystemExit(2)

    if args.output is None:
        pdf_path = os.path.join(root_dir, "contrast_vs_coupling_summary.pdf")
    else:
        pdf_path = os.path.abspath(args.output)

    if args.stable_json is None:
        stable_json_path = os.path.join(root_dir, "stable_region_stats.json")
    else:
        stable_json_path = os.path.abspath(args.stable_json)

    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    os.makedirs(os.path.dirname(stable_json_path), exist_ok=True)

    make_plots_and_analyze(
        root_dir=root_dir,
        pdf_path=pdf_path,
        c_min=args.c_min,
        p_min=args.p_min,
        bin_decimals=args.bin_decimals,
        stable_json_path=stable_json_path,
        add_stability_page=args.add_stability_page,
    )


if __name__ == "__main__":
    main()
