#!/usr/bin/env python3
"""
Aggregate detuning sweep results and plot:

  1) Raw scatter: contrast_rare_center vs coupling metric
     η = DeltaOmega_over_geff = ΔΩ / |g_eff|

  2) Raw scatter: contrast_rare_center vs scaled detuning
     δ_A / f1A = delta_Hz / f1A_Hz

  3) Raw scatter: |Δslope_center| vs coupling metric
     |Δslope_center| = |I_z_slope_on_center - I_z_slope_off_center|

  4) Raw scatter: |Δslope_center| vs scaled detuning
     δ_A / f1A = delta_Hz / f1A_Hz

No averaging or binning: every detuning point from every sweep is shown.

This script also saves each plotted page as a PNG image in a "graphs" subfolder
next to the output PDF.
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

# Your f1A values are expected to be in the range 5–50 kHz (steps of 2.5 kHz).
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
        # Try to bring dialog to front on Windows.
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


def _to_float(val):
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _first_float(dct, keys):
    for k in keys:
        if k in dct:
            v = _to_float(dct.get(k))
            if v is not None:
                return v
    return None



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

        # Compute y limits within the zoomed x-range to prevent near-zero-x outliers
        # from crushing the y-scale.
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
    """Scatter with continuous coloring (and gray for missing color values).

    Parameters
    ----------
    vmin, vmax : float | None
        If provided, clamps the colormap range.
    cbar_ticks : array-like | None
        If provided, sets explicit ticks on the colorbar.
    """
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
    """
    Recursively find all summary.json files under root_dir.

    Yields full paths to summary.json.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "summary.json" in filenames:
            yield os.path.join(dirpath, "summary.json")


def load_data_from_summary(summary_path: str):
    """
    Load per-detuning metrics for one sweep.

    Returns
    -------
    points : list of dict
        Each dict has keys:
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
        # If missing, skip this sweep
        return []

    points = []
    for row in sweep_results:
        coupling = row.get("DeltaOmega_over_geff", float("nan"))
        contrast = row.get("contrast_rare_center", float("nan"))
        delta_Hz = row.get("delta_Hz", float("nan"))

        # Slopes for rare-center; may be absent in older summaries
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

        if coupling is None or contrast is None or delta_Hz is None:
            continue

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
    """
    Gather all points from all summary.json files under root_dir.

    Returns
    -------
    all_points : list of dict
        Each with keys "coupling_metric", "contrast", "f1A_Hz",
        "delta_Hz", "abs_delta_slope_center".
    """
    all_points = []
    for summary_path in find_sweep_summaries(root_dir):
        pts = load_data_from_summary(summary_path)
        all_points.extend(pts)

    return all_points


def make_plots(root_dir: str, pdf_path: str):
    """
    Main plotting routine: gathers data, builds all four plots, and writes a PDF.
    """
    all_points = aggregate_points(root_dir)

    if not all_points:
        raise RuntimeError(f"No valid data points found under {root_dir!r}")

    # --- Extract raw arrays ---
    coupling = np.array([p["coupling_metric"] for p in all_points], dtype=float)
    contrast = np.array([p["contrast"] for p in all_points], dtype=float)
    f1A_Hz = np.array([p["f1A_Hz"] for p in all_points], dtype=float)
    delta_Hz = np.array([p["delta_Hz"] for p in all_points], dtype=float)
    abs_delta_slope = np.array(
        [p["abs_delta_slope_center"] for p in all_points], dtype=float
    )

    # mask any non-finite (just in case)
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
    # scaled detuning: δ_A / f1A
    detuning_ratio = delta_Hz / f1A_Hz

    # Color metric (kHz) for readability
    f1A_kHz = f1A_Hz / 1000.0

    # Output folder for individual graphs
    graphs_dir = os.path.join(os.path.dirname(pdf_path), "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        # --------------------------------------------------
        # Page 1: raw scatter - contrast vs coupling metric
        # --------------------------------------------------
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
        ax1.set_title(
            "Contrast vs coupling metric\n"
            "(all detuning points across all sweeps)"
        )
        ax1.grid(True, alpha=0.3)

        _save_fig(fig1, os.path.join(graphs_dir, "01_contrast_vs_eta.png"), pdf)

        # --------------------------------------------------
        # Page 2: raw scatter - contrast vs δ_A / f1A
        # --------------------------------------------------
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
        ax2.set_xlabel(r"Scaled detuning $\delta_A / f_{1A}$")
        ax2.set_ylabel(r"Contrast")
        ax2.set_title(
            r"Contrast vs $\delta_A / f_{1A}$"
            "\n(all detuning points across all sweeps)"
        )
        ax2.grid(True, alpha=0.3)

        _save_fig(fig2, os.path.join(graphs_dir, "02_contrast_vs_scaled_detuning.png"), pdf)

        # --------------------------------------------------
        # Page 3: raw scatter - |Δslope_center| vs coupling metric
        # --------------------------------------------------
        # Use only points where we actually have a finite abs_delta_slope
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
            ax3.set_title(
                r"Absolute slope difference vs coupling metric"
                "\n(all detuning points across all sweeps)"
            )
            ax3.grid(True, alpha=0.3)

            _save_fig(fig3, os.path.join(graphs_dir, "03_abs_slope_diff_vs_eta_zoom.png"), pdf)

        # --------------------------------------------------
        # Page 4: raw scatter - |Δslope_center| vs δ_A / f1A
        # --------------------------------------------------
        if coupling_s.size > 0:
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

            ax4.set_xlabel(r"Scaled detuning $\delta_A / f_{1A}$")
            ax4.set_ylabel(r"$| \Delta I^z_{\mathrm{slope,center}} |$")
            ax4.set_title(
                r"Absolute slope difference vs $\delta_A / f_{1A}$"
                "\n(all detuning points across all sweeps)"
            )
            ax4.grid(True, alpha=0.3)

            _save_fig(fig4, os.path.join(graphs_dir, "04_abs_slope_diff_vs_scaled_detuning_zoom.png"), pdf)

    print(f"Wrote summary PDF to: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Gather detuning sweep results and plot:\n"
            "  (1) Raw scatter of contrast_rare_center vs coupling metric "
            "(ΔΩ / |g_eff|).\n"
            "  (2) Raw scatter of contrast_rare_center vs δ_A / f1A.\n"
            "  (3) Raw scatter of |Δslope_center| vs coupling metric.\n"
            "  (4) Raw scatter of |Δslope_center| vs δ_A / f1A."
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

    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    make_plots(root_dir, pdf_path)


if __name__ == "__main__":
    main()
