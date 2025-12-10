# reprocess_sweep_results.py (updated to reuse helpers from sweep_sea_detuning)
"""
Reprocess an existing sea-detuning sweep WITHOUT re-running simulations.

Assumes the sweep directory was produced by the *new* sweep_sea_detuning.py,
which, for each detuning, saves:

  time_and_obs_center_off.npz
  time_and_obs_center_on.npz
  time_and_obs_shell_off.npz

and a top-level summary.json of the form:

  {
    "global_params": { ... },
    "sweep_results": [
      {
        "delta_Hz": ...,
        "I_z_slope_off_center": ...,
        "R_off_center": ...,
        "I_z_slope_on_center": ...,
        "R_on_center": ...,
        "contrast_rare_center": ...,
        "I_z_slope_off_sea_center": ...,
        "R_off_sea_center": ...,
        "contrast_sea_center": ...
      },
      ...
    ]
  }

This script:

  * Lets you pick a sweep directory via a folder picker (starting in 'results/').
  * Reloads saved time series (center_off/on, shell_off).
  * Re-applies coarse-graining with a user-selected window size.
  * Recomputes the I_z_slope, R values, contrast_rare_center, and
    contrast_sea_center using the SAME coarse_grain and iz_slope_from_coarse
    helpers as the sweep script (imported from sweep_sea_detuning).
  * Writes a new PDF report called

        sea_detuning_report_reprocessed[_winNN].pdf

    in the chosen sweep folder, with the same 4-plot layout as the new sweep
    (rare center OFF/ON, envelopes + slopes, sea-center control envelope, norm).
  * Writes a JSON file

        summary_reprocessed_winNN.json

    with the recomputed metrics (no detection_metric).
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# GUI folder picker
import tkinter as tk
from tkinter import filedialog

# Reuse helpers from the sweep script so definitions stay in sync
from sweep_sea_detuning import (
    coarse_grain,
    iz_slope_from_coarse,
    detuning_label,
)


# ---------------------------------------------------------------------------
# Utility: normalization helper (kept consistent with sweep script)
# ---------------------------------------------------------------------------

def _safe_normalized_difference(
    num: float,
    denom: float,
) -> float:
    """
    Return num / denom, but guard against denom = 0.

    If denom is zero (or NaN), returns NaN.
    """
    if denom == 0.0 or np.isnan(denom):
        return float("nan")
    return num / denom


# ---------------------------------------------------------------------------
# Reprocessing logic (new layout)
# ---------------------------------------------------------------------------

def det_label_from_delta(delta_hz: float) -> str:
    """
    Wrapper around sweep_sea_detuning.detuning_label for backwards clarity.
    """
    return detuning_label(delta_hz)


def reprocess_sweep(base_dir: str, window: int = 50) -> str:
    """
    Reprocess an existing sweep in base_dir and create a new PDF with
    updated coarse-graining and contrast metrics.

    Parameters
    ----------
    base_dir : str
        Path to the sweep directory (e.g. 'results/sea_detuning_sweep_...').
    window : int
        Coarse-grain window size to use for the envelopes.

    Returns
    -------
    new_pdf_path : str
        Path to the newly created PDF.
    """
    base_dir = os.path.abspath(base_dir)

    # Load summary (gives us global_params and the list of detunings)
    summary_path = os.path.join(base_dir, "summary.json")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"summary.json not found in {base_dir}")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary: Dict[str, Any] = json.load(f)

    global_params = summary.get("global_params", {})
    sweep_results_orig = summary.get("sweep_results", [])

    # Set up new PDF name
    orig_pdf = os.path.join(base_dir, "sea_detuning_report.pdf")
    if window > 0:
        new_pdf = os.path.join(
            base_dir, f"sea_detuning_report_reprocessed_win{window}.pdf"
        )
        new_summary_json = os.path.join(
            base_dir, f"summary_reprocessed_win{window}.json"
        )
    else:
        new_pdf = os.path.join(base_dir, "sea_detuning_report_reprocessed.pdf")
        new_summary_json = os.path.join(base_dir, "summary_reprocessed.json")

    print(f"Reprocessing sweep in: {base_dir}")
    print(f"  Original PDF : {orig_pdf if os.path.exists(orig_pdf) else '(not found)'}")
    print(f"  New PDF      : {new_pdf}")
    print(f"  Envelope window size: {window}")
    print("------------------------------------------------------------")

    # Collect recomputed metrics
    new_sweep_results: List[Dict[str, Any]] = []

    with PdfPages(new_pdf) as pdf:
        # --- Page 1: global parameter summary (adapted to new layout) ---
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4-ish
        ax.axis("off")

        lines: List[str] = []
        lines.append("Sea detuning sweep report (REPROCESSED)")
        lines.append("")
        lines.append(f"Reprocessed coarse-grain window = {window}")
        lines.append("")
        lines.append("Global parameters (from original sweep):")

        # Try to mirror the new sweep's header if keys are present
        f_Az = global_params.get("f_Az_Hz", None)
        f_Rz = global_params.get("f_Rz_Hz", None)
        f1A = global_params.get("f1A_Hz", None)
        f1R = global_params.get("f1R_Hz", None)
        target_det = global_params.get("target_sea_detuning", None)
        gamma_sea = global_params.get("gamma_sea", None)
        gamma_rare = global_params.get("gamma_rare", None)
        B0_common = global_params.get("B0_common_T", None)
        B1_sea = global_params.get("B1_sea_T", None)
        B1_rare = global_params.get("B1_rare_T", None)
        dip_scale = global_params.get("dipolar_scale_SI", None)
        shell_scale_m = global_params.get("shell_scale_m", None)
        t_final = global_params.get("t_final_s", None)
        steps = global_params.get("steps", None)
        n_sea = global_params.get("n_sea", None)
        phi_sea = global_params.get("phi_sea_rad", None)
        phi_rare = global_params.get("phi_rare_rad", None)
        sea_spin_type = global_params.get("sea_spin_type", None)
        rare_spin_type = global_params.get("rare_spin_type", None)
        solver_atol = global_params.get("solver_atol", None)
        solver_rtol = global_params.get("solver_rtol", None)
        solver_nsteps = global_params.get("solver_nsteps", None)
        solver_max_step = global_params.get("solver_max_step", None)
        sea_detunings = global_params.get("sea_detunings_Hz", [])

        if f_Az is not None:
            lines.append(f"  f_Az (sea Larmor)     = {f_Az/1e6:.3f} MHz")
        if f_Rz is not None:
            lines.append(f"  f_Rz (rare Larmor)    = {f_Rz/1e6:.3f} MHz")
        if f1A is not None:
            lines.append(f"  f1A (sea Rabi)        = {f1A/1e3:.3f} kHz")
        if f1R is not None:
            lines.append(f"  f1R (rare Rabi)       = {f1R/1e3:.3f} kHz")
        if target_det is not None:
            lines.append(f"  Target sea detuning   = {target_det/1e3:.3f} kHz")
        if gamma_sea is not None:
            lines.append(f"  gamma_sea             = {gamma_sea:.3e} rad·s⁻¹·T⁻¹")
        if gamma_rare is not None:
            lines.append(f"  gamma_rare            = {gamma_rare:.3e} rad·s⁻¹·T⁻¹")
        if B0_common is not None:
            lines.append(f"  B0_common             = {B0_common:.3f} T")
        if B1_sea is not None:
            lines.append(f"  B1_sea                = {B1_sea:.3e} T")
        if B1_rare is not None:
            lines.append(f"  B1_rare               = {B1_rare:.3e} T")
        if dip_scale is not None:
            lines.append(f"  dipolar_scale_SI      = {dip_scale:.3e}")
        if shell_scale_m is not None:
            lines.append(f"  shell_scale           = {shell_scale_m*1e9:.3f} nm")
        if t_final is not None:
            lines.append(f"  t_final               = {t_final:.3e} s")
        if steps is not None:
            lines.append(f"  steps                 = {steps:d}")
        if n_sea is not None:
            lines.append(f"  n_sea                 = {n_sea:d}")
        if phi_sea is not None:
            lines.append(f"  phi_sea               = {phi_sea:.3f} rad")
        if phi_rare is not None:
            lines.append(f"  phi_rare              = {phi_rare:.3f} rad")
        if sea_spin_type is not None:
            lines.append(f"  sea_spin_type         = {sea_spin_type}")
        if rare_spin_type is not None:
            lines.append(f"  rare_spin_type        = {rare_spin_type}")

        lines.append("")
        lines.append(f"  solver_atol           = {solver_atol}")
        lines.append(f"  solver_rtol           = {solver_rtol}")
        lines.append(f"  solver_nsteps         = {solver_nsteps}")
        lines.append(f"  solver_max_step       = {solver_max_step}")
        lines.append("")
        lines.append(f"  coarse_window (orig)  = {global_params.get('coarse_window', 'NA')}")
        lines.append(f"  coarse_window (this)  = {window}")
        lines.append("")
        if sea_detunings:
            lines.append("Sea detunings (δ_A = f_Az - f_rf,A) in Hz:")
            det_strs = [f"{d:+.1f}" for d in sea_detunings]
            per_line = 6
            for i in range(0, len(det_strs), per_line):
                lines.append("  " + ", ".join(det_strs[i : i + per_line]))

        ax.text(
            0.02,
            0.98,
            "\n".join(lines),
            transform=ax.transAxes,
            va="top",
            family="monospace",
        )
        pdf.savefig(fig)
        plt.close(fig)

        # --- Per-detuning pages (mirror new sweep layout) ---
        for row in sweep_results_orig:
            delta_hz = float(row["delta_Hz"])
            det_dir = os.path.join(base_dir, det_label_from_delta(delta_hz))
            if not os.path.isdir(det_dir):
                print(f"Warning: directory for δ_A={delta_hz:+.1f} Hz not found, skipping.")
                continue

            print(f"Reprocessing δ_A = {delta_hz:+.1f} Hz ...")

            # Load NPZ files for the 3 geometries we now use
            path_center_off = os.path.join(det_dir, "time_and_obs_center_off.npz")
            path_center_on  = os.path.join(det_dir, "time_and_obs_center_on.npz")
            path_sea_center_off  = os.path.join(det_dir, "time_and_obs_shell_off.npz")

            if not (
                os.path.isfile(path_center_off)
                and os.path.isfile(path_center_on)
                and os.path.isfile(path_sea_center_off)
            ):
                print(
                    f"  Missing center_off/center_on/sea-center_off NPZ for "
                    f"δ_A={delta_hz:+.1f} Hz, skipping."
                )
                continue

            data_center_off = np.load(path_center_off)
            data_center_on  = np.load(path_center_on)
            data_sea_center_off  = np.load(path_sea_center_off)

            # Time arrays
            t_center_off = data_center_off["t"]
            t_center_on  = data_center_on["t"]
            t_sea_center_off  = data_sea_center_off["t"]

            # Observables
            Iz_center_off = data_center_off["Iz_sea"]
            Iz_center_on  = data_center_on["Iz_sea"]
            Iz_sea_center_off  = data_sea_center_off["Iz_sea"]

            norm_center_off = data_center_off.get("state_norm", None)
            norm_center_on  = data_center_on.get("state_norm", None)

            # Recompute coarse-grained envelopes and slopes
            # Rare-at-center geometry
            t_c_off_center, Iz_c_off_center = coarse_grain(
                t_center_off, Iz_center_off, window=window
            )
            t_c_on_center, Iz_c_on_center = coarse_grain(
                t_center_on, Iz_center_on, window=window
            )

            slope_off_center = iz_slope_from_coarse(t_c_off_center, Iz_c_off_center)
            slope_on_center  = iz_slope_from_coarse(t_c_on_center,  Iz_c_on_center)

            I_z_slope_off_center = slope_off_center["I_z_slope"]
            I_z_slope_on_center  = slope_on_center["I_z_slope"]
            R_off_center = slope_off_center["R_value"]
            R_on_center = slope_on_center["R_value"]

            # Sea-as-center control geometry
            t_c_off_sea_center, Iz_c_off_sea_center = coarse_grain(
                t_sea_center_off, Iz_sea_center_off, window=window
            )
            slope_off_sea_center = iz_slope_from_coarse(
                t_c_off_sea_center, Iz_c_off_sea_center
            )

            I_z_slope_off_sea_center = slope_off_sea_center["I_z_slope"]
            R_off_sea_center = slope_off_sea_center["R_value"]

            # Normalized contrast metrics
            contrast_rare_center = _safe_normalized_difference(
                I_z_slope_on_center - I_z_slope_off_center,
                I_z_slope_off_center,
            )
            contrast_sea_center = _safe_normalized_difference(
                I_z_slope_on_center - I_z_slope_off_sea_center,
                I_z_slope_off_sea_center,
            )

            # Store recomputed metrics
            new_metrics = {
                "delta_Hz": float(delta_hz),
                "I_z_slope_off_center": float(I_z_slope_off_center),
                "R_off_center": float(R_off_center),
                "I_z_slope_on_center": float(I_z_slope_on_center),
                "R_on_center": float(R_on_center),
                "contrast_rare_center": float(contrast_rare_center),
                "I_z_slope_off_sea_center": float(I_z_slope_off_sea_center),
                "R_off_sea_center": float(R_off_sea_center),
                "contrast_sea_center": float(contrast_sea_center),
            }
            new_sweep_results.append(new_metrics)

            # --- Plot helpers (same structure as sweep script) ---

            def _plot_slope_segment(ax, slope_info: Dict[str, float], style: str, label: str) -> None:
                if np.isnan(slope_info["I_z_slope"]):
                    return
                ax.plot(
                    [slope_info["t_start"], slope_info["t_end"]],
                    [slope_info["I_z_start"], slope_info["I_z_end"]],
                    style,
                    linewidth=2.0,
                    markersize=6,
                    label=label,
                )

            def _annotate_slope_text(
                ax,
                slope_info: Dict[str, float],
                slope_value: float,
                dy: float,
                offset_sign: float,
                text_label: Optional[str] = None,
            ) -> None:
                if np.isnan(slope_value) or np.isnan(slope_info["t_start"]):
                    return
                t_mid = 0.5 * (slope_info["t_start"] + slope_info["t_end"])
                iz_mid = 0.5 * (slope_info["I_z_start"] + slope_info["I_z_end"])
                iz_mid += offset_sign * 0.03 * dy
                label = text_label or f"{slope_value:+.2e}"
                ax.text(
                    t_mid,
                    iz_mid,
                    label,
                    fontsize=6,
                    ha="center",
                    va="bottom",
                    family="monospace",
                    bbox=dict(boxstyle="round", alpha=0.2, linewidth=0),
                )

            # ------------------------------------------------------------------
            # Plot 1: ⟨I^z_sea⟩ (rare OFF vs ON) - full resolution, rare-at-center
            # ------------------------------------------------------------------
            fig1, ax1 = plt.subplots()
            ax1.plot(
                t_center_off,
                Iz_center_off,
                label=r"$\langle I^z_{\mathrm{sea}}\rangle$, rare OFF (center)",
            )
            ax1.plot(
                t_center_on,
                Iz_center_on,
                label=r"$\langle I^z_{\mathrm{sea}}\rangle$, rare ON (center)",
            )

            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel(r"$\langle I^z_{\mathrm{sea}}\rangle$")
            ax1.set_title(
                f"δ_A = {delta_hz:+.1f} Hz (rare at center, reprocessed)"
            )
            ax1.legend()
            fig1.tight_layout()
            pdf.savefig(fig1)
            plt.close(fig1)

            # ------------------------------------------------------------------
            # Plot 2: Coarse-grained ⟨I^z_sea⟩ (rare-at-center only) + slopes
            # ------------------------------------------------------------------
            fig2, ax2 = plt.subplots()

            ax2.plot(
                t_c_off_center,
                Iz_c_off_center,
                "o-",
                markersize=3,
                label=r"OFF, rare center (envelope)",
            )
            ax2.plot(
                t_c_on_center,
                Iz_c_on_center,
                "o--",
                markersize=3,
                label=r"ON, rare center (envelope)",
            )

            _plot_slope_segment(
                ax2,
                slope_off_center,
                "s-",
                r"OFF slope, rare center",
            )
            _plot_slope_segment(
                ax2,
                slope_on_center,
                "s--",
                r"ON slope, rare center",
            )

            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel(r"$\langle I^z_{\mathrm{sea}}\rangle$")
            ax2.set_title(
                "δ_A = "
                f"{delta_hz:+.1f} Hz (coarse envelopes, rare at center; reprocessed)"
            )

            all_env_center = np.concatenate([Iz_c_off_center, Iz_c_on_center])
            y_min_c = float(np.min(all_env_center))
            y_max_c = float(np.max(all_env_center))
            if y_max_c > y_min_c:
                pad_c = 0.05 * (y_max_c - y_min_c)
                ax2.set_ylim(y_min_c - pad_c, y_max_c + pad_c)
            dy_c = max(1e-8, y_max_c - y_min_c)

            _annotate_slope_text(
                ax2,
                slope_off_center,
                I_z_slope_off_center,
                dy_c,
                offset_sign=-1.0,
                text_label=f"OFF slope = {I_z_slope_off_center:+.2e}",
            )
            _annotate_slope_text(
                ax2,
                slope_on_center,
                I_z_slope_on_center,
                dy_c,
                offset_sign=+1.0,
                text_label=f"ON slope = {I_z_slope_on_center:+.2e}",
            )

            metrics_text_center = (
                f"I_z_slope_off(center)   = {I_z_slope_off_center:+.3e}\n"
                f"R_off(center)           = {R_off_center:+.3f}\n"
                f"I_z_slope_on(center)    = {I_z_slope_on_center:+.3e}\n"
                f"R_on(center)            = {R_on_center:+.3f}\n"
                f"contrast_rare_center    = {contrast_rare_center:+.3e}"
            )
            ax2.text(
                0.02,
                0.98,
                metrics_text_center,
                transform=ax2.transAxes,
                va="top",
                fontsize=7,
                family="monospace",
                bbox=dict(boxstyle="round", alpha=0.08),
            )

            ax2.legend(fontsize=7, loc="best")
            fig2.tight_layout()
            pdf.savefig(fig2)
            plt.close(fig2)

            # ------------------------------------------------------------------
            # Plot 3: Coarse-grained ⟨I^z_sea⟩ (sea-center control) + slopes
            # ------------------------------------------------------------------
            fig3, ax3 = plt.subplots()

            ax3.plot(
                t_c_off_sea_center,
                Iz_c_off_sea_center,
                "x-",
                markersize=3,
                label=r"Sea-center control (envelope)",
            )

            _plot_slope_segment(
                ax3,
                slope_off_sea_center,
                "D-",
                r"Slope, sea-center control",
            )

            ax3.set_xlabel("Time (s)")
            ax3.set_ylabel(r"$\langle I^z_{\mathrm{sea}}\rangle$)")
            ax3.set_title(
                "δ_A = "
                f"{delta_hz:+.1f} Hz (coarse envelope, sea-center control; reprocessed)"
            )

            all_env_shell = Iz_c_off_sea_center
            y_min_s = float(np.min(all_env_shell))
            y_max_s = float(np.max(all_env_shell))
            if y_max_s > y_min_s:
                pad_s = 0.05 * (y_max_s - y_min_s)
                ax3.set_ylim(y_min_s - pad_s, y_max_s + pad_s)
            dy_s = max(1e-8, y_max_s - y_min_s)

            _annotate_slope_text(
                ax3,
                slope_off_sea_center,
                I_z_slope_off_sea_center,
                dy_s,
                offset_sign=+1.0,
                text_label=f"Slope = {I_z_slope_off_sea_center:+.2e}",
            )

            metrics_text_shell = (
                f"I_z_slope_sea-center    = {I_z_slope_off_sea_center:+.3e}\n"
                f"R_sea-center            = {R_off_sea_center:+.3f}\n"
                f"contrast_sea_center     = {contrast_sea_center:+.3e}"
            )
            ax3.text(
                0.02,
                0.98,
                metrics_text_shell,
                transform=ax3.transAxes,
                va="top",
                fontsize=7,
                family="monospace",
                bbox=dict(boxstyle="round", alpha=0.08),
            )

            ax3.legend(fontsize=7, loc="best")
            fig3.tight_layout()
            pdf.savefig(fig3)
            plt.close(fig3)

            # ------------------------------------------------------------------
            # Plot 4: State norm ‖ψ(t)‖ (rare OFF vs ON, rare-at-center)
            # ------------------------------------------------------------------
            if norm_center_off is not None and norm_center_on is not None:
                figN, axN = plt.subplots()
                axN.plot(
                    t_center_off,
                    norm_center_off,
                    label=r"$\|\psi(t)\|$, rare OFF (center)",
                )
                axN.plot(
                    t_center_on,
                    norm_center_on,
                    label=r"$\|\psi(t)\|$, rare ON (center)",
                )

                axN.set_xlabel("Time (s)")
                axN.set_ylabel(r"State norm $\|\psi\|$")
                axN.set_title(
                    f"δ_A = {delta_hz:+.1f} Hz (state norm, rare at center; reprocessed)"
                )
                axN.legend()
                figN.tight_layout()
                pdf.savefig(figN)
                plt.close(figN)

        # -------- Final page: contrast-metric summary table --------
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")

        col_labels = [
            "δ_A (Hz)",
            "slope_off(center)",
            "R_off(center)",
            "slope_on(center)",
            "R_on(center)",
            "contrast_rare_center",
            "slope_sea-center",
            "R_sea-center",
            "contrast_sea_center",
        ]

        table_vals: List[List[str]] = []
        for row in new_sweep_results:
            table_vals.append(
                [
                    f"{row['delta_Hz']:+.1f}",
                    f"{row['I_z_slope_off_center']:+.3e}",
                    f"{row['R_off_center']:+.3f}",
                    f"{row['I_z_slope_on_center']:+.3e}",
                    f"{row['R_on_center']:+.3f}",
                    f"{row['contrast_rare_center']:+.3e}",
                    f"{row['I_z_slope_off_sea_center']:+.3e}",
                    f"{row['R_off_sea_center']:+.3f}",
                    f"{row['contrast_sea_center']:+.3e}",
                ]
            )

        table = ax.table(
            cellText=table_vals,
            colLabels=col_labels,
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(6)
        table.scale(1.0, 1.3)

        ax.set_title(
            "Reprocessed contrast metrics from coarse-grained ⟨I^z_sea⟩ slopes",
            pad=20,
        )

        pdf.savefig(fig)
        plt.close(fig)

    # -------- Save JSON with recomputed metrics --------
    reproc_summary = {
        "global_params": global_params,
        "sweep_results": new_sweep_results,
        "coarse_window": window,
    }
    with open(new_summary_json, "w", encoding="utf-8") as f:
        json.dump(reproc_summary, f, indent=2, default=float)

    print("Reprocessing complete.")
    print(f"  New PDF      : {new_pdf}")
    print(f"  New summary  : {new_summary_json}")
    return new_pdf


# ---------------------------------------------------------------------------
# GUI-based selector
# ---------------------------------------------------------------------------

def main() -> None:
    # Start folder picker in 'results' if it exists
    default_root = os.path.abspath("results")

    root = tk.Tk()
    root.withdraw()

    print("Select a sweep folder to reprocess (e.g. a 'sea_detuning_sweep_...' folder).")
    base_dir = filedialog.askdirectory(
        initialdir=default_root if os.path.isdir(default_root) else os.getcwd(),
        title="Select sea-detuning sweep folder",
        mustexist=True,
    )

    if not base_dir:
        raise SystemExit("No folder selected; aborting.")

    # Ask for coarse-grain window (still via CLI)
    win_str = input("Coarse-grain window size (default 50): ").strip()
    window = 50 if not win_str else int(win_str)

    new_pdf = reprocess_sweep(base_dir, window=window)
    print(f"New PDF written to: {new_pdf}")


if __name__ == "__main__":
    main()
