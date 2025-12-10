"""
Reprocess an existing sea-detuning sweep WITHOUT re-running simulations.

Assumes the sweep directory was produced by the current version of
`sweep_sea_detuning.py`, which, for each detuning, saves:

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
        "t_off_center": ...,
        "I_z_slope_on_center": ...,
        "R_on_center": ...,
        "t_on_center": ...,
        "contrast_rare_center": ...,
        "I_z_slope_off_sea_center": ...,
        "R_off_sea_center": ...,
        "t_off_sea_center": ...,
        "contrast_sea_center": ...,
        "DeltaOmega_Hz": ...,
        "g_eff_Hz": ...,
        "DeltaOmega_over_geff": ...
      },
      ...
    ]
  }

This script:

  * Lets you pick a sweep directory via a folder picker (starting in 'results/').
  * Reloads saved time series (center_off/on, shell_off).
  * Re-applies coarse-graining with a user-selected window size.
  * Recomputes I_z_slope, t-values, R values, contrast_rare_center, and
    contrast_sea_center using the SAME coarse_grain, iz_slope_from_coarse, and
    contrast_michelson_with_t_gate helpers as the sweep script.
  * Recomputes ΔΩ, g_eff, and ΔΩ/|g_eff| using:
        f1A_Hz, f1R_Hz, and rms_b_AR_Hz from global_params.
  * Writes a new PDF report called

        sea_detuning_report_reprocessed[_winNN].pdf

    in the chosen sweep folder, with the same 4-plot layout as the sweep
    (rare center OFF/ON, envelopes + slopes, sea-center control envelope, norm),
    plus a final contrast_rare_center vs ΔΩ/|g_eff| page.
  * Writes a JSON file

        summary_reprocessed_winNN.json

    with the recomputed metrics (no detection_metric), including ΔΩ, g_eff,
    ΔΩ/|g_eff|, and t-values for the slopes.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

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
    contrast_michelson_with_t_gate,
)


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

    # New output filenames
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
    print(f"  New PDF      : {new_pdf}")
    print(f"  Envelope window size: {window}")
    print("------------------------------------------------------------")

    # Pull out global quantities needed for ΔΩ/|g_eff|
    f1A_Hz = float(global_params.get("f1A_Hz", np.nan))
    f1R_Hz = float(global_params.get("f1R_Hz", np.nan))
    rms_b_AR_Hz = float(global_params.get("rms_b_AR_Hz", np.nan))

    # Collect recomputed metrics
    new_sweep_results: List[Dict[str, Any]] = []

    with PdfPages(new_pdf) as pdf:
        # --- Page 1: global parameter summary ---
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4-ish
        ax.axis("off")

        lines: List[str] = []
        lines.append("Sea detuning sweep report (REPROCESSED)")
        lines.append("")
        lines.append(f"Reprocessed coarse-grain window = {window}")
        lines.append("")
        lines.append("Global parameters (from original sweep):")

        f_Az = global_params.get("f_Az_Hz", None)
        f_Rz = global_params.get("f_Rz_Hz", None)
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
        if f1A_Hz == f1A_Hz:  # not NaN
            lines.append(f"  f1A (sea Rabi)        = {f1A_Hz/1e3:.3f} kHz")
        if f1R_Hz == f1R_Hz:
            lines.append(f"  f1R (rare Rabi)       = {f1R_Hz/1e3:.3f} kHz")
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

        # --- Helpers for plotting slope segments/text (same as sweep) ---

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
            text_label: str,
        ) -> None:
            if np.isnan(slope_value) or np.isnan(slope_info["t_start"]):
                return
            t_mid = 0.5 * (slope_info["t_start"] + slope_info["t_end"])
            iz_mid = 0.5 * (slope_info["I_z_start"] + slope_info["I_z_end"])
            iz_mid += offset_sign * 0.03 * dy
            ax.text(
                t_mid,
                iz_mid,
                text_label,
                fontsize=6,
                ha="center",
                va="bottom",
                family="monospace",
                bbox=dict(boxstyle="round", alpha=0.2, linewidth=0),
            )

        # --- Per-detuning pages ---

        # Ensure we process in order of delta_Hz
        rows_sorted = sorted(sweep_results_orig, key=lambda r: r["delta_Hz"])

        for row in rows_sorted:
            delta_hz = float(row["delta_Hz"])
            det_dir = os.path.join(base_dir, detuning_label(delta_hz))
            if not os.path.isdir(det_dir):
                print(f"Warning: directory for δ_A={delta_hz:+.1f} Hz not found, skipping.")
                continue

            print(f"Reprocessing δ_A = {delta_hz:+.1f} Hz ...")

            path_center_off = os.path.join(det_dir, "time_and_obs_center_off.npz")
            path_center_on = os.path.join(det_dir, "time_and_obs_center_on.npz")
            path_sea_center_off = os.path.join(det_dir, "time_and_obs_shell_off.npz")

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
            data_center_on = np.load(path_center_on)
            data_sea_center_off = np.load(path_sea_center_off)

            # Time arrays
            t_center_off = data_center_off["t"]
            t_center_on = data_center_on["t"]
            t_sea_center_off = data_sea_center_off["t"]

            # Observables
            Iz_center_off = data_center_off["Iz_sea"]
            Iz_center_on = data_center_on["Iz_sea"]
            Iz_sea_center_off = data_sea_center_off["Iz_sea"]

            # State norms (if stored)
            norm_center_off = (
                data_center_off["state_norm"]
                if "state_norm" in data_center_off.files
                else None
            )
            norm_center_on = (
                data_center_on["state_norm"]
                if "state_norm" in data_center_on.files
                else None
            )

            # Recompute coarse-grained envelopes and slopes
            t_c_off_center, Iz_c_off_center = coarse_grain(
                t_center_off, Iz_center_off, window=window
            )
            t_c_on_center, Iz_c_on_center = coarse_grain(
                t_center_on, Iz_center_on, window=window
            )

            slope_off_center = iz_slope_from_coarse(t_c_off_center, Iz_c_off_center)
            slope_on_center = iz_slope_from_coarse(t_c_on_center, Iz_c_on_center)

            I_z_slope_off_center = slope_off_center["I_z_slope"]
            I_z_slope_on_center = slope_on_center["I_z_slope"]
            R_off_center = slope_off_center["R_value"]
            R_on_center = slope_on_center["R_value"]
            t_off_center = slope_off_center["t_value"]
            t_on_center = slope_on_center["t_value"]

            # Sea-as-center control
            t_c_off_sea_center, Iz_c_off_sea_center = coarse_grain(
                t_sea_center_off, Iz_sea_center_off, window=window
            )
            slope_off_sea_center = iz_slope_from_coarse(
                t_c_off_sea_center, Iz_c_off_sea_center
            )

            I_z_slope_off_sea_center = slope_off_sea_center["I_z_slope"]
            R_off_sea_center = slope_off_sea_center["R_value"]
            t_off_sea_center = slope_off_sea_center["t_value"]

            # Normalized contrast metrics (Michelson-style, t-gated)
            contrast_rare_center = contrast_michelson_with_t_gate(
                I_z_slope_on_center,
                I_z_slope_off_center,
                t_on_center,
                t_off_center,
            )
            contrast_sea_center = contrast_michelson_with_t_gate(
                I_z_slope_on_center,
                I_z_slope_off_sea_center,
                t_on_center,
                t_off_sea_center,
            )

            # ΔΩ/|g_eff| for this detuning, using global f1A, f1R, rms_b_AR
            DeltaOmega_Hz = float("nan")
            g_eff_Hz = float("nan")
            DeltaOmega_over_geff = float("nan")

            if (
                np.isfinite(f1A_Hz)
                and np.isfinite(f1R_Hz)
                and np.isfinite(rms_b_AR_Hz)
            ):
                OmegaA_Hz = float(np.sqrt(delta_hz**2 + f1A_Hz**2))
                OmegaR_Hz = float(np.sqrt(0.0**2 + f1R_Hz**2))  # ≈ f1R_Hz

                DeltaOmega_Hz = OmegaA_Hz - OmegaR_Hz

                sin_theta_A = f1A_Hz / OmegaA_Hz if OmegaA_Hz != 0.0 else 0.0
                sin_theta_R = f1R_Hz / OmegaR_Hz if OmegaR_Hz != 0.0 else 0.0

                g_eff_Hz = (rms_b_AR_Hz / 4.0) * sin_theta_A * sin_theta_R

                if g_eff_Hz != 0.0 and not np.isnan(g_eff_Hz):
                    DeltaOmega_over_geff = float(DeltaOmega_Hz / abs(g_eff_Hz))

            # Store recomputed metrics
            new_metrics = {
                "delta_Hz": float(delta_hz),
                "I_z_slope_off_center": float(I_z_slope_off_center),
                "R_off_center": float(R_off_center),
                "t_off_center": float(t_off_center),
                "I_z_slope_on_center": float(I_z_slope_on_center),
                "R_on_center": float(R_on_center),
                "t_on_center": float(t_on_center),
                "contrast_rare_center": float(contrast_rare_center),
                "I_z_slope_off_sea_center": float(I_z_slope_off_sea_center),
                "R_off_sea_center": float(R_off_sea_center),
                "t_off_sea_center": float(t_off_sea_center),
                "contrast_sea_center": float(contrast_sea_center),
                "DeltaOmega_Hz": float(DeltaOmega_Hz),
                "g_eff_Hz": float(g_eff_Hz),
                "DeltaOmega_over_geff": float(DeltaOmega_over_geff),
            }
            new_sweep_results.append(new_metrics)

            # ---------------- Plot 1: Iz(t) OFF vs ON (center) ----------------
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
            ax1.set_title(f"δ_A = {delta_hz:+.1f} Hz (rare at center)")
            ax1.legend()
            fig1.tight_layout()
            pdf.savefig(fig1)
            plt.close(fig1)

            # ---------------- Plot 2: envelopes (center) + slopes --------------
            fig2, ax2 = plt.subplots()
            fig2.subplots_adjust(right=0.75)

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
                f"δ_A = {delta_hz:+.1f} Hz (coarse envelopes, rare at center)"
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
                f"t_off(center)           = {t_off_center:+.3f}\n"
                f"I_z_slope_on(center)    = {I_z_slope_on_center:+.3e}\n"
                f"t_on(center)            = {t_on_center:+.3f}\n"
                f"contrast_rare_center    = {contrast_rare_center:+.3e}\n"
                f"ΔΩ/|g_eff|              = {DeltaOmega_over_geff:+.3e}"
            )
            ax2.text(
                1.02,  # just outside the axes, to the right
                0.98,
                metrics_text_center,
                transform=ax2.transAxes,
                va="top",
                ha="left",
                fontsize=7,
                family="monospace",
                bbox=dict(boxstyle="round", alpha=0.08),
                clip_on=False,  # don't clip at axes boundary
            )

            ax2.legend(fontsize=7, loc="upper left")
            fig2.tight_layout()
            pdf.savefig(fig2)
            plt.close(fig2)

            # ---------------- Plot 3: sea-center control envelope --------------
            fig3, ax3 = plt.subplots()
            fig3.subplots_adjust(right=0.75)

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
            ax3.set_ylabel(r"$\langle I^z_{\mathrm{sea}}\rangle$")
            ax3.set_title(
                f"δ_A = {delta_hz:+.1f} Hz (coarse envelope, sea-center control)"
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
                f"t_sea-center            = {t_off_sea_center:+.3f}\n"
                f"contrast_sea_center     = {contrast_sea_center:+.3e}"
            )
            ax3.text(
                1.02,
                0.98,
                metrics_text_shell,
                transform=ax3.transAxes,
                va="top",
                ha="left",
                fontsize=7,
                family="monospace",
                bbox=dict(boxstyle="round", alpha=0.08),
                clip_on=False,
            )

            ax3.legend(fontsize=7, loc="upper left")
            fig3.tight_layout()
            pdf.savefig(fig3)
            plt.close(fig3)

            # ---------------- Plot 4: state norm (if available) ----------------
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
                    f"δ_A = {delta_hz:+.1f} Hz (state norm, rare at center)"
                )
                axN.legend()
                figN.tight_layout()
                pdf.savefig(figN)
                plt.close(figN)

        # --- Summary table page ---
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")

        col_labels = [
            "δ_A (Hz)",
            "slope_off(center)",
            "t_off(center)",
            "slope_on(center)",
            "t_on(center)",
            "contrast_rare_center",
            "slope_sea-center",
            "t_sea-center",
            "contrast_sea_center",
        ]

        table_vals: List[List[str]] = []
        for row in new_sweep_results:
            table_vals.append(
                [
                    f"{row['delta_Hz']:+.1f}",
                    f"{row['I_z_slope_off_center']:+.3e}",
                    f"{row['t_off_center']:+.3f}",
                    f"{row['I_z_slope_on_center']:+.3e}",
                    f"{row['t_on_center']:+.3f}",
                    f"{row['contrast_rare_center']:+.3e}",
                    f"{row['I_z_slope_off_sea_center']:+.3e}",
                    f"{row['t_off_sea_center']:+.3f}",
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

        # --- Final plot: contrast_rare_center vs ΔΩ/|g_eff| ---
        x_vals = np.array(
            [row.get("DeltaOmega_over_geff", np.nan) for row in new_sweep_results],
            dtype=float,
        )
        y_vals = np.array(
            [row.get("contrast_rare_center", np.nan) for row in new_sweep_results],
            dtype=float,
        )

        mask = ~np.isnan(x_vals) & ~np.isnan(y_vals)
        x_vals = x_vals[mask]
        y_vals = y_vals[mask]

        if x_vals.size > 0:
            order = np.argsort(x_vals)
            x_sorted = x_vals[order]
            y_sorted = y_vals[order]

            figc, axc = plt.subplots(figsize=(6, 4))
            axc.plot(x_sorted, y_sorted, "o-", markersize=4)

            axc.set_xlabel(r"$\Delta\Omega / |g_{\mathrm{eff}}|$")
            axc.set_ylabel(r"$\mathrm{contrast\_rare\_center}$")
            axc.set_title(
                r"Rare-center contrast vs $\Delta\Omega/|g_{\mathrm{eff}}|$ (reprocessed)"
            )
            axc.grid(True, alpha=0.3)

            figc.tight_layout()
            pdf.savefig(figc)
            plt.close(figc)

    # --- Write reprocessed summary JSON ---
    summary_reprocessed = {
        "global_params": global_params,
        "sweep_results": new_sweep_results,
        "coarse_window_reprocessed": int(window),
    }

    with open(new_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_reprocessed, f, indent=2, default=float)

    print("------------------------------------------------------------")
    print("Reprocessing complete.")
    print(f"  New PDF: {new_pdf}")
    print(f"  New summary JSON: {new_summary_json}")
    print("------------------------------------------------------------")

    return new_pdf


def _choose_sweep_dir(initial_dir: str = "results") -> str | None:
    """
    Open a folder picker dialog to choose a sweep directory.
    """
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(
        initialdir=os.path.abspath(initial_dir),
        title="Select sea-detuning sweep directory",
    )
    root.destroy()
    if not folder:
        return None
    return folder


if __name__ == "__main__":
    sweep_dir = _choose_sweep_dir()
    if not sweep_dir:
        print("No directory selected. Exiting.")
    else:
        try:
            window_str = input("Coarse-grain window size (integer, default 50): ").strip()
            if window_str:
                window = int(window_str)
            else:
                window = 50
        except Exception:
            print("Invalid window size, using default 50.")
            window = 50

        reprocess_sweep(sweep_dir, window=window)
