"""
Reprocess an existing sea-detuning sweep folder produced by sweep_sea_detuning.py,
but replace the *linear slope* envelope metric with two nonlinear relaxation metrics:

  - tau: time constant from an exponential-to-plateau fit on the coarse-grained envelope
  - t90: time to reach 90% relaxation toward the plateau (model-free)

Outputs:
  - A PDF report similar in structure to the original per-detuning PDF, but annotated
    with tau/t90 instead of slope/contrast
  - Per-plot PNGs saved into a subfolder (default: graphs_exponential)

Usage:
  python reprocess_detuning_sweep_exponential.py --root "path/to/sea_detuning_sweep_YYYYMMDD_HHMMSS"

If --root is omitted, a UI folder picker will open (Windows-friendly).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# -----------------------------
# UI folder picker (optional)
# -----------------------------
def pick_folder_ui(title: str = "Select sweep folder") -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title=title, mustexist=True)
    root.destroy()
    if not folder:
        return None
    return folder


# -----------------------------
# I/O helpers
# -----------------------------
def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Coarse graining
# -----------------------------
def coarse_grain(t: np.ndarray, y: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    n = (len(t) // window) * window
    if n == 0 or window <= 1:
        return t, y
    t_block = t[:n].reshape(-1, window)
    y_block = y[:n].reshape(-1, window)
    return t_block.mean(axis=1), y_block.mean(axis=1)


# -----------------------------
# Robust stats
# -----------------------------
def mad_sigma(x: np.ndarray) -> float:
    """
    Robust sigma estimate using MAD: sigma ≈ 1.4826 * median(|x - median(x)|).
    """
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return 1.4826 * mad


@dataclass
class RelaxFit:
    status: str
    I_inf: float
    tau_s: float
    A: float
    t0_s: float
    r2: float
    n_fit: int
    sigma_noise: float
    R0: float
    t90_s: float


def _interp_threshold_time(t: np.ndarray, r: np.ndarray, thr: float) -> float:
    """
    First time when r <= thr using linear interpolation between samples.
    Returns NaN if never crosses.
    Assumes r >= 0.
    """
    if t.size == 0:
        return float("nan")
    below = np.where(r <= thr)[0]
    if below.size == 0:
        return float("nan")
    i = int(below[0])
    if i == 0:
        return float(t[0])
    t0, t1 = float(t[i - 1]), float(t[i])
    r0, r1 = float(r[i - 1]), float(r[i])
    if not (math.isfinite(r0) and math.isfinite(r1)) or (r1 == r0):
        return float(t1)
    # interpolate r(t) linearly
    frac = (thr - r0) / (r1 - r0)
    frac = max(0.0, min(1.0, frac))
    return t0 + frac * (t1 - t0)


def fit_exponential_to_plateau(
    t: np.ndarray,
    y: np.ndarray,
    *,
    plateau_frac: float = 0.15,
    early_frac: float = 0.10,
    alpha_noise: float = 3.0,
    min_points: int = 6,
) -> RelaxFit:
    """
    Fit y(t) to an exponential approach to a plateau:

        y(t) = I_inf + sign0 * A * exp(-(t - t0)/tau)

    We estimate I_inf robustly from the last plateau_frac of samples (median),
    and fit tau by linear regression on ln(residual) using points where:
      - residual has the same sign as the initial residual
      - residual magnitude is above max(0.1*R0, alpha_noise*sigma_noise)

    t90 is computed model-free from |y - I_inf| reaching 0.1*R0.

    Returns RelaxFit with status in {"OK","FLAT","CENSORED","BAD_FIT","TOO_FEW"}.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    mask_finite = np.isfinite(t) & np.isfinite(y)
    t = t[mask_finite]
    y = y[mask_finite]

    if t.size < 4:
        return RelaxFit(
            status="TOO_FEW",
            I_inf=float("nan"),
            tau_s=float("nan"),
            A=float("nan"),
            t0_s=float("nan"),
            r2=float("nan"),
            n_fit=0,
            sigma_noise=float("nan"),
            R0=float("nan"),
            t90_s=float("nan"),
        )

    # Estimate plateau I_inf from late samples
    k_plateau = max(3, int(round(plateau_frac * t.size)))
    y_late = y[-k_plateau:]
    I_inf = float(np.median(y_late))

    # Noise estimate from late window (about plateau)
    sigma_noise = float(mad_sigma(y_late))
    if not math.isfinite(sigma_noise):
        sigma_noise = float("nan")

    # Residuals to plateau
    res = y - I_inf
    abs_res = np.abs(res)

    # Initial residual amplitude from early samples
    k_early = max(3, int(round(early_frac * t.size)))
    R0 = float(np.median(abs_res[:k_early]))

    # t90 (model-free)
    if R0 <= 0.0 or not math.isfinite(R0):
        t90_s = float("nan")
    else:
        thr90 = 0.1 * R0
        t90_s = _interp_threshold_time(t, abs_res, thr90)

    # Flat classification: starts at plateau within noise
    if math.isfinite(sigma_noise) and R0 <= alpha_noise * sigma_noise:
        return RelaxFit(
            status="FLAT",
            I_inf=I_inf,
            tau_s=float("nan"),
            A=float("nan"),
            t0_s=float(t[0]),
            r2=float("nan"),
            n_fit=0,
            sigma_noise=sigma_noise,
            R0=R0,
            t90_s=0.0 if math.isfinite(t[0]) else float("nan"),
        )

    # Choose sign based on early residual median
    sign0 = float(np.sign(np.median(res[:k_early])))
    if sign0 == 0.0:
        # Fall back to first non-zero residual sign
        nz = res[np.nonzero(res)]
        sign0 = float(np.sign(nz[0])) if nz.size else 1.0

    # Work with signed-positive residual
    rpos = res * sign0

    # Fit window: points with rpos > threshold
    if not math.isfinite(R0) or R0 <= 0.0:
        thr_fit = float("nan")
    else:
        thr_fit = max(0.1 * R0, (alpha_noise * sigma_noise) if math.isfinite(sigma_noise) else 0.0)

    fit_mask = np.isfinite(rpos) & (rpos > thr_fit)
    t_fit = t[fit_mask]
    r_fit = rpos[fit_mask]

    if t_fit.size < min_points:
        # Not enough usable points for tau
        status = "CENSORED" if math.isfinite(t90_s) is False else "TOO_FEW"
        return RelaxFit(
            status=status,
            I_inf=I_inf,
            tau_s=float("nan"),
            A=float("nan"),
            t0_s=float(t[0]),
            r2=float("nan"),
            n_fit=int(t_fit.size),
            sigma_noise=sigma_noise,
            R0=R0,
            t90_s=t90_s,
        )

    # Linear fit: ln(r) = ln(A) - (t - t0)/tau  => slope = -1/tau
    # Use absolute time to keep it simple
    ln_r = np.log(r_fit)
    slope, intercept = np.polyfit(t_fit, ln_r, 1)

    if not math.isfinite(slope) or slope >= 0.0:
        return RelaxFit(
            status="BAD_FIT",
            I_inf=I_inf,
            tau_s=float("nan"),
            A=float("nan"),
            t0_s=float(t_fit[0]),
            r2=float("nan"),
            n_fit=int(t_fit.size),
            sigma_noise=sigma_noise,
            R0=R0,
            t90_s=t90_s,
        )

    tau_s = float(-1.0 / slope)
    A = float(math.exp(intercept))

    # R^2 in ln-space on fit points
    ln_pred = slope * t_fit + intercept
    ss_res = float(np.sum((ln_r - ln_pred) ** 2))
    ss_tot = float(np.sum((ln_r - float(np.mean(ln_r))) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")

    return RelaxFit(
        status="OK",
        I_inf=I_inf,
        tau_s=tau_s,
        A=A,
        t0_s=float(t_fit[0]),
        r2=r2,
        n_fit=int(t_fit.size),
        sigma_noise=sigma_noise,
        R0=R0,
        t90_s=t90_s,
    )


def eval_fit_curve(t: np.ndarray, fit: RelaxFit, sign0: float, t_ref: float) -> np.ndarray:
    """
    Evaluate y_fit(t) = I_inf + sign0 * A * exp(-(t - 0)/tau)
    using A fitted against absolute t in ln-space. For plotting only.
    """
    if not (math.isfinite(fit.I_inf) and math.isfinite(fit.A) and math.isfinite(fit.tau_s)):
        return np.full_like(t, np.nan, dtype=float)
    # Because we fit ln(r) = slope * t + intercept with intercept = ln(A),
    # r(t) = A * exp(slope * t) = A * exp(-t/tau).
    r = fit.A * np.exp(-t / fit.tau_s)
    return fit.I_inf + sign0 * r


# -----------------------------
# Reprocessing logic
# -----------------------------
def discover_detuning_dirs(root: str) -> List[str]:
    det_dirs: List[str] = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue
        metrics_path = os.path.join(p, "metrics.json")
        if os.path.isfile(metrics_path):
            det_dirs.append(p)
    return sorted(det_dirs)


def load_npz(det_dir: str, tag: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    npz_path = os.path.join(det_dir, f"time_and_obs_{tag}.npz")
    data = np.load(npz_path, allow_pickle=False)
    t = data["t"]
    obs = {k: data[k] for k in data.files if k != "t"}
    return t, obs


def make_report(
    *,
    root: str,
    out_pdf: str,
    graphs_dir: str,
    coarse_window: int = 100,
) -> None:
    safe_mkdir(graphs_dir)

    det_dirs = discover_detuning_dirs(root)
    if not det_dirs:
        raise RuntimeError(f"No detuning subfolders with metrics.json found under: {root}")

    # Try to load global params if present
    global_params_path = os.path.join(root, "global_params.json")
    global_params = read_json(global_params_path) if os.path.isfile(global_params_path) else {}

    # Storage for summary pages
    rows: List[Dict[str, Any]] = []

    with PdfPages(out_pdf) as pdf:
        # -------- Global parameter page --------
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        lines: List[str] = []
        lines.append("Sea detuning sweep report (REPROCESSED: exponential-to-plateau metrics)")
        lines.append("")
        lines.append(f"Source sweep folder: {root}")
        lines.append(f"Generated: {_dt.datetime.now().isoformat(timespec='seconds')}")
        lines.append("")
        if global_params:
            lines.append("Global params (from global_params.json):")
            keys_show = [
                "f_Az_Hz","f_Rz_Hz","f1A_Hz","f1R_Hz","t_final_s","steps","n_sea",
                "target_sea_detuning","coarse_window","avg_b_AR_Hz","rms_b_AR_Hz",
            ]
            for k in keys_show:
                if k in global_params:
                    lines.append(f"  {k:20s} = {global_params[k]}")
            lines.append("")
        lines.append(f"Coarse window (reprocess): {coarse_window}")
        lines.append("")
        lines.append("Metrics computed per detuning (rare-at-center):")
        lines.append("  - tau_off_center, tau_on_center   (s)")
        lines.append("  - t90_off_center, t90_on_center   (s)")
        lines.append("  - eta = ΔΩ/|g_eff| (from metrics.json)")
        ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", family="monospace")
        pdf.savefig(fig)
        fig.savefig(os.path.join(graphs_dir, "00_global_params.png"), dpi=300)
        plt.close(fig)

        # -------- Per-detuning pages --------
        for det_i, det_dir in enumerate(det_dirs):
            metrics = read_json(os.path.join(det_dir, "metrics.json"))
            delta_Hz = float(metrics.get("delta_Hz", float("nan")))
            eta = float(metrics.get("DeltaOmega_over_geff", float("nan")))

            # Load traces
            t_off, obs_off = load_npz(det_dir, "center_off")
            t_on, obs_on = load_npz(det_dir, "center_on")

            # Coarse envelopes
            t_c_off, y_c_off = coarse_grain(t_off, obs_off["Iz_sea"], coarse_window)
            t_c_on, y_c_on = coarse_grain(t_on, obs_on["Iz_sea"], coarse_window)

            # Fits
            fit_off = fit_exponential_to_plateau(t_c_off, y_c_off)
            fit_on = fit_exponential_to_plateau(t_c_on, y_c_on)

            # Determine sign0 for plotting residual direction (based on early residual)
            def sign0_for(t_c: np.ndarray, y_c: np.ndarray, I_inf: float) -> float:
                k_early = max(3, int(round(0.10 * t_c.size)))
                res = y_c - I_inf
                s = float(np.sign(np.median(res[:k_early])))
                if s == 0.0:
                    nz = res[np.nonzero(res)]
                    s = float(np.sign(nz[0])) if nz.size else 1.0
                return s

            s_off = sign0_for(t_c_off, y_c_off, fit_off.I_inf) if np.isfinite(fit_off.I_inf) else 1.0
            s_on = sign0_for(t_c_on, y_c_on, fit_on.I_inf) if np.isfinite(fit_on.I_inf) else 1.0

            # 1) Full Iz vs time (raw)
            fig1, ax1 = plt.subplots()
            ax1.plot(t_off, obs_off["Iz_sea"], label=r"$\langle I^z_{\mathrm{sea}}\rangle$, rare OFF (center)")
            ax1.plot(t_on,  obs_on["Iz_sea"],  label=r"$\langle I^z_{\mathrm{sea}}\rangle$, rare ON (center)")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel(r"$\langle I^z_{\mathrm{sea}}\rangle$")
            ax1.set_title(f"δ_A = {delta_Hz:+.1f} Hz (rare at center)")
            ax1.legend(fontsize=8)
            fig1.tight_layout()
            p1 = os.path.join(graphs_dir, f"{det_i:03d}_Iz_raw.png")
            fig1.savefig(p1, dpi=300)
            pdf.savefig(fig1)
            plt.close(fig1)

            # 2) Coarse envelopes + exponential fits + tau/t90 annotations
            fig2, ax2 = plt.subplots()
            fig2.subplots_adjust(right=0.78)

            ax2.plot(t_c_off, y_c_off, "o-", markersize=3, label="OFF (envelope)")
            ax2.plot(t_c_on,  y_c_on,  "o--", markersize=3, label="ON (envelope)")

            y_fit_off = eval_fit_curve(t_c_off, fit_off, s_off, float(t_c_off[0]))
            y_fit_on  = eval_fit_curve(t_c_on,  fit_on,  s_on,  float(t_c_on[0]))

            if np.any(np.isfinite(y_fit_off)):
                ax2.plot(t_c_off, y_fit_off, "-", linewidth=2, label="OFF exp fit")
            if np.any(np.isfinite(y_fit_on)):
                ax2.plot(t_c_on, y_fit_on, "--", linewidth=2, label="ON exp fit")

            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel(r"$\langle I^z_{\mathrm{sea}}\rangle$")
            ax2.set_title(f"δ_A = {delta_Hz:+.1f} Hz (coarse + exp-to-plateau fits)")

            # y zoom
            all_env = np.concatenate([y_c_off, y_c_on])
            y_min, y_max = float(np.min(all_env)), float(np.max(all_env))
            if y_max > y_min:
                pad = 0.06 * (y_max - y_min)
                ax2.set_ylim(y_min - pad, y_max + pad)

            # Side text
            def fmt_fit(name: str, fit: RelaxFit) -> str:
                tau = fit.tau_s
                t90 = fit.t90_s
                return (
                    f"{name}:\n"
                    f"  status   = {fit.status}\n"
                    f"  tau (s)  = {tau:.3e}" + ("" if math.isfinite(tau) else "") + "\n"
                    f"  t90 (s)  = {t90:.3e}" + ("" if math.isfinite(t90) else "") + "\n"
                    f"  I_inf    = {fit.I_inf:+.3e}\n"
                    f"  R0       = {fit.R0:.3e}\n"
                    f"  sigma    = {fit.sigma_noise:.3e}\n"
                    f"  R2(ln)   = {fit.r2:.3f}\n"
                    f"  n_fit    = {fit.n_fit}\n"
                )

            side = (
                fmt_fit("OFF", fit_off)
                + "\n"
                + fmt_fit("ON", fit_on)
                + "\n"
                + f"eta = ΔΩ/|g_eff| = {eta:+.3e}\n"
            )

            ax2.text(
                1.02, 0.98, side,
                transform=ax2.transAxes, va="top", ha="left",
                fontsize=7, family="monospace",
                bbox=dict(boxstyle="round", alpha=0.08),
                clip_on=False,
            )

            ax2.legend(fontsize=8, loc="upper left")
            fig2.tight_layout()
            p2 = os.path.join(graphs_dir, f"{det_i:03d}_Iz_env_fit.png")
            fig2.savefig(p2, dpi=300)
            pdf.savefig(fig2)
            plt.close(fig2)

            # 3) State norm (raw)
            if "state_norm" in obs_off and "state_norm" in obs_on:
                fig3, ax3 = plt.subplots()
                ax3.plot(t_off, obs_off["state_norm"], label=r"$\|\psi(t)\|$, OFF (center)")
                ax3.plot(t_on,  obs_on["state_norm"],  label=r"$\|\psi(t)\|$, ON (center)")
                ax3.set_xlabel("Time (s)")
                ax3.set_ylabel(r"State norm $\|\psi\|$")
                ax3.set_title(f"δ_A = {delta_Hz:+.1f} Hz (state norm)")
                ax3.legend(fontsize=8)
                fig3.tight_layout()
                p3 = os.path.join(graphs_dir, f"{det_i:03d}_state_norm.png")
                fig3.savefig(p3, dpi=300)
                pdf.savefig(fig3)
                plt.close(fig3)

            # Record row for summary
            rows.append({
                "delta_Hz": delta_Hz,
                "eta": eta,
                "tau_off_s": fit_off.tau_s,
                "tau_on_s": fit_on.tau_s,
                "t90_off_s": fit_off.t90_s,
                "t90_on_s": fit_on.t90_s,
                "status_off": fit_off.status,
                "status_on": fit_on.status,
                "R2ln_off": fit_off.r2,
                "R2ln_on": fit_on.r2,
            })

        # -------- Summary table page --------
        figT, axT = plt.subplots(figsize=(8.27, 11.69))
        axT.axis("off")
        axT.set_title("Relaxation metrics from exponential-to-plateau fits (coarse envelopes)", pad=20)

        col_labels = [
            "δ_A (Hz)", "eta", "tau_off (s)", "tau_on (s)", "t90_off (s)", "t90_on (s)", "status_off", "status_on"
        ]

        table_vals: List[List[str]] = []
        for r in rows:
            table_vals.append([
                f"{r['delta_Hz']:+.1f}",
                f"{r['eta']:+.3e}",
                f"{r['tau_off_s']:.3e}" if math.isfinite(r["tau_off_s"]) else "NaN",
                f"{r['tau_on_s']:.3e}" if math.isfinite(r["tau_on_s"]) else "NaN",
                f"{r['t90_off_s']:.3e}" if math.isfinite(r["t90_off_s"]) else "NaN",
                f"{r['t90_on_s']:.3e}" if math.isfinite(r["t90_on_s"]) else "NaN",
                str(r["status_off"]),
                str(r["status_on"]),
            ])

        table = axT.table(cellText=table_vals, colLabels=col_labels, loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(6)
        table.scale(1.0, 1.25)

        pdf.savefig(figT)
        figT.savefig(os.path.join(graphs_dir, "summary_table.png"), dpi=300)
        plt.close(figT)

        # -------- Summary plots: tau vs eta and t90 vs eta --------
        # Sort by eta
        eta_arr = np.array([r["eta"] for r in rows], dtype=float)
        order = np.argsort(eta_arr)
        eta_s = eta_arr[order]

        tau_off = np.array([rows[i]["tau_off_s"] for i in order], dtype=float)
        tau_on  = np.array([rows[i]["tau_on_s"] for i in order], dtype=float)

        t90_off = np.array([rows[i]["t90_off_s"] for i in order], dtype=float)
        t90_on  = np.array([rows[i]["t90_on_s"] for i in order], dtype=float)

        # Plot helper
        def _plot_metric_vs_eta(metric_off: np.ndarray, metric_on: np.ndarray, ylabel: str, title: str, fname: str):
            fig, ax = plt.subplots(figsize=(6.5, 4.0))
            # Use only finite points
            m_off = np.isfinite(eta_s) & np.isfinite(metric_off)
            m_on  = np.isfinite(eta_s) & np.isfinite(metric_on)

            if np.any(m_off):
                ax.plot(eta_s[m_off], metric_off[m_off], "o-", markersize=4, label="OFF (center)")
            if np.any(m_on):
                ax.plot(eta_s[m_on],  metric_on[m_on],  "o--", markersize=4, label="ON (center)")

            ax.set_xlabel(r"$\eta = \Delta\Omega / |g_{\mathrm{eff}}|$")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            out_png = os.path.join(graphs_dir, fname)
            fig.savefig(out_png, dpi=300)
            pdf.savefig(fig)
            plt.close(fig)

        _plot_metric_vs_eta(
            tau_off, tau_on,
            ylabel=r"$\tau$ (s)",
            title=r"Exponential-to-plateau time constant vs $\eta$",
            fname="tau_vs_eta.png",
        )

        _plot_metric_vs_eta(
            t90_off, t90_on,
            ylabel=r"$t_{90}$ (s)",
            title=r"Time-to-90% relaxation vs $\eta$",
            fname="t90_vs_eta.png",
        )

    # Save a machine-readable summary
    out_json = os.path.join(os.path.dirname(out_pdf), "summary_exponential_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"root": root, "rows": rows}, f, indent=2, default=float)

    print("------------------------------------------------------------")
    print("Reprocessing complete.")
    print(f"  Source sweep folder : {root}")
    print(f"  Output PDF          : {out_pdf}")
    print(f"  Graphs folder       : {graphs_dir}")
    print(f"  Summary JSON        : {out_json}")
    print("------------------------------------------------------------")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reprocess a detuning sweep using exponential-to-plateau metrics (tau and t90).")
    parser.add_argument("--root", type=str, default=None, help="Path to the sweep folder (contains detuning subfolders with metrics.json).")
    parser.add_argument("--coarse-window", type=int, default=None, help="Override coarse window (block size) for envelope coarse-graining.")
    args = parser.parse_args()

    root = args.root
    if not root:
        root = pick_folder_ui("Select the sweep output folder to reprocess")
        if not root:
            print("No folder selected. Exiting.")
            return

    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise RuntimeError(f"Not a directory: {root}")

    # Determine coarse window default: prefer global_params.json if present
    coarse_window = 100
    gp_path = os.path.join(root, "global_params.json")
    if os.path.isfile(gp_path):
        try:
            gp = read_json(gp_path)
            if isinstance(gp, dict) and "coarse_window" in gp:
                coarse_window = int(gp["coarse_window"])
        except Exception:
            pass
    if args.coarse_window is not None:
        coarse_window = int(args.coarse_window)

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_pdf = os.path.join(root, f"sea_detuning_report_exponential_{ts}.pdf")
    graphs_dir = os.path.join(root, f"graphs_exponential_{ts}")

    make_report(root=root, out_pdf=out_pdf, graphs_dir=graphs_dir, coarse_window=coarse_window)


if __name__ == "__main__":
    main()
