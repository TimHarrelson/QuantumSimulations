# reprocess_sweep_results.py (GUI picker version)

"""
Reprocess an existing sea-detuning sweep without re-running simulations.

- Lets you pick a sweep directory via a folder picker (starting in 'results/').
- Reloads saved time series (time_and_obs_off/on.npz).
- Re-applies coarse-graining with a user-selected window size.
- Re-fits pseudo-T1 curves.
- Writes a new PDF report called
    sea_detuning_report_reprocessed[_winNN].pdf
  in the chosen sweep folder.
"""

import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# NEW: GUI folder picker
import tkinter as tk
from tkinter import filedialog

def fit_exponential_decay(t: np.ndarray, y: np.ndarray) -> Optional[Dict[str, float]]:
    """
    Fit y(t) ≈ C + A * exp(-t / T) by:
      1) Estimate C as the mean of the last 10% of samples.
      2) Fit ln(y - C) = ln(A) - t / T via linear least squares.

    Returns dict {A, C, T} or None if fit fails (e.g. non-positive values).
    """
    if t.ndim != 1 or y.ndim != 1 or len(t) != len(y):
        raise ValueError("t and y must be 1D arrays of the same length.")

    n = len(t)
    if n < 10:
        return None

    # Estimate asymptote C from the tail
    n_tail = max(1, n // 10)
    C = float(np.mean(y[-n_tail:]))

    y_eff = y - C
    mask = y_eff > 0.0
    if mask.sum() < 5:
        return None

    t_fit = t[mask]
    y_fit = y_eff[mask]
    ln_y = np.log(y_fit)

    # Linear least squares: ln_y = ln(A) - t/T
    A_mat = np.vstack([np.ones_like(t_fit), -t_fit]).T
    coeffs, *_ = np.linalg.lstsq(A_mat, ln_y, rcond=None)
    lnA, invT = coeffs
    if invT <= 0.0:
        return None

    A0 = float(np.exp(lnA))
    T = float(1.0 / invT)
    return {"A": A0, "C": C, "T": T}


def coarse_grain(t: np.ndarray, y: np.ndarray, window: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    """
    Block-average y(t) over 'window' points to obtain a smooth envelope.
    """
    n = (len(t) // window) * window
    if n == 0 or window <= 1:
        return t, y
    t_block = t[:n].reshape(-1, window)
    y_block = y[:n].reshape(-1, window)
    t_coarse = t_block.mean(axis=1)
    y_coarse = y_block.mean(axis=1)
    return t_coarse, y_coarse


# ---------------------------------------------------------------------------
# Reprocessing logic
# ---------------------------------------------------------------------------

def det_label_from_delta(delta_hz: float) -> str:
    """
    Reproduce the folder naming convention used in sweep_sea_detuning.py:
      delta_{+1000.0}Hz -> 'delta_p1000.0Hz'
      delta_{-1000.0}Hz -> 'delta_m1000.0Hz'
    """
    return f"delta_{delta_hz:+.1f}Hz".replace("+", "p").replace("-", "m")


def reprocess_sweep(base_dir: str, window: int = 50) -> str:
    """
    Reprocess an existing sweep in base_dir and create a new PDF with
    updated coarse-graining.

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
    sweep_results = summary.get("sweep_results", [])

    # Set up new PDF name
    orig_pdf = os.path.join(base_dir, "sea_detuning_report.pdf")
    if window > 0:
        new_pdf = os.path.join(
            base_dir, f"sea_detuning_report_reprocessed_win{window}.pdf"
        )
    else:
        new_pdf = os.path.join(base_dir, "sea_detuning_report_reprocessed.pdf")

    print(f"Reprocessing sweep in: {base_dir}")
    print(f"  Original PDF : {orig_pdf if os.path.exists(orig_pdf) else '(not found)'}")
    print(f"  New PDF      : {new_pdf}")
    print(f"  Envelope window size: {window}")
    print("------------------------------------------------------------")

    # Helper to format T for labels
    def fmt_T(fit: Optional[Dict[str, float]]) -> str:
        if fit is None or fit.get("T", None) is None:
            return "NA"
        return f"{fit['T']:.3g}"

    # Rebuild a fresh summary with new fits
    new_sweep_results = []

    with PdfPages(new_pdf) as pdf:
        # --- Page 1: global parameter summary (copied from original info) ---
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4-ish
        ax.axis("off")
        lines = []
        lines.append("Sea detuning sweep report (REPROCESSED)")
        lines.append("")
        lines.append("Global parameters (copied from original sweep):")
        for key, val in global_params.items():
            # Pretty-print a subset in a semi-readable way
            if isinstance(val, float):
                lines.append(f"  {key:20s} = {val:.6g}")
            else:
                lines.append(f"  {key:20s} = {val}")
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

        # --- Per-detuning pages ---
        for row in sweep_results:
            delta_hz = float(row["delta_Hz"])
            det_dir = os.path.join(base_dir, det_label_from_delta(delta_hz))
            if not os.path.isdir(det_dir):
                print(f"Warning: directory for δ_A={delta_hz:+.1f} Hz not found, skipping.")
                continue

            print(f"Reprocessing δ_A = {delta_hz:+.1f} Hz ...")

            off_npz = os.path.join(det_dir, "time_and_obs_off.npz")
            on_npz = os.path.join(det_dir, "time_and_obs_on.npz")

            data_off = np.load(off_npz)
            data_on = np.load(on_npz)

            t_off = data_off["t"]
            t_on = data_on["t"]

            Iz_off = data_off["Iz_sea"]
            Iz_on = data_on["Iz_sea"]

            Ix_off = data_off["Ix_sea"]
            Iy_off = data_off["Iy_sea"]

            norm_off = data_off.get("state_norm", None)
            norm_on = data_on.get("state_norm", None)

            # Recompute fits from full-resolution data
            fit_off = fit_exponential_decay(t_off, Iz_off)
            fit_on = fit_exponential_decay(t_on, Iz_on)

            new_sweep_results.append(
                {
                    "delta_Hz": delta_hz,
                    "fit_Iz_sea_off": fit_off,
                    "fit_Iz_sea_on": fit_on,
                }
            )

            # ---------------- Plot 1: Iz_sea full resolution -----------------
            fig1, ax1 = plt.subplots()
            ax1.plot(t_off, Iz_off,
                     label=r"$\langle I^z_{\mathrm{sea}}\rangle$, rare OFF")
            ax1.plot(t_on, Iz_on,
                     label=r"$\langle I^z_{\mathrm{sea}}\rangle$, rare ON")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel(r"$\langle I^z_{\mathrm{sea}}\rangle$")
            ax1.set_title(f"δ_A = {delta_hz:+.1f} Hz")
            ax1.legend()
            fig1.tight_layout()
            pdf.savefig(fig1)
            plt.close(fig1)

            # ---------------- Plot 2: envelope + fits -----------------------
            fig2, ax2 = plt.subplots()

            t_off_c, Iz_off_c = coarse_grain(t_off, Iz_off, window=window)
            t_on_c, Iz_on_c = coarse_grain(t_on, Iz_on, window=window)

            ax2.plot(
                t_off_c,
                Iz_off_c,
                "o-",
                markersize=3,
                label=r"$\langle I^z_{\mathrm{sea}}\rangle$, rare OFF (envelope)",
            )
            ax2.plot(
                t_on_c,
                Iz_on_c,
                "o--",
                markersize=3,
                label=r"$\langle I^z_{\mathrm{sea}}\rangle$, rare ON (envelope)",
            )

            if fit_off is not None:
                f = fit_off
                y_model = f["C"] + f["A"] * np.exp(-t_off / f["T"])
                ax2.plot(
                    t_off,
                    y_model,
                    "-",
                    linewidth=1.5,
                    label=f"Fit OFF (T ≈ {fmt_T(f)} s)",
                )
            if fit_on is not None:
                f = fit_on
                y_model = f["C"] + f["A"] * np.exp(-t_on / f["T"])
                ax2.plot(
                    t_on,
                    y_model,
                    "--",
                    linewidth=1.5,
                    label=f"Fit ON (T ≈ {fmt_T(f)} s)",
                )

            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel(r"$\langle I^z_{\mathrm{sea}}\rangle$")
            ax2.set_title(
                f"δ_A = {delta_hz:+.1f} Hz (pseudo $T_1$ envelope, window={window})"
            )

            y_min = min(Iz_off_c.min(), Iz_on_c.min())
            y_max = max(Iz_off_c.max(), Iz_on_c.max())
            if y_max > y_min:
                pad = 0.05 * (y_max - y_min)
                ax2.set_ylim(y_min - pad, y_max + pad)

            ax2.legend(fontsize=8)
            fig2.tight_layout()
            pdf.savefig(fig2)
            plt.close(fig2)

            # ---------------- Plot 3: Ix/Iy (rare OFF) ----------------------
            fig3, ax3 = plt.subplots()
            ax3.plot(t_off, Ix_off, label=r"$\langle I^x_{\mathrm{sea}}\rangle$")
            ax3.plot(t_off, Iy_off, label=r"$\langle I^y_{\mathrm{sea}}\rangle$")
            ax3.set_xlabel("Time (s)")
            ax3.set_ylabel(r"$\langle I^{x,y}_{\mathrm{sea}}\rangle$")
            ax3.set_title(f"δ_A = {delta_hz:+.1f} Hz (rare drive OFF)")
            ax3.legend(fontsize=8)
            fig3.tight_layout()
            pdf.savefig(fig3)
            plt.close(fig3)

            # ---------------- Plot 4: state norm ----------------------------
            if norm_off is not None and norm_on is not None:
                figN, axN = plt.subplots()
                axN.plot(t_off, norm_off, label=r"$\|\psi(t)\|$, rare OFF")
                axN.plot(t_on, norm_on, label=r"$\|\psi(t)\|$, rare ON")
                axN.set_xlabel("Time (s)")
                axN.set_ylabel(r"State norm $\|\psi\|$")
                axN.set_title(f"δ_A = {delta_hz:+.1f} Hz (state norm)")
                axN.legend()
                figN.tight_layout()
                pdf.savefig(figN)
                plt.close(figN)

        # ------------- Final page: new T-fit summary -----------------------
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        lines = []
        lines.append("Reprocessed T-like decay fits from ⟨I^z_sea⟩ traces")
        lines.append(f"(coarse-grain window = {window})")
        lines.append("")
        header = "delta_Hz      T_Iz_sea_off       T_Iz_sea_on"
        lines.append(header)
        lines.append("-" * len(header))

        def fmt_T_line(fit: Optional[Dict[str, float]]) -> str:
            if fit is None or fit.get("T", None) is None:
                return "NA".ljust(18)
            return f"{fit['T']:.3g}".ljust(18)

        for row in new_sweep_results:
            delta = row["delta_Hz"]
            l = f"{delta:+9.1f}  "
            l += fmt_T_line(row["fit_Iz_sea_off"])
            l += fmt_T_line(row["fit_Iz_sea_on"])
            lines.append(l)

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

    print("Reprocessing complete.")
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
