"""
Sweep the sea detuning δ_A = f_Az - f_rf,A for a Ga (sea) / Al (rare) system.

For each detuning, run two simulations:
  1) rare drive OFF
  2) rare drive ON

For each detuning:
  - Save all raw data and parameters to disk.
  - Produce three plots:
      (a) ⟨I^z_sea⟩ vs time (rare OFF & ON).
      (b) Same, with exponential T-like fits overlaid on a coarse-grained
          envelope of ⟨I^z_sea⟩ (to visualize the pseudo T1).
      (c) ⟨I^x_sea⟩ and ⟨I^y_sea⟩ vs time (rare OFF only).

A PDF report is generated with:
  - Global parameter summary.
  - T-fit summary table.
  - All plots inserted.

This version uses:
  - Physical gyromagnetic ratios for Ga (effective 69/71 mix) and 27Al.
  - B0 = 3 T.
  - Rabi frequencies f1A = 20 kHz (sea), f1R = 10 kHz (rare).
  - Dipolar scale dipolar_scale = (μ0 / 4π) * ħ with shell_scale ≈ 0.4 nm,
    giving nearest-neighbor dipolar couplings of ≈ 75 Hz.
  - Time window t_final = 3e-2 s, steps = 2000 (good envelope view).
  - Detuning sweep δ_A ∈ [-10 kHz, 10 kHz].
"""

from dataclasses import asdict, replace
import datetime as _dt
import json
import os
from typing import Any, Dict, List, Optional, Sequence
import time

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from dipolar_ensemble_with_rare import (
    DipolarRareParams,
    get_derived_frequencies,
    simulate_rare,
    shell_positions_with_rare_center,
    dipolar_couplings_from_positions,
)


# ---------------------------------------------------------------------------
# Utility: simple exponential decay fit
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Utility: coarse-grain traces to reveal the envelope
# ---------------------------------------------------------------------------

def coarse_grain(t: np.ndarray, y: np.ndarray, window: int = 25):
    """
    Block-average y(t) over 'window' points to obtain a smooth envelope.

    Parameters
    ----------
    t : np.ndarray
        Time points (1D).
    y : np.ndarray
        Observable values (1D).
    window : int
        Number of consecutive samples per averaging block.

    Returns
    -------
    t_coarse, y_coarse : np.ndarray
        Coarse-grained time and observable arrays.
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
# Main sweep function
# ---------------------------------------------------------------------------

def run_sweep_sea_detuning(
    *,
    f_Az: float,
    f1A: float,
    target_sea_detuning: float,
    gamma_sea: float,
    gamma_rare: float,
    sea_detunings_Hz: Sequence[float],
    n_sea: int = 12,
    t_final: float = 3.0e-2,
    steps: int = 2000,
    phi_sea: float = 0.0,
    phi_rare: float = 0.0,
    out_root: str = "results",
    is_spin_three_half = False,
    solver_atol: float | None = None,
    solver_rtol: float | None = None,
    solver_nsteps: int | None = None,
    solver_max_step: float | None = None,
) -> str:
    """
    Run a sweep over sea detunings δ_A = f_Az - f_rf,A.

    Parameters
    ----------
    f_Az : float
        Sea Larmor frequency (Hz).
    f1A : float
        Sea Rabi frequency f1A (Hz).
    target_sea_detuning : float
        The sea detuning (Hz) used to calculate the rare Rabi frequency (f1R) such that it will satisfy
         the resonance condition the model assumes.
    gamma_sea : float
        Sea gyromagnetic ratio (rad·s^-1·T^-1).
    gamma_rare : float
        Rare gyromagnetic ratio (same units).
    sea_detunings_Hz : sequence of float
        Detuning values δ_A = f_Az - f_rf,A (Hz) to sweep over.
    t_final : float
        Final simulation time (seconds).
    steps : int
        Number of time steps at which expectations are recorded.
    phi_sea, phi_rare : float
        RF phases for sea and rare drives (radians).
    out_root : str
        Root directory where result folders will be created.
    is_spin_three_half : bool
        Whether the rare spin is J = 3/2 or I = 1/2

    Returns
    -------
    base_dir : str
        Path to the directory containing this sweep's results.
    """
    f1R = f1R_for_resonance(f1A, target_sea_detuning, 0)
    sea_detunings_Hz = np.asarray(sea_detunings_Hz, dtype=float)
    n_det = len(sea_detunings_Hz)

    # -------- Derive B0 and B1 from frequencies and gammas --------
    # Common B0 chosen so that ω_Az = γ_sea * B0 = 2π f_Az.
    B0_common = 2 * np.pi * f_Az / gamma_sea

    # Implied rare Larmor frequency (from same B0)
    omega_Rz = gamma_rare * B0_common
    f_Rz = omega_Rz / (2 * np.pi)

    # Rabi fields B1 such that ω1 = γ * B1 = 2π f1
    B1_sea = 2 * np.pi * f1A / gamma_sea
    B1_rare = 2 * np.pi * f1R / gamma_rare

    # -------- Physical dipolar scale and shell length scale --------
    # dipolar_scale_SI = (μ0 / 4π) * ħ, in SI units,
    # used in b_ij = dipolar_scale * γ_i γ_j (1 - 3 cos^2 θ) / r^3.
    mu0_over_4pi = 1.0e-7             # N / A^2
    hbar = 1.054571817e-34            # J·s
    dipolar_scale_SI = mu0_over_4pi * hbar  # ~1.05e-41 in SI units
    shell_scale = 0.282393e-9    # meters; ~0.3 nm characteristic length

    # -------- One-shot computation of b_ij couplings for this geometry --------
    positions = shell_positions_with_rare_center(n_sea=n_sea, radius=shell_scale)
    n_total = n_sea + 1
    idx_rare = n_sea

    b = dipolar_couplings_from_positions(
        positions=positions,
        scale=dipolar_scale_SI,
        gamma_sea=gamma_sea,
        gamma_rare=gamma_rare,
    )

    sea_indices = list(range(n_sea))

    # Sea–rare couplings (all i-R pairs)
    sea_rare_vals = np.array([b[i, idx_rare] for i in sea_indices], dtype=float)
    sea_rare_abs = np.abs(sea_rare_vals) / (2 * np.pi)  # Hz

    # Sea–sea couplings (all i<j pairs)
    sea_sea_vals = []
    for i in sea_indices:
        for j in sea_indices:
            if j > i:
                sea_sea_vals.append(b[i, j])
    sea_sea_vals = np.array(sea_sea_vals, dtype=float)
    sea_sea_abs = np.abs(sea_sea_vals) / (2 * np.pi)  # Hz

    # Optional "gamma_rare = 0" sanity check (heteronuclear part should vanish)
    b_gamma0 = dipolar_couplings_from_positions(
        positions=positions,
        scale=dipolar_scale_SI,
        gamma_sea=gamma_sea,
        gamma_rare=0.0,
    )
    sea_rare_gamma0 = np.array([b_gamma0[i, idx_rare] for i in sea_indices], dtype=float)
    max_sea_rare_gamma0_Hz = np.max(np.abs(sea_rare_gamma0)) / (2 * np.pi)

    # Print a concise coupling summary to the console
    print("Estimated dipolar couplings from geometry + physical scales:")
    print(f"  Sea–rare b_ij (all sea ↔ rare), |b| in Hz:")
    print(f"    avg |b_AR| ≈ {sea_rare_abs.mean():.2f} Hz")
    print(f"    min |b_AR| ≈ {sea_rare_abs.min():.2f} Hz")
    print(f"    max |b_AR| ≈ {sea_rare_abs.max():.2f} Hz")
    print("  Sea–sea b_ij (all i<j), |b| in Hz:")
    print(f"    avg |b_AA| ≈ {sea_sea_abs.mean():.2f} Hz")
    print(f"    min |b_AA| ≈ {sea_sea_abs.min():.2f} Hz")
    print(f"    max |b_AA| ≈ {sea_sea_abs.max():.2f} Hz")

    print(
        "  Sanity check (gamma_rare = 0): "
        f"max |b_AR| ≈ {max_sea_rare_gamma0_Hz:.3e} Hz (should be ~0)."
    )
    print("------------------------------------------------------------", flush=True)

    # -------- Output directory & report setup --------
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(out_root, f"sea_detuning_sweep_{timestamp}")
    os.makedirs(base_dir, exist_ok=True)

    pdf_path = os.path.join(base_dir, "sea_detuning_report.pdf")
    summary: Dict[str, Any] = {
        "global_params": {},
        "sweep_results": [],
    }

    # Global parameters that are constant across the sweep
    summary["global_params"] = {
        "f_Az_Hz": float(f_Az),
        "f_Rz_Hz": float(f_Rz),
        "f1A_Hz": float(f1A),
        "f1R_Hz": float(f1R),
        "gamma_sea": float(gamma_sea),
        "gamma_rare": float(gamma_rare),
        "B0_common_T": float(B0_common),
        "B1_sea_T": float(B1_sea),
        "B1_rare_T": float(B1_rare),
        "dipolar_scale_SI": float(dipolar_scale_SI),
        "shell_scale_m": float(shell_scale),
        "t_final_s": float(t_final),
        "steps": int(steps),
        "n_sea": int(n_sea),
        "phi_sea_rad": float(phi_sea),
        "phi_rare_rad": float(phi_rare),
        "sea_detunings_Hz": [float(x) for x in sea_detunings_Hz],
        "sea_spin_type": "1/2",
        "rare_spin_type": "3/2" if is_spin_three_half else "1/2",
        "solver_atol": solver_atol,
        "solver_rtol": solver_rtol,
        "solver_nsteps": solver_nsteps,
        "solver_max_step": solver_max_step,
        "target_sea_detuning": target_sea_detuning,
    }

    print("------------------------------------------------------------")
    print("Starting sea detuning sweep (Ga sea, Al rare)")
    print(f"  Output directory    : {base_dir}")
    print(f"  Number of points    : {n_det}")
    print(f"  f_Az (Ga Larmor)    : {f_Az/1e6:.3f} MHz")
    print(f"  f_Rz (Al Larmor)    : {f_Rz/1e6:.3f} MHz")
    print(f"  Target sea detuning : {target_sea_detuning/1e6:.3f} MHz")
    print(f"  f1A (sea Rabi)      : {f1A/1e3:.3f} kHz")
    print(f"  f1R (rare Rabi)     : {f1R/1e3:.3f} kHz")
    print(f"  B0 (common)         : {B0_common:.3f} T")
    print("  Detunings δ_A (Hz):")
    print("   ", ", ".join(f"{d:+.1f}" for d in sea_detunings_Hz))
    print("------------------------------------------------------------", flush=True)

    # Small helper for JSON dumping
    def _json_dump(path: str, obj: Any) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=float)

    with PdfPages(pdf_path) as pdf:
        # -------- PDF: global parameter summary page --------
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4-like
        ax.axis("off")
        lines: List[str] = []
        lines.append("Sea detuning sweep report (Ga sea / Al rare)")
        lines.append("")
        lines.append("Global parameters (constant across sweep):")
        lines.append(f"  f_Az (sea Larmor)     = {f_Az/1e6:.3f} MHz")
        lines.append(f"  f_Rz (rare Larmor)    = {f_Rz/1e6:.3f} MHz")
        lines.append(f"  f1A (sea Rabi)        = {f1A/1e3:.3f} kHz")
        lines.append(f"  f1R (rare Rabi)       = {f1R/1e3:.3f} kHz")
        lines.append(f"  Target sea detuning   = {target_sea_detuning / 1e3:.3f} kHz")
        lines.append(f"  gamma_sea             = {gamma_sea:.3e} rad·s⁻¹·T⁻¹")
        lines.append(f"  gamma_rare            = {gamma_rare:.3e} rad·s⁻¹·T⁻¹")
        lines.append(f"  B0_common             = {B0_common:.3f} T")
        lines.append(f"  B1_sea                = {B1_sea:.3e} T")
        lines.append(f"  B1_rare               = {B1_rare:.3e} T")
        lines.append(f"  dipolar_scale_SI      = {dipolar_scale_SI:.3e}")
        lines.append(f"  shell_scale           = {shell_scale*1e9:.3f} nm")
        lines.append(f"  t_final               = {t_final:.3e} s")
        lines.append(f"  steps                 = {steps:d}")
        lines.append(f"  n_sea                 = {n_sea:d}")
        lines.append(f"  phi_sea               = {phi_sea:.3f} rad")
        lines.append(f"  phi_rare              = {phi_rare:.3f} rad")
        lines.append(f"  sea_spin_type         = 1/2")
        lines.append(f"  rare_spin_type        = {"3/2" if is_spin_three_half else "1/2"}")
        lines.append("")
        lines.append(f"  solver_atol           = {solver_atol}")
        lines.append(f"  solver_rtol           = {solver_rtol}")
        lines.append(f"  solver_nsteps         = {solver_nsteps}")
        lines.append(f"  solver_max_step       = {solver_max_step}")
        lines.append("")
        lines.append("Sea detunings (δ_A = f_Az - f_rf,A) in Hz:")
        lines.append("  " + ", ".join(f"{d:+.1f}" for d in sea_detunings_Hz))
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

        # -------- Sweep loop --------
        for idx, delta_Hz in enumerate(sea_detunings_Hz):
            # Progress update
            print(
                f"[{idx + 1}/{n_det}] Running δ_A = {delta_Hz:+.1f} Hz ...",
                flush=True,
            )

            # f_rf,A = f_Az - δ_A
            f_rf_sea = f_Az - delta_Hz
            omega_rf_sea = 2 * np.pi * f_rf_sea

            # Keep rare on resonance with its own Larmor
            f_rf_rare = f_Rz
            omega_rf_rare = 2 * np.pi * f_rf_rare

            base_params = DipolarRareParams(
                n_sea=n_sea,
                gamma_sea=gamma_sea,
                gamma_rare=gamma_rare,
                B0_sea=B0_common,
                B0_rare=B0_common,
                B1_sea=B1_sea,
                B1_rare=B1_rare,
                omega_rf_sea=omega_rf_sea,
                omega_rf_rare=omega_rf_rare,
                phi_sea=phi_sea,
                phi_rare=phi_rare,
                dipolar_scale=dipolar_scale_SI,
                shell_scale=shell_scale,
                t_final=t_final,
                steps=steps,
                drive_sea=True,
                drive_rare=False,   # toggled below
                init_x_sign=-1,
                init_rare_level=3,
                is_spin_three_half=is_spin_three_half,
                solver_atol=solver_atol,
                solver_rtol=solver_rtol,
                solver_nsteps=solver_nsteps,
                solver_max_step=solver_max_step,
            )

            # Rare drive OFF and ON
            params_off = replace(base_params, drive_rare=False)
            params_on = replace(base_params, drive_rare=True)

            # Run simulations
            t0_off = time.perf_counter()
            t_off, obs_off = simulate_rare(params_off)
            t1_off = time.perf_counter()
            dt_off = t1_off - t0_off
            print(
                f"[{idx + 1}/{n_det}] |||| Finished drive OFF in {dt_off:.2f} s",
                flush=True,
            )

            t0_on = time.perf_counter()
            t_on, obs_on = simulate_rare(params_on)
            t1_on = time.perf_counter()
            dt_on = t1_on - t0_on
            print(
                f"[{idx + 1}/{n_det}] |||| Finished drive ON in {dt_on:.2f} s",
                flush=True,
            )

            dt_total = dt_off + dt_on

            freqs_off = get_derived_frequencies(params_off)
            freqs_on = get_derived_frequencies(params_on)

            # Create per-detuning directory
            det_label = f"delta_{delta_Hz:+.1f}Hz".replace("+", "p").replace("-", "m")
            det_dir = os.path.join(base_dir, det_label)
            os.makedirs(det_dir, exist_ok=True)

            # Save raw data (all observables, even if we only plot some)
            np.savez(
                os.path.join(det_dir, "time_and_obs_off.npz"),
                t=t_off,
                **obs_off,
            )
            np.savez(
                os.path.join(det_dir, "time_and_obs_on.npz"),
                t=t_on,
                **obs_on,
            )

            # Save params and derived frequencies
            _json_dump(os.path.join(det_dir, "params_off.json"), asdict(params_off))
            _json_dump(os.path.join(det_dir, "params_on.json"), asdict(params_on))
            _json_dump(os.path.join(det_dir, "freqs_off.json"), freqs_off)
            _json_dump(os.path.join(det_dir, "freqs_on.json"), freqs_on)

            # -------- T-like fits for Iz_sea (only) --------
            fit_Iz_sea_off = fit_exponential_decay(t_off, obs_off["Iz_sea"])
            fit_Iz_sea_on = fit_exponential_decay(t_on, obs_on["Iz_sea"])

            # Store these in summary
            summary["sweep_results"].append(
                {
                    "delta_Hz": float(delta_Hz),
                    "f_rf_sea_Hz": float(f_rf_sea),
                    "fit_Iz_sea_off": fit_Iz_sea_off,
                    "fit_Iz_sea_on": fit_Iz_sea_on,
                }
            )

            # Helper to format T for labels
            def fmt_T(fit: Optional[Dict[str, float]]) -> str:
                if fit is None or fit.get("T", None) is None:
                    return "NA"
                return f"{fit['T']:.3g}"

            # ------------------------------------------------------------------
            # Plot 1: ⟨I^z_sea⟩ (rare OFF vs ON) - full resolution
            # ------------------------------------------------------------------
            fig1, ax1 = plt.subplots()
            ax1.plot(t_off, obs_off["Iz_sea"],
                     label=r"$\langle I^z_{\mathrm{sea}}\rangle$, rare OFF")
            ax1.plot(t_on,  obs_on["Iz_sea"],
                     label=r"$\langle I^z_{\mathrm{sea}}\rangle$, rare ON")

            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel(r"$\langle I^z_{\mathrm{sea}}\rangle$")
            ax1.set_title(f"δ_A = {delta_Hz:+.1f} Hz")
            ax1.legend()
            fig1.tight_layout()
            fig1_path = os.path.join(det_dir, "Iz_sea_off_on.png")
            fig1.savefig(fig1_path, dpi=300)
            pdf.savefig(fig1)
            plt.close(fig1)

            # ------------------------------------------------------------------
            # Plot 2: ⟨I^z_sea⟩ (OFF/ON) with T-fit overlays, using envelope
            # ------------------------------------------------------------------
            fig2, ax2 = plt.subplots()

            # Coarse-grain to show the envelope clearly
            t_off_c, Iz_off_c = coarse_grain(t_off, obs_off["Iz_sea"], window=50)
            t_on_c,  Iz_on_c  = coarse_grain(t_on,  obs_on["Iz_sea"],  window=50)

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

            # Overlay fits (computed from full-resolution data)
            if fit_Iz_sea_off is not None:
                f = fit_Iz_sea_off
                y_model = f["C"] + f["A"] * np.exp(-t_off / f["T"])
                ax2.plot(
                    t_off,
                    y_model,
                    "-",
                    linewidth=1.5,
                    label=f"Fit OFF (T ≈ {fmt_T(f)} s)",
                )
            if fit_Iz_sea_on is not None:
                f = fit_Iz_sea_on
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
            ax2.set_title(f"δ_A = {delta_Hz:+.1f} Hz (pseudo $T_1$ envelope)")

            # Zoom y-limits to the envelope range for clarity
            y_min = min(Iz_off_c.min(), Iz_on_c.min())
            y_max = max(Iz_off_c.max(), Iz_on_c.max())
            if y_max > y_min:
                pad = 0.05 * (y_max - y_min)
                ax2.set_ylim(y_min - pad, y_max + pad)

            ax2.legend(fontsize=8)
            fig2.tight_layout()
            fig2_path = os.path.join(det_dir, "Iz_sea_with_fits.png")
            fig2.savefig(fig2_path, dpi=300)
            pdf.savefig(fig2)
            plt.close(fig2)

            # ------------------------------------------------------------------
            # Plot 3: ⟨I^x_sea⟩ and ⟨I^y_sea⟩ (rare OFF)
            # ------------------------------------------------------------------
            fig3, ax3 = plt.subplots()
            ax3.plot(t_off, obs_off["Ix_sea"],
                     label=r"$\langle I^x_{\mathrm{sea}}\rangle$")
            ax3.plot(t_off, obs_off["Iy_sea"],
                     label=r"$\langle I^y_{\mathrm{sea}}\rangle$")

            ax3.set_xlabel("Time (s)")
            ax3.set_ylabel(r"$\langle I^{x,y}_{\mathrm{sea}}\rangle$")
            ax3.set_title(f"δ_A = {delta_Hz:+.1f} Hz (rare drive OFF)")
            ax3.legend(fontsize=8)
            fig3.tight_layout()
            fig3_path = os.path.join(det_dir, "Ix_Iy_sea_off.png")
            fig3.savefig(fig3_path, dpi=300)
            pdf.savefig(fig3)
            plt.close(fig3)

            # ------------------------------------------------------------------
            # Plot 4: State norm ‖ψ(t)‖ (rare OFF vs ON)
            # ------------------------------------------------------------------
            figN, axN = plt.subplots()
            axN.plot(t_off, obs_off["state_norm"],
                     label=r"$\|\psi(t)\|$, rare OFF")
            axN.plot(t_on, obs_on["state_norm"],
                     label=r"$\|\psi(t)\|$, rare ON")

            axN.set_xlabel("Time (s)")
            axN.set_ylabel(r"State norm $\|\psi\|$")
            axN.set_title(f"δ_A = {delta_Hz:+.1f} Hz (state norm)")
            axN.legend()
            figN.tight_layout()
            figN_path = os.path.join(det_dir, "state_norm_off_on.png")
            figN.savefig(figN_path, dpi=300)
            pdf.savefig(figN)
            plt.close(figN)

            print(
                f"[{idx + 1}/{n_det}] Finished δ_A = {delta_Hz:+.1f} Hz, "
                f"in {dt_total:.2f} s, "
                f"results in {det_dir}",
                flush=True,
            )

        # -------- PDF: T-fit summary page --------
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        lines = []
        lines.append("T-like decay fits from ⟨I^z_sea⟩ traces")
        lines.append("")
        header = "delta_Hz      T_Iz_sea_off       T_Iz_sea_on"
        lines.append(header)
        lines.append("-" * len(header))

        def _fmt_T_line(fit: Optional[Dict[str, float]]) -> str:
            if fit is None or fit.get("T", None) is None:
                return "NA".ljust(18)
            return f"{fit['T']:.3g}".ljust(18)

        for row in summary["sweep_results"]:
            delta = row["delta_Hz"]
            l = f"{delta:+9.1f}  "
            l += _fmt_T_line(row["fit_Iz_sea_off"])
            l += _fmt_T_line(row["fit_Iz_sea_on"])
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

    # -------- Save JSON summaries --------
    with open(os.path.join(base_dir, "global_params.json"), "w", encoding="utf-8") as f:
        json.dump(summary["global_params"], f, indent=2, default=float)

    with open(os.path.join(base_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)

    print("------------------------------------------------------------")
    print("Sweep complete.")
    print(f"  Results directory: {base_dir}")
    print(f"  PDF report       : {pdf_path}")
    print("------------------------------------------------------------", flush=True)

    return base_dir


def f1R_for_resonance(f1A_Hz: float,
                      deltaA_Hz: float,
                      deltaR_Hz: float = 0.0) -> float:
    """
    Compute f1R (Hz) such that the sea and rare effective fields match:

        sqrt(deltaA_Hz**2 + f1A_Hz**2) = sqrt(deltaR_Hz**2 + f1R_Hz**2)

    Parameters
    ----------
    f1A_Hz : float
        Sea Rabi frequency in Hz.
    deltaA_Hz : float
        Target sea detuning (δ_A) in Hz where you want the match.
    deltaR_Hz : float, optional
        Rare detuning (δ_R) in Hz. Default is 0 (rare on resonance).

    Returns
    -------
    float
        Required rare Rabi frequency f1R in Hz.
    """
    lhs_sq = deltaA_Hz**2 + f1A_Hz**2
    rhs_sq = lhs_sq - deltaR_Hz**2
    return rhs_sq**0.5


# ---------------------------------------------------------------------------
# Main entry point with Ga/Al physical parameters
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Ga (sea) & Al (rare) gyromagnetic ratios (rad/s/T) ---
    # See https://www.kherb.io/docs/nmr_table.html for gyromagnetic ratio sources and abundances
    # gamma69 = 6.4389e7    # 69Ga
    gamma71 = 8.1812e7    # 71Ga
    # p69, p71 = 0.60108, 0.39892  # natural abundances (approx)

    gamma_sea = gamma71  # p69 * gamma69 + p71 * gamma71   # effective Ga, ~7.13e7
    gamma_rare = 6.976e7                        # 27Al

    # --- Static field ---
    B0_common = 3.0  # Tesla

    # Sea Larmor f_Az determined by B0 and gamma_sea
    f_Az = gamma_sea * B0_common / (2 * np.pi)

    # --- Rabi frequencies (Hz) ---
    f1A = 2500   # sea Rabi
    target_sea_detuning = 1250
    # rare Rabi freq determined to make it match resonance condition with target sea detuning

    # --- Time grid (extended to see the envelope clearly) ---
    t_final = 30 # in seconds
    steps = 20000

    # --- RF phases ---
    phi_sea = (np.pi / 2.0) * 1.0
    phi_rare = (np.pi / 2.0) * 1.0

    # --- Detuning sweep (δ_A in Hz): -10 kHz ... +10 kHz ---
    sea_detunings_Hz = np.linspace(0, 5000, 21)

    run_sweep_sea_detuning(
        f_Az=f_Az,
        f1A=f1A,
        target_sea_detuning=target_sea_detuning,
        gamma_sea=gamma_sea,
        gamma_rare=gamma_rare,
        sea_detunings_Hz=sea_detunings_Hz,
        n_sea=6,
        t_final=t_final,
        steps=steps,
        phi_sea=phi_sea,
        phi_rare=phi_rare,
        out_root="results",
        is_spin_three_half=False,
        solver_atol=1e-10,
        solver_rtol=1e-9,
        solver_nsteps=10_000_000,
        solver_max_step=1e-5,
    )
