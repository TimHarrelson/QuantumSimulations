"""
Sweep the sea detuning δ_A = f_Az - f_rf,A for a Ga (sea) / Al (rare) system.

For each detuning, we now run THREE simulations:

  Geometry A: rare at center (is_center_rare = True)
    1) rare drive OFF
    2) rare drive ON

  Geometry B: sea-as-center control (is_center_rare = False)
    3) sea-center control (drive_rare = OFF only)

For each detuning:
  - Save all raw data and parameters to disk as each simulation finishes.
  - Compute window-averaged ⟨I^z_sea⟩ and define a drift "slope" metric from a
    linear regression on the central portion of the coarse-grained envelope.


  - For the rare-at-center geometry, define a *normalized* contrast metric
    comparing ON vs OFF, using a Michelson-style normalized difference of
    the magnitudes of the slopes:

        contrast_rare_center
            = (|I_z_slope_on_center| - |I_z_slope_off_center|)
              / (|I_z_slope_on_center| + |I_z_slope_off_center|),

    but only if both ON and OFF slopes are statistically significant,
    i.e. |t_on| ≥ SLOPE_T_MIN and |t_off| ≥ SLOPE_T_MIN. Otherwise we set
    contrast_rare_center = NaN.

  - For the sea-as-center control, define a *normalized* contrast metric that
    compares the rare-at-center ON slope to the sea-center OFF slope, again
    using a Michelson-style normalized difference and t-stat gating:

        contrast_sea_center
            = (|I_z_slope_on_center| - |I_z_slope_off_sea_center|)
              / (|I_z_slope_on_center| + |I_z_slope_off_sea_center|),

    but only if |t_on_center| ≥ SLOPE_T_MIN and
    |t_off_sea_center| ≥ SLOPE_T_MIN.

  - Save all of the above to metrics.json in the detuning directory.
  - We still track the linear-regression correlation coefficient R for each
    slope fit (as diagnostics), but t-values are used for significance gating:
        R_off_center, R_on_center, R_off_sea_center,
        t_off_center, t_on_center, t_off_sea_center.

Additionally, we define a dimensionless parameter

    x = ΔΩ / |g_eff|,

where:

  - ΔΩ = Ω_A - Ω_R is the difference between the sea and rare *effective*
    nutation frequencies in Hz:
        Ω_A = sqrt(δ_A^2 + f1A^2),
        Ω_R = sqrt(δ_R^2 + f1R^2) ≈ f1R for δ_R ≈ 0.

  - g_eff is defined from the RMS sea–rare coupling |b_AR|_rms in Hz via
        |g_eff| = (|b_AR|_rms / 4) · sin θ_A · sin θ_R,
        sin θ_A = f1A / Ω_A,
        sin θ_R = f1R / Ω_R.

We store ΔΩ, g_eff, and ΔΩ/|g_eff| for each detuning in metrics.json and in
the top-level summary, and we add a final PDF page plotting:

    contrast_rare_center vs ΔΩ / |g_eff|.

Plots for each detuning (added to a PDF report and saved as PNGs):

  (1) ⟨I^z_sea⟩ vs time (rare OFF & ON, rare-at-center geometry).
  (2) Coarse-grained (window-averaged) ⟨I^z_sea⟩ for the rare-at-center
      geometry (OFF & ON), with slope endpoints highlighted and labelled,
      plus contrast_rare_center, t-values, and ΔΩ/|g_eff|.
  (3) Coarse-grained ⟨I^z_sea⟩ for the sea-as-center control (OFF only),
      with its slope endpoints highlighted and labelled, plus
      contrast_sea_center and its t-value.
  (4) State norm ‖ψ(t)‖ vs time (rare OFF & ON, rare-at-center geometry).

A final PDF page summarises, for each detuning:

  - I_z_slope_OFF/ON(center), t_off/center, t_on/center, contrast_rare_center
  - I_z_slope_OFF(sea-center), t_off(sea-center), contrast_sea_center

This version removes the previous exponential “T-like” fits entirely, removes
the detection_metric, keeps the window-averaged plots, separates the control
geometry into its own graph using only the OFF sea-center run, and adds a
contrast_rare_center vs ΔΩ/|g_eff| plot using an RMS g_eff based on the
sea–rare couplings for this geometry.
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
# Utility: slope/drift metric from coarse-grained ⟨I^z_sea⟩
# ---------------------------------------------------------------------------

def iz_slope_from_coarse(
    t_coarse: np.ndarray,
    iz_coarse: np.ndarray,
) -> Dict[str, float]:
    """
    Compute a drift metric from a coarse-grained ⟨I^z_sea⟩ trace using
    a linear fit to the *central* part of the envelope.

    We fit a line iz ≈ a + b t over the central ~60% of points, then define:

        I_z_slope = I_z_fit(t_end) - I_z_fit(t_start)

    where t_start/t_end are the endpoints of the fitted interval and
    I_z_fit is the fitted line evaluated at those times.

    We also compute:
        - slope      : the fitted slope b (units of 1/time),
        - slope_std  : standard error of the slope,
        - t_value    : t-statistic b / slope_std,
        - R_value    : Pearson correlation coefficient for the segment,
        - R2_value   : R_value**2.

    Returns a dict with keys:
        - I_z_slope
        - t_start, t_end
        - I_z_start, I_z_end  (fitted values at these endpoints)
        - slope, slope_std, t_value
        - R_value, R2_value

    If too few coarse points are available, returns NaNs.
    """
    n = t_coarse.size
    if n < 4 or iz_coarse.size < 4:
        return {
            "I_z_slope": np.nan,
            "t_start": np.nan,
            "t_end": np.nan,
            "I_z_start": np.nan,
            "I_z_end": np.nan,
            "slope": np.nan,
            "slope_std": np.nan,
            "t_value": np.nan,
            "R_value": np.nan,
            "R2_value": np.nan,
        }

    # Use the central ~60% to avoid early/late transients and edge noise.
    frac_edge = 0.2
    i0 = int(frac_edge * n)
    i1 = int((1.0 - frac_edge) * n)

    # Clamp indices so we always have at least two points.
    i0 = max(0, min(i0, n - 2))
    i1 = max(i0 + 2, min(i1, n))

    t_seg = t_coarse[i0:i1]
    iz_seg = iz_coarse[i0:i1]

    if t_seg.size < 2:
        return {
            "I_z_slope": np.nan,
            "t_start": np.nan,
            "t_end": np.nan,
            "I_z_start": np.nan,
            "I_z_end": np.nan,
            "slope": np.nan,
            "slope_std": np.nan,
            "t_value": np.nan,
            "R_value": np.nan,
            "R2_value": np.nan,
        }

    # Linear least-squares fit: iz ≈ a + b t
    b, a = np.polyfit(t_seg, iz_seg, 1)

    t_start = float(t_seg[0])
    t_end = float(t_seg[-1])
    iz_start = float(a + b * t_start)
    iz_end = float(a + b * t_end)
    I_z_slope = iz_end - iz_start

    # Correlation coefficient R and R² on the fitted segment
    t_mean = np.mean(t_seg)
    iz_mean = np.mean(iz_seg)
    t_d = t_seg - t_mean
    iz_d = iz_seg - iz_mean
    ss_t = float(np.sum(t_d * t_d))
    ss_iz = float(np.sum(iz_d * iz_d))

    if ss_t > 0.0 and ss_iz > 0.0:
        R_value = float(np.dot(t_d, iz_d) / np.sqrt(ss_t * ss_iz))
        R2_value = float(R_value * R_value)
    else:
        R_value = np.nan
        R2_value = np.nan

    # t-statistic for slope b
    if t_seg.size > 2 and ss_t > 0.0:
        y_fit = a + b * t_seg
        resid = iz_seg - y_fit
        sse = float(np.sum(resid**2))
        s2 = sse / (t_seg.size - 2)
        slope_var = s2 / ss_t if ss_t > 0.0 else np.nan
        slope_std = float(np.sqrt(slope_var)) if slope_var > 0.0 else np.nan
        t_value = float(b / slope_std) if (slope_std > 0.0 and np.isfinite(slope_std)) else np.nan
    else:
        slope_std = np.nan
        t_value = np.nan

    return {
        "I_z_slope": float(I_z_slope),
        "t_start": t_start,
        "t_end": t_end,
        "I_z_start": iz_start,
        "I_z_end": iz_end,
        "slope": float(b),
        "slope_std": slope_std,
        "t_value": t_value,
        "R_value": R_value,
        "R2_value": R2_value,
    }


# ---------------------------------------------------------------------------
# Significance threshold and contrast helper
# ---------------------------------------------------------------------------

# Minimum |t| value required to treat a slope as reliably non-zero.
SLOPE_T_MIN: float = 1.0


def contrast_michelson_with_t_gate(
    slope_on: float,
    slope_off: float,
    t_on: float,
    t_off: float,
    t_min: float = SLOPE_T_MIN,
) -> float:
    """
    Michelson-style normalized difference of slope magnitudes with *soft*
    t-based handling of noisy near-zero slopes.

        C = (|s_on_eff| - |s_off_eff|) / (|s_on_eff| + |s_off_eff|),

    where:
      - If |t| >= t_min the corresponding slope is used as-is.
      - If |t| <  t_min the corresponding slope is treated as "effectively 0".

    If both effective slopes are ~0, returns 0.0 (no measurable contrast).
    If either slope or t is non-finite, returns NaN.
    """
    # Slopes must be finite
    if not (np.isfinite(slope_on) and np.isfinite(slope_off)):
        return float("nan")

    # We still require finite t-values; if the fit completely failed, bail out.
    if not (np.isfinite(t_on) and np.isfinite(t_off)):
        return float("nan")

    # Effective slopes: small-|t| => treat as zero baseline
    eff_on = 0.0 if abs(t_on) < t_min else slope_on
    eff_off = 0.0 if abs(t_off) < t_min else slope_off

    denom = abs(eff_on) + abs(eff_off)

    # If both are effectively zero, define contrast as 0 (no detectable difference)
    if not np.isfinite(denom) or denom <= 1e-16:
        return 0.0

    return (abs(eff_on) - abs(eff_off)) / denom


# ---------------------------------------------------------------------------
# Utility: safe normalization helper (unused by new contrasts, kept for legacy)
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
# Utility: consistent detuning label for folder names
# ---------------------------------------------------------------------------

def detuning_label(delta_Hz: float) -> str:
    """
    Produce the per-detuning directory name used by this sweep, e.g.:

      +1000.0 Hz -> 'delta_p1000.0Hz'
      -1000.0 Hz -> 'delta_m1000.0Hz'
    """
    return f"delta_{delta_Hz:+.1f}Hz".replace("+", "p").replace("-", "m")


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
    is_spin_three_half: bool = False,
    solver_atol: float | None = None,
    solver_rtol: float | None = None,
    solver_nsteps: int | None = None,
    solver_max_step: float | None = None,
    coarse_window: int = 50,
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
        The sea detuning (Hz) used to calculate the rare Rabi frequency (f1R)
        such that it will satisfy the resonance condition the model assumes.
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
        Whether the rare spin is J = 3/2 or I = 1/2.
    coarse_window : int
        Block size for coarse-graining the ⟨I^z_sea⟩ traces.

    Returns
    -------
    base_dir : str
        Path to the directory containing this sweep's results.
    """
    # Rare Rabi frequency from your original Hartmann–Hahn style condition
    f1R = f1R_for_resonance(f1A, target_sea_detuning, 0.0)

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
    B1_rare = 2 * np.pi * f1R / gamma_rare if gamma_rare != 0.0 else 0.0

    # -------- Physical dipolar scale and shell length scale --------
    # dipolar_scale_SI = (μ0 / 4π) * ħ, in SI units,
    # used in b_ij = dipolar_scale * γ_i γ_j (1 - 3 cos^2 θ) / r^3.
    mu0_over_4pi = 1.0e-7             # N / A^2
    hbar = 1.054571817e-34            # J·s
    dipolar_scale_SI = mu0_over_4pi * hbar  # ~1.05e-41 in SI units
    shell_scale = 0.282393e-9         # meters; ~0.3 nm characteristic length

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
    sea_rare_abs_Hz = np.abs(sea_rare_vals) / (2 * np.pi)  # Hz
    sea_rare_rms_Hz = np.sqrt(np.mean(np.abs(sea_rare_vals) ** 2)) / (2 * np.pi)

    # Sea–sea couplings (all i<j pairs)
    sea_sea_vals = []
    for i in sea_indices:
        for j in sea_indices:
            if j > i:
                sea_sea_vals.append(b[i, j])
    sea_sea_vals = np.array(sea_sea_vals, dtype=float)
    sea_sea_abs_Hz = np.abs(sea_sea_vals) / (2 * np.pi)  # Hz
    sea_sea_rms_Hz = np.sqrt(np.mean(np.abs(sea_sea_vals) ** 2)) / (2 * np.pi)

    # Print a concise coupling summary to the console
    print("Estimated dipolar couplings from geometry + physical scales:")
    print(f"  Sea–rare b_ij (all sea ↔ rare), |b| in Hz:")
    print(f"    avg |b_AR| ≈ {sea_rare_abs_Hz.mean():.2f} Hz")
    print(f"    rms |b_AR| ≈ {sea_rare_rms_Hz:.2f} Hz")
    print(f"    min |b_AR| ≈ {sea_rare_abs_Hz.min():.2f} Hz")
    print(f"    max |b_AR| ≈ {sea_rare_abs_Hz.max():.2f} Hz")
    print("  Sea–sea b_ij (all i<j), |b| in Hz:")
    print(f"    avg |b_AA| ≈ {sea_sea_abs_Hz.mean():.2f} Hz")
    print(f"    rms |b_AA| ≈ {sea_sea_rms_Hz:.2f} Hz")
    print(f"    min |b_AA| ≈ {sea_sea_abs_Hz.min():.2f} Hz")
    print(f"    max |b_AA| ≈ {sea_sea_abs_Hz.max():.2f} Hz")
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

    # Save geometry and couplings so that other scripts can reuse them
    np.savez(
        os.path.join(base_dir, "geometry_and_couplings.npz"),
        positions=positions,
        b=b,
        sea_indices=np.array(sea_indices, dtype=int),
        idx_rare=int(idx_rare),
        sea_rare_vals=sea_rare_vals,
        sea_sea_vals=sea_sea_vals,
    )

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
        "coarse_window": int(coarse_window),
        # Coupling statistics (Hz)
        "avg_b_AR_Hz": float(sea_rare_abs_Hz.mean()),
        "rms_b_AR_Hz": float(sea_rare_rms_Hz),
        "avg_b_AA_Hz": float(sea_sea_abs_Hz.mean()),
        "rms_b_AA_Hz": float(sea_sea_rms_Hz),
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
    print(f"  Detunings δ_A (Hz):")
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
        lines.append("  sea_spin_type         = 1/2")
        lines.append(
            "  rare_spin_type        = " + ("3/2" if is_spin_three_half else "1/2")
        )
        lines.append("")
        lines.append(f"  solver_atol           = {solver_atol}")
        lines.append(f"  solver_rtol           = {solver_rtol}")
        lines.append(f"  solver_nsteps         = {solver_nsteps}")
        lines.append(f"  solver_max_step       = {solver_max_step}")
        lines.append("")
        lines.append(f"  coarse_window         = {coarse_window}")
        lines.append("")
        lines.append("Sea detunings (δ_A = f_Az - f_rf,A) in Hz:")
        det_strs = [f"{d:+.1f}" for d in sea_detunings_Hz]
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

        # -------- Sweep loop --------
        for idx, delta_Hz in enumerate(sea_detunings_Hz):
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

            # Create per-detuning directory *before* running sims
            det_label = detuning_label(delta_Hz)
            det_dir = os.path.join(base_dir, det_label)
            os.makedirs(det_dir, exist_ok=True)

            # Base parameters (rare-at-center geometry by default)
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
                drive_rare=False,   # toggled per variant
                init_x_sign=-1,
                init_rare_level=3,
                is_spin_three_half=is_spin_three_half,
                is_center_rare=True,
                solver_atol=solver_atol,
                solver_rtol=solver_rtol,
                solver_nsteps=solver_nsteps,
                solver_max_step=solver_max_step,
            )

            # Variant parameters
            params_center_off = replace(base_params, drive_rare=False, is_center_rare=True)
            params_center_on  = replace(base_params, drive_rare=True,  is_center_rare=True)

            # Sea-as-center control geometry (drive_rare always OFF here)
            params_sea_center_off = replace(
                base_params,
                drive_rare=False,
                is_center_rare=False,
            )

            # Helper to run a simulation, save immediately, and report timing
            def run_and_save(tag: str, params: DipolarRareParams):
                t0 = time.perf_counter()
                t_arr, obs = simulate_rare(params)
                t1 = time.perf_counter()
                dt_sim = t1 - t0

                npz_name = f"time_and_obs_{tag}.npz"
                np.savez(
                    os.path.join(det_dir, npz_name),
                    t=t_arr,
                    **obs,
                )
                _json_dump(os.path.join(det_dir, f"params_{tag}.json"), asdict(params))
                freqs = get_derived_frequencies(params)
                _json_dump(os.path.join(det_dir, f"freqs_{tag}.json"), freqs)

                print(
                    f"[{idx + 1}/{n_det}] |||| Finished {tag} in {dt_sim:.2f} s",
                    flush=True,
                )
                return t_arr, obs, freqs

            # Run the three simulations for this detuning
            t_center_off, obs_center_off, freqs_center_off = run_and_save(
                "center_off", params_center_off
            )
            t_center_on,  obs_center_on,  freqs_center_on  = run_and_save(
                "center_on", params_center_on
            )
            t_sea_center_off, obs_sea_center_off, freqs_sea_center_off = run_and_save(
                "shell_off", params_sea_center_off
            )

            # -------- Coarse-grain Iz_sea traces and compute slopes --------
            # Rare-at-center geometry
            t_c_off_center, Iz_c_off_center = coarse_grain(
                t_center_off, obs_center_off["Iz_sea"], window=coarse_window
            )
            t_c_on_center, Iz_c_on_center = coarse_grain(
                t_center_on, obs_center_on["Iz_sea"], window=coarse_window
            )

            slope_off_center = iz_slope_from_coarse(t_c_off_center, Iz_c_off_center)
            slope_on_center  = iz_slope_from_coarse(t_c_on_center,  Iz_c_on_center)

            I_z_slope_off_center = slope_off_center["I_z_slope"]
            I_z_slope_on_center  = slope_on_center["I_z_slope"]
            R_off_center = slope_off_center["R_value"]
            R_on_center = slope_on_center["R_value"]
            t_off_center = slope_off_center["t_value"]
            t_on_center = slope_on_center["t_value"]

            # Sea-as-center control geometry (sea-center OFF only)
            t_c_off_sea_center, Iz_c_off_sea_center = coarse_grain(
                t_sea_center_off, obs_sea_center_off["Iz_sea"], window=coarse_window
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

            # -------- Dimensionless mismatch parameter: ΔΩ / |g_eff| --------
            # Use RMS sea–rare coupling to define a single effective g_eff (Hz).
            # Δ_A = δ_A = delta_Hz; Δ_R ≈ 0 (rare driven on resonance).
            Omega_A = np.sqrt(delta_Hz ** 2 + f1A ** 2)  # Hz
            Omega_R = np.sqrt(0.0 ** 2 + f1R ** 2)       # Hz (just f1R)

            DeltaOmega = Omega_A - Omega_R               # Hz

            # sin(theta) = f1 / Omega for each species
            sin_theta_A = f1A / Omega_A if Omega_A != 0.0 else 0.0
            sin_theta_R = f1R / Omega_R if Omega_R != 0.0 else 0.0

            # RMS g_eff in Hz using RMS |b_AR|
            g_eff_Hz = (sea_rare_rms_Hz / 4.0) * sin_theta_A * sin_theta_R

            if g_eff_Hz == 0.0 or np.isnan(g_eff_Hz):
                DeltaOmega_over_geff = float("nan")
            else:
                # Keep the sign of ΔΩ, normalize by |g_eff|
                DeltaOmega_over_geff = float(DeltaOmega / abs(g_eff_Hz))

            # -------- Save metrics for this detuning --------
            metrics = {
                "delta_Hz": float(delta_Hz),
                "f_rf_sea_Hz": float(f_rf_sea),
                # Rare-at-center geometry
                "I_z_slope_off_center": float(I_z_slope_off_center),
                "R_off_center": float(R_off_center),
                "t_off_center": float(t_off_center),
                "I_z_slope_on_center": float(I_z_slope_on_center),
                "R_on_center": float(R_on_center),
                "t_on_center": float(t_on_center),
                "contrast_rare_center": float(contrast_rare_center),
                # Sea-as-center control geometry (sea-center)
                "I_z_slope_off_sea_center": float(I_z_slope_off_sea_center),
                "R_off_sea_center": float(R_off_sea_center),
                "t_off_sea_center": float(t_off_sea_center),
                "contrast_sea_center": float(contrast_sea_center),
                # ΔΩ / |g_eff| diagnostics
                "DeltaOmega_Hz": float(DeltaOmega),
                "g_eff_Hz": float(g_eff_Hz),
                "DeltaOmega_over_geff": float(DeltaOmega_over_geff),
            }
            _json_dump(os.path.join(det_dir, "metrics.json"), metrics)
            summary["sweep_results"].append(metrics)

            # ------------------------------------------------------------------
            # Plot 1: ⟨I^z_sea⟩ (rare OFF vs ON) - full resolution, rare-at-center
            # ------------------------------------------------------------------
            fig1, ax1 = plt.subplots()
            ax1.plot(
                t_center_off,
                obs_center_off["Iz_sea"],
                label=r"$\langle I^z_{\mathrm{sea}}\rangle$, rare OFF (center)",
            )
            ax1.plot(
                t_center_on,
                obs_center_on["Iz_sea"],
                label=r"$\langle I^z_{\mathrm{sea}}\rangle$, rare ON (center)",
            )

            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel(r"$\langle I^z_{\mathrm{sea}}\rangle$")
            ax1.set_title(f"δ_A = {delta_Hz:+.1f} Hz (rare at center)")
            ax1.legend()
            fig1.tight_layout()
            fig1_path = os.path.join(det_dir, "Iz_sea_off_on_center.png")
            fig1.savefig(fig1_path, dpi=300)
            pdf.savefig(fig1)
            plt.close(fig1)

            # Helper: plot a slope segment
            def _plot_slope_segment(ax, slope_info: Dict[str, float], style: str, label: str):
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

            # Helper: annotate slope value at the midpoint of the segment
            def _annotate_slope_text(
                ax,
                slope_info: Dict[str, float],
                slope_value: float,
                dy: float,
                offset_sign: float,
                text_label: Optional[str] = None,
            ):
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
            # Plot 2: Coarse-grained ⟨I^z_sea⟩ (rare-at-center only) + slope
            # ------------------------------------------------------------------
            fig2, ax2 = plt.subplots()
            fig2.subplots_adjust(right=0.75)

            # Coarse envelopes (rare-at-center)
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

            # Slope segments
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
                f"{delta_Hz:+.1f} Hz (coarse envelopes, rare at center)"
            )

            # Zoom y-limits to the envelope range for clarity
            all_env_center = np.concatenate([Iz_c_off_center, Iz_c_on_center])
            y_min_c = float(np.min(all_env_center))
            y_max_c = float(np.max(all_env_center))
            if y_max_c > y_min_c:
                pad_c = 0.05 * (y_max_c - y_min_c)
                ax2.set_ylim(y_min_c - pad_c, y_max_c + pad_c)
            dy_c = max(1e-8, y_max_c - y_min_c)

            # Annotate slopes along their lines
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

            # Annotate contrast and t values
            metrics_text_center = (
                f"I_z_slope_off(center)   = {I_z_slope_off_center:+.3e}\n"
                f"t_off(center)           = {t_off_center:+.3f}\n"
                f"I_z_slope_on(center)    = {I_z_slope_on_center:+.3e}\n"
                f"t_on(center)            = {t_on_center:+.3f}\n"
                f"contrast_rare_center    = {contrast_rare_center:+.3e}\n"
                f"ΔΩ/|g_eff|              = {DeltaOmega_over_geff:+.3e}"
            )
            ax2.text(
                1.02,
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
            fig2_path = os.path.join(det_dir, "Iz_sea_detection_envelopes_center.png")
            fig2.savefig(fig2_path, dpi=300)
            pdf.savefig(fig2)
            plt.close(fig2)

            # ------------------------------------------------------------------
            # Plot 3: Coarse-grained ⟨I^z_sea⟩ (sea-center control) + slope
            # ------------------------------------------------------------------
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
                "δ_A = "
                f"{delta_Hz:+.1f} Hz (coarse envelope, sea-center control)"
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
            fig3_path = os.path.join(det_dir, "Iz_sea_detection_envelopes_sea_center.png")
            fig3.savefig(fig3_path, dpi=300)
            pdf.savefig(fig3)
            plt.close(fig3)

            # ------------------------------------------------------------------
            # Plot 4: State norm ‖ψ(t)‖ (rare OFF vs ON, rare-at-center)
            # ------------------------------------------------------------------
            figN, axN = plt.subplots()
            axN.plot(
                t_center_off,
                obs_center_off["state_norm"],
                label=r"$\|\psi(t)\|$, rare OFF (center)",
            )
            axN.plot(
                t_center_on,
                obs_center_on["state_norm"],
                label=r"$\|\psi(t)\|$, rare ON (center)",
            )

            axN.set_xlabel("Time (s)")
            axN.set_ylabel(r"State norm $\|\psi\|$")
            axN.set_title(
                f"δ_A = {delta_Hz:+.1f} Hz (state norm, rare at center)"
            )
            axN.legend()
            figN.tight_layout()
            figN_path = os.path.join(det_dir, "state_norm_off_on_center.png")
            figN.savefig(figN_path, dpi=300)
            pdf.savefig(figN)
            plt.close(figN)

            print(
                f"[{idx + 1}/{n_det}] Finished δ_A = {delta_Hz:+.1f} Hz, "
                f"results in {det_dir}",
                flush=True,
            )

        # -------- PDF: contrast-metric summary page (as a table) --------
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
        for row in summary["sweep_results"]:
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
            "Contrast metrics from coarse-grained ⟨I^z_sea⟩ slopes",
            pad=20,
        )

        pdf.savefig(fig)
        plt.close(fig)

        # -------- PDF: contrast_rare_center vs ΔΩ/|g_eff| 2D plot --------
        try:
            sweep_rows = summary["sweep_results"]
            if sweep_rows:
                x_vals = np.array(
                    [row.get("DeltaOmega_over_geff", np.nan) for row in sweep_rows],
                    dtype=float,
                )
                y_vals = np.array(
                    [row.get("contrast_rare_center", np.nan) for row in sweep_rows],
                    dtype=float,
                )

                # Drop NaNs
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
                        r"Rare-center contrast vs $\Delta\Omega/|g_{\mathrm{eff}}|$"
                    )
                    axc.grid(True, alpha=0.3)

                    figc.tight_layout()
                    figc_path = os.path.join(
                        base_dir, "contrast_rare_center_vs_DeltaOmega_over_geff.png"
                    )
                    figc.savefig(figc_path, dpi=300)
                    pdf.savefig(figc)
                    plt.close(figc)
        except Exception as exc:
            print(f"Warning: could not build ΔΩ/|g_eff| contrast plot: {exc}")

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


def f1R_for_resonance(
    f1A_Hz: float,
    deltaA_Hz: float,
    deltaR_Hz: float = 0.0,
) -> float:
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

    gamma_sea = gamma71  # effective Ga
    gamma_rare = 6.976e7  # 27Al

    # --- Static field ---
    B0_common = 3.0  # Tesla

    # Sea Larmor f_Az determined by B0 and gamma_sea
    f_Az = gamma_sea * B0_common / (2 * np.pi)

    # --- Rabi frequencies (Hz) ---
    f1A = 50_000   # sea Rabi
    target_sea_detuning = f1A
    # rare Rabi freq determined to make it match resonance condition with target sea detuning

    # --- Time grid (extended to see the envelope clearly) ---
    t_final = 30  # in seconds
    steps = 20000

    # --- RF phases ---
    phi_sea = (np.pi / 2.0) * 1.0
    phi_rare = (np.pi / 2.0) * 1.0

    # --- Detuning sweep (δ_A in Hz) ---
    sea_detunings_Hz = np.linspace(0.0, 3.0*target_sea_detuning, 13)

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
        out_root="results/sweep_f1A_3x_target_detune_extra_long",
        is_spin_three_half=False,
        solver_atol=1e-10,
        solver_rtol=1e-9,
        solver_nsteps=10_000_000,
        solver_max_step=1e-5,
        coarse_window=100,
    )
