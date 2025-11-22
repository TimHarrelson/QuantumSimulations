# sweep_sea_detuning.py
"""
Sweep the sea detuning δ_A = f_Az - f_rf_A.
For each detuning, run two simulations:
  1) rare drive OFF
  2) rare drive ON

Plot sum(<Iz_sea>) vs time for both cases on the same figure, and
annotate each figure with the parameters used.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace

# Make sure this module name matches your file: dipolar_ensemble_with_rare.py
from dipolar_ensemble_with_rare import (
    DipolarRareParams,
    simulate_rare,
    shell_positions_with_rare_center,
)


def run_sweep_sea_detuning() -> None:
    # -------- Base physical parameters (in Hz) --------
    f_Az = 700.0        # sea Larmor frequency
    f_Rz = 1100.0       # rare Larmor frequency

    f1A = 100.0         # sea RF amplitude
    f1R = 50.0          # rare RF amplitude

    phi_sea = 0.0       # RF phases
    phi_rare = 0.0

    gamma_sea = 4.2
    gamma_rare = 6.6

    # Convert to rad/s
    omega_Az = 2 * np.pi * f_Az
    omega_Rz = 2 * np.pi * f_Rz
    omega1A = 2 * np.pi * f1A
    omega1R = 2 * np.pi * f1R

    # Geometry: 12 sea spins on a shell + rare at origin
    positions = shell_positions_with_rare_center(a=0.282393)
    n_sea = positions.shape[0] - 1

    # -------- Detuning sweep (in Hz): δ_A = f_Az - f_rf_A --------
    # You can adjust the range and number of points here.
    sea_detunings_Hz = np.linspace(-300.0, 300.0, 7)

    for delta_Hz in sea_detunings_Hz:
        # f_rf_A = f_Az - δ_A
        f_rf_A = f_Az - delta_Hz
        omega_rf_sea = 2 * np.pi * f_rf_A

        # Keep the rare on resonance with its own Larmor
        f_rf_R = f_Rz
        omega_rf_rare = 2 * np.pi * f_rf_R

        # Base parameter set (sea driven, rare drive will be toggled)
        base_params = DipolarRareParams(
            n_sea=n_sea,
            omega_nz_sea=omega_Az,
            omega_nz_rare=omega_Rz,
            omega_rf_sea=omega_rf_sea,
            omega_rf_rare=omega_rf_rare,
            omega1_sea=omega1A,
            omega1_rare=omega1R,
            phi_sea=phi_sea,
            phi_rare=phi_rare,
            delta_std_sea=0.0,
            delta_rare=0.0,
            positions=positions,
            dipolar_scale=2 * np.pi,
            gamma_sea=gamma_sea,
            gamma_rare=gamma_rare,
            t_final=1.0,
            steps=2000,
            drive_sea=True,
            drive_rare=False,     # overridden below for "ON" case
            init_x_sign=-1,
            init_rare_level=3,
        )

        # -------- Run simulations: rare drive OFF and ON --------
        params_off = replace(base_params, drive_rare=False)
        t_off, obs_off = simulate_rare(params_off)

        params_on = replace(base_params, drive_rare=True)
        t_on, obs_on = simulate_rare(params_on)

        # -------- Plot sum(<Iz_sea>) for both cases --------
        fig, ax = plt.subplots()

        ax.plot(
            t_off,
            obs_off["Iz_sea"],
            label="⟨Iz_sea⟩ (rare drive OFF)",
        )
        ax.plot(
            t_on,
            obs_on["Iz_sea"],
            "--",
            label="⟨Iz_sea⟩ (rare drive ON)",
        )

        ax.set_xlabel("t (s)")
        ax.set_ylabel("Total ⟨Iz_sea⟩")
        ax.set_title(
            f"Sea detuning δ_A = {delta_Hz:.1f} Hz\n"
            f"(f_Az = {f_Az:.1f} Hz, f_rf,A = {f_rf_A:.1f} Hz)"
        )

        # Annotate with other key parameters for this iteration
        text = (
            f"f_Rz = {f_Rz:.1f} Hz, f_rf,R = {f_rf_R:.1f} Hz\n"
            f"f1A = {f1A:.1f} Hz, f1R = {f1R:.1f} Hz\n"
            f"ϕ_sea = {phi_sea:.2f}, ϕ_rare = {phi_rare:.2f}"
        )
        ax.text(
            0.98,
            0.02,
            text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round", alpha=0.15, lw=0.0),
        )

        ax.legend()
        fig.tight_layout()

    # Show all figures at once when the sweep is done
    plt.show()


if __name__ == "__main__":
    run_sweep_sea_detuning()
