from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt


def sim(
        zeeman_angular_freq: float,  # ω_eZ
        drive_angular_freq: float,  # ω_MW
        rabi_angular_freq: float,  # ω_1
        t_final: float = 5.0,
        steps: int = 2_000,
) -> None:

    if steps < 2:
        raise ValueError("steps must be >= 2")
    if t_final <= 0.0:
        raise ValueError("t_final must be > 0")

    # Spin-1/2 operators (ħ=1 ⇒ S = σ/2)
    sx = 0.5 * qt.sigmax()
    sy = 0.5 * qt.sigmay()
    sz = 0.5 * qt.sigmaz()

    # Resonant when drive freq == zeeman freq
    # rabi freq << zeeman freq
    # H_Z = ω_eZ * S_z
    # H_ESR = ω_1(cos(ω_MW * t)*s_x - sin(ω_MW * t)*s_y)
    # H_total = H_Z + H_ESR
    #         = (ω_eZ * S_z) + ω_1(cos(ω_MW * t)*s_x - sin(ω_MW * t)*s_y)

    zeeman_hamiltonian = zeeman_angular_freq * sz

    # Time-dependent drive coefficients
    def esr_coeff_x(t: float) -> float:
        return rabi_angular_freq * np.cos(drive_angular_freq * t)

    def esr_coeff_y(t: float) -> float:
        return -rabi_angular_freq * np.sin(drive_angular_freq * t)

    total_hamiltonian = [zeeman_hamiltonian, [sx, esr_coeff_x], [sy, esr_coeff_y]]

    psi0 = qt.basis(2, 0)

    sample_times = np.linspace(0.0, t_final, steps)

    e_ops = [sx, sy, sz]

    result = qt.sesolve(total_hamiltonian, psi0, sample_times, e_ops=e_ops)

    obs: Dict[str, np.ndarray] = {
        "Sx": np.real(result.expect[0]),
        "Sy": np.real(result.expect[1]),
        "Sz": np.real(result.expect[2]),
    }
    return sample_times, obs

def plot_spin_components(t: np.ndarray, obs: dict[str, np.ndarray], title: str) -> None:
    """Plot ⟨Sx⟩, ⟨Sy⟩, ⟨Sz⟩ vs time."""
    plt.figure()
    plt.plot(t, obs["Sx"], label="⟨Sx⟩")
    plt.plot(t, obs["Sy"], label="⟨Sy⟩")
    plt.plot(t, obs["Sz"], label="⟨Sz⟩")
    plt.xlabel("t")
    plt.ylabel("Expectation value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()


if __name__ == "__main__":
    # ---------------- Parameters (ħ = 1;) ----------------
    zeeman_angular_freq = 2 * np.pi * 1  # Zeeman/Larmor ω_eZ
    rabi_angular_freq = 2 * np.pi * 20  # drive amplitude (Rabi rate) ω_1
    detuning = 2 * np.pi * .1  # Δ = ω_MW - ω_eZ

    t_final = 2
    steps = 10000

    # ---------------- Resonant (Δ = 0) ----------------
    t_res, obs_res = sim(
        zeeman_angular_freq=zeeman_angular_freq,
        drive_angular_freq=zeeman_angular_freq,  # Δ = 0 ⇒ ω_MW = ω_eZ
        rabi_angular_freq=rabi_angular_freq,
        t_final=t_final,
        steps=steps,
    )
    plot_spin_components(t_res, obs_res, "Spin expectations (Δ = 0)")

    # ---------------- Off-resonant (Δ ≠ 0) ----------------
    t_det, obs_det = sim(
        zeeman_angular_freq=zeeman_angular_freq,
        drive_angular_freq=zeeman_angular_freq + detuning,  # ω_MW = ω_eZ + Δ
        rabi_angular_freq=rabi_angular_freq,
        t_final=t_final,
        steps=steps,
    )
    plot_spin_components(t_det, obs_det, "Spin expectations (Δ ≠ 0)")

    plt.show()
