from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Mapping, Tuple

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt


# ---------- Helpers: spin-1/2 single-site operators and tensors ----------

_Ix = 0.5 * qt.sigmax()
_Iy = 0.5 * qt.sigmay()
_Iz = 0.5 * qt.sigmaz()
_I = qt.qeye(2)


def embed_site_op(op: qt.Qobj, site: int, n: int) -> qt.Qobj:
    """Embed a single-site operator at `site` in an n-spin tensor product."""
    ops = [_I] * n
    ops[site] = op
    return qt.tensor(ops)


def total_op(op: qt.Qobj, n: int) -> qt.Qobj:
    """Sum of a single-site operator over all spins."""
    return sum(embed_site_op(op, j, n) for j in range(n))


def basis_x(sign: int = +1) -> qt.Qobj:
    """|±x> eigenstates of Ix (sign=+1 gives +1/2, sign=-1 gives -1/2)."""
    up, dn = qt.basis(2, 0), qt.basis(2, 1)
    ket = (up + sign * dn).unit()
    return ket


# ---------- Coupling matrix generation ----------

def dipolar_couplings_from_positions(
    positions: np.ndarray,
    scale: float,
) -> np.ndarray:
    """
    Compute secular dipolar couplings:
        b_ij = scale * (1 - 3 cos^2 θ_ij) / r_ij^3
    where θ_ij is the angle to the z-axis (Bz direction), r_ij = |r_i - r_j|.
    Units: `scale` should be in rad/s * (distance)^3 so that b_ij ends in rad/s.
    """
    n = positions.shape[0]
    b = np.zeros((n, n), dtype=float)
    for i, j in combinations(range(n), 2):
        r = positions[i] - positions[j]
        rij = np.linalg.norm(r)
        if rij == 0:
            raise ValueError("Two sites have identical positions.")
        cos_th = r[2] / rij  # z-component over distance
        geom = (1.0 - 3.0 * cos_th**2) / (rij**3)
        val = scale * geom
        b[i, j] = b[j, i] = val
    return b


def random_positions_in_cube(n: int, side: float, seed: int | None = None) -> np.ndarray:
    """Random 3D positions in a cube [0, side]^3 (no min-separation enforcement)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, side, size=(n, 3))

def b_matrix_chain_1over_r3(n: int, b_nn: float) -> np.ndarray:
    """
    Deterministic dipolar-like couplings on a 1D chain:
        b_ij =  b_nn / |i-j|^3   for i != j,  0 on diagonal.
    b_nn is the nearest-neighbor magnitude in rad/s.
    """
    B = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            val = b_nn / ((j - i) ** 3)
            B[i, j] = B[j, i] = val
    return B

def b_matrix_chain_nearest(n: int, b_nn: float) -> np.ndarray:
    """Deterministic nearest-neighbor couplings only."""
    B = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        B[i, i + 1] = B[i + 1, i] = b_nn
    return B

# ---------- Simulation core ----------

@dataclass
class DipolarParams:
    n: int = 6
    omega_nz: float = 2 * np.pi * 1_000.0      # base nuclear Larmor (rad/s)
    omega_rf: float | None = None              # RF rotating-frame freq; default: = omega_nz
    omega1_rf: float = 2 * np.pi * 100.0       # RF Rabi amplitude (rad/s)
    phi: float = 0.0                           # RF phase (0 spin-lock, π/2 Rabi-like)
    delta_std: float = 0.0                     # per-site Larmor disorder std (rad/s)
    b_matrix: np.ndarray | None = None         # [n,n] dipolar couplings b_ij (rad/s)
    positions: np.ndarray | None = None        # if provided, compute b_ij from positions
    dipolar_scale: float = 2 * np.pi * 100.0   # scale for b_ij if using positions
    t_final: float = 0.02                      # seconds
    steps: int = 2_000
    drive: bool = False                        # False → FID; True → continuous drive
    init_x_sign: int = -1                      # start in Ix = -1/2 eigenstate (full transverse)


def build_hamiltonian(params: DipolarParams) -> Tuple[qt.Qobj, Dict[str, qt.Qobj]]:
    """Construct the rotating-frame Hamiltonian and some useful observables."""
    n = params.n
    omega_rf = params.omega_rf if params.omega_rf is not None else params.omega_nz

    # Per-site detunings Δ_j = ω_{N,z,j} - ω_RF (allowing static disorder)
    rng = np.random.default_rng(0)
    delta_j = (params.omega_nz - omega_rf) + rng.normal(0.0, params.delta_std, size=n)

    # Single-site sums
    hamiltonian_detune = sum(delta_j[j] * embed_site_op(_Iz, j, n) for j in range(n))

    # RF term (static in the rotating frame)
    if params.drive:
        hamiltonian_drive = params.omega1_rf * (
            np.cos(params.phi) * total_op(_Ix, n) + np.sin(params.phi) * total_op(_Iy, n)
        )
    else:
        hamiltonian_drive = 0

    # Dipolar couplings b_ij (secular, quantization axis = z)
    if params.b_matrix is not None:
        b = np.array(params.b_matrix, dtype=float)
        if b.shape != (n, n):
            raise ValueError("b_matrix must be shape (n, n).")
    elif params.positions is not None:
        b = dipolar_couplings_from_positions(params.positions, scale=params.dipolar_scale)
    else:
        # simple random symmetric couplings centered at 0, scaled
        b = rng.normal(0.0, 1.0, size=(n, n))
        b = 0.5 * (b + b.T)
        np.fill_diagonal(b, 0.0)
        b *= params.dipolar_scale / np.sqrt(n)

    # hamiltonian_dipolar = Σ_{i<j} b_ij [ 3 Izi Izj − (Ix_i Ix_j + Iy_i Iy_j + Iz_i Iz_j) ]
    #       = Σ_{i<j} b_ij [ 2 Izi Izj − Ix_i Ix_j − Iy_i Iy_j ]
    hamiltonian_dipolar = 0
    for i, j in combinations(range(n), 2):
        hamiltonian_dipolar += b[i, j] * (
                2 * embed_site_op(_Iz, i, n) * embed_site_op(_Iz, j, n)
                - embed_site_op(_Ix, i, n) * embed_site_op(_Ix, j, n)
                - embed_site_op(_Iy, i, n) * embed_site_op(_Iy, j, n)
        )

    hamiltonian_total = qt.Qobj(hamiltonian_detune + hamiltonian_drive + hamiltonian_dipolar)

    # Observables: total magnetization components
    obs = {
        "Ix_tot": total_op(_Ix, n),
        "Iy_tot": total_op(_Iy, n),
        "Iz_tot": total_op(_Iz, n),
    }
    return hamiltonian_total, obs


def initial_state(n: int, sign: int = -1) -> qt.Qobj:
    """Product state with all spins in |±x> (full transverse polarization)."""
    ket = basis_x(sign)
    return qt.tensor([ket] * n)

def simulate(params: DipolarParams) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Run the time evolution and return totals ⟨Ix⟩, ⟨Iy⟩, ⟨Iz⟩."""
    if params.steps < 2 or params.t_final <= 0.0:
        raise ValueError("Bad time grid: steps>=2 and t_final>0.")

    hamiltonian, eops = build_hamiltonian(params)
    psi0 = initial_state(params.n, sign=params.init_x_sign)

    t = np.linspace(0.0, params.t_final, params.steps)
    res = qt.sesolve(hamiltonian, psi0, t, e_ops=[eops["Ix_tot"], eops["Iy_tot"], eops["Iz_tot"]])
    out = {"Ix": np.real(res.expect[0]), "Iy": np.real(res.expect[1]), "Iz": np.real(res.expect[2])}
    return t, out


# ---------- Small plotting helper ----------

def annotate_freqs(ax: plt.Axes, info: Mapping[str, float]) -> None:
    parts = []
    if "f_nz" in info:
        parts.append(fr"$f_{{N,z}}$={info['f_nz']:.3g} Hz")
    if "f_rf" in info:
        parts.append(fr"$f_{{RF}}$={info['f_rf']:.3g} Hz")
    if "f1" in info:
        parts.append(fr"$f_1$={info['f1']:.3g} Hz")
    if "phi" in info:
        parts.append(fr"$\varphi$={info['phi']:.2f} rad")
    ax.text(
        0.98,
        0.02,
        ", ".join(parts),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.15, lw=0.0),
    )


def plot_totals(t: np.ndarray, obs: Mapping[str, np.ndarray], title: str, info: Mapping[str, float]) -> None:
    fig, ax = plt.subplots()
    ax.plot(t, obs["Ix"], label="⟨Ix_tot⟩")
    ax.plot(t, obs["Iy"], label="⟨Iy_tot⟩")
    ax.plot(t, obs["Iz"], label="⟨Iz_tot⟩")
    ax.set_xlabel("t")
    ax.set_ylabel("Expectation value")
    ax.set_title(title)
    ax.legend()
    annotate_freqs(ax, info)
    fig.tight_layout()


# ---------- Demos (FID, spin-lock, Rabi-like) ----------

if __name__ == "__main__":
    # Frequencies in Hz for readability
    f_nz = 1_000.0      # nuclear Larmor
    f1 = 100.0          # RF Rabi amplitude
    f_rf = f_nz         # resonant

    # Angular versions (rad/s)
    omega_nz = 2 * np.pi * f_nz
    omega1 = 2 * np.pi * f1
    omega_rf = 2 * np.pi * f_rf

    params_base = DipolarParams(
        n=8,
        omega_nz=omega_nz,
        omega_rf=omega_rf,
        omega1_rf=omega1,
        phi=0.0,
        delta_std=0.0,
        # positions=random_positions_in_cube(n=8, side=2.0, seed=7),
        b_matrix=b_matrix_chain_nearest(8, 2*np.pi*100.0),
        dipolar_scale=2 * np.pi * 100.0,   # ~100 Hz couplings
        t_final=0.05,
        steps=2_000,
    )

    # 1) FID (no drive)
    p_fid = params_base
    p_fid.drive = False
    t, obs = simulate(p_fid)
    plot_totals(t, obs, "FID (no RF drive)", info={"f_nz": f_nz})

    # 2) Spin locking (continuous resonant drive, φ = 0)
    p_lock = params_base
    p_lock.drive = True
    p_lock.phi = 0.0
    t, obs = simulate(p_lock)
    plot_totals(t, obs, "Spin locking (resonant, φ=0)", info={"f_nz": f_nz, "f_rf": f_rf, "f1": f1, "phi": 0.0})

    # 3) Rabi-like (continuous resonant drive, φ = π/2)
    p_rabi = params_base
    p_rabi.drive = True
    p_rabi.phi = np.pi / 2
    t, obs = simulate(p_rabi)
    plot_totals(t, obs, "Continuous drive (resonant, φ=π/2)", info={"f_nz": f_nz, "f_rf": f_rf, "f1": f1, "phi": np.pi / 2})

    plt.show()
