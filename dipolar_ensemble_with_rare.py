from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Mapping, Tuple

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt


# ---------- Helpers: local operators for sea (I=1/2) and rare (I=3/2) ----------

# Sea spins: spin-1/2
_Ix_sea = 0.5 * qt.sigmax()
_Iy_sea = 0.5 * qt.sigmay()
_Iz_sea = 0.5 * qt.sigmaz()
_I_sea = qt.qeye(2)

# Rare spin: spin-3/2
_Jx_rare = qt.jmat(1.5, "x")     # 4x4
_Jy_rare = qt.jmat(1.5, "y")
_Jz_rare = qt.jmat(1.5, "z")
_J_rare = qt.qeye(4)


def dims_with_rare(n_sea: int) -> list[int]:
    """Local dimensions: n_sea spins of dim=2, plus one rare spin of dim=4."""
    return [2] * n_sea + [4]


def embed_site_op_hetero(local_op: qt.Qobj, site: int, dims: list[int]) -> qt.Qobj:
    """
    Embed a single-site operator with given local dims into the full tensor space.

    dims[k] is the local Hilbert dimension at site k.
    """
    op_list = [qt.qeye(d) for d in dims]
    op_list[site] = local_op
    return qt.tensor(op_list)


def total_op_sea(local_op: qt.Qobj, n_sea: int, dims: list[int]) -> qt.Qobj:
    """
    Sum of the same local operator over all sea spins (indices 0..n_sea-1).
    """
    return sum(embed_site_op_hetero(local_op, j, dims) for j in range(n_sea))


def basis_x_sea(sign: int = +1) -> qt.Qobj:
    """|±x> eigenstates of Ix for a spin-1/2 (sign=+1 gives +1/2, sign=-1 gives -1/2)."""
    up, dn = qt.basis(2, 0), qt.basis(2, 1)
    ket = (up + sign * dn).unit()
    return ket

# ---------- Coupling matrix generation (unchanged) ----------

def dipolar_couplings_from_positions(
    positions: np.ndarray,
    scale: float,
    gammaSea: float,
    gammaRare: float,
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
        gamma1 = gammaRare if (i == n - 1) else gammaSea
        gamma2 = gammaRare if (j == n - 1) else gammaSea
        val = gamma1 * gamma2 * scale * geom
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

def shell_positions_with_rare_center(a: float = 0.282393) -> np.ndarray:
    """
    Positions for a single rare nucleus at the origin and 12 sea nuclei
    on a symmetric shell around it, as described in the notes:

        S = { (±1, ±1, 0), (±1, 0, ±1), (0, ±1, ±1) } * a

    Returns an array of shape (13, 3) with the first 12 rows the sea spins
    and the last row the rare spin at (0,0,0).
    """
    shell_vectors = [
        (+1, +1, 0), (+1, -1, 0), (-1, +1, 0), (-1, -1, 0),
        (+1, 0, +1), (+1, 0, -1), (-1, 0, +1), (-1, 0, -1),
        (0, +1, +1), (0, +1, -1), (0, -1, +1), (0, -1, -1),
    ]
    sea_positions = a * np.array(shell_vectors, dtype=float)
    rare_position = np.array([[0.0, 0.0, 0.0]], dtype=float)
    positions = np.vstack([sea_positions, rare_position])  # shape (13,3)
    return positions



# ---------- Simulation core: sea (I=1/2) + rare (I=3/2) ----------

@dataclass
class DipolarRareParams:
    # number of sea spins (I=1/2). Total spins = n_sea + 1 (rare).
    n_sea: int = 6

    # Zeeman frequencies (rad/s)
    omega_nz_sea: float = 2 * np.pi * 700.0   # base sea nuclear Larmor
    omega_nz_rare: float = 2 * np.pi * 1_100.0    # rare nuclear Larmor

    # RF rotating-frame frequency (common frame for both species)
    omega_rf_sea: float | None = None              # default: = omega_nz_sea
    omega_rf_rare: float | None = None  # default: = omega_nz_rare

    # RF amplitudes and phases (rad/s, rad)
    omega1_sea: float = 2 * np.pi * 100.0
    omega1_rare: float = 2 * np.pi * 50.0
    phi_sea: float = 0.0
    phi_rare: float = 0.0

    # static disorder / detunings
    delta_std_sea: float = 0.0                 # per-sea-site Larmor disorder std (rad/s)
    delta_rare: float = 0.0                    # extra rare detuning (rad/s)

    # dipolar couplings: full (n_sea+1 x n_sea+1) matrix, or positions, or random
    b_matrix_full: np.ndarray | None = None    # including sea-sea and sea-rare
    positions: np.ndarray | None = None        # positions for all spins (sea then rare)
    dipolar_scale: float = 2 * np.pi * 100.0   # scale for b_ij if using positions / random

    gamma_sea: float = 1.0
    gamma_rare: float = 1.0

    # time grid
    t_final: float = 0.02                      # seconds
    steps: int = 2_000

    # drive flags
    drive_sea: bool = False
    drive_rare: bool = False

    # initial state
    init_x_sign: int = -1                      # sea: |±x> eigenstates
    init_rare_level: int = 3                   # rare: basis index 0..3 (0= m=+3/2, 3 = m=-3/2)


def build_hamiltonian_rare(params: DipolarRareParams) -> Tuple[qt.Qobj, Dict[str, qt.Qobj]]:
    """
    Construct the rotating-frame Hamiltonian for:

        - n_sea spins of I = 1/2
        - 1 rare spin of I = 3/2 (last index)

    Returns the Hamiltonian and useful observables.
    """
    n_sea = params.n_sea
    n_total = n_sea + 1
    idx_rare = n_sea
    dims = dims_with_rare(n_sea)

    omega_rf_sea = params.omega_rf_sea if params.omega_rf_sea is not None else params.omega_nz_sea
    omega_rf_rare = params.omega_rf_rare if params.omega_rf_rare is not None else params.omega_nz_rare

    # ---- Per-site detunings Δ_j = ω_{z,j} - ω_RF ----
    rng = np.random.default_rng(0)

    # Sea detunings with disorder
    delta_sea = (params.omega_nz_sea - omega_rf_sea) + rng.normal(
        0.0, params.delta_std_sea, size=n_sea
    )

    # Rare detuning (no site disorder, just a single extra delta_rare)
    delta_rare = (params.omega_nz_rare - omega_rf_rare) + params.delta_rare

    # Detuning Hamiltonian
    H_detune = 0
    for j in range(n_sea):
        H_detune += delta_sea[j] * embed_site_op_hetero(_Iz_sea, j, dims)
    H_detune += delta_rare * embed_site_op_hetero(_Jz_rare, idx_rare, dims)

    # ---- RF drive terms (static in this rotating frame) ----
    if params.drive_sea:
        H_drive_sea = params.omega1_sea * (
            np.cos(params.phi_sea) * total_op_sea(_Ix_sea, n_sea, dims)
            + np.sin(params.phi_sea) * total_op_sea(_Iy_sea, n_sea, dims)
        )
    else:
        H_drive_sea = 0

    if params.drive_rare:
        Ix_R = embed_site_op_hetero(_Jx_rare, idx_rare, dims)
        Iy_R = embed_site_op_hetero(_Jy_rare, idx_rare, dims)
        H_drive_rare = params.omega1_rare * (
            np.cos(params.phi_rare) * Ix_R + np.sin(params.phi_rare) * Iy_R
        )
    else:
        H_drive_rare = 0

    # ---- Dipolar couplings ----
    # Expect a full (n_total x n_total) matrix if provided.
    if params.b_matrix_full is not None:
        b = np.array(params.b_matrix_full, dtype=float)
        if b.shape != (n_total, n_total):
            raise ValueError(f"b_matrix_full must be shape ({n_total}, {n_total}).")
    elif params.positions is not None:
        if params.positions.shape != (n_total, 3):
            raise ValueError(f"positions must have shape ({n_total}, 3).")
        b = dipolar_couplings_from_positions(params.positions, params.dipolar_scale,
                                             params.gamma_sea, params.gamma_rare)
    else:
        # simple random symmetric couplings centered at 0, scaled
        b = rng.normal(0.0, 1.0, size=(n_total, n_total))
        b = 0.5 * (b + b.T)
        np.fill_diagonal(b, 0.0)
        b *= params.dipolar_scale / np.sqrt(n_total)

    # Build dipolar Hamiltonian:
    #
    #  - sea-sea:   secular homonuclear dipolar
    #               H_ij^(AA) = b_ij [ 2 Izi Izj − Ixi Ixj − Iyi Iyj ]
    #
    #  - sea-rare:  heteronuclear Ising (no flip-flops)
    #               H_iR^(AR) = b_iR Izi IzR
    #
    H_dipolar = 0
    for i, j in combinations(range(n_total), 2):
        if i < n_sea and j < n_sea:
            # sea-sea
            Iz_i = embed_site_op_hetero(_Iz_sea, i, dims)
            Iz_j = embed_site_op_hetero(_Iz_sea, j, dims)
            Ix_i = embed_site_op_hetero(_Ix_sea, i, dims)
            Ix_j = embed_site_op_hetero(_Ix_sea, j, dims)
            Iy_i = embed_site_op_hetero(_Iy_sea, i, dims)
            Iy_j = embed_site_op_hetero(_Iy_sea, j, dims)
            H_dipolar += b[i, j] * (
                Iz_i * Iz_j - (0.25 * (Ix_i * Ix_j - Iy_i * Iy_j))
            )
        else:
            # sea-rare (Ising only: Izi * IzR)
            if i == idx_rare or j == idx_rare:
                sea_idx = i if j == idx_rare else j
                Iz_i = embed_site_op_hetero(_Iz_sea, sea_idx, dims)
                Iz_R = embed_site_op_hetero(_Jz_rare, idx_rare, dims)
                H_dipolar += b[i, j] * (Iz_i * Iz_R)
            else:
                # (This case won't occur for n_total = n_sea + 1, but left for clarity.)
                pass

    H_total = qt.Qobj(H_detune + H_drive_sea + H_drive_rare + H_dipolar)

    # ---- Observables ----
    Ix_tot_sea = total_op_sea(_Ix_sea, n_sea, dims)
    Iy_tot_sea = total_op_sea(_Iy_sea, n_sea, dims)
    Iz_tot_sea = total_op_sea(_Iz_sea, n_sea, dims)
    Iz_R = embed_site_op_hetero(_Jz_rare, idx_rare, dims)
    Ix_R = embed_site_op_hetero(_Jx_rare, idx_rare, dims)

    obs = {
        "Ix_sea": Ix_tot_sea,
        "Iy_sea": Iy_tot_sea,
        "Iz_sea": Iz_tot_sea,   # abundant-bath longitudinal magnetization
        "Iz_R": Iz_R,           # rare spin longitudinal magnetization
        "Ix_R": Ix_R,
    }
    return H_total, obs


def initial_state_rare(params: DipolarRareParams) -> qt.Qobj:
    """
    Product state for sea + rare:

      - sea spins: all in |±x> eigenstate (as before)
      - rare spin: bare basis level index init_rare_level (0..3)
        (0 → |m=+3/2>, 1 → |m=+1/2>, 2 → |m=-1/2>, 3 → |m=-3/2>)
    """
    n_sea = params.n_sea
    dims = dims_with_rare(n_sea)
    dim_rare = dims[-1]

    sea_ket = basis_x_sea(params.init_x_sign)
    # rare_level = int(params.init_rare_level)
    # if not (0 <= rare_level < dim_rare):
    #     raise ValueError(f"init_rare_level must be in [0, {dim_rare-1}]")

    rare_ket = basis_x_rare(sign=+1)

    return qt.tensor([sea_ket] * n_sea + [rare_ket])

def basis_x_rare(sign: int = +1) -> qt.Qobj:
    """
    |±x> eigenstates of Jx for a spin-3/2.
    sign=+1 → largest +Jx eigenvalue, sign=-1 → most negative.
    """
    evals, evecs = _Jx_rare.eigenstates()
    if sign >= 0:
        idx = int(np.argmax(evals))   # +3/2 eigenstate of Jx
    else:
        idx = int(np.argmin(evals))   # -3/2 eigenstate of Jx
    return evecs[idx]


def simulate_rare(params: DipolarRareParams) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Run time evolution and return:

        - ⟨Ix_sea⟩, ⟨Iy_sea⟩, ⟨Iz_sea⟩ (bath magnetization components)
        - ⟨Iz_R⟩ (rare spin longitudinal magnetization)
    """
    if params.steps < 2 or params.t_final <= 0.0:
        raise ValueError("Bad time grid: steps>=2 and t_final>0.")

    H, eops = build_hamiltonian_rare(params)
    psi0 = initial_state_rare(params)

    t = np.linspace(0.0, params.t_final, params.steps)
    res = qt.sesolve(H, psi0, t, e_ops=[eops["Ix_sea"], eops["Iy_sea"], eops["Iz_sea"], eops["Iz_R"], eops["Ix_R"]])

    out = {
        "Ix_sea": np.real(res.expect[0]),
        "Iy_sea": np.real(res.expect[1]),
        "Iz_sea": np.real(res.expect[2]),
        "Iz_R": np.real(res.expect[3]),
        "Ix_R": np.real(res.expect[4]),
    }
    return t, out


# ---------- Small plotting helper ----------

def annotate_freqs(ax: plt.Axes, info: Mapping[str, float]) -> None:
    parts = []
    if "f_Az" in info:
        parts.append(fr"$f_{{A,z}}$={info['f_Az']:.3g} Hz")
    if "f_Rz" in info:
        parts.append(fr"$f_{{R,z}}$={info['f_Rz']:.3g} Hz")
    if "f_rf" in info:
        parts.append(fr"$f_{{RF}}$={info['f_rf']:.3g} Hz")
    if "f1A" in info:
        parts.append(fr"$f_1^A$={info['f1A']:.3g} Hz")
    if "f1R" in info:
        parts.append(fr"$f_1^R$={info['f1R']:.3g} Hz")
    if "phi_sea" in info:
        parts.append(fr"$φ^A$={info['phi_sea']:.3g}")
    if "phi_rare" in info:
        parts.append(fr"$φ^R$={info['phi_rare']:.3g}")
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


def plot_totals_rare(t: np.ndarray, obs: Mapping[str, np.ndarray], title: str, info: Mapping[str, float]) -> None:
    fig, ax = plt.subplots()
    ax.plot(t, obs["Ix_sea"], label="⟨Ix_sea⟩")
    ax.plot(t, obs["Iy_sea"], label="⟨Iy_sea⟩")
    ax.plot(t, obs["Iz_sea"], label="⟨Iz_sea⟩")
    ax.plot(t, obs["Iz_R"], label="⟨Iz_R⟩")
    ax.plot(t, obs["Ix_R"], label="⟨Ix_R⟩")
    ax.set_xlabel("t")
    ax.set_ylabel("Expectation value")
    ax.set_title(title)
    ax.legend()
    annotate_freqs(ax, info)
    fig.tight_layout()


# ---------- Example demo ----------

if __name__ == "__main__":
    # Frequencies in Hz for readability
    f_Az = 700.0      # abundant species Larmor
    f_Rz = 1_100.0        # rare species Larmor
    f1A = 100.0         # sea RF amplitude
    f1R = 50.0          # rare RF amplitude
    f_rf_A = f_Az + 100.0        # rotating-frame reference frequency
    f_rf_R = f_Rz

    omega_Az = 2 * np.pi * f_Az
    omega_Rz = 2 * np.pi * f_Rz
    omega1A = 2 * np.pi * f1A
    omega1R = 2 * np.pi * f1R
    omega_rf_sea = 2 * np.pi * f_rf_A
    omega_rf_rare = 2 * np.pi * f_rf_R

    # Example: 8 sea spins on a chain, plus one rare spin (index 8)
    # n_sea = 8
    # bAA = b_matrix_chain_nearest(n_sea, 2 * np.pi * 100.0)
    # # simple choice: same coupling b_AR between each sea spin and rare
    # bAR_val = 2 * np.pi * 30.0
    # b_full = np.zeros((n_sea + 1, n_sea + 1))
    # b_full[:n_sea, :n_sea] = bAA
    # for i in range(n_sea):
    #
    #     b_full[i, n_sea] = b_full[n_sea, i] = bAR_val

    # --- Shell geometry: 12 sea spins + 1 rare at origin ---
    a_lattice = 0.282393
    positions = shell_positions_with_rare_center(a=a_lattice)
    n_sea = 12  # must match the shell construction

    phi_sea = 0.0  # np.pi / 2.0
    phi_rare = 0.0  # np.pi / 2.0

    params = DipolarRareParams(
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
        gamma_sea=4.2,
        gamma_rare=6.6,
        # b_matrix_full=b_full,
        positions=positions,
        dipolar_scale=2 * np.pi,
        t_final=1,
        steps=2_000,
        drive_sea=True,
        drive_rare=False,
        init_x_sign=-1,
        init_rare_level=3,
    )

    t, obs = simulate_rare(params)
    plot_totals_rare(
        t,
        obs,
        "Sea (I=1/2) + rare (I=3/2) no rare drive",
        info={"f_Az": f_Az, "f_Rz": f_Rz, "f_rf": f_rf_A, "f1A": f1A, "f1R": f1R, "phi_sea": phi_sea,
              "phi_rare": phi_rare},
    )
    plt.show()
