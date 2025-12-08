from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Mapping, Tuple

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Local operators for sea (I = 1/2) and rare (J = 3/2) or (I = 1/2)
# ---------------------------------------------------------------------------

# Spin-1/2
_Ix = 0.5 * qt.sigmax()
_Iy = 0.5 * qt.sigmay()
_Iz = 0.5 * qt.sigmaz()
_I = qt.qeye(2)

# Spin-3/2
_Jx = qt.jmat(1.5, "x")  # 4x4
_Jy = qt.jmat(1.5, "y")
_Jz = qt.jmat(1.5, "z")
_J = qt.qeye(4)


def dims_with_rare(n_sea: int, is_spin_three_half: bool = False) -> list[int]:
    """
    Local dimensions for n_sea sea spins and one rare spin.
    Indices: 0..n_sea-1 are sea (dim=2), index n_sea is rare (dim=4).
    """
    return [2] * n_sea + ([4] if is_spin_three_half else [2])


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

def basis_sea(axis: str = "x", sign: int = +1) -> qt.Qobj:
    """
    Single-spin basis state for a sea spin (I = 1/2).

    axis ∈ {"x", "z"}
        "x": |±x⟩ eigenstate of I_x
        "z": |↑⟩ (sign ≥ 0) or |↓⟩ (sign < 0) eigenstate of I_z

    sign = +1 → "upper" eigenstate (largest eigenvalue)
    sign = -1 → "lower" eigenstate (smallest eigenvalue)
    """
    axis = axis.lower()
    if axis == "x":
        up, dn = qt.basis(2, 0), qt.basis(2, 1)
        ket = (up + sign * dn).unit()
        return ket
    elif axis == "z":
        # I_z eigenstates are just |↑⟩ and |↓⟩ in the computational basis
        idx = 0 if sign >= 0 else 1
        return qt.basis(2, idx)
    else:
        raise ValueError("axis must be 'x' or 'z' for basis_sea()")


def basis_rare(axis: str = "x", sign: int = +1, is_spin_three_half = False) -> qt.Qobj:
    """
    Single-spin basis state for the rare spin (I = 3/2).

    axis ∈ {"x", "z"}
        "x": |±⟩ eigenstate of J_x (max / min eigenvalue)
        "z": |m = +3/2⟩ (sign ≥ 0) or |m = -3/2⟩ (sign < 0) eigenstate of J_z

    sign = +1 → "upper" eigenstate (largest eigenvalue)
    sign = -1 → "lower" eigenstate (smallest eigenvalue)

    isSpinThreeHalf = True → use J (3/2) rather than I (1/2) for the basis state.
    isSpinThreeHalf = True → use I (1/2) rather than J (3/2) for the basis state.
    """
    axis = axis.lower()
    if axis == "x":
        evals, evecs = _Jx.eigenstates() if is_spin_three_half else _Ix.eigenstates()
    elif axis == "z":
        evals, evecs = _Jz.eigenstates() if is_spin_three_half else _Iz.eigenstates()
    else:
        raise ValueError("axis must be 'x' or 'z' for basis_rare()")

    idx = int(np.argmax(evals) if sign >= 0 else np.argmin(evals))
    return evecs[idx]


# ---------------------------------------------------------------------------
# Shell geometry + dipolar couplings
# ---------------------------------------------------------------------------

def _platonic_vertices(n_sea: int) -> np.ndarray:
    """
    Return unscaled vertex coordinates for the Platonic solid
    with n_sea vertices, normalized to lie on the unit sphere.

    Supported n_sea: 4 (tetra), 6 (octa), 8 (cube), 12 (icosa), 20 (dodeca).
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0  # golden ratio
    inv_phi = 1.0 / phi               # = phi - 1

    if n_sea == 4:
        # Regular tetrahedron
        pts = np.array([
            [ 1.0,  1.0,  1.0],
            [-1.0, -1.0,  1.0],
            [-1.0,  1.0, -1.0],
            [ 1.0, -1.0, -1.0],
        ], dtype=float)

    elif n_sea == 6:
        # Regular octahedron
        pts = np.array([
            [ 1.0,  0.0,  0.0],
            [-1.0,  0.0,  0.0],
            [ 0.0,  1.0,  0.0],
            [ 0.0, -1.0,  0.0],
            [ 0.0,  0.0,  1.0],
            [ 0.0,  0.0, -1.0],
        ], dtype=float)

    elif n_sea == 8:
        # Cube
        pts = np.array([
            [ 1.0,  1.0,  1.0],
            [ 1.0,  1.0, -1.0],
            [ 1.0, -1.0,  1.0],
            [ 1.0, -1.0, -1.0],
            [-1.0,  1.0,  1.0],
            [-1.0,  1.0, -1.0],
            [-1.0, -1.0,  1.0],
            [-1.0, -1.0, -1.0],
        ], dtype=float)

    elif n_sea == 12:
        # Icosahedron
        pts = np.array([
            [ 0.0,  1.0,  phi],
            [ 0.0, -1.0,  phi],
            [ 0.0,  1.0, -phi],
            [ 0.0, -1.0, -phi],

            [ 1.0,  phi,  0.0],
            [-1.0,  phi,  0.0],
            [ 1.0, -phi,  0.0],
            [-1.0, -phi,  0.0],

            [ phi,  0.0,  1.0],
            [ phi,  0.0, -1.0],
            [-phi,  0.0,  1.0],
            [-phi,  0.0, -1.0],
        ], dtype=float)

    elif n_sea == 20:
        # Dodecahedron: cube vertices + 12 others
        pts = []

        # (±1, ±1, ±1)
        for x in (-1.0, 1.0):
            for y in (-1.0, 1.0):
                for z in (-1.0, 1.0):
                    pts.append([x, y, z])

        # (0, ±1/phi, ±phi)
        for y in (-inv_phi, inv_phi):
            for z in (-phi, phi):
                pts.append([0.0, y, z])

        # (±1/phi, ±phi, 0)
        for x in (-inv_phi, inv_phi):
            for y in (-phi, phi):
                pts.append([x, y, 0.0])

        # (±phi, 0, ±1/phi)
        for x in (-phi, phi):
            for z in (-inv_phi, inv_phi):
                pts.append([x, 0.0, z])

        pts = np.array(pts, dtype=float)

    else:
        raise ValueError(f"No Platonic solid with {n_sea} vertices.")

    # Normalize each vertex to lie on the unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts_unit = pts / norms
    return pts_unit


def shell_positions_with_rare_center(
    n_sea: int,
    radius: float = 0.282393,
) -> np.ndarray:
    """
    Positions for `n_sea` sea nuclei on a shell around a rare nucleus at the origin.

    For n_sea in {4, 6, 8, 12, 20}, use the vertices of the corresponding
    Platonic solid (exactly symmetric on that polyhedral surface), scaled
    to the given radius.

    For all other n_sea, fall back to a Fibonacci-sphere construction that
    gives quasi-uniform points on a spherical shell.

    Returns
    -------
    positions : ndarray, shape (n_sea + 1, 3)
        First n_sea rows: sea nuclei.
        Last row: rare nucleus at the origin (0, 0, 0).
    """
    if n_sea < 1:
        raise ValueError("n_sea must be at least 1.")

    try:
        # Try Platonic solid first
        sea_unit = _platonic_vertices(n_sea)
        sea_positions = radius * sea_unit
    except ValueError:
        # Fallback: Fibonacci sphere
        sea_positions = np.zeros((n_sea, 3), dtype=float)
        golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0

        for i in range(n_sea):
            # y in [-1, 1]
            y = 1.0 - 2.0 * (i + 0.5) / n_sea
            r_xy = np.sqrt(max(0.0, 1.0 - y * y))

            phi_angle = 2.0 * np.pi * i / golden_ratio

            x = r_xy * np.cos(phi_angle)
            z = r_xy * np.sin(phi_angle)

            sea_positions[i] = radius * np.array([x, y, z], dtype=float)

    rare_position = np.array([[0.0, 0.0, 0.0]], dtype=float)
    positions = np.vstack([sea_positions, rare_position])
    return positions



def dipolar_couplings_from_positions(
    positions: np.ndarray,
    scale: float,
    gamma_sea: float,
    gamma_rare: float,
) -> np.ndarray:
    """
    Compute secular dipolar couplings in frequency units:

        b_ij = γ_i γ_j · scale · (1 - 3 cos^2 θ_ij) / r_ij^3,

    where θ_ij is the angle of the vector r_i - r_j to the z-axis (B0 direction),
    and r_ij = |r_i - r_j|.

    - positions: array of shape (n, 3). The last site is assumed to be the rare spin.
    - scale: sets the overall energy scale; in physical units it would absorb
      μ0/(4π), ħ, and any unit conversions.
    - gamma_sea, gamma_rare: gyromagnetic ratios for sea and rare species.

    Returns
    -------
    b : ndarray of shape (n, n)
        Symmetric coupling matrix with zeros on the diagonal.
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

        # Heteronuclear case: sea vs rare have different γ
        idx_rare = n - 1
        gamma_i = gamma_rare if (i == idx_rare) else gamma_sea
        gamma_j = gamma_rare if (j == idx_rare) else gamma_sea

        val = gamma_i * gamma_j * scale * geom
        b[i, j] = b[j, i] = val

    return b



# ---------------------------------------------------------------------------
# Simulation core: sea (I = 1/2) + rare (I = 3/2)
# ---------------------------------------------------------------------------

@dataclass
class DipolarRareParams:
    """
    Parameters for a dipolar-coupled ensemble of:

      - n_sea: I = 1/2 "sea" spins
      - 1 rare spin of I = 3/2 (last index)

    All angular frequencies (Larmor, Rabi) are DERIVED from:

      ω_z^(sea)  = γ_sea  · B0_sea
      ω_z^(rare) = γ_rare · B0_rare
      ω_1^(sea)  = γ_sea  · B1_sea
      ω_1^(rare) = γ_rare · B1_rare

    where γ are gyromagnetic ratios and B0/B1 are static and transverse RF
    fields respectively (units are arbitrary but must be consistent).

    When a given drive is OFF (drive_sea or drive_rare = False) we choose the
    rotating frame for that species to be exactly at its Larmor frequency,
    so the Zeeman term contributes no detuning for that species (Δ = 0 and
    that part of H_Z is omitted).
    """

    # Number of sea spins.
    n_sea: int = 12

    # Gyromagnetic ratios (units: rad·s^-1·(field_unit)^-1, or consistent internals)
    gamma_sea: float = 1.0
    gamma_rare: float = 1.0

    # Static B0 fields along z (field units, e.g. Tesla).
    # If both species see the same B0, set B0_sea = B0_rare.
    B0_sea: float = 1.0
    B0_rare: float = 1.0

    # Transverse RF B1 amplitudes (same field units as B0).
    # Rabi angular frequencies are ω1 = γ · B1.
    B1_sea: float = 0.0
    B1_rare: float = 0.0

    # RF rotating-frame angular frequencies (rad/s).
    # If None, defaults to the respective Larmor frequency (on resonance).
    omega_rf_sea: float | None = None
    omega_rf_rare: float | None = None

    # RF phases (radians) for sea and rare drives.
    phi_sea: float = 0.0
    phi_rare: float = 0.0

    # Dipolar coupling scale (in angular frequency units).
    dipolar_scale: float = 2 * np.pi

    # Shell geometry scale factor "a" in shell_positions_with_rare_center().
    shell_scale: float = 0.282393

    # Time grid for evolution
    t_final: float = 0.02  # seconds (or arbitrary time units)
    steps: int = 2_000

    # Drive flags: whether to apply the (static) transverse RF terms.
    drive_sea: bool = False
    drive_rare: bool = False

    # Initial state choices
    # sea: all |±x> eigenstates of Ix (I = 1/2)
    init_x_sign: int = -1
    # rare: currently prepared as an |+x> eigenstate of Jx (see initial_state_rare)
    init_rare_level: int = 3

    is_spin_three_half: bool = True

    # Solver settings
    solver_atol: float | None = None
    solver_rtol: float | None = None
    solver_nsteps: int | None = None
    solver_max_step: float | None = None


def get_derived_frequencies(params: DipolarRareParams) -> Dict[str, float]:
    """
    Compute derived angular frequencies and detunings from DipolarRareParams.

    Returns a dictionary with both angular frequencies (rad/s) and linear
    frequencies (Hz), e.g.:
        - omega_Az, omega_Rz, omega1_sea, omega1_rare, omega_rf_sea, omega_rf_rare
        - delta_sea, delta_rare         (detunings in rad/s)
        - f_Az, f_Rz, f1_sea, f1_rare, f_rf_sea, f_rf_rare, delta_sea_Hz, delta_rare_Hz
    """
    gamma_sea = params.gamma_sea
    gamma_rare = params.gamma_rare
    B0_sea = params.B0_sea
    B0_rare = params.B0_rare
    B1_sea = params.B1_sea
    B1_rare = params.B1_rare

    # Larmor angular frequencies
    omega_Az = gamma_sea * B0_sea
    omega_Rz = gamma_rare * B0_rare

    # Rabi angular frequencies
    omega1_sea = gamma_sea * B1_sea
    omega1_rare = gamma_rare * B1_rare

    # Rotating-frame (RF carrier) frequencies
    omega_rf_sea = params.omega_rf_sea if params.omega_rf_sea is not None else omega_Az
    omega_rf_rare = params.omega_rf_rare if params.omega_rf_rare is not None else omega_Rz

    # Detunings: only meaningful if the corresponding drive is on.
    if params.drive_sea:
        delta_sea = omega_Az - omega_rf_sea
    else:
        delta_sea = 0.0

    if params.drive_rare:
        delta_rare = omega_Rz - omega_rf_rare
    else:
        delta_rare = 0.0

    # Helper to convert angular to linear frequency
    def to_hz(omega: float) -> float:
        return omega / (2 * np.pi)

    return {
        # Angular frequencies (rad/s)
        "omega_Az": omega_Az,
        "omega_Rz": omega_Rz,
        "omega1_sea": omega1_sea,
        "omega1_rare": omega1_rare,
        "omega_rf_sea": omega_rf_sea,
        "omega_rf_rare": omega_rf_rare,
        "delta_sea": delta_sea,
        "delta_rare": delta_rare,
        # Linear frequencies (Hz)
        "f_Az": to_hz(omega_Az),
        "f_Rz": to_hz(omega_Rz),
        "f1_sea": to_hz(omega1_sea),
        "f1_rare": to_hz(omega1_rare),
        "f_rf_sea": to_hz(omega_rf_sea),
        "f_rf_rare": to_hz(omega_rf_rare),
        "delta_sea_Hz": to_hz(delta_sea),
        "delta_rare_Hz": to_hz(delta_rare),
    }


def build_hamiltonian_rare(params: DipolarRareParams) -> Tuple[qt.Qobj, Dict[str, qt.Qobj]]:
    """
    Construct the rotating-frame Hamiltonian for:

        - n_sea spins of I = 1/2
        - 1 rare spin of J = 3/2 or I = 3/2 (last index)

    The Hamiltonian includes:
      - Zeeman detunings in the rotating frames (sea + rare), but only for
        species with an active drive:
            H_Z = Σ_j Δ_A I_{zj}   (if drive_sea)
                + Δ_R J_{zR}       (if drive_rare)
        where
            Δ_A = ω_Az - ω_rf_sea,
            Δ_R = ω_Rz - ω_rf_rare.

      - RF drive terms (sea + rare, if enabled):
            H_rf^A = ω_1^A [ cos φ_A · I_x^A,tot + sin φ_A · I_y^A,tot ]
            H_rf^R = ω_1^R [ cos φ_R · J_xR    + sin φ_R · J_yR     ]

      - Secular dipolar couplings:
          * sea-sea (homonuclear):
                H_dip^(AA) = Σ_{i<j} b_ij
                    [ I_{iz} I_{jz}
                      - (1/4) ( I_{ix} I_{jx} - I_{iy} I_{jy} ) ].

          * sea-rare (heteronuclear):

                H_dip^(AR) = Σ_i b_{iR} I_{iz} J_{zR}.
    """
    n_sea = params.n_sea
    n_total = n_sea + 1
    idx_rare = n_sea
    dims = dims_with_rare(n_sea, params.is_spin_three_half)

    # ---- Derived frequencies and detunings ----
    freqs = get_derived_frequencies(params)
    omega1_sea = freqs["omega1_sea"]
    omega1_rare = freqs["omega1_rare"]
    delta_sea = freqs["delta_sea"]
    delta_rare = freqs["delta_rare"]

    rare_operator_z = _Jz if params.is_spin_three_half else _Iz
    rare_operator_y = _Jy if params.is_spin_three_half else _Iy
    rare_operator_x = _Jx if params.is_spin_three_half else _Ix

    # ---- Zeeman detunings in the rotating frames ----
    H_detune = 0
    if params.drive_sea and delta_sea != 0.0:
        # All sea spins share the same Δ_A.
        Iz_tot_sea = total_op_sea(_Iz, n_sea, dims)
        H_detune += delta_sea * Iz_tot_sea

    if params.drive_rare and delta_rare != 0.0:
        Iz_R = embed_site_op_hetero(rare_operator_z, idx_rare, dims)
        H_detune += delta_rare * Iz_R

    # ---- RF drive terms (static in this rotating frame) ----
    if params.drive_sea and omega1_sea != 0.0:
        H_drive_sea = omega1_sea * (
            np.cos(params.phi_sea) * total_op_sea(_Ix, n_sea, dims)
            + np.sin(params.phi_sea) * total_op_sea(_Iy, n_sea, dims)
        )
    else:
        H_drive_sea = 0

    if params.drive_rare and omega1_rare != 0.0:
        Ix_R = embed_site_op_hetero(rare_operator_x, idx_rare, dims)
        Iy_R = embed_site_op_hetero(rare_operator_y, idx_rare, dims)
        H_drive_rare = omega1_rare * (
            np.cos(params.phi_rare) * Ix_R + np.sin(params.phi_rare) * Iy_R
        )
    else:
        H_drive_rare = 0

    # ---- Dipolar couplings from shell geometry ----
    positions = shell_positions_with_rare_center(
        n_sea=n_sea,
        radius=params.shell_scale,
    )
    if positions.shape != (n_total, 3):
        raise RuntimeError("Shell geometry returned unexpected number of sites.")

    b = dipolar_couplings_from_positions(
        positions,
        params.dipolar_scale,
        params.gamma_sea,
        params.gamma_rare,
    )

    # Build dipolar Hamiltonian:
    H_dipolar = 0
    for i, j in combinations(range(n_total), 2):
        if i < n_sea and j < n_sea:
            # sea-sea couplings: homonuclear secular dipolar
            Iz_i = embed_site_op_hetero(_Iz, i, dims)
            Iz_j = embed_site_op_hetero(_Iz, j, dims)
            Ix_i = embed_site_op_hetero(_Ix, i, dims)
            Ix_j = embed_site_op_hetero(_Ix, j, dims)
            Iy_i = embed_site_op_hetero(_Iy, i, dims)
            Iy_j = embed_site_op_hetero(_Iy, j, dims)

            H_dipolar += b[i, j] * (
                Iz_i * Iz_j - (0.25 * (Ix_i * Ix_j - Iy_i * Iy_j))
            )
        else:
            # sea-rare (Ising only: Izi · JzR)
            if i == idx_rare or j == idx_rare:
                sea_idx = i if j == idx_rare else j
                Iz_i = embed_site_op_hetero(_Iz, sea_idx, dims)
                Iz_R = embed_site_op_hetero(rare_operator_z, idx_rare, dims)
                H_dipolar += b[i, j] * (Iz_i * Iz_R)

    H_total = qt.Qobj(H_detune + H_drive_sea + H_drive_rare + H_dipolar)

    # ---- Observables ----
    Ix_tot_sea = total_op_sea(_Ix, n_sea, dims)
    Iy_tot_sea = total_op_sea(_Iy, n_sea, dims)
    Iz_tot_sea = total_op_sea(_Iz, n_sea, dims)
    Iz_R = embed_site_op_hetero(rare_operator_z, idx_rare, dims)
    Ix_R = embed_site_op_hetero(rare_operator_x, idx_rare, dims)
    Iy_R = embed_site_op_hetero(rare_operator_y, idx_rare, dims)

    obs = {
        "Ix_sea": Ix_tot_sea,
        "Iy_sea": Iy_tot_sea,
        "Iz_sea": Iz_tot_sea,
        "Iz_R": Iz_R,
        "Ix_R": Ix_R,
        "Iy_R": Iy_R,
    }
    return H_total, obs


def initial_state_rare(params: DipolarRareParams) -> qt.Qobj:
    """
    Product state for sea + rare:

      - sea spins: all in |±x> eigenstate (sign set by init_x_sign)
      - rare spin: currently prepared in an |+x> eigenstate of Jx
    """
    n_sea = params.n_sea
    dims = dims_with_rare(n_sea, params.is_spin_three_half)
    dim_rare = dims[-1]

    sea_ket = basis_sea(axis="z", sign=params.init_x_sign)

    rare_ket = basis_rare(axis="z", sign=-params.init_x_sign, is_spin_three_half=params.is_spin_three_half)

    return qt.tensor([sea_ket] * n_sea + [rare_ket])


def simulate_rare(params: DipolarRareParams) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Run time evolution and return time array and expectation values:

      - ⟨Ix_sea⟩, ⟨Iy_sea⟩, ⟨Iz_sea⟩ : bath magnetization components
      - ⟨Iz_R⟩                       : rare spin longitudinal magnetization
      - ⟨Ix_R⟩, ⟨Iy_R⟩               : rare spin transverse (x,y) magnetization
      - state_norm                   : ‖ψ(t)‖ at each time point (should stay ≈ 1)
    """
    if params.steps < 2 or params.t_final <= 0.0:
        raise ValueError("Bad time grid: steps >= 2 and t_final > 0.")

    H, eops = build_hamiltonian_rare(params)
    psi0 = initial_state_rare(params)

    t = np.linspace(0.0, params.t_final, params.steps)

    # Build QuTiP Options object only if any override is provided
    options = None
    if any(
            v is not None
            for v in (
                    getattr(params, "solver_atol", None),
                    getattr(params, "solver_rtol", None),
                    getattr(params, "solver_nsteps", None),
                    getattr(params, "solver_max_step", None),
            )
    ):
        options = {}

        options["store_states"] = True
        options["store_final_state"] = True

        if getattr(params, "solver_atol", None) is not None:
            options["atol"] = params.solver_atol
        if getattr(params, "solver_rtol", None) is not None:
            options["rtol"] = params.solver_rtol
        if getattr(params, "solver_nsteps", None) is not None:
            options["nsteps"] = params.solver_nsteps
        if getattr(params, "solver_max_step", None) is not None:
            options["max_step"] = params.solver_max_step

    res = qt.sesolve(
        H,
        psi0,
        t,
        e_ops=[
            eops["Ix_sea"],
            eops["Iy_sea"],
            eops["Iz_sea"],
            eops["Iz_R"],
            eops["Ix_R"],
            eops["Iy_R"],
        ],
        options=options,
    )

    # State norms as an extra diagnostic observable
    norms = np.array([psi.norm() for psi in res.states])

    out = {
        "Ix_sea": np.real(res.expect[0]),
        "Iy_sea": np.real(res.expect[1]),
        "Iz_sea": np.real(res.expect[2]),
        "Iz_R":   np.real(res.expect[3]),
        "Ix_R":   np.real(res.expect[4]),
        "Iy_R":   np.real(res.expect[5]),
        "state_norm": norms,
    }
    return t, out
