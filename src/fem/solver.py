"""
fem/solver.py
=============
Global stiffness assembly and linear-elastic FEA solve K(x) U = F for the
SIMP-interpolated plane-stress problem.

    E_e(x_e) = Emin + x_e^p (E0 - Emin)
    c        = F^T U = sum_e E_e * u_e^T KE u_e
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import factorized, spsolve


def simp_modulus(x_phys: np.ndarray, p: float = 3.0, E0: float = 1.0, Emin: float = 1e-9) -> np.ndarray:
    """SIMP Young's modulus of every element, same shape as x_phys."""
    return Emin + x_phys ** p * (E0 - Emin)


def compliance_sensitivity(
    x_phys: np.ndarray, ce: np.ndarray, p: float = 3.0, E0: float = 1.0, Emin: float = 1e-9
) -> np.ndarray:
    """dc / dx_phys = -p x_phys^(p-1) (E0 - Emin) u_e^T KE u_e  (always <= 0)."""
    return -p * x_phys ** (p - 1.0) * (E0 - Emin) * ce


def solve(
    x_phys: np.ndarray,
    case: Dict[str, Any],
    KE: np.ndarray,
    edofMat: np.ndarray,
    iK: np.ndarray,
    jK: np.ndarray,
    p: float = 3.0,
    E0: float = 1.0,
    Emin: float = 1e-9,
    factorize: bool = False,
) -> Tuple[np.ndarray, float, np.ndarray, Optional[Any]]:
    """
    Assemble K(x_phys) and solve K U = F.

    Parameters
    ----------
    x_phys : physical densities, shape (nely, nelx).
    case   : dict with keys nelx, nely, fixed_dofs, F.
    KE, edofMat, iK, jK : element matrix and mesh maps (see fem.element / fem.mesh).
    p, E0, Emin : SIMP parameters.
    factorize : also return the factorisation of the reduced stiffness matrix.

    Returns
    -------
    U  : displacement vector, shape (ndof,).
    c  : compliance F^T U (float).
    ce : unit-modulus element energies u_e^T KE u_e, shape (nely, nelx).
    solver : callable applying K_free^{-1}, or None.
    """
    nelx, nely = int(case["nelx"]), int(case["nely"])
    ndof = 2 * (nelx + 1) * (nely + 1)
    if x_phys.shape != (nely, nelx):
        raise ValueError(f"x_phys has shape {x_phys.shape}, expected {(nely, nelx)}")
    F = np.asarray(case["F"], dtype=np.float64)

    E_e = simp_modulus(x_phys.ravel(order="F"), p, E0, Emin)

    # Triplet assembly (duplicates are summed) and symmetrisation against round-off.
    sK = (E_e[:, None] * KE.ravel()[None, :]).ravel()
    K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
    K = (K + K.T) * 0.5

    free = np.ones(ndof, dtype=bool)
    free[np.asarray(case["fixed_dofs"], dtype=np.int64)] = False
    free = np.flatnonzero(free)
    K_free = K[free, :][:, free].tocsc()

    U = np.zeros(ndof)
    solver = None
    if factorize:
        solver = factorized(K_free)
        U[free] = solver(F[free])
    else:
        U[free] = spsolve(K_free, F[free])

    u_e = U[edofMat]                                  # (nel, 8)
    ce_flat = np.einsum("ij,jk,ik->i", u_e, KE, u_e)  # u_e^T KE u_e
    c = float(E_e @ ce_flat)
    ce = ce_flat.reshape((nely, nelx), order="F")
    return U, c, ce, solver


def strain_energy_density(
    U: np.ndarray,
    x_phys: np.ndarray,
    KE: np.ndarray,
    edofMat: np.ndarray,
    p: float = 3.0,
    E0: float = 1.0,
    Emin: float = 1e-9,
) -> np.ndarray:
    """
    Element strain energy w_e = 0.5 * E_e * u_e^T KE u_e, shape (nely, nelx).
    sum(w) equals half the compliance. Used as an RL observation channel.
    """
    nely, nelx = x_phys.shape
    u_e = U[edofMat]
    ce_flat = np.einsum("ij,jk,ik->i", u_e, KE, u_e)
    E_e = simp_modulus(x_phys.ravel(order="F"), p, E0, Emin)
    return (0.5 * E_e * ce_flat).reshape((nely, nelx), order="F")
