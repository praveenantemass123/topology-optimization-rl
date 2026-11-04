"""
fem/solver.py
=============
Global stiffness assembly and Finite Element Analysis (FEA) solver
for 2D linear elasticity under plane stress.

Conforms to SPEC.md Section 0 & Section 3.
"""

from typing import Tuple, Optional, Any
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve, factorized


def solve(
    x_phys: np.ndarray,
    case: dict,
    KE: np.ndarray,
    edofMat: np.ndarray,
    iK: np.ndarray,
    jK: np.ndarray,
    p: float = 3.0,
    E0: float = 1.0,
    Emin: float = 1e-9,
    factorize: bool = False
) -> Tuple[np.ndarray, float, np.ndarray, Optional[Any]]:
    """
    Assembles the global stiffness matrix K(x_phys) and solves K * U = F.
    
    Parameters:
        x_phys: 2D array of element physical densities, shape (nely, nelx).
        case: Case dictionary conforming to SPEC.md Section 0.
        KE: 8x8 element stiffness matrix.
        edofMat: Element DOF mapping array, shape (nel, 8).
        iK, jK: Coordinate arrays for sparse assembly, length nel * 64.
        p: SIMP penalization power exponent. Default = 3.0.
        E0: Young's modulus of solid material. Default = 1.0.
        Emin: Artificial Young's modulus of void to prevent singularity. Default = 1e-9.
        factorize: If True, also returns the SciPy factorized solver object.
        
    Returns:
        U: Global displacement vector, shape (ndof,).
        c: Total structural compliance (scalar float), c = U^T K U.
        ce: Element compliance array, shape (nely, nelx) in Fortran order.
        solver_fac: Optional factorized solver object if factorize=True, else None.
    """
    nelx = case["nelx"]
    nely = case["nely"]
    ndof = 2 * (nelx + 1) * (nely + 1)
    fixed_dofs = case["fixed_dofs"]
    F = case["F"]
    
    # 1. Flatten densities in column-major (Fortran) order matching e = ix * nely + iy
    x_phys_flat = x_phys.flatten(order="F")
    
    # 2. SIMP material interpolation law
    E_e = Emin + (x_phys_flat ** p) * (E0 - Emin)
    
    # 3. Assemble global sparse stiffness matrix K
    sK = (E_e[:, None] * KE.flatten()[None, :]).flatten()
    K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
    K = (K + K.T) / 2.0  # Enforce exact mathematical symmetry
    
    # 4. Extract free degrees of freedom
    all_dofs = np.arange(ndof, dtype=np.int64)
    free = np.setdiff1d(all_dofs, fixed_dofs)
    
    # 5. Solve linear system K_free * U_free = F_free
    K_free = K[free, :][:, free]
    F_free = F[free]
    
    U = np.zeros(ndof, dtype=np.float64)
    solver_fac = None
    
    if factorize:
        solver_fac = factorized(K_free)
        U[free] = solver_fac(F_free)
    else:
        U[free] = spsolve(K_free, F_free)
        
    # 6. Element compliance ce = u_e^T * KE * u_e
    U_elem = U[edofMat]  # shape (nel, 8)
    ce_flat = np.sum((U_elem @ KE) * U_elem, axis=1)
    
    # Total compliance c = sum(E_e * ce)
    c = float(np.sum(E_e * ce_flat))
    
    # Reshape ce to (nely, nelx) in Fortran order
    ce = ce_flat.reshape((nely, nelx), order="F")
    
    return U, c, ce, solver_fac


def strain_energy_density(
    U: np.ndarray,
    x_phys: np.ndarray,
    KE: np.ndarray,
    edofMat: np.ndarray,
    p: float = 3.0,
    E0: float = 1.0,
    Emin: float = 1e-9
) -> np.ndarray:
    """
    Computes the element strain energy density field w_e = 0.5 * E_e * u_e^T * KE * u_e.
    Used as a physical observation channel for Reinforcement Learning.
    
    Returns:
        w: 2D array of strain energy density, shape (nely, nelx) in Fortran order.
    """
    nely, nelx = x_phys.shape
    x_phys_flat = x_phys.flatten(order="F")
    E_e = Emin + (x_phys_flat ** p) * (E0 - Emin)
    
    U_elem = U[edofMat]
    ce_flat = np.sum((U_elem @ KE) * U_elem, axis=1)
    
    sed_flat = 0.5 * E_e * ce_flat
    return sed_flat.reshape((nely, nelx), order="F")
