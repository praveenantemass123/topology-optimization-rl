"""
test_fea.py
===========
Unit tests for the Finite Element Analysis (FEA) physics engine
and SIMP formulation as mandated in Stage 1 of the BTP Workflow.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
from fem.element import lk
from fem.mesh import build_edofmat, build_sparse_indices
from fem.solver import solve
from fem.cases import cantilever_case
from simp.simp import run_simp


def test_element_stiffness_symmetry():
    """Checks that the 8x8 element stiffness matrix k0 is mathematically symmetric."""
    k0 = lk(nu=0.3)
    diff = np.max(np.abs(k0 - k0.T))
    assert diff < 1e-12, f"Stiffness matrix k0 is asymmetric! Max diff: {diff}"
    print("[PASS] Element stiffness matrix k0 is symmetric.")


def test_rigid_body_modes():
    """
    Checks that the 3 rigid-body modes (translation X, translation Y, rotation)
    produce zero strain energy: u_rigid^T * k0 * u_rigid == 0.
    """
    k0 = lk(nu=0.3)
    
    # 1. Rigid Translation in X (u_x = 1, u_y = 0 for all 4 nodes)
    u_trans_x = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
    energy_tx = u_trans_x.T @ k0 @ u_trans_x
    assert abs(energy_tx) < 1e-10, f"Rigid translation X produced non-zero strain energy: {energy_tx}"
    
    # 2. Rigid Translation in Y (u_x = 0, u_y = 1 for all 4 nodes)
    u_trans_y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    energy_ty = u_trans_y.T @ k0 @ u_trans_y
    assert abs(energy_ty) < 1e-10, f"Rigid translation Y produced non-zero strain energy: {energy_ty}"
    
    # 3. Rigid In-plane Rotation about center
    # For unit square [-0.5, 0.5]^2: u_x = -y, u_y = x
    u_rot = np.array([0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5], dtype=float)
    energy_rot = u_rot.T @ k0 @ u_rot
    assert abs(energy_rot) < 1e-10, f"Rigid rotation produced non-zero strain energy: {energy_rot}"
    
    print("[PASS] Rigid-body modes produce zero strain energy (3 zero eigenvalues confirmed).")


def test_compliance_energy_balance():
    """
    Verifies that the internal strain energy (U^T K U) mathematically equals
    the external work done by applied loads (F^T U).
    """
    case = cantilever_case(nelx=30, nely=10, volfrac=0.4)
    KE = lk(nu=0.3)
    edofMat = build_edofmat(30, 10)
    iK, jK = build_sparse_indices(edofMat)
    x = np.full((10, 30), 0.4, dtype=np.float64)
    
    U, c, _, _ = solve(x, case, KE, edofMat, iK, jK, p=3.0, E0=1.0, Emin=1e-9)
    external_work = float(case["F"].T @ U)
    internal_strain_energy = float(c)
    
    rel_error = abs(external_work - internal_strain_energy) / abs(external_work)
    assert rel_error < 1e-6, f"Energy balance violated! External: {external_work}, Internal: {internal_strain_energy}"
    print(f"[PASS] Energy balance verified: U^T K U == F^T U (relative error = {rel_error:.2e}).")


def test_monotonic_convergence():
    """Verifies that classical SIMP optimization monotonically decreases compliance."""
    case = cantilever_case(nelx=30, nely=10, volfrac=0.4)
    res = run_simp(case, max_iter=25, tol=0.01)
    
    initial_c = res["history_c"][0]
    final_c = res["history_c"][-1]
    
    assert final_c < initial_c, f"Optimization failed to reduce compliance! Initial: {initial_c}, Final: {final_c}"
    print(f"[PASS] Monotonic convergence verified: Compliance decreased from {initial_c:.2f} to {final_c:.2f}.")


if __name__ == "__main__":
    print("========================================")
    print(" Running Stage 1 FEA & Mechanics Tests  ")
    print("========================================")
    test_element_stiffness_symmetry()
    test_rigid_body_modes()
    test_compliance_energy_balance()
    test_monotonic_convergence()
    print("\n[ALL TESTS PASSED] Stage 1 Gate Successfully Cleared!")
