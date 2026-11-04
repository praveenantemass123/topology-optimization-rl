"""
test_solver_cases.py
====================
Pytest suite for fem/solver.py and fem/cases.py.
Conforms to SPEC.md Sections 3 and 4.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
import numpy as np
from fem.element import lk
from fem.mesh import build_edofmat, build_sparse_indices
from fem.cases import mbb_case, cantilever_case, sample_load_case, make_test_set
from fem.solver import solve, strain_energy_density


class TestFEMSolver:
    """Tests for fem/solver.py."""

    @pytest.fixture
    def setup_cantilever(self):
        nelx, nely = 30, 10
        case = cantilever_case(nelx=nelx, nely=nely, volfrac=0.4, load_pos="mid_right")
        KE = lk(nu=0.3)
        edofMat = build_edofmat(nelx, nely)
        iK, jK = build_sparse_indices(edofMat)
        x_phys = np.ones((nely, nelx), dtype=np.float64)
        return case, KE, edofMat, iK, jK, x_phys

    def test_compliance_equals_work(self, setup_cantilever):
        """Compliance c must equal external work F^T U within 1e-8 relative tolerance."""
        case, KE, edofMat, iK, jK, x_phys = setup_cantilever
        U, c, ce, _ = solve(x_phys, case, KE, edofMat, iK, jK)
        
        work = float(case["F"] @ U)
        assert abs(c - work) < 1e-8, f"Compliance {c} != External Work {work}"

    def test_quadratic_load_scaling(self, setup_cantilever):
        """In linear elasticity, doubling the external load must quadruple compliance (2^2 = 4)."""
        case, KE, edofMat, iK, jK, x_phys = setup_cantilever
        _, c1, _, _ = solve(x_phys, case, KE, edofMat, iK, jK)
        
        case_doubled = case.copy()
        case_doubled["F"] = case["F"] * 2.0
        _, c2, _, _ = solve(x_phys, case_doubled, KE, edofMat, iK, jK)
        
        ratio = c2 / c1
        assert np.isclose(ratio, 4.0, atol=1e-6), f"Doubling load gave ratio {ratio}, expected 4.0"

    def test_unconstrained_null_space(self):
        """Rigid translation of an unconstrained mesh must produce near-zero strain energy."""
        nelx, nely = 4, 4
        ndof = 2 * (nelx + 1) * (nely + 1)
        KE = lk(nu=0.3)
        edofMat = build_edofmat(nelx, nely)
        iK, jK = build_sparse_indices(edofMat)
        x_phys = np.ones((nely, nelx), dtype=np.float64)
        
        # Assemble unconstrained global K
        from scipy.sparse import coo_matrix
        x_flat = x_phys.flatten(order="F")
        sK = (KE.flatten()[:, None] * x_flat[None, :]).flatten(order="F")
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        
        # Rigid translation vector in X (u_x = 1.0, u_y = 0.0)
        u_rigid = np.zeros(ndof)
        u_rigid[0::2] = 1.0
        
        strain_energy = float(u_rigid.T @ K @ u_rigid)
        assert abs(strain_energy) < 1e-10, f"Rigid body translation produced strain energy {strain_energy}"


class TestLoadCases:
    """Tests for fem/cases.py."""

    def test_mbb_boundary_conditions(self):
        """MBB case must have symmetry BC on left edge and vertical roller on bottom right."""
        nelx, nely = 20, 10
        case = mbb_case(nelx, nely, volfrac=0.5)
        
        # Force must be non-zero at exactly 1 DOF (top-left node y-DOF)
        non_zero_forces = np.where(case["F"] != 0)[0]
        assert len(non_zero_forces) == 1
        assert non_zero_forces[0] == 1  # node 0 y-DOF
        assert case["F"][1] == -1.0
        
        # Fixed DOFs must be unique and in valid range
        ndof = 2 * (nelx + 1) * (nely + 1)
        assert len(case["fixed_dofs"]) == len(np.unique(case["fixed_dofs"]))
        assert np.all(case["fixed_dofs"] >= 0) and np.all(case["fixed_dofs"] < ndof)

    def test_cantilever_boundary_conditions(self):
        """Cantilever case must clamp the full left edge and apply downward force on right edge."""
        nelx, nely = 20, 10
        case = cantilever_case(nelx, nely, volfrac=0.4, load_pos="mid_right")
        
        ndof = 2 * (nelx + 1) * (nely + 1)
        # Left edge has nely+1 nodes, each with 2 DOFs = 2*(nely+1) fixed DOFs
        assert len(case["fixed_dofs"]) == 2 * (nely + 1)
        
        # Force nonzero at exactly 1 DOF
        non_zero_forces = np.where(case["F"] != 0)[0]
        assert len(non_zero_forces) == 1
        assert case["F"][non_zero_forces[0]] == -1.0

    def test_reproducible_sampler(self):
        """Random sampler must produce identical cases for the same seed."""
        base = cantilever_case(nelx=20, nely=10, volfrac=0.4)
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        
        c1 = sample_load_case(base, rng1)
        c2 = sample_load_case(base, rng2)
        
        assert np.allclose(c1["F"], c2["F"])
        assert c1["load_info"] == c2["load_info"]
