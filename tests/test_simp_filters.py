"""
test_simp_filters.py
====================
Pytest suite for simp/filters.py and simp/simp.py.
Conforms to SPEC.md Sections 5 and 6.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
import numpy as np
from fem.cases import cantilever_case, mbb_case
from fem.element import lk
from fem.mesh import build_edofmat, build_sparse_indices
from fem.solver import solve
from simp.filters import build_filter, density_filter
from simp.simp import run_simp


class TestFilters:
    """Tests for simp/filters.py."""

    def test_filter_rmin_one_identity(self):
        """With rmin = 1.0, distance to any neighbor is >= 1, so H is strictly diagonal."""
        nelx, nely = 8, 4
        H, Hs = build_filter(nelx, nely, rmin=1.0)
        
        # Diagonal elements must equal rmin = 1.0
        diag = H.diagonal()
        assert np.allclose(diag, 1.0)
        # Off-diagonal elements must be zero
        assert H.nnz == nelx * nely

    def test_hs_strictly_positive(self):
        """Row sum vector Hs must be strictly positive for all elements."""
        nelx, nely = 15, 5
        _, Hs = build_filter(nelx, nely, rmin=1.8)
        assert np.all(Hs > 0.0)

    def test_uniform_field_invariance(self):
        """A uniform density field must remain exactly invariant under density_filter."""
        nelx, nely = 12, 6
        H, Hs = build_filter(nelx, nely, rmin=1.5)
        
        x_uniform = np.full((nely, nelx), 0.42, dtype=np.float64)
        x_filtered = density_filter(x_uniform, H, Hs)
        
        assert np.allclose(x_filtered, 0.42, atol=1e-12)


class TestSIMPOptimizer:
    """Tests for simp/simp.py."""

    def test_finite_difference_gradient(self):
        """
        Finite-difference verification of analytic sensitivity dc.
        Central difference: dc_num = (c(x + eps) - c(x - eps)) / (2 * eps).
        Must match analytic dc within 1e-4 relative error.
        """
        nelx, nely = 6, 4
        case = cantilever_case(nelx=nelx, nely=nely, volfrac=0.5)
        KE = lk(nu=0.3)
        edofMat = build_edofmat(nelx, nely)
        iK, jK = build_sparse_indices(edofMat)
        
        # Test on a non-uniform density field
        rng = np.random.default_rng(42)
        x = rng.uniform(0.2, 0.8, size=(nely, nelx))
        
        # Analytic solve
        p = 3.0
        U, c, ce, _ = solve(x, case, KE, edofMat, iK, jK, p=p)
        dc_analytic = -p * (x ** (p - 1.0)) * ce
        
        # Central difference perturbation on element (iy, ix) = (2, 3)
        iy, ix = 2, 3
        eps = 1e-6
        
        x_plus = x.copy()
        x_plus[iy, ix] += eps
        _, c_plus, _, _ = solve(x_plus, case, KE, edofMat, iK, jK, p=p)
        
        x_minus = x.copy()
        x_minus[iy, ix] -= eps
        _, c_minus, _, _ = solve(x_minus, case, KE, edofMat, iK, jK, p=p)
        
        dc_num = (c_plus - c_minus) / (2.0 * eps)
        dc_ana = dc_analytic[iy, ix]
        
        rel_error = abs(dc_num - dc_ana) / abs(dc_ana)
        assert rel_error < 1e-4, f"Analytic {dc_ana} vs Numerical {dc_num} (rel err: {rel_error})"

    def test_cantilever_optimization_convergence(self):
        """Classical SIMP on Cantilever must decrease compliance and meet volume constraint."""
        case = cantilever_case(nelx=30, nely=10, volfrac=0.4)
        res = run_simp(case, rmin=1.5, max_iter=30, tol=0.01)
        
        history_c = res["history_c"]
        assert len(history_c) > 5
        assert history_c[-1] < history_c[0]  # Compliance decreased
        
        final_vol = float(np.mean(res["x_phys"]))
        assert abs(final_vol - 0.4) < 1e-3, f"Volume fraction {final_vol} != 0.4"

    def test_mbb_optimization_convergence(self):
        """Classical SIMP on MBB beam must decrease compliance and meet volume constraint."""
        case = mbb_case(nelx=30, nely=10, volfrac=0.5)
        res = run_simp(case, rmin=1.5, max_iter=30, tol=0.01)
        
        assert res["history_c"][-1] < res["history_c"][0]
        final_vol = float(np.mean(res["x_phys"]))
        assert abs(final_vol - 0.5) < 1e-3
