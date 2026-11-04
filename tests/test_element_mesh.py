"""
test_element_mesh.py
====================
Pytest suite for fem/element.py and fem/mesh.py.
Conforms to SPEC.md Sections 1 and 2.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
import numpy as np
from fem.element import lk, lk_closed_form, lk_gauss
from fem.mesh import build_edofmat, build_sparse_indices, node_from_xy


class TestElementStiffness:
    """Tests for fem/element.py."""

    @pytest.mark.parametrize("nu", [0.3, 0.0, 0.25, 0.45])
    def test_closed_form_equals_gauss_quadrature(self, nu):
        """Route 1 (closed form) must equal Route 2 (Gauss quadrature) within machine precision."""
        ke_closed = lk_closed_form(nu)
        ke_gauss = lk_gauss(nu)
        assert np.allclose(ke_closed, ke_gauss, atol=1e-10), f"Mismatch for nu={nu}"

    def test_ke_symmetry(self):
        """Element stiffness matrix KE must be strictly symmetric."""
        ke = lk(nu=0.3)
        diff = np.max(np.abs(ke - ke.T))
        assert diff < 1e-12, f"KE is not symmetric! Max diff: {diff}"

    def test_rigid_body_eigenvalues(self):
        """
        Unconstrained 2D element stiffness matrix must have exactly 3 near-zero eigenvalues
        (corresponding to 2 rigid translations and 1 rigid rotation), and the rest (5) must be strictly positive.
        """
        ke = lk(nu=0.3)
        eigenvalues = np.sort(np.linalg.eigvalsh(ke))
        
        # 3 near-zero eigenvalues (< 1e-10)
        assert np.all(np.abs(eigenvalues[:3]) < 1e-10), f"First 3 eigenvalues not near zero: {eigenvalues[:3]}"
        # Remaining 5 eigenvalues strictly positive
        assert np.all(eigenvalues[3:] > 0.05), f"Remaining eigenvalues not positive: {eigenvalues[3:]}"


class TestMeshConnectivity:
    """Tests for fem/mesh.py."""

    def test_single_element_mesh(self):
        """For a 1x1 mesh, the 8 DOFs must be an exact permutation of integers 0 through 7."""
        edofMat = build_edofmat(nelx=1, nely=1)
        assert edofMat.shape == (1, 8)
        assert set(edofMat[0].tolist()) == set(range(8))

    @pytest.mark.parametrize("nelx, nely", [(2, 2), (10, 5), (30, 10)])
    def test_unique_dofs_count(self, nelx, nely):
        """The total number of unique DOFs across all elements must equal 2*(nelx+1)*(nely+1)."""
        edofMat = build_edofmat(nelx, nely)
        expected_total_dofs = 2 * (nelx + 1) * (nely + 1)
        unique_dofs = np.unique(edofMat)
        
        assert len(unique_dofs) == expected_total_dofs
        assert unique_dofs.min() == 0
        assert unique_dofs.max() == expected_total_dofs - 1

    @pytest.mark.parametrize("nelx, nely", [(4, 3), (12, 6)])
    def test_element_dofs_are_distinct(self, nelx, nely):
        """Within each individual element, all 8 DOFs must be strictly distinct."""
        edofMat = build_edofmat(nelx, nely)
        for e in range(nelx * nely):
            assert len(set(edofMat[e].tolist())) == 8, f"Element {e} has duplicate DOFs!"

    def test_sparse_indices_dimensions(self):
        """build_sparse_indices must return (iK, jK) vectors of length nel * 64."""
        nelx, nely = 8, 4
        nel = nelx * nely
        edofMat = build_edofmat(nelx, nely)
        iK, jK = build_sparse_indices(edofMat)
        
        assert len(iK) == nel * 64
        assert len(jK) == nel * 64

    def test_node_from_xy(self):
        """node_from_xy must match column-major node numbering n = ix*(nely+1) + iy."""
        nely = 10
        assert node_from_xy(0, 0, nely) == 0       # Top-left node
        assert node_from_xy(0, nely, nely) == nely # Bottom-left node
        assert node_from_xy(1, 0, nely) == nely + 1# Top node of second column
