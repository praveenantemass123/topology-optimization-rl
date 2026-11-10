"""
fem/mesh.py
===========
Node numbering, element numbering and DOF maps for a structured grid of
nelx x nely unit-square Q4 elements.

Conventions (same as the 88-line code, 0-based):
    * ix = 0..nelx runs left to right, iy = 0..nely runs TOP to BOTTOM.
    * node      n = ix * (nely + 1) + iy          (column-major)
    * element   e = ix * nely + iy                (column-major)
    * DOFs of node n: x -> 2n, y -> 2n + 1
    * 2-D element fields are stored as arrays of shape (nely, nelx); flattening
      them with order="F" gives the element numbering above.
"""

from typing import Tuple

import numpy as np


def node_from_xy(ix: int, iy: int, nely: int) -> int:
    """Global node index of grid column ix and row iy (iy = 0 is the top row)."""
    return ix * (nely + 1) + iy


def build_edofmat(nelx: int, nely: int) -> np.ndarray:
    """
    Element-to-DOF map, shape (nelx * nely, 8), dtype int64.

    With n1 the top-left node of an element and n2 its top-right node, the row is
        [2n1+2, 2n1+3, 2n2+2, 2n2+3, 2n2, 2n2+1, 2n1, 2n1+1]
    i.e. lower-left, lower-right, upper-right, upper-left (counter-clockwise),
    which is the local order assumed by fem.element.lk.
    """
    ix, iy = np.meshgrid(np.arange(nelx), np.arange(nely), indexing="ij")
    n1 = (ix * (nely + 1) + iy).ravel()           # C-order ravel of (ix, iy) == e = ix*nely + iy
    n2 = n1 + (nely + 1)
    offsets = np.stack([
        2 * n1 + 2, 2 * n1 + 3,
        2 * n2 + 2, 2 * n2 + 3,
        2 * n2,     2 * n2 + 1,
        2 * n1,     2 * n1 + 1,
    ], axis=1)
    return offsets.astype(np.int64)


def build_sparse_indices(edofMat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Row (iK) and column (jK) index vectors, each of length nel * 64, for the
    triplet assembly of the global stiffness matrix. Entry 8*a + b of an
    element's block is KE[a, b].
    """
    iK = np.repeat(edofMat, 8, axis=1).ravel()
    jK = np.tile(edofMat, (1, 8)).ravel()
    return iK, jK
