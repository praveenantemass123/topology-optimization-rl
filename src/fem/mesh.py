"""
fem/mesh.py
===========
Mesh connectivity, node numbering, and degree of freedom (DOF) mapping
for structured 2D grids of Q4 quad elements.

Conforms to SPEC.md Section 0 & Section 2.
"""

from typing import Tuple
import numpy as np


def node_from_xy(ix: int, iy: int, nely: int) -> int:
    """
    Returns the global node index for column ix and row iy.
    Node indexing runs top-to-bottom, then left-to-right (column-major / Fortran order):
    n = ix * (nely + 1) + iy
    """
    return ix * (nely + 1) + iy


def build_edofmat(nelx: int, nely: int) -> np.ndarray:
    """
    Builds the element degree-of-freedom mapping matrix (edofMat) of shape (nel, 8).
    
    Element numbering:
        e = ix * nely + iy  (column-major)
        
    Element corner nodes:
        Top-left node: n1 = ix * (nely + 1) + iy
        Top-right node: n2 = (ix + 1) * (nely + 1) + iy
        
    DOFs ordering (8 DOFs counter-clockwise starting from lower-left):
        1. Lower-left node (n1 + 1):  dof_x = 2*n1 + 2, dof_y = 2*n1 + 3
        2. Lower-right node (n2 + 1): dof_x = 2*n2 + 2, dof_y = 2*n2 + 3
        3. Upper-right node (n2):     dof_x = 2*n2,     dof_y = 2*n2 + 1
        4. Upper-left node (n1):      dof_x = 2*n1,     dof_y = 2*n1 + 1
        
    Matches the 88-line MATLAB code definition in 0-based indexing:
        [2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3, 2*n2, 2*n2+1, 2*n1, 2*n1+1]
        
    Returns:
        edofMat: int64 array of shape (nelx * nely, 8).
    """
    nel = nelx * nely
    edofMat = np.zeros((nel, 8), dtype=np.int64)
    
    for ix in range(nelx):
        for iy in range(nely):
            e = ix * nely + iy
            n1 = ix * (nely + 1) + iy
            n2 = (ix + 1) * (nely + 1) + iy
            
            edofMat[e, :] = [
                2 * n1 + 2,
                2 * n1 + 3,
                2 * n2 + 2,
                2 * n2 + 3,
                2 * n2,
                2 * n2 + 1,
                2 * n1,
                2 * n1 + 1
            ]
            
    return edofMat


def build_sparse_indices(edofMat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds the global row (iK) and column (jK) coordinate vectors for sparse matrix assembly.
    
    Parameters:
        edofMat: ndarray of shape (nel, 8)
        
    Returns:
        iK: 1D array of length nel * 64
        jK: 1D array of length nel * 64
    """
    # For each element, row index repeats 8 times per row entry,
    # and column index tiles the 8 DOFs across each row.
    iK = np.repeat(edofMat, 8, axis=1).flatten()
    jK = np.tile(edofMat, (1, 8)).flatten()
    return iK, jK
