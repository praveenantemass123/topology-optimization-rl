"""
simp/filters.py
===============
Mesh-independency sensitivity and density filters for topology optimization.

Conforms to SPEC.md Section 0 & Section 5.
"""

from typing import Tuple
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix


def build_filter(nelx: int, nely: int, rmin: float) -> Tuple[csc_matrix, np.ndarray]:
    """
    Builds the sparse convolution weight matrix H and normalization vector Hs.
    
    Parameters:
        nelx, nely: Grid dimensions.
        rmin: Filter radius in element units.
        
    Returns:
        H: Sparse csc_matrix of shape (nel, nel).
        Hs: 1D array of row sums, length nel.
    """
    nel = nelx * nely
    iH = []
    jH = []
    sH = []
    
    r_int = int(np.ceil(rmin)) - 1
    
    for ix in range(nelx):
        for iy in range(nely):
            e1 = ix * nely + iy
            
            ix_min = max(0, ix - r_int)
            ix_max = min(nelx, ix + r_int + 1)
            iy_min = max(0, iy - r_int)
            iy_max = min(nely, iy + r_int + 1)
            
            for jx in range(ix_min, ix_max):
                for jy in range(iy_min, iy_max):
                    e2 = jx * nely + jy
                    dist = np.hypot(ix - jx, iy - jy)
                    weight = max(0.0, rmin - dist)
                    if weight > 0.0:
                        iH.append(e1)
                        jH.append(e2)
                        sH.append(weight)
                        
    H = coo_matrix((sH, (iH, jH)), shape=(nel, nel)).tocsc()
    Hs = np.array(H.sum(axis=1)).flatten()
    return H, Hs


def density_filter(x: np.ndarray, H: csc_matrix, Hs: np.ndarray) -> np.ndarray:
    """
    Computes physical density x_phys = (H @ x) / Hs.
    """
    nely, nelx = x.shape
    x_flat = x.flatten(order="F")
    x_phys_flat = (H @ x_flat) / Hs
    return x_phys_flat.reshape((nely, nelx), order="F")


def sensitivity_filter(
    x: np.ndarray,
    dc: np.ndarray,
    H: csc_matrix,
    Hs: np.ndarray
) -> np.ndarray:
    """
    Applies Sigmund's sensitivity filter:
    dcn = H @ (x * dc) / (Hs * max(1e-3, x))
    """
    nely, nelx = x.shape
    x_flat = x.flatten(order="F")
    dc_flat = dc.flatten(order="F")
    
    dcn_flat = (H @ (x_flat * dc_flat)) / (Hs * np.maximum(1e-3, x_flat))
    return dcn_flat.reshape((nely, nelx), order="F")


def chain_rule_density(
    dc: np.ndarray,
    dv: np.ndarray,
    H: csc_matrix,
    Hs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies adjoint chain rule for the density filter:
    dc_mod = H^T @ (dc / Hs)
    dv_mod = H^T @ (dv / Hs)
    """
    nely, nelx = dc.shape
    dc_flat = dc.flatten(order="F")
    dv_flat = dv.flatten(order="F")
    
    dc_mod_flat = H.T @ (dc_flat / Hs)
    dv_mod_flat = H.T @ (dv_flat / Hs)
    
    return (
        dc_mod_flat.reshape((nely, nelx), order="F"),
        dv_mod_flat.reshape((nely, nelx), order="F")
    )
