"""
simp/filters.py
===============
Linear "hat" filters for mesh-independent topology optimization.

    H[e, f] = max(0, rmin - dist(e, f)),   Hs[e] = sum_f H[e, f]

Element fields are (nely, nelx) arrays; internally they are flattened with
order="F" so that the flat index is e = ix * nely + iy (see fem.mesh).
"""

from typing import Tuple

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix


def build_filter(nelx: int, nely: int, rmin: float) -> Tuple[csc_matrix, np.ndarray]:
    """
    Build the filter weight matrix H (nel x nel, symmetric) and its row sums Hs.
    rmin is the filter radius in element widths; rmin <= 1 gives the identity.
    """
    if rmin <= 0:
        raise ValueError(f"rmin must be positive, got {rmin}")
    nel = nelx * nely
    reach = int(np.ceil(rmin)) - 1

    ix, iy = np.meshgrid(np.arange(nelx), np.arange(nely), indexing="ij")
    ix, iy = ix.ravel(), iy.ravel()                    # element e = ix * nely + iy
    rows, cols, vals = [], [], []
    for dx in range(-reach, reach + 1):
        for dy in range(-reach, reach + 1):
            w = rmin - np.hypot(dx, dy)
            if w <= 0.0:
                continue
            jx, jy = ix + dx, iy + dy
            ok = (jx >= 0) & (jx < nelx) & (jy >= 0) & (jy < nely)
            rows.append((ix * nely + iy)[ok])
            cols.append((jx * nely + jy)[ok])
            vals.append(np.full(int(ok.sum()), w))

    H = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nel, nel)
    ).tocsc()
    Hs = np.asarray(H.sum(axis=1)).ravel()
    return H, Hs


def _apply(H: csc_matrix, field: np.ndarray) -> np.ndarray:
    """H @ field for a (nely, nelx) field, returned as a flat (order='F') vector."""
    return H @ field.ravel(order="F")


def density_filter(x: np.ndarray, H: csc_matrix, Hs: np.ndarray) -> np.ndarray:
    """Physical density x_phys = (H x) / Hs."""
    return (_apply(H, x) / Hs).reshape(x.shape, order="F")


def sensitivity_filter(x: np.ndarray, dc: np.ndarray, H: csc_matrix, Hs: np.ndarray) -> np.ndarray:
    """Sigmund's sensitivity filter: dc_f = H (x dc) / (Hs max(1e-3, x))."""
    x_flat = x.ravel(order="F")
    out = _apply(H, x * dc) / (Hs * np.maximum(1e-3, x_flat))
    return out.reshape(x.shape, order="F")


def chain_rule_density(
    dc: np.ndarray, dv: np.ndarray, H: csc_matrix, Hs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chain rule through the density filter: given derivatives with respect to the
    physical density, return derivatives with respect to the design variable,
        d/dx = H^T (d/dx_phys / Hs).
    """
    def back(g: np.ndarray) -> np.ndarray:
        return (H.T @ (g.ravel(order="F") / Hs)).reshape(g.shape, order="F")

    return back(dc), back(dv)
