"""
fem/element.py
==============
Stiffness matrix of a 4-node bilinear (Q4) unit-square element in plane stress,
for Young's modulus E = 1.

Local DOF order (matches the 88-line code of Andreassen et al., 2011):
    [u1, v1, u2, v2, u3, v3, u4, v4]
with the nodes taken counter-clockwise starting from the lower-left corner.

Two independent implementations are provided so that each can verify the other:
the closed-form expression and 2x2 Gauss quadrature.
"""

from typing import Literal

import numpy as np


def lk_closed_form(nu: float = 0.3) -> np.ndarray:
    """Closed-form 8x8 element stiffness matrix (E = 1)."""
    k = np.array([
        1 / 2 - nu / 6,
        1 / 8 + nu / 8,
        -1 / 4 - nu / 12,
        -1 / 8 + 3 * nu / 8,
        -1 / 4 + nu / 12,
        -1 / 8 - nu / 8,
        nu / 6,
        1 / 8 - 3 * nu / 8,
    ])
    # Index pattern of the symmetric matrix in terms of the 8 coefficients.
    pattern = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 0, 7, 6, 5, 4, 3, 2],
        [2, 7, 0, 5, 6, 3, 4, 1],
        [3, 6, 5, 0, 7, 2, 1, 4],
        [4, 5, 6, 7, 0, 1, 2, 3],
        [5, 4, 3, 2, 1, 0, 7, 6],
        [6, 3, 4, 1, 2, 7, 0, 5],
        [7, 2, 1, 4, 3, 6, 5, 0],
    ])
    return k[pattern] / (1.0 - nu ** 2)


def lk_gauss(nu: float = 0.3) -> np.ndarray:
    """8x8 element stiffness matrix (E = 1) by 2x2 Gauss quadrature."""
    D = np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0],
    ]) / (1.0 - nu ** 2)

    # Parent-domain corner signs of nodes 1..4 (counter-clockwise from lower-left).
    xi_n = np.array([-1.0, 1.0, 1.0, -1.0])
    eta_n = np.array([-1.0, -1.0, 1.0, 1.0])

    # Unit square mapped from [-1, 1]^2: dx/dxi = 1/2, so d/dx = 2 d/dxi, det J = 1/4.
    det_j = 0.25
    g = 1.0 / np.sqrt(3.0)

    KE = np.zeros((8, 8))
    for xi in (-g, g):
        for eta in (-g, g):
            dN_dx = 2.0 * 0.25 * xi_n * (1.0 + eta_n * eta)
            dN_dy = 2.0 * 0.25 * eta_n * (1.0 + xi_n * xi)
            B = np.zeros((3, 8))
            B[0, 0::2] = dN_dx
            B[1, 1::2] = dN_dy
            B[2, 0::2] = dN_dy
            B[2, 1::2] = dN_dx
            KE += B.T @ D @ B * det_j  # Gauss weights are all 1
    return KE


def lk(nu: float = 0.3, method: Literal["closed_form", "gauss"] = "closed_form") -> np.ndarray:
    """Element stiffness matrix of a unit-square Q4 plane-stress element (E = 1)."""
    if method == "closed_form":
        return lk_closed_form(nu)
    if method == "gauss":
        return lk_gauss(nu)
    raise ValueError(f"Unknown method '{method}'. Choose 'closed_form' or 'gauss'.")
