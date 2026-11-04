"""
fem/element.py
==============
Element stiffness matrix computation for a 4-node bilinear (Q4) unit square element
under plane stress conditions.

Conforms to SPEC.md Section 1.
"""

from typing import Literal
import numpy as np


def lk_closed_form(nu: float = 0.3) -> np.ndarray:
    """
    Computes the 8x8 element stiffness matrix using the closed-form analytical
    representation from the 88-line topology optimization MATLAB code (Andreassen et al., 2011).
    
    Parameters:
        nu: Poisson's ratio (float). Default = 0.3.
        
    Returns:
        KE: 8x8 symmetric element stiffness matrix for E = 1.0.
    """
    E = 1.0
    k = np.array([
        1/2 - nu/6,
        1/8 + nu/8,
        -1/4 - nu/12,
        -1/8 + 3*nu/8,
        -1/4 + nu/12,
        -1/8 - nu/8,
        nu/6,
        1/8 - 3*nu/8
    ], dtype=np.float64)
    
    KE = (E / (1.0 - nu**2)) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
    ], dtype=np.float64)
    return KE


def lk_gauss(nu: float = 0.3) -> np.ndarray:
    """
    Computes the 8x8 element stiffness matrix using 2x2 numerical Gauss quadrature
    as a rigorous reference verification.
    
    Parameters:
        nu: Poisson's ratio (float). Default = 0.3.
        
    Returns:
        KE: 8x8 symmetric element stiffness matrix for E = 1.0.
    """
    E = 1.0
    D = (E / (1.0 - nu**2)) * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0]
    ], dtype=np.float64)
    
    # 2x2 Gauss integration points and weights for standard parent domain [-1, 1]^2
    gp = 1.0 / np.sqrt(3.0)
    gauss_points = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
    
    # Unit square physical domain [0, 1] x [0, 1] => det(J) = 1/4
    # Shape function derivatives dN/dx = 2 * dN/dxi, dN/dy = 2 * dN/deta
    detJ = 0.25
    weight = 1.0
    
    KE = np.zeros((8, 8), dtype=np.float64)
    
    for xi, eta in gauss_points:
        dNdxi = np.array([
            -0.25 * (1.0 - eta),
             0.25 * (1.0 - eta),
             0.25 * (1.0 + eta),
            -0.25 * (1.0 + eta)
        ], dtype=np.float64)
        
        dNdeta = np.array([
            -0.25 * (1.0 - xi),
            -0.25 * (1.0 + xi),
             0.25 * (1.0 + xi),
             0.25 * (1.0 - xi)
        ], dtype=np.float64)
        
        dNdx = 2.0 * dNdxi
        dNdy = 2.0 * dNdeta
        
        B = np.zeros((3, 8), dtype=np.float64)
        for i in range(4):
            B[0, 2*i] = dNdx[i]
            B[1, 2*i + 1] = dNdy[i]
            B[2, 2*i] = dNdy[i]
            B[2, 2*i + 1] = dNdx[i]
            
        KE += (B.T @ D @ B) * detJ * weight
        
    return KE


def lk(nu: float = 0.3, method: Literal["closed_form", "gauss"] = "closed_form") -> np.ndarray:
    """
    Standard interface for element stiffness matrix of a unit square Q4 plane-stress element.
    """
    if method == "closed_form":
        return lk_closed_form(nu)
    elif method == "gauss":
        return lk_gauss(nu)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'closed_form' or 'gauss'.")
