"""
evaluate.py
===========
Rigorous evaluation and benchmarking suite for RL Topology Optimization.
Computes compliance ratio (C_RL / C_SIMP), volume error, grayness,
and structural connectivity.

Conforms to SPEC.md Section 10.
"""

from typing import Dict, Any, List
import numpy as np
from scipy.ndimage import label

from fem.element import lk
from fem.mesh import build_edofmat, build_sparse_indices
from fem.solver import solve


def evaluate_metrics(
    x_phys: np.ndarray,
    case: Dict[str, Any],
    c_simp_target: float,
    p: float = 3.0
) -> Dict[str, Any]:
    """
    Computes rigorous metrics on a generated structural topology with a fresh FEA solve.
    """
    nelx = case["nelx"]
    nely = case["nely"]
    volfrac = case["volfrac"]
    
    # 1. Fresh independent FEA solve (do not rely on cached env values)
    KE = lk(nu=0.3)
    edofMat = build_edofmat(nelx, nely)
    iK, jK = build_sparse_indices(edofMat)
    
    U, c_actual, _, _ = solve(x_phys, case, KE, edofMat, iK, jK, p=p)
    
    # 2. Metric calculations
    compliance_ratio = float(c_actual / c_simp_target)
    
    actual_vol = float(np.mean(x_phys))
    vol_error = float(abs(actual_vol - volfrac) / volfrac)
    
    # Grayness: fraction of intermediate elements with 0.1 < x < 0.9
    grayness = float(np.mean((x_phys > 0.1) & (x_phys < 0.9)))
    
    # Connectivity of solid phase (threshold 0.5)
    solid_mask = (x_phys >= 0.5).astype(int)
    labeled_array, num_features = label(solid_mask)
    
    return {
        "compliance": round(float(c_actual), 4),
        "c_simp_target": round(float(c_simp_target), 4),
        "compliance_ratio": round(compliance_ratio, 4),
        "volume": round(actual_vol, 4),
        "volume_error": round(vol_error, 4),
        "grayness": round(grayness, 4),
        "num_connected_components": int(num_features),
        "success": bool(vol_error < 0.05 and compliance_ratio < 1.5)
    }
