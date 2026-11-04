"""
simp/simp.py
============
Classical baseline SIMP (Solid Isotropic Material with Penalization)
topology optimization solver using Optimality Criteria (OC).

Conforms to SPEC.md Section 0 & Section 6.
"""

from typing import Dict, Any
import time
import numpy as np

from fem.element import lk
from fem.mesh import build_edofmat, build_sparse_indices
from fem.solver import solve
from simp.filters import build_filter, density_filter, sensitivity_filter, chain_rule_density


def run_simp(
    case: Dict[str, Any],
    rmin: float = 1.5,
    p: float = 3.0,
    filter_type: str = "sensitivity",
    max_iter: int = 200,
    tol: float = 0.01,
    move: float = 0.2,
    return_history: bool = True
) -> Dict[str, Any]:
    """
    Executes the classical SIMP topology optimization loop on a given case.
    
    Parameters:
        case: Case dictionary (nelx, nely, volfrac, fixed_dofs, F).
        rmin: Filter radius in element units. Default = 1.5.
        p: SIMP penalization power exponent. Default = 3.0.
        filter_type: 'sensitivity' or 'density'. Default = 'sensitivity'.
        max_iter: Maximum optimization iterations. Default = 200.
        tol: Convergence tolerance on maximum density change. Default = 0.01.
        move: Maximum allowable density change per OC step. Default = 0.2.
        return_history: Whether to track iteration history. Default = True.
        
    Returns:
        dict: {"x_phys", "compliance", "history_c", "n_fea", "time_s", "converged"}
    """
    nelx = case["nelx"]
    nely = case["nely"]
    nel = nelx * nely
    volfrac = case["volfrac"]
    E0 = 1.0
    Emin = 1e-9
    
    # 1. Precompute FEA structures (cached for the entire optimization)
    KE = lk(nu=0.3)
    edofMat = build_edofmat(nelx, nely)
    iK, jK = build_sparse_indices(edofMat)
    H, Hs = build_filter(nelx, nely, rmin)
    
    # 2. Initialize design variable field x uniformly
    x = np.full((nely, nelx), volfrac, dtype=np.float64)
    x_phys = x.copy()
    
    history_c = []
    converged = False
    t0 = time.time()
    
    dv = np.ones((nely, nelx), dtype=np.float64)
    
    for it in range(1, max_iter + 1):
        x_old = x.copy()
        
        # Step 1: Compute physical density
        if filter_type == "density":
            x_phys = density_filter(x, H, Hs)
        else:
            x_phys = x.copy()
            
        # Step 2: FEA Solve
        U, c, ce, _ = solve(x_phys, case, KE, edofMat, iK, jK, p=p, E0=E0, Emin=Emin)
        history_c.append(c)
        
        # Step 3: Sensitivity Analysis
        # dc = -p * x_phys^(p-1) * (E0 - Emin) * ce
        dc = -p * (x_phys ** (p - 1.0)) * (E0 - Emin) * ce
        
        # Step 4: Spatial Filtering
        if filter_type == "density":
            dc_filtered, dv_filtered = chain_rule_density(dc, dv, H, Hs)
        else:
            dc_filtered = sensitivity_filter(x, dc, H, Hs)
            dv_filtered = dv
            
        # Step 5: Optimality Criteria (OC) Bisection Update
        l1 = 0.0
        l2 = 1e9
        
        while (l2 - l1) / (l1 + l2 + 1e-12) > 1e-3 and (l2 - l1) > 1e-4:
            lmid = 0.5 * (l1 + l2)
            
            # OC update formula
            ratio = np.maximum(-dc_filtered / (dv_filtered * lmid + 1e-12), 1e-12)
            x_step = x * np.sqrt(ratio)
            
            xnew = np.clip(
                x_step,
                np.maximum(0.001, x - move),
                np.minimum(1.0, x + move)
            )
            
            if filter_type == "density":
                x_phys_new = density_filter(xnew, H, Hs)
            else:
                x_phys_new = xnew
                
            if np.sum(x_phys_new) > volfrac * nel:
                l1 = lmid
            else:
                l2 = lmid
                
        x = xnew
        change = float(np.max(np.abs(x - x_old)))
        
        if change < tol:
            converged = True
            break
            
    runtime = time.time() - t0
    
    if filter_type == "density":
        x_phys_final = density_filter(x, H, Hs)
    else:
        x_phys_final = x.copy()
        
    return {
        "x_phys": x_phys_final,
        "compliance": history_c[-1] if history_c else c,
        "history_c": history_c,
        "n_fea": it,
        "time_s": runtime,
        "converged": converged
    }
