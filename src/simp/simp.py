"""
simp/simp.py
============
Classical SIMP minimum-compliance topology optimization with the Optimality
Criteria (OC) update (Sigmund 2001; Andreassen et al. 2011).

    min_x  c(x) = F^T U(x)      s.t.  mean(x_phys) = volfrac,  xmin <= x <= 1

Bookkeeping used throughout this project:
    * one "iteration" = one FEA solve + one OC design update;
    * history_c[k] is the compliance of the design BEFORE update k+1
      (history_c[0] is the uniform starting design);
    * the returned "compliance" is always a fresh FEA of the returned design, so
      it corresponds to the design after `n_updates` updates and costs
      n_fea = n_updates + 1 solves in total.
"""

import time
from typing import Any, Dict

import numpy as np

from fem.element import lk
from fem.mesh import build_edofmat, build_sparse_indices
from fem.solver import compliance_sensitivity, solve
from simp.filters import build_filter, chain_rule_density, density_filter, sensitivity_filter

XMIN = 1e-3  # lower bound of the design variable


def oc_update(
    x: np.ndarray,
    dc: np.ndarray,
    dv: np.ndarray,
    volfrac: float,
    move: float,
    to_physical,
    eta: float = 0.5,
    max_bisect: int = 200,
    vol_tol: float = 1e-9,
) -> np.ndarray:
    """
    One OC update. The Lagrange multiplier of the volume constraint is found by
    bisection on a geometric scale until mean(x_phys) matches volfrac to vol_tol
    (or the bracket is exhausted).

    dc must be <= 0 and dv > 0 (derivatives w.r.t. the design variable x).
    to_physical maps a candidate design to its physical density.
    """
    lower = np.maximum(XMIN, x - move)
    upper = np.minimum(1.0, x + move)
    scale = np.maximum(-dc, 0.0) / dv           # B_e * lambda

    def candidate(lam: float) -> np.ndarray:
        return np.clip(x * (scale / lam) ** eta, lower, upper)

    lo, hi = 1e-12, 1e12
    x_new = candidate(np.sqrt(lo * hi))
    for _ in range(max_bisect):
        lam = np.sqrt(lo * hi)
        x_new = candidate(lam)
        err = float(np.mean(to_physical(x_new))) - volfrac
        if abs(err) <= vol_tol:
            break
        if err > 0.0:
            lo = lam        # too much material -> larger multiplier
        else:
            hi = lam
        if hi / lo < 1.0 + 1e-14:
            break
    return x_new


def run_simp(
    case: Dict[str, Any],
    rmin: float = 1.5,
    p: float = 3.0,
    filter_type: str = "sensitivity",
    max_iter: int = 200,
    tol: float = 0.01,
    move: float = 0.2,
    return_history: bool = True,
    E0: float = 1.0,
    Emin: float = 1e-9,
) -> Dict[str, Any]:
    """
    Run SIMP + OC on a case.

    Parameters
    ----------
    case        : case dict (nelx, nely, volfrac, fixed_dofs, F).
    rmin        : filter radius in element widths.
    p           : SIMP penalisation exponent.
    filter_type : "sensitivity" (99-line) or "density" (88-line).
    max_iter    : maximum number of design updates.
    tol         : stop when max|x_new - x_old| < tol.
    move        : OC move limit.

    Returns
    -------
    dict with
        x_phys      final physical density, shape (nely, nelx)
        compliance  compliance of x_phys from a fresh FEA
        volume      mean(x_phys)
        history_c   compliance before each update, followed by the final value
                    (length n_updates + 1)
        n_updates   number of OC updates performed
        n_fea       number of FEA solves, n_updates + 1
        change      last max|dx|
        converged   whether change < tol was reached
        time_s      wall time
    """
    if filter_type not in ("sensitivity", "density"):
        raise ValueError(f"filter_type must be 'sensitivity' or 'density', got {filter_type!r}")
    nelx, nely = int(case["nelx"]), int(case["nely"])
    volfrac = float(case["volfrac"])

    KE = lk(nu=0.3)
    edofMat = build_edofmat(nelx, nely)
    iK, jK = build_sparse_indices(edofMat)
    H, Hs = build_filter(nelx, nely, rmin)
    use_density = filter_type == "density"

    def to_physical(x: np.ndarray) -> np.ndarray:
        return density_filter(x, H, Hs) if use_density else x

    def fea(x_phys: np.ndarray):
        _, c, ce, _ = solve(x_phys, case, KE, edofMat, iK, jK, p=p, E0=E0, Emin=Emin)
        return c, ce

    x = np.full((nely, nelx), volfrac)
    x_phys = to_physical(x)
    ones = np.ones((nely, nelx))

    history_c = []
    converged = False
    change = np.inf
    n_updates = 0
    t0 = time.perf_counter()

    c, ce = fea(x_phys)
    for _ in range(max_iter):
        history_c.append(c)

        dc = compliance_sensitivity(x_phys, ce, p, E0, Emin)
        if use_density:
            dc, dv = chain_rule_density(dc, ones, H, Hs)
        else:
            dc, dv = sensitivity_filter(x, dc, H, Hs), ones

        x_new = oc_update(x, dc, dv, volfrac, move, to_physical)
        change = float(np.max(np.abs(x_new - x)))
        x = x_new
        x_phys = to_physical(x)
        n_updates += 1

        c, ce = fea(x_phys)          # compliance of the updated design
        if change < tol:
            converged = True
            break
    history_c.append(c)

    return {
        "x_phys": x_phys.copy(),
        "compliance": float(c),
        "volume": float(np.mean(x_phys)),
        "history_c": history_c if return_history else [],
        "n_updates": n_updates,
        "n_fea": n_updates + 1,
        "change": change,
        "converged": converged,
        "time_s": time.perf_counter() - t0,
    }
