"""
evaluate.py
===========
Metrics of a finished design, computed from an independent FEA:
compliance, ratio to a SIMP reference, volume error, gray fraction and the
number of connected solid components.
"""

from typing import Any, Dict

import numpy as np
from scipy.ndimage import label

from common import fresh_compliance, gray_fraction


def evaluate_metrics(
    x_phys: np.ndarray,
    case: Dict[str, Any],
    c_simp_target: float,
    p: float = 3.0,
) -> Dict[str, Any]:
    """
    Parameters
    ----------
    x_phys        : physical density, shape (nely, nelx).
    case          : case dict the design was made for.
    c_simp_target : reference compliance (e.g. converged SIMP) for the ratio.

    A design counts as a "success" if its volume is within 5 % of the target
    and its compliance is below 1.5 times the reference.
    """
    c = fresh_compliance(x_phys, case, p=p)
    volume = float(np.mean(x_phys))
    vol_error = abs(volume - case["volfrac"]) / case["volfrac"]
    ratio = c / c_simp_target
    _, n_components = label(x_phys >= 0.5)   # 4-connected solid regions
    return {
        "compliance": round(float(c), 4),
        "c_simp_target": round(float(c_simp_target), 4),
        "compliance_ratio": round(float(ratio), 4),
        "volume": round(volume, 4),
        "volume_error": round(float(vol_error), 4),
        "grayness": round(gray_fraction(x_phys), 4),
        "num_connected_components": int(n_components),
        "success": bool(vol_error < 0.05 and ratio < 1.5),
    }
