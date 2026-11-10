"""
viz.py
======
Plot helpers for density fields.
"""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_density(
    x: np.ndarray,
    title: str = "Density Map",
    ax: Optional[plt.Axes] = None,
    case: Optional[Dict[str, Any]] = None,
) -> plt.Axes:
    """
    Draw an element density field: 0 = void (white), 1 = solid (black).
    If a case is given, the load is drawn as a red arrow whose head is at the
    loaded node (row 0 of the array is the top of the structure).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))
    nely, nelx = x.shape
    # Element (iy, ix) covers [ix, ix+1] x [iy, iy+1]; nodes sit on integer coordinates.
    ax.imshow(1.0 - x, cmap="gray", vmin=0.0, vmax=1.0, aspect="equal",
              interpolation="none", extent=(0, nelx, nely, 0))
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("X elements")
    ax.set_ylabel("Y elements")

    if case and "load_info" in case:
        info = case["load_info"]
        node = int(info["node"])
        ix, iy = divmod(node, case["nely"] + 1)
        theta = np.radians(info.get("angle_deg", info.get("theta", -90.0)))
        length = 0.2 * nely
        dx, dy = length * np.cos(theta), -length * np.sin(theta)   # image y axis points down
        ax.annotate("", xy=(ix, iy), xytext=(ix - dx, iy - dy), annotation_clip=False,
                    arrowprops=dict(color="red", arrowstyle="->", lw=2.0))
    return ax


def plot_comparison(
    case: Dict[str, Any],
    x_simp: np.ndarray,
    x_rl: np.ndarray,
    c_simp: float,
    c_rl: float,
    save_path: Optional[str] = None,
):
    """Side-by-side figure: classical SIMP against an RL design."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_density(x_simp, title=f"Classical SIMP (OC)\nCompliance: {c_simp:.2f}", ax=axes[0], case=case)
    plot_density(x_rl, ax=axes[1], case=case,
                 title=f"Reinforcement Learning (PPO)\nCompliance: {c_rl:.2f} (Ratio: {c_rl / c_simp:.2f}x)")
    fig.suptitle(f"Topology Optimization Benchmark: {case.get('name', 'Case')}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    plt.close(fig)
