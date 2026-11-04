"""
viz.py
======
Visualization and plotting routines for topology optimization.

Conforms to SPEC.md Section 12.
"""

from typing import Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt


def plot_density(
    x: np.ndarray,
    title: str = "Density Map",
    ax: Optional[plt.Axes] = None,
    case: Optional[Dict[str, Any]] = None
) -> plt.Axes:
    """
    Renders element density: 0 = void (white), 1 = solid (black).
    Optionally overlays load arrows and support anchorage markers.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        
    ax.imshow(1.0 - x, cmap="gray", vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("X elements")
    ax.set_ylabel("Y elements")
    
    # Overlay load arrow if case provided
    if case and "load_info" in case:
        node = case["load_info"]["node"]
        nely = case["nely"]
        ix = node // (nely + 1)
        iy = node % (nely + 1)
        # Force arrow
        theta = np.radians(case["load_info"].get("theta", -90.0))
        dx = 2.0 * np.cos(theta)
        dy = -2.0 * np.sin(theta)  # Inverted y for image coordinates
        ax.annotate(
            "",
            xy=(ix, iy),
            xytext=(ix - dx, iy - dy),
            arrowprops=dict(facecolor="red", edgecolor="darkred", arrowstyle="->", lw=2.5)
        )
        
    return ax


def plot_comparison(
    case: Dict[str, Any],
    x_simp: np.ndarray,
    x_rl: np.ndarray,
    c_simp: float,
    c_rl: float,
    save_path: Optional[str] = None
):
    """Generates side-by-side comparison figure: Classical SIMP vs Trained RL Agent."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    plot_density(x_simp, title=f"Classical SIMP (OC)\nCompliance: {c_simp:.2f}", ax=axes[0], case=case)
    plot_density(x_rl, title=f"Reinforcement Learning (PPO)\nCompliance: {c_rl:.2f} (Ratio: {c_rl/c_simp:.2f}x)", ax=axes[1], case=case)
    
    plt.suptitle(f"Topology Optimization Benchmark: {case.get('name', 'Case')}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"[Saved] Comparison plot saved to {save_path}")
    plt.close()
