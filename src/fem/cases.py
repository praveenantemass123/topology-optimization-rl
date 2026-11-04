"""
fem/cases.py
============
Boundary condition definitions, benchmark load cases (MBB, Cantilever, L-bracket),
and reproducible random load distribution samplers.

Conforms to SPEC.md Section 0 & Section 4.
"""

from typing import Dict, Any, List, Optional
import json
import numpy as np
from fem.mesh import node_from_xy


def mbb_case(nelx: int, nely: int, volfrac: float = 0.5) -> Dict[str, Any]:
    """
    Half MBB Beam Benchmark.
    
    Domain:
        Symmetric half of a simply-supported beam loaded centrally.
    Boundary Conditions:
        - Left vertical edge: Symmetry boundary condition (roller fixing x-displacement).
        - Bottom-right corner node: Pinned vertically (roller fixing y-displacement).
    Load:
        - Downward unit force on top-left corner node (node 0, y DOF, F = -1.0).
    """
    ndof = 2 * (nelx + 1) * (nely + 1)
    
    # 1. Fixed DOFs
    # All x DOFs of left-edge nodes (ix = 0, iy in 0..nely)
    left_nodes = [node_from_xy(0, iy, nely) for iy in range(nely + 1)]
    fixed_left_x = [2 * n for n in left_nodes]
    
    # Bottom-right corner node (ix = nelx, iy = nely) y DOF
    bottom_right_node = node_from_xy(nelx, nely, nely)
    fixed_bottom_right_y = [2 * bottom_right_node + 1]
    
    fixed_dofs = np.unique(np.array(fixed_left_x + fixed_bottom_right_y, dtype=np.int64))
    
    # 2. Force Vector
    # Downward force at top-left node (ix=0, iy=0 => node 0, y DOF = 1)
    F = np.zeros(ndof, dtype=np.float64)
    load_node = node_from_xy(0, 0, nely)
    F[2 * load_node + 1] = -1.0  # Negative y = downward
    
    return {
        "nelx": nelx,
        "nely": nely,
        "volfrac": volfrac,
        "fixed_dofs": fixed_dofs,
        "F": F,
        "name": "mbb",
        "load_info": {
            "node": load_node,
            "angle_deg": -90.0,
            "magnitude": 1.0,
            "load_pos": "top_left"
        }
    }


def cantilever_case(
    nelx: int,
    nely: int,
    volfrac: float = 0.4,
    load_pos: str = "mid_right"
) -> Dict[str, Any]:
    """
    Cantilever Beam Benchmark.
    
    Domain:
        Solid rectangular beam clamped on the left wall, loaded on the right edge.
    Boundary Conditions:
        - Clamped along the entire left vertical edge (both x and y DOFs fixed).
    Load:
        - Downward unit force on chosen right-edge node (mid_right, bottom_right, or top_right).
    """
    ndof = 2 * (nelx + 1) * (nely + 1)
    
    # 1. Fixed DOFs (All DOFs of left-edge nodes ix=0)
    left_nodes = [node_from_xy(0, iy, nely) for iy in range(nely + 1)]
    fixed_dofs = []
    for n in left_nodes:
        fixed_dofs.extend([2 * n, 2 * n + 1])
    fixed_dofs = np.unique(np.array(fixed_dofs, dtype=np.int64))
    
    # 2. Load Node Selection on right edge (ix = nelx)
    if load_pos == "mid_right":
        load_iy = nely // 2
    elif load_pos == "bottom_right":
        load_iy = nely
    elif load_pos == "top_right":
        load_iy = 0
    else:
        raise ValueError(f"Unknown load_pos: '{load_pos}'. Choose 'mid_right', 'bottom_right', or 'top_right'.")
        
    load_node = node_from_xy(nelx, load_iy, nely)
    F = np.zeros(ndof, dtype=np.float64)
    F[2 * load_node + 1] = -1.0  # Downward force
    
    return {
        "nelx": nelx,
        "nely": nely,
        "volfrac": volfrac,
        "fixed_dofs": fixed_dofs,
        "F": F,
        "name": f"cantilever_{load_pos}",
        "load_info": {
            "node": load_node,
            "angle_deg": -90.0,
            "magnitude": 1.0,
            "load_pos": load_pos
        }
    }


def sample_load_case(
    base_case: Dict[str, Any],
    rng: np.random.Generator,
    edge: str = "right",
    angle_range_deg: tuple = (-90.0, -90.0),
    magnitude: float = 1.0
) -> Dict[str, Any]:
    """
    Samples a randomized load position and angle along a specified edge,
    retaining the boundary conditions and volume budget of base_case.
    """
    case = {
        "nelx": base_case["nelx"],
        "nely": base_case["nely"],
        "volfrac": base_case["volfrac"],
        "fixed_dofs": base_case["fixed_dofs"].copy(),
        "name": f"{base_case['name']}_random",
    }
    
    nelx = case["nelx"]
    nely = case["nely"]
    ndof = 2 * (nelx + 1) * (nely + 1)
    
    # 1. Pick random node along specified edge
    if edge == "right":
        candidate_nodes = [node_from_xy(nelx, iy, nely) for iy in range(nely + 1)]
    elif edge == "top":
        candidate_nodes = [node_from_xy(ix, 0, nely) for ix in range(nelx + 1)]
    elif edge == "bottom":
        candidate_nodes = [node_from_xy(ix, nely, nely) for ix in range(nelx + 1)]
    else:
        raise ValueError(f"Unknown edge: '{edge}'")
        
    chosen_node = int(rng.choice(candidate_nodes))
    
    # 2. Pick random angle in range
    angle_deg = float(rng.uniform(angle_range_deg[0], angle_range_deg[1]))
    angle_rad = np.radians(angle_deg)
    
    # 3. Construct force vector
    F = np.zeros(ndof, dtype=np.float64)
    # dof_x = 2n (horizontal), dof_y = 2n+1 (vertical, positive points up)
    fx = magnitude * np.cos(angle_rad)
    fy = magnitude * np.sin(angle_rad)
    
    F[2 * chosen_node] = fx
    F[2 * chosen_node + 1] = fy
    case["F"] = F
    
    case["load_info"] = {
        "node": chosen_node,
        "theta": angle_deg,
        "magnitude": magnitude,
        "edge": edge
    }
    return case


def make_test_set(
    base_case: Dict[str, Any],
    n: int,
    seed: int = 42,
    in_range: tuple = (-120.0, -60.0),
    out_range: Optional[tuple] = (-180.0, 0.0)
) -> List[Dict[str, Any]]:
    """Generates a standardized evaluation dataset of randomized load cases."""
    rng = np.random.default_rng(seed)
    test_set = []
    
    n_in = n if out_range is None else n // 2
    n_out = 0 if out_range is None else n - n_in
    
    for i in range(n_in):
        c = sample_load_case(base_case, rng, edge="right", angle_range_deg=in_range)
        c["name"] = f"test_in_dist_{i}"
        test_set.append(c)
        
    for j in range(n_out):
        c = sample_load_case(base_case, rng, edge="right", angle_range_deg=out_range)
        c["name"] = f"test_out_dist_{j}"
        test_set.append(c)
        
    return test_set
