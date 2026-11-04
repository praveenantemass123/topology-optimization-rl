"""
evaluate_baselines.py
=====================
Audited sanity baselines evaluation conforming to SPEC.md and Claude's audit:
- Proper seeding of both env.reset(seed=s) and env.action_space.seed(s)
- Canonical strain-energy sensitivity heuristic
- Random agent taking normalized [-1, 1] actions scaled by delta_max inside env
- Uses src/ modules with Emin = 1e-9 and strict bisection volume projection
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import csv
import numpy as np

from env.topopt_env import TopOptEnv
from fem.cases import cantilever_case
from fem.element import lk
from fem.mesh import build_edofmat, build_sparse_indices
from fem.solver import solve, strain_energy_density


def run_uniform_baseline(nelx=30, nely=10, volfrac=0.4):
    case = cantilever_case(nelx=nelx, nely=nely, volfrac=volfrac)
    KE = lk(nu=0.3)
    edofMat = build_edofmat(nelx, nely)
    iK, jK = build_sparse_indices(edofMat)
    x = np.full((nely, nelx), volfrac, dtype=np.float64)
    _, c, _, _ = solve(x, case, KE, edofMat, iK, jK, p=3.0, E0=1.0, Emin=1e-9)
    return float(c)


def run_random_agent(seeds=[0, 1, 2], max_steps=30):
    compliances = []
    config = {"nelx": 30, "nely": 10, "volfrac": 0.4, "max_steps": max_steps, "project_volume": True}
    
    for s in seeds:
        env = TopOptEnv(config=config, seed=s)
        obs, info = env.reset(seed=s)
        env.action_space.seed(s)
        
        for _ in range(max_steps):
            action = env.action_space.sample()  # shape (10, 30) in [-1, 1]
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        compliances.append(info["compliance"])
        
    return float(np.mean(compliances)), float(np.std(compliances))


def run_canonical_heuristic(max_steps=30):
    """
    Canonical sensitivity-guided heuristic:
    Step delta proportional to -dc/max|dc|, projected to volume fraction.
    """
    config = {"nelx": 30, "nely": 10, "volfrac": 0.4, "max_steps": max_steps, "project_volume": True}
    env = TopOptEnv(config=config, seed=42)
    obs, info = env.reset(seed=42)
    
    for _ in range(max_steps):
        # Solve FEA to get current sensitivities
        x_phys = env.x_phys
        _, _, ce, _ = solve(
            x_phys, env.current_case, env.fea_cache["KE"],
            env.fea_cache["edofMat"], env.fea_cache["iK"], env.fea_cache["jK"],
            p=3.0, E0=1.0, Emin=1e-9
        )
        dc = -3.0 * (x_phys ** 2.0) * ce
        dc_norm = dc / (np.max(np.abs(dc)) + 1e-12)
        
        # Action in [-1, 1] directing material towards high sensitivity
        action = -np.sign(dc_norm).astype(np.float32) * np.abs(dc_norm).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
            
    return float(info["compliance"])


def main():
    print("===============================================================")
    print(" Audited Sanity Baselines Benchmark (Cantilever 30x10, Emin=1e-9) ")
    print("===============================================================")
    
    c_uniform = run_uniform_baseline()
    print(f"1. Uniform Density Baseline (x_e = 0.4) : {c_uniform:.2f}")
    
    rand_mean, rand_std = run_random_agent(seeds=[0, 1, 2], max_steps=30)
    print(f"2. Random Agent (3 seeded runs)         : {rand_mean:.2f} ± {rand_std:.2f}")
    
    c_heuristic = run_canonical_heuristic(max_steps=30)
    print(f"3. Canonical Sensitivity Heuristic (30 steps): {c_heuristic:.2f}")
    
    c_simp = 399.38  # Converged SIMP from baselines.csv
    print(f"4. Converged Classical SIMP (Ground Truth): {c_simp:.2f}")
    
    os.makedirs("results", exist_ok=True)
    res_path = "results/sanity_baselines.csv"
    with open(res_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Mean Compliance", "Std Compliance", "FEA Solves"])
        writer.writerow(["Uniform Baseline", f"{c_uniform:.2f}", "0.00", 1])
        writer.writerow(["Random Agent (3 seeds)", f"{rand_mean:.2f}", f"{rand_std:.2f}", 30])
        writer.writerow(["Canonical Sensitivity Heuristic", f"{c_heuristic:.2f}", "0.00", 30])
        writer.writerow(["Converged Classical SIMP", f"{c_simp:.2f}", "0.00", 95])
        
    print(f"\n[Saved] Audited sanity baselines saved to {res_path}")

if __name__ == "__main__":
    main()
