"""
evaluate_baselines.py
=====================
Audited sanity baselines evaluation conforming strictly to BTP specifications and Claude's audit:
- Evaluates Uniform Baseline
- Evaluates Random Agent across 3 seeded runs (properly seeding action_space and env)
- Evaluates Trained PPO Agent across 3 separately trained checkpoints (seeds 42, 43, 44)
- Evaluates Canonical Sensitivity Heuristic
- Evaluates Classical SIMP at the EXACT SAME FEA budgets (20 and 30 solves)
- Evaluates Converged Classical SIMP (95 solves)
- Automatically writes results/sanity_baselines.csv from code (never hand-edited)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import csv
import numpy as np
from stable_baselines3 import PPO

from env.topopt_env import TopOptEnv
from fem.cases import cantilever_case
from fem.element import lk
from fem.mesh import build_edofmat, build_sparse_indices
from fem.solver import solve
from simp.simp import run_simp
from train import FlatObsWrapper


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
    config = {
        "nelx": 30, "nely": 10, "volfrac": 0.4, "max_steps": max_steps,
        "project_volume": True, "delta_max": 0.05, "Emin": 1e-9
    }
    
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


def run_ppo_agents(seeds=[42, 43, 44], max_steps=30):
    compliances = []
    config = {
        "nelx": 30, "nely": 10, "volfrac": 0.4, "max_steps": max_steps,
        "project_volume": True, "delta_max": 0.05, "Emin": 1e-9, "reward_clip": 2.0
    }
    
    for s in seeds:
        ckpt_path = f"results/checkpoints/ppo_gate3_seed{s}.zip"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Run src/train.py first.")
            
        model = PPO.load(ckpt_path)
        test_env = FlatObsWrapper(config=config, seed=s)
        obs, info = test_env.reset(seed=s)
        test_env.action_space.seed(s)
        
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            if terminated or truncated:
                break
        compliances.append(info["compliance"])
        
    return float(np.mean(compliances)), float(np.std(compliances)), compliances


def run_canonical_heuristic(max_steps=30):
    """
    Canonical sensitivity-guided heuristic:
    Step delta proportional to -dc/max|dc|, projected to volume fraction.
    """
    config = {
        "nelx": 30, "nely": 10, "volfrac": 0.4, "max_steps": max_steps,
        "project_volume": True, "delta_max": 0.05, "Emin": 1e-9
    }
    env = TopOptEnv(config=config, seed=42)
    obs, info = env.reset(seed=42)
    
    for _ in range(max_steps):
        x_phys = env.x_phys
        _, _, ce, _ = solve(
            x_phys, env.current_case, env.fea_cache["KE"],
            env.fea_cache["edofMat"], env.fea_cache["iK"], env.fea_cache["jK"],
            p=3.0, E0=1.0, Emin=1e-9
        )
        dc = -3.0 * (x_phys ** 2.0) * ce
        dc_norm = dc / (np.max(np.abs(dc)) + 1e-12)
        
        action = -np.sign(dc_norm).astype(np.float32) * np.abs(dc_norm).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
            
    return float(info["compliance"])


def run_simp_at_budgets(nelx=30, nely=10, volfrac=0.4):
    case = cantilever_case(nelx=nelx, nely=nely, volfrac=volfrac)
    
    # 1. SIMP @ 20 iterations
    res_20 = run_simp(case, max_iter=20, tol=1e-6)
    c_20 = float(res_20["compliance"])
    
    # 2. SIMP @ 30 iterations
    res_30 = run_simp(case, max_iter=30, tol=1e-6)
    c_30 = float(res_30["compliance"])
    
    # 3. Converged Classical SIMP
    res_conv = run_simp(case, max_iter=250, tol=0.005)
    c_conv = float(res_conv["compliance"])
    n_conv = int(res_conv["n_fea"])
    
    return c_20, c_30, c_conv, n_conv


def main():
    print("======================================================================")
    print(" Audited Sanity Baselines Benchmark (Cantilever 30x10, Emin=1e-9)     ")
    print(" Strict Script Regeneration - Zero Hand-Editing                        ")
    print("======================================================================")
    
    c_uniform = run_uniform_baseline()
    print(f"1. Uniform Density Baseline (x_e = 0.4)        : {c_uniform:.2f}")
    
    rand_mean, rand_std = run_random_agent(seeds=[0, 1, 2], max_steps=30)
    print(f"2. Random Agent (3 seeds, delta_max=0.05)      : {rand_mean:.2f} ± {rand_std:.2f}")
    
    ppo_mean, ppo_std, ppo_ind = run_ppo_agents(seeds=[42, 43, 44], max_steps=30)
    print(f"3. Trained PPO Agent (3 separately trained seeds): {ppo_mean:.2f} ± {ppo_std:.2f} (seeds: {ppo_ind})")
    
    c_heuristic = run_canonical_heuristic(max_steps=30)
    print(f"4. Canonical Sensitivity Heuristic (30 steps)  : {c_heuristic:.2f}")
    
    c_simp_20, c_simp_30, c_simp_conv, n_conv = run_simp_at_budgets()
    print(f"5. Classical SIMP @ 20 iterations (Equal Budget): {c_simp_20:.2f}")
    print(f"6. Classical SIMP @ 30 iterations (Equal Budget): {c_simp_30:.2f}")
    print(f"7. Converged Classical SIMP (tol=0.005)        : {c_simp_conv:.2f} (in {n_conv} iters)")
    
    os.makedirs("results", exist_ok=True)
    res_path = "results/sanity_baselines.csv"
    with open(res_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Mean Compliance", "Std Compliance", "FEA Solves", "Description"])
        writer.writerow(["Uniform Baseline", f"{c_uniform:.2f}", "0.00", 1, "Homogeneous x_e = 0.40"])
        writer.writerow(["Random Agent (3 seeds)", f"{rand_mean:.2f}", f"{rand_std:.2f}", 30, "Seeded uniform random noise in [-1, 1] scaled by delta_max"])
        writer.writerow(["Trained PPO Agent (Gate 3)", f"{ppo_mean:.2f}", f"{ppo_std:.2f}", 30, "3 separately trained PPO models (seeds 42, 43, 44)"])
        writer.writerow(["Canonical Sensitivity Heuristic", f"{c_heuristic:.2f}", "0.00", 30, "Greedy sensitivity step -dc/|max dc| with volume projection"])
        writer.writerow(["Classical SIMP @ 20 iters", f"{c_simp_20:.2f}", "0.00", 20, "SIMP OC with sensitivity filter at equal FEA budget 20"])
        writer.writerow(["Classical SIMP @ 30 iters", f"{c_simp_30:.2f}", "0.00", 30, "SIMP OC with sensitivity filter at equal FEA budget 30"])
        writer.writerow(["Converged Classical SIMP", f"{c_simp_conv:.2f}", "0.00", n_conv, "SIMP OC converged to tol=0.005 (ground truth)"])
        
    print(f"\n[Saved] Audited sanity baselines regenerated strictly by script to {res_path}")


if __name__ == "__main__":
    main()
