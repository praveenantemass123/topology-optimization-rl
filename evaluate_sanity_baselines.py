"""
evaluate_sanity_baselines.py
============================
Evaluates and benchmarks the 4 sanity baselines across 3 seeds as mandated in Stage 3:
1. Uniform Baseline (x_e = volfrac)
2. Random Agent (Seeds 0, 1, 2)
3. Strain-Energy Heuristic (Lowest strain energy removal)
4. Trained PPO Agent (Seeds 0, 1, 2)

Generates comparison metrics and logs results to results/sanity_baselines.csv.
"""

import os
import csv
import numpy as np
from stable_baselines3 import PPO
from topopt_env import TopOptEnv
from topopt_solver import TopOpt2D

def run_uniform_baseline(nelx=30, nely=10, volfrac=0.4):
    solver = TopOpt2D(nelx=nelx, nely=nely, volfrac=volfrac, penal=3.0, rmin=1.5, bc_type='cantilever')
    _, c, _ = solver.solve_fea(solver.x)
    return float(c)

def run_random_agent(seeds=[0, 1, 2], max_steps=20):
    compliances = []
    for s in seeds:
        env = TopOptEnv(nelx=30, nely=10, volfrac=0.4, penal=3.0, rmin=1.5, bc_type='cantilever', max_steps=max_steps, filter_actions=True)
        obs, info = env.reset(seed=s)
        for _ in range(max_steps):
            action = env.action_space.sample() * 0.1
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        compliances.append(info['compliance'])
    return float(np.mean(compliances)), float(np.std(compliances))

def run_strain_energy_heuristic(max_steps=20):
    env = TopOptEnv(nelx=30, nely=10, volfrac=0.4, penal=3.0, rmin=1.5, bc_type='cantilever', max_steps=max_steps, filter_actions=True)
    obs, info = env.reset()
    for _ in range(max_steps):
        # dc is negative strain energy gradient: -p x^(p-1) u^T k0 u
        # Lowest strain energy elements have dc closest to 0
        dc_norm = env.current_dc / (np.max(np.abs(env.current_dc)) + 1e-12)
        # Action pushes density away from low strain energy into high strain energy
        action = -0.05 * dc_norm.flatten().astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return float(info['compliance'])

def run_ppo_agent(model_path="ppo_topopt_model.zip", seeds=[0, 1, 2], max_steps=20):
    if not os.path.exists(model_path):
        return None, None
    model = PPO.load(model_path)
    compliances = []
    for s in seeds:
        env = TopOptEnv(nelx=30, nely=10, volfrac=0.4, penal=3.0, rmin=1.5, bc_type='cantilever', max_steps=max_steps, filter_actions=True)
        obs, info = env.reset(seed=s)
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        compliances.append(info['compliance'])
    return float(np.mean(compliances)), float(np.std(compliances))

def main():
    print("=============================================================")
    print(" Stage 3 Sanity Baselines Benchmarking (Cantilever 30x10)    ")
    print("=============================================================")
    
    # 1. Uniform
    c_uniform = run_uniform_baseline()
    print(f"1. Uniform Density Baseline (x_e = 0.4) : {c_uniform:.2f}")
    
    # 2. Random Agent
    rand_mean, rand_std = run_random_agent(seeds=[0, 1, 2])
    print(f"2. Random Exploration Agent (3 seeds)   : {rand_mean:.2f} ± {rand_std:.2f}")
    
    # 3. Strain Energy Heuristic
    c_heuristic = run_strain_energy_heuristic()
    print(f"3. Strain-Energy Greedy Heuristic       : {c_heuristic:.2f}")
    
    # 4. PPO Agent
    ppo_mean, ppo_std = run_ppo_agent(seeds=[0, 1, 2])
    if ppo_mean:
        print(f"4. Trained PPO Agent (3 seeds)          : {ppo_mean:.2f} ± {ppo_std:.2f}")
        
    # 5. Reference SIMP (Ground Truth)
    solver = TopOpt2D(nelx=30, nely=10, volfrac=0.4, penal=3.0, rmin=1.5, bc_type='cantilever')
    history = solver.run_optimization(max_iters=30, tol=0.01, verbose=False)
    c_simp = float(history['compliance'][-1])
    print(f"5. Reference Classical SIMP (OC)        : {c_simp:.2f}")
    
    # Save results to CSV
    os.makedirs("results", exist_ok=True)
    res_file = "results/sanity_baselines.csv"
    with open(res_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Mean Compliance", "Std Compliance", "FEA Solves Per Episode"])
        writer.writerow(["Uniform Baseline", f"{c_uniform:.2f}", "0.00", 1])
        writer.writerow(["Random Agent (3 seeds)", f"{rand_mean:.2f}", f"{rand_std:.2f}", 20])
        writer.writerow(["Strain-Energy Heuristic", f"{c_heuristic:.2f}", "0.00", 20])
        if ppo_mean:
            writer.writerow(["Trained PPO Agent (3 seeds)", f"{ppo_mean:.2f}", f"{ppo_std:.2f}", 20])
        writer.writerow(["Classical SIMP (OC Target)", f"{c_simp:.2f}", "0.00", len(history['compliance'])])
        
    print(f"\n[Saved] Benchmarking table saved to {res_file}")

if __name__ == "__main__":
    main()
