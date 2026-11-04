"""
plot_hierarchy.py
=================
Generates the 6-panel benchmark visualization showing the complete hierarchy:
1. Uniform Baseline (C = 1826.1)
2. Random Exploration (C = 2391.8)
3. Standard PPO Agent (Gate 3, C = 905.6)
4. Sensitivity Heuristic (C = 602.3)
5. Residual PPO Agent (Stage 4, C = 440.9)
6. Classical SIMP Ground Truth (C = 399.3)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from env.topopt_env import TopOptEnv
from fem.cases import cantilever_case
from simp.simp import run_simp
from train import FlatObsWrapper
from viz import plot_density
from fem.solver import solve


def main():
    case = cantilever_case(30, 10, 0.4)
    res_simp = run_simp(case, max_iter=200, tol=0.005)

    # 1. Uniform
    x_uniform = np.full((10, 30), 0.4)

    # 2. Random (seed 0)
    env_r = TopOptEnv(config={'nelx': 30, 'nely': 10, 'volfrac': 0.4, 'max_steps': 30, 'project_volume': True}, seed=0)
    env_r.reset(seed=0)
    env_r.action_space.seed(0)
    for _ in range(30):
        _, _, term, trunc, info_r = env_r.step(env_r.action_space.sample())
        if term or trunc:
            break
    x_rand = env_r.x_phys
    c_rand = info_r['compliance']

    # 3. Standard PPO (Seed 44)
    model_std = PPO.load('results/checkpoints/ppo_gate3_seed44.zip')
    env_std = FlatObsWrapper(config={'nelx': 30, 'nely': 10, 'volfrac': 0.4, 'max_steps': 30, 'project_volume': True}, seed=44)
    obs, info_std = env_std.reset(seed=44)
    env_std.action_space.seed(44)
    for _ in range(30):
        a, _ = model_std.predict(obs, deterministic=True)
        obs, _, term, trunc, info_std = env_std.step(a)
        if term or trunc:
            break
    x_std = env_std.x_phys
    c_std = info_std['compliance']

    # 4. Sensitivity Heuristic
    env_h = TopOptEnv(config={'nelx': 30, 'nely': 10, 'volfrac': 0.4, 'max_steps': 30, 'project_volume': True}, seed=42)
    env_h.reset(seed=42)
    for _ in range(30):
        x_phys = env_h.x_phys
        _, _, ce, _ = solve(x_phys, env_h.current_case, env_h.fea_cache['KE'], env_h.fea_cache['edofMat'], env_h.fea_cache['iK'], env_h.fea_cache['jK'], p=3.0, E0=1.0, Emin=1e-9)
        dc = -3.0 * (x_phys ** 2.0) * ce
        dc_norm = dc / (np.max(np.abs(dc)) + 1e-12)
        action = -np.sign(dc_norm).astype(np.float32) * np.abs(dc_norm).astype(np.float32)
        _, _, term, trunc, info_h = env_h.step(action)
        if term or trunc:
            break
    x_heur = env_h.x_phys
    c_heur = info_h['compliance']

    # 5. Residual PPO (Stage 4, Seed 43)
    model_res = PPO.load('results/checkpoints/ppo_stage4_residual_seed43.zip')
    env_res = FlatObsWrapper(config={'nelx': 30, 'nely': 10, 'volfrac': 0.4, 'max_steps': 30, 'project_volume': True, 'residual': True, 'alpha': 0.2}, seed=43)
    obs, info_res = env_res.reset(seed=43)
    env_res.action_space.seed(43)
    for _ in range(30):
        a, _ = model_res.predict(obs, deterministic=True)
        obs, _, term, trunc, info_res = env_res.step(a)
        if term or trunc:
            break
    x_res = env_res.x_phys
    c_res = info_res['compliance']

    # Plot 6-panel layout
    fig, axes = plt.subplots(1, 6, figsize=(21, 3.5))
    plot_density(x_uniform, title='1. Uniform\nC = 1826.1', ax=axes[0], case=case)
    plot_density(x_rand, title=f'2. Random\nC = {c_rand:.1f}', ax=axes[1], case=case)
    plot_density(x_std, title=f'3. Standard PPO\nC = {c_std:.1f}', ax=axes[2], case=case)
    plot_density(x_heur, title=f'4. Heuristic\nC = {c_heur:.1f}', ax=axes[3], case=case)
    plot_density(x_res, title=f'5. Residual PPO\nC = {c_res:.1f}', ax=axes[4], case=case)
    plot_density(res_simp['x_phys'], title=f'6. SIMP (Ground Truth)\nC = {res_simp["compliance"]:.1f}', ax=axes[5], case=case)

    plt.suptitle('Topology Optimization Benchmark Progression on Cantilever 30x10 (Emin = 1e-9)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_img = 'rl_trained_agent_result.png'
    plt.savefig(out_img, dpi=200)
    plt.close()
    print(f'[Done] Saved 6-panel progression comparison to {out_img}')


if __name__ == '__main__':
    main()
