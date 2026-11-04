"""
plot_hierarchy.py
=================
Generates the 5-panel benchmark visualization:
1. Uniform Baseline (C = 1826.10)
2. Random Agent (C = 2395.38)
3. Trained PPO Agent (Gate 3, C = 906.17)
4. Canonical Sensitivity Heuristic (C = 609.84)
5. Classical SIMP Ground Truth (C = 399.27)
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

    # 3. Trained PPO (Seed 44)
    model = PPO.load('results/checkpoints/ppo_gate3_seed44.zip')
    env_p = FlatObsWrapper(config={'nelx': 30, 'nely': 10, 'volfrac': 0.4, 'max_steps': 30, 'project_volume': True}, seed=44)
    obs, info_p = env_p.reset(seed=44)
    env_p.action_space.seed(44)
    for _ in range(30):
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info_p = env_p.step(a)
        if term or trunc:
            break
    x_ppo = env_p.x_phys
    c_ppo = info_p['compliance']

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

    # Plot 5-panel layout
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))
    plot_density(x_uniform, title='1. Uniform Baseline\nC = 1826.1', ax=axes[0], case=case)
    plot_density(x_rand, title=f'2. Random Exploration\nC = {c_rand:.1f}', ax=axes[1], case=case)
    plot_density(x_ppo, title=f'3. Trained PPO (Gate 3)\nC = {c_ppo:.1f}', ax=axes[2], case=case)
    plot_density(x_heur, title=f'4. Sensitivity Heuristic\nC = {c_heur:.1f}', ax=axes[3], case=case)
    plot_density(res_simp['x_phys'], title=f'5. SIMP Ground Truth\nC = {res_simp["compliance"]:.1f}', ax=axes[4], case=case)

    plt.suptitle('Benchmark Hierarchy on Cantilever 30x10 (Emin = 1e-9)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_img = 'rl_trained_agent_result.png'
    plt.savefig(out_img, dpi=200)
    plt.close()
    print(f'[Done] Saved 5-panel comparison to {out_img}')


if __name__ == '__main__':
    main()
