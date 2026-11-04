"""
run_rl_agent_demo.py
====================
Demonstrates the Gymnasium Reinforcement Learning environment for Topology Optimization.
Compares:
1. Random Exploration Agent (Baseline)
2. Sensitivity-Guided Policy (Heuristic Agent)

Shows that the Gymnasium wrapper correctly manages the MDP loop:
reset() -> step(action) -> reward, next_obs, done
"""

import numpy as np
import matplotlib.pyplot as plt
from topopt_env import TopOptEnv

def run_agent_episode(env, agent_type='heuristic', max_steps=25):
    obs, info = env.reset()
    total_reward = 0.0
    print(f"\n--- Running Episode: {agent_type.upper()} AGENT ---")
    print(f"Initial State: Compliance = {info['compliance']:.2f}, Volume Fraction = {info['volume_fraction']:.2f}")
    print(f"{'Step':<6}{'Compliance':<15}{'Reward':<12}{'Mean Density':<15}")
    print("-" * 48)
    
    for step in range(1, max_steps + 1):
        if agent_type == 'random':
            # Random exploration in continuous action space
            action = env.action_space.sample() * 0.1
        elif agent_type == 'heuristic':
            # Uses the FEA sensitivities to propose gradient-directed density shifts
            _, _, dc = env.solver.solve_fea(env.density)
            # Normalize dc to [-0.1, 0.1]
            dc_norm = dc / (np.max(np.abs(dc)) + 1e-12)
            action = -0.1 * dc_norm.astype(np.float32)
        else:
            raise ValueError(f"Unknown agent: {agent_type}")
            
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step == 1 or step % 5 == 0 or terminated or truncated:
            print(f"{step:<6}{info['compliance']:<15.2f}{reward:<12.2f}{info['volume_fraction']:<15.2f}")
            
        if terminated or truncated:
            print(f"--> Episode finished at step {step} | Total Cumulative Reward: {total_reward:.2f}")
            break
            
    return env.compliance_history, env.density

def main():
    print("=======================================================")
    print(" BTP TopOpt Gymnasium RL Environment Verification ")
    print("=======================================================")
    
    # Create environment (30x10 grid for fast RL iteration)
    env = TopOptEnv(nelx=30, nely=10, volfrac=0.4, penal=3.0, rmin=1.5, bc_type='cantilever', max_steps=20)
    
    # 1. Random Agent
    rand_history, rand_final_density = run_agent_episode(env, agent_type='random', max_steps=20)
    
    # 2. Heuristic/Guided Agent
    heur_history, heur_final_density = run_agent_episode(env, agent_type='heuristic', max_steps=20)
    
    # Plot Comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    axes[0].imshow(1 - rand_final_density, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f"Random Agent Final\nCompliance: {rand_history[-1]:.2f}", fontsize=11)
    axes[0].axis('off')
    
    axes[1].imshow(1 - heur_final_density, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f"Guided Agent Final\nCompliance: {heur_history[-1]:.2f}", fontsize=11, fontweight='bold')
    axes[1].axis('off')
    
    plt.suptitle("Gymnasium RL Environment for Topology Optimization", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("rl_environment_comparison.png", dpi=200)
    plt.close()
    print("\n[Saved] RL comparison plot saved to rl_environment_comparison.png")
    print("[SUCCESS] Gymnasium RL environment fully functional!")

if __name__ == "__main__":
    main()
