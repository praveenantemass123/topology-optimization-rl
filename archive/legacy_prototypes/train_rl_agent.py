"""
train_rl_agent.py
=================
Trains a Deep Reinforcement Learning Agent (PPO - Proximal Policy Optimization)
to perform 2D Topology Optimization (Compliance Minimization).

Physics-Informed RL:
- State includes both the Density Map and the Strain Energy Sensitivity Field
- Action is regularized through the Mesh Filter to enforce structural continuity
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from topopt_env import TopOptEnv
from topopt_solver import TopOpt2D


def train_agent(total_timesteps=4000):
    print("=======================================================")
    print(" BTP Deep RL Training: PPO on Topology Optimization    ")
    print("=======================================================")
    
    # 1. Setup Environment
    raw_env = TopOptEnv(
        nelx=30, nely=10, volfrac=0.4, penal=3.0, rmin=1.5,
        bc_type='cantilever', max_steps=20, filter_actions=True
    )
    env = Monitor(raw_env)
    
    # 2. Configure PPO Agent
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=64,
        batch_size=32,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        verbose=0
    )
    
    # 3. Train Policy Network
    t0 = time.time()
    print(f"\n--> Training PPO Agent for {total_timesteps} environment timesteps...")
    model.learn(total_timesteps=total_timesteps)
    train_time = time.time() - t0
    print(f"[Done] Training completed in {train_time:.2f} seconds!")
    
    # 4. Save Trained Model
    model_path = "ppo_topopt_model.zip"
    model.save(model_path)
    print(f"[Saved] Trained PPO policy saved to {model_path}")
    
    # 5. Evaluate Trained Agent on a fresh design episode
    print("\n--> Evaluating Trained Agent on a fresh design episode...")
    obs, info = raw_env.reset()
    initial_compliance = info['compliance']
    
    agent_compliance_history = [initial_compliance]
    
    for step in range(1, 21):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = raw_env.step(action)
        agent_compliance_history.append(info['compliance'])
        if step % 5 == 0 or step == 1 or terminated or truncated:
            print(f"Step {step:<3} | Compliance: {info['compliance']:<10.2f} | Volume: {info['volume_fraction']:.2f}")
        if terminated or truncated:
            break
            
    final_density = raw_env.density.copy()
    final_compliance = info['compliance']
    print(f"--> Initial Compliance: {initial_compliance:.2f} | Final Compliance: {final_compliance:.2f}")
    if final_compliance < initial_compliance:
        print(f"--> Compliance Reduced by: {((initial_compliance - final_compliance)/initial_compliance)*100:.1f}%")
    
    # 6. Benchmark with Classical SIMP for comparison
    classical_solver = TopOpt2D(nelx=30, nely=10, volfrac=0.4, penal=3.0, rmin=1.5, bc_type='cantilever')
    classical_history = classical_solver.run_optimization(max_iters=20, tol=0.01, verbose=False)
    
    # 7. Generate Comparison Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    
    # Classical SIMP
    axes[0].imshow(1 - classical_solver.x, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f"Classical SIMP (OC)\nFinal Compliance: {classical_history['compliance'][-1]:.2f}", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("X elements")
    axes[0].set_ylabel("Y elements")
    
    # Trained RL Agent
    axes[1].imshow(1 - final_density, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f"Trained PPO Agent\nFinal Compliance: {final_compliance:.2f}", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("X elements")
    axes[1].set_ylabel("Y elements")
    
    plt.suptitle("Topology Optimization: Classical SIMP vs. Deep Reinforcement Learning (PPO)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    output_fig = "rl_trained_agent_result.png"
    plt.savefig(output_fig, dpi=200)
    plt.close()
    print(f"[Saved] Comparison plot saved to {output_fig}")
    print("\n[SUCCESS] Deep RL Agent training and benchmark completed successfully!")


if __name__ == "__main__":
    train_agent(total_timesteps=4000)
