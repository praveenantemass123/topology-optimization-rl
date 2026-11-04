"""
train.py
========
Audited & Hardened PPO training pipeline for RL Topology Optimization.

Supports:
1. Standard Mode (Gate 3): Trains raw continuous PPO policy to beat Random & Uniform.
2. Residual Mode (Stage 4): Trains Heuristic Residual PPO policy (alpha = 0.2) to close the gap with Classical SIMP.
- Multi-seed training (e.g. seeds 42, 43, 44) for rigorous evaluation
- Tracks reward clipping events to guarantee gradients are never artificially squashed
- Evaluates across all seeds to compute mean ± std
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import time
import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from env.topopt_env import TopOptEnv
from fem.cases import cantilever_case


class FlatObsWrapper(TopOptEnv):
    """Flattens 4-channel observation and 2D action space for standard SB3 continuous policies."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        flat_dim = int(np.prod(self.observation_space.shape))
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(flat_dim,), dtype=np.float32
        )
        act_dim = int(np.prod(self.action_space.shape))
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return obs.flatten().astype(np.float32), info
        
    def step(self, action):
        action_2d = np.array(action, dtype=np.float32).reshape((self.nely, self.nelx))
        obs, r, term, trunc, info = super().step(action_2d)
        return obs.flatten().astype(np.float32), r, term, trunc, info


def train_single_seed(
    env_config,
    total_timesteps=50000,
    seed=42,
    mode="standard",
    lr=3e-4,
    log_std_init=-1.5,
    ckpt_prefix="ppo_gate3"
):
    print(f"\n==========================================================")
    print(f" Training PPO on TopOptEnv [{mode.upper()}] | Seed: {seed} | Steps: {total_timesteps}")
    print(f"==========================================================")
    
    train_env = FlatObsWrapper(config=env_config, seed=seed)
    monitored_env = Monitor(train_env)
    
    policy_kwargs = dict(
        log_std_init=log_std_init,
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )
    
    model = PPO(
        "MlpPolicy",
        monitored_env,
        learning_rate=lr,
        n_steps=256,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=0
    )
    
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps)
    train_time = time.time() - t0
    
    os.makedirs("results/checkpoints", exist_ok=True)
    ckpt = f"results/checkpoints/{ckpt_prefix}_seed{seed}.zip"
    model.save(ckpt)
    print(f"[Done] Seed {seed} trained in {train_time:.1f}s ({total_timesteps/train_time:.0f} FPS). Checkpoint saved to {ckpt}")
    
    # Deterministic Evaluation
    test_env = FlatObsWrapper(config=env_config, seed=seed)
    obs, info = test_env.reset(seed=seed)
    test_env.action_space.seed(seed)
    
    for step in range(env_config["max_steps"]):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        if terminated or truncated:
            break
            
    final_c = info["compliance"]
    final_v = info["volume"]
    clip_events = info.get("total_clip_events", 0)
    print(f"  Seed {seed} -> Final Compliance: {final_c:.2f} | Vol: {final_v:.4f} | Clip Events: {clip_events}")
    
    return final_c, clip_events


def train_multi_seed(
    seeds=[42, 43, 44],
    total_timesteps=30000,
    mode="residual",
    alpha=0.2
):
    is_residual = (mode == "residual")
    lr = 1e-4 if is_residual else 3e-4
    log_std_init = -2.0 if is_residual else -1.5
    ckpt_prefix = "ppo_stage4_residual" if is_residual else "ppo_gate3"
    
    env_config = {
        "nelx": 30,
        "nely": 10,
        "volfrac": 0.4,
        "rmin": 1.5,
        "delta_max": 0.05,
        "max_steps": 30,
        "project_volume": True,
        "lam_v": 10.0,
        "beta": 0.0,
        "filter_type": "density",
        "channels": 4,
        "Emin": 1e-9,
        "reward_clip": 2.0,
        "residual": is_residual,
        "alpha": alpha
    }
    
    results = []
    total_clips = 0
    t_start = time.time()
    
    for seed in seeds:
        c, clips = train_single_seed(
            env_config,
            total_timesteps=total_timesteps,
            seed=seed,
            mode=mode,
            lr=lr,
            log_std_init=log_std_init,
            ckpt_prefix=ckpt_prefix
        )
        results.append(c)
        total_clips += clips
        
    mean_c = float(np.mean(results))
    std_c = float(np.std(results))
    total_time = time.time() - t_start
    
    print("\n==========================================================")
    print(f" Multi-Seed Training Summary [{mode.upper()}] ({len(seeds)} Seeds, {total_timesteps} Steps each)")
    print("==========================================================")
    for s, c in zip(seeds, results):
        print(f"  Seed {s}: Compliance = {c:.2f}")
    print(f"  Aggregate Compliance: {mean_c:.2f} ± {std_c:.2f}")
    print(f"  Total Reward Clip Events during evaluation: {total_clips}")
    print(f"  Total Elapsed Time: {total_time:.1f}s")
    print("==========================================================")
    return mean_c, std_c


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on TopOptEnv across multiple seeds.")
    parser.add_argument("--timesteps", type=int, default=30000, help="Timesteps per seed")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44], help="Random seeds")
    parser.add_argument("--mode", type=str, default="residual", choices=["standard", "residual"], help="Training mode")
    parser.add_argument("--alpha", type=float, default=0.2, help="Residual scaling alpha")
    args = parser.parse_args()
    
    train_multi_seed(seeds=args.seeds, total_timesteps=args.timesteps, mode=args.mode, alpha=args.alpha)
