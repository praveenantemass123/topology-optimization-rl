"""
train.py
========
PPO training on the 30x10 cantilever benchmark.

Modes
    standard  PPO acts directly on the design (Gate 3 prototype), 50 000 steps.
    residual  PPO adds a correction (weight alpha = 0.2) to the physics base
              action (Stage 4), 30 000 steps.

Each seed is trained in its own process with a single torch thread, so a run is
reproducible on a given machine. After training, the deterministic policy is
rolled out once and its final compliance is reported (mean and sample standard
deviation over the seeds).

Usage (from the repo root):
    python src/train.py --mode residual
    python src/train.py --mode standard --timesteps 50000 --seeds 42 43 44
"""

import argparse
import csv
import multiprocessing as mp
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from common import (ALPHA, CHECKPOINT_DIR, PPO_MODES, RESULTS_DIR, TRAIN_SEEDS, FlatObsWrapper,
                    make_config, mean_std, model_policy, rollout)

__all__ = ["FlatObsWrapper", "train_ppo", "train_many", "train_multi_seed"]


def train_ppo(
    env_config: Dict[str, Any],
    seed: int,
    timesteps: int,
    lr: float,
    log_std_init: float,
    ckpt_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train one PPO model and evaluate its deterministic policy for one episode.

    Returns the rollout summary (see common.rollout) extended with the training
    log: "returns" and "train_final_c" per training episode, and "train_time".
    """
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    class EpisodeLog(BaseCallback):
        """Records the return and the final compliance of every training episode."""

        def __init__(self):
            super().__init__()
            self.returns: List[float] = []
            self.final_c: List[float] = []
            self._ret = 0.0

        def _on_step(self) -> bool:
            self._ret += float(self.locals["rewards"][0])
            if self.locals["dones"][0]:
                self.returns.append(self._ret)
                self.final_c.append(float(self.locals["infos"][0]["compliance"]))
                self._ret = 0.0
            return True

    env = FlatObsWrapper(config=env_config, seed=seed)
    model = PPO(
        "MlpPolicy", env,
        learning_rate=lr, n_steps=256, batch_size=64, n_epochs=5,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.001,
        policy_kwargs=dict(log_std_init=log_std_init, net_arch=dict(pi=[128, 128], vf=[128, 128])),
        seed=seed, device="cpu", verbose=0,
    )
    log = EpisodeLog()
    t0 = time.perf_counter()
    model.learn(total_timesteps=timesteps, callback=log)
    train_time = time.perf_counter() - t0

    if ckpt_path:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        model.save(ckpt_path)

    result = rollout(FlatObsWrapper(config=env_config, seed=seed), model_policy(model), seed)
    result.update(seed=seed, returns=log.returns, train_final_c=log.final_c,
                  train_time=train_time, timesteps=int(model.num_timesteps))
    return result


def _worker(job: Dict[str, Any]) -> Dict[str, Any]:
    res = train_ppo(**job["kwargs"])
    res["tag"] = job["tag"]
    print(f"  [{job['tag']:9s}] seed {res['seed']}: C = {res['compliance']:.2f}  "
          f"V = {res['volume']:.6f}  ({res['train_time']:.0f} s)", flush=True)
    return res


def train_many(jobs: Sequence[Dict[str, Any]], workers: int) -> List[Dict[str, Any]]:
    """Run training jobs ({"tag": str, "kwargs": train_ppo kwargs}), in parallel if workers > 1."""
    if workers > 1 and len(jobs) > 1:
        with mp.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            return pool.map(_worker, jobs, chunksize=1)
    return [_worker(j) for j in jobs]


def train_multi_seed(
    seeds: Sequence[int] = TRAIN_SEEDS,
    total_timesteps: Optional[int] = None,
    mode: str = "residual",
    alpha: float = ALPHA,
    workers: int = 1,
):
    """Train one model per seed; write checkpoints and results/train_<mode>.csv."""
    hp = PPO_MODES[mode]
    timesteps = int(total_timesteps or hp["timesteps"])
    cfg = make_config("audited", residual=(mode == "residual"), alpha=alpha)
    print(f"== PPO [{mode}] {timesteps} steps x seeds {list(seeds)} ==", flush=True)
    jobs = [{
        "tag": mode,
        "kwargs": dict(env_config=cfg, seed=s, timesteps=timesteps, lr=hp["lr"],
                       log_std_init=hp["log_std_init"],
                       ckpt_path=os.path.join(CHECKPOINT_DIR, f"{hp['ckpt']}_seed{s}.zip")),
    } for s in seeds]
    results = train_many(jobs, workers)

    mean_c, std_c = mean_std([r["compliance"] for r in results])
    with open(os.path.join(RESULTS_DIR, f"train_{mode}.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["mode", "seed", "timesteps", "final_compliance", "volume", "gray_fraction",
                     "episode_return", "reward_clip_events", "fea_solves"])
        for r in results:
            wr.writerow([mode, r["seed"], r["timesteps"], f"{r['compliance']:.4f}", f"{r['volume']:.6f}",
                         f"{r['grayness']:.4f}", f"{r['return']:.4f}", r["clip_events"], r["n_fea"]])
    print(f"  {mode}: C = {mean_c:.2f} +/- {std_c:.2f} (sample std over {len(results)} seeds)")
    return mean_c, std_c


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train PPO on the topology optimization environment.")
    ap.add_argument("--mode", choices=sorted(PPO_MODES), default="residual")
    ap.add_argument("--timesteps", type=int, default=None, help="default: 50000 standard, 30000 residual")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(TRAIN_SEEDS))
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--workers", type=int, default=max(1, min(3, os.cpu_count() or 1)))
    a = ap.parse_args()
    train_multi_seed(a.seeds, a.timesteps, a.mode, a.alpha, a.workers)
