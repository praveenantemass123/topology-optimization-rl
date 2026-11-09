"""
evaluate_baselines.py
=====================
Sanity baselines on the 30x10 cantilever (V* = 0.4), all evaluated in the same
environment and with the same budget of 30 design updates (31 FEA solves:
one for the starting design and one after every update).

    Uniform design                       no update
    Random agent                         3 seeds
    Standard PPO (Gate 3)                3 trained seeds, deterministic policy
    Greedy sensitivity heuristic         action = -dC/dx / max|dC/dx|
    Residual base (zero residual)        physics base action only, no learning
    Residual PPO (Stage 4)               3 trained seeds, deterministic policy
    SIMP at 20 / 30 updates              sensitivity filter and density filter
    SIMP converged                       sensitivity filter and density filter

The PPO rows need the checkpoints written by train.py; they are skipped (with a
message) if a checkpoint is missing. Writes results/sanity_baselines.csv.

Usage (from the repo root):
    python src/evaluate_baselines.py
"""

import csv
import os
from typing import List, Optional, Sequence

import numpy as np

from common import (ALPHA, CHECKPOINT_DIR, PPO_MODES, RANDOM_SEEDS, RESULTS_DIR, TRAIN_SEEDS,
                    FlatObsWrapper, greedy_policy, make_config, mean_std, model_policy,
                    random_policy, rollout, zero_policy)
from env.topopt_env import TopOptEnv
from fem.cases import cantilever_case
from simp.simp import run_simp

N_UPDATES = 30


def eval_checkpoints(mode: str, seeds: Sequence[int] = TRAIN_SEEDS) -> Optional[List[float]]:
    """Final compliance of each trained seed of a mode, or None if a checkpoint is missing."""
    from stable_baselines3 import PPO

    cfg = make_config("audited", residual=(mode == "residual"), alpha=ALPHA)
    out = []
    for s in seeds:
        path = os.path.join(CHECKPOINT_DIR, f"{PPO_MODES[mode]['ckpt']}_seed{s}.zip")
        if not os.path.exists(path):
            print(f"  (skipping {mode} PPO: {path} not found, run src/train.py --mode {mode})")
            return None
        model = PPO.load(path, device="cpu")
        out.append(rollout(FlatObsWrapper(config=cfg, seed=s), model_policy(model), s)["compliance"])
    return out


def main():
    cfg = make_config("audited")
    case = cantilever_case(cfg["nelx"], cfg["nely"], cfg["volfrac"])
    rows = []

    def add(method, values, updates, description):
        values = list(np.atleast_1d(values))
        m, s = mean_std(values)
        rows.append([method, f"{m:.2f}", f"{s:.2f}", " ".join(f"{v:.2f}" for v in values),
                     updates, updates + 1, description])
        print(f"  {method:44s}: C = {m:8.2f} +/- {s:6.2f}", flush=True)

    print("== Sanity baselines, cantilever 30x10, V* = 0.4 ==")
    env = TopOptEnv(config=cfg)
    _, info = env.reset(seed=0)
    add("Uniform Baseline", info["compliance"], 0, "Homogeneous x_e = 0.40")

    rand = [rollout(TopOptEnv(config=cfg, seed=s), random_policy, s)["compliance"] for s in RANDOM_SEEDS]
    add("Random Agent (3 seeds)", rand, N_UPDATES,
        "Uniform random action in [-1, 1] scaled by delta_max, seeds 0 1 2")

    std_ppo = eval_checkpoints("standard")
    if std_ppo:
        add("Standard PPO Agent (Gate 3)", std_ppo, N_UPDATES,
            "3 separately trained PPO models (seeds 42 43 44), deterministic policy")

    add("Canonical Sensitivity Heuristic", rollout(TopOptEnv(config=cfg), greedy_policy, 0)["compliance"],
        N_UPDATES, "Greedy step -dC/dx / max|dC/dx| (chain rule through the density filter) with volume projection")

    res_cfg = make_config("audited", residual=True, alpha=ALPHA)
    add("Residual Base (zero residual)", rollout(TopOptEnv(config=res_cfg), zero_policy, 0)["compliance"],
        N_UPDATES, "Standardised sensitivity base action only; no learned correction")

    res_ppo = eval_checkpoints("residual")
    if res_ppo:
        add("Residual PPO Agent (Stage 4)", res_ppo, N_UPDATES,
            f"3 separately trained residual PPO models (alpha={ALPHA}), deterministic policy")

    for filt in ("sensitivity", "density"):
        for n in (20, N_UPDATES):
            r = run_simp(case, filter_type=filt, max_iter=n, tol=0.0)
            add(f"Classical SIMP @ {n} updates ({filt} filter)", r["compliance"], n,
                f"SIMP OC, {filt} filter, compliance of the design after {n} updates")
    for filt in ("sensitivity", "density"):
        r = run_simp(case, filter_type=filt, max_iter=1000, tol=0.005)
        add(f"Converged Classical SIMP ({filt} filter)", r["compliance"], r["n_updates"],
            f"SIMP OC, {filt} filter, converged to max|dx| < 0.005")

    path = os.path.join(RESULTS_DIR, "sanity_baselines.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["Method", "Mean Compliance", "Std Compliance", "Per-seed Compliance",
                     "Design Updates", "FEA Solves", "Description"])
        wr.writerows(rows)
    print(f"Saved {path} (Std = sample standard deviation, ddof = 1)")


if __name__ == "__main__":
    main()
