"""
plot_hierarchy.py
=================
Six-panel figure of the benchmark hierarchy on the 30x10 cantilever:
uniform design, random agent, standard PPO, greedy heuristic, residual PPO and
converged SIMP. Every title shows the compliance computed in this run; for the
two PPO panels the best of the trained seeds is shown.

Needs the checkpoints written by train.py. Writes
results/plots/rl_trained_agent_result.png.

Usage (from the repo root):
    python src/plot_hierarchy.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (ALPHA, CHECKPOINT_DIR, PPO_MODES, RANDOM_SEEDS, RESULTS_DIR, TRAIN_SEEDS,
                    FlatObsWrapper, greedy_policy, make_config, model_policy, random_policy, rollout)
from env.topopt_env import TopOptEnv
from fem.cases import cantilever_case
from simp.simp import run_simp
from viz import plot_density


def best_checkpoint(mode):
    """(seed, rollout) of the trained seed with the lowest final compliance."""
    from stable_baselines3 import PPO

    cfg = make_config("audited", residual=(mode == "residual"), alpha=ALPHA)
    runs = []
    for s in TRAIN_SEEDS:
        model = PPO.load(os.path.join(CHECKPOINT_DIR, f"{PPO_MODES[mode]['ckpt']}_seed{s}.zip"), device="cpu")
        runs.append((s, rollout(FlatObsWrapper(config=cfg, seed=s), model_policy(model), s)))
    return min(runs, key=lambda sr: sr[1]["compliance"])


def main():
    cfg = make_config("audited")
    case = cantilever_case(cfg["nelx"], cfg["nely"], cfg["volfrac"])

    env = TopOptEnv(config=cfg)
    _, info = env.reset(seed=0)
    uniform = (env.x_phys.copy(), info["compliance"])
    rand = rollout(TopOptEnv(config=cfg, seed=RANDOM_SEEDS[0]), random_policy, RANDOM_SEEDS[0])
    s_std, std = best_checkpoint("standard")
    greedy = rollout(TopOptEnv(config=cfg), greedy_policy, 0)
    s_res, res = best_checkpoint("residual")
    simp = run_simp(case, max_iter=1000, tol=0.005)

    panels = [
        ("1. Uniform", *uniform),
        (f"2. Random (seed {RANDOM_SEEDS[0]})", rand["x"], rand["compliance"]),
        (f"3. Standard PPO (seed {s_std})", std["x"], std["compliance"]),
        ("4. Greedy heuristic", greedy["x"], greedy["compliance"]),
        (f"5. Residual PPO (seed {s_res})", res["x"], res["compliance"]),
        ("6. SIMP (converged)", simp["x_phys"], simp["compliance"]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 5.2))
    for ax, (name, x, c) in zip(axes.ravel(), panels):
        plot_density(np.asarray(x), title=f"{name}\nC = {c:.1f}", ax=ax, case=case)
    fig.suptitle("Benchmark hierarchy on the 30x10 cantilever (V* = 0.4, Emin = 1e-9)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "plots", "rl_trained_agent_result.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
