"""
run_weighted_reward.py
======================
End-to-end run of the weighted multi-factor reward on the 30x10 cantilever (V* = 0.4).

1. Baselines in the same environment: uniform design, random agent, canonical
   greedy sensitivity heuristic, the residual base direction alone (residual env
   with zero residual action), and SIMP at 30 FEAs.
2. Residual PPO (alpha = 0.2) trained with two reward weightings:
     - "audited":  w_stiff=1, w_weight=10, w_gray=0, w_smooth=0, w_term=0
                   (identical to the Stage 3/4 reward)
     - "multi":    w_stiff=1, w_weight=10, w_gray=0.05, w_smooth=0.01, w_term=1.0
                   (all five factors active)
3. Writes to results/weighted_reward/:
     results.csv           final compliance / grayness / return for every method
     reward_terms.csv      per-step reward factors of the trained policies
     training_curves.png   episode return and final compliance during training
     compliance_history.png  compliance vs step for every method
     topologies.png        final density fields

Usage (from the repo root):
    python src/run_weighted_reward.py --timesteps 20000 --seeds 42 43 44
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import csv
import time
import argparse
import multiprocessing as mp

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env.topopt_env import TopOptEnv
from fem.cases import cantilever_case
from simp.simp import run_simp

OUT_DIR = os.path.join("results", "weighted_reward")

BASE_CONFIG = {
    "nelx": 30, "nely": 10, "volfrac": 0.4, "rmin": 1.5,
    "delta_max": 0.05, "max_steps": 30, "project_volume": True,
    "filter_type": "density", "channels": 4, "Emin": 1e-9,
}
REWARDS = {
    "audited": {"w_stiff": 1.0, "w_weight": 10.0, "w_gray": 0.0, "w_smooth": 0.0,
                "w_term": 0.0, "reward_clip": 2.0},
    "multi":   {"w_stiff": 1.0, "w_weight": 10.0, "w_gray": 0.05, "w_smooth": 0.01,
                "w_term": 1.0, "reward_clip": 2.0},
}
ALPHA = 0.2


def gray_fraction(x):
    return float(np.mean((x > 0.1) & (x < 0.9)))


def rollout(env, policy, seed):
    """Run one episode. policy(obs, env) -> action. Returns summary + histories."""
    obs, info = env.reset(seed=seed)
    comp, rewards, terms = [info["compliance"]], [], []
    for _ in range(env.max_steps):
        action = policy(obs, env)
        obs, r, term, trunc, info = env.step(action)
        comp.append(info["compliance"])
        rewards.append(r)
        terms.append(info["reward_terms"])
        if term or trunc:
            break
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    return {
        "compliance": float(info["compliance"]),
        "volume": float(info["volume"]),
        "grayness": gray_fraction(base.x_phys),
        "return": float(np.sum(rewards)),
        "comp_hist": comp,
        "terms": terms,
        "x": base.x_phys.copy(),
    }


# ----------------------------------------------------------------------------- PPO
def train_one(args):
    """Train residual PPO for one (reward name, seed). Runs in a worker process."""
    reward_name, seed, timesteps = args
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback
    from train import FlatObsWrapper

    cfg = dict(BASE_CONFIG, residual=True, alpha=ALPHA, **REWARDS[reward_name])

    class EpisodeLog(BaseCallback):
        def __init__(self):
            super().__init__()
            self.returns, self.final_c, self._ret = [], [], 0.0

        def _on_step(self):
            self._ret += float(self.locals["rewards"][0])
            if self.locals["dones"][0]:
                self.returns.append(self._ret)
                self.final_c.append(float(self.locals["infos"][0]["compliance"]))
                self._ret = 0.0
            return True

    env = Monitor(FlatObsWrapper(config=cfg, seed=seed))
    model = PPO(
        "MlpPolicy", env, learning_rate=1e-4, n_steps=256, batch_size=64, n_epochs=5,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.001,
        policy_kwargs=dict(log_std_init=-2.0, net_arch=dict(pi=[128, 128], vf=[128, 128])),
        seed=seed, verbose=0,
    )
    cb = EpisodeLog()
    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=cb)
    train_time = time.time() - t0

    ckpt_dir = os.path.join(OUT_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save(os.path.join(ckpt_dir, f"residual_{reward_name}_seed{seed}.zip"))

    test_env = FlatObsWrapper(config=cfg, seed=seed)
    res = rollout(test_env, lambda o, e: model.predict(o, deterministic=True)[0], seed)
    res.update(returns=cb.returns, train_final_c=cb.final_c, train_time=train_time,
               reward_name=reward_name, seed=seed)
    print(f"  [{reward_name:7s}] seed {seed}: C = {res['compliance']:.2f} "
          f"(trained {train_time:.0f}s, {len(cb.returns)} episodes)", flush=True)
    return res


# ------------------------------------------------------------------------ baselines
def run_baselines(reward_cfg):
    out = {}
    cfg = dict(BASE_CONFIG, **reward_cfg)

    env = TopOptEnv(config=cfg)
    out["Uniform design"] = rollout(env, lambda o, e: np.zeros((e.nely, e.nelx), np.float32), 0)
    # Uniform: zero action keeps x = V* everywhere, so compliance stays at C0
    out["Uniform design"]["comp_hist"] = [out["Uniform design"]["comp_hist"][0]] * 31

    rand = []
    for s in (0, 1, 2):
        env = TopOptEnv(config=cfg, seed=s)
        env.action_space.seed(s)
        rand.append(rollout(env, lambda o, e: e.action_space.sample(), s))
    out["Random agent"] = rand

    # Canonical greedy sensitivity heuristic: action = -dc / max|dc| (same as evaluate_baselines.py)
    def greedy(o, e):
        dc = -e.config["p"] * (e.x_phys ** (e.config["p"] - 1.0)) * e.ce
        return (-dc / (np.max(np.abs(dc)) + 1e-12)).astype(np.float32)
    env = TopOptEnv(config=cfg)
    out["Greedy heuristic"] = rollout(env, greedy, 0)

    # Residual base direction alone = residual env with zero residual action (what PPO refines)
    env = TopOptEnv(config=dict(cfg, residual=True, alpha=ALPHA))
    out["Residual base (zero residual)"] = rollout(env, lambda o, e: np.zeros((e.nely, e.nelx), np.float32), 0)

    case = cantilever_case(nelx=30, nely=10, volfrac=0.4)
    simp = run_simp(case, max_iter=30, tol=1e-6)
    out["SIMP @ 30 FEAs"] = {"compliance": float(simp["compliance"]),
                             "x": np.asarray(simp["x_phys"]),
                             "comp_hist": list(simp["history_c"])}
    return out


def summarize(runs):
    if isinstance(runs, dict):
        runs = [runs]
    c = [r["compliance"] for r in runs]
    g = [r.get("grayness", np.nan) for r in runs]
    ret = [r.get("return", np.nan) for r in runs]
    return np.mean(c), np.std(c), np.mean(g), np.mean(ret)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=20000)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2))))
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("== Baselines (30x10 cantilever, V* = 0.4) ==", flush=True)
    base = run_baselines(REWARDS["audited"])
    for k, v in base.items():
        m, s, _, _ = summarize(v)
        print(f"  {k:18s}: C = {m:.2f} ± {s:.2f}", flush=True)

    print(f"== Residual PPO, {args.timesteps} steps x {len(args.seeds)} seeds x {len(REWARDS)} rewards ==",
          flush=True)
    jobs = [(name, s, args.timesteps) for name in REWARDS for s in args.seeds]
    if args.workers > 1:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            ppo = pool.map(train_one, jobs)
    else:
        ppo = [train_one(j) for j in jobs]
    by_reward = {n: [r for r in ppo if r["reward_name"] == n] for n in REWARDS}

    # ---------------------------------------------------------------- results.csv
    rows = []
    label = {"audited": "Residual PPO, audited reward", "multi": "Residual PPO, multi-factor reward"}
    for k in ("Uniform design", "Random agent", "Greedy heuristic", "Residual base (zero residual)"):
        rows.append((k, *summarize(base[k]), 1 if k == "Uniform design" else 30, "-"))
    for n in REWARDS:
        w = REWARDS[n]
        wtxt = (f"w_stiff={w['w_stiff']} w_weight={w['w_weight']} w_gray={w['w_gray']} "
                f"w_smooth={w['w_smooth']} w_term={w['w_term']}")
        rows.append((label[n], *summarize(by_reward[n]), 30, wtxt))
    rows.append(("SIMP @ 30 FEAs", base["SIMP @ 30 FEAs"]["compliance"], 0.0,
                 gray_fraction(base["SIMP @ 30 FEAs"]["x"]), np.nan, 30, "-"))
    with open(os.path.join(OUT_DIR, "results.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["Method", "Mean compliance", "Std compliance", "Gray fraction (0.1<x<0.9)",
                     "Episode return", "FEA solves", "Reward weights"])
        for r in rows:
            wr.writerow([r[0], f"{r[1]:.2f}", f"{r[2]:.2f}", f"{r[3]:.3f}",
                         "" if np.isnan(r[4]) else f"{r[4]:.4f}", r[5], r[6]])
        wr.writerow([])
        wr.writerow([f"# timesteps per seed = {args.timesteps}; seeds = {args.seeds}; alpha = {ALPHA}"])

    # ---------------------------------------------------------- reward_terms.csv
    keys = ["F_stiffness", "F_weight", "F_grayness", "F_smoothness", "F_terminal",
            "r_stiffness", "r_weight", "r_grayness", "r_smoothness", "r_terminal", "total"]
    with open(os.path.join(OUT_DIR, "reward_terms.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["reward", "seed", "step"] + keys)
        for r in ppo:
            for t, d in enumerate(r["terms"], 1):
                wr.writerow([r["reward_name"], r["seed"], t] + [f"{d[k]:.6g}" for k in keys])

    # ------------------------------------------------------------------- figures
    colors = {"audited": "#2a6fdb", "multi": "#d1495b"}

    def smooth(y, k=10):
        y = np.asarray(y, float)
        return np.convolve(y, np.ones(k) / k, mode="valid") if len(y) > k else y

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for n, runs in by_reward.items():
        for i, r in enumerate(runs):
            ax[0].plot(smooth(r["returns"]), color=colors[n], alpha=0.8,
                       label=label[n] if i == 0 else None)
            ax[1].plot(smooth(r["train_final_c"]), color=colors[n], alpha=0.8,
                       label=label[n] if i == 0 else None)
    ax[0].set(title="Episode return during training", xlabel="Episode", ylabel="Return (10-ep mean)")
    ax[1].axhline(base["Residual base (zero residual)"]["compliance"], ls="--", c="gray",
                  label="Residual base (zero residual)")
    ax[1].axhline(base["SIMP @ 30 FEAs"]["compliance"], ls=":", c="k", label="SIMP @ 30 FEAs")
    ax[1].set(title="Final compliance during training", xlabel="Episode", ylabel="Compliance (10-ep mean)")
    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(base["Random agent"][0]["comp_hist"], c="#999", label="Random agent")
    ax.plot(base["Greedy heuristic"]["comp_hist"], c="#e8a33d", label="Greedy heuristic")
    ax.plot(base["Residual base (zero residual)"]["comp_hist"], c="gray", ls="--",
            label="Residual base (zero residual)")
    if base["SIMP @ 30 FEAs"]["comp_hist"]:
        ax.plot(base["SIMP @ 30 FEAs"]["comp_hist"], c="k", ls=":", label="SIMP")
    for n, runs in by_reward.items():
        best = min(runs, key=lambda r: r["compliance"])
        ax.plot(best["comp_hist"], c=colors[n], label=f"{label[n]} (seed {best['seed']})")
    ax.set_yscale("log"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    ax.set(title="Compliance over the 30-step episode", xlabel="Step (FEA solve)", ylabel="Compliance")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "compliance_history.png"), dpi=150); plt.close(fig)

    panels = [("Random agent", base["Random agent"][0]), ("Greedy heuristic", base["Greedy heuristic"]),
              ("Residual base (zero residual)", base["Residual base (zero residual)"])]
    for n, runs in by_reward.items():
        best = min(runs, key=lambda r: r["compliance"])
        panels.append((label[n], best))
    panels.append(("SIMP @ 30 FEAs", base["SIMP @ 30 FEAs"]))
    fig, axs = plt.subplots(len(panels), 1, figsize=(6, 1.9 * len(panels)))
    for a, (name, r) in zip(axs, panels):
        a.imshow(1 - np.asarray(r["x"]).reshape(10, 30), cmap="gray", vmin=0, vmax=1, interpolation="none")
        a.set_title(f"{name}: C = {r['compliance']:.1f}", fontsize=9); a.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "topologies.png"), dpi=150); plt.close(fig)

    print("\n== Summary ==")
    for r in rows:
        print(f"  {r[0]:36s} C = {r[1]:8.2f} ± {r[2]:.2f}   gray = {r[3]:.3f}")
    print(f"\nSaved results to {OUT_DIR}/")


if __name__ == "__main__":
    main()
