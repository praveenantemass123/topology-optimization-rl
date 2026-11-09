"""
run_weighted_reward.py
======================
End-to-end run of the weighted multi-factor reward on the 30x10 cantilever
(V* = 0.4, 30 design updates).

1. Baselines in the same environment: uniform design, random agent, greedy
   sensitivity heuristic, the residual base action alone (zero residual), and
   SIMP after 30 updates (density filter, i.e. the same design parametrisation
   as the environment, and sensitivity filter).
2. Residual PPO (alpha = 0.2) trained with two reward weightings:
     audited  w_stiff=1, w_weight=10, w_gray=0,    w_smooth=0,    w_term=0
     multi    w_stiff=1, w_weight=10, w_gray=0.05, w_smooth=0.01, w_term=1
3. Writes to results/weighted_reward/:
     results.csv             final compliance / gray fraction / return per method
     per_seed.csv            the same for every trained seed
     reward_terms.csv        per-step reward factors of the trained policies
     training_curves.png     episode return and final compliance during training
     compliance_history.png  compliance against design update for every method
     topologies.png          final density fields
     checkpoints/            trained models

Usage (from the repo root):
    python src/run_weighted_reward.py --timesteps 30000 --seeds 42 43 44
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (ALPHA, PPO_MODES, RANDOM_SEEDS, RESULTS_DIR, REWARDS, TRAIN_SEEDS, greedy_policy,
                    gray_fraction, make_config, mean_std, random_policy, rollout, zero_policy)
from env.topopt_env import TopOptEnv
from fem.cases import cantilever_case
from simp.simp import run_simp
from train import train_many

OUT_DIR = os.path.join(RESULTS_DIR, "weighted_reward")
LABEL = {"audited": "Residual PPO, audited reward", "multi": "Residual PPO, multi-factor reward"}
COLOR = {"audited": "#2a6fdb", "multi": "#d1495b"}
BASE = "Residual base (zero residual)"
SIMP_D = "SIMP @ 30 updates (density filter)"
SIMP_S = "SIMP @ 30 updates (sensitivity filter)"
TERM_KEYS = ["F_stiffness", "F_weight", "F_grayness", "F_smoothness", "F_terminal",
             "r_stiffness", "r_weight", "r_grayness", "r_smoothness", "r_terminal", "total"]


def run_baselines():
    """Baselines scored with the audited reward. Values are rollout dicts (a list for the random agent)."""
    cfg = make_config("audited")
    n = cfg["max_steps"]
    out = {}

    uniform = rollout(TopOptEnv(config=cfg), zero_policy, 0)   # zero action keeps x = V*
    assert np.allclose(uniform["comp_hist"], uniform["comp_hist"][0], rtol=1e-9)
    out["Uniform design"] = uniform
    out["Random agent"] = [rollout(TopOptEnv(config=cfg, seed=s), random_policy, s) for s in RANDOM_SEEDS]
    out["Greedy heuristic"] = rollout(TopOptEnv(config=cfg), greedy_policy, 0)
    out[BASE] = rollout(TopOptEnv(config=make_config("audited", residual=True)), zero_policy, 0)

    case = cantilever_case(cfg["nelx"], cfg["nely"], cfg["volfrac"])
    for name, filt in ((SIMP_D, "density"), (SIMP_S, "sensitivity")):
        r = run_simp(case, filter_type=filt, max_iter=n, tol=0.0)
        out[name] = {"compliance": r["compliance"], "volume": r["volume"],
                     "grayness": gray_fraction(r["x_phys"]), "return": float("nan"),
                     "comp_hist": r["history_c"], "x": r["x_phys"]}
    return out


def summarize(runs):
    runs = [runs] if isinstance(runs, dict) else runs
    mean_c, std_c = mean_std([r["compliance"] for r in runs])
    return (mean_c, std_c, float(np.mean([r["grayness"] for r in runs])),
            float(np.mean([r["return"] for r in runs])))


def write_csvs(base, by_reward, timesteps, seeds):
    n = make_config("audited")["max_steps"]
    rows = []
    for k in ("Uniform design", "Random agent", "Greedy heuristic", BASE):
        updates = 0 if k == "Uniform design" else n
        rows.append((k, *summarize(base[k]), updates, updates + 1, "-"))
    for name, runs in by_reward.items():
        w = REWARDS[name]
        wtxt = " ".join(f"{k}={w[k]}" for k in ("w_stiff", "w_weight", "w_gray", "w_smooth", "w_term"))
        rows.append((LABEL[name], *summarize(runs), n, n + 1, wtxt))
    for k in (SIMP_D, SIMP_S):
        rows.append((k, *summarize(base[k]), n, n + 1, "-"))

    with open(os.path.join(OUT_DIR, "results.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["Method", "Mean compliance", "Std compliance", "Gray fraction (0.1<x<0.9)",
                     "Episode return", "Design updates", "FEA solves", "Reward weights"])
        for r in rows:
            wr.writerow([r[0], f"{r[1]:.2f}", f"{r[2]:.2f}", f"{r[3]:.3f}",
                         "" if np.isnan(r[4]) else f"{r[4]:.4f}", r[5], r[6], r[7]])
        wr.writerow([])
        wr.writerow([f"# timesteps per seed = {timesteps}; seeds = {list(seeds)}; alpha = {ALPHA}; "
                     f"Std = sample standard deviation (ddof = 1); "
                     f"baseline returns use the audited reward"])

    with open(os.path.join(OUT_DIR, "per_seed.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["reward", "seed", "timesteps", "final_compliance", "volume", "gray_fraction",
                     "episode_return", "reward_clip_events", "base_compliance", "compliance_minus_base"])
        c_base = base[BASE]["compliance"]
        for name, runs in by_reward.items():
            for r in runs:
                wr.writerow([name, r["seed"], r["timesteps"], f"{r['compliance']:.4f}", f"{r['volume']:.6f}",
                             f"{r['grayness']:.4f}", f"{r['return']:.4f}", r["clip_events"],
                             f"{c_base:.4f}", f"{r['compliance'] - c_base:+.4f}"])

    with open(os.path.join(OUT_DIR, "reward_terms.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["reward", "seed", "step"] + TERM_KEYS)
        for name, runs in by_reward.items():
            for r in runs:
                for t, d in enumerate(r["terms"], 1):
                    wr.writerow([name, r["seed"], t] + [f"{d[k]:.6g}" for k in TERM_KEYS])
    return rows


def make_figures(base, by_reward):
    def smooth(y, k=10):
        y = np.asarray(y, float)
        return np.convolve(y, np.ones(k) / k, mode="valid") if len(y) > k else y

    best = {n: min(runs, key=lambda r: r["compliance"]) for n, runs in by_reward.items()}

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for n, runs in by_reward.items():
        for i, r in enumerate(runs):
            ax[0].plot(smooth(r["returns"]), color=COLOR[n], alpha=0.8, label=LABEL[n] if i == 0 else None)
            ax[1].plot(smooth(r["train_final_c"]), color=COLOR[n], alpha=0.8, label=LABEL[n] if i == 0 else None)
    ax[0].set(title="Episode return during training", xlabel="Episode", ylabel="Return (10-ep mean)")
    ax[1].axhline(base[BASE]["compliance"], ls="--", c="gray", label=BASE)
    ax[1].axhline(base[SIMP_D]["compliance"], ls=":", c="k", label=SIMP_D)
    ax[1].set(title="Final compliance during training (stochastic policy)", xlabel="Episode",
              ylabel="Compliance (10-ep mean)")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(base["Random agent"][0]["comp_hist"], c="#999", label=f"Random agent (seed {RANDOM_SEEDS[0]})")
    ax.plot(base["Greedy heuristic"]["comp_hist"], c="#e8a33d", label="Greedy heuristic")
    ax.plot(base[BASE]["comp_hist"], c="gray", ls="--", label=BASE)
    ax.plot(base[SIMP_D]["comp_hist"], c="k", ls=":", label=SIMP_D)
    ax.plot(base[SIMP_S]["comp_hist"], c="k", ls="-.", lw=0.9, label=SIMP_S)
    for n, r in best.items():
        ax.plot(r["comp_hist"], c=COLOR[n], label=f"{LABEL[n]} (seed {r['seed']})")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set(title="Compliance over the 30-update episode", xlabel="Design update", ylabel="Compliance")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "compliance_history.png"), dpi=150)
    plt.close(fig)

    panels = [(f"Random agent (seed {RANDOM_SEEDS[0]})", base["Random agent"][0]),
              ("Greedy heuristic", base["Greedy heuristic"]), (BASE, base[BASE])]
    panels += [(f"{LABEL[n]} (seed {r['seed']})", r) for n, r in best.items()]
    panels += [(SIMP_D, base[SIMP_D]), (SIMP_S, base[SIMP_S])]
    fig, axs = plt.subplots(len(panels), 1, figsize=(6, 1.9 * len(panels)))
    for a, (name, r) in zip(axs, panels):
        a.imshow(1 - np.asarray(r["x"]), cmap="gray", vmin=0, vmax=1, interpolation="none")
        a.set_title(f"{name}: C = {r['compliance']:.1f}", fontsize=9)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "topologies.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=PPO_MODES["residual"]["timesteps"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(TRAIN_SEEDS))
    ap.add_argument("--workers", type=int, default=max(1, min(6, os.cpu_count() or 1)))
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("== Baselines (30x10 cantilever, V* = 0.4) ==", flush=True)
    base = run_baselines()
    for k, v in base.items():
        m, s, _, _ = summarize(v)
        print(f"  {k:40s}: C = {m:.2f} +/- {s:.2f}", flush=True)

    print(f"== Residual PPO, {args.timesteps} steps x {len(args.seeds)} seeds x {len(REWARDS)} rewards ==",
          flush=True)
    hp = PPO_MODES["residual"]
    jobs = [{
        "tag": name,
        "kwargs": dict(env_config=make_config(name, residual=True), seed=s, timesteps=args.timesteps,
                       lr=hp["lr"], log_std_init=hp["log_std_init"],
                       ckpt_path=os.path.join(OUT_DIR, "checkpoints", f"residual_{name}_seed{s}.zip")),
    } for name in REWARDS for s in args.seeds]
    ppo = train_many(jobs, args.workers)
    by_reward = {n: [r for r in ppo if r["tag"] == n] for n in REWARDS}

    rows = write_csvs(base, by_reward, args.timesteps, args.seeds)
    make_figures(base, by_reward)

    print("\n== Summary ==")
    for r in rows:
        print(f"  {r[0]:40s} C = {r[1]:8.2f} +/- {r[2]:.2f}   gray = {r[3]:.3f}")
    print(f"\nSaved results to {OUT_DIR}/")


if __name__ == "__main__":
    main()
