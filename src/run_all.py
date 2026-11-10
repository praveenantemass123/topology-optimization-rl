"""
run_all.py
==========
Regenerates every result of this semester from scratch, in order:

    1. generate_baselines.py     classical SIMP references      results/baselines.csv
    2. train.py (standard)       Gate 3 PPO, 3 seeds            results/train_standard.csv
    3. train.py (residual)       Stage 4 residual PPO, 3 seeds  results/train_residual.csv
    4. evaluate_baselines.py     sanity baselines               results/sanity_baselines.csv
    5. run_weighted_reward.py    weighted multi-factor reward   results/weighted_reward/
    6. plot_hierarchy.py         six-panel figure               results/plots/

Usage (from the repo root):
    python src/run_all.py            # about 15 minutes on 4 CPU cores
    python src/run_all.py --quick    # short smoke test, numbers are NOT meaningful
"""

import argparse
import os
import subprocess
import sys

SRC = os.path.dirname(os.path.abspath(__file__))


def run(script, *args):
    print(f"\n>>> python src/{script} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, os.path.join(SRC, script), *args], check=True,
                   cwd=os.path.dirname(SRC))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2048 training steps per run (smoke test)")
    ap.add_argument("--workers", type=int, default=max(1, min(6, os.cpu_count() or 1)))
    a = ap.parse_args()
    w = ["--workers", str(a.workers)]
    quick = ["--timesteps", "2048"] if a.quick else []

    run("generate_baselines.py")
    run("train.py", "--mode", "standard", *quick, *w)
    run("train.py", "--mode", "residual", *quick, *w)
    run("evaluate_baselines.py")
    run("run_weighted_reward.py", *quick, *w)
    run("plot_hierarchy.py")


if __name__ == "__main__":
    main()
