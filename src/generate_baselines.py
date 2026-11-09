"""
generate_baselines.py
=====================
Classical SIMP + OC reference solutions for the four benchmark problems
(cantilever and half MBB beam at 30x10 and 60x20), with both the sensitivity
filter (99-line code) and the density filter (88-line code).

Writes
    results/baselines.csv
    results/plots/cantilever_optimized.png     (60x20, sensitivity filter)
    results/plots/mbb_beam_optimized.png       (60x20, sensitivity filter)
    results/plots/convergence_history.png

Settings: p = 3, rmin = 1.5, Emin = 1e-9, move = 0.2, stop when max|dx| < 0.005,
at most 1000 updates. Every compliance is a fresh FEA of the final design.

Usage (from the repo root):
    python src/generate_baselines.py
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS_DIR, fresh_compliance, gray_fraction
from fem.cases import cantilever_case, mbb_case
from simp.simp import run_simp
from viz import plot_density

TOL = 0.005
MAX_ITER = 1000
RMIN = 1.5

CASES = [
    ("cantilever_30x10", lambda: cantilever_case(30, 10, volfrac=0.4, load_pos="mid_right")),
    ("cantilever_60x20", lambda: cantilever_case(60, 20, volfrac=0.4, load_pos="mid_right")),
    ("mbb_30x10", lambda: mbb_case(30, 10, volfrac=0.5)),
    ("mbb_60x20", lambda: mbb_case(60, 20, volfrac=0.5)),
]
FIELDS = ["case_name", "bc_type", "filter", "nelx", "nely", "target_volfrac", "final_compliance",
          "iterations", "fea_solves", "runtime_seconds", "final_volfrac", "grayness",
          "max_change", "converged"]


def run_all():
    """Returns (rows for the CSV, {(case_name, filter): (case, simp result)})."""
    rows, runs = [], {}
    for filter_type in ("sensitivity", "density"):
        for name, make_case in CASES:
            case = make_case()
            res = run_simp(case, rmin=RMIN, p=3.0, filter_type=filter_type, max_iter=MAX_ITER, tol=TOL)
            c = fresh_compliance(res["x_phys"], case)
            assert np.isclose(c, res["compliance"], rtol=1e-9), "SIMP compliance is not that of its design"
            runs[(name, filter_type)] = (case, res)
            rows.append({
                "case_name": name, "bc_type": case["name"], "filter": filter_type,
                "nelx": case["nelx"], "nely": case["nely"], "target_volfrac": case["volfrac"],
                "final_compliance": f"{c:.4f}", "iterations": res["n_updates"],
                "fea_solves": res["n_fea"], "runtime_seconds": f"{res['time_s']:.2f}",
                "final_volfrac": f"{res['volume']:.6f}",
                "grayness": f"{gray_fraction(res['x_phys']):.4f}",
                "max_change": f"{res['change']:.5f}", "converged": res["converged"],
            })
            print(f"  {name:17s} {filter_type:11s}: C = {c:9.4f}  V = {res['volume']:.6f}  "
                  f"updates = {res['n_updates']:4d}  converged = {res['converged']}", flush=True)
    return rows, runs


def save_plots(runs, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    cant_case, cant = runs[("cantilever_60x20", "sensitivity")]
    mbb_case_, mbb = runs[("mbb_60x20", "sensitivity")]

    for fname, case, res, label in [
        ("cantilever_optimized.png", cant_case, cant, "Cantilever Beam"),
        ("mbb_beam_optimized.png", mbb_case_, mbb, "Half MBB Beam"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_density(res["x_phys"], ax=ax, case=case,
                     title=f"{label} ({case['nelx']}x{case['nely']}, Volfrac={case['volfrac']}, "
                           f"C={res['compliance']:.2f})")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, fname), dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(mbb["history_c"], "b-", label=f"MBB Beam 60x20 (V=0.5), final C={mbb['compliance']:.2f}")
    ax.plot(cant["history_c"], "r-", label=f"Cantilever Beam 60x20 (V=0.4), final C={cant['compliance']:.2f}")
    ax.set(title="SIMP Convergence: Compliance Minimization (sensitivity filter)",
           xlabel="Design update", ylabel="Compliance (Objective C)", yscale="log")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "convergence_history.png"), dpi=200)
    plt.close(fig)


def main():
    print(f"== SIMP baselines (tol = {TOL}, max {MAX_ITER} updates, rmin = {RMIN}, Emin = 1e-9) ==")
    rows, runs = run_all()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "baselines.csv")
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    save_plots(runs, os.path.join(RESULTS_DIR, "plots"))
    print(f"Saved {path} and the SIMP figures in results/plots/")


if __name__ == "__main__":
    main()
