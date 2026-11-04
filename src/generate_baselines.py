"""
generate_baselines.py
=====================
Regenerates baselines.csv using the authoritative src/ modules with:
- Emin = 1e-9 (strict global convention)
- max_iter = 250, tol = 0.005 (true mathematical convergence, not arbitrary caps)
- Fresh FEA solve of final x_phys
- Reports grayness fraction (0.1 < x < 0.9) alongside compliance
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import csv
import time
import numpy as np

from fem.cases import cantilever_case, mbb_case
from simp.simp import run_simp
from fem.element import lk
from fem.mesh import build_edofmat, build_sparse_indices
from fem.solver import solve

cases_to_run = [
    ("cantilever_30x10", cantilever_case(30, 10, volfrac=0.4, load_pos="mid_right")),
    ("cantilever_60x20", cantilever_case(60, 20, volfrac=0.4, load_pos="mid_right")),
    ("mbb_30x10", mbb_case(30, 10, volfrac=0.5)),
    ("mbb_60x20", mbb_case(60, 20, volfrac=0.5)),
]

print("=======================================================================")
print(" Regenerating Ground-Truth baselines.csv (Emin=1e-9, True Convergence) ")
print("=======================================================================")

rows = []

for name, case in cases_to_run:
    print(f"\nRunning {name} (tol=0.005, max_iter=250)...")
    res = run_simp(case, rmin=1.5, p=3.0, max_iter=250, tol=0.005)
    
    x_final = res["x_phys"]
    
    # Fresh independent FEA solve
    KE = lk(nu=0.3)
    edofMat = build_edofmat(case["nelx"], case["nely"])
    iK, jK = build_sparse_indices(edofMat)
    U, c_fresh, _, _ = solve(x_final, case, KE, edofMat, iK, jK, p=3.0, E0=1.0, Emin=1e-9)
    
    actual_vol = float(np.mean(x_final))
    grayness = float(np.mean((x_final > 0.1) & (x_final < 0.9)))
    
    print(f"--> Converged: {res['converged']} in {res['n_fea']} iters | Fresh C: {c_fresh:.4f} | Grayness: {grayness*100:.1f}%")
    
    rows.append({
        "case_name": name,
        "bc_type": case.get("name", "custom"),
        "nelx": case["nelx"],
        "nely": case["nely"],
        "target_volfrac": case["volfrac"],
        "final_compliance": round(float(c_fresh), 4),
        "iterations": res["n_fea"],
        "runtime_seconds": round(float(res["time_s"]), 4),
        "final_volfrac": round(actual_vol, 4),
        "grayness": round(grayness, 4),
        "converged": res["converged"]
    })

csv_file = "baselines.csv"
fieldnames = [
    "case_name", "bc_type", "nelx", "nely", "target_volfrac",
    "final_compliance", "iterations", "runtime_seconds", "final_volfrac", "grayness", "converged"
]

with open(csv_file, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\n[Saved] Authoritative baselines saved to {csv_file}")
