"""
generate_baselines_csv.py
=========================
Generates the foundational baselines.csv recording reference numbers
(Compliance, Iterations, Runtime, Mean Density) across standard test cases.
This serves as the ground truth against which all future RL runs are evaluated.
"""

import csv
import time
from topopt_solver import TopOpt2D

cases = [
    {"name": "cantilever_30x10", "nelx": 30, "nely": 10, "volfrac": 0.4, "penal": 3.0, "rmin": 1.5, "bc": "cantilever", "max_iters": 30},
    {"name": "cantilever_60x20", "nelx": 60, "nely": 20, "volfrac": 0.4, "penal": 3.0, "rmin": 1.5, "bc": "cantilever", "max_iters": 50},
    {"name": "mbb_30x10", "nelx": 30, "nely": 10, "volfrac": 0.5, "penal": 3.0, "rmin": 1.5, "bc": "mbb", "max_iters": 30},
    {"name": "mbb_60x20", "nelx": 60, "nely": 20, "volfrac": 0.5, "penal": 3.0, "rmin": 1.5, "bc": "mbb", "max_iters": 50},
]

rows = []
print("==================================================")
print(" Generating Ground-Truth baselines.csv (SIMP OC) ")
print("==================================================")

for c in cases:
    print(f"\nRunning benchmark: {c['name']}...")
    solver = TopOpt2D(nelx=c["nelx"], nely=c["nely"], volfrac=c["volfrac"], penal=c["penal"], rmin=c["rmin"], bc_type=c["bc"])
    t0 = time.time()
    history = solver.run_optimization(max_iters=c["max_iters"], tol=0.01, verbose=False)
    runtime = time.time() - t0
    
    final_c = history["compliance"][-1]
    iters = len(history["compliance"])
    mean_density = float(solver.x.mean())
    
    print(f"--> Converged in {iters} iters | Compliance: {final_c:.4f} | Runtime: {runtime:.3f}s")
    
    rows.append({
        "case_name": c["name"],
        "bc_type": c["bc"],
        "nelx": c["nelx"],
        "nely": c["nely"],
        "target_volfrac": c["volfrac"],
        "final_compliance": round(final_c, 4),
        "iterations": iters,
        "runtime_seconds": round(runtime, 4),
        "final_volfrac": round(mean_density, 4)
    })

csv_file = "baselines.csv"
fieldnames = ["case_name", "bc_type", "nelx", "nely", "target_volfrac", "final_compliance", "iterations", "runtime_seconds", "final_volfrac"]

with open(csv_file, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\n[Saved] Reference baselines successfully saved to {csv_file}")
