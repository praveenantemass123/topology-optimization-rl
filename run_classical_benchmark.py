"""
run_classical_benchmark.py
==========================
Runs the 99-line SIMP topology optimization solver on standard benchmarks:
1. Half MBB Beam (Symmetric Half Bridge)
2. Cantilever Beam

Generates visual figures to verify the mechanics and optimization loop.
"""

import time
import matplotlib.pyplot as plt
from topopt_solver import TopOpt2D

def main():
    print("====================================================")
    print(" BTP TopOpt Benchmark: Testing Classical SIMP Solver ")
    print("====================================================")
    
    # ----------------------------------------------------
    # Test 1: Half MBB Beam Benchmark
    # ----------------------------------------------------
    t0 = time.time()
    mbb_solver = TopOpt2D(nelx=60, nely=20, volfrac=0.5, penal=3.0, rmin=1.5, bc_type='mbb')
    mbb_history = mbb_solver.run_optimization(max_iters=40, tol=0.01, verbose=True)
    mbb_time = time.time() - t0
    print(f"MBB Optimization finished in {mbb_time:.2f} seconds.")
    mbb_solver.plot_structure("mbb_beam_optimized.png", title=f"Half MBB Beam (60x20, Volfrac=0.5, C={mbb_history['compliance'][-1]:.2f})")
    
    # ----------------------------------------------------
    # Test 2: Cantilever Beam Benchmark
    # ----------------------------------------------------
    t0 = time.time()
    cant_solver = TopOpt2D(nelx=60, nely=20, volfrac=0.4, penal=3.0, rmin=1.5, bc_type='cantilever')
    cant_history = cant_solver.run_optimization(max_iters=40, tol=0.01, verbose=True)
    cant_time = time.time() - t0
    print(f"Cantilever Optimization finished in {cant_time:.2f} seconds.")
    cant_solver.plot_structure("cantilever_optimized.png", title=f"Cantilever Beam (60x20, Volfrac=0.4, C={cant_history['compliance'][-1]:.2f})")
    
    # ----------------------------------------------------
    # Plot Convergence Curve
    # ----------------------------------------------------
    plt.figure(figsize=(7, 4))
    plt.plot(mbb_history['compliance'], 'b-o', markersize=4, label='MBB Beam (V=0.5)')
    plt.plot(cant_history['compliance'], 'r-s', markersize=4, label='Cantilever Beam (V=0.4)')
    plt.xlabel('Iteration')
    plt.ylabel('Compliance (Objective C)')
    plt.title('SIMP Convergence: Compliance Minimization')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('convergence_history.png', dpi=200)
    plt.close()
    print("[Saved] Convergence plot saved to convergence_history.png")
    
    print("\n[SUCCESS] Classical TopOpt benchmarks completed successfully!")

if __name__ == "__main__":
    main()
