# Reinforcement Learning for 2D Structural Topology Optimization
**Bachelor of Technology Project (BTP)**  
**Department of Mechanical Engineering, IIT Kharagpur**  
* **Student:** Praveen Kumar  
* **Supervisor:** Prof. Asur Vijaya Kumar  
* **One-Sentence Thesis:** *Train an RL agent that produces near-SIMP-quality topologies for unseen load cases in one forward pass, and measure honestly when it beats, matches, or loses to classical SIMP.*

---

## 1. Repository Directory Structure

The project has been organized into clean, modular, and dedicated directories:

```text
btp/
├── configs/                     # Hyperparameter & experiment configurations
│   └── exp_baseline.yaml        # Baseline PPO, FEA, and mesh hyperparameters
│
├── docs/                        # Project specifications & academic references
│   ├── SPEC.md                  # Professor's Code Logic Spec (module contracts & conventions)
│   ├── WORKFLOW.md              # Professor's 7-Stage Research Workflow & Gate Criteria
│   ├── mdp_specification.md     # Formal Markov Decision Process (MDP) specification
│   └── lectures/                # Foundational RL course lectures & study guides
│       ├── RL Lecture 1_ Complete Guide.md
│       └── Reinforcement_Learning_Lectures_1_2_and_3_Minimal_Study_Guide.html
│
├── presentations/               # Viva presentations & deck automation
│   ├── BTP_TopOpt_RL_Progress_Praveen.pptx  # Progress presentation for supervisor review
│   └── generate_presentation.py             # Script generating slide decks from audited CSVs
│
├── results/                     # Experimental data, model weights & visualizations
│   ├── baselines.csv            # Ground-truth SIMP reference benchmarks (Emin = 1e-9)
│   ├── sanity_baselines.csv     # Audited 3-seed benchmark hierarchy across methods
│   ├── checkpoints/             # Trained PPO RL model checkpoints (.zip)
│   │   ├── ppo_gate3_seed{42,43,44}.zip
│   │   └── ppo_stage4_residual_seed{42,43,44}.zip
│   └── plots/                   # High-resolution figures for reports and slides
│       ├── cantilever_optimized.png
│       ├── mbb_beam_optimized.png
│       ├── convergence_history.png
│       └── rl_trained_agent_result.png
│
├── src/                         # Authoritative source code
│   ├── fem/                     # Finite Element Analysis (FEA) engine
│   │   ├── cases.py             # Boundary condition definitions (Cantilever, MBB)
│   │   ├── element.py           # Q4 bilinear element stiffness (Gauss & Analytical)
│   │   ├── mesh.py              # Connectivity & degree-of-freedom maps
│   │   └── solver.py            # Sparse COO/CSC Cholesky/LU linear elastic solver
│   ├── simp/                    # Classical SIMP baseline (Ground Truth)
│   │   ├── filters.py           # Convolutional spatial sensitivity & density filters
│   │   └── simp.py              # 88/99-line SIMP optimizer with Optimality Criteria (OC)
│   ├── env/                     # Gymnasium RL environment
│   │   └── topopt_env.py        # TopOptEnv with 4-channel tensor & volume bisection
│   ├── models/                  # Neural network architectures
│   │   └── policy.py            # CNN spatial feature extractor & Actor-Critic networks
│   ├── evaluate.py              # Agent evaluation script
│   ├── evaluate_baselines.py    # Audited multi-seed benchmark hierarchy evaluator
│   ├── generate_baselines.py    # Ground-truth SIMP benchmark generator
│   ├── plot_hierarchy.py        # 6-panel progression comparison visualizer
│   ├── train.py                 # Multi-seed PPO training runner
│   └── viz.py                   # Plotting & domain rendering routines
│
├── tests/                       # Automated pytest verification suite (43 passing tests)
│   ├── test_element_mesh.py     # Element stiffness symmetry & connectivity tests
│   ├── test_fea.py              # Rigid-body modes & work-energy balance tests
│   ├── test_policy.py           # CNN & residual action dimension tests
│   ├── test_simp_filters.py     # Filter radius & sensitivity smoothing tests
│   ├── test_solver_cases.py     # FEA solver accuracy & boundary condition tests
│   └── test_topopt_env.py       # Gymnasium API & exact volume bisection tests
│
├── archive/                     # Historical prototype code (for audit trail)
│   └── legacy_prototypes/
│
├── .gitignore                   # Git ignore rules for caches and artifacts
└── README.md                    # Project navigation & documentation
```

---

## 2. Complete Inventory of Required Files

### A. Professor & Supervision Files ("Prof's Files")
These files govern academic requirements, theoretical foundations, and formal evaluation gates:
1. **`docs/SPEC.md`**: Master code logic specification detailing mathematical equations, global conventions ($E_{\min} = 10^{-9}$, $p = 3$, column-major flattening), and module interfaces.
2. **`docs/WORKFLOW.md`**: The 7-Stage Research Workflow mandated by supervisor Prof. Asur, establishing gate criteria before moving across stages.
3. **`docs/mdp_specification.md`**: Formal definition of State ($\mathcal{S}$), Action ($\mathcal{A}$), Transition ($\mathcal{P}$), Reward ($\mathcal{R}$), and Terminal conditions for the RL formulation.
4. **`docs/lectures/RL Lecture 1_ Complete Guide.md`**: In-depth course study guide covering Markov Decision Processes, Bellman equations, and value/policy search.
5. **`docs/lectures/Reinforcement_Learning_Lectures_1_2_and_3_Minimal_Study_Guide.html`**: Interactive HTML revision guide for RL foundational theory.
6. **`presentations/BTP_TopOpt_RL_Progress_Praveen.pptx`**: Review presentation delivered to the supervisor detailing audited Stage 1–4 results.
7. **`presentations/generate_presentation.py`**: Automated script compiling verified benchmark tables and plots into the supervisor slide deck.

### B. Core Mechanics & Algorithmic Implementation (`src/`)
* **Finite Element Analysis (`src/fem/`)**:
  * `element.py`: Element stiffness matrix ($k_0$) verified analytically and via $2 \times 2$ Gauss quadrature.
  * `mesh.py`: Degree-of-freedom indices and sparse Kronecker assembly maps.
  * `solver.py`: Sparse linear system solver ($K U = F$) computing compliance $C = U^T K U$ and element strain energies.
  * `cases.py`: Standard benchmark boundary conditions (Cantilever and Half-MBB beam).
* **Classical SIMP Baseline (`src/simp/`)**:
  * `simp.py`: Solid Isotropic Material with Penalization optimizer using Optimality Criteria (OC).
  * `filters.py`: Sensitivity filtering with $r_{\min} = 1.5$ preventing mesh-dependent checkerboard artifacts.
* **Gymnasium Environment (`src/env/`)**:
  * `topopt_env.py`: 4-channel observation space (Density, Strain Energy, Load mask, Support mask), bounded continuous density changes ($\Delta x \in [-0.05, 0.05]$), exact volume enforcement via bisection shift, and residual action co-adaptation.
* **Policy Networks (`src/models/`)**:
  * `policy.py`: Fully convolutional spatial feature extractor handling arbitrary 2D domain grids.
* **Pipelines & Evaluation**:
  * `train.py`: PPO reinforcement learning pipeline with 3-seed reproducibility and tensorboard logging.
  * `evaluate_baselines.py`: Multi-seed evaluation comparing Random, Uniform, Standard PPO, Sensitivity Heuristic, Residual PPO, and SIMP.
  * `plot_hierarchy.py`: Renders the publication-grade 6-panel comparison figure.

### C. Test Suite (`tests/`)
* **43 Automated Tests** verifying element stiffness symmetry, zero rigid-body energy, energy conservation ($U^T K U = F^T U$), filter normalization, and Gym compliance.

### D. Benchmark Results & Checkpoints (`results/`)
* `results/baselines.csv`: Authoritative SIMP benchmarks on Cantilever & MBB (30x10 and 60x20).
* `results/sanity_baselines.csv`: Statistical performance table across 3 seeds.
* `results/checkpoints/`: Saved model weights for Gate 3 Standard PPO and Stage 4 Residual PPO.
* `results/plots/`: High-resolution figures displaying optimized topologies and convergence curves.

---

## 3. Workflow Progress & Gate Status

| Stage | Focus | Gate Condition | Status |
|:---:|:---|:---|:---:|
| **1** | Classical SIMP Baseline | Compliance & topology match published literature | **PASSED** (Gate 1 Cleared) |
| **2** | MDP Formalization | Formal mathematical spec ready for supervisor sign-off | **PASSED** (Gate 2 Cleared) |
| **3** | Prototype PPO Agent | Beats random & uniform baselines across 3 seeds | **PASSED** (Gate 3 Cleared) |
| **4** | Residual Policy & Architecture | Final compliance within narrow margin of SIMP | **PASSED** (Gate 4 Cleared) |
| **5** | Multi-Load Generalization | Generalization across unseen continuous load angles/positions | *Upcoming* |
| **6** | Benchmarking & Ablation | Comprehensive scaling & speedup analysis | *Pending* |
| **7** | Viva & Thesis Write-up | Complete thesis submission & clean repository | *Pending* |

---

## 4. Benchmark Hierarchy (Cantilever $30 \times 10$, $V^* = 0.40$)

All values audited using true $E_{\min} = 10^{-9}$ and evaluated across 3 independent seeds ($\pm 1 \text{ std}$):

| Rank | Method | Mean Compliance ($C$) | FEA Solves | Gap to SIMP@30 |
|:---:|:---|:---:|:---:|:---:|
| 1 | **Classical SIMP (Converged)** | **399.27** | 95 | — |
| 2 | **Classical SIMP (Fixed @ 30)** | **415.65** | 30 | Baseline |
| 3 | **Stage 4 Residual PPO Agent** | **441.88 $\pm$ 1.01** | **30** | **+6.3%** |
| 4 | Sensitivity Heuristic Baseline | 602.26 | 30 | +44.9% |
| 5 | Gate 3 Standard PPO Agent | 994.30 $\pm$ 85.81 | 30 | +139.2% |
| 6 | Uniform Density ($x = 0.4$) | 1826.13 | 1 | +339.3% |
| 7 | Random Action Baseline | 2391.78 $\pm$ 154.2 | 30 | +475.4% |

---

## 5. Quickstart & Verification Commands

### Run Full Test Suite
```powershell
pytest tests/
```

### Regenerate SIMP Ground-Truth Benchmarks
```powershell
python src/generate_baselines.py
```

### Run Multi-Seed Benchmark Evaluation
```powershell
python src/evaluate_baselines.py
```

### Generate 6-Panel Progression Plot
```powershell
python src/plot_hierarchy.py
```

### Rebuild Supervisor Presentation Deck
```powershell
python presentations/generate_presentation.py
```
