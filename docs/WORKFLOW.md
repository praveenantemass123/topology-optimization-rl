# BTP Workflow: Reinforcement Learning for Topology Optimization

**Supervisor:** Prof. Asur · **Institute:** IIT Kharagpur  
**One-sentence thesis:** *Train an RL agent that produces near-SIMP-quality topologies for unseen load cases in one forward pass, and measure honestly when it beats, matches or loses to classical SIMP.*

---

## 0. The idea in one picture

```
Problem (domain, loads, supports, volume fraction)
        │
        ▼
 State s_t ──► Policy π_θ (CNN) ──► Action a_t (density update)
        ▲                                   │
        │                                   ▼
        └────────  Reward r_t  ◄──── FEA solver (K U = F → compliance)
```

Baseline to beat or match: SIMP + sensitivity filter + OC (99/88-line codes).

---

## 1. Workflow overview (7 stages)

| # | Stage | Output | Gate to move on |
|---|-------|--------|-----------------|
| 1 | Validate classical baseline | SIMP code matching published benchmark results | Compliance and shapes match Sigmund/Andreassen on cantilever and MBB |
| 2 | Formalize the MDP | Written spec: state, action, reward, termination | Prof. Asur signs off on the formulation |
| 3 | Prototype agent | PPO runs on small grid (e.g. 32×16), fixed load | Beats random and a greedy heuristic baseline |
| 4 | Improve architecture and reward | CNN policy, physics channels, shaped reward | Final compliance within a small gap of SIMP on a fixed case |
| 5 | Generalization | Agent trained on a distribution of loads | Works on held-out load positions and angles |
| 6 | Benchmark and ablate | Tables, plots, ablation study | Every claim backed by a number and a seed count |
| 7 | Write up and present | Thesis, repo, viva deck (paper if results allow) | Reproducible from a clean checkout |

Rule: **never start a stage before its gate is passed.** Each gate gives you something concrete to show your supervisor.

---

## 2. Stage details

### Stage 1: Classical baseline (the ground truth)
- Reproduce the 99-line SIMP code in vectorized Python/SciPy (Andreassen-style assembly).
- Validate against known cases: cantilever and MBB beam (volume fraction about 0.4 to 0.5, p = 3, filter radius about 1.5 elements).
- **Checks:** objective decreases monotonically, volume constraint is met, no checkerboards, mesh-independence when you refine the grid.
- **Unit tests to write now:** element stiffness matrix symmetry; rigid-body modes give zero energy; a single-element patch test; compliance from `U^T K U` equals `F^T U`.
- Record reference numbers (compliance, iterations, runtime) for each benchmark in a `baselines.csv`. Everything later is compared to this file.

### Stage 2: MDP formulation (decide this carefully, since it drives everything)

| Component | Options to compare | Suggested starting choice |
|-----------|--------------------|---------------------------|
| **State** | density map only; + strain energy; + load/BC masks; + volume remaining | 3 to 4 channels: density, strain-energy density, load mask, support mask |
| **Action** | (a) add or remove one element; (b) small continuous density change for all elements; (c) predict full density in one shot | (b): continuous per-element delta, bounded (e.g. ±0.05), which scales better than per-element picking |
| **Transition** | deterministic: apply action, clip to [x_min, 1], optional filter, re-solve FEA | As stated; run FEA each step |
| **Reward** | compliance improvement per step; final-compliance bonus; volume-violation penalty | `r_t = −(C_t − C_{t−1}) / C_0 − λ·|V_t − V*|`, normalized so scale is comparable across load cases |
| **Termination** | fixed step budget; convergence; volume satisfied | Fixed budget (e.g. 50 to 100 steps) |
| **γ** | 0.9 to 1.0 | Start near 0.99; episodes are short and credit assignment matters |

Why normalizing reward matters: compliance values differ by orders of magnitude between load cases, which destabilizes PPO if you do not divide by an initial or SIMP reference value.

**Hard-constraint handling:** enforce the volume fraction by projection (rescale densities to hit V*) instead of relying only on a penalty. Penalty-only designs often stall at a wrong volume.

### Stage 3: Prototype agent
- Small grids first (e.g. 32×16). FEA per step is the bottleneck, so keep it fast and cache the factorization where possible.
- PPO with Stable-Baselines3; log to TensorBoard or Weights & Biases.
- **Sanity baselines you must beat:** random actions; uniform density at V*; a heuristic that removes lowest strain-energy elements. If PPO cannot beat these, debug the environment before the network.
- Run at least 3 seeds. RL results from one seed are not evidence.

### Stage 4: CNN policy and reward refinement
- Replace the flattened MLP by a fully convolutional policy (same-size output map), so one model works across grid sizes and keeps spatial structure.
- Input channels: density, strain-energy density, load mask, support mask (and optionally the volume-gap as a constant plane).
- Reward ablations: with and without final-compliance bonus; with and without volume penalty; sparse vs dense.
- Add a light smoothing or density filter to the action, to avoid checkerboards, and report its effect.
- **Gate:** on one fixed case, the RL final compliance is within a few percent of SIMP, with a binary-ness measure (fraction of elements with density between 0.1 and 0.9) reported.

### Stage 5: Generalization (the main scientific contribution)
- Train on randomized loads (position along the free edge, angle) and possibly randomized supports or volume fractions.
- Protocol: **train / validation / test split by load case**, with test cases never seen in training, including some out of the training range to expose limits.
- Curriculum idea: start with a narrow load range and widen it as the success rate rises.
- Report: success rate (volume within tolerance and compliance within X% of SIMP), median and worst-case compliance ratio, and inference time.
- Important honest comparison: **a supervised CNN trained on SIMP solutions** can also give one-pass designs. Include it as a baseline (or at least discuss it), so that the RL contribution is clearly stated. For example, RL may avoid needing a dataset of converged SIMP solutions, or may produce an iterative refinement process. Decide which claim you can actually support.

### Stage 6: Benchmarking and ablation

| Metric | How to measure |
|--------|----------------|
| Compliance ratio | `C_RL / C_SIMP` per case (converged SIMP, same mesh, same V*) |
| Volume error | `|V − V*| / V*` |
| Grayness | fraction of elements with 0.1 < x < 0.9 |
| Connectivity | connected components of the solid phase; load-path continuity |
| Time | RL inference (single pass or K refinement steps) vs full SIMP run, on the same hardware |
| Robustness | success rate on held-out and out-of-distribution loads |

Ablations to include: CNN vs MLP; number of input channels; reward terms; with and without filter; training-set size or load range; number of refinement steps at inference.
Plot: compliance ratio vs number of FEA calls, for SIMP and for RL, which is the fairest speed comparison.

### Stage 7: Writing and presentation
Write continuously from Stage 1 onward (see Section 6), instead of in the last month.

---

## 3. 14-week timeline (adjust to your semester)

| Weeks | Focus | Deliverable |
|-------|-------|-------------|
| 1 | Baseline validation and unit tests | `baselines.csv`, passing tests |
| 2 | MDP spec and environment cleanup | 2-page MDP document |
| 3 to 4 | Prototype PPO, sanity baselines, 3 seeds | First learning curves |
| 5 to 6 | CNN policy, physics channels, reward ablations | Fixed-case result within a few percent of SIMP |
| 7 to 9 | Generalization over loads, curriculum | Held-out test results |
| 10 | Supervised-CNN baseline and extra cases (L-bracket) | Comparison table |
| 11 | Full benchmarking, ablations, figures | Final tables and plots |
| 12 to 13 | Thesis writing and repo cleanup | Draft thesis, reproducible repo |
| 14 | Viva rehearsal and demo | Slide deck, tested live demo |

Buffer: Weeks 7 to 9 are the riskiest. If generalization stalls, fall back to a narrower load family and a clearly scoped claim.

---

## 4. Repository structure

```
btp-rl-topopt/
├── README.md                # how to reproduce every figure
├── pyproject.toml / requirements.txt
├── src/
│   ├── fem/                 # element stiffness, assembly, solver
│   ├── simp/                # baseline: filter, OC update
│   ├── env/                 # gymnasium environment
│   ├── models/              # CNN policy, feature extractors
│   ├── train.py             # config-driven training
│   └── evaluate.py          # benchmarks, metrics
├── configs/                 # one YAML per experiment
├── tests/                   # FEA and environment tests
├── results/                 # CSVs, figures, checkpoints (git-ignored if large)
└── notebooks/               # visualization only
```

Reproducibility checklist: fixed seeds, config saved with each run, library versions pinned, git commit hash logged, same random test-set file used by every method.

---

## 5. Experiment log template (use for every run)

```
Run ID:      2026-xx-xx_cnn_v3
Goal:        does adding strain-energy channel help?
Change:      only the input channels (everything else identical to run cnn_v2)
Seeds:       0, 1, 2
Result:      compliance ratio mean ± std, volume error, grayness
Conclusion:  keep / drop / needs follow-up
```

Change one thing at a time. A log like this turns directly into the ablation chapter.

---

## 5b. Meeting agenda for Prof. Asur (use this for the next meeting)

1. **Where I am:** baseline reproduced (show the benchmark numbers), environment and a prototype PPO run.
2. **Open decisions:** action space (continuous delta vs element picking), reward design, how generalization will be tested.
3. **Proposed claim:** generalization to unseen loads; ask whether the supervised-CNN baseline should be included.
4. **Plan:** the timeline above with gates; ask for feedback on scope (2D only, with 3D and fracture as future work).
5. **Ask:** compute access (GPU), preferred benchmark cases, and expected thesis format.

---

## 6. Thesis writing plan (write alongside the work)

| Chapter | Write after | Key content |
|---------|-------------|-------------|
| 1. Introduction | Stage 2 | Motivation, problem statement, contributions |
| 2. Literature review | Stage 1 | SIMP (Bendsøe, Sigmund 2001, Andreassen 2011), graded materials (Paulino), RL and learning-based topology optimization |
| 3. Formulation | Stage 2 | Elasticity, FEA, SIMP, sensitivity, MDP |
| 4. RL framework | Stage 4 | Environment, CNN policy, PPO, reward, filter |
| 5. Results | Stage 6 | Benchmarks, generalization, ablations, limitations |
| 6. Conclusion | Stage 7 | Findings, limitations, future work (3D, fracture) |

Before citing learning-based topology optimization papers, search and verify them yourself (for example work on CNN-based topology optimization by Sosnovik and Oseledets, and on RL for truss topology). Do not cite from memory.

---

## 7. What makes the final output strongest

1. **A clear, defensible claim.** "RL generalizes to unseen load cases at X% compliance gap with Y× fewer FEA calls" is stronger than "RL can do topology optimization".
2. **Fair baselines.** Converged SIMP, a supervised CNN, and trivial heuristics, all on the same test set.
3. **Statistics, not single runs.** Multiple seeds, mean ± std, and failure cases shown.
4. **Honest limitations.** Where the agent fails (out-of-range loads, grayness, thin members) and why.
5. **Reproducibility.** A reviewer can clone the repo and regenerate every figure.
6. **A good demo.** Animation of the agent building the structure, plus a live "change the load, get a design" demo.

### Final deliverables checklist
- [ ] Repository with README, tests and configs
- [ ] Trained checkpoints and the exact test set
- [ ] Benchmark tables and plots (compliance ratio, time, grayness, success rate)
- [ ] Thesis (40 to 60 pages) with all six chapters
- [ ] 15-minute viva deck with animation and live demo (plus a recorded backup)
- [ ] Optional: conference or journal draft if the generalization results are strong

---

## 8. Risks and fallbacks

| Risk | Early warning | Fallback |
|------|---------------|----------|
| FEA per step too slow | Training takes days | Smaller grids, sparse-solver reuse, coarse-to-fine, fewer steps per episode |
| PPO does not learn | Reward flat across seeds | Check reward scale, simplify action space, add dense reward, verify environment with a heuristic policy |
| Volume constraint drifts | Final volume off target | Projection step instead of penalty |
| Checkerboards or gray designs | High grayness | Density filter, projection or sharpening step |
| Poor generalization | Test compliance ratio far above train | Narrower load family, curriculum, more varied training data |
| Time overrun | Stage 5 not done by Week 9 | Present fixed-case result plus a scoped generalization study |

---

## 9. Immediate next actions (this week)

1. Save the benchmark numbers from your working SIMP code into `baselines.csv`.
2. Write the 2-page MDP specification (state, action, reward, termination, γ).
3. Add the sanity baselines (random, uniform, strain-energy heuristic) and confirm PPO beats them over 3 seeds.
4. Prepare the meeting agenda in Section 5b and bring one figure: SIMP result next to the current agent result.
