# BTP Code Logic Spec: hand this to a coding model

Purpose: precise logic, interfaces and tests for every module of the RL topology-optimization project, so a coding model can write the code **one module at a time** and you can verify each one.

---

## How to use this with another model

1. Paste **Section 0 (Global conventions)** into every request. It keeps all modules consistent.
2. Then paste **one module section** at a time, in the build order of Section 12.
3. Ask for **the tests first or together with the code**. Run the tests yourself before moving on.
4. Do not ask for the entire project in one message.

Starter prompt for each module:

```
You are writing one module of a Python project (topology optimization with RL).
Follow the global conventions and the module spec exactly. Use only NumPy, SciPy,
Matplotlib, Gymnasium, PyTorch, Stable-Baselines3 (and PyYAML, pytest).
Write: (1) the module code with type hints and docstrings, (2) pytest tests listed
in the spec. Do not change function names or signatures. If something in the spec
is ambiguous, state your assumption in a comment and continue.
```

---

## 0. Global conventions (paste into every request)

- **Language/libs:** Python 3.10+, NumPy, SciPy (sparse), PyTorch, Gymnasium, Stable-Baselines3, Matplotlib, PyYAML, pytest.
- **Problem:** 2D linear elasticity, plane stress, unit-thickness, structured grid of `nelx × nely` square 4-node bilinear (Q4) elements. Unit element size.
- **Material:** `E0 = 1.0`, `Emin = 1e-9`, `nu = 0.3`, SIMP penalty `p = 3`.
- **Density arrays:** shape `(nely, nelx)`, float64. Row 0 is the **top** of the domain. Flatten in **column-major (Fortran) order** `x.flatten(order="F")` so that element index `e = ix * nely + iy` (matches the MATLAB 99/88-line codes).
- **Nodes:** `(nelx+1)*(nely+1)` nodes, node index `n = ix*(nely+1) + iy`, with `ix` from 0 to nelx, `iy` from 0 to nely (iy = 0 at top). Two DOFs per node: `dof_x = 2n`, `dof_y = 2n+1`. Positive y DOF points **up** (so a downward load has negative y force).
- **Element DOF ordering (8 DOFs):** nodes in order lower-left, lower-right, upper-right, upper-left, i.e. counter-clockwise, as in the 88-line code. Build `edofMat` of shape `(nel, 8)`.
- **Case dictionary** (used by all modules):

```python
case = {
  "nelx": int, "nely": int, "volfrac": float,
  "fixed_dofs": np.ndarray[int],        # constrained DOF indices
  "F": np.ndarray[float],               # global force vector, shape (ndof,)
  "name": str,                          # e.g. "mbb", "cantilever", "random_17"
  "load_info": dict                     # node, angle, magnitude (for masks/logging)
}
```

- **Randomness:** every random function takes an `np.random.Generator` (or seed). No global seeding inside modules.
- **Config:** all hyperparameters from a YAML file; no magic numbers in code.

---

## 1. `fem/element.py`: element stiffness matrix

**Goal:** `lk(nu) -> ndarray (8, 8)`, stiffness of a unit square Q4 plane-stress element with `E = 1`.

**Logic (two equivalent routes, implement both for a test):**
1. *Closed form (88-line code).* With
   `k = [1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3nu/8, -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3nu/8]` (k1..k8),
   `KE = 1/(1-nu²) *` the 8×8 matrix below (1-indexed entries of `k`):
   ```
   [k1 k2 k3 k4 k5 k6 k7 k8]
   [k2 k1 k8 k7 k6 k5 k4 k3]
   [k3 k8 k1 k6 k7 k4 k5 k2]
   [k4 k7 k6 k1 k8 k3 k2 k5]
   [k5 k6 k7 k8 k1 k2 k3 k4]
   [k6 k5 k4 k3 k2 k1 k8 k7]
   [k7 k4 k5 k2 k3 k8 k1 k6]
   [k8 k3 k2 k5 k4 k7 k6 k1]
   ```
2. *Gauss quadrature (reference).* Constitutive matrix `D = 1/(1-nu²) * [[1, nu, 0],[nu, 1, 0],[0, 0, (1-nu)/2]]`. For each 2×2 Gauss point `(ξ, η) = (±1/√3, ±1/√3)` compute shape-function derivatives, strain-displacement matrix `B (3×8)`, and sum `Bᵀ D B · det(J) · weight`. For a unit square, `det(J) = 1/4`, weights = 1.

**Tests:**
- Route 1 equals route 2 (`np.allclose`, tol 1e-10) for `nu = 0.3` and `nu = 0.0`.
- `KE` symmetric; eigenvalues: exactly 3 near-zero (rigid-body modes), rest positive.

---

## 2. `fem/mesh.py`: connectivity and DOF maps

**Functions**
- `build_edofmat(nelx, nely) -> ndarray (nel, 8)`:
  for each element `(ix, iy)` with `e = ix*nely + iy`: `n1 = ix*(nely+1)+iy` (top-left node), `n2 = (ix+1)*(nely+1)+iy` (top-right). Return DOFs in the counter-clockwise order required by Section 0: `[2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3, 2*n2, 2*n2+1, 2*n1, 2*n1+1]` (matches the 88-line code, where node numbering runs top to bottom). Verify against the 88-line MATLAB definition.
- `build_sparse_indices(edofMat) -> (iK, jK)`: `iK = np.kron(edofMat, np.ones((8,1))).T.flatten()`, `jK = np.kron(edofMat, np.ones((1,8))).T.flatten()`.
- `node_from_xy(ix, iy, nely) -> int`.

**Tests:** number of unique DOFs equals `2*(nelx+1)*(nely+1)`; every element's 8 DOFs are distinct; for a 1-element mesh the DOF list is a permutation of 0..7.

---

## 3. `fem/solver.py`: assemble and solve

**Function:** `solve(x_phys, case, KE, edofMat, iK, jK, p=3, E0=1.0, Emin=1e-9) -> (U, c, ce)`

**Logic**
1. Element Young's modulus: `E_e = Emin + x_phys_flat**p * (E0 - Emin)`.
2. Values: `sK = (KE.flatten()[:, None] * E_e[None, :]).flatten(order="F")`.
3. `K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc(); K = (K + K.T)/2`.
4. `free = setdiff1d(arange(ndof), fixed_dofs)`.
5. Solve `K[free][:, free] U[free] = F[free]` using `scipy.sparse.linalg.spsolve` (or `splu`; allow a `factorize` option that returns the LU object for reuse).
6. Element compliance `ce = sum((U[edofMat] @ KE) * U[edofMat], axis=1)`; total `c = sum(E_e * ce)`.
7. Return `U` (ndof,), `c` (float), `ce` reshaped to `(nely, nelx)` in the Section 0 ordering.

**Also provide:** `strain_energy_density(U, x_phys, ...) -> (nely, nelx)` = `E_e * ce`, used as an RL observation channel.

**Tests**
- `c == F @ U` (tol 1e-8).
- Uniform `x = 1`, cantilever with end load: result finite; doubling the load multiplies `c` by 4.
- Rigid translation of an unconstrained mesh is in the null space of `K` (before applying BCs).

---

## 4. `fem/cases.py`: benchmark cases and random load sampler

**Functions**
- `mbb_case(nelx, nely, volfrac)`: half MBB beam. Force: node at top-left corner, y DOF, value `-1`. Fixed: all x DOFs of left-edge nodes (symmetry), plus the y DOF of the bottom-right node.
- `cantilever_case(nelx, nely, volfrac, load_pos="mid_right")`: all DOFs of the left-edge nodes fixed; downward unit force on the chosen right-edge node.
- `lbracket_case(...)` (optional): L-shaped domain via an element mask (passive void region). If included, the density arrays need a `passive` mask applied after every update.
- `sample_load_case(base, rng, edge, angle_range_deg, ...)`: copy a base case (supports + volfrac), choose a random node on the allowed edge(s) (and optionally away from the supports), a random angle `θ`, force `(Fx, Fy) = (cos θ, sin θ) * magnitude`. Fill `case["load_info"]` with `{node, theta, magnitude}`.
- `make_test_set(n, seed, in_range, out_range) -> list[case]` and `save_cases(path)/load_cases(path)` (JSON or npz) so every method is evaluated on the **same** test cases.

**Tests:** forces nonzero at exactly the intended DOFs; fixed DOFs unique and within range; sampler is reproducible for a given seed.

---

## 5. `simp/filters.py`: filter matrix

**Function:** `build_filter(nelx, nely, rmin) -> (H (sparse nel×nel), Hs (nel,))`

**Logic:** for every element `e` (centre coordinates `(ix, iy)`), for every element `f` with centres within `ceil(rmin)-1` cells, compute `d = hypot(dx, dy)`; weight `w = max(0, rmin - d)`; store if `w > 0`. `Hs = H.sum(axis=1)`. Build with COO triplets (vectorized or loops over offsets).

**Functions**
- `density_filter(x, H, Hs)` returns `x_phys = H @ x / Hs`.
- `sensitivity_filter(x, dc, H, Hs)` returns `H @ (x * dc) / (Hs * max(1e-3, x))`.
- `chain_rule_density(dc, dv, H, Hs)` returns `H.T @ (dc / Hs)` and `H.T @ (dv / Hs)`.

**Tests:** with `rmin = 1`, `H` is the identity (up to scaling); `Hs` positive; a uniform field stays uniform under `density_filter`.

---

## 6. `simp/simp.py`: classical baseline (ground truth)

**Function:** `run_simp(case, rmin=1.5, p=3, filter_type="sensitivity"|"density", max_iter=200, tol=0.01, move=0.2, return_history=False) -> dict`

**Logic per iteration**
1. `x_phys` from `x` (identity for sensitivity filter, `density_filter` for density filter).
2. FEA: `U, c, ce = solve(...)`.
3. Sensitivities: `dc = -p * x_phys**(p-1) * (E0 - Emin) * ce`; `dv = ones`.
4. Filter: sensitivity filter, or chain rule for the density filter.
5. **OC update** by bisection on the Lagrange multiplier:
   ```
   l1, l2 = 0, 1e9
   while (l2 - l1) / (l1 + l2) > 1e-3:
       lmid = (l1 + l2) / 2
       xnew = clip(x * sqrt(-dc / dv / lmid), max(0.001, x - move), min(1, x + move))
       (x_phys_new = filtered xnew if density filter, else xnew)
       if sum(x_phys_new) > volfrac * nel: l1 = lmid else: l2 = lmid
   ```
6. `change = max|xnew - x|`; stop when `change < tol` or `max_iter`.
7. Initial `x = volfrac` everywhere.

**Return:** `{"x_phys", "compliance", "history_c", "n_fea", "time_s", "converged"}`

**Tests**
- Compliance decreases (almost monotonically) and the final volume equals `volfrac` within 1e-3.
- On a 60×20 MBB and cantilever, the picture is a recognizable truss without checkerboards; record final compliance in `baselines.csv`.
- **Finite-difference check:** perturb a random element density by ±1e-6, central-difference the compliance, compare with the analytic `dc` (relative error < 1e-4).
- Mesh independence: with `rmin` scaled to the mesh, finer meshes give similar topology.

---

## 7. `env/topopt_env.py`: Gymnasium environment

**Class:** `TopOptEnv(gym.Env)` with `__init__(config, case_sampler, seed=None)`

**Spaces**
- `observation_space = Box(0, 1, shape=(C, nely, nelx), float32)` with channels:
  1. `x_phys` (current density),
  2. normalized strain-energy density (divide by its max, or by a running scale),
  3. load mask (1 at load node's adjacent elements; optionally separate Fx, Fy channels for direction),
  4. support mask (elements adjacent to fixed DOFs),
  5. optional constant plane: `(V_current - volfrac)` and step fraction `t/T`.
- `action_space = Box(-1, 1, shape=(nely, nelx), float32)`.

**`reset(seed, options)`**
1. Sample a case from `case_sampler` (fixed case for training stage 1).
2. `x = volfrac * ones`; run FEA; store `C0 = c` and `C_prev = c`; `t = 0`.
3. Return observation and `info = {case_name, C0}`.

**`step(action)`**
1. `delta = delta_max * action` (e.g. `delta_max = 0.05`); `x = clip(x + delta, xmin, 1)`.
2. Optional density filter for `x_phys`.
3. **Volume projection:** find shift `s` (bisection) such that `mean(clip(x + s, xmin, 1)) = volfrac`; apply. (Switch controlled by `config["project_volume"]`.)
4. FEA: `c`, `ce`.
5. **Reward:** `r = -(c - C_prev) / C0 - lam_v * abs(V - volfrac)` (+ optional terminal bonus `-beta * c / C0`). Store `C_prev = c`.
6. `t += 1`; `terminated = False`; `truncated = (t >= max_steps)`.
7. Return `obs, r, terminated, truncated, info` with `info = {compliance, volume, grayness, n_fea}`.

**Notes for the coder:** keep FEA objects (KE, edofMat, filter matrices) cached per grid size; never rebuild them per step. Observation normalization must be deterministic. Compliance may blow up if the structure disconnects (near-singular `K`), so clip rewards to a sane range (e.g. [-10, 10]) and use `Emin` as defined.

**Tests:** `gymnasium.utils.env_checker.check_env` passes; zero action leaves density unchanged (up to projection); reward is zero if the compliance does not change; the episode length equals `max_steps`; two resets with the same seed give identical observations.

---

## 8. `models/policy.py`: CNN policy

**Two stages, implement in this order:**

**8a. Feature extractor for SB3 (quick start).** Subclass `BaseFeaturesExtractor`: `Conv2d(C,32,3,pad=1) → ReLU → Conv2d(32,64,3,pad=1) → ReLU → Conv2d(64,64,3,pad=1) → ReLU → Flatten → Linear(features_dim)`. Use with `PPO("CnnPolicy", env, policy_kwargs=dict(features_extractor_class=..., features_extractor_kwargs=...))`. Note: SB3's image preprocessing may expect `uint8` channel-last or channel-first images; set `normalize_images=False` and keep float inputs, or write a small wrapper. The action head is then a flat `nely*nelx` Gaussian.

**8b. Fully convolutional actor-critic (preferred).** A custom `ActorCriticPolicy` (or a handwritten PPO) where:
- Encoder: a small U-Net or stack of dilated convolutions, keeping spatial size `(nely, nelx)`.
- **Actor head:** `Conv2d → (1, nely, nelx)` giving the action **mean map** (use `tanh`); `log_std` is a learnable scalar or per-channel parameter.
- **Critic head:** global average pooling of encoder features → MLP → scalar value.
- This makes the policy independent of grid size and preserves locality.

**Tests:** output shapes for several grid sizes; gradients flow to all parameters; a forward pass on a batch of observations is fast (less than the FEA time per step).

---

## 9. `train.py`: training script

**Logic**
1. Load YAML config (grid size, rmin, volfrac, delta_max, max_steps, reward weights, PPO hyperparameters, load-sampler ranges, seeds, total timesteps).
2. Build vectorized envs: `SubprocVecEnv` (FEA is CPU-bound), each with its own seed; wrap with `VecMonitor`; optionally `VecNormalize(reward=True, obs=False)`.
3. Create PPO with the policy from Section 8. Suggested start: `n_steps = 256..512`, `batch_size = 64..128`, `gamma = 0.99`, `gae_lambda = 0.95`, `clip_range = 0.2`, `ent_coef = 0.0..0.01`, `learning_rate = 3e-4`.
4. Callbacks: periodic checkpoint; **evaluation callback on a held-out validation set** of load cases, logging compliance ratio vs SIMP; early stop if no improvement.
5. Log to TensorBoard (and optionally Weights & Biases): episode reward, final compliance ratio, volume error, grayness.
6. Save the config, git commit hash and library versions beside the checkpoint.
7. CLI: `python train.py --config configs/exp.yaml --seed 0 --out runs/exp_seed0`.

**Curriculum option:** a wrapper that widens the load-angle/position range once rolling success rate exceeds a threshold.

---

## 10. `evaluate.py`: benchmark and metrics

**Logic**
1. Load the saved test set (Section 4), the trained checkpoint, and the cached SIMP results (compute with Section 6 once and store as `simp_results.json`).
2. For each test case: run the policy deterministically for `K` steps (or one pass for a single-step design), final `x_phys`.
3. Recompute the final compliance with a **fresh** FEA call (do not trust the env's cached value).
4. Metrics per case:
   - `compliance_ratio = C_RL / C_SIMP`
   - `volume_error = |V - volfrac| / volfrac`
   - `grayness = mean((x > 0.1) & (x < 0.9))` (fraction of elements; or `4*x*(1-x)` mean)
   - `connected_components` of the solid phase (`scipy.ndimage.label`, threshold 0.5), and whether load and supports are in the same component
   - `n_fea_calls`, `wall_time_s` for both RL and SIMP
   - `success = (volume_error < tol_v) & (compliance_ratio < tol_c)`
5. Aggregate: mean, median, std, worst case, success rate, for in-distribution vs out-of-distribution cases.
6. Write `results/metrics.csv` and a summary table.

**Baselines to include** (each as a function with the same interface): `uniform_density`, `random_actions`, `strain_energy_heuristic` (remove lowest-energy elements until volume is met), and optionally the supervised CNN (Section 11).

---

## 11. `supervised/` (optional but strongly recommended baseline)

- `make_dataset.py`: sample N load cases, run SIMP, save `(input_channels, x_final)` pairs.
- `model.py`: small U-Net, input = same channels as the RL observation at the initial density, output = density map.
- `train_sup.py`: loss = MSE (+ volume penalty `(mean(x_pred) - volfrac)²`); output passes through a sigmoid; evaluate with the same `evaluate.py` metrics.

---

## 12. `viz.py`: plots and animation

- `plot_density(x, title, ax)`: `imshow(1 - x, cmap="gray", vmin=0, vmax=1)` with load and support markers.
- `plot_comparison(case, x_simp, x_rl)`: side by side with compliance in the titles.
- `animate_episode(env, policy, path)`: save the density every step to a GIF/MP4 (`matplotlib.animation` or `imageio`).
- `plot_learning_curves(logdir)`, `plot_compliance_vs_fea(history_simp, history_rl)`, and a metrics bar chart with error bars over seeds.

---

## 13. Build order and acceptance checks

| Order | Module | Done when |
|-------|--------|-----------|
| 1 | element, mesh | Tests in Sections 1 and 2 pass |
| 2 | solver, cases | `c == F @ U`; MBB and cantilever load correctly |
| 3 | filters, simp | FD gradient check passes; MBB and cantilever images match known results; baselines saved |
| 4 | env | `check_env` passes; heuristic policies run end to end |
| 5 | policy (8a), train | PPO beats random/uniform/heuristic on a fixed case, 3 seeds |
| 6 | policy (8b), reward ablations | Fixed-case compliance within a few percent of SIMP |
| 7 | load sampler, curriculum, validation callback | Learning curves on held-out loads |
| 8 | evaluate, baselines, supervised | Full metrics table on the fixed test set |
| 9 | viz | All thesis figures regenerate from one script |

---

## 14. Common mistakes to warn the coding model about

- Mixing row-major and column-major flattening, which silently scrambles element order.
- Wrong sign of the sensitivity (it must be negative; OC takes `sqrt(-dc/dv/lmid)`).
- Forgetting `Emin`, which makes `K` singular for void regions.
- Rebuilding KE, `edofMat` or the filter matrix every step, which makes training far too slow.
- Using the env's cached compliance for evaluation instead of a fresh FEA solve.
- Unnormalized rewards, so PPO diverges when compliance scales differ between cases.
- Evaluating on training load cases, which hides the generalization gap.
