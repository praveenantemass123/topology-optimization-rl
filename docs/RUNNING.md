# Running the Code (Windows)

These steps reproduce every result of this semester (SIMP baselines, sanity baselines, PPO runs and the weighted multi-factor reward run) on a Windows machine. Use **PowerShell** or **Command Prompt**, with Python 3.10 or newer.

## 1. Get the code

If you already have the repo in `E:\7th sem\btp`:
```powershell
cd "E:\7th sem\btp"
git fetch origin
git checkout feature/weighted-multifactor-reward
git pull
```
(After PR #1 is merged, use `git checkout main` and `git pull` instead.)

Fresh copy:
```powershell
git clone https://github.com/praveenantemass123/topology-optimization-rl.git
cd topology-optimization-rl
git checkout feature/weighted-multifactor-reward
```

## 2. Create an environment and install dependencies (one time)
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PowerShell blocks `activate`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Command Prompt (`.venv\Scripts\activate.bat`).

`requirements.txt` installs the CPU build of PyTorch, which is enough for this run.

## 3. Run the tests
```powershell
python -m pytest -q
```
Expected: `79 passed`.

## 4. Regenerate everything in one command
```powershell
python src\run_all.py
```
This runs, in order, `generate_baselines.py`, `train.py` (standard and residual), `evaluate_baselines.py`, `run_weighted_reward.py` and `plot_hierarchy.py`. It takes about 15 minutes on 4 CPU cores. `python src\run_all.py --quick` is a short smoke test (its numbers are not meaningful).

Every training run uses a single torch thread and a fixed seed, so repeating a run on the same machine gives the same numbers. Across machines or library versions the PPO numbers can differ in the last digits; the SIMP and heuristic baselines do not depend on torch.

## 5. Run only the weighted-reward experiment
```powershell
python src\run_weighted_reward.py --timesteps 30000 --seeds 42 43 44
```
This takes about 3 minutes per training run (6 runs in parallel over your CPU cores, so roughly 5 to 10 minutes total). Use `--workers 1` if your machine runs out of memory. For a quick smoke test: `python src\run_weighted_reward.py --timesteps 2000 --seeds 42`.

It runs:
- Baselines: uniform design, random agent (3 seeds), greedy sensitivity heuristic, residual base action alone, SIMP after 30 updates (density filter and sensitivity filter).
- Residual PPO (alpha = 0.2), trained with two reward weightings:
  - **audited:** `w_stiff=1, w_weight=10, w_gray=0, w_smooth=0, w_term=0` (the Stage 3/4 reward)
  - **multi-factor:** `w_stiff=1, w_weight=10, w_gray=0.05, w_smooth=0.01, w_term=1.0`

Outputs in `results\weighted_reward\`:

| File | Contents |
| :--- | :--- |
| `results.csv` | Final compliance (mean ± sample std), gray fraction and episode return for every method |
| `per_seed.csv` | Final compliance of every trained seed and its difference to the residual base |
| `reward_terms.csv` | Every reward factor and its weighted contribution, per step, for each trained policy |
| `training_curves.png` | Episode return and final compliance during training |
| `compliance_history.png` | Compliance over the 30-step episode for each method |
| `topologies.png` | Final density fields |
| `checkpoints\` | Trained PPO models |

## 6. Changing the reward weights

The weights that the scripts train with are in the `REWARDS` dictionary in `src/common.py` (`configs/exp_baseline.yaml` documents the same defaults). See `docs/mdp_specification.md` Section 2.D for the formulation.

## Other scripts
```powershell
python src\generate_baselines.py                                         # SIMP references -> results\baselines.csv
python src\train.py --mode standard                                      # Gate 3 PPO, 50000 steps x 3 seeds
python src\train.py --mode residual                                      # Stage 4 residual PPO, 30000 steps x 3 seeds
python src\evaluate_baselines.py                                         # results\sanity_baselines.csv (needs the checkpoints)
python src\plot_hierarchy.py                                             # six-panel figure
```

## How the numbers are counted

- One design update is one change of the density field. An episode has 30 updates.
- FEA solves = updates + 1: one solve for the starting design and one after every update. SIMP is counted the same way, so "SIMP @ 30 updates" and a 30-step episode cost the same 31 solves.
- Every reported compliance is the compliance of the reported design (checked with an independent FEA), and every design meets the volume constraint to better than 1e-8.
- `±` values are sample standard deviations (ddof = 1) over the seeds.
