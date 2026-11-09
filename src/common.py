"""
common.py
=========
Shared pieces of the experiment scripts: the benchmark configuration, the
flattened environment used with Stable-Baselines3, hand-written policies,
a single rollout routine and small statistics helpers.

Every script imports from here so that all reported numbers come from the same
environment configuration and the same evaluation code.
"""

import os
import sys
from typing import Any, Callable, Dict, List, Sequence, Tuple

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
from gymnasium import spaces

from env.topopt_env import TopOptEnv
from fem.element import lk
from fem.mesh import build_edofmat, build_sparse_indices
from fem.solver import solve

RESULTS_DIR = os.path.join(REPO_DIR, "results")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")

# Benchmark problem of this semester: 30x10 cantilever, V* = 0.4, 30 design updates.
BASE_CONFIG: Dict[str, Any] = {
    "nelx": 30, "nely": 10, "volfrac": 0.4, "rmin": 1.5,
    "delta_max": 0.05, "max_steps": 30, "project_volume": True,
    "filter_type": "density", "channels": 4,
    "p": 3.0, "E0": 1.0, "Emin": 1e-9,
}

# Reward weightings that are trained and compared.
REWARDS: Dict[str, Dict[str, float]] = {
    # Stage 3/4 reward: r = (C_{t-1} - C_t)/C0 - 10 |V - V*|
    "audited": {"w_stiff": 1.0, "w_weight": 10.0, "w_gray": 0.0, "w_smooth": 0.0,
                "w_term": 0.0, "reward_clip": 2.0},
    # All five factors active.
    "multi":   {"w_stiff": 1.0, "w_weight": 10.0, "w_gray": 0.05, "w_smooth": 0.01,
                "w_term": 1.0, "reward_clip": 2.0},
}

ALPHA = 0.2                    # weight of the learned residual in residual mode
TRAIN_SEEDS = (42, 43, 44)     # PPO training seeds
RANDOM_SEEDS = (0, 1, 2)       # seeds of the random-agent baseline

# PPO hyper-parameters per training mode (unchanged from the earlier runs).
PPO_MODES: Dict[str, Dict[str, Any]] = {
    "standard": {"lr": 3e-4, "log_std_init": -1.5, "timesteps": 50_000, "ckpt": "ppo_gate3"},
    "residual": {"lr": 1e-4, "log_std_init": -2.0, "timesteps": 30_000, "ckpt": "ppo_stage4_residual"},
}


def make_config(reward: str = "audited", residual: bool = False, alpha: float = ALPHA, **overrides) -> Dict[str, Any]:
    """Environment config of the benchmark with the named reward weighting."""
    cfg = dict(BASE_CONFIG, **REWARDS[reward], residual=residual, alpha=alpha)
    cfg.update(overrides)
    return cfg


class FlatObsWrapper(TopOptEnv):
    """TopOptEnv with flattened observation and action vectors (for SB3's MlpPolicy)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._grid_obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(int(np.prod(self._grid_obs_shape)),), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.nely * self.nelx,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return obs.ravel(), info

    def step(self, action):
        obs, r, term, trunc, info = super().step(np.asarray(action).reshape(self.nely, self.nelx))
        return obs.ravel(), r, term, trunc, info


# ------------------------------------------------------------------- policies
Policy = Callable[[np.ndarray, TopOptEnv], np.ndarray]


def zero_policy(obs: np.ndarray, env: TopOptEnv) -> np.ndarray:
    """No policy contribution. In residual mode this applies the base action alone."""
    return np.zeros(env.action_space.shape, dtype=np.float32)


def random_policy(obs: np.ndarray, env: TopOptEnv) -> np.ndarray:
    """Uniform random action from the (seeded) action space."""
    return env.action_space.sample()


def greedy_policy(obs: np.ndarray, env: TopOptEnv) -> np.ndarray:
    """
    Greedy sensitivity heuristic: step along -dC/dx scaled by its largest
    magnitude, so every component lies in [0, 1] (material is only added; the
    volume projection then removes the same amount uniformly).
    """
    g = -env.design_sensitivity()
    return (g / (np.max(np.abs(g)) + 1e-12)).astype(np.float32).reshape(env.action_space.shape)


def model_policy(model) -> Policy:
    """Deterministic (mean) action of a trained SB3 model."""
    return lambda obs, env: model.predict(obs, deterministic=True)[0]


# -------------------------------------------------------------------- rollout
def fresh_compliance(x_phys: np.ndarray, case: Dict[str, Any], p: float = 3.0,
                     E0: float = 1.0, Emin: float = 1e-9) -> float:
    """Compliance of a design from an independent FEA (no cached structures)."""
    edofMat = build_edofmat(case["nelx"], case["nely"])
    iK, jK = build_sparse_indices(edofMat)
    return solve(x_phys, case, lk(nu=0.3), edofMat, iK, jK, p=p, E0=E0, Emin=Emin)[1]


def gray_fraction(x: np.ndarray) -> float:
    """Fraction of elements with intermediate density 0.1 < x < 0.9."""
    return float(np.mean((x > 0.1) & (x < 0.9)))


def rollout(env: TopOptEnv, policy: Policy, seed: int) -> Dict[str, Any]:
    """
    Run one full episode and return its summary. The final compliance is
    re-computed with an independent FEA and must agree with the environment.
    """
    obs, info = env.reset(seed=seed)
    comp_hist: List[float] = [info["compliance"]]
    rewards: List[float] = []
    terms: List[Dict[str, float]] = []
    done = False
    while not done:
        obs, r, terminated, truncated, info = env.step(policy(obs, env))
        comp_hist.append(info["compliance"])
        rewards.append(r)
        terms.append(info["reward_terms"])
        done = terminated or truncated

    c_check = fresh_compliance(env.x_phys, env.current_case, env.p, env.E0, env.Emin)
    if not np.isclose(c_check, info["compliance"], rtol=1e-8):
        raise RuntimeError(f"Environment compliance {info['compliance']} != fresh FEA {c_check}")
    return {
        "compliance": float(info["compliance"]),
        "volume": float(info["volume"]),
        "grayness": gray_fraction(env.x_phys),
        "return": float(np.sum(rewards)),
        "n_fea": int(info["n_fea"]),
        "clip_events": int(sum(abs(r) >= env.reward_clip for r in rewards)),
        "comp_hist": comp_hist,
        "terms": terms,
        "x": env.x_phys.copy(),
    }


# ----------------------------------------------------------------- statistics
def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and SAMPLE standard deviation (ddof = 1); std is 0 for a single value."""
    v = np.asarray(values, dtype=np.float64)
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0
