"""
env/topopt_env.py
=================
Gymnasium environment that casts 2-D minimum-compliance topology optimization
as a finite-horizon MDP.

State        design variable x (nely, nelx), starts uniform at volfrac.
Observation  (C, nely, nelx) float32 in [0, 1]: physical density, strain energy
             density (scaled by its maximum), load mask, support mask.
Action       (nely, nelx) in [-1, 1]; the design changes by delta_max * action
             and is then shifted so that mean(x_phys) = volfrac exactly.
Reward       weighted multi-factor sum, see TopOptEnv.reward_terms.
Episode      exactly max_steps steps (truncation only, never early termination).

FEA cost: 1 solve at reset plus 1 solve per step, so info["n_fea"] = t + 1.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from fem.cases import cantilever_case
from fem.element import lk
from fem.mesh import build_edofmat, build_sparse_indices
from fem.solver import compliance_sensitivity, solve, strain_energy_density
from simp.filters import build_filter, chain_rule_density, density_filter

XMIN = 1e-3  # lower bound of the design variable (same as SIMP)

DEFAULT_CONFIG: Dict[str, Any] = {
    "nelx": 30,
    "nely": 10,
    "volfrac": 0.4,
    "rmin": 1.5,
    "delta_max": 0.05,
    "max_steps": 30,
    "project_volume": True,
    # Weighted multi-factor reward. The defaults reproduce the Stage 3/4 reward
    # r = (C_{t-1} - C_t)/C0 - 10 |V - V*|.
    "w_stiff": 1.0,           # stiffness factor (compliance drop)
    "w_weight": None,         # volume factor; None -> legacy lam_v
    "w_gray": 0.0,            # grayness factor
    "w_smooth": 0.0,          # action-magnitude factor
    "w_term": None,           # terminal factor; None -> legacy beta
    "weight_penalty": "abs",  # "abs": |V - V*|, "sq_rel": ((V - V*)/V*)^2
    "reward_clip": 2.0,
    "lam_v": 10.0,            # legacy alias of w_weight
    "beta": 0.0,              # legacy alias of w_term
    "p": 3.0,
    "E0": 1.0,
    "Emin": 1e-9,
    "filter_type": "density",  # "density" or anything else for no filter
    "channels": 4,
    "residual": False,         # add the physics base action to the policy action
    "alpha": 0.5,              # weight of the policy action in residual mode
}


class TopOptEnv(gym.Env):
    """Topology optimization as a Gymnasium environment (see module docstring)."""

    metadata = {"render_modes": []}

    # Mesh / filter structures shared by all environments of the same size.
    _CACHE: Dict[Tuple[int, int, float], Dict[str, Any]] = {}

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        case_sampler: Optional[Callable[[np.random.Generator], Dict[str, Any]]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.config = dict(DEFAULT_CONFIG)
        if config:
            self.config.update(config)
        cfg = self.config

        self.nelx = int(cfg["nelx"])
        self.nely = int(cfg["nely"])
        self.volfrac = float(cfg["volfrac"])
        self.rmin = float(cfg["rmin"])
        self.delta_max = float(cfg["delta_max"])
        self.max_steps = int(cfg["max_steps"])
        self.num_channels = int(cfg["channels"])
        self.p = float(cfg["p"])
        self.E0 = float(cfg["E0"])
        self.Emin = float(cfg["Emin"])
        self.use_density_filter = cfg.get("filter_type") == "density"
        self.project_volume = bool(cfg.get("project_volume", True))
        self.residual = bool(cfg.get("residual", False))
        self.alpha = float(cfg.get("alpha", 0.5))
        self.reward_clip = float(cfg.get("reward_clip", 2.0))
        if not 1 <= self.num_channels <= 4:
            raise ValueError(f"channels must be between 1 and 4, got {self.num_channels}")

        # Reward weights (new names take precedence over the legacy lam_v / beta).
        w_weight, w_term = cfg.get("w_weight"), cfg.get("w_term")
        self.w_stiff = float(cfg.get("w_stiff", 1.0))
        self.w_weight = float(cfg.get("lam_v", 10.0) if w_weight is None else w_weight)
        self.w_gray = float(cfg.get("w_gray", 0.0))
        self.w_smooth = float(cfg.get("w_smooth", 0.0))
        self.w_term = float(cfg.get("beta", 0.0) if w_term is None else w_term)
        self.weight_penalty = str(cfg.get("weight_penalty", "abs"))
        if self.weight_penalty not in ("abs", "sq_rel"):
            raise ValueError(f"weight_penalty must be 'abs' or 'sq_rel', got {self.weight_penalty!r}")

        self.case_sampler = case_sampler
        self.rng = np.random.default_rng(seed)

        shape = (self.nely, self.nelx)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=shape, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.num_channels, *shape), dtype=np.float32
        )

        key = (self.nelx, self.nely, self.rmin)
        if key not in self._CACHE:
            edofMat = build_edofmat(self.nelx, self.nely)
            iK, jK = build_sparse_indices(edofMat)
            H, Hs = build_filter(self.nelx, self.nely, self.rmin)
            self._CACHE[key] = {"KE": lk(nu=0.3), "edofMat": edofMat, "iK": iK, "jK": jK, "H": H, "Hs": Hs}
        self.fea_cache = self._CACHE[key]

        self.current_case: Optional[Dict[str, Any]] = None
        self.x = np.full(shape, self.volfrac)
        self.x_phys = self.x.copy()
        self.U = np.zeros(2 * (self.nelx + 1) * (self.nely + 1))
        self.ce = np.zeros(shape)
        self.C0 = 1.0
        self.C_prev = 1.0
        self.t = 0
        self.clip_count = 0   # cumulative over the lifetime of the environment
        self.load_mask = np.zeros(shape, dtype=np.float32)
        self.support_mask = np.zeros(shape, dtype=np.float32)

    # ------------------------------------------------------------------ physics
    def _to_physical(self, x: np.ndarray) -> np.ndarray:
        if self.use_density_filter:
            return density_filter(x, self.fea_cache["H"], self.fea_cache["Hs"])
        return x.copy()

    def _fea(self) -> float:
        """Solve for the current x_phys; stores U and ce and returns the compliance."""
        fc = self.fea_cache
        self.U, c, self.ce, _ = solve(
            self.x_phys, self.current_case, fc["KE"], fc["edofMat"], fc["iK"], fc["jK"],
            p=self.p, E0=self.E0, Emin=self.Emin,
        )
        return float(c)

    def design_sensitivity(self) -> np.ndarray:
        """dC/dx of the current design (chain rule through the density filter). Always <= 0."""
        dc = compliance_sensitivity(self.x_phys, self.ce, self.p, self.E0, self.Emin)
        if self.use_density_filter:
            dc, _ = chain_rule_density(dc, np.ones_like(dc), self.fea_cache["H"], self.fea_cache["Hs"])
        return dc

    def base_action(self) -> np.ndarray:
        """
        Physics base action of the residual formulation: the descent direction
        -dC/dx, standardised to zero mean and unit standard deviation over the
        elements and clipped to [-1, 1].
        """
        g = -self.design_sensitivity()
        g = g / (np.max(np.abs(g)) + 1e-12)
        a = (g - np.mean(g)) / (np.std(g) + 1e-12)
        return np.clip(a, -1.0, 1.0)

    def _project_volume(self, x_in: np.ndarray) -> np.ndarray:
        """
        Return clip(x_in + s, XMIN, 1) with the uniform shift s found by bisection
        such that mean(x_phys) = volfrac (error below 1e-9).
        """
        def volume(s: float) -> float:
            return float(np.mean(self._to_physical(np.clip(x_in + s, XMIN, 1.0))))

        lo, hi = -1.0, 1.0     # volume(-1) = XMIN <= volfrac <= 1 = volume(1)
        for _ in range(60):
            s = 0.5 * (lo + hi)
            if volume(s) > self.volfrac:
                hi = s
            else:
                lo = s
        return np.clip(x_in + 0.5 * (lo + hi), XMIN, 1.0)

    # -------------------------------------------------------------- observation
    def _build_spatial_masks(self) -> None:
        """Mark the elements that touch a loaded node / a supported node."""
        nodes_per_elem = self.fea_cache["edofMat"][:, ::2] // 2      # (nel, 4)
        n_nodes = (self.nelx + 1) * (self.nely + 1)

        def mask(dofs: np.ndarray) -> np.ndarray:
            flag = np.zeros(n_nodes, dtype=bool)
            flag[np.asarray(dofs, dtype=np.int64) // 2] = True
            m = flag[nodes_per_elem].any(axis=1)
            return m.reshape((self.nely, self.nelx), order="F").astype(np.float32)

        self.load_mask = mask(np.flatnonzero(self.current_case["F"]))
        self.support_mask = mask(self.current_case["fixed_dofs"])

    def _get_obs(self) -> np.ndarray:
        sed = strain_energy_density(
            self.U, self.x_phys, self.fea_cache["KE"], self.fea_cache["edofMat"],
            p=self.p, E0=self.E0, Emin=self.Emin,
        )
        sed = np.nan_to_num(sed, nan=0.0, posinf=0.0, neginf=0.0)
        max_sed = float(np.max(sed))
        sed = sed / max_sed if max_sed > 1e-12 else np.zeros_like(sed)
        channels = [
            np.clip(self.x_phys, 0.0, 1.0), np.clip(sed, 0.0, 1.0), self.load_mask, self.support_mask
        ]
        return np.stack(channels[: self.num_channels], axis=0).astype(np.float32)

    # ------------------------------------------------------------------- reward
    def reward_terms(
        self, c: float, volume: float, policy_action: np.ndarray, is_last: bool
    ) -> Optional[Dict[str, float]]:
        """
        Weighted multi-factor reward (docs/mdp_specification.md, Section 2.D):

            R_t = w_stiff  * F_stiffness   (C_{t-1} - C_t) / C0
                + w_weight * F_weight      -|V_t - V*|  or  -((V_t - V*)/V*)^2
                + w_gray   * F_grayness    -mean(4 x_phys (1 - x_phys))
                + w_smooth * F_smoothness  -mean(a_t^2) of the raw policy action
                + w_term   * F_terminal    -C_T / C0 on the last step, else 0

        Returns the factors, their weighted contributions and the unclipped
        total, or None if the compliance is not finite.
        """
        if not (np.isfinite(c) and np.isfinite(self.C_prev) and self.C0 > 0):
            return None

        if self.weight_penalty == "sq_rel":
            f_weight = -((volume - self.volfrac) / self.volfrac) ** 2
        else:
            f_weight = -abs(volume - self.volfrac)
        terms = {
            "F_stiffness": float((self.C_prev - c) / self.C0),
            "F_weight": float(f_weight),
            "F_grayness": -float(np.mean(4.0 * self.x_phys * (1.0 - self.x_phys))),
            "F_smoothness": -float(np.mean(np.asarray(policy_action, dtype=np.float64) ** 2)),
            "F_terminal": float(-c / self.C0) if is_last else 0.0,
        }
        weighted = {
            "r_stiffness": self.w_stiff * terms["F_stiffness"],
            "r_weight": self.w_weight * terms["F_weight"],
            "r_grayness": self.w_gray * terms["F_grayness"],
            "r_smoothness": self.w_smooth * terms["F_smoothness"],
            "r_terminal": self.w_term * terms["F_terminal"],
        }
        terms.update(weighted)
        terms["total"] = float(sum(weighted.values()))
        return terms

    # ---------------------------------------------------------------- gym API
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.action_space.seed(seed)

        if self.case_sampler is not None:
            self.current_case = self.case_sampler(self.rng)
        else:
            self.current_case = cantilever_case(self.nelx, self.nely, self.volfrac, load_pos="mid_right")
        self._build_spatial_masks()

        self.x = np.full((self.nely, self.nelx), self.volfrac)
        self.x_phys = self._to_physical(self.x)
        self.C0 = self.C_prev = self._fea()
        self.t = 0

        info = {
            "case_name": self.current_case.get("name", "topopt"),
            "C0": self.C0,
            "compliance": self.C0,
            "volume": float(np.mean(self.x_phys)),
            "volume_fraction": float(np.mean(self.x_phys)),
            "n_fea": 1,
        }
        return self._get_obs(), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.current_case is None:
            raise RuntimeError("Call reset() before step().")
        self.t += 1

        policy_action = np.clip(
            np.asarray(action, dtype=np.float64).reshape(self.nely, self.nelx), -1.0, 1.0
        )
        if self.residual:
            applied = np.clip(self.base_action() + self.alpha * policy_action, -1.0, 1.0)
        else:
            applied = policy_action

        x_new = np.clip(self.x + self.delta_max * applied, XMIN, 1.0)
        if self.project_volume:
            x_new = self._project_volume(x_new)
        self.x = x_new
        self.x_phys = self._to_physical(self.x)
        c = self._fea()

        terminated = False
        truncated = self.t >= self.max_steps

        volume = float(np.mean(self.x_phys))
        terms = self.reward_terms(c, volume, policy_action, is_last=truncated)
        if terms is None:
            # Non-finite compliance (singular stiffness matrix): worst reward.
            reward, clipped, delta_c = -self.reward_clip, True, 0.0
        else:
            clipped = bool(abs(terms["total"]) >= self.reward_clip)
            reward = float(np.clip(terms["total"], -self.reward_clip, self.reward_clip))
            delta_c = terms["F_stiffness"]
        self.clip_count += int(clipped)
        self.C_prev = c

        info = {
            "compliance": c,
            "volume": volume,
            "grayness": float(np.mean((self.x_phys > 0.1) & (self.x_phys < 0.9))),
            "n_fea": self.t + 1,
            "delta_compliance": float(delta_c),
            "reward_clipped": clipped,
            "total_clip_events": self.clip_count,
            "reward_terms": terms if terms is not None else {},
        }
        return self._get_obs(), reward, terminated, bool(truncated), info
