"""
env/topopt_env.py
=================
Audited & Hardened Gymnasium environment for 2D Structural Topology Optimization.

Conforms strictly to SPEC.md Section 0 & Section 7:
- Emin = 1e-9 (uniform with FEM and SIMP)
- Action space: Box(-1.0, 1.0, shape=(nely, nelx)), scaled by delta_max = 0.05
- Volume projection: Exact bisection shift ensuring mean(x) == volfrac to < 1e-6
- Reward: weighted multi-factor sum (stiffness, weight, grayness, smoothness,
  terminal), normalized by C0, clipped to [-reward_clip, reward_clip], finite check
- Termination: Fixed-horizon truncation ONLY (no early stopping loopholes)
"""

from typing import Dict, Any, Optional, Callable, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from fem.element import lk
from fem.mesh import build_edofmat, build_sparse_indices
from fem.solver import solve, strain_energy_density
from fem.cases import cantilever_case
from simp.filters import build_filter, density_filter, chain_rule_density


class TopOptEnv(gym.Env):
    """
    Audited Gymnasium Environment for 2D Topology Optimization.
    Observation shape: (C, nely, nelx) float32 in [0, 1]
    Action shape:      (nely, nelx) float32 in [-1.0, 1.0]
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    _CACHE: Dict[Tuple[int, int, float], Dict[str, Any]] = {}

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        case_sampler: Optional[Callable[[np.random.Generator], Dict[str, Any]]] = None,
        seed: Optional[int] = None
    ):
        super().__init__()
        
        self.config = {
            "nelx": 30,
            "nely": 10,
            "volfrac": 0.4,
            "rmin": 1.5,
            "delta_max": 0.05,
            "max_steps": 30,
            "project_volume": True,
            # Weighted multi-factor reward (defaults reproduce the audited Stage 3/4 reward)
            "w_stiff": 1.0,     # stiffness / compliance-drop factor
            "w_weight": None,   # volume factor; None -> falls back to legacy lam_v
            "w_gray": 0.0,      # grayness factor 4x(1-x)
            "w_smooth": 0.0,    # action-churn factor mean(a^2)
            "w_term": None,     # terminal factor; None -> falls back to legacy beta
            "weight_penalty": "abs",  # "abs": |V-V*|, "sq_rel": ((V-V*)/V*)^2
            "reward_clip": 2.0,
            "lam_v": 10.0,      # legacy alias of w_weight
            "beta": 0.0,        # legacy alias of w_term
            "p": 3.0,
            "E0": 1.0,
            "Emin": 1e-9,
            "filter_type": "density",
            "channels": 4,  # density, strain_energy, load_mask, support_mask
            "residual": False,  # If True, blends sensitivity base direction with residual action
            "alpha": 0.5,       # Residual action scaling factor
        }
        if config:
            self.config.update(config)
            
        self.nelx = int(self.config["nelx"])
        self.nely = int(self.config["nely"])
        self.volfrac = float(self.config["volfrac"])
        self.rmin = float(self.config["rmin"])
        self.delta_max = float(self.config["delta_max"])
        self.max_steps = int(self.config["max_steps"])
        self.num_channels = int(self.config["channels"])

        # Resolve reward weights (new names take precedence over legacy lam_v / beta)
        w_weight = self.config.get("w_weight")
        w_term = self.config.get("w_term")
        self.w_stiff = float(self.config.get("w_stiff", 1.0))
        self.w_weight = float(self.config.get("lam_v", 10.0) if w_weight is None else w_weight)
        self.w_gray = float(self.config.get("w_gray", 0.0))
        self.w_smooth = float(self.config.get("w_smooth", 0.0))
        self.w_term = float(self.config.get("beta", 0.0) if w_term is None else w_term)
        self.weight_penalty = str(self.config.get("weight_penalty", "abs"))
        if self.weight_penalty not in ("abs", "sq_rel"):
            raise ValueError(f"weight_penalty must be 'abs' or 'sq_rel', got {self.weight_penalty!r}")
        
        self.case_sampler = case_sampler
        self.rng = np.random.default_rng(seed)
        
        # Action Space: Box(-1.0, 1.0, shape=(nely, nelx))
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.nely, self.nelx),
            dtype=np.float32
        )
        
        # Observation Space: (C, nely, nelx) float32
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_channels, self.nely, self.nelx),
            dtype=np.float32
        )
        
        # Cache static FEA structures
        cache_key = (self.nelx, self.nely, self.rmin)
        if cache_key not in self._CACHE:
            KE = lk(nu=0.3)
            edofMat = build_edofmat(self.nelx, self.nely)
            iK, jK = build_sparse_indices(edofMat)
            H, Hs = build_filter(self.nelx, self.nely, self.rmin)
            self._CACHE[cache_key] = {
                "KE": KE,
                "edofMat": edofMat,
                "iK": iK,
                "jK": jK,
                "H": H,
                "Hs": Hs
            }
            
        self.fea_cache = self._CACHE[cache_key]
        
        self.current_case: Optional[Dict[str, Any]] = None
        self.x: np.ndarray = np.full((self.nely, self.nelx), self.volfrac, dtype=np.float64)
        self.x_phys: np.ndarray = self.x.copy()
        self.U: np.ndarray = np.zeros(2 * (self.nelx + 1) * (self.nely + 1), dtype=np.float64)
        self.ce: np.ndarray = np.zeros((self.nely, self.nelx), dtype=np.float64)
        self.C0: float = 1.0
        self.C_prev: float = 1.0
        self.t: int = 0
        self.clip_count: int = 0
        self.load_mask: np.ndarray = np.zeros((self.nely, self.nelx), dtype=np.float32)
        self.support_mask: np.ndarray = np.zeros((self.nely, self.nelx), dtype=np.float32)

    def _build_spatial_masks(self):
        nelx, nely = self.nelx, self.nely
        self.load_mask = np.zeros((nely, nelx), dtype=np.float32)
        self.support_mask = np.zeros((nely, nelx), dtype=np.float32)
        
        F = self.current_case["F"]
        loaded_dofs = np.where(F != 0)[0]
        loaded_nodes = set(loaded_dofs // 2)
        
        fixed_dofs = self.current_case["fixed_dofs"]
        fixed_nodes = set(fixed_dofs // 2)
        
        for ix in range(nelx):
            for iy in range(nely):
                n1 = ix * (nely + 1) + iy
                n2 = (ix + 1) * (nely + 1) + iy
                elem_nodes = {n1, n2, n2 + 1, n1 + 1}
                
                if elem_nodes.intersection(loaded_nodes):
                    self.load_mask[iy, ix] = 1.0
                if elem_nodes.intersection(fixed_nodes):
                    self.support_mask[iy, ix] = 1.0

    def _get_obs(self, sed: Optional[np.ndarray] = None) -> np.ndarray:
        if sed is None:
            sed = np.zeros((self.nely, self.nelx), dtype=np.float64)
            
        max_sed = float(np.max(sed))
        sed_norm = (sed / max_sed).astype(np.float32) if max_sed > 1e-12 else np.zeros_like(sed, dtype=np.float32)
        
        channels = [
            self.x_phys.astype(np.float32),
            sed_norm,
            self.load_mask,
            self.support_mask
        ]
        return np.stack(channels[:self.num_channels], axis=0)

    def _project_volume(self, x_in: np.ndarray) -> np.ndarray:
        """Exact bisection shift projection ensuring mean(x_phys) == volfrac to < 1e-6."""
        xmin = 0.001
        target_v = self.volfrac
        use_density_filter = (self.config.get("filter_type") == "density")
        H = self.fea_cache["H"]
        Hs = self.fea_cache["Hs"]
        
        l1 = -2.0
        l2 = 2.0
        
        for _ in range(35):
            s = 0.5 * (l1 + l2)
            x_cand = np.clip(x_in + s, xmin, 1.0)
            if use_density_filter:
                v_mean = float(np.mean(density_filter(x_cand, H, Hs)))
            else:
                v_mean = float(np.mean(x_cand))
                
            if v_mean > target_v:
                l2 = s
            else:
                l1 = s
                
        s_opt = 0.5 * (l1 + l2)
        return np.clip(x_in + s_opt, xmin, 1.0)

    def reward_terms(
        self,
        c: float,
        volume: float,
        policy_action: np.ndarray,
        is_last: bool,
    ) -> Optional[Dict[str, float]]:
        """
        Weighted multi-factor reward (see docs/mdp_specification.md, Section 2.D):

            R_t = w_stiff  * F_stiffness   ( (C_{t-1} - C_t) / C0 )
                + w_weight * F_weight      ( -|V_t - V*|  or  -((V_t - V*)/V*)^2 )
                + w_gray   * F_grayness    ( -mean(4 x_phys (1 - x_phys)) )
                + w_smooth * F_smoothness  ( -mean(a_t^2), raw policy action )
                + w_term   * F_terminal    ( -C_T / C0 at t = T_max, else 0 )

        Returns the individual factors, their weighted contributions and the
        unclipped total, or None if compliance is non-finite.
        """
        if not (np.isfinite(c) and np.isfinite(self.C_prev) and self.C0 > 0):
            return None

        F_stiff = -(c - self.C_prev) / self.C0
        if self.weight_penalty == "sq_rel":
            F_weight = -((volume - self.volfrac) / self.volfrac) ** 2
        else:
            F_weight = -abs(volume - self.volfrac)
        F_gray = -float(np.mean(4.0 * self.x_phys * (1.0 - self.x_phys)))
        F_smooth = -float(np.mean(np.asarray(policy_action, dtype=np.float64) ** 2))
        F_term = -(c / self.C0) if is_last else 0.0

        terms = {
            "F_stiffness": float(F_stiff),
            "F_weight": float(F_weight),
            "F_grayness": float(F_gray),
            "F_smoothness": float(F_smooth),
            "F_terminal": float(F_term),
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

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.action_space.seed(seed)
            
        if self.case_sampler is not None:
            self.current_case = self.case_sampler(self.rng)
        else:
            self.current_case = cantilever_case(
                nelx=self.nelx,
                nely=self.nely,
                volfrac=self.volfrac,
                load_pos="mid_right"
            )
            
        self._build_spatial_masks()
        
        self.x = np.full((self.nely, self.nelx), self.volfrac, dtype=np.float64)
        
        if self.config.get("filter_type") == "density":
            self.x_phys = density_filter(self.x, self.fea_cache["H"], self.fea_cache["Hs"])
        else:
            self.x_phys = self.x.copy()
            
        U, c, ce, _ = solve(
            self.x_phys,
            self.current_case,
            self.fea_cache["KE"],
            self.fea_cache["edofMat"],
            self.fea_cache["iK"],
            self.fea_cache["jK"],
            p=self.config["p"],
            E0=self.config["E0"],
            Emin=self.config["Emin"]
        )
        self.U = U
        self.ce = ce
        self.ce = ce
        self.C0 = float(c)
        self.C_prev = float(c)
        self.t = 0
        
        sed = strain_energy_density(
            self.U, self.x_phys, self.fea_cache["KE"], self.fea_cache["edofMat"],
            p=self.config["p"], E0=self.config["E0"], Emin=self.config["Emin"]
        )
        
        obs = self._get_obs(sed)
        info = {
            "case_name": self.current_case.get("name", "topopt"),
            "C0": self.C0,
            "compliance": self.C0,
            "volume_fraction": float(np.mean(self.x_phys)),
            "n_fea": 1
        }
        return obs, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.t += 1
        
        # 1. Bounded action scaled by delta_max
        action_arr = np.clip(np.array(action, dtype=np.float64), -1.0, 1.0)
        policy_action = action_arr.copy()  # raw policy output, used by the smoothness term
        
        # Physics Residual Policy blend (Stage 4)
        if self.config.get("residual", False):
            p = float(self.config["p"])
            E0 = float(self.config["E0"])
            Emin = float(self.config["Emin"])
            dc = -p * (self.x_phys ** (p - 1.0)) * (E0 - Emin) * self.ce
            dv = np.ones_like(dc)
            dc_filt, _ = chain_rule_density(dc, dv, self.fea_cache["H"], self.fea_cache["Hs"])
            dc_norm = -dc_filt / (np.max(np.abs(dc_filt)) + 1e-12)
            a_base = (dc_norm - np.mean(dc_norm)) / (np.std(dc_norm) + 1e-12)
            a_base = np.clip(a_base, -1.0, 1.0)
            alpha = float(self.config.get("alpha", 0.5))
            action_arr = np.clip(a_base + alpha * action_arr, -1.0, 1.0)
            
        delta = self.delta_max * action_arr
        
        x_new = np.clip(self.x + delta, 0.001, 1.0)
        
        # 2. Strict Bisection Volume Projection
        if self.config.get("project_volume", True):
            x_new = self._project_volume(x_new)
            
        self.x = x_new
        
        # 3. Physical density filtering
        if self.config.get("filter_type") == "density":
            self.x_phys = density_filter(self.x, self.fea_cache["H"], self.fea_cache["Hs"])
        else:
            self.x_phys = self.x.copy()
            
        # 4. FEA Solve
        U, c, ce, _ = solve(
            self.x_phys,
            self.current_case,
            self.fea_cache["KE"],
            self.fea_cache["edofMat"],
            self.fea_cache["iK"],
            self.fea_cache["jK"],
            p=self.config["p"],
            E0=self.config["E0"],
            Emin=self.config["Emin"]
        )
        self.U = U
        self.ce = ce
        
        # 5. Fixed-Horizon Truncation ONLY (No early-stopping loopholes)
        terminated = False
        truncated = bool(self.t >= self.max_steps)

        # 6. Weighted multi-factor reward (C0-normalized, clipped, finite-checked)
        actual_vol = float(np.mean(self.x_phys))
        terms = self.reward_terms(c, actual_vol, policy_action, is_last=(terminated or truncated))
        reward_clip = float(self.config.get("reward_clip", 2.0))

        if terms is not None:
            r_unclipped = terms["total"]
            clipped = bool(abs(r_unclipped) >= reward_clip)
            r = float(np.clip(r_unclipped, -reward_clip, reward_clip))
            delta_c_norm = terms["F_stiffness"]
        else:
            # Non-finite compliance (disconnected structure / singular K)
            r = -reward_clip
            clipped = True
            delta_c_norm = 0.0
        if clipped:
            self.clip_count += 1

        self.C_prev = float(c)

        sed = strain_energy_density(
            self.U, self.x_phys, self.fea_cache["KE"], self.fea_cache["edofMat"],
            p=self.config["p"], E0=self.config["E0"], Emin=self.config["Emin"]
        )
        
        obs = self._get_obs(sed)
        grayness = float(np.mean((self.x_phys > 0.1) & (self.x_phys < 0.9)))
        
        info = {
            "compliance": float(c),
            "volume": actual_vol,
            "grayness": grayness,
            "n_fea": self.t,
            "delta_compliance": float(delta_c_norm),
            "reward_clipped": clipped,
            "total_clip_events": self.clip_count,
            "reward_terms": terms if terms is not None else {},
        }
        return obs, r, terminated, truncated, info
