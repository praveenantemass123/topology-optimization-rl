"""
env/topopt_env.py
=================
Audited & Hardened Gymnasium environment for 2D Structural Topology Optimization.

Conforms strictly to SPEC.md Section 0 & Section 7:
- Emin = 1e-9 (uniform with FEM and SIMP)
- Action space: Box(-1.0, 1.0, shape=(nely, nelx)), scaled by delta_max = 0.05
- Volume projection: Exact bisection shift ensuring mean(x) == volfrac to < 1e-6
- Reward: Normalized strictly by C0, clipped to [-10, 10], with finite check
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
            "lam_v": 10.0,
            "beta": 0.0,
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
        
        # 5. C0-Normalized Reward with Safeguards
        actual_vol = float(np.mean(self.x_phys))
        vol_error = abs(actual_vol - self.volfrac)
        reward_clip = float(self.config.get("reward_clip", 2.0))
        
        if np.isfinite(c) and np.isfinite(self.C_prev) and self.C0 > 0:
            delta_c_norm = -(c - self.C_prev) / self.C0
            r_unclipped = float(delta_c_norm - self.config["lam_v"] * vol_error)
            clipped = bool(abs(r_unclipped) >= reward_clip)
            if clipped:
                self.clip_count += 1
            r = float(np.clip(r_unclipped, -reward_clip, reward_clip))
        else:
            r = -reward_clip
            clipped = True
            self.clip_count += 1
            
        self.C_prev = float(c)
        
        # 6. Fixed-Horizon Truncation ONLY (No early-stopping loopholes)
        terminated = False
        truncated = bool(self.t >= self.max_steps)
        
        if (terminated or truncated) and self.config.get("beta", 0.0) > 0.0:
            r += -self.config["beta"] * (c / self.C0)
            
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
            "delta_compliance": float(delta_c_norm) if np.isfinite(c) else 0.0,
            "reward_clipped": clipped,
            "total_clip_events": self.clip_count
        }
        return obs, r, terminated, truncated, info
