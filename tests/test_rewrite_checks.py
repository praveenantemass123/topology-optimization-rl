"""
test_rewrite_checks.py
======================
Checks that pin down the bookkeeping fixed in the code rewrite:
the SIMP result describes the design it returns, the volume constraint is met
exactly, FEA counts are right, and the shared experiment helpers behave.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import pytest

from common import (FlatObsWrapper, fresh_compliance, greedy_policy, make_config, mean_std,
                    random_policy, rollout, zero_policy)
from env.topopt_env import TopOptEnv
from fem.cases import cantilever_case, mbb_case
from simp.filters import build_filter, chain_rule_density, density_filter
from simp.simp import run_simp

SMALL = {"nelx": 12, "nely": 6, "volfrac": 0.4, "max_steps": 6}


class TestSimpBookkeeping:
    @pytest.mark.parametrize("filter_type", ["sensitivity", "density"])
    @pytest.mark.parametrize("make_case", [lambda: cantilever_case(20, 8, 0.4), lambda: mbb_case(20, 8, 0.5)])
    def test_compliance_belongs_to_returned_design(self, filter_type, make_case):
        case = make_case()
        res = run_simp(case, filter_type=filter_type, max_iter=15, tol=0.0)
        assert res["compliance"] == pytest.approx(fresh_compliance(res["x_phys"], case), rel=1e-10)
        assert res["history_c"][-1] == res["compliance"]

    @pytest.mark.parametrize("filter_type", ["sensitivity", "density"])
    def test_volume_constraint_is_exact(self, filter_type):
        case = cantilever_case(30, 10, 0.4)
        res = run_simp(case, filter_type=filter_type, max_iter=30, tol=0.0)
        assert abs(res["volume"] - 0.4) < 1e-8
        assert res["x_phys"].min() >= 0.0 and res["x_phys"].max() <= 1.0 + 1e-12

    def test_update_and_fea_counts(self):
        res = run_simp(cantilever_case(12, 6, 0.4), max_iter=7, tol=0.0)
        assert res["n_updates"] == 7 and res["n_fea"] == 8
        assert len(res["history_c"]) == 8 and not res["converged"]

    def test_converges_and_flags_it(self):
        res = run_simp(cantilever_case(30, 10, 0.4), max_iter=1000, tol=0.005)
        assert res["converged"] and res["change"] < 0.005
        assert res["compliance"] < 0.25 * res["history_c"][0]

    def test_invalid_filter_raises(self):
        with pytest.raises(ValueError):
            run_simp(cantilever_case(6, 4, 0.4), filter_type="none")


class TestFilterAdjoint:
    def test_chain_rule_is_adjoint_of_density_filter(self):
        """<g, filter(v)> == <chain_rule(g), v> for arbitrary fields."""
        nelx, nely = 9, 5
        H, Hs = build_filter(nelx, nely, 1.5)
        rng = np.random.default_rng(0)
        g, v = rng.normal(size=(nely, nelx)), rng.normal(size=(nely, nelx))
        gt, _ = chain_rule_density(g, np.ones_like(g), H, Hs)
        assert np.sum(g * density_filter(v, H, Hs)) == pytest.approx(np.sum(gt * v), rel=1e-12)

    def test_filter_matrix_is_symmetric(self):
        H, _ = build_filter(7, 4, 2.3)
        assert abs(H - H.T).max() == 0.0


class TestEnvPhysics:
    def test_design_sensitivity_matches_finite_difference(self):
        """dC/dx from the environment (through the density filter) against central differences."""
        env = TopOptEnv(config=SMALL)
        env.reset(seed=0)
        env.step(np.random.default_rng(3).uniform(-1, 1, (env.nely, env.nelx)))
        dc = env.design_sensitivity()
        assert np.all(dc <= 0.0)
        x0, eps = env.x.copy(), 1e-6
        for iy, ix in [(0, 0), (3, 5), (5, 11)]:
            cs = []
            for sgn in (1.0, -1.0):
                x = x0.copy()
                x[iy, ix] += sgn * eps
                cs.append(fresh_compliance(env._to_physical(x), env.current_case))
            fd = (cs[0] - cs[1]) / (2 * eps)
            assert dc[iy, ix] == pytest.approx(fd, rel=1e-5)

    def test_base_action_is_standardised_descent_direction(self):
        env = TopOptEnv(config=dict(SMALL, residual=True, alpha=0.2))
        env.reset(seed=0)
        a = env.base_action()
        assert a.shape == (env.nely, env.nelx) and np.all(np.abs(a) <= 1.0)
        # Larger |dC/dx| must never get a smaller action.
        g = -env.design_sensitivity().ravel()
        order = np.argsort(g)
        assert np.all(np.diff(a.ravel()[order]) >= -1e-12)

    def test_fea_count_and_volume_every_step(self):
        env = TopOptEnv(config=SMALL)
        _, info = env.reset(seed=1)
        assert info["n_fea"] == 1
        for t in range(1, env.max_steps + 1):
            _, _, _, _, info = env.step(env.action_space.sample())
            assert info["n_fea"] == t + 1
            assert abs(info["volume"] - env.volfrac) < 1e-9

    def test_masks_mark_load_and_support_elements(self):
        env = TopOptEnv(config=SMALL)
        env.reset(seed=0)
        assert np.all(env.support_mask[:, 0] == 1.0) and env.support_mask[:, 1:].sum() == 0.0
        # Mid-right load node touches the two right-edge elements around mid-height.
        assert env.load_mask.sum() == 2.0 and env.load_mask[:, -1].sum() == 2.0

    def test_observation_inside_declared_space(self):
        env = TopOptEnv(config=SMALL)
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        obs, *_ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)


class TestExperimentHelpers:
    def test_mean_std_uses_sample_std(self):
        m, s = mean_std([1.0, 2.0, 3.0])
        assert m == pytest.approx(2.0) and s == pytest.approx(1.0)
        assert mean_std([5.0]) == (5.0, 0.0)

    def test_rollout_is_reproducible_and_self_checked(self):
        cfg = make_config("audited", **SMALL)
        a = rollout(TopOptEnv(config=cfg, seed=5), random_policy, 5)
        b = rollout(TopOptEnv(config=cfg, seed=5), random_policy, 5)
        assert a["compliance"] == b["compliance"]
        assert a["n_fea"] == SMALL["max_steps"] + 1 and len(a["comp_hist"]) == a["n_fea"]

    def test_uniform_stays_uniform_and_greedy_improves(self):
        cfg = make_config("audited", **SMALL)
        uni = rollout(TopOptEnv(config=cfg), zero_policy, 0)
        assert np.allclose(uni["x"], SMALL["volfrac"]) and uni["return"] == pytest.approx(0.0, abs=1e-9)
        greedy = rollout(TopOptEnv(config=cfg), greedy_policy, 0)
        assert greedy["compliance"] < uni["compliance"]

    def test_flat_wrapper_matches_grid_env(self):
        cfg = make_config("audited", **SMALL)
        grid, flat = TopOptEnv(config=cfg), FlatObsWrapper(config=cfg)
        og, _ = grid.reset(seed=0)
        of, _ = flat.reset(seed=0)
        assert flat.observation_space.contains(of) and np.array_equal(og.ravel(), of)
        act = np.random.default_rng(0).uniform(-1, 1, (grid.nely, grid.nelx)).astype(np.float32)
        _, rg, _, _, ig = grid.step(act)
        _, rf, _, _, i_f = flat.step(act.ravel())
        assert rg == rf and ig["compliance"] == i_f["compliance"]

    def test_ppo_training_is_deterministic(self):
        from train import train_ppo
        cfg = make_config("audited", residual=True, **SMALL)
        runs = [train_ppo(cfg, seed=7, timesteps=256, lr=1e-4, log_std_init=-2.0) for _ in range(2)]
        assert runs[0]["compliance"] == runs[1]["compliance"]
