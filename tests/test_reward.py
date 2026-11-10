"""
test_reward.py
==============
Pytest suite for the weighted multi-factor reward in env/topopt_env.py:

    R_t = w_stiff * F_stiffness + w_weight * F_weight + w_gray * F_grayness
        + w_smooth * F_smoothness + w_term * F_terminal,
    clipped to [-reward_clip, reward_clip].
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
import numpy as np
from env.topopt_env import TopOptEnv

SMALL = {"nelx": 12, "nely": 6, "volfrac": 0.4, "max_steps": 6}


def make_env(**overrides):
    cfg = dict(SMALL)
    cfg.update(overrides)
    env = TopOptEnv(config=cfg)
    obs, info = env.reset(seed=0)
    return env, info


def random_actions(env, n, seed=1):
    rng = np.random.default_rng(seed)
    return [rng.uniform(-1, 1, size=(env.nely, env.nelx)).astype(np.float32) for _ in range(n)]


class TestBackwardCompatibility:

    def test_defaults_reproduce_audited_reward(self):
        """Default weights must give r = dC/C0 - lam_v*|V - V*| (the Stage 3/4 reward)."""
        env, info = make_env()
        c_prev = info["compliance"]
        for a in random_actions(env, env.max_steps):
            _, r, _, _, info = env.step(a)
            c = info["compliance"]
            legacy = -(c - c_prev) / env.C0 - 10.0 * abs(info["volume"] - env.volfrac)
            assert r == pytest.approx(float(np.clip(legacy, -2.0, 2.0)), abs=1e-12)
            c_prev = c

    def test_legacy_aliases(self):
        """lam_v / beta still work; w_weight / w_term take precedence when given."""
        env, _ = make_env(lam_v=5.0, beta=0.3)
        assert env.w_weight == 5.0 and env.w_term == 0.3
        env, _ = make_env(lam_v=5.0, w_weight=7.0, beta=0.3, w_term=0.0)
        assert env.w_weight == 7.0 and env.w_term == 0.0

    def test_invalid_weight_penalty_raises(self):
        with pytest.raises(ValueError):
            TopOptEnv(config=dict(SMALL, weight_penalty="cubic"))


class TestIndividualFactors:

    def test_stiffness_factor(self):
        """F_stiffness = (C_{t-1} - C_t) / C0, positive when the design gets stiffer."""
        env, info = make_env(residual=True, alpha=0.0)
        c_prev = info["compliance"]
        _, _, _, _, info = env.step(np.zeros((env.nely, env.nelx), dtype=np.float32))
        F = info["reward_terms"]["F_stiffness"]
        assert F == pytest.approx((c_prev - info["compliance"]) / env.C0, rel=1e-12)
        assert F > 0, "Sensitivity step should make the cantilever stiffer"

    def test_weight_factor_abs_without_projection(self):
        """Without projection, F_weight = -|V - V*|."""
        env, _ = make_env(project_volume=False)
        _, _, _, _, info = env.step(np.ones((env.nely, env.nelx), dtype=np.float32))
        V = info["volume"]
        assert abs(V - env.volfrac) > 0.01
        assert info["reward_terms"]["F_weight"] == pytest.approx(-abs(V - env.volfrac), abs=1e-12)

    def test_weight_factor_sq_rel(self):
        """weight_penalty='sq_rel' gives F_weight = -((V - V*)/V*)^2."""
        env, _ = make_env(project_volume=False, weight_penalty="sq_rel")
        _, _, _, _, info = env.step(-np.ones((env.nely, env.nelx), dtype=np.float32))
        V = info["volume"]
        expected = -((V - env.volfrac) / env.volfrac) ** 2
        assert info["reward_terms"]["F_weight"] == pytest.approx(expected, abs=1e-12)

    def test_weight_factor_vanishes_with_projection(self):
        """With exact volume projection the weight factor is ~0 at every step."""
        env, _ = make_env(project_volume=True)
        for a in random_actions(env, env.max_steps):
            _, _, _, _, info = env.step(a)
            assert abs(info["reward_terms"]["F_weight"]) < 1e-5

    def test_grayness_factor(self):
        """F_grayness = -mean(4 x (1 - x)) over x_phys; 0 for crisp, -1 for all 0.5."""
        env, _ = make_env()
        _, _, _, _, info = env.step(np.zeros((env.nely, env.nelx), dtype=np.float32))
        expected = -np.mean(4.0 * env.x_phys * (1.0 - env.x_phys))
        assert info["reward_terms"]["F_grayness"] == pytest.approx(expected, abs=1e-12)
        # Uniform x = 0.4 everywhere -> 4*0.4*0.6 = 0.96
        assert expected == pytest.approx(-0.96, abs=1e-6)

        # Mathematical properties of the factor itself
        env.x_phys = np.where(np.arange(env.nelx)[None, :] % 2 == 0, 1.0, 0.0) * np.ones((env.nely, 1))
        crisp = env.reward_terms(env.C_prev, 0.5, np.zeros_like(env.x_phys), False)
        assert crisp["F_grayness"] == pytest.approx(0.0, abs=1e-12)
        env.x_phys = np.full((env.nely, env.nelx), 0.5)
        gray = env.reward_terms(env.C_prev, 0.5, np.zeros_like(env.x_phys), False)
        assert gray["F_grayness"] == pytest.approx(-1.0, abs=1e-12)

    def test_smoothness_factor(self):
        """F_smoothness = -mean(a^2): 0 for zero action, -1 for saturated action."""
        env, _ = make_env()
        _, _, _, _, info = env.step(np.zeros((env.nely, env.nelx), dtype=np.float32))
        assert info["reward_terms"]["F_smoothness"] == pytest.approx(0.0, abs=1e-12)
        _, _, _, _, info = env.step(-np.ones((env.nely, env.nelx), dtype=np.float32))
        assert info["reward_terms"]["F_smoothness"] == pytest.approx(-1.0, abs=1e-12)
        a = np.full((env.nely, env.nelx), 0.5, dtype=np.float32)
        _, _, _, _, info = env.step(a)
        assert info["reward_terms"]["F_smoothness"] == pytest.approx(-0.25, abs=1e-7)

    def test_smoothness_uses_raw_policy_action_in_residual_mode(self):
        """In residual mode the churn term penalizes the policy's residual, not the blended update."""
        env, _ = make_env(residual=True, alpha=0.2)
        _, _, _, _, info = env.step(np.zeros((env.nely, env.nelx), dtype=np.float32))
        assert info["reward_terms"]["F_smoothness"] == pytest.approx(0.0, abs=1e-12)

    def test_terminal_factor_only_at_last_step(self):
        """F_terminal = -C_T / C0 on the final step and 0 before it."""
        env, _ = make_env(w_term=1.0)
        for i, a in enumerate(random_actions(env, env.max_steps)):
            _, r, _, truncated, info = env.step(a)
            F_term = info["reward_terms"]["F_terminal"]
            if i < env.max_steps - 1:
                assert not truncated and F_term == 0.0
            else:
                assert truncated
                assert F_term == pytest.approx(-info["compliance"] / env.C0, rel=1e-12)
                assert info["reward_terms"]["r_terminal"] == pytest.approx(F_term, rel=1e-12)


class TestWeightedSum:

    def test_total_is_weighted_sum_of_factors(self):
        """r = clip(sum_i w_i F_i) for arbitrary non-zero weights."""
        w = {"w_stiff": 1.5, "w_weight": 3.0, "w_gray": 0.1, "w_smooth": 0.02, "w_term": 0.7}
        env, _ = make_env(project_volume=False, **w)
        for a in random_actions(env, env.max_steps, seed=7):
            _, r, _, _, info = env.step(a)
            t = info["reward_terms"]
            total = (w["w_stiff"] * t["F_stiffness"] + w["w_weight"] * t["F_weight"]
                     + w["w_gray"] * t["F_grayness"] + w["w_smooth"] * t["F_smoothness"]
                     + w["w_term"] * t["F_terminal"])
            assert t["total"] == pytest.approx(total, abs=1e-12)
            assert r == pytest.approx(float(np.clip(total, -2.0, 2.0)), abs=1e-12)

    def test_zero_weight_disables_term(self):
        """Changing a factor's weight from 0 changes the reward by exactly w * F."""
        a = random_actions(TopOptEnv(config=SMALL), 1)[0]
        env0, _ = make_env(w_gray=0.0)
        _, r0, _, _, _ = env0.step(a)
        env1, _ = make_env(w_gray=0.2)
        _, r1, _, _, info = env1.step(a)
        assert r1 - r0 == pytest.approx(0.2 * info["reward_terms"]["F_grayness"], abs=1e-12)

    def test_reward_clipping(self):
        """Huge weights saturate at +/- reward_clip and are counted as clip events."""
        env, _ = make_env(residual=True, alpha=0.0, w_stiff=1e4, reward_clip=1.5)
        _, r, _, _, info = env.step(np.zeros((env.nely, env.nelx), dtype=np.float32))
        assert r == pytest.approx(1.5)
        assert info["reward_clipped"] is True and info["total_clip_events"] == 1

    def test_non_finite_compliance_returns_none(self):
        """Non-finite compliance yields no factors (env then gives -reward_clip)."""
        env, _ = make_env()
        assert env.reward_terms(float("inf"), 0.4, np.zeros((env.nely, env.nelx)), False) is None
        assert env.reward_terms(float("nan"), 0.4, np.zeros((env.nely, env.nelx)), False) is None
