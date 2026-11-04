"""
test_topopt_env.py
==================
Pytest suite for env/topopt_env.py.
Conforms to SPEC.md Section 7.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
import numpy as np
from gymnasium.utils.env_checker import check_env
from env.topopt_env import TopOptEnv


class TestTopOptEnv:
    """Tests for env/topopt_env.py."""

    def test_gym_env_checker(self):
        """Standard Gymnasium environment checker must pass without errors."""
        config = {"nelx": 10, "nely": 5, "max_steps": 10}
        env = TopOptEnv(config=config)
        # check_env raises exceptions if spaces or step/reset signatures violate Gym API
        check_env(env.unwrapped)

    def test_zero_action_preserves_density(self):
        """Applying a zero action must keep densities invariant."""
        config = {"nelx": 12, "nely": 6, "max_steps": 5, "project_volume": True}
        env = TopOptEnv(config=config)
        obs, _ = env.reset(seed=42)
        
        initial_x = env.x.copy()
        zero_action = np.zeros((env.nely, env.nelx), dtype=np.float32)
        
        obs_next, r, _, _, _ = env.step(zero_action)
        assert np.allclose(env.x, initial_x, atol=1e-5), "Zero action modified design densities!"

    def test_zero_compliance_change_reward(self):
        """If compliance does not change and volume is satisfied, reward must be ~0."""
        config = {"nelx": 10, "nely": 5, "project_volume": True, "lam_v": 0.0}
        env = TopOptEnv(config=config)
        env.reset(seed=42)
        
        zero_action = np.zeros((env.nely, env.nelx), dtype=np.float32)
        _, r, _, _, _ = env.step(zero_action)
        assert abs(r) < 1e-4, f"Zero action gave non-zero reward: {r}"

    def test_episode_truncation(self):
        """Episode must truncate at exactly max_steps."""
        max_steps = 15
        env = TopOptEnv(config={"nelx": 10, "nely": 5, "max_steps": max_steps})
        env.reset(seed=42)
        
        step_count = 0
        truncated = False
        while not truncated:
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            step_count += 1
            if step_count > max_steps + 5:
                break
                
        assert step_count == max_steps, f"Episode ended at step {step_count}, expected {max_steps}"
        assert truncated is True

    def test_seed_reproducibility(self):
        """Two resets with identical seed must yield identical observations."""
        config = {"nelx": 12, "nely": 6}
        env = TopOptEnv(config=config)
        
        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=123)
        
        assert np.allclose(obs1, obs2, atol=1e-12), "Identical seeds produced different initial observations!"

    def test_residual_mode_zero_action_stiffness(self):
        """In residual mode, zero residual action must achieve significant stiffness reduction."""
        config = {
            "nelx": 16, "nely": 8, "volfrac": 0.4, "max_steps": 10,
            "project_volume": True, "residual": True, "alpha": 0.5
        }
        env = TopOptEnv(config=config)
        obs, info = env.reset(seed=42)
        initial_c = info["compliance"]
        
        for _ in range(10):
            zero_action = np.zeros((env.nely, env.nelx), dtype=np.float32)
            obs, r, term, trunc, info = env.step(zero_action)
            if term or trunc:
                break
                
        final_c = info["compliance"]
        assert final_c < initial_c * 0.6, f"Residual zero action failed to achieve >40% stiffness reduction: {initial_c} -> {final_c}"
        assert abs(info["volume"] - 0.4) < 1e-5, f"Residual volume constraint violated: {info['volume']}"

    def test_residual_alpha_zero_insensitivity(self):
        """When alpha=0.0, arbitrary actions must yield identical results to zero action."""
        config = {
            "nelx": 10, "nely": 5, "volfrac": 0.4, "max_steps": 5,
            "project_volume": True, "residual": True, "alpha": 0.0
        }
        env1 = TopOptEnv(config=config)
        env1.reset(seed=42)
        _, _, _, _, info1 = env1.step(np.zeros((5, 10), dtype=np.float32))
        
        env2 = TopOptEnv(config=config)
        env2.reset(seed=42)
        _, _, _, _, info2 = env2.step(np.ones((5, 10), dtype=np.float32))
        
        assert np.isclose(info1["compliance"], info2["compliance"], atol=1e-6), "alpha=0.0 did not nullify residual actions!"
