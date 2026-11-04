"""
test_policy.py
==============
Pytest suite for models/policy.py.
Conforms to SPEC.md Section 8.
"""

import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
import torch
from gymnasium import spaces
from models.policy import TopOptCNNFeatureExtractor, FullyConvolutionalNetwork


class TestPolicyArchitectures:
    """Tests for models/policy.py."""

    def test_cnn_feature_extractor_shape(self):
        """TopOptCNNFeatureExtractor must output a 1D vector of length features_dim."""
        obs_space = spaces.Box(low=0.0, high=1.0, shape=(4, 10, 30), dtype=float)
        features_dim = 128
        extractor = TopOptCNNFeatureExtractor(obs_space, features_dim=features_dim)
        
        batch_size = 8
        dummy_input = torch.randn(batch_size, 4, 10, 30)
        output = extractor(dummy_input)
        
        assert output.shape == (batch_size, features_dim)

    @pytest.mark.parametrize("nely, nelx", [(10, 30), (20, 60), (16, 32)])
    def test_fcn_spatial_shape_preservation(self, nely, nelx):
        """FullyConvolutionalNetwork must output action maps exactly matching input (nely, nelx)."""
        fcn = FullyConvolutionalNetwork(in_channels=4, hidden_dim=32)
        batch_size = 4
        dummy_input = torch.randn(batch_size, 4, nely, nelx)
        
        action_mean, value = fcn(dummy_input)
        
        assert action_mean.shape == (batch_size, nely, nelx)
        assert value.shape == (batch_size, 1)

    def test_gradient_flow(self):
        """Backpropagation must compute valid non-zero gradients for all trainable parameters."""
        fcn = FullyConvolutionalNetwork(in_channels=4, hidden_dim=32)
        dummy_input = torch.randn(2, 4, 10, 20)
        
        action_mean, value = fcn(dummy_input)
        loss = action_mean.sum() + value.sum()
        loss.backward()
        
        for name, param in fcn.named_parameters():
            assert param.grad is not None, f"Parameter {name} received no gradient!"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients!"

    def test_forward_pass_speed(self):
        """Forward pass on a batch must execute in < 20 ms on CPU."""
        fcn = FullyConvolutionalNetwork(in_channels=4, hidden_dim=32)
        dummy_input = torch.randn(1, 4, 20, 60)
        
        # Warmup
        _ = fcn(dummy_input)
        
        t0 = time.time()
        for _ in range(50):
            _ = fcn(dummy_input)
        elapsed = (time.time() - t0) / 50.0
        
        assert elapsed < 0.02, f"Inference took {elapsed*1000:.2f} ms (expected < 20 ms)"
