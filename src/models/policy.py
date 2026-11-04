"""
models/policy.py
================
Convolutional Neural Network (CNN) feature extractors and Fully-Convolutional
Actor-Critic architectures for Topology Optimization.

Conforms to SPEC.md Section 0 & Section 8.
"""

from typing import Tuple, Dict, Any, Type, Optional
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution


class TopOptCNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Stage 8a: Custom CNN Feature Extractor for Stable-Baselines3.
    Input:  (Batch, C, nely, nelx) float32 tensor
    Output: (Batch, features_dim) embedding vector
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 8)),  # Fixed spatial representation
            nn.Flatten()
        )
        
        # Compute flattened dimension
        self.linear = nn.Sequential(
            nn.Linear(64 * 4 * 8, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class FullyConvolutionalNetwork(nn.Module):
    """
    Stage 8b: Fully-Convolutional Encoder-Decoder for grid-independent topology design.
    Maintains spatial resolution (nely, nelx) from input to action output.
    """
    def __init__(self, in_channels: int = 4, hidden_dim: int = 64):
        super().__init__()
        
        # Spatial feature encoder (preserves H x W via padding=1)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Actor Head: Outputs 1-channel continuous action mean map
        self.actor_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1),
            nn.Tanh()  # Action delta in [-1.0, 1.0]
        )
        
        # Critic Head: Global pooling -> scalar value estimate V(s)
        self.critic_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            action_mean: Tensor of shape (Batch, nely, nelx)
            value: Tensor of shape (Batch, 1)
        """
        features = self.encoder(x)
        action_mean = self.actor_head(features).squeeze(1)  # shape: (Batch, H, W)
        value = self.critic_head(features)                  # shape: (Batch, 1)
        return action_mean, value
