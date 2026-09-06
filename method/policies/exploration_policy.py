"""pi_explore_i(a_i | o_i) — method-spec.md §2, §4. NOT conditioned on belief b_i."""

from __future__ import annotations

import torch

from .gaussian_policy import GaussianPolicy


class ExplorationPolicy(GaussianPolicy):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128, action_scale: float = 1.0):
        super().__init__(input_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim, action_scale=action_scale)

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sample(obs)
