"""pi_execute_i(a_i | o_i, b_i) — method-spec.md §2, §8. Conditioned on the
*whole* belief distribution, never a sample of it."""

from __future__ import annotations

import torch

from ..belief import belief_dim, flatten_belief
from .gaussian_policy import GaussianPolicy


class ExecutionPolicy(GaussianPolicy):
    def __init__(self, obs_dim: int, code_dim: int, rho_dim: int, action_dim: int, hidden_dim: int = 128, action_scale: float = 1.0):
        input_dim = obs_dim + belief_dim(code_dim, rho_dim)
        super().__init__(input_dim=input_dim, action_dim=action_dim, hidden_dim=hidden_dim, action_scale=action_scale)

    def act(
        self,
        obs: torch.Tensor,
        mu_z: torch.Tensor,
        sigma2_z: torch.Tensor,
        mu_rho: torch.Tensor,
        sigma2_rho: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        belief = flatten_belief(mu_z, sigma2_z, mu_rho, sigma2_rho)
        return self.sample(torch.cat([obs, belief], dim=-1))
