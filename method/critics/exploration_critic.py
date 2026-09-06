"""Q_exp_i (method-spec.md §4). Per-agent, independent, no mixer — invariant 1
(§5): exploration critics never feed Q_tot."""

from __future__ import annotations

import torch

from .qnetwork import TwinQNetwork


class ExplorationCritic(TwinQNetwork):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__(input_dim=obs_dim + action_dim, hidden_dim=hidden_dim)

    def q(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(torch.cat([obs, action], dim=-1))
