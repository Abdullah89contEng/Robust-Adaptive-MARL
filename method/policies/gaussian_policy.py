"""Shared squashed-Gaussian SAC actor network.

Both `pi_explore_i` and `pi_execute_i` (method-spec.md §2) are SAC-style
stochastic continuous-control policies; they differ only in what they
condition on (exploration: obs only; execution: obs + belief, §6) and in
which critic/temperature they're trained against (§4 vs §8). Rather than
duplicate the network, `ExplorationPolicy`/`ExecutionPolicy` (in this
package) both wrap this one class and just pick their input.
"""

from __future__ import annotations

import torch
import torch.nn as nn

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


class GaussianPolicy(nn.Module):
    """tanh-squashed Gaussian actor, standard SAC parameterization.

    Action is scaled to `[-action_scale, action_scale]`, matching VMAS's
    default per-agent `u_range=1.0`.
    """

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128, action_scale: float = 1.0):
        super().__init__()
        self.action_scale = action_scale
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (action, log_prob, deterministic_action=tanh(mean)*scale)."""
        mean, log_std = self.forward(x)
        std = log_std.exp()
        eps = torch.randn_like(mean)
        pre_tanh = mean + std * eps
        squashed = torch.tanh(pre_tanh)
        action = squashed * self.action_scale

        # log N(pre_tanh; mean, std) - log|d(tanh*scale)/d(pre_tanh)|, standard SAC correction.
        log_prob = (-0.5 * ((pre_tanh - mean) / std) ** 2 - log_std - 0.5 * torch.log(torch.tensor(2 * torch.pi))).sum(
            dim=-1
        )
        log_prob = log_prob - torch.log(self.action_scale * (1 - squashed.pow(2)) + 1e-6).sum(dim=-1)

        deterministic_action = torch.tanh(mean) * self.action_scale
        return action, log_prob, deterministic_action
