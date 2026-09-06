"""g_omega(s, {Q_i}) -> Q_tot — the IGM-satisfying mixer (method-spec.md §8).

Standard QMIX-style monotonic mixing network (Rashid et al.): a
hypernetwork produces the mixing weights from the global state `s`, and
weights are passed through `abs()` so `dQ_tot/dQ_i >= 0` for every agent —
the sufficient condition for the Individual-Global-Max property. This is
the same mixing architecture torchrl's `QMixer` implements; a small local
version is used here (rather than importing torchrl's) because that one
is wired for discrete state-action-value tensors in BenchMARL's shape
convention, whereas here `{Q_i}` are already scalar continuous-critic
outputs and `s` is this project's own state representation (the
concatenation of every agent's observation — this env exposes no separate
centralized state).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QMixer(nn.Module):
    def __init__(self, n_agents: int, state_dim: int, mixing_embed_dim: int = 32):
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = mixing_embed_dim

        self.hyper_w1 = nn.Linear(state_dim, n_agents * mixing_embed_dim)
        self.hyper_w2 = nn.Linear(state_dim, mixing_embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, mixing_embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1),
        )

    def forward(self, q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """q_values: (..., n_agents), state: (..., state_dim) -> Q_tot: (...,)"""
        batch_shape = q_values.shape[:-1]

        w1 = self.hyper_w1(state).abs().view(*batch_shape, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(*batch_shape, 1, self.embed_dim)
        hidden = torch.nn.functional.elu(q_values.unsqueeze(-2) @ w1 + b1)  # (..., 1, embed_dim)

        w2 = self.hyper_w2(state).abs().view(*batch_shape, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(*batch_shape, 1, 1)
        q_tot = (hidden @ w2 + b2).squeeze(-1).squeeze(-1)
        return q_tot
