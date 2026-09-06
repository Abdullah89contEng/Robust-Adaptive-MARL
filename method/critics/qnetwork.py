"""Per-agent SAC critic Q(x) -> scalar, with the standard twin-Q trick
(two independently-initialized networks, take the min at the TD target
to control overestimation bias). Used for both `Q_exp_i` and `Q_exe_i`
(method-spec.md §4, §8) — they differ only in what's concatenated into
`x` (obs+action vs. obs+action+belief) and in whether their output feeds
a mixer (invariant 1, §5).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TwinQNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.q1 = QNetwork(input_dim, hidden_dim)
        self.q2 = QNetwork(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(x), self.q2(x)

    def min_q(self, x: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(x)
        return torch.minimum(q1, q2)
