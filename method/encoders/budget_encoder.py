"""Team-shared perturbation-budget encoder q_eta / decoder p_eta (method-spec.md §6).

Structured like the context encoder in spirit (infers a latent belief from
execution histories) but architecturally simpler: a single, team-shared,
*amortized* (non-recursive) Gaussian posterior pooled permutation-
invariantly over every agent's execution transitions, rather than a
per-agent O(1) recursive update. The spec doesn't prescribe pooling
architecture beyond "pools execution histories across all agents", so a
mean-pooled set encoder (Zaheer et al. "Deep Sets"-style) is used here as
the simplest permutation-invariant choice.

The decoder reconstructs the perturbation-ball *radius* `eps_a`, not the
realized MAAL displacement (spec's stated rationale, §6) — so p_eta is a
plain non-negative scalar regressor.

Prior `p(rho)`: the spec doesn't pin this down explicitly; N(0, I) is used
here, matching the usual VAE convention (also what the context encoder's
`p(z)` most naturally means in §3's ELBO).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BudgetEncoder(nn.Module):
    """q_eta(x_exe) -> (mu_rho, sigma2_rho), team-shared, permutation-invariant over the input set."""

    def __init__(self, input_dim: int, rho_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.rho_dim = rho_dim
        self.embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.head = nn.Linear(hidden_dim, 2 * rho_dim)

    def forward(self, x_exe: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x_exe: (..., N, input_dim) — N execution (obs, action) pairs pooled over agents and time."""
        pooled = self.embed(x_exe).mean(dim=-2)
        mu, log_sigma2 = self.head(pooled).chunk(2, dim=-1)
        sigma2 = F.softplus(log_sigma2) + 1e-6
        return mu, sigma2

    def sample(self, mu: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
        return mu + torch.sqrt(sigma2) * torch.randn_like(mu)


class BudgetDecoder(nn.Module):
    """p_eta(rho) -> eps_hat, a non-negative scalar (the reconstructed ball radius)."""

    def __init__(self, rho_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(rho_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(rho)).squeeze(-1)


def budget_loss(mu_rho: torch.Tensor, sigma2_rho: torch.Tensor, eps_hat: torch.Tensor, eps_a: torch.Tensor) -> torch.Tensor:
    """L_rho = KL(q_eta(rho|x_exe) || N(0,I)) + (eps_a - eps_hat)^2."""
    kl = 0.5 * (sigma2_rho + mu_rho**2 - 1 - torch.log(sigma2_rho)).sum(dim=-1)
    reconstruction = (eps_a - eps_hat) ** 2
    return kl + reconstruction
