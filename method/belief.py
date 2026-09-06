"""The belief object b_i (method-spec.md §2, §6) and how it's fed to a network.

b_i = { (mu_z_i, Sigma_z_i), (mu_rho, Sigma_rho) } — always the *whole
distribution*, never a sample (VariBAD-style), so both execution policy
and execution critic condition on this flattened vector rather than on a
drawn z_i/rho.
"""

from __future__ import annotations

import torch


def flatten_belief(mu_z: torch.Tensor, sigma2_z: torch.Tensor, mu_rho: torch.Tensor, sigma2_rho: torch.Tensor) -> torch.Tensor:
    """Concatenate the belief's distribution parameters into one vector.

    mu_z:      (..., code_dim)   context posterior mean
    sigma2_z:  (...,)            context posterior variance (isotropic)
    mu_rho:    (..., rho_dim)    budget posterior mean (team-shared)
    sigma2_rho:(..., rho_dim)    budget posterior variance (diagonal)

    All leading dims must already be broadcast to match (e.g. mu_rho
    expanded to each agent before calling this, since it's team-shared).
    """
    return torch.cat([mu_z, sigma2_z.unsqueeze(-1), mu_rho, sigma2_rho], dim=-1)


def belief_dim(code_dim: int, rho_dim: int) -> int:
    return code_dim + 1 + rho_dim + rho_dim
