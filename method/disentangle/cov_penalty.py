"""L_cov: cross-covariance disentanglement penalty (method-spec.md §7).

Cleanup only — the primary defense is the data path (adversary-free flow
input, enforced by never writing MAAL's output to a buffer, see
buffers/replay.py and adversary/maal.py's docstring). This penalizes only
residual *linear* dependence between the posterior *means* of z and rho
(not full joint covariance, not reparameterized samples).
"""

from __future__ import annotations

import torch


def cross_covariance_penalty(mu_z: torch.Tensor, mu_rho: torch.Tensor) -> torch.Tensor:
    """mu_z, mu_rho: (B, *) posterior means at the same step t, over a minibatch of size B.

    C_z_rho = (1/B) * sum_b (mu_z_b - mean(mu_z)) (mu_rho_b - mean(mu_rho))^T
    L_cov   = ||C_z_rho||_F^2
    """
    batch_size = mu_z.shape[0]
    mu_z_centered = mu_z - mu_z.mean(dim=0, keepdim=True)
    mu_rho_centered = mu_rho - mu_rho.mean(dim=0, keepdim=True)
    cross_cov = (mu_z_centered.T @ mu_rho_centered) / batch_size
    return (cross_cov**2).sum()
