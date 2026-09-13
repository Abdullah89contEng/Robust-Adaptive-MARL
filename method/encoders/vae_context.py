"""Amortized (VariBAD-style) context encoder -- Option A of ch-proposed.tex's
"Two admissible choices for context inference" section, offered here as a
configurable alternative to the exchangeable-process posterior
(context_encoder.py, Option B, adopted by default as `context_mode="bruno"`).

Motivation. BRUNO assumes every per-step code is an i.i.d. draw from
N(z_i, nu-kappa) around a single slowly-updating mean, with (nu, kappa)
fixed by hand. Diagnostics on this codebase's own training runs (a large,
non-decreasing l_elbo) showed that assumption is too tight: a moving
agent's raw per-step transitions (lidar, ears, position, velocity) vary far
more within one physics regime than a fixed nu-kappa=0.8 band allows, and
BRUNO's posterior variance shrinks with step count regardless of whether
the mean is actually tracking well, which inflates the mismatch further.

This module (`context_mode="vae"`) replaces the fixed noise band with a
LEARNED one: a GRUCell accumulates the trajectory into a hidden state, a
linear head reads off (mu_z, sigma2_z), and `ContextDecoder` (a separate
MLP) predicts its own (mean, variance) for the reconstruction target
`y = (r, o')` from a sampled z -- rather than living inside (nu, kappa).

Trade-off (ch-proposed.tex, Option A "Against"): no closed-form one-step
information gain (no exact Gaussian recursion under composition), no
calibration guarantee, and the usual VAE difficulties (KL balancing,
posterior collapse). `information_gain_reward` in context_encoder.py is
still applied to whichever (sigma2_prev, sigma2_new) pair this encoder
produces -- under "vae" that is an approximate, uncalibrated
variance-reduction proxy, not an exact information gain, exactly the
"entropy- or bound-difference estimate" fallback the spec allows for
Option A.

`VAEContextEncoder` matches `ExchangeablePosterior`'s state-machine
interface exactly (`init_state`, `step`, `step_where`, `reset_where`) so
every call site that drives a per-agent context posterior --
`Phase1Trainer.rollout_iteration`, `MetaTestRunner.step`, the Phase-2
detector's frozen embeddings -- works unchanged regardless of which mode
is configured; only `Phase1Trainer._update_representation`'s
reconstruction term needs to branch on `context_mode`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VAEPosteriorState:
    """The context half of the belief object `b_i`, batched -- same shape
    contract as `context_encoder.PosteriorState` (mu, sigma2), plus the
    GRU hidden state driving them.

    hidden: (..., hidden_dim) GRU hidden state
    mu:     (..., code_dim)   posterior mean of z_i, read off `hidden`
    sigma2: (...,)            posterior variance (isotropic: Sigma = sigma2 * I)
    """

    hidden: torch.Tensor
    mu: torch.Tensor
    sigma2: torch.Tensor

    def detach(self) -> "VAEPosteriorState":
        return VAEPosteriorState(self.hidden.detach(), self.mu.detach(), self.sigma2.detach())


class VAEContextEncoder(nn.Module):
    """q_phi(z_i | tau_i): a GRUCell reads the flow's per-step code and
    accumulates it into a hidden state; a linear head reads off (mu_z,
    sigma2_z) from that hidden state. Where BRUNO's update is an exact,
    parameter-free conjugate recursion, this update is learned end-to-end
    through the reconstruction + KL loss (`vae_elbo_loss` in
    encoder_losses.py), the same way VariBAD's RNN posterior is trained.
    """

    def __init__(self, code_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        self.cell = nn.GRUCell(code_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 2 * code_dim)

    def _read(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_sigma2 = self.head(hidden).chunk(2, dim=-1)
        # Isotropic sigma2 (scalar per batch entry), matching
        # ExchangeablePosterior's convention and belief.flatten_belief's
        # expected shape -- averaged rather than summed so it stays on the
        # same numeric scale as a per-dimension variance.
        sigma2 = F.softplus(log_sigma2).mean(dim=-1) + 1e-6
        return mu, sigma2

    def init_state(
        self, batch_shape: torch.Size | tuple[int, ...], device=None, dtype: torch.dtype = torch.float32
    ) -> VAEPosteriorState:
        hidden = torch.zeros((*batch_shape, self.hidden_dim), device=device, dtype=dtype)
        with torch.no_grad():
            mu, sigma2 = self._read(hidden)
        return VAEPosteriorState(hidden=hidden, mu=mu, sigma2=sigma2)

    def step(self, state: VAEPosteriorState, code: torch.Tensor) -> VAEPosteriorState:
        """Absorb one new code for every batch entry, unconditionally.

        Use `step_where` instead when only some (env, agent) entries in the
        batch actually observed a new code this step.
        """
        batch_shape = state.hidden.shape[:-1]
        flat_hidden = state.hidden.reshape(-1, self.hidden_dim)
        flat_code = code.reshape(-1, self.code_dim)
        new_hidden = self.cell(flat_code, flat_hidden).reshape(*batch_shape, self.hidden_dim)
        mu, sigma2 = self._read(new_hidden)
        return VAEPosteriorState(hidden=new_hidden, mu=mu, sigma2=sigma2)

    def step_where(self, state: VAEPosteriorState, code: torch.Tensor, mask: torch.Tensor) -> VAEPosteriorState:
        """Like `step`, but only where `mask` (bool, shape == state.sigma2.shape) is True."""
        updated = self.step(state, code)
        mask_h = mask.unsqueeze(-1)
        return VAEPosteriorState(
            hidden=torch.where(mask_h, updated.hidden, state.hidden),
            mu=torch.where(mask_h, updated.mu, state.mu),
            sigma2=torch.where(mask, updated.sigma2, state.sigma2),
        )

    def reset_where(self, state: VAEPosteriorState, mask: torch.Tensor) -> VAEPosteriorState:
        """Reset entries selected by `mask` (bool) back to the (zero-hidden-state) prior."""
        prior = self.init_state(state.sigma2.shape, device=state.sigma2.device, dtype=state.mu.dtype)
        mask_h = mask.unsqueeze(-1)
        return VAEPosteriorState(
            hidden=torch.where(mask_h, prior.hidden, state.hidden),
            mu=torch.where(mask_h, prior.mu, state.mu),
            sigma2=torch.where(mask, prior.sigma2, state.sigma2),
        )


class ContextDecoder(nn.Module):
    """p_psi(y | z, x): predicts its own (mean, variance) for the
    reconstruction target `y = (r, o')`, rather than relying on a fixed
    (nu, kappa) noise band (Option B) or the flow's own change-of-variables
    density. `condition = (o, a) = x`, matching the flow's own conditioning
    input.
    """

    def __init__(self, code_dim: int, cond_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(code_dim + cond_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.head = nn.Linear(hidden_dim, 2 * code_dim)

    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(torch.cat([z, condition], dim=-1))
        mean, log_sigma2 = self.head(h).chunk(2, dim=-1)
        sigma2 = F.softplus(log_sigma2) + 1e-6
        return mean, sigma2
