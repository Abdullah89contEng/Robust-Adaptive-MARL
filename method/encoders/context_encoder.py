"""Exchangeable-process (BRUNO-style) per-agent context posterior.

method-spec.md §3. Within a regime, a per-agent stream of codes
``{c_i,t}`` (each in R^d, produced by the invertible flow ``f_phi`` from a
transition) is modeled as an exchangeable Gaussian process:
``Sigma_tt = nu``, ``Sigma_tt' = kappa`` for ``t != t'``. By de Finetti this
is equivalent to ``c_i,t`` being conditionally i.i.d. given a latent
``z_i`` whose posterior admits an exact O(1) recursive conjugate update —
no attention over history, no growing state.

This module implements only the recursion itself (state in, code out,
state out) plus the closed-form one-step information gain it makes
possible (§4). It is deliberately independent of the flow ``f_phi``: any
module producing a code ``c_i,t`` in R^d can drive this posterior.

Everything here is batched over an arbitrary leading shape (e.g.
``(batch_dim, n_agents)``), since every parallel VMAS environment tracks
every agent's posterior independently, and different (env, agent) pairs
can be at different counts `t` — a per-agent, ground-truth or
detector-triggered reset (§3, §10) only ever resets the entries it
applies to.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PosteriorState:
    """The context half of the belief object `b_i` (spec §2), batched.

    mu:     (..., code_dim)  posterior mean of z_i
    sigma2: (...,)            posterior variance (isotropic: Sigma = sigma2 * I)
    count:  (...,) long       number of codes absorbed so far (0 at init)
    """

    mu: torch.Tensor
    sigma2: torch.Tensor
    count: torch.Tensor

    def detach(self) -> "PosteriorState":
        return PosteriorState(self.mu.detach(), self.sigma2.detach(), self.count.detach())


class ExchangeablePosterior:
    """The recursive conjugate update of spec §3.

    ``nu`` (marginal variance) and ``kappa`` (cross-covariance) are the
    exchangeable process's fixed hyperparameters, shared across every
    agent and every code dimension — not learned per agent.
    """

    def __init__(self, code_dim: int, nu: float, kappa: float):
        if nu <= 0:
            raise ValueError(f"nu must be positive, got {nu}")
        if not (0 < kappa < nu):
            raise ValueError(f"kappa must be in (0, nu); got kappa={kappa}, nu={nu}")
        self.code_dim = code_dim
        self.nu = nu
        self.kappa = kappa

    def init_state(
        self, batch_shape: torch.Size | tuple[int, ...], device=None, dtype: torch.dtype = torch.float32
    ) -> PosteriorState:
        """mu_i,0 = 0, sigma2_i,0 = kappa, count = 0 (spec §3 init)."""
        mu = torch.zeros((*batch_shape, self.code_dim), device=device, dtype=dtype)
        sigma2 = torch.full(batch_shape, float(self.kappa), device=device, dtype=dtype)
        count = torch.zeros(batch_shape, device=device, dtype=torch.long)
        return PosteriorState(mu=mu, sigma2=sigma2, count=count)

    def _beta(self, new_count: torch.Tensor) -> torch.Tensor:
        """beta_(t-1) = kappa / (nu + kappa * (t - 2)), t = new_count (>= 1)."""
        t = new_count.to(torch.get_default_dtype())
        return self.kappa / (self.nu + self.kappa * (t - 2.0))

    def step(self, state: PosteriorState, code: torch.Tensor) -> PosteriorState:
        """Absorb one new code `c_i,(t-1)` for every batch entry, unconditionally.

        Use `step_where` instead when only some (env, agent) entries in the
        batch actually observed a new code this step (e.g. padded agents).

        NOTE on a spec discrepancy: method-spec.md §3 literally gives
        ``sigma2_t = (1 - beta) * (sigma2_(t-1) - nu + kappa)``. That is
        inconsistent with its own stated init ``sigma2_0 = kappa`` — it
        drives the variance negative on the very first update for any
        valid ``nu > kappa > 0`` (verified below and in tests). Deriving
        the same recursion from bruno-sac's `nn_gp_layer.py` (a working
        reference implementation of this exact exchangeable-Gaussian
        update) and converting its internal state — which starts at `nu`
        and is only exposed as `internal_sigma - (nu - kappa)` — into the
        "exposed variance starting at kappa" convention this spec uses
        gives a clean multiplicative decay instead:
        ``sigma2_t = (1 - beta) * sigma2_(t-1)``. That is what's
        implemented here. This should be checked against `ch-proposed.tex`
        (the declared source of truth) directly.
        """
        new_count = state.count + 1
        beta = self._beta(new_count)  # (...,)

        mu_new = (1.0 - beta).unsqueeze(-1) * state.mu + beta.unsqueeze(-1) * code
        sigma2_new = (1.0 - beta) * state.sigma2

        return PosteriorState(mu=mu_new, sigma2=sigma2_new, count=new_count)

    def step_where(self, state: PosteriorState, code: torch.Tensor, mask: torch.Tensor) -> PosteriorState:
        """Like `step`, but only where `mask` (bool, shape == state.sigma2.shape) is True."""
        updated = self.step(state, code)
        mask_mu = mask.unsqueeze(-1)
        return PosteriorState(
            mu=torch.where(mask_mu, updated.mu, state.mu),
            sigma2=torch.where(mask, updated.sigma2, state.sigma2),
            count=torch.where(mask, updated.count, state.count),
        )

    def reset_where(self, state: PosteriorState, mask: torch.Tensor) -> PosteriorState:
        """Reset entries selected by `mask` (bool) back to the prior (spec §3, §10).

        Used both for the ground-truth reset at `t == vartheta_i_tilde` in
        Phase 1, and for the detector-triggered reset at meta-test time.
        """
        prior = self.init_state(state.sigma2.shape, device=state.sigma2.device, dtype=state.mu.dtype)
        mask_mu = mask.unsqueeze(-1)
        return PosteriorState(
            mu=torch.where(mask_mu, prior.mu, state.mu),
            sigma2=torch.where(mask, prior.sigma2, state.sigma2),
            count=torch.where(mask, prior.count, state.count),
        )


def information_gain_reward(sigma2_prev: torch.Tensor, sigma2_new: torch.Tensor) -> torch.Tensor:
    """r_aux_i,t = I(z_i; c_i,t | c_i,1:t-1) = 0.5 * log(sigma2_(t-1) / sigma2_t)  (spec §4).

    Exact in closed form because the posterior is exactly Gaussian: the
    one-step information gain of an exchangeable-Gaussian update is just
    the log ratio of variances before/after absorbing the new code.
    """
    return 0.5 * torch.log(sigma2_prev / sigma2_new)
