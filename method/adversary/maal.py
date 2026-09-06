"""MAAL: one-step projected-gradient teammate-action perturbation (method-spec.md §6).

Approximates the intractable inner min of the robust critic target with a
single gradient step on the *target* critic, then projects back onto the
epsilon_a-ball:

    check_a^-i = Proj_eps_a[ a_hat^-i - alpha_adv * grad_{a^-i} Q_tot_target(...) ]

No adversary network (unlike RARL) — purely a closed-form quantity
computed at update time. Per §9, the result must never be written to any
replay buffer (that's what keeps the context encoder's input adversary-
free, §7) — enforced by the caller, not this function.
"""

from __future__ import annotations

from typing import Callable

import torch


def maal_perturb(
    q_tot_target_fn: Callable[[torch.Tensor], torch.Tensor],
    teammate_actions: torch.Tensor,
    eps_a: torch.Tensor,
    alpha_adv: float,
) -> torch.Tensor:
    """One projected-gradient-descent step minimizing Q_tot_target w.r.t. the
    teammates' joint action, projected onto an L2 ball of radius `eps_a`
    around the nominal `teammate_actions`.

    q_tot_target_fn: differentiable map from a candidate teammate-action
        tensor (same shape as `teammate_actions`) to Q_tot_target, shape (batch,).
    teammate_actions: (batch, n_teammates, action_dim) — a_hat^-i, nominal
        (already-sampled) teammate actions; NOT modified in place.
    eps_a: (batch,) perturbation-ball radius for that batch element (the
        team-wide draw, broadcast to every agent's call).
    alpha_adv: MAAL step size.

    Returns check_a^-i, same shape as `teammate_actions`, detached (this is
    a critic-side quantity, not something to backprop the outer loss through).
    """
    a = teammate_actions.detach().clone().requires_grad_(True)
    q_tot = q_tot_target_fn(a)
    (grad,) = torch.autograd.grad(q_tot.sum(), a)

    candidate = teammate_actions.detach() - alpha_adv * grad
    delta = candidate - teammate_actions.detach()

    flat_delta = delta.flatten(start_dim=1)
    norm = flat_delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = torch.clamp(eps_a.reshape(-1, 1) / norm, max=1.0)
    projected_delta = (flat_delta * scale).view_as(delta)

    return teammate_actions.detach() + projected_delta
