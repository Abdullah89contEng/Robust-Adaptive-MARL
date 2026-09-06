"""PGD/FGSM-style self-perturbation: an agent's own position or own action.

Both are distinct from `maal.py`'s teammate-*action* attack, which
perturbs what *other* agents did. These perturb what *this* agent itself
perceives (`perturb_position` — its own position, the same style of
attack SA-MDP and RADIAL-RL use, Ch.5's adversarial-regularization
branch) or does (`perturb_own_action` — its own chosen action, an
actuator-style attack, distinct from MAAL's opponent/teammate-action
attack). Both are training-time robustifications of the execution
critic, the same role MAAL plays for the teammate-action channel.

    s_hat_{u+1} = clip(s_hat_u + beta * sign(grad_{s_hat} L))

`sign()` makes this an L_inf-ball attack (every perturbed coordinate
saturates to +-beta per step, unlike MAAL's L2-ball projection), and
`clip` here means projecting back into the L_inf ball of radius `eps`
around the *original* (unperturbed) value — the standard PGD box
constraint — not clamping to some fixed absolute range. With `n_steps=1`
this is exactly one-step FGSM, matching MAAL's "no separate adversary
network, one closed-form step" design by default; set `n_steps > 1` for
iterative PGD.

Scoping (mirrors invariant 2, method-spec.md §5, extended to this second
channel): only ever apply this to execution-side observations, and never
write the perturbed value to a replay buffer — same reason MAAL's output
never is (method-spec.md §7's data-path guarantee for the context
encoder).
"""

from __future__ import annotations

from typing import Callable

import torch


def pgd_perturb(
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    eps: torch.Tensor | float,
    beta: float,
    n_steps: int = 1,
) -> torch.Tensor:
    """Iterative sign-gradient ascent on `loss_fn`, L_inf-projected each step.

    loss_fn: differentiable map from a candidate `x` to a loss to be
        *maximized* (so the returned perturbation makes `loss_fn` worse
        for whoever's objective it represents), shape (batch,).
    x: (batch, dim) nominal value being attacked; NOT modified in place.
    eps: L_inf radius (scalar or (batch,)) — how far `x_hat` may stray
        from the original `x` in any single coordinate.
    beta: step size per PGD iteration.
    n_steps: number of iterations; 1 = single-step FGSM (default, matches
        MAAL's "one closed-form step" design).

    Returns x_hat, same shape as `x`, detached (a critic-side quantity,
    not something to backprop the outer loss through).
    """
    x0 = x.detach()
    eps_t = eps if torch.is_tensor(eps) else torch.as_tensor(eps, device=x0.device, dtype=x0.dtype)
    if eps_t.dim() > 0:
        eps_t = eps_t.reshape(-1, *([1] * (x0.dim() - 1)))

    x_hat = x0.clone()
    for _ in range(n_steps):
        x_hat = x_hat.detach().requires_grad_(True)
        loss = loss_fn(x_hat)
        (grad,) = torch.autograd.grad(loss.sum(), x_hat)

        x_hat = x_hat.detach() + beta * grad.sign()
        x_hat = torch.clamp(x_hat, min=x0 - eps_t, max=x0 + eps_t)

    return x_hat.detach()


def perturb_position(
    q_fn: Callable[[torch.Tensor], torch.Tensor],
    position: torch.Tensor,
    eps_pos: torch.Tensor | float,
    beta_pos: float,
    n_steps: int = 1,
) -> torch.Tensor:
    """Convenience wrapper: attack an agent's own *position* to minimize `q_fn`.

    `q_fn`: differentiable map from a candidate position to a value to be
        made *worse* (this function maximizes `-q_fn`, i.e. drives `q_fn`
        down — the same "worst case for the agent" framing as MAAL, just
        aimed at the agent's own perceived position instead of a
        teammate's action).
    position: (batch, 2) nominal (x, y) position, e.g. `obs[..., :2]`.
    """
    return pgd_perturb(lambda p: -q_fn(p), position, eps_pos, beta_pos, n_steps)


def perturb_own_action(
    q_fn: Callable[[torch.Tensor], torch.Tensor],
    action: torch.Tensor,
    eps_action: torch.Tensor | float,
    beta_action: float,
    n_steps: int = 1,
) -> torch.Tensor:
    """Convenience wrapper: attack an agent's own *action* to minimize `q_fn`.

    Unlike MAAL (which perturbs teammates' actions, contained to the
    execution stream per invariant 2), this perturbs the acting agent's
    *own* chosen action — e.g. an actuator fault/attack downstream of the
    policy's decision, rather than another agent's behavior or this
    agent's own perception. Same L_inf, sign-gradient mechanism as
    `perturb_position`; can be composed with a preceding position attack
    by passing in an already-position-attacked action as `action`.

    `q_fn`: differentiable map from a candidate action to a value to be
        made *worse* (maximizes `-q_fn`, same worst-case framing as MAAL
        and `perturb_position`).
    action: (batch, action_dim) nominal action, e.g. the policy's sampled output.
    """
    return pgd_perturb(lambda a: -q_fn(a), action, eps_action, beta_action, n_steps)
