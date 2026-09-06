"""Q_exe_i and the FMASAC mixer Q_tot (method-spec.md §8).

Per-agent critic conditioned on agent i's own belief b_i (spec §6 restates
actor *and* critic conditioning together as "always on the full belief":
`a_i_exe ~ pi_execute_i(.|o_i, b_i)`, `b_i = {(mu_z_i,Sigma_z_i),
(mu_rho,Sigma_rho)}` — used here for the critic too, since that's the
unambiguous statement; §8's `Q_i(tau_i, a_i, b; zeta_i)` with an
un-subscripted `b` is read as shorthand for the same `b_i`, not a
switch to a separate whole-team belief). Evaluated at either the nominal
or the MAAL-perturbed joint action. `Q_tot` is built only from these —
invariant 1 (§5). The mixer g_omega itself lives in `mixer.py` and is
owned by the training loop (one shared mixer, not one per critic).
"""

from __future__ import annotations

import torch

from .qnetwork import TwinQNetwork


class ExecutionCritic(TwinQNetwork):
    def __init__(self, obs_dim: int, action_dim: int, belief_dim_total: int, hidden_dim: int = 128):
        super().__init__(input_dim=obs_dim + action_dim + belief_dim_total, hidden_dim=hidden_dim)

    def q(self, obs: torch.Tensor, action: torch.Tensor, belief_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(torch.cat([obs, action, belief_flat], dim=-1))
