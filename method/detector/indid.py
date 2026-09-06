"""InDiD-style change-point detector h_xi (method-spec.md / thesis-v2.pdf §10.9,
Algorithm 10.2 "Phase 2 -- supervised change-point detection").

h_xi is a single, shared-parameter, per-agent-input transformer over the
*frozen* context-encoder embeddings c_i,1:t, producing p_i,t in [0, 1]:
the detector's running estimate that agent i's regime has already
switched by step t. Supervised directly against the ground-truth switch
time vartheta_i_tilde available in the labeled (simulator-only) episodes
Phase 1 already generates.

InDiD [66] (Romanenkova, Stepikin, Morozov, Zaytsev) is cited by the
thesis as an existing method being reused, not re-derived — the thesis
text (Algorithm 10.2) names two loss terms, L_delay and L_FA, without
giving their formulas ("the constituent mechanisms ... are each taken
from prior work"). What's implemented below is a standard, principled
realization of that delay/false-alarm decomposition (see `delay_loss`/
`false_alarm_loss` docstrings for the exact reasoning), not a transcribed
formula from the InDiD paper itself — check the original paper if exact
fidelity to it specifically matters, as opposed to fidelity to what
Algorithm 10.2 actually specifies (which is only the two-term structure
and the causal/local-window transformer).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CausalLocalWindowTransformer(nn.Module):
    """h_xi: causal, local-window transformer over a code sequence.

    "Causal": position t only ever attends to positions <= t (an
    unresolved regime at t+1 must not influence p_i,t — matches Algorithm
    10.2's "p_i,t <- h_xi(c_i,1:t)").
    "Local-window": position t only attends back `window` steps, not the
    full history — bounds compute, and biases the detector toward recent
    evidence (thesis doesn't give a window size; left as a hyperparameter).

    One instance is shared across all agents (the thesis: "a single
    (shared parameters, per-agent input)" detector) — call it once per
    agent's own code sequence, batched.
    """

    def __init__(
        self,
        code_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        window: int = 16,
        dim_feedforward: int = 128,
        max_len: int = 512,
    ):
        super().__init__()
        self.window = window
        self.input_proj = nn.Linear(code_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def _causal_local_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """(seq_len, seq_len) bool mask, True = blocked (nn.Transformer convention)."""
        idx = torch.arange(seq_len, device=device)
        query, key = idx.unsqueeze(1), idx.unsqueeze(0)
        future = key > query
        too_far_back = (query - key) >= self.window
        return future | too_far_back

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (batch, seq_len, code_dim) -> p: (batch, seq_len) in (0, 1)."""
        batch, seq_len, _ = codes.shape
        if seq_len > self.pos_embedding.shape[1]:
            raise ValueError(f"sequence length {seq_len} exceeds max_len {self.pos_embedding.shape[1]}")
        x = self.input_proj(codes) + self.pos_embedding[:, :seq_len]
        mask = self._causal_local_mask(seq_len, codes.device)
        hidden = self.transformer(x, mask=mask)
        return torch.sigmoid(self.head(hidden)).squeeze(-1)


def _log_survival(p: torch.Tensor) -> torch.Tensor:
    """L[:, t] = sum_{k=0}^{t-1} log(1 - p_k), for t = 0..T (L[:, 0] = 0).

    exp(L[:, t2] - L[:, t1]) = prod_{k=t1}^{t2-1} (1 - p_k) for t2 >= t1 —
    the survival-product building block both InDiD loss terms below are
    made of. Computed once in log-space (a running sum, not a running
    product) so it's numerically stable over a long horizon and fully
    vectorized over the batch, rather than looping per example.
    """
    log_q = torch.log((1.0 - p).clamp(min=1e-8))
    zero = torch.zeros(p.shape[0], 1, device=p.device, dtype=p.dtype)
    return torch.cat([zero, torch.cumsum(log_q, dim=1)], dim=1)


def delay_loss(p: torch.Tensor, switch_time: torch.Tensor) -> torch.Tensor:
    r"""L_delay, InDiD's exact formula (Romanenkova et al. [66]):

        L_delay = (1/N) sum_i [ sum_{t=theta_i}^{T} (t - theta_i) p_t prod_{k=theta_i}^{t-1} (1-p_k)
                                 + (T + 1 - theta_i) prod_{k=theta_i}^{T} (1-p_k) ]

    approximating E[tau - theta]: `p_t * prod_{k=theta}^{t-1}(1-p_k)` is the
    probability the detector *first* fires exactly at t (a discrete
    first-passage/hazard model), so the sum is an expected delay, and the
    closing term assigns the worst-case delay (T+1-theta) to whatever
    probability mass is left over having never fired by the end of the
    window.

    Indices here are 0-based array positions (`p`'s shape is (batch, T)
    with T = the episode length used, `switch_time` is 1-indexed with
    the horizon itself as the "no switch this episode" sentinel — the
    convention already used elsewhere in this codebase, e.g.
    `sample_episode_schedule`'s vartheta_tilde = min(vartheta, H)). So
    `theta0 = switch_time - 1` is the first post-switch array index, and
    the closing coefficient `(T + 1 - theta) `becomes `(T - theta0)` after
    that shift. No-switch episodes contribute 0 here (there is no delay
    to measure); they drive `false_alarm_loss` instead.

    Per-example values, not yet averaged over the batch — matching
    `false_alarm_loss`'s convention (`detection_loss`/callers `.mean()`).
    """
    batch, horizon = p.shape
    switch_time = switch_time.long()
    theta0 = (switch_time - 1).clamp(0, horizon - 1)
    has_switch = (switch_time < horizon).to(p.dtype)

    log_surv = _log_survival(p)  # (batch, horizon + 1)
    log_surv_theta = torch.gather(log_surv, 1, theta0.unsqueeze(1)).squeeze(1)

    t_idx = torch.arange(horizon, device=p.device, dtype=p.dtype).unsqueeze(0)
    theta0_f = theta0.unsqueeze(1).to(p.dtype)
    survival = torch.exp(log_surv[:, :horizon] - log_surv_theta.unsqueeze(1))  # t = 0..T-1
    post_mask = (t_idx >= theta0_f).to(p.dtype)
    running_term = ((t_idx - theta0_f) * p * survival * post_mask).sum(dim=1)

    survival_full = torch.exp(log_surv[:, horizon] - log_surv_theta)
    closing_term = (horizon - theta0.to(p.dtype)) * survival_full

    return (running_term + closing_term) * has_switch


def false_alarm_loss(p: torch.Tensor, switch_time: torch.Tensor) -> torch.Tensor:
    r"""L_FA, corrected from the formula handed to me (see NOTE in the
    function body for why: the pasted version had a `-` where a `+` is
    required for this to be a valid expectation, and empirically rewards
    the opposite of the intended behavior):

        L_FA = -(1/N) sum_i [ sum_{t=0}^{T_i} t p_t prod_{k=0}^{t-1} (1-p_k)
                               + (T_i + 1) prod_{k=0}^{T_i} (1-p_k) ]

    approximating -E[min(tau, T_i+1)] (maximize the expected time of a
    false alarm, i.e. push any premature firing as late as possible, or
    avoid it within the window entirely): the bracketed sum is
    E[min(tau, cap)] under the same first-passage model as `delay_loss`
    — a law-of-total-expectation sum over a partition of outcomes
    (P(tau=t) for each t, plus P(tau > T_i) for "never fires"), which
    must add (both are non-negative probability weights), not subtract.
    The outer `-` is what turns "maximize time-to-false-alarm" into a
    quantity to minimize, matching `delay_loss` (which needs no such
    flip, since a *small* delay is directly what's wanted).

    `T_i` here is the length of the *stationary prefix* being scored —
    `switch_time - 1` (0-indexed, exclusive) for an episode with a
    genuine switch, or the *whole* window for a no-switch episode (every
    step of a stationary episode is a valid false-alarm negative — the
    "stationary negatives" role method-spec.md §2 calls out for
    vartheta_tilde == H episodes). This is what makes `T_i` in this
    formula distinct from `T` in `delay_loss`'s: the two losses score
    disjoint parts of the same episode (`false_alarm_loss` the pre-switch
    prefix, `delay_loss` the post-switch tail), not the same window twice.

    Per-example values, not yet averaged over the batch.
    """
    batch, horizon = p.shape
    switch_time = switch_time.long()
    has_switch = switch_time < horizon
    theta0 = (switch_time - 1).clamp(0, horizon - 1)
    fa_end = torch.where(has_switch, theta0, torch.full_like(theta0, horizon))  # exclusive upper bound

    log_surv = _log_survival(p)  # (batch, horizon + 1)
    survival = torch.exp(log_surv[:, :horizon])  # t = 0..T-1, survival from the start of the episode

    t_idx = torch.arange(horizon, device=p.device, dtype=p.dtype).unsqueeze(0)
    fa_end_f = fa_end.unsqueeze(1).to(p.dtype)
    pre_mask = (t_idx < fa_end_f).to(p.dtype)
    running_term = (t_idx * p * survival * pre_mask).sum(dim=1)

    log_surv_end = torch.gather(log_surv, 1, fa_end.unsqueeze(1)).squeeze(1)
    closing_term = fa_end.to(p.dtype) * torch.exp(log_surv_end)

    # NOTE on a formula correction: the version handed to me had `running_term
    # - closing_term` here (then negated once more outside), i.e. an overall
    # `-(sum - cap*P(escape))`. Verified empirically (see PR discussion) that
    # this rewards firing immediately and penalizes never firing -- exactly
    # backwards. `running_term + closing_term` is `E[min(tau, cap)]`, a law-
    # of-total-expectation sum over a partition of outcomes (P(tau=t) for
    # each t, plus P(tau>cap) for the escape case) -- both non-negative
    # probability weights, so they must add, not subtract, for this to be a
    # valid expectation at all. Negating that sum once (matching this
    # function's docstring, `L_FA = -E[...]`) is what turns "maximize the
    # time-to-false-alarm" into a quantity to minimize.
    return -(running_term + closing_term)


def detection_loss(p: torch.Tensor, switch_time: torch.Tensor, lambda_fa: float) -> torch.Tensor:
    """L = L_delay + c * L_FA for one agent (Algorithm 10.2, line 4, before the sum over agents).

    `p` is clamped to [1e-4, 1 - 1e-4] first. Both loss terms pass `p` through
    `log(1 - p)` (via `_log_survival`) and divide by `p` / `(1 - p)` implicitly
    in the backward pass; as `p -> 1` the gradient of `log(1 - p)` is
    `-1 / (1 - p)`, which reaches ~1e8 against the `clamp(min=1e-8)` floor and
    overflows a single batch's gradient to NaN even while the forward `p`
    itself still looks finite and unsaturated. Clamping to 1e-4 caps that
    gradient magnitude at ~1e4 and removes the isolated-batch NaNs that Adam
    would otherwise absorb permanently. The clamp is symmetric so `L_delay`'s
    `p` factor (needing `p` bounded away from 0) is covered too.
    """
    p = p.clamp(1e-4, 1.0 - 1e-4)
    # Both terms are expected step counts, O(horizon) in magnitude, and
    # `false_alarm_loss` is unbounded below (-> -E[min(tau, cap)] -> -horizon
    # as the detector learns not to fire early), so at raw scale it dominates
    # `delay_loss` and the objective has no floor -- training oscillates
    # instead of converging. Dividing both by the horizon makes them
    # commensurable, O(1), and bounds the combined objective; it only
    # rescales, it does not change which detector minimizes it.
    horizon = p.shape[1]
    d = delay_loss(p, switch_time) / horizon
    fa = false_alarm_loss(p, switch_time) / horizon
    return d + lambda_fa * fa
