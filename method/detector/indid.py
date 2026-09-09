"""Change-point detector h_xi (thesis-v2.pdf §10.9, Algorithm 10.2).

Backbone: a CAUSAL, LOCAL-WINDOW Transformer encoder (Algorithm 10.2). The
"special mask" is `_causal_local_mask`: position t attends only to
positions [t-window+1, t] -- causal (never the future) and local (never
more than `window` steps back). There is no position embedding; within its
window the detector treats the feature set roughly exchangeably, which is
the right inductive bias for "the recent distribution has shifted" and
removes any train/deploy absolute-position mismatch across detector resets.

Training signal: InDiD's `CPDLoss` (Romanenkova, Stepikin, Morozov,
Zaytsev, "InDiD: Instant Disorder Detection via a Principled Neural
Network", ACM MM'22), ported verbatim from
`third-party/InDiD/utils/loss.py`:

    per sequence:  alpha * L_delay(p[theta : theta+T])  +  beta * L_FA(p[:theta])
    no-change:     beta * L_FA(p[:])
    with alpha = 2 * batch / T, beta = 1, T = len_segment.

Note: InDiD's own backbone is an LSTM seq2seq (its paper appendix and
`utils/core_models.py::BaseRnn`); the Transformer here is this thesis's
architecture choice (Algorithm 10.2). What is taken from InDiD unchanged is
the loss and the evaluation metrics (`calculate_errors`).

Metrics (`find_first_change`, `calculate_errors`) ported verbatim from
`third-party/InDiD/utils/metrics.py`: a prediction firing at ANY step >=
the true change is a TRUE POSITIVE (delay = pred - true, unbounded); a
prediction before the change, or on a no-change sequence, is a false
positive.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CausalLocalWindowTransformer(nn.Module):
    """h_xi: causal, local-window transformer. One shared instance, called
    per agent on that agent's own feature sequence, batched over envs."""

    def __init__(
        self,
        code_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        window: int = 48,
        dim_feedforward: int = 128,
        max_len: int = 512,        # unused (no position embedding); kept for signature compat
        dropout: float = 0.1,
    ):
        super().__init__()
        self.window = window
        # Running input-feature normalization (buffers; frozen at eval).
        self.register_buffer("feat_mean", torch.zeros(code_dim))
        self.register_buffer("feat_var", torch.ones(code_dim))
        self.register_buffer("feat_count", torch.zeros(()))

        self.input_proj = nn.Linear(code_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def _causal_local_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """(seq_len, seq_len) bool, True = blocked (nn.Transformer convention).
        Position t may attend only to [t-window+1, t]."""
        idx = torch.arange(seq_len, device=device)
        q, k = idx.unsqueeze(1), idx.unsqueeze(0)
        future = k > q
        too_far_back = (q - k) >= self.window
        return future | too_far_back

    @torch.no_grad()
    def update_norm(self, x: torch.Tensor) -> None:
        """Fold a batch of feature vectors into the running mean/var
        (parallel/Chan update). `x`: (..., code_dim)."""
        x = x.reshape(-1, x.shape[-1]).float()
        n = x.shape[0]
        if n == 0:
            return
        new_count = self.feat_count + n
        delta = x.mean(0) - self.feat_mean
        self.feat_mean += delta * (n / new_count)
        m_a = self.feat_var * self.feat_count
        m_b = x.var(0, unbiased=False) * n
        self.feat_var = (m_a + m_b + delta ** 2 * (self.feat_count * n / new_count)) / new_count
        self.feat_count = new_count

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (batch, seq_len, code_dim) -> p: (batch, seq_len) in (0, 1)."""
        x = (codes - self.feat_mean) / (self.feat_var.sqrt() + 1e-5)
        x = self.input_proj(x.float())
        mask = self._causal_local_mask(x.shape[1], x.device)
        h = self.transformer(x, mask=mask)
        return torch.sigmoid(self.head(h)).squeeze(-1)


# =========================================================================
# InDiD CPDLoss  --  ported from third-party/InDiD/utils/loss.py
# =========================================================================
def _calculate_delays(prob: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """utils/loss.py :: CPDLoss.calculate_delays_ (1-D `prob`)."""
    device = prob.device
    n = prob.size(0)
    prob_no_change = torch.ones(n, device=device)
    prob_no_change[1:] = prob_no_change[1:] - prob[:-1]
    prob_no_change_before = torch.cumprod(prob_no_change, dim=0)
    delays = torch.arange(1, n + 1, device=device) * prob * prob_no_change_before
    prod_prob_no_change = torch.prod(prob_no_change) * (1.0 - prob[-1])
    return delays, prod_prob_no_change


def _delay_detection_loss(prob: torch.Tensor) -> torch.Tensor:
    """utils/loss.py :: CPDLoss.delay_detection_loss_ (1-D, from the change on)."""
    if prob.numel() == 0:
        return prob.new_zeros(())
    delays, prod_pnc = _calculate_delays(prob)
    return torch.sum(delays) + (len(prob) + 1) * torch.prod(prod_pnc)


def _false_alarms_loss(prob: torch.Tensor) -> torch.Tensor:
    """utils/loss.py :: CPDLoss.false_alarms_loss_ (1-D, on_intervals=False)."""
    if prob.numel() == 0:
        return prob.new_zeros(())
    delays, prod_pnc = _calculate_delays(prob)
    return -torch.sum(delays) - (len(prob) + 1) * torch.prod(prod_pnc)


def cpd_loss(prob: torch.Tensor, true_labels: torch.Tensor, len_segment: int) -> torch.Tensor:
    """utils/loss.py :: CPDLoss.forward.  prob, true_labels: (batch, seq_len);
    `true_labels[i,t]` is 0 before the change and 1 from the change on
    (0 throughout for a no-change sequence)."""
    batch = prob.shape[0]
    alpha = 2.0 * batch / len_segment   # reference: 2 * len(prob) / T ; caller batches ~64
    beta = 1.0
    total = prob.new_zeros(())
    for i in range(batch):
        label = true_labels[i]
        change = torch.nonzero(label != label[0])
        if change.numel() == 0:
            total = total + beta * _false_alarms_loss(prob[i, :])
        else:
            c = int(change[0])
            total = total + alpha * _delay_detection_loss(prob[i, c: c + len_segment])
            total = total + beta * _false_alarms_loss(prob[i, :c])
    return total / batch


def labels_from_switch_time(switch_time: torch.Tensor, seq_len: int) -> torch.Tensor:
    """(batch,) 1-indexed vartheta_tilde (== seq_len -> "no switch") ->
    (batch, seq_len) 0/1 'has switched' labels."""
    t_idx = torch.arange(seq_len, device=switch_time.device)
    sw = torch.where(switch_time < seq_len, switch_time,
                     torch.full_like(switch_time, seq_len + 1))
    return (t_idx.unsqueeze(0) >= (sw.unsqueeze(1) - 1)).long()


def detection_loss(p: torch.Tensor, switch_time: torch.Tensor, len_segment: int) -> torch.Tensor:
    """Adapter for `phase2.py`: build per-timestep labels, call InDiD `cpd_loss`."""
    p = p.clamp(1e-4, 1.0 - 1e-4)
    labels = labels_from_switch_time(switch_time.long(), p.shape[1])
    return cpd_loss(p, labels, len_segment)


# =========================================================================
# InDiD metrics  --  ported from third-party/InDiD/utils/metrics.py
# =========================================================================
def find_first_change(mask: torch.Tensor) -> torch.Tensor:
    """utils/metrics.py :: find_first_change. mask: (batch, seq) bool ->
    (batch,) first True index, or -1 if never."""
    change_ind = torch.argmax(mask.int(), dim=1)
    change_ind[mask.sum(dim=1) == 0] = -1
    return change_ind


def calculate_errors(real: torch.Tensor, pred: torch.Tensor, seq_len: int):
    """utils/metrics.py :: calculate_errors. real, pred: (batch,) first-change
    indices (-1 = none). A prediction at ANY step >= the true change is a TP
    (delay = pred - real). Returns (TN, FP, FN, TP, fp_delay, delay)."""
    fp_delay = torch.zeros_like(real)
    delay = torch.zeros_like(real)
    tn_mask = (real == pred) & (real == -1)
    fn_mask = (real != pred) & (pred == -1)
    tp_mask = (real <= pred) & (real != -1)
    fp_mask = (((real > pred) & (real != -1) & (pred != -1))
               | ((pred != -1) & (real == -1)))
    TN, FN, TP, FP = (int(tn_mask.sum()), int(fn_mask.sum()),
                      int(tp_mask.sum()), int(fp_mask.sum()))
    fp_delay[tn_mask] = seq_len
    fp_delay[fn_mask] = seq_len
    fp_delay[tp_mask] = real[tp_mask]
    fp_delay[fp_mask] = pred[fp_mask]
    delay[fn_mask] = seq_len - real[fn_mask]
    delay[tp_mask] = pred[tp_mask] - real[tp_mask]
    return TN, FP, FN, TP, fp_delay, delay


def f1_score(TN: int, FP: int, FN: int, TP: int) -> float:
    denom = 2 * TP + FN + FP
    return 2.0 * TP / denom if denom else float("nan")

# =========================================================================
# Paper CPD loss  --  arXiv:2510.24988v1 "Enhancing Hierarchical RL through
# Change Point Detection in Time Series", Eq. 7 + Algorithm 1/2.
#
# The paper trains its sigmoid boundary classifier with a plain per-step
# BINARY CROSS-ENTROPY (Eq. 7) on labels that are 1 in a +/-Delta window
# around the change and 0 elsewhere ("pseudo-labels ... smoothed in +/-Delta
# window"), with
#   - label smoothing            y~_t = (1 - eps) y_t + eps/2       (Alg. 1)
#   - near-boundary up-weighting  w_t  = 1 + alpha * 1[t in N]       (Alg. 2)
# There is NO delay / false-alarm decomposition (that is InDiD, above); this
# is the objective used when Phase2Config.detector_loss == "paper".  Our
# setting supplies GROUND-TRUTH switch times, so y_t is the true boundary
# indicator rather than a pseudo-label from intrinsic-signal peaks.
# =========================================================================
def boundary_labels_from_switch_time(
    switch_time: torch.Tensor, seq_len: int, half_width: int = 2
) -> torch.Tensor:
    """(batch,) 1-indexed vartheta_tilde (>= seq_len == "no switch") ->
    (batch, seq_len) float labels: 1.0 for |t - theta| <= half_width,
    0.0 elsewhere; all-zero for a no-change sequence.

    theta (0-indexed) == switch_time - 1: `collect_labeled_episode` applies
    mu_2 just before the env.step whose transition is stored at position
    switch_time - 1, so that transition is the first one under the new
    regime (same convention as `labels_from_switch_time`)."""
    device = switch_time.device
    st = switch_time.to(device).long()
    t_idx = torch.arange(seq_len, device=device).unsqueeze(0)     # (1, T), 0-indexed
    theta0 = (st - 1).unsqueeze(1)                                # (B, 1)
    changed = (st < seq_len).unsqueeze(1)                         # (B, 1)
    near = (t_idx - theta0).abs() <= half_width
    return (near & changed).float()


def paper_cpd_loss(
    p: torch.Tensor,
    switch_time: torch.Tensor,
    half_width: int = 2,
    smooth_eps: float = 0.1,
    near_alpha: float = 3.0,
) -> torch.Tensor:
    """arXiv:2510.24988v1 Eq. 7 (+ Alg. 1/2): near-boundary-weighted,
    label-smoothed BCE.  `p`, `switch_time`: (batch, seq_len), (batch,).

        y~_t = (1 - eps) * y_t + eps/2
        w_t  = 1 + near_alpha * y_t         (y_t == 1 exactly on t in N)
        L    = sum_t w_t * BCE(p_t, y~_t) / sum_t w_t
    """
    seq_len = p.shape[1]
    y = boundary_labels_from_switch_time(switch_time, seq_len, half_width)  # (B, T) in {0,1}
    y_smooth = (1.0 - smooth_eps) * y + smooth_eps / 2.0
    p = p.clamp(1e-4, 1.0 - 1e-4)
    bce = -(y_smooth * torch.log(p) + (1.0 - y_smooth) * torch.log(1.0 - p))
    w = 1.0 + near_alpha * y
    return (w * bce).sum() / w.sum()
