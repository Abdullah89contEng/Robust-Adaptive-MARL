"""Encoder training losses (method-spec.md §3): L_enc = L_ELBO - lambda_CPC * L_CPC.

L_ELBO's reconstruction term uses the flow's own change-of-variables
density rather than a separate decoder network: since f_phi is invertible,
`log p_psi(y | z, x) = log N(f_phi(y|x); mu_z, sigma2_z) + log|det df_phi/dy|`
is an exact likelihood, not an approximation — this is exactly how
bruno-sac's `get_sequence_model_likelihoods` computes its reconstruction
term (`llp_model + jacob`), and it avoids training a second, redundant
network to invert what the flow already inverts.
"""

from __future__ import annotations

import torch


def gaussian_log_prob(code: torch.Tensor, mu: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
    """log N(code; mu, sigma2 * I), summed over the code dimension.

    code, mu: (..., code_dim); sigma2: (...,) isotropic variance.
    """
    sigma2 = sigma2.unsqueeze(-1)
    log_pdf = -0.5 * torch.log(2 * torch.pi * sigma2) - (code - mu) ** 2 / (2 * sigma2)
    return log_pdf.sum(dim=-1)


def elbo_loss(
    code: torch.Tensor,
    log_det_jacobian: torch.Tensor,
    predictive_mu: torch.Tensor,
    predictive_sigma2: torch.Tensor,
    posterior_mu: torch.Tensor,
    posterior_sigma2: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_sigma2: torch.Tensor,
) -> torch.Tensor:
    """L_ELBO = -E_q[log p_psi(y|z,x)] + KL(q(z|tau) || p(z)).

    The reconstruction term scores `code` under the *predictive* (pre-
    update) posterior — (mu_i,(t-1), sigma2_i,(t-1)) — not the posterior
    that already absorbed `code` itself: scoring a point under a
    distribution partly fit from that same point is circular and would
    trivially reward larger `beta` (faster forgetting of history) rather
    than an actually-predictive code. This mirrors bruno-sac's own
    reconstruction term, which evaluates held-out points under a posterior
    fit only from an earlier context (`get_sequence_model_likelihoods`).
    The KL term instead regularizes the fully-updated posterior — (mu_i,t,
    sigma2_i,t), i.e. "given everything up to and including this point" —
    against the prior, which is the standard `q(z|tau)` reading (posterior
    given the whole history so far).

    Both terms are evaluated in closed form (both q and p are diagonal
    Gaussians, and the reconstruction term uses the flow's exact change-
    of-variables density — see module docstring). Returns a scalar per
    batch element (before any further reduction).
    """
    log_p_y_given_z = gaussian_log_prob(code, predictive_mu, predictive_sigma2) + log_det_jacobian
    reconstruction = -log_p_y_given_z

    code_dim = code.shape[-1]
    kl = 0.5 * (
        code_dim * (posterior_sigma2 / prior_sigma2 - 1 - torch.log(posterior_sigma2 / prior_sigma2))
        + ((posterior_mu - prior_mu) ** 2).sum(dim=-1) / prior_sigma2
    )
    return reconstruction + kl


def infonce_cpc_loss(
    z_query: torch.Tensor, z_positive: torch.Tensor, z_negatives: torch.Tensor,
    w: torch.Tensor, temperature: float = 0.1,
) -> torch.Tensor:
    """L_CPC (InfoNCE), spec eq:pm-enc: f(z, z') = z^T W z', W learnable.

    z_query:     (B, d)      online embedding of a same-regime segment
    z_positive:  (B, d)      momentum embedding of a *different* same-regime segment
    z_negatives: (B, K, d)   momentum embeddings of other-regime segments
    w:           (d, d)      learnable bilinear form

    The embeddings are L2-normalized before the bilinear score (standard
    CPC/SimCLR practice, not in the spec's one-line `z^T W z'`): without it
    the encoder minimizes the loss partly by inflating ||z||, which sends
    the score -> +/- inf and made l_cpc blow up to ~200 mid-run. With unit
    z the score is a bounded quadratic form and `temperature` sets the
    logit scale so `lambda_cpc` is meaningful.

    Returns the per-example cross-entropy loss (mean not yet taken).
    """
    zq = torch.nn.functional.normalize(z_query, dim=-1)
    zp = torch.nn.functional.normalize(z_positive, dim=-1)
    zn = torch.nn.functional.normalize(z_negatives, dim=-1)
    pos_score = torch.einsum("bd,de,be->b", zq, w, zp)
    neg_scores = torch.einsum("bd,de,bke->bk", zq, w, zn)
    logits = torch.cat([pos_score.unsqueeze(-1), neg_scores], dim=-1) / temperature
    labels = torch.zeros(zq.shape[0], dtype=torch.long, device=zq.device)
    return torch.nn.functional.cross_entropy(logits, labels, reduction="none")
