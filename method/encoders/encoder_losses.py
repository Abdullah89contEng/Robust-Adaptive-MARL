"""Encoder training losses (method-spec.md §3): L_enc = L_ELBO + lambda_CPC * L_CPC.

Two ELBO variants live here, one per `context_mode` (`phase1.py`).

`elbo_loss` (Option B / "bruno", the default) scores the reconstruction
term under the flow's own change-of-variables density rather than a
separate decoder network: since f_phi is invertible,
`log p_psi(y | z, x) = log N(f_phi(y|x); mu_z, sigma2_z) + log|det df_phi/dy|`
is an exact likelihood, not an approximation — this is exactly how
bruno-sac's `get_sequence_model_likelihoods` computes its reconstruction
term (`llp_model + jacob`), and it avoids training a second, redundant
network to invert what the flow already inverts. It requires the assumed
per-code noise band (nu, kappa) to actually match how much a transition
varies within one regime — see `vae_elbo_loss` below for the alternative
that does not.

`vae_elbo_loss` (Option A / "vae") scores reconstruction under a learned
decoder (`method/encoders/vae_context.ContextDecoder`) instead, so the
noise band is learned rather than fixed by (nu, kappa).
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


def gaussian_log_prob_diag(x: torch.Tensor, mu: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
    """log N(x; mu, diag(sigma2)), summed over the last dimension.

    x, mu, sigma2: all (..., dim) -- a per-dimension (not isotropic)
    diagonal covariance, unlike `gaussian_log_prob`'s single isotropic
    sigma2. Used by `vae_elbo_loss`'s decoder likelihood: a learned
    decoder should be free to give each observation channel (lidar, ears,
    position, ...) its own noise scale, since those channels have very
    different natural scales. Forcing one shared scalar here would just
    move the fixed-noise-band mismatch `vae_elbo_loss` exists to avoid
    from (nu, kappa) to a single learned number instead of removing it.
    """
    log_pdf = -0.5 * torch.log(2 * torch.pi * sigma2) - (x - mu) ** 2 / (2 * sigma2)
    return log_pdf.sum(dim=-1)


def diag_gaussian_kl(
    posterior_mu: torch.Tensor,
    posterior_sigma2: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_sigma2: torch.Tensor,
) -> torch.Tensor:
    """KL(N(posterior_mu, posterior_sigma2*I) || N(prior_mu, prior_sigma2*I)),
    both isotropic diagonal Gaussians over the last (code) dimension.
    Shared by `elbo_loss` (Option B) and `vae_elbo_loss` (Option A) —
    the two modes differ in how the reconstruction term is scored, not
    in how the posterior is regularized toward the prior.
    """
    code_dim = posterior_mu.shape[-1]
    return 0.5 * (
        code_dim * (posterior_sigma2 / prior_sigma2 - 1 - torch.log(posterior_sigma2 / prior_sigma2))
        + ((posterior_mu - prior_mu) ** 2).sum(dim=-1) / prior_sigma2
    )


def elbo_loss(
    code: torch.Tensor,
    log_det_jacobian: torch.Tensor,
    predictive_mu: torch.Tensor,
    predictive_sigma2: torch.Tensor,
    posterior_mu: torch.Tensor,
    posterior_sigma2: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_sigma2: torch.Tensor,
    kl_weight: float = 1.0,
) -> torch.Tensor:
    """L_ELBO = -E_q[log p_psi(y|z,x)] + kl_weight * KL(q(z|tau) || p(z)).

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

    `kl_weight` mirrors CCM's own treatment of this term: CCM (the
    contrastive-context baseline this encoder is compared against,
    `third-party/baselines-ccm`) is PEARL's information-bottleneck KL
    term with its coefficient taken down to near zero (CLI default
    `kl_lambda=0.0001` against `ce_coeff=1.0`,
    `ccm_launch_experiment.py:217-218`) and, in the vendored reference
    code, the KL branch is never even wired into the optimizer call
    (`kl_lambda`/`ce_coeff` are commented out of the `PEARLSoftActorCritic`
    constructor call, `rlkit/torch/sac/sac.py:22-29,164-165`) — i.e. CCM's
    context encoder is trained almost entirely by the contrastive loss,
    with the prior-regularizing KL term left negligible rather than at
    full (PEARL-style) strength. A full-strength KL here pulls every
    regime's posterior mean toward the same shared `prior_mu`, which
    directly fights `lambda_cpc`'s need for inter-regime separation
    (method-spec.md §3); `kl_weight` lets the caller down-weight it the
    same way CCM does instead of removing the term outright.

    Both terms are evaluated in closed form (both q and p are diagonal
    Gaussians, and the reconstruction term uses the flow's exact change-
    of-variables density — see module docstring). Returns a scalar per
    batch element (before any further reduction).
    """
    log_p_y_given_z = gaussian_log_prob(code, predictive_mu, predictive_sigma2) + log_det_jacobian
    reconstruction = -log_p_y_given_z
    kl = diag_gaussian_kl(posterior_mu, posterior_sigma2, prior_mu, prior_sigma2)
    return reconstruction + kl_weight * kl


def vae_elbo_loss(
    y: torch.Tensor,
    decoder_mean: torch.Tensor,
    decoder_sigma2: torch.Tensor,
    posterior_mu: torch.Tensor,
    posterior_sigma2: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_sigma2: torch.Tensor,
    kl_weight: float = 1.0,
) -> torch.Tensor:
    """L_ELBO for `context_mode="vae"` (Option A):
    -log p_psi(y|z,x) + kl_weight * KL(q(z|tau) || p(z)).

    Unlike `elbo_loss` (Option B), reconstruction is scored under a
    *learned* decoder density (`decoder_mean`, `decoder_sigma2`, from
    `vae_context.ContextDecoder`, evaluated at a `z` sampled from the
    *predictive*, pre-update posterior — same anti-circularity reasoning
    as `elbo_loss`'s docstring) rather than the flow's own change-of-
    variables density under a fixed (nu, kappa) band, so the reconstruction
    noise is whatever the decoder learns rather than a hand-set constant.
    The KL term is identical in form to `elbo_loss`'s and, likewise,
    regularizes the fully-updated (post-update) posterior against the
    prior.
    """
    reconstruction = -gaussian_log_prob_diag(y, decoder_mean, decoder_sigma2)
    kl = diag_gaussian_kl(posterior_mu, posterior_sigma2, prior_mu, prior_sigma2)
    return reconstruction + kl_weight * kl


def infonce_cpc_loss(
    z_query: torch.Tensor, z_positive: torch.Tensor, z_negatives: torch.Tensor,
    w: torch.Tensor, temperature: float = 0.1,
) -> torch.Tensor:
    """L_CPC (InfoNCE), spec eq:pm-enc: f(z, z') = z^T W z', W learnable.

    z_query:     (B, d)      online embedding of a same-regime segment
    z_positive:  (B, d)      stop-gradient embedding of a *different* same-regime segment
    z_negatives: (B, K, d)   stop-gradient embeddings of other-regime segments
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
