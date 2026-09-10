"""Phase 1 — joint context/exploration/execution training (method-spec.md §10).

Wires together every module built so far against the real search-and-
rescue `Scenario`. This is a first, fully-runnable vertical slice, not a
polished/tuned implementation — several points where the spec leaves an
implementation choice unstated are resolved here with a documented,
explicit decision (search "DESIGN CHOICE" below); check those against
`ch-proposed.tex` before trusting results from this code.

DESIGN CHOICE — the adversarial branch here is *not* method-spec.md §6's
MAAL (teammate-action attack); that mechanism (`method/adversary/maal.py`)
is built and tested but intentionally unused in `_update_execution` by
request — teammates 1..n-1 keep their clean, nominal next-actions in the
TD target. Instead, two threat models attack agent 0's *own* channel
(`method/adversary/state_attack.py`, same PGD/FGSM, L_inf mechanism,
chained): its perceived *position* (`perturb_position` — the same
SA-MDP/RADIAL-RL-style observation attack Ch.5 covers for other methods),
then its own *resulting action* (`perturb_own_action` — an actuator-style
attack downstream of the policy's decision). Both are beyond what R1-R7
ask for — added on request, not implied by the spec, and neither is R2's
teammate-action channel — so if `ch-proposed.tex` specifies MAAL as the
adversarial branch, treat what's wired in here as an experimental
substitution, not the method being verified against the thesis text.

DESIGN CHOICE — what's actually executed in the environment each step.
Spec's per-step loop samples *both* `a_i_exp` and `a_i_exe` for every
agent, then says "execute joint action" without saying which. Here, only
`a_i_exp` is ever sent to the environment: exploration is what needs to
happen on-policy to drive context inference (§4), and §9 explicitly says
D_exe "also receives every exploration transition with the intrinsic
reward stripped" — i.e. D_exe's primary inflow during this rollout *is*
the exploration stream. `pi_execute_i` is trained purely off-policy from
buffer samples (standard for SAC), so it never needs to touch the env
during this loop; the reference for `pi_execute_i` actually acting is the
meta-test procedure (§10), once an agent has left EXPLORE mode.

DESIGN CHOICE — both buffers store *joint* per-timestep rows (every
agent's data at that timestep, stacked along an agent axis), not
independent per-agent rows. An earlier version of this file stored flat
per-agent rows and silently lost agent identity (every agent's SAC update
sampled the same agent-unlabeled minibatch) and had no way to assemble the
joint action `Q_tot` actually needs. Joint rows fix both: `batch["obs"][:,
i]` is agent i's data, and `batch["obs"]` as a whole is the joint state
needed for the mixer (and, for the adversarial branch, for evaluating
`Q_tot_target` at a candidate perturbed action/position).

DESIGN CHOICE — CPC segments (§3) are single transitions, not multi-step
segments. An anchor transition's positive is another transition of the
SAME regime mode (from any episode — same-mode transitions are
exchangeable given z, so episode identity is a nuisance factor, not a
constraint); its negatives are K transitions of a DIFFERENT mode, drawn
across the whole minibatch and weighted toward the hardest
(closest-in-(mass,friction)-space) different-mode rows. See
`_sample_cpc_indices`. The regime-mode label rides on each replay row
(`regime_mode`, from the `DEFAULT_MODES` mixture); with the flat-box task
distribution (`modes=()`) the label is -1 and the sampler falls back to a
pure normalized-distance rule (positive within `cpc_pos_tol`, negative
beyond `cpc_neg_guard`). `regime_half` is still stored but no longer
drives CPC.

DESIGN CHOICE — the budget encoder's `x_exe` pooling set (§6) is, for a
given minibatch row, that timestep's own (o_i, a_i) pairs pooled over
agents — a per-timestep pool, not a whole-episode execution history
(reconstructing full-episode grouping from a flat replay sample adds
real complexity, deferred).

`mu_z` (the context posterior mean) is stored exactly at collection time —
`mu_prev`/`sigma2_prev` (pre-update, what the agent actually conditioned
on when it acted) and `mu_new`/`sigma2_new` (post-update, one step later)
both travel with each row, the same way bruno-sac's belief state
(`current_bruno_state`) travels alongside each stored transition in its
own replay buffer (`sac_bruno.py::train`) rather than being reconstructed
after the fact. An earlier version of this file used a zero stand-in for
`mu_z` because it assumed reconstructing the running mean required
replaying the whole recursion from episode start at sample time — it
doesn't; the live value just needed to be captured when it was already
sitting in `PosteriorState.mu` during rollout. The ELBO reconstruction
term scores a code under the *predictive* (pre-update) posterior rather
than the posterior that already absorbed it, to avoid circularity (see
`encoder_losses.elbo_loss`'s docstring); the KL term and `L_cov` use the
fully-updated (post-update) posterior, matching `q(z|tau)`'s usual
"given everything so far" reading.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import vmas

from ..adversary.state_attack import perturb_own_action, perturb_position
from ..belief import belief_dim, flatten_belief
from ..buffers.replay import ReplayBuffer
from ..critics.exploration_critic import ExplorationCritic
from ..critics.execution_critic import ExecutionCritic
from ..critics.mixer import QMixer
from ..disentangle.cov_penalty import cross_covariance_penalty
from ..encoders.budget_encoder import BudgetDecoder, BudgetEncoder, budget_loss
from ..encoders.context_encoder import ExchangeablePosterior, information_gain_reward
from ..encoders.encoder_losses import elbo_loss, infonce_cpc_loss
from ..encoders.flow import ConditionalFlow
from ..policies.execution_policy import ExecutionPolicy
from ..policies.exploration_policy import ExplorationPolicy
from .regime import apply_regime, regime_vector, sample_episode_schedule


@dataclass
class Phase1Config:
    n_envs: int = 8
    horizon: int = 32
    nu: float = 1.0
    kappa: float = 0.2
    rho_dim: int = 4
    mixing_embed_dim: int = 32
    hidden_dim: int = 64
    gamma: float = 0.99
    tau_polyak: float = 0.01
    tau_m: float = 0.99
    alpha_exp_init: float = 0.2
    log_alpha_exp_min: float = -3.0   # floor on log alpha_exp (exp(-3) ~ 0.05); stops entropy collapse
    alpha_exe_init: float = 0.2
    log_alpha_exe_min: float = -3.0   # same floor for the (now auto-tuned) execution temperature
    eps_pos: float = 0.2
    beta_pos: float = 0.05
    pgd_steps: int = 1
    eps_self_action: float = 0.1
    beta_self_action: float = 0.05
    # lambda_cpc was 1.0: with the ELBO term at O(100-1000) the (correctly
    # signed) contrastive CE at O(3) was <1% of the loss, so z never became
    # regime-discriminative. Raised so the two terms are comparable once the
    # ELBO has settled.
    lambda_cpc: float = 10.0
    lambda_cov: float = 1e-3
    p_sw: float = 0.05
    # span the DEFAULT_MODES mixture (icy..heavy); also used to normalize
    # regime distance in the CPC negative sampler.
    mass_range: tuple[float, float] = (0.65, 1.75)
    friction_range: tuple[float, float] = (0.0, 0.45)
    eps_a_range: tuple[float, float] = (0.0, 0.3)
    buffer_capacity: int = 20_000
    batch_size: int = 64
    cpc_pool_size: int = 128
    # CPC negative sampling (see `_sample_cpc_indices`). Negatives are drawn
    # across the whole minibatch, stratified by regime mode, and weighted
    # toward the hardest (closest-in-parameter-space) different-mode rows.
    n_cpc_negatives: int = 16      # K in InfoNCE; the bound on I(z; mode) is ~log(K+1)
    cpc_neg_guard: float = 0.30    # min normalized (mass,friction) distance for a pair to be a valid negative
    cpc_neg_temp: float = 0.25     # exp(-d / temp) negative weighting; smaller -> sample harder negatives
    cpc_pos_tol: float = 0.10      # flat-box fallback only: max normalized distance for a positive
    lr: float = 3e-4


def _make_flow(code_dim: int, cond_dim: int, device: torch.device) -> ConditionalFlow:
    return ConditionalFlow(dim=code_dim, cond_dim=cond_dim, hidden_dim=32).to(device)


class Phase1Trainer:
    def __init__(self, scenario_factory, config: Phase1Config, device: torch.device | None = None):
        self.cfg = config
        self.device = device or torch.device("cpu")

        self.env = vmas.make_env(
            scenario=scenario_factory(),
            num_envs=config.n_envs,
            device=self.device,
            continuous_actions=True,
            wrapper=None,
            max_steps=config.horizon,
        )
        self.n_agents = len(self.env.agents)
        obs = self.env.reset()
        self.obs_dim = obs[0].shape[-1]
        self.action_dim = 2  # VMAS holonomic default

        self.code_dim = 1 + self.obs_dim  # y = (r, o'); flow is dimension-preserving
        self.rho_dim = config.rho_dim
        self.b_dim = belief_dim(self.code_dim, self.rho_dim)
        cond_dim = self.obs_dim + self.action_dim

        self.posterior = ExchangeablePosterior(code_dim=self.code_dim, nu=config.nu, kappa=config.kappa)

        self.flow = _make_flow(self.code_dim, cond_dim, self.device)
        self.flow_momentum = _make_flow(self.code_dim, cond_dim, self.device)
        self.flow_momentum.load_state_dict(self.flow.state_dict())
        for p in self.flow_momentum.parameters():
            p.requires_grad_(False)
        # Identity init (was 0.1*I): with L2-normalized z the bilinear score
        # z^T W z' is a cosine-like quantity in ~[-1, 1]; a 0.1 scale would
        # keep logits near 0 and the InfoNCE softmax stuck at uniform (CE
        # pinned at the log(K+1) chance floor) regardless of separability.
        self.cpc_w = nn.Parameter(torch.eye(self.code_dim, device=self.device))

        self.budget_encoder = BudgetEncoder(cond_dim, self.rho_dim).to(self.device)
        self.budget_decoder = BudgetDecoder(self.rho_dim).to(self.device)

        def make_module_list(factory):
            return nn.ModuleList([factory() for _ in range(self.n_agents)]).to(self.device)

        self.exploration_policies = make_module_list(lambda: ExplorationPolicy(self.obs_dim, self.action_dim, config.hidden_dim))
        self.exploration_critics = make_module_list(lambda: ExplorationCritic(self.obs_dim, self.action_dim, config.hidden_dim))
        self.exploration_critics_target = make_module_list(lambda: ExplorationCritic(self.obs_dim, self.action_dim, config.hidden_dim))
        for tgt, src in zip(self.exploration_critics_target, self.exploration_critics):
            tgt.load_state_dict(src.state_dict())
        self.log_alpha_exp = nn.Parameter(torch.full((self.n_agents,), float(torch.log(torch.tensor(config.alpha_exp_init)))))

        self.execution_policies = make_module_list(
            lambda: ExecutionPolicy(self.obs_dim, self.code_dim, self.rho_dim, self.action_dim, config.hidden_dim)
        )
        self.execution_critics = make_module_list(lambda: ExecutionCritic(self.obs_dim, self.action_dim, self.b_dim, config.hidden_dim))
        self.execution_critics_target = make_module_list(
            lambda: ExecutionCritic(self.obs_dim, self.action_dim, self.b_dim, config.hidden_dim)
        )
        for tgt, src in zip(self.execution_critics_target, self.execution_critics):
            tgt.load_state_dict(src.state_dict())

        state_dim = self.n_agents * self.obs_dim
        self.mixer = QMixer(self.n_agents, state_dim=state_dim, mixing_embed_dim=config.mixing_embed_dim).to(self.device)
        self.mixer_target = QMixer(self.n_agents, state_dim=state_dim, mixing_embed_dim=config.mixing_embed_dim).to(self.device)
        self.mixer_target.load_state_dict(self.mixer.state_dict())
        self.log_alpha_exe = nn.Parameter(torch.log(torch.tensor(config.alpha_exe_init)))

        self.d_exp = ReplayBuffer(config.buffer_capacity)
        self.d_exe = ReplayBuffer(config.buffer_capacity)

        self.rep_optimizer = torch.optim.Adam(
            list(self.flow.parameters()) + [self.cpc_w] + list(self.budget_encoder.parameters()) + list(self.budget_decoder.parameters()),
            lr=config.lr,
        )
        self.exp_policy_optimizer = torch.optim.Adam(self.exploration_policies.parameters(), lr=config.lr)
        self.exp_critic_optimizer = torch.optim.Adam(self.exploration_critics.parameters(), lr=config.lr)
        self.alpha_exp_optimizer = torch.optim.Adam([self.log_alpha_exp], lr=config.lr)
        self.exe_policy_optimizer = torch.optim.Adam(self.execution_policies.parameters(), lr=config.lr)
        self.exe_critic_optimizer = torch.optim.Adam(
            list(self.execution_critics.parameters()) + list(self.mixer.parameters()), lr=config.lr
        )
        self.alpha_exe_optimizer = torch.optim.Adam([self.log_alpha_exe], lr=config.lr)

        self.target_entropy = -float(self.action_dim)
        self._episode_counter = 0

    # ------------------------------------------------------------------
    # Rollout — fills D_exp and D_exe with joint per-timestep rows.
    # ------------------------------------------------------------------
    def rollout_iteration(self) -> dict[str, float]:
        cfg = self.cfg
        n_envs, n_agents = cfg.n_envs, self.n_agents
        schedule = sample_episode_schedule(n_agents, cfg.horizon, cfg.p_sw, cfg.mass_range, cfg.friction_range)
        eps_a = float(torch.empty(1).uniform_(*cfg.eps_a_range))
        episode_id = self._episode_counter
        self._episode_counter += 1

        for i, agent in enumerate(self.env.agents):
            apply_regime(agent, schedule.mu_1[i])
        obs = self.env.reset()

        posterior_state = self.posterior.init_state((n_envs, n_agents), device=self.device)
        # per-agent episode return, each step averaged over the n_envs
        episode_return_per_agent = torch.zeros(n_agents)
        t = 0
        for t in range(1, cfg.horizon + 1):
            regime_half = torch.zeros(n_agents, dtype=torch.long)
            regime_mode = torch.full((n_agents,), -1, dtype=torch.long)
            regime_vec = torch.zeros(n_agents, 2)
            for i, agent in enumerate(self.env.agents):
                if t == schedule.switch_time[i].item():
                    apply_regime(agent, schedule.mu_2[i])
                    reset_mask = torch.zeros((n_envs, n_agents), dtype=torch.bool, device=self.device)
                    reset_mask[:, i] = True
                    posterior_state = self.posterior.reset_where(posterior_state, reset_mask)
                post = t >= schedule.switch_time[i].item()
                regime_half[i] = 1 if post else 0
                active = schedule.mu_2[i] if post else schedule.mu_1[i]
                regime_vec[i] = torch.tensor(regime_vector(active))
                if schedule.mode_1 is not None:
                    regime_mode[i] = schedule.mode_2[i] if post else schedule.mode_1[i]

            with torch.no_grad():
                actions = [self.exploration_policies[i].act(obs[i])[0] for i in range(n_agents)]

            next_obs, rewards, dones, _infos = self.env.step(actions)

            joint_obs = torch.stack(obs, dim=1)  # (n_envs, n_agents, obs_dim)
            joint_action = torch.stack(actions, dim=1)
            joint_next_obs = torch.stack(next_obs, dim=1)
            joint_reward = torch.stack(rewards, dim=1)  # (n_envs, n_agents)

            with torch.no_grad():
                cond = torch.cat([joint_obs, joint_action], dim=-1)
                y = torch.cat([joint_reward.unsqueeze(-1), joint_next_obs], dim=-1)
                codes, _ = self.flow(y, cond)  # (n_envs, n_agents, code_dim)

                # b_i = {(mu_i,(t-1), sigma2_i,(t-1) * I), ...} (spec §10): the
                # belief an agent actually conditioned on at this step is the
                # *pre*-update posterior, so store it exactly as-is rather than
                # trying to reconstruct it later from a flat replay sample.
                mu_prev = posterior_state.mu.clone()
                sigma2_prev = posterior_state.sigma2.clone()
                posterior_state = self.posterior.step(posterior_state, codes)
                mu_new = posterior_state.mu.clone()
                sigma2_new = posterior_state.sigma2.clone()

            episode_id_field = torch.full((n_envs, n_agents), episode_id, dtype=torch.long)
            regime_half_field = regime_half.unsqueeze(0).expand(n_envs, n_agents).clone()
            regime_mode_field = regime_mode.unsqueeze(0).expand(n_envs, n_agents).clone()
            regime_vec_field = regime_vec.unsqueeze(0).expand(n_envs, n_agents, 2).clone()
            eps_a_field = torch.full((n_envs, n_agents), eps_a)

            row = dict(
                obs=joint_obs,
                action=joint_action,
                next_obs=joint_next_obs,
                reward=joint_reward,
                mu_prev=mu_prev,
                sigma2_prev=sigma2_prev,
                mu_new=mu_new,
                sigma2_new=sigma2_new,
                episode_id=episode_id_field,
                regime_half=regime_half_field,
                regime_mode=regime_mode_field,
                regime_vec=regime_vec_field,
                eps_a=eps_a_field,
            )
            self.d_exp.add(**row)
            self.d_exe.add(**row)

            obs = next_obs
            episode_return_per_agent += joint_reward.mean(dim=0).detach().cpu()
            if bool(dones.all()):
                break

        logs = {f"episode_return_agent{i}": episode_return_per_agent[i].item()
                for i in range(n_agents)}
        logs["episode_return_team"] = episode_return_per_agent.sum().item()
        logs["episode_return_mean"] = episode_return_per_agent.mean().item()
        logs["steps"] = t
        survivals = getattr(self.env.scenario, "_survivals", None)
        if survivals:
            rescued = torch.stack([sv.rescued for sv in survivals], dim=1).float()  # (n_envs, n_surv)
            logs["rescued_frac"] = rescued.mean().item()
            logs["rescued_count"] = rescued.sum(dim=1).mean().item()
        return logs

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------
    def update(self) -> dict[str, float]:
        if not self.d_exe.is_ready(self.cfg.batch_size):
            return {}
        logs = {}
        logs.update(self._update_representation())
        logs.update(self._update_exploration())
        logs.update(self._update_execution())
        self._polyak_update()
        return logs

    def _sample_cpc_indices(self, mode: torch.Tensor, vec: torch.Tensor):
        """Vectorized CPC index sampler for one agent's minibatch column.

        `mode`: (B,) long regime-mode id per row (-1 if the flat-box task
        distribution was used). `vec`: (B, 2) the row's (mass, friction).

        Returns (anchors, pos_idx, neg_idx) with shapes (A,), (A,), (A, K),
        or None if no row has both a positive and a negative. For each anchor:

          * positive  = one row of the SAME regime mode (any episode; de
            Finetti: same-mode transitions are exchangeable), chosen
            uniformly. With no mode labels, any row within `cpc_pos_tol`
            normalized distance.
          * negatives = K rows of a DIFFERENT mode (or, unlabeled, any row
            at least `cpc_neg_guard` away), sampled with probability
            proportional to exp(-d / cpc_neg_temp) so the hardest
            (closest-but-different) negatives dominate. The `cpc_neg_guard`
            floor is unconditional: it removes false negatives from
            overlapping / adjacent regimes.
        """
        cfg = self.cfg
        B = mode.shape[0]
        dev = mode.device
        K = cfg.n_cpc_negatives

        span = torch.tensor(
            [cfg.mass_range[1] - cfg.mass_range[0], cfg.friction_range[1] - cfg.friction_range[0]],
            device=dev,
        ).clamp_min(1e-6)
        d = torch.cdist(vec / span, vec / span)  # (B, B) normalized regime distance
        eye = torch.eye(B, dtype=torch.bool, device=dev)

        labelled = mode >= 0
        pair_labelled = labelled.unsqueeze(0) & labelled.unsqueeze(1)
        same_mode = mode.unsqueeze(0) == mode.unsqueeze(1)

        pos_mask = ~eye & ((pair_labelled & same_mode) | (~pair_labelled & (d <= cfg.cpc_pos_tol)))
        neg_mask = (d >= cfg.cpc_neg_guard) & ((pair_labelled & ~same_mode) | ~pair_labelled)

        valid = pos_mask.any(1) & neg_mask.any(1)
        if not valid.any():
            return None
        anchors = valid.nonzero(as_tuple=True)[0]

        pos_idx = torch.multinomial(pos_mask[anchors].float(), 1).squeeze(1)
        w_neg = neg_mask[anchors].float() * torch.exp(-d[anchors] / cfg.cpc_neg_temp) + 1e-12
        neg_idx = torch.multinomial(w_neg, K, replacement=True)  # (A, K)
        return anchors, pos_idx, neg_idx

    def _flow_code(self, flow: ConditionalFlow, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, reward: torch.Tensor):
        cond = torch.cat([obs, action], dim=-1)
        y = torch.cat([reward.unsqueeze(-1), next_obs], dim=-1)
        return flow(y, cond)

    def _update_representation(self) -> dict[str, float]:
        cfg = self.cfg
        batch = self.d_exp.sample(cfg.batch_size, device=self.device)  # fields: (B, n_agents, ...)

        elbo_total = 0.0
        cpc_total = torch.tensor(0.0, device=self.device)
        n_cpc_terms = 0
        for i in range(self.n_agents):
            obs_i, action_i, next_obs_i, reward_i = batch["obs"][:, i], batch["action"][:, i], batch["next_obs"][:, i], batch["reward"][:, i]
            code, log_det = self._flow_code(self.flow, obs_i, action_i, next_obs_i, reward_i)

            elbo = elbo_loss(
                code=code,
                log_det_jacobian=log_det,
                predictive_mu=batch["mu_prev"][:, i],
                predictive_sigma2=batch["sigma2_prev"][:, i],
                posterior_mu=batch["mu_new"][:, i],
                posterior_sigma2=batch["sigma2_new"][:, i],
                prior_mu=torch.zeros_like(code),
                prior_sigma2=torch.full_like(batch["sigma2_new"][:, i], cfg.nu),
            ).mean()
            elbo_total = elbo_total + elbo

            idx = self._sample_cpc_indices(batch["regime_mode"][:, i], batch["regime_vec"][:, i])
            if idx is not None:
                anchor_idx, pos_idx, neg_idx = idx
                with torch.no_grad():
                    pool_code, _ = self._flow_code(self.flow_momentum, obs_i, action_i, next_obs_i, reward_i)
                z_query = code[anchor_idx]        # (A, d)    online encoder, carries grad
                z_positive = pool_code[pos_idx]   # (A, d)    momentum encoder
                z_negatives = pool_code[neg_idx]  # (A, K, d) momentum encoder
                cpc_total = cpc_total + infonce_cpc_loss(z_query, z_positive, z_negatives, self.cpc_w).mean()
                n_cpc_terms += 1

        elbo_total = elbo_total / self.n_agents
        cpc_loss = cpc_total / max(n_cpc_terms, 1)

        # Budget encoder: pool this timestep's (o_i, a_i) across agents.
        x_exe = torch.cat([batch["obs"], batch["action"]], dim=-1)  # (B, n_agents, obs+act)
        mu_rho, sigma2_rho = self.budget_encoder(x_exe)
        rho = self.budget_encoder.sample(mu_rho, sigma2_rho)
        eps_hat = self.budget_decoder(rho)
        l_rho = budget_loss(mu_rho, sigma2_rho, eps_hat, batch["eps_a"][:, 0]).mean()

        # spec §7: L_cov uses the context posterior *mean* mu_z, not the raw
        # code — one agent's is enough (the penalty targets any residual
        # z-rho leakage, and every agent's mu_z shares the same mu_rho here).
        l_cov = cross_covariance_penalty(batch["mu_new"][:, 0].detach(), mu_rho)

        # eq:pm-enc:  L_enc = L_ELBO - lambda_CPC * log softmax(f(z_q, z_pos))
        #                   = L_ELBO + lambda_CPC * CE(...)      [ CE = -log softmax ]
        # `infonce_cpc_loss` returns the cross-entropy (>= 0, to be MINIMIZED),
        # so it is ADDED. The previous `-` maximized the InfoNCE loss, i.e.
        # trained z to make same-/different-regime segments *indistinguishable*
        # -- l_cpc never dropped below the log(K+1) random-chance floor.
        loss = elbo_total + cfg.lambda_cpc * cpc_loss + l_rho + cfg.lambda_cov * l_cov
        self.rep_optimizer.zero_grad()
        loss.backward()
        self.rep_optimizer.step()

        with torch.no_grad():
            for p_bar, p in zip(self.flow_momentum.parameters(), self.flow.parameters()):
                p_bar.mul_(cfg.tau_m).add_(p, alpha=1 - cfg.tau_m)

        return {"l_elbo": elbo_total.item(), "l_cpc": cpc_loss.item(), "l_rho": l_rho.item(), "l_cov": l_cov.item()}

    def _update_exploration(self) -> dict[str, float]:
        cfg = self.cfg
        batch = self.d_exp.sample(cfg.batch_size, device=self.device)

        # Pass 1: critic update. Must fully backward+step before the policy
        # pass below builds any graph through these same critic parameters —
        # interleaving the two (as an earlier version did) backwards through
        # a critic forward pass computed *before* optimizer.step() bumped
        # its parameters' versions, which autograd rejects.
        total_critic_loss = torch.tensor(0.0, device=self.device)
        for i in range(self.n_agents):
            obs_i, action_i, next_obs_i = batch["obs"][:, i], batch["action"][:, i], batch["next_obs"][:, i]
            r_aux = information_gain_reward(batch["sigma2_prev"][:, i], batch["sigma2_new"][:, i])
            r_e = batch["reward"][:, i] + self.log_alpha_exp[i].exp().detach() * r_aux

            with torch.no_grad():
                next_action, next_log_prob, _ = self.exploration_policies[i].sample(next_obs_i)
                q1_t, q2_t = self.exploration_critics_target[i].q(next_obs_i, next_action)
                target_v = torch.minimum(q1_t, q2_t) - self.log_alpha_exp[i].exp() * next_log_prob
                target_q = r_e + cfg.gamma * target_v

            q1, q2 = self.exploration_critics[i].q(obs_i, action_i)
            critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
            total_critic_loss = total_critic_loss + critic_loss

        self.exp_critic_optimizer.zero_grad()
        total_critic_loss.backward()
        self.exp_critic_optimizer.step()

        # Pass 2: policy (+ alpha) update, with a fresh critic forward pass.
        total_policy_loss = torch.tensor(0.0, device=self.device)
        total_alpha_loss = torch.tensor(0.0, device=self.device)
        for i in range(self.n_agents):
            obs_i = batch["obs"][:, i]
            new_action, log_prob, _ = self.exploration_policies[i].sample(obs_i)
            q_new = self.exploration_critics[i].min_q(torch.cat([obs_i, new_action], dim=-1))
            policy_loss = (self.log_alpha_exp[i].exp().detach() * log_prob - q_new).mean()
            alpha_loss = -(self.log_alpha_exp[i].exp() * (log_prob.detach() + self.target_entropy)).mean()

            total_policy_loss = total_policy_loss + policy_loss
            total_alpha_loss = total_alpha_loss + alpha_loss

        self.exp_policy_optimizer.zero_grad()
        total_policy_loss.backward()
        self.exp_policy_optimizer.step()

        self.alpha_exp_optimizer.zero_grad()
        total_alpha_loss.backward()
        self.alpha_exp_optimizer.step()
        # Floor the exploration entropy coefficient: in earlier runs it
        # annealed to ~0.01 and the exploration policy went deterministic,
        # so it stopped covering the arena and never found victims.
        with torch.no_grad():
            self.log_alpha_exp.clamp_(min=cfg.log_alpha_exp_min)

        return {
            "exp_critic_loss": total_critic_loss.item(),
            "exp_policy_loss": total_policy_loss.item(),
            "alpha_exp_mean": self.log_alpha_exp.exp().mean().item(),
        }

    def _belief_for_batch(self, batch: dict[str, torch.Tensor], use_next: bool = False) -> torch.Tensor:
        """(B, n_agents, b_dim) belief, one per agent, from a joint minibatch.

        mu_z/sigma2_z: the exact posterior each agent actually held at the
        relevant step, stored at collection time rather than reconstructed.
        `use_next=False` (default) gives the *pre*-update posterior
        (mu_prev, sigma2_prev) — what the agent conditioned on when
        choosing `batch["action"]` (spec §10: `b_i = {(mu_i,(t-1),
        sigma2_i,(t-1)*I), ...}`). `use_next=True` gives the posterior one
        step later (mu_new, sigma2_new) — i.e. what's held at `next_obs`,
        needed for the TD target's next-action sampling.
        mu_rho/sigma2_rho: from the budget encoder, team-shared (same value
        broadcast to every agent in a given row); not similarly advanced
        for `use_next` since it depends on (obs, action), not next_obs.
        """
        b, n = batch["obs"].shape[0], self.n_agents
        mu_z = batch["mu_new"] if use_next else batch["mu_prev"]
        sigma2_z = batch["sigma2_new"] if use_next else batch["sigma2_prev"]
        x_exe = torch.cat([batch["obs"], batch["action"]], dim=-1)
        with torch.no_grad():
            mu_rho, sigma2_rho = self.budget_encoder(x_exe)
        mu_rho = mu_rho.unsqueeze(1).expand(b, n, self.rho_dim)
        sigma2_rho = sigma2_rho.unsqueeze(1).expand(b, n, self.rho_dim)
        return flatten_belief(mu_z, sigma2_z, mu_rho, sigma2_rho)

    def _q_tot(self, critics, mixer, obs, actions, belief, state) -> torch.Tensor:
        """obs, actions, belief: (B, n_agents, *). Returns Q_tot: (B,)."""
        qs = []
        for i in range(self.n_agents):
            q1, q2 = critics[i].q(obs[:, i], actions[:, i], belief[:, i])
            qs.append(torch.minimum(q1, q2))
        return mixer(torch.stack(qs, dim=-1), state)

    def _update_execution(self) -> dict[str, float]:
        cfg = self.cfg
        batch = self.d_exe.sample(cfg.batch_size, device=self.device)
        obs, action, next_obs = batch["obs"], batch["action"], batch["next_obs"]
        team_reward = batch["reward"].mean(dim=1)  # shared team reward for Q_tot's TD target
        belief = self._belief_for_batch(batch)
        next_belief = self._belief_for_batch(batch, use_next=True)
        state = obs.reshape(obs.shape[0], -1)
        alpha_exe = self.log_alpha_exe.exp()

        with torch.no_grad():
            next_actions, next_log_probs = [], []
            for i in range(self.n_agents):
                a, lp, _ = self.execution_policies[i].sample(torch.cat([next_obs[:, i], next_belief[:, i]], dim=-1))
                next_actions.append(a)
                next_log_probs.append(lp)
            next_actions = torch.stack(next_actions, dim=1)  # (B, n_agents, action_dim)

        # Teammates 1..n-1 keep their clean, nominal next_actions —
        # MAAL (teammate-action attack) is intentionally not used here.
        # Only agent 0's own channel is attacked below (position, then
        # its own action), not what teammates do.
        perturbed_next_actions = next_actions

        # Agent 0's own perceived *position* is attacked (PGD/FGSM, L_inf
        # ball) to make its chosen action worse. The critic still
        # evaluates Q against the TRUE next_obs (position attack only
        # changes what agent 0's *policy* saw when choosing its action,
        # not the environment's actual state) — mirrors SA-MDP/RADIAL-RL's
        # framing (Ch.5): the agent perceives a corrupted observation but
        # the world, and the value function's notion of state, doesn't
        # actually change.
        agent0_next_obs = next_obs[:, 0]

        def q_for_agent0_position(candidate_pos: torch.Tensor) -> torch.Tensor:
            perturbed_obs0 = torch.cat([candidate_pos, agent0_next_obs[:, 2:]], dim=-1)
            action0, _, _ = self.execution_policies[0].sample(torch.cat([perturbed_obs0, next_belief[:, 0]], dim=-1))
            candidate_actions = torch.cat([action0.unsqueeze(1), perturbed_next_actions[:, 1:]], dim=1)
            return self._q_tot(self.execution_critics_target, self.mixer_target, next_obs, candidate_actions, next_belief, state)

        perturbed_position = perturb_position(
            q_for_agent0_position, agent0_next_obs[:, :2], cfg.eps_pos, cfg.beta_pos, cfg.pgd_steps
        )
        with torch.no_grad():
            perturbed_obs0 = torch.cat([perturbed_position, agent0_next_obs[:, 2:]], dim=-1)
            attacked_action0, _, _ = self.execution_policies[0].sample(torch.cat([perturbed_obs0, next_belief[:, 0]], dim=-1))
            perturbed_next_actions = torch.cat([attacked_action0.unsqueeze(1), perturbed_next_actions[:, 1:]], dim=1)

        # Third threat model, chained on top of the (already position-
        # attacked) action above: agent 0's own action is *also* directly
        # perturbed — an actuator-style attack downstream of the policy's
        # decision, distinct from MAAL (which attacks teammates' actions,
        # not this agent's own) and from the position attack (which
        # attacks what feeds the policy, not what comes out of it).
        def q_for_agent0_action(candidate_action: torch.Tensor) -> torch.Tensor:
            candidate_actions = torch.cat([candidate_action.unsqueeze(1), perturbed_next_actions[:, 1:]], dim=1)
            return self._q_tot(self.execution_critics_target, self.mixer_target, next_obs, candidate_actions, next_belief, state)

        attacked_action0_final = perturb_own_action(
            q_for_agent0_action, perturbed_next_actions[:, 0], cfg.eps_self_action, cfg.beta_self_action, cfg.pgd_steps
        )
        perturbed_next_actions = torch.cat([attacked_action0_final.unsqueeze(1), perturbed_next_actions[:, 1:]], dim=1)

        with torch.no_grad():
            q_tot_next = self._q_tot(self.execution_critics_target, self.mixer_target, next_obs, perturbed_next_actions, next_belief, state)
            entropy_term = alpha_exe * torch.stack(next_log_probs, dim=1).sum(dim=1)
            y_tot = team_reward + cfg.gamma * (q_tot_next - entropy_term)

        q_tot = self._q_tot(self.execution_critics, self.mixer, obs, action, belief, state)
        critic_loss = ((q_tot - y_tot) ** 2).mean()

        self.exe_critic_optimizer.zero_grad()
        critic_loss.backward()
        self.exe_critic_optimizer.step()

        policy_loss_total = torch.tensor(0.0, device=self.device)
        log_probs_pi = []
        for i in range(self.n_agents):
            new_action, log_prob, _ = self.execution_policies[i].sample(torch.cat([obs[:, i], belief[:, i]], dim=-1))
            candidate_actions = action.clone()
            candidate_actions[:, i] = new_action
            q_tot_pi = self._q_tot(self.execution_critics, self.mixer, obs, candidate_actions, belief, state)
            policy_loss_total = policy_loss_total + (alpha_exe.detach() * log_prob - q_tot_pi).mean()
            log_probs_pi.append(log_prob)

        self.exe_policy_optimizer.zero_grad()
        policy_loss_total.backward()
        self.exe_policy_optimizer.step()

        # Auto-tune the execution entropy temperature against the same
        # target entropy as exploration (ch-proposed.tex: alpha_exe is
        # adapted, independently of alpha_exp). Previously log_alpha_exe was
        # a Parameter with a live optimizer that was never stepped -- alpha_exe
        # stayed pinned at its init 0.2 for the whole run.
        log_prob_all = torch.stack(log_probs_pi, dim=0).mean()
        alpha_exe_loss = -(self.log_alpha_exe.exp() * (log_prob_all.detach() + self.target_entropy))
        self.alpha_exe_optimizer.zero_grad()
        alpha_exe_loss.backward()
        self.alpha_exe_optimizer.step()
        with torch.no_grad():
            self.log_alpha_exe.clamp_(min=self.cfg.log_alpha_exe_min)

        return {
            "exe_critic_loss": critic_loss.item(),
            "exe_policy_loss": policy_loss_total.item(),
            "alpha_exe": self.log_alpha_exe.exp().item(),
        }

    def _polyak_update(self) -> None:
        tau = self.cfg.tau_polyak
        for tgt, src in zip(self.exploration_critics_target, self.exploration_critics):
            for pt, ps in zip(tgt.parameters(), src.parameters()):
                pt.data.mul_(1 - tau).add_(ps.data, alpha=tau)
        for tgt, src in zip(self.execution_critics_target, self.execution_critics):
            for pt, ps in zip(tgt.parameters(), src.parameters()):
                pt.data.mul_(1 - tau).add_(ps.data, alpha=tau)
        for pt, ps in zip(self.mixer_target.parameters(), self.mixer.parameters()):
            pt.data.mul_(1 - tau).add_(ps.data, alpha=tau)
