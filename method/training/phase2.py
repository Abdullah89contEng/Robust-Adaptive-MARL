"""Phase 2 -- supervised change-point detection (method-spec.md / thesis-v2.pdf
§10.9, Algorithm 10.2).

Freezes everything from a trained `Phase1Trainer` (its context encoder
`f_phi` and its per-agent exploration policies) and trains the InDiD-style
detector `h_xi` on episodes generated the same way Phase 1's own rollouts
are (same regime-sampling distribution, same environment) — reusing
`Phase1Trainer.cfg`/`.env` rather than duplicating those settings in a
separate config, since the thesis explicitly frames this as training the
detector "on the same episodes" Phase 1 used.

DESIGN CHOICE: episodes are collected fresh here with the *frozen*
exploration policy acting, rather than replayed from D_exp/D_exe. Phase 1's
buffers store flat, order-shuffled rows (see phase1.py's own design-choice
notes on this) -- exactly the wrong shape for a detector that needs whole,
temporally-ordered per-agent code sequences c_i,1:H paired with a single
ground-truth switch time. Re-rolling out episodes (cheap: no gradient
tracked through the frozen encoder/policy) sidesteps reconstructing that
structure from the flat buffers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import random

from ..detector.indid import CausalLocalWindowTransformer, detection_loss, paper_cpd_loss
from .phase1 import Phase1Trainer
from .regime import DEFAULT_MODES, apply_regime, sample_episode_schedule, sample_far_regime


@dataclass
class Phase2Config:
    # Backbone: causal local-window Transformer (Algorithm 10.2). `window`
    # is the attention span AND the deployment history cap.
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 4        # arXiv:2510.24988v1 Sec. 4: "4-layer multi-head attention network (4 heads)"
    dim_feedforward: int = 128
    window: int = 20         # arXiv:2510.24988v1 Sec. 4: "trajectory windows of length 20"
    # "raw" -> detector runs on the transition (o, a, r, o'), where the
    # change is strongly visible (probe AUC ~0.8-0.9); "code" -> frozen
    # f_phi embedding c_{i,t}.
    detector_input: str = "raw"

    # Training objective:
    #   "paper" -> arXiv:2510.24988v1 Eq. 7 (+ Alg. 1/2): near-boundary
    #              weighted, label-smoothed BCE on +/-Delta boundary labels
    #              (default).
    #   "indid" -> InDiD CPDLoss (delay + false-alarm), kept for comparison.
    detector_loss: str = "paper"
    label_half_width: int = 2        # +/-Delta window: y_t = 1 for |t - theta| <= this
    label_smooth_eps: float = 0.1    # y~_t = (1 - eps) y_t + eps/2                 (Alg. 1)
    near_boundary_alpha: float = 3.0  # w_t = 1 + alpha for t in the +/-Delta window (Alg. 2)
    input_noise_std: float = 0.01    # additive Gaussian token noise, Sec. 5.2 ...
    input_noise_prob: float = 0.30   # ... applied to the batch with this probability
    weight_decay: float = 1e-4       # Adam weight decay, Sec. 5.2

    # InDiD-path only (detector_loss == "indid"): CPDLoss segment length T
    # (paper: 5..32), delay weighted alpha = 2*batch/T, FA at beta = 1.
    len_segment: int = 32
    # Reproduce a batch of ~64 sequences: stack this many rollouts and merge
    # the (env, agent) axes before one loss/step (4 x 8 envs x 2 agents = 64).
    rollouts_per_step: int = 4
    p_no_change: float = 0.35   # fraction of episodes with NO regime switch (stationary negatives)
    grad_clip_norm: float = 1.0  # arXiv:2510.24988v1 Sec. 5.2: "gradient clipping 1.0" (0 = off)
    lr: float = 1e-3


class Phase2Trainer:
    def __init__(self, phase1_trainer: Phase1Trainer, config: Phase2Config):
        self.phase1 = phase1_trainer
        self.cfg = config
        self.device = phase1_trainer.device

        # Freeze everything from Phase 1 (thesis §10.9: "Phase 2 ... freezes
        # everything from Phase 1"). Only f_phi and the exploration policies
        # are actually exercised during episode collection below, but
        # freezing is a correctness statement, not just an optimization:
        # nothing here should ever receive a gradient.
        for module in [self.phase1.flow, *self.phase1.exploration_policies]:
            module.eval()
            for p in module.parameters():
                p.requires_grad_(False)

        self.detector_input = config.detector_input
        obs_dim, act_dim = self.phase1.obs_dim, self.phase1.action_dim
        self.raw_dim = obs_dim + act_dim + 1 + obs_dim   # (o, a, r, o')
        feat_dim = self.raw_dim if self.detector_input == "raw" else self.phase1.code_dim

        self.detector = CausalLocalWindowTransformer(
            code_dim=feat_dim,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            window=config.window,
            dim_feedforward=config.dim_feedforward,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.detector.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

    @torch.no_grad()
    def collect_labeled_episode(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Roll out one episode with the frozen exploration policy.

        Returns (codes, switch_time):
            codes: (n_envs, n_agents, T, code_dim) -- frozen f_phi embeddings.
            switch_time: (n_agents,) ground-truth vartheta_i_tilde, clamped
                to the realized episode length T (T == cfg.horizon unless
                every env's episode ended early).
        """
        p1_cfg = self.phase1.cfg
        env = self.phase1.env
        n_agents = self.phase1.n_agents

        H = p1_cfg.horizon
        schedule = sample_episode_schedule(
            n_agents, H, p1_cfg.p_sw, p1_cfg.mass_range, p1_cfg.friction_range
        )

        # Rebalance the supervision: a fixed fraction of episodes have NO
        # switch (switch_time == H sentinel, mu_2 == mu_1); the rest get a
        # switch time drawn UNIFORMLY over the usable window (not GEOM,
        # which piles every positive near step ~20). Because this path
        # OVERRIDES switch_time, sample_episode_schedule's own separation
        # guarantee (which only fires where the GEOM draw already produced a
        # switch) does not cover every agent here -- so re-draw mu_2 with
        # the same guarantee (different mode AND >= min-distance from mu_1).
        ms = p1_cfg.mass_range[1] - p1_cfg.mass_range[0]
        fs = p1_cfg.friction_range[1] - p1_cfg.friction_range[0]
        if random.random() < self.cfg.p_no_change:
            schedule.mu_2 = list(schedule.mu_1)
            schedule.switch_time = torch.full((n_agents,), H, dtype=torch.long)
            if schedule.mode_2 is not None:
                schedule.mode_2 = schedule.mode_1.clone()
        else:
            lo, hi = 10, max(11, H - 10)   # switch lands well inside the episode
            schedule.switch_time = torch.randint(lo, hi, (n_agents,))
            mode_2 = schedule.mode_2.clone() if schedule.mode_2 is not None else None
            for i in range(n_agents):
                m1 = int(schedule.mode_1[i]) if schedule.mode_1 is not None else -1
                r2, k2 = sample_far_regime(schedule.mu_1[i], m1, DEFAULT_MODES, ms, fs)
                schedule.mu_2[i] = r2
                if mode_2 is not None:
                    mode_2[i] = k2
            schedule.mode_2 = mode_2

        for i, agent in enumerate(env.agents):
            apply_regime(agent, schedule.mu_1[i])
        obs = env.reset()

        feat_over_time = []
        t = 0
        for t in range(1, p1_cfg.horizon + 1):
            for i, agent in enumerate(env.agents):
                if t == schedule.switch_time[i].item():
                    apply_regime(agent, schedule.mu_2[i])

            actions = [self.phase1.exploration_policies[i].act(obs[i])[0] for i in range(n_agents)]
            next_obs, rewards, dones, _infos = env.step(actions)

            joint_obs = torch.stack(obs, dim=1)
            joint_action = torch.stack(actions, dim=1)
            joint_next_obs = torch.stack(next_obs, dim=1)
            joint_reward = torch.stack(rewards, dim=1)
            if self.detector_input == "raw":
                feat = torch.cat([joint_obs, joint_action, joint_reward.unsqueeze(-1), joint_next_obs], dim=-1)
            else:
                cond = torch.cat([joint_obs, joint_action], dim=-1)
                y = torch.cat([joint_reward.unsqueeze(-1), joint_next_obs], dim=-1)
                feat, _ = self.phase1.flow(y, cond)  # (n_envs, n_agents, code_dim)
            feat_over_time.append(feat)
            obs = next_obs
            # NB: no early `done` break -- the detector needs full-length
            # labeled sequences with a real post-switch segment (episodes
            # otherwise end when the agents rescue everyone, often before the
            # scheduled switch, leaving nothing to detect).

        feat_seq = torch.stack(feat_over_time, dim=2)  # (n_envs, n_agents, T, feat_dim)
        switch_time = schedule.switch_time.clamp(max=t).to(self.device)
        return feat_seq, switch_time

    def train_iteration(self) -> dict[str, float]:
        # Accumulate several rollouts and merge the (env, agent) axes into
        # one batch so the loss sees ~64 sequences, matching InDiD's batch.
        feats, sws = [], []
        for _ in range(self.cfg.rollouts_per_step):
            fs, st = self.collect_labeled_episode()      # (n_envs, n_agents, T, D), (n_agents,)
            ne, na, T, D = fs.shape
            feats.append(fs.reshape(ne * na, T, D))
            sws.append(st.unsqueeze(0).expand(ne, na).reshape(ne * na))
        feat_batch = torch.cat(feats, 0)                 # (B, T, D)
        sw_batch = torch.cat(sws, 0)                     # (B,)
        self.detector.update_norm(feat_batch)            # online analogue of the paper's offline z-norm

        # arXiv:2510.24988v1 Sec. 5.2: additive Gaussian token noise on a
        # fraction of batches.
        if self.cfg.input_noise_std > 0 and random.random() < self.cfg.input_noise_prob:
            feat_batch = feat_batch + self.cfg.input_noise_std * torch.randn_like(feat_batch)

        p = self.detector(feat_batch)                    # (B, T)
        if self.cfg.detector_loss == "paper":
            total_loss = paper_cpd_loss(
                p, sw_batch,
                half_width=self.cfg.label_half_width,
                smooth_eps=self.cfg.label_smooth_eps,
                near_alpha=self.cfg.near_boundary_alpha,
            )
        else:
            total_loss = detection_loss(p, sw_batch, self.cfg.len_segment)

        self.optimizer.zero_grad()
        total_loss.backward()
        # arXiv:2510.24988v1 clips grad norm at 1.0 (Phase2Config default);
        # set grad_clip_norm = 0 to disable. The skip-on-non-finite guard
        # below is ours: one bad batch can otherwise poison Adam's moments
        # permanently on this noisier rollout data.
        if self.cfg.grad_clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.detector.parameters(), self.cfg.grad_clip_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.detector.parameters(), float("inf"))
        if torch.isfinite(grad_norm) and torch.isfinite(total_loss):
            self.optimizer.step()
        else:
            self.optimizer.zero_grad()
            self.optimizer.state.clear()
        return {"l_cpd": total_loss.item(), "grad_norm": grad_norm.item()}
