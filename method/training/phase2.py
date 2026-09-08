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

from ..detector.indid import CausalLocalWindowTransformer, detection_loss
from .phase1 import Phase1Trainer
from .regime import DEFAULT_MODES, apply_regime, sample_episode_schedule, sample_far_regime


@dataclass
class Phase2Config:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    window: int = 16
    # The InDiD delay/FA loss alone is unbalanced on this task: with
    # lambda_fa >= 1 "never fire" is the global optimum even when the code
    # stream carries a strong change signal (probe AUC ~0.8) -- missing a
    # real change costs less delay than the FA term saves. Fix: a direct
    # per-step BCE against the ground-truth "has switched" label carries the
    # supervision, the InDiD terms only shape delay/FA around it, and
    # lambda_fa is dropped well below 1.
    bce_weight: float = 1.0
    lambda_fa: float = 0.5
    p_no_change: float = 0.35      # fraction of episodes with NO regime switch (stationary negatives)
    warmup_p_reg: float = 0.05     # penalty on mean p over the first `window` steps
    # 3e-4 (a common default elsewhere in this codebase) reliably diverges
    # this transformer to NaN within ~20-60 iterations -- verified directly
    # by inspecting detector.parameters() every iteration. 1e-4 ran 150
    # iterations with zero non-finite batches and a cleanly decreasing
    # loss; the train_iteration()'s skip-on-non-finite-gradient guard stays
    # as defense in depth, not as the primary fix.
    lr: float = 1e-4


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

        self.detector = CausalLocalWindowTransformer(
            code_dim=self.phase1.code_dim,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            window=config.window,
            max_len=self.phase1.cfg.horizon + 1,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.detector.parameters(), lr=config.lr)

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
            lo = self.cfg.window + 4
            hi = max(lo + 1, H - 8)
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

        codes_over_time = []
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
            cond = torch.cat([joint_obs, joint_action], dim=-1)
            y = torch.cat([joint_reward.unsqueeze(-1), joint_next_obs], dim=-1)
            codes, _ = self.phase1.flow(y, cond)  # (n_envs, n_agents, code_dim)
            codes_over_time.append(codes)

            obs = next_obs
            if bool(dones.all()):
                break

        codes_seq = torch.stack(codes_over_time, dim=2)  # (n_envs, n_agents, T, code_dim)
        switch_time = schedule.switch_time.clamp(max=t).to(self.device)
        return codes_seq, switch_time

    def train_iteration(self) -> dict[str, float]:
        codes_seq, switch_time = self.collect_labeled_episode()
        n_envs, n_agents, _T, _code_dim = codes_seq.shape

        total_loss = torch.tensor(0.0, device=self.device)
        w = self.detector.window
        for i in range(n_agents):
            p_i = self.detector(codes_seq[:, i])  # (n_envs, T)
            T = p_i.shape[1]
            switch_time_i = switch_time[i].expand(n_envs)
            total_loss = total_loss + detection_loss(p_i, switch_time_i, self.cfg.lambda_fa).mean()

            # direct per-step supervision: label = 1 once the regime has
            # switched (all-zero for no-switch episodes). This is the term
            # that actually anchors the detector; it cannot be minimized by
            # p == 0 (that is punished on every post-switch step).
            t_idx = torch.arange(T, device=p_i.device)
            sw = torch.where(switch_time_i < T, switch_time_i, torch.full_like(switch_time_i, T + 1))
            label = (t_idx.unsqueeze(0) >= (sw.unsqueeze(1) - 1)).float()
            bce = torch.nn.functional.binary_cross_entropy(p_i.clamp(1e-4, 1 - 1e-4), label)
            total_loss = total_loss + self.cfg.bce_weight * bce

            # keep the detector quiet during its warm-up window (incomplete context)
            total_loss = total_loss + self.cfg.warmup_p_reg * p_i[:, :w].mean()

        self.optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping alone is not enough here and was verified
        # insufficient directly: an isolated batch's loss occasionally goes
        # NaN on its own (confirmed via direct inspection -- the detector's
        # own p outputs stayed finite and unsaturated for every iteration
        # leading up to the failure, so this isn't sigmoid saturation
        # feeding the log-survival trick's clamp). clip_grad_norm_ computes
        # a single scalar norm over *all* parameters' gradients together, so
        # one NaN component makes that norm (and therefore the rescaling
        # applied to every other parameter) NaN too -- one bad batch
        # poisons every parameter's gradient. Adam's internal moving
        # averages (exp_avg/exp_avg_sq) then absorb that NaN and never
        # recover, even once later batches are fine again -- confirmed
        # directly: loss went transiently NaN, then finite again for one
        # iteration, then permanently NaN once the optimizer stepped a
        # second time. Skipping the step entirely on a non-finite batch
        # (standard practice for exactly this failure mode) avoids ever
        # touching the optimizer state with it.
        # Per-element clip first: one exploded component can no longer poison
        # the global norm below (and therefore every other parameter's
        # rescaled gradient) before the finiteness check gets to see it.
        torch.nn.utils.clip_grad_value_(self.detector.parameters(), clip_value=5.0)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.detector.parameters(), max_norm=1.0)
        if torch.isfinite(grad_norm) and torch.isfinite(total_loss):
            self.optimizer.step()
        else:
            # Skipping the step is not enough on its own: Adam's exp_avg /
            # exp_avg_sq can already hold a NaN from an earlier step and never
            # recover. Clearing the optimizer state makes a transient bad
            # batch fully recoverable -- the moments just rebuild from the
            # next finite gradients.
            self.optimizer.zero_grad()
            self.optimizer.state.clear()
        return {"l_cpd": total_loss.item(), "grad_norm": grad_norm.item()}
