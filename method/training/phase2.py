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

from ..detector.indid import CausalLocalWindowTransformer, detection_loss
from .phase1 import Phase1Trainer
from .regime import apply_regime, sample_episode_schedule


@dataclass
class Phase2Config:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    window: int = 16
    lambda_fa: float = 1.0
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

        schedule = sample_episode_schedule(
            n_agents, p1_cfg.horizon, p1_cfg.p_sw, p1_cfg.mass_range, p1_cfg.friction_range
        )
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
        for i in range(n_agents):
            p_i = self.detector(codes_seq[:, i])  # (n_envs, T)
            switch_time_i = switch_time[i].expand(n_envs)
            total_loss = total_loss + detection_loss(p_i, switch_time_i, self.cfg.lambda_fa).mean()

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
