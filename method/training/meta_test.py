"""Meta-test: change-point-triggered re-exploration (method-spec.md / thesis-v2.pdf
Algorithm 10.3), run interactively on a dedicated *test* environment.

Everything here is frozen (phi, psi, eta, all policies/critics, xi): no
gradient updates happen during meta-test, matching the thesis's framing —
this is deployment, not training. The environment is deliberately a
*separate* VMAS instance from whatever env a `Phase1Trainer` was trained
against (num_envs=1, single trajectory), since the point here is to watch
one specific, scripted storyline unfold, not to batch-evaluate.

Two kinds of scripted events can be injected at specific steps, matching
the two things this thesis (Ch.10) actually models as non-stationarity:
- `RegimeChangeEvent`: an environmental change (Section 10.1's mu_i ->
  mu_2,i), the thing the context posterior + detector are meant to catch.
- `AdversarialAttackEvent`: activates the position/own-action PGD attacks
  from `method/adversary/state_attack.py` against a given agent from a
  given step onward, using that agent's own trained (frozen) execution
  critic as the attack objective -- the same mechanism used at Phase-1
  training time, now applied live at test time instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import vmas

from ..adversary.state_attack import perturb_own_action, perturb_position
from ..belief import flatten_belief
from ..detector.indid import CausalLocalWindowTransformer
from .phase1 import Phase1Trainer
from .regime import Regime, apply_regime


@dataclass
class MetaTestConfig:
    threshold_C: float = 0.5
    trigger_persistence: int = 3   # require p >= threshold_C this many steps in a row before a reset fires
    re_exploration_budget_K: int = 8
    sigma2_min: float = 0.05
    eps_pos: float = 0.2
    beta_pos: float = 0.05
    eps_action: float = 0.1
    beta_action: float = 0.05
    pgd_steps: int = 1


@dataclass
class RegimeChangeEvent:
    step: int
    agent: int
    regime: Regime


@dataclass
class AdversarialAttackEvent:
    step: int
    agent: int
    attack_position: bool = True
    attack_action: bool = True


@dataclass
class StepResult:
    step: int
    frame: "object"  # rgb_array from env.render(), or None if not rendered this step
    rewards: list[float]
    mode: list[str]
    detector_p: list[Optional[float]]
    reset_fired: list[bool]
    active_attacks: list[bool]


class MetaTestRunner:
    """Owns its own single-instance test environment; drives Algorithm 10.3."""

    def __init__(
        self,
        phase1_trainer: Phase1Trainer,
        detector: CausalLocalWindowTransformer,
        scenario_factory,
        config: MetaTestConfig,
        device: torch.device | None = None,
    ):
        self.p1 = phase1_trainer
        self.detector = detector
        self.cfg = config
        self.device = device or phase1_trainer.device
        self.n_agents = phase1_trainer.n_agents

        # Freeze everything -- meta-test performs no gradient updates at all.
        modules = [
            self.p1.flow,
            self.p1.budget_encoder,
            self.p1.budget_decoder,
            self.detector,
            *self.p1.exploration_policies,
            *self.p1.execution_policies,
            *self.p1.execution_critics,
        ]
        for m in modules:
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

        self.env = vmas.make_env(
            scenario=scenario_factory(), num_envs=1, device=self.device, continuous_actions=True, wrapper=None
        )

        self._regime_events: dict[int, list[RegimeChangeEvent]] = {}
        self._attack_events: dict[int, list[AdversarialAttackEvent]] = {}
        self.reset_state()

    def schedule(self, events: list[RegimeChangeEvent | AdversarialAttackEvent]) -> None:
        for event in events:
            table = self._regime_events if isinstance(event, RegimeChangeEvent) else self._attack_events
            table.setdefault(event.step, []).append(event)

    def reset_state(self) -> None:
        self.obs = self.env.reset()
        self.mode = ["explore"] * self.n_agents
        self.k = [0] * self.n_agents
        self.p_streak = [0] * self.n_agents  # consecutive steps with detector_p >= threshold_C
        self.posterior_state = self.p1.posterior.init_state((1, self.n_agents), device=self.device)
        self.active_attacks = {i: {"position": False, "action": False} for i in range(self.n_agents)}
        # Rolling per-agent code history, capped at the detector's own
        # window: h_xi(c_i,1:t) only ever attends `window` steps back, so
        # keeping more than that is wasted compute, not extra correctness --
        # feeding exactly the last `window` codes and reading the final
        # position's output is equivalent to feeding the whole history.
        # Cleared on a per-agent reset (below): post-reset, the detector
        # should judge the *new* regime from scratch, not carry pre-reset
        # codes into its local window.
        self.code_history: list[list[torch.Tensor]] = [[] for _ in range(self.n_agents)]
        self.step_idx = 0

    def _own_belief(self, agent: int, mu_rho: torch.Tensor, sigma2_rho: torch.Tensor) -> torch.Tensor:
        mu_z = self.posterior_state.mu[:, agent]
        sigma2_z = self.posterior_state.sigma2[:, agent]
        return flatten_belief(mu_z, sigma2_z, mu_rho, sigma2_rho)

    def step(self, render: bool = True) -> StepResult:
        cfg = self.cfg
        t = self.step_idx

        for event in self._regime_events.get(t, []):
            apply_regime(self.env.agents[event.agent], event.regime)
        for event in self._attack_events.get(t, []):
            self.active_attacks[event.agent] = {"position": event.attack_position, "action": event.attack_action}

        with torch.no_grad():
            x_exe = torch.cat([self.obs[0], torch.zeros(1, self.p1.action_dim, device=self.device)], dim=-1).unsqueeze(1)
            mu_rho, sigma2_rho = self.p1.budget_encoder(x_exe)

        actions: list[torch.Tensor] = []
        reset_fired = [False] * self.n_agents
        detector_p: list[Optional[float]] = [None] * self.n_agents

        for i in range(self.n_agents):
            obs_i = self.obs[i]
            if self.mode[i] == "explore":
                with torch.no_grad():
                    action, _, _ = self.p1.exploration_policies[i].act(obs_i)
                self.k[i] += 1
            else:
                belief = self._own_belief(i, mu_rho, sigma2_rho)

                if self.active_attacks[i]["position"]:

                    def q_for_position(candidate_pos: torch.Tensor) -> torch.Tensor:
                        perturbed_obs = torch.cat([candidate_pos, obs_i[:, 2:]], dim=-1)
                        a, _, _ = self.p1.execution_policies[i].sample(torch.cat([perturbed_obs, belief], dim=-1))
                        q1, q2 = self.p1.execution_critics[i].q(obs_i, a, belief)
                        return torch.minimum(q1, q2)

                    attacked_pos = perturb_position(q_for_position, obs_i[:, :2], cfg.eps_pos, cfg.beta_pos, cfg.pgd_steps)
                    obs_i = torch.cat([attacked_pos, obs_i[:, 2:]], dim=-1)

                with torch.no_grad():
                    action, _, _ = self.p1.execution_policies[i].sample(torch.cat([obs_i, belief], dim=-1))

                if self.active_attacks[i]["action"]:

                    def q_for_action(candidate_action: torch.Tensor) -> torch.Tensor:
                        q1, q2 = self.p1.execution_critics[i].q(obs_i, candidate_action, belief)
                        return torch.minimum(q1, q2)

                    action = perturb_own_action(q_for_action, action, cfg.eps_action, cfg.beta_action, cfg.pgd_steps)
                    # An actuator attack still has to issue a physically valid
                    # command -- the L_inf perturbation has no reason to
                    # respect the environment's action bounds on its own
                    # (the nominal action is already inside them via the
                    # policy's own tanh squashing, but nominal +/- eps can
                    # step back out), so clip to the agent's actual u_range
                    # rather than let VMAS reject it.
                    u_range = self.env.agents[i].u_range
                    action = action.clamp(min=-u_range, max=u_range)

            actions.append(action)

        next_obs, rewards, dones, _infos = self.env.step(actions)

        with torch.no_grad():
            cond = torch.cat([torch.stack(list(self.obs), dim=1), torch.stack(actions, dim=1)], dim=-1)
            y = torch.cat([torch.stack(rewards, dim=1).unsqueeze(-1), torch.stack(list(next_obs), dim=1)], dim=-1)
            codes, _ = self.p1.flow(y, cond)  # (1, n_agents, code_dim)

            for i in range(self.n_agents):
                agent_mask = torch.tensor([[j == i for j in range(self.n_agents)]], device=self.device)
                self.posterior_state = self.p1.posterior.step_where(self.posterior_state, codes, agent_mask)

                self.code_history[i].append(codes[:, i])  # each (1, code_dim)
                if len(self.code_history[i]) > self.detector.window:
                    self.code_history[i] = self.code_history[i][-self.detector.window :]
                code_seq_i = torch.stack(self.code_history[i], dim=1)  # (1, min(t, window), code_dim)
                p_i = self.detector(code_seq_i)[0, -1].item()
                detector_p[i] = p_i

                self.p_streak[i] = self.p_streak[i] + 1 if p_i >= cfg.threshold_C else 0
                if self.p_streak[i] >= cfg.trigger_persistence:
                    self.posterior_state = self.p1.posterior.reset_where(self.posterior_state, agent_mask)
                    self.mode[i] = "explore"
                    self.k[i] = 0
                    self.code_history[i] = []
                    self.p_streak[i] = 0
                    reset_fired[i] = True

                if self.mode[i] == "explore" and (
                    self.k[i] >= cfg.re_exploration_budget_K or self.posterior_state.sigma2[0, i].item() <= cfg.sigma2_min
                ):
                    self.mode[i] = "execute"

        frame = self.env.render(mode="rgb_array", env_index=0) if render else None
        self.obs = next_obs
        self.step_idx += 1

        return StepResult(
            step=t,
            frame=frame,
            rewards=[r[0].item() for r in rewards],
            mode=list(self.mode),
            detector_p=detector_p,
            reset_fired=reset_fired,
            active_attacks=[self.active_attacks[i]["position"] or self.active_attacks[i]["action"] for i in range(self.n_agents)],
        )
