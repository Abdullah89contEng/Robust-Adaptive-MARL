"""Per-agent environment regime sampling and application (method-spec.md §2).

A "regime" mu is realized here as a `(mass, linear_friction)` pair — reusing
this project's existing physics-parameter framing (`PhysicsTaskSampler`)
rather than inventing a new non-stationarity mechanism. Both are plain
mutable Python attributes on a live VMAS `Agent` (`agent.mass`,
`agent.linear_friction`), not batched per vectorized env, so switching an
agent's regime mid-rollout changes it for every environment in the current
batch at once. That's consistent with §10 Phase 1's pseudocode, which
samples one regime schedule per *training iteration* (not per env within
the batch) and rolls the whole vectorized batch out against it — batching
here buys sample efficiency for one task, not task diversity within a
batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from vmas.simulator.core import Agent


@dataclass
class Regime:
    mass: float
    linear_friction: float


def sample_regime(mass_range: tuple[float, float], friction_range: tuple[float, float], rng: torch.Generator | None = None) -> Regime:
    mass = torch.empty(1).uniform_(*mass_range, generator=rng).item()
    friction = torch.empty(1).uniform_(*friction_range, generator=rng).item()
    return Regime(mass=mass, linear_friction=friction)


def apply_regime(agent: Agent, regime: Regime) -> None:
    agent.mass = regime.mass
    agent.linear_friction = regime.linear_friction


@dataclass
class EpisodeRegimeSchedule:
    """One training iteration's regime draw for every agent (§2)."""

    mu_1: list[Regime]  # per agent, first regime
    mu_2: list[Regime]  # per agent, second regime
    switch_time: torch.Tensor  # (n_agents,) long, in [1, H] -- vartheta_i_tilde

    def active_regime(self, agent_index: int, t: int) -> Regime:
        return self.mu_1[agent_index] if t < self.switch_time[agent_index].item() else self.mu_2[agent_index]


def sample_episode_schedule(
    n_agents: int,
    horizon: int,
    p_sw: float,
    mass_range: tuple[float, float],
    friction_range: tuple[float, float],
    rng: torch.Generator | None = None,
) -> EpisodeRegimeSchedule:
    """for each agent i: mu_1i, mu_2i ~ p(mu); vartheta_i ~ GEOM(p_sw);
    vartheta_i_tilde = min(vartheta_i, H)  (§2)."""
    mu_1 = [sample_regime(mass_range, friction_range, rng) for _ in range(n_agents)]
    mu_2 = [sample_regime(mass_range, friction_range, rng) for _ in range(n_agents)]
    # torch has no direct Geometric sampler taking a generator kwarg pre-2.x-consistently;
    # invert the CDF of Geometric(p_sw) supported on {1, 2, ...} from a uniform draw instead.
    u = torch.empty(n_agents).uniform_(1e-6, 1.0, generator=rng)
    vartheta = torch.ceil(torch.log(u) / torch.log(torch.tensor(1.0 - p_sw))).long().clamp(min=1)
    switch_time = torch.clamp(vartheta, max=horizon)
    return EpisodeRegimeSchedule(mu_1=mu_1, mu_2=mu_2, switch_time=switch_time)
