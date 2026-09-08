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

TASK DISTRIBUTION — meta-training draws mu from a finite mixture
`p(mu) = sum_k pi_k p_k(mu)` over a few interpretable dynamics regimes
(`DEFAULT_MODES`: icy / normal / heavy, each a box in (mass, friction)
space, well separated in both axes), not from one flat
U(mass_range) x U(friction_range) box. Two reasons: (i) it gives reproducible, nameable meta-test conditions
(the eval harness's `NOMINAL`/`SHIFTED` become specific modes), and (ii) the
contrastive encoder term (method-spec.md §3) needs a ground-truth regime
label per transition to build positives (same mode) and negatives
(different mode) — a mixture supplies that label for free, where a flat box
supplies only a continuous vector and forces a distance threshold.
Pass `modes=()` to `sample_episode_schedule` to recover the old flat box
(regime-mode ids are then -1 and the encoder falls back to a distance rule).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from vmas.simulator.core import Agent


@dataclass
class Regime:
    mass: float
    linear_friction: float


@dataclass(frozen=True)
class RegimeMode:
    """One mixture component of the meta-training task distribution."""

    name: str
    mass_range: tuple[float, float]
    friction_range: tuple[float, float]
    weight: float = 1.0


# icy / normal / heavy: three qualitatively distinct dynamics regimes,
# separated in BOTH mass and linear friction so that the transition
# (o, a, r, o') visibly differs between them even for a slow-moving agent
# -- a narrow gap (the earlier 0.85-1.15 mass, 0.0-0.15 friction spread)
# left the regimes nearly indistinguishable in code space, so the
# change-point detector had no signal to learn (probe AUC ~0.56).
DEFAULT_MODES: tuple[RegimeMode, ...] = (
    RegimeMode("icy",    (0.70, 0.90), (0.00, 0.04)),
    RegimeMode("normal", (0.95, 1.25), (0.12, 0.20)),
    RegimeMode("heavy",  (1.35, 1.70), (0.30, 0.42)),
)


def regime_vector(regime: Regime) -> tuple[float, float]:
    """The (mass, linear_friction) coordinates used for regime-space distance."""
    return (regime.mass, regime.linear_friction)


def sample_regime(mass_range: tuple[float, float], friction_range: tuple[float, float], rng: torch.Generator | None = None) -> Regime:
    mass = torch.empty(1).uniform_(*mass_range, generator=rng).item()
    friction = torch.empty(1).uniform_(*friction_range, generator=rng).item()
    return Regime(mass=mass, linear_friction=friction)


def sample_regime_from_modes(modes: tuple[RegimeMode, ...], rng: torch.Generator | None = None) -> tuple[Regime, int]:
    """Draw a mixture component by weight, then a regime uniformly inside it.
    Returns (regime, mode_index)."""
    weights = torch.tensor([m.weight for m in modes], dtype=torch.float)
    k = int(torch.multinomial(weights, 1, generator=rng).item())
    mode = modes[k]
    return sample_regime(mode.mass_range, mode.friction_range, rng), k


def apply_regime(agent: Agent, regime: Regime) -> None:
    agent.mass = regime.mass
    agent.linear_friction = regime.linear_friction


@dataclass
class EpisodeRegimeSchedule:
    """One training iteration's regime draw for every agent (§2)."""

    mu_1: list[Regime]  # per agent, first regime
    mu_2: list[Regime]  # per agent, second regime
    switch_time: torch.Tensor  # (n_agents,) long, in [1, H] -- vartheta_i_tilde
    mode_1: torch.Tensor | None = None  # (n_agents,) long, mixture component of mu_1 (-1 if flat box)
    mode_2: torch.Tensor | None = None  # (n_agents,) long, mixture component of mu_2 (-1 if flat box)

    def active_regime(self, agent_index: int, t: int) -> Regime:
        return self.mu_1[agent_index] if t < self.switch_time[agent_index].item() else self.mu_2[agent_index]


def sample_episode_schedule(
    n_agents: int,
    horizon: int,
    p_sw: float,
    mass_range: tuple[float, float],
    friction_range: tuple[float, float],
    rng: torch.Generator | None = None,
    modes: tuple[RegimeMode, ...] | None = None,
) -> EpisodeRegimeSchedule:
    """for each agent i: mu_1i, mu_2i ~ p(mu); vartheta_i ~ GEOM(p_sw);
    vartheta_i_tilde = min(vartheta_i, H)  (§2).

    p(mu) is the `modes` mixture (default `DEFAULT_MODES`); pass `modes=()`
    to use the flat U(mass_range) x U(friction_range) box instead.
    """
    if modes is None:
        modes = DEFAULT_MODES

    if modes:
        drawn_1 = [sample_regime_from_modes(modes, rng) for _ in range(n_agents)]
        drawn_2 = [sample_regime_from_modes(modes, rng) for _ in range(n_agents)]
        mu_1 = [r for r, _ in drawn_1]
        mu_2 = [r for r, _ in drawn_2]
        mode_1 = torch.tensor([k for _, k in drawn_1], dtype=torch.long)
        mode_2 = torch.tensor([k for _, k in drawn_2], dtype=torch.long)
    else:
        mu_1 = [sample_regime(mass_range, friction_range, rng) for _ in range(n_agents)]
        mu_2 = [sample_regime(mass_range, friction_range, rng) for _ in range(n_agents)]
        mode_1 = torch.full((n_agents,), -1, dtype=torch.long)
        mode_2 = torch.full((n_agents,), -1, dtype=torch.long)

    # torch has no direct Geometric sampler taking a generator kwarg pre-2.x-consistently;
    # invert the CDF of Geometric(p_sw) supported on {1, 2, ...} from a uniform draw instead.
    u = torch.empty(n_agents).uniform_(1e-6, 1.0, generator=rng)
    vartheta = torch.ceil(torch.log(u) / torch.log(torch.tensor(1.0 - p_sw))).long().clamp(min=1)
    switch_time = torch.clamp(vartheta, max=horizon)
    return EpisodeRegimeSchedule(mu_1=mu_1, mu_2=mu_2, switch_time=switch_time, mode_1=mode_1, mode_2=mode_2)
