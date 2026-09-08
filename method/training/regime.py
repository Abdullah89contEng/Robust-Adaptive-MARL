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

import math
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


def regime_distance(a: Regime, b: Regime, mass_span: float, friction_span: float) -> float:
    """L2 distance between two regimes in (mass, friction) space, each axis
    normalized by its full range. Used to guarantee that a sampled switch
    mu_1 -> mu_2 is an actual regime change, not two draws from the same blob."""
    return math.hypot(
        (a.mass - b.mass) / max(mass_span, 1e-6),
        (a.linear_friction - b.linear_friction) / max(friction_span, 1e-6),
    )


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


DEFAULT_MIN_SWITCH_DIST = 0.35


def sample_far_regime(
    mu_1: Regime,
    mode_1: int,
    modes: tuple[RegimeMode, ...],
    mass_span: float,
    friction_span: float,
    min_switch_dist: float = DEFAULT_MIN_SWITCH_DIST,
    rng: torch.Generator | None = None,
    max_tries: int = 25,
) -> tuple[Regime, int]:
    """Draw a post-switch regime that is a *genuine* change from ``mu_1``:
    a different mixture mode (when the mixture has >= 2 components) AND at
    least ``min_switch_dist`` away in range-normalized (mass, friction)
    space. Rejection-samples up to ``max_tries``; the disjoint DEFAULT_MODES
    boundaries can still sit closer than the threshold, so the distance
    check is not redundant with the mode check. Returns (regime, mode_index).
    """
    multi = len(modes) >= 2
    r2, k2 = sample_regime_from_modes(modes, rng)
    for _ in range(max_tries):
        mode_ok = (not multi) or (k2 != mode_1)
        if mode_ok and regime_distance(mu_1, r2, mass_span, friction_span) >= min_switch_dist:
            return r2, k2
        r2, k2 = sample_regime_from_modes(modes, rng)
    return r2, k2


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
    min_switch_dist: float = DEFAULT_MIN_SWITCH_DIST,
) -> EpisodeRegimeSchedule:
    """for each agent i: mu_1i, mu_2i ~ p(mu); vartheta_i ~ GEOM(p_sw);
    vartheta_i_tilde = min(vartheta_i, H)  (§2).

    p(mu) is the `modes` mixture (default `DEFAULT_MODES`); pass `modes=()`
    to use the flat U(mass_range) x U(friction_range) box instead.

    SEPARATION GUARANTEE: for every agent that *actually switches within the
    horizon* (vartheta_i_tilde < H), mu_2 is resampled until it is (a) a
    different mixture mode than mu_1 and (b) at least `min_switch_dist` away
    in range-normalized (mass, friction) space. Two independent draws from a
    3-mode mixture land in the same mode ~1/3 of the time; without this a
    labelled "switch" is often no dynamics change at all, which is noise for
    both the encoder and the change-point detector. Agents that do not
    switch (vartheta_i_tilde == H) are left untouched -- mu_2 is never
    applied for them.
    """
    if modes is None:
        modes = DEFAULT_MODES
    mass_span = mass_range[1] - mass_range[0]
    fric_span = friction_range[1] - friction_range[0]

    # switch times first, so the separation guarantee is only enforced where
    # a switch actually occurs.
    u = torch.empty(n_agents).uniform_(1e-6, 1.0, generator=rng)
    vartheta = torch.ceil(torch.log(u) / torch.log(torch.tensor(1.0 - p_sw))).long().clamp(min=1)
    switch_time = torch.clamp(vartheta, max=horizon)
    switches = (switch_time < horizon).tolist()

    if modes:
        drawn_1 = [sample_regime_from_modes(modes, rng) for _ in range(n_agents)]
        mu_1 = [r for r, _ in drawn_1]
        mode_1 = torch.tensor([k for _, k in drawn_1], dtype=torch.long)
        mu_2, mode_2 = [], []
        for i in range(n_agents):
            if switches[i]:
                r2, k2 = sample_far_regime(mu_1[i], int(mode_1[i]), modes, mass_span, fric_span, min_switch_dist, rng)
            else:
                r2, k2 = sample_regime_from_modes(modes, rng)  # unused (mu_2 never applied)
            mu_2.append(r2)
            mode_2.append(k2)
        mode_2 = torch.tensor(mode_2, dtype=torch.long)
    else:
        mu_1 = [sample_regime(mass_range, friction_range, rng) for _ in range(n_agents)]
        mu_2 = []
        for i in range(n_agents):
            r2 = sample_regime(mass_range, friction_range, rng)
            for _ in range(50):
                if not switches[i] or regime_distance(mu_1[i], r2, mass_span, fric_span) >= min_switch_dist:
                    break
                r2 = sample_regime(mass_range, friction_range, rng)
            mu_2.append(r2)
        mode_1 = torch.full((n_agents,), -1, dtype=torch.long)
        mode_2 = torch.full((n_agents,), -1, dtype=torch.long)

    return EpisodeRegimeSchedule(mu_1=mu_1, mu_2=mu_2, switch_time=switch_time, mode_1=mode_1, mode_2=mode_2)
