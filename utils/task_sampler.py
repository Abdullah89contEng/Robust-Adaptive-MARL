"""Sampling of physics tasks for meta-reinforcement learning.

A task is a set of dynamics parameters that remains constant during an
episode (and during its MAML support/query rollouts).  Sample a new task for
each member of a meta-batch; do *not* resample it at every environment reset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterator

import random


@dataclass(frozen=True)
class PhysicsTask:
    """The dynamics defining one environment/task.

    ``drag`` belongs to the VMAS ``World``.  The other parameters are passed
    to each VMAS agent.  They are deliberately scalar: a task changes the
    dynamics of the complete world rather than giving every agent a different
    hidden task.
    """

    drag: float
    linear_friction: float
    angular_friction: float
    agent_mass: float

    def as_dict(self) -> dict[str, float]:
        """Return serialisable task metadata for experiment logs."""
        return asdict(self)


@dataclass(frozen=True)
class UniformRange:
    """Closed interval used by :class:`PhysicsTaskSampler`."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low < 0 or self.high < 0:
            raise ValueError("Physics parameter ranges must be non-negative")
        if self.low > self.high:
            raise ValueError("Range low must not be greater than high")

    def sample(self, rng: random.Random) -> float:
        return rng.uniform(self.low, self.high)


class PhysicsTaskSampler:
    """Reproducibly sample a distribution of VMAS physics tasks.

    The defaults are intentionally modest around the existing scenario's
    ``drag=0.25``.  Widen them only after a policy can solve this range.
    """

    def __init__(
        self,
        *,
        drag: UniformRange = UniformRange(0.05, 0.45),
        linear_friction: UniformRange = UniformRange(0.0, 0.15),
        angular_friction: UniformRange = UniformRange(0.0, 0.05),
        agent_mass: UniformRange = UniformRange(0.8, 1.2),
        seed: int | None = None,
    ) -> None:
        self.drag = drag
        self.linear_friction = linear_friction
        self.angular_friction = angular_friction
        self.agent_mass = agent_mass
        self._rng = random.Random(seed)

    def sample(self) -> PhysicsTask:
        """Sample one task. Its values should stay fixed for an episode."""
        return PhysicsTask(
            drag=self.drag.sample(self._rng),
            linear_friction=self.linear_friction.sample(self._rng),
            angular_friction=self.angular_friction.sample(self._rng),
            agent_mass=self.agent_mass.sample(self._rng),
        )

    def sample_batch(self, num_tasks: int) -> list[PhysicsTask]:
        """Sample the tasks for one MAML meta-batch."""
        if num_tasks < 1:
            raise ValueError("num_tasks must be at least one")
        return [self.sample() for _ in range(num_tasks)]

    def iter_tasks(self) -> Iterator[PhysicsTask]:
        """Yield an unbounded deterministic stream of tasks."""
        while True:
            yield self.sample()
