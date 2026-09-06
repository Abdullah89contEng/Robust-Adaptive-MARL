"""DrIGM exemplar (thesis Ch.5.6/9): rho-contamination robust VDN/QMIX on this
repo's VMAS search-and-rescue Scenario, built on top of BenchMARL/TorchRL.

third-party/robust-coMARL implements the same robust Bellman target
(`target = reward + gamma * (1 - rho) * next_max_q * (1 - done)`) but is
hard-wired to SustainGym's building-energy env (discrete MultiDiscrete
actions, a fixed obs_dim, a single-env python training loop). Rather than
adapting that env-specific code, this module gets the identical robust
target by pointing BenchMARL's own VDN/QMIX (which already wrap
torchrl's IGM-preserving VDNMixer/QMixer) at an effective discount of
`gamma * (1 - rho)`: TD0's target is `reward + gamma * next_value *
(1 - done)`, so scaling gamma reproduces the rho-contamination target
exactly, without reimplementing the loss.
"""

from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Any, Callable, Dict, Optional, Type

from torchrl.envs import EnvBase
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.objectives import ValueEstimators

from benchmarl.algorithms.common import Algorithm
from benchmarl.algorithms.qmix import Qmix, QmixConfig
from benchmarl.algorithms.vdn import Vdn, VdnConfig
from benchmarl.environments.vmas.common import VmasClass
from benchmarl.utils import DEVICE_TYPING

from .scenario import Scenario


class RescueTaskClass(VmasClass):
    """Wraps this repo's Scenario as a BenchMARL VMAS task.

    ``VmasClass.get_env_fun`` resolves ``scenario`` as a name string
    against VMAS's built-in scenario registry, which our Scenario isn't
    part of. A Scenario instance also can't be reused across the several
    VmasEnv instances BenchMARL builds (train env, eval env): it is
    mutated in place by ``make_world``/``reset_world_at``. So this builds
    a fresh ``Scenario(**scenario_kwargs)`` inside the returned factory,
    once per env construction, instead.

    ``self.config`` holds both ``max_steps`` (read directly by the base
    class's ``max_steps()``) and the kwargs forwarded to ``Scenario()``.
    """

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        scenario_kwargs = {k: v for k, v in self.config.items() if k != "max_steps"}
        max_steps = self.config.get("max_steps", 200)

        def make() -> EnvBase:
            return VmasEnv(
                scenario=Scenario(**scenario_kwargs),
                num_envs=num_envs,
                continuous_actions=continuous_actions,
                seed=seed,
                device=device,
                categorical_actions=True,
                clamp_actions=True,
                max_steps=max_steps,
            )

        return make


def rescue_task(config: Optional[Dict[str, Any]] = None) -> RescueTaskClass:
    """Build a BenchMARL task for this repo's Scenario.

    Example:
        >>> task = rescue_task({"config_file": "world_config.yaml", "max_steps": 200})
        >>> experiment = Experiment(task=task, algorithm_config=..., ...)
    """
    return RescueTaskClass(name="rescue", config=dict(config or {}))


def _robust_gamma(experiment_config, rho: float) -> float:
    """The rho-contamination target's effective discount: gamma * (1 - rho)."""
    return experiment_config.gamma * (1.0 - rho)


class RobustQmix(Qmix):
    """QMIX with DrIGM's rho-contamination robust Bellman target.

    Everything (mixer, IGM, exploration, replay) is identical to
    BenchMARL's Qmix; only the value estimator's discount changes, from
    `gamma` to `gamma * (1 - rho)`, matching robust-coMARL's
    `target = reward + gamma * (1 - rho) * next_max_q * (1 - done)`.
    """

    def __init__(self, rho: float, **kwargs):
        super().__init__(**kwargs)
        self.rho = rho

    def _get_loss(self, group, policy_for_loss, continuous):
        loss_module, use_target = super()._get_loss(group, policy_for_loss, continuous)
        loss_module.make_value_estimator(
            ValueEstimators.TD0, gamma=_robust_gamma(self.experiment_config, self.rho)
        )
        return loss_module, use_target


@dataclass
class RobustQmixConfig(QmixConfig):
    """Configuration for :class:`RobustQmix`. ``rho`` is the contamination budget."""

    rho: float = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return RobustQmix


class RobustVdn(Vdn):
    """VDN with DrIGM's rho-contamination robust Bellman target (see RobustQmix)."""

    def __init__(self, rho: float, **kwargs):
        super().__init__(**kwargs)
        self.rho = rho

    def _get_loss(self, group, policy_for_loss, continuous):
        loss_module, use_target = super()._get_loss(group, policy_for_loss, continuous)
        loss_module.make_value_estimator(
            ValueEstimators.TD0, gamma=_robust_gamma(self.experiment_config, self.rho)
        )
        return loss_module, use_target


@dataclass
class RobustVdnConfig(VdnConfig):
    """Configuration for :class:`RobustVdn`. ``rho`` is the contamination budget."""

    rho: float = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return RobustVdn
