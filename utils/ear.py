from __future__ import annotations

import torch

from vmas.simulator.core import World
from vmas.simulator.sensors import Sensor

from .survival import Survival


class Ear(Sensor):
    """Binaural distance sensor.

    For each :class:`Survival` entity, returns the distance from the
    agent's left and right ear points to it. This is the only channel
    through which survivals are perceived (no raw relative position) —
    obstacles are sensed separately, via lidar.
    """

    def __init__(
        self,
        world: World,
        survivals: list[Survival],
        ear_offset: float | None = None,
        rescued_value: float = 0.0,
    ):
        super().__init__(world)
        # `survivals` is shared by reference with Scenario.make_world, which
        # populates it after every agent (and its Ear) has been constructed.
        self._survivals = survivals
        self._ear_offset = ear_offset
        # Distance reported for a survival once it has been rescued: a fixed
        # "out of earshot" value so a rescued victim also disappears from the
        # observation, not just from the render.
        self._rescued_value = rescued_value
        self._last_measurement: torch.Tensor | None = None

    def _ear_offset_value(self) -> float:
        """Distance from the agent's center to each ear."""
        return self._ear_offset if self._ear_offset is not None else self.agent.shape.radius

    def measure(self) -> torch.Tensor:
        """Returns (batch_dim, n_survivals * 2): a (left, right) distance pair per survival."""
        theta = self.agent.state.rot.squeeze(-1)

        # Unit vector pointing to the agent's left.
        left_normal = torch.stack((-torch.sin(theta), torch.cos(theta)), dim=-1)

        offset = self._ear_offset_value()
        left_ear = self.agent.state.pos + offset * left_normal
        right_ear = self.agent.state.pos - offset * left_normal

        if not self._survivals:
            measurement = torch.zeros(self._world.batch_dim, 0, device=self._world.device)
        else:
            per_survival = []
            for survival in self._survivals:
                survival_pos = survival.state.pos
                left_distance = torch.norm(survival_pos - left_ear, dim=-1)
                right_distance = torch.norm(survival_pos - right_ear, dim=-1)
                pair = torch.stack((left_distance, right_distance), dim=-1)  # (batch, 2)
                rescued = getattr(survival, "rescued", False)
                if torch.is_tensor(rescued):
                    pair = torch.where(
                        rescued.unsqueeze(-1),
                        torch.full_like(pair, self._rescued_value),
                        pair,
                    )
                per_survival.append(pair)
            measurement = torch.cat(per_survival, dim=-1)

        self._last_measurement = measurement
        return measurement

    def render(self, env_index: int = 0):
        return []

    def to(self, device: torch.device):
        pass
