import torch
from vmas.simulator.core import Landmark


class Survival(Landmark):
    """A rescue target.

    ``health`` and ``rescued`` start as plain Python values here and are
    replaced with per-environment tensors of shape ``(batch_dim,)`` by
    ``Scenario.make_world`` once ``batch_dim`` is known (VMAS entities are
    vectorized across environments, so a single scalar cannot track whether
    each parallel environment has rescued this survival).
    """

    def __init__(
        self,
        required_rescuers: int = 2,
        health: float = 100,
        decay_rate: float = 0.2,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.required_rescuers = required_rescuers
        self.initial_health = health
        self.health = health
        self.decay_rate = decay_rate
        self.rescued = False

    def render(self, env_index: int = 0):
        """Emit no geometry for environments where this survival has already
        been rescued, so a rescued victim visually disappears from the frame.

        Before ``Scenario.make_world`` replaces ``rescued`` with a
        per-environment tensor it is still the scalar ``False`` (no batch to
        index); fall through to normal landmark rendering in that case.
        """
        rescued = getattr(self, "rescued", False)
        if torch.is_tensor(rescued) and bool(rescued[env_index]):
            return []
        return super().render(env_index)
