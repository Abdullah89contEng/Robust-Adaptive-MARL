from vmas.simulator.core import Agent
import torch


class EaredAgent(Agent):

    @property
    def ear_offset(self):
        """Distance from the center to each ear."""
        return self.shape.radius 

    def get_ear_distance(self, victim_pos: torch.Tensor):
        """
        Args:
            victim_pos: Tensor of shape (batch_size, 2)

        Returns:
            left_distance:  (batch_size,)
            right_distance: (batch_size,)
        """

        theta = self.state.rot.squeeze(-1)

        # Unit vector pointing to the agent's left
        left_normal = torch.stack(
            (
                -torch.sin(theta),
                 torch.cos(theta),
            ),
            dim=-1,
        )

        left_ear = self.state.pos + self.ear_offset * left_normal
        right_ear = self.state.pos - self.ear_offset * left_normal

        left_distance = torch.norm(victim_pos - left_ear, dim=-1)
        right_distance = torch.norm(victim_pos - right_ear, dim=-1)

        return left_distance, right_distance