import torch
from vmas.simulator.core import Landmark, Sphere, World, Agent, Color
from vmas.simulator.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device, **kwargs):
        world = World(
            batch_dim=batch_dim,
            device=device,
        )
        self._world = world

        self.target = Landmark(
            name="target",
            collide=False,
            movable=False,
            shape=Sphere(radius=0.2),
        )
        world.add_landmark(self.target)
        agent = Agent(name="agent_0",shape=Sphere(radius=2),color=Color.BLUE)
        world.add_agent(agent)
        return world

    def reset_world_at(self, env_index=None):
        self._world.agents[0].set_pos(torch.tensor([-3,1]),batch_index= env_index)
        return
        

    def reward(self, agent):
        return torch.zeros(self.world.batch_dim, device=self.world.device)

    def observation(self, agent):
        return torch.zeros((self.world.batch_dim, 1), device=self.world.device)
