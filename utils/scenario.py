from __future__ import annotations

import random

import torch
from typing import cast

from vmas.simulator.core import Agent, Box, Landmark, Sphere, World, Color
from vmas.simulator.scenario import BaseScenario
#from vmas.simulator.utils import Color

from .map import Map, make_map_from_file
from .ear import EaredAgent 
from .task_sampler import PhysicsTask
from .victim import Victim
class Scenario(BaseScenario):
	def __init__(
		self,
		config_file: str = "./world_config.yaml",
		map_: Map | None = None,
		task: PhysicsTask | None = None,
	):
		super().__init__()
		self.map = map_ if map_ is not None else make_map_from_file(config_file)
		# One Scenario instance represents one MAML task. Keep this immutable
		# while collecting its support and query trajectories.
		self.task = task
		self.render_origin = torch.tensor([self.map.width / 2, self.map.height / 2], dtype=torch.float32)

	def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
		dt = kwargs.get("dt", 0.1)
		drag = kwargs.get("drag", self.task.drag if self.task is not None else 0.25)
		world = World(
			batch_dim=batch_dim,
			device=device,
			dt=dt,
			drag=drag,
			x_semidim=self.map.width / 2,
			y_semidim=self.map.height / 2,
			dim_c=0,
		)

		# VMAS entities are batched: create each object once, then reset its
		# state for all environments (or one selected environment).
		for obs in self.map.obstacles:
			world.add_landmark(Landmark(name=obs.name, shape=obs.shape, color=Color.BLACK))
		agent_physics = {}
		if self.task is not None:
			agent_physics = {
				"mass": self.task.agent_mass,
				"linear_friction": self.task.linear_friction,
				"angular_friction": self.task.angular_friction,
			}
		for ag in self.map.agents:
			world.add_agent(
				EaredAgent(name=ag.name, shape=ag.shape, color=ag.color, **agent_physics)
			)
		for vic in self.map.victims:
			world.add_landmark(
				Victim(
					name=vic.name,
					shape=vic.shape,
					movable=False,
					collide=False,
					color=Color.PINK,
				)
			)
		return world

	def reset_world_at(self, env_index: int | None = None):
		obstacle_count = len(self.map.obstacles)
		agent_count = len(self.map.agents)
		offset = torch.tensor(
			[self.map.width / 2.0, self.map.height / 2.0],
			device=self.world.device,
		)
		zero_pos = torch.zeros(2, device=self.world.device)
		zero_rot = torch.zeros(1, device=self.world.device)

		# Preserve the map's reset behaviour while keeping VMAS entities intact.
		if env_index is None:
			self.map.reset_agents()

		for landmark, obs in zip(self.world.landmarks[:obstacle_count], self.map.obstacles):
			landmark.set_pos(pos=obs.position.to(self.world.device) - offset, batch_index=env_index)
			landmark.set_vel(vel=zero_pos, batch_index=env_index)
			landmark.set_rot(
				rot=torch.tensor([obs.angle], device=self.world.device),
				batch_index=env_index,
			)
			landmark.set_ang_vel(ang_vel=zero_rot, batch_index=env_index)

		for agent, ag in zip(self.world.agents[:agent_count], self.map.agents):
			agent.set_pos(pos=ag.position.to(self.world.device) - offset, batch_index=env_index)
			agent.set_vel(vel=zero_pos, batch_index=env_index)
			agent.set_rot(rot=zero_rot, batch_index=env_index)
			agent.set_ang_vel(ang_vel=zero_rot, batch_index=env_index)

		for victim, vic in zip(self.world.landmarks[obstacle_count:], self.map.victims):
			victim.set_pos(pos=vic.position.to(self.world.device) - offset, batch_index=env_index)
			victim.set_vel(vel=zero_pos, batch_index=env_index)
			victim.set_rot(rot=zero_rot, batch_index=env_index)
			victim.set_ang_vel(ang_vel=zero_rot, batch_index=env_index)

	def observation(self, agent: Agent):
		agent_pos = agent.state.pos
		agent_vel = agent.state.vel
		assert agent_pos is not None and agent_vel is not None

		obs_parts: list[torch.Tensor] = [agent_pos, agent_vel]
		for landmark in self.world.landmarks:
			landmark_pos = landmark.state.pos
			assert landmark_pos is not None
			obs_parts.append(landmark_pos - agent_pos)

		return torch.cat(obs_parts, dim=-1)

	def reward(self, agent: Agent):
		return torch.zeros(self.world.batch_dim, device=self.world.device)
