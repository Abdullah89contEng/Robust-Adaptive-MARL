from __future__ import annotations

import random

import torch
from typing import cast

from vmas.simulator.core import Agent, Box, Landmark, Sphere, World, Color
from vmas.simulator.scenario import BaseScenario
#from vmas.simulator.utils import Color

from .map import Map, make_map_from_file


class Scenario(BaseScenario):
	def __init__(self, config_file: str = "./world_config.yaml", map_: Map | None = None):
		super().__init__()
		self.map = map_ if map_ is not None else make_map_from_file(config_file)

	def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
		dt = kwargs.get("dt", 0.1)
		drag = kwargs.get("drag", 0.25)
		world = World(
			batch_dim=batch_dim,
			device=device,
			dt=dt,
			drag=drag,
			x_semidim=self.map.width / 2,
			y_semidim=self.map.height / 2,
			dim_c=0,
		)

		self._world = world

		offset = torch.tensor([self.map.width / 2.0, self.map.height / 2.0], device=device)
		batch_index_for_init = 0 if batch_dim == 1 else None
		colors = [Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW, Color.PURPLE, Color.YELLOW]
		for index, agent_info in enumerate(self.map.agents):
			agent = Agent(
				name=agent_info.name or f"agent_{index}",
				shape=Sphere(radius=agent_info.radius),
				movable=True,
				rotatable=False,
				collide=True,
				color=colors[index % len(colors)],
				silent=True,
			)
			world.add_agent(agent)
			# set initial position in world-centered coordinates
			p = agent_info.position.to(device)
			if (p >= 0).all() and (p[:,] <= torch.tensor([self.map.width, self.map.height], device=device)).all():
				pos_world = p - offset
			else:
				pos_world = p
			agent.set_pos(pos_world, batch_index=cast(int, batch_index_for_init))
			agent.set_vel(torch.zeros(2, device=device), batch_index=cast(int, batch_index_for_init))
			# verify the position was applied; if not, fall back to per-env assignment
			if pos_world.dim() == 1:
				expected = pos_world.unsqueeze(0).to(device)
			else:
				expected = pos_world.to(device)
			stored = agent.state.pos
			assert stored is not None
			if not torch.allclose(stored, expected, atol=1e-6):
				print(f"make_world: fallback per-env set_pos for agent {agent.name}")
				for bi in range(batch_dim):
					if pos_world.dim() == 1:
						agent.set_pos(pos_world, batch_index=bi)
					else:
						agent.set_pos(pos_world[bi], batch_index=bi)
				stored2 = agent.state.pos
				assert stored2 is not None
				if not torch.allclose(stored2, expected, atol=1e-6):
					raise AssertionError(f"Failed to set agent pos for {agent.name}: expected {expected}, got {stored2}")

		for index, obstacle_info in enumerate(self.map.obstacles):
			landmark = Landmark(
				name=obstacle_info.name or f"obstacle_{index}",
				shape=obstacle_info.shape,
				movable=False,
				rotatable=True,
				collide=True,
				color=Color.GRAY,
			)
			world.add_landmark(landmark)
			# set landmark position/rotation in world-centered coordinates
			q = obstacle_info.position.to(device)
			if (q >= 0).all() and (q[:,] <= torch.tensor([self.map.width, self.map.height], device=device)).all():
				pos_world = q - offset
			else:
				pos_world = q
			landmark.set_pos(pos_world, batch_index=cast(int, batch_index_for_init))
			landmark.set_rot(torch.tensor([obstacle_info.angle], device=device), batch_index=cast(int, batch_index_for_init))
			# verify and fallback per-env if needed
			if pos_world.dim() == 1:
				expected_lm = pos_world.unsqueeze(0).to(device)
			else:
				expected_lm = pos_world.to(device)
			stored_lm = landmark.state.pos
			assert stored_lm is not None
			if not torch.allclose(stored_lm, expected_lm, atol=1e-6):
				print(f"make_world: fallback per-env set_pos for landmark {landmark.name}")
				for bi in range(batch_dim):
					if pos_world.dim() == 1:
						landmark.set_pos(pos_world, batch_index=bi)
					else:
						landmark.set_pos(pos_world[bi], batch_index=bi)
				stored2_lm = landmark.state.pos
				assert stored2_lm is not None
				if not torch.allclose(stored2_lm, expected_lm, atol=1e-6):
					raise AssertionError(f"Failed to set landmark pos for {landmark.name}: expected {expected_lm}, got {stored2_lm}")

		return world

	def reset_world_at(self, env_index: int | None = None):
		zero_pos = torch.zeros(2, device=self.world.device)
		zero_rot = torch.zeros(1, device=self.world.device)

		# VMAS world coordinates are centered at (0,0) with semidims set in make_world.
		# Convert map positions (0..width, 0..height) to world-centered coords.
		offset = torch.tensor([self.map.width / 2.0, self.map.height / 2.0], device=self.world.device)

		for agent_index, (agent, agent_info) in enumerate(zip(self.world.agents, self.map.agents)):
			if self.map.free_cell:
				free_cell = random.choice(self.map.free_cell)
				pos_map = free_cell.to(self.world.device)
			else:
				pos_map = agent_info.position.to(self.world.device)
			pos_world = pos_map - offset
			batch_indices = [env_index] if env_index is not None else list(range(self.world.batch_dim))
			for batch_index in batch_indices:
				agent.set_pos(pos_world, batch_index=batch_index)
				agent.set_vel(zero_pos, batch_index=batch_index)
				agent.set_rot(zero_rot, batch_index=batch_index)
				agent.set_ang_vel(zero_rot, batch_index=batch_index)
			# verify and fallback per-env if needed
			expected = pos_world.unsqueeze(0) if pos_world.dim() == 1 else pos_world
			expected = expected.to(self.world.device)
			stored = agent.state.pos
			assert stored is not None
			if not torch.allclose(stored, expected, atol=1e-6):
				print(f"reset_world_at: fallback per-env set_pos for agent {agent.name}")
				for bi in batch_indices:
					agent.set_pos(pos_world, batch_index=bi)
				stored2 = agent.state.pos
				assert stored2 is not None
				if not torch.allclose(stored2, expected, atol=1e-6):
					raise AssertionError(f"Failed to set agent pos for {agent.name}: expected {expected}, got {stored2}")

		for landmark, obstacle_info in zip(self.world.landmarks, self.map.obstacles):
			pos_world = obstacle_info.position.to(self.world.device) - offset
			batch_indices = [env_index] if env_index is not None else list(range(self.world.batch_dim))
			for batch_index in batch_indices:
				landmark.set_pos(pos_world, batch_index=batch_index)
				landmark.set_vel(zero_pos, batch_index=batch_index)
				landmark.set_rot(torch.tensor([obstacle_info.angle], device=self.world.device), batch_index=batch_index)
				landmark.set_ang_vel(zero_rot, batch_index=batch_index)
			# verify and fallback per-env if needed
			expected_lm = pos_world.unsqueeze(0) if pos_world.dim() == 1 else pos_world
			expected_lm = expected_lm.to(self.world.device)
			stored_lm = landmark.state.pos
			assert stored_lm is not None
			if not torch.allclose(stored_lm, expected_lm, atol=1e-6):
				print(f"reset_world_at: fallback per-env set_pos for landmark {landmark.name}")
				for bi in batch_indices:
					landmark.set_pos(pos_world, batch_index=bi)
				stored2_lm = landmark.state.pos
				assert stored2_lm is not None
				if not torch.allclose(stored2_lm, expected_lm, atol=1e-6):
					raise AssertionError(f"Failed to set landmark pos for {landmark.name}: expected {expected_lm}, got {stored2_lm}")

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
