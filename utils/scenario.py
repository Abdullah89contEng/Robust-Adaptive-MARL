from __future__ import annotations

import math
import torch

from vmas.simulator.core import Agent, Landmark, World, Color
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar

from .map import Map, make_map_from_file
from .ear import Ear
from .task_sampler import PhysicsTask
from .survival import Survival


class Scenario(BaseScenario):
	def __init__(
		self,
		config_file: str = "./world_config.yaml",
		map_: Map | None = None,
		task: PhysicsTask | None = None,
		rescue_range: float = 0.5,
		rescue_reward: float = 10.0,
		decay_penalty: float = 0.1,
		time_penalty: float = 0.0,
		fit_map: bool = False,
		lidar_n_rays: int = 12,
		lidar_range: float = 10.0,
	):
		super().__init__()
		self.map = map_ if map_ is not None else make_map_from_file(config_file)
		# One Scenario instance represents one MAML task. Keep this immutable
		# while collecting its support and query trajectories.
		self.task = task
		# World coordinates are centered at the map's midpoint (entities are
		# placed at `map_position - offset` in reset_world_at), so the render
		# camera must be centered on the origin too, not on (width/2, height/2).
		self.render_origin = torch.tensor([0.0, 0.0], dtype=torch.float32)
		# VMAS's shared-viewer render auto-fits the camera to the agents'
		# spread, but the formula it uses effectively squares viewer_zoom
		# once it goes above ~1 (displayed half-extent ends up
		# `max(agent_extent, viewer_zoom ** 2)`), so leave it at
		# BaseScenario's default (1.2) unless `fit_map` asks to guarantee
		# the whole map (walls included, which auto-fit ignores) is visible.
		if fit_map:
			self.viewer_zoom = math.sqrt(max(self.map.width, self.map.height) / 2)

		# Rescue mechanics: a survival is rescued once one agent is within
		# `rescue_range` of it (Survival.required_rescuers is always 1).
		# Its health decays every step it spends unrescued, and stops
		# decaying once it is rescued or dead.
		self.rescue_range = rescue_range
		self.rescue_reward = rescue_reward
		self.decay_penalty = decay_penalty
		self.time_penalty = time_penalty
		self.lidar_n_rays = lidar_n_rays
		self.lidar_range = lidar_range
		self._survivals: list[Survival] = []

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

		# `self._survivals` is passed into each Ear below before it is
		# populated (survivals are created after agents); Ear holds the
		# same list by reference and only reads it inside measure(), by
		# which point the list has been filled in.
		self._survivals = []
		obstacle_filter = lambda e: isinstance(e, Landmark) and not isinstance(e, Survival)
		for ag in self.map.agents:
			world.add_agent(
				Agent(
					name=ag.name,
					shape=ag.shape,
					color=ag.color,
					sensors=[
						Lidar(
							world,
							n_rays=self.lidar_n_rays,
							max_range=self.lidar_range,
							entity_filter=obstacle_filter,
						),
						Ear(world, self._survivals),
					],
					**agent_physics,
				)
			)

		for sur in self.map.survivals:
			survival = Survival(
				required_rescuers=1,
				name=sur.name,
				shape=sur.shape,
				movable=False,
				collide=False,
				color=Color.PINK,
			)
			# Survival.health/rescued start as plain Python values; VMAS is
			# vectorized across batch_dim environments, so each needs its own
			# per-environment tensor once batch_dim is known.
			survival.health = torch.full((batch_dim,), survival.initial_health, device=device)
			survival.rescued = torch.zeros(batch_dim, dtype=torch.bool, device=device)
			world.add_landmark(survival)
			self._survivals.append(survival)

		n_agents = len(self.map.agents)
		self._agent_rescue_bonus = torch.zeros(batch_dim, n_agents, device=device)
		self._shared_reward = torch.zeros(batch_dim, device=device)

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

		for survival, sur in zip(self._survivals, self.map.survivals):
			survival.set_pos(pos=sur.position.to(self.world.device) - offset, batch_index=env_index)
			survival.set_vel(vel=zero_pos, batch_index=env_index)
			survival.set_rot(rot=zero_rot, batch_index=env_index)
			survival.set_ang_vel(ang_vel=zero_rot, batch_index=env_index)
			if env_index is None:
				survival.health[:] = survival.initial_health
				survival.rescued[:] = False
			else:
				survival.health[env_index] = survival.initial_health
				survival.rescued[env_index] = False

		if env_index is None:
			self._agent_rescue_bonus[:] = 0.0
			self._shared_reward[:] = 0.0
		else:
			self._agent_rescue_bonus[env_index] = 0.0
			self._shared_reward[env_index] = 0.0

	def observation(self, agent: Agent):
		agent_pos = agent.state.pos
		agent_vel = agent.state.vel
		agent_ang_vel = agent.state.ang_vel
		assert agent_pos is not None and agent_vel is not None and agent_ang_vel is not None

		# Obstacles are sensed via lidar (sensors[0]): one bounded-range
		# distance per ray, not an oracle relative position. Survivals are
		# only perceived through the agent's binaural ears (sensors[1]): a
		# (left, right) distance pair per survival, never a raw position.
		lidar, ear = agent.sensors
		obs_parts: list[torch.Tensor] = [
			agent_pos,
			agent_vel,
			agent_ang_vel,
			lidar.measure(),
			ear.measure(),
		]

		return torch.cat(obs_parts, dim=-1)

	def post_step(self):
		"""Resolve survival rescue/decay once per physics step (not once per agent)."""
		n_agents = len(self.world.agents)
		self._agent_rescue_bonus = torch.zeros(self.world.batch_dim, n_agents, device=self.world.device)
		total_decay = torch.zeros(self.world.batch_dim, device=self.world.device)

		if self._survivals:
			agents_pos = torch.stack([a.state.pos for a in self.world.agents], dim=1)  # (batch, n_agents, 2)
			survivals_pos = torch.stack([s.state.pos for s in self._survivals], dim=1)  # (batch, n_survivals, 2)
			dists = torch.cdist(agents_pos, survivals_pos)  # (batch, n_agents, n_survivals)
			in_range = dists < self.rescue_range
			rescuers_present = in_range.sum(dim=1)  # (batch, n_survivals)

			for i, survival in enumerate(self._survivals):
				alive = survival.health > 0
				active = (~survival.rescued) & alive
				newly_rescued = active & (rescuers_present[:, i] >= survival.required_rescuers)

				# Reward is proportional to the survival's remaining health
				# at the moment of rescue, so waiting (health decays every
				# step it's left unrescued) directly costs reward, not just
				# via the separate decay penalty below.
				contributed = in_range[:, :, i] & newly_rescued.unsqueeze(-1)
				health_fraction = survival.health / survival.initial_health
				self._agent_rescue_bonus += contributed.float() * self.rescue_reward * health_fraction.unsqueeze(-1)

				decaying = active & ~newly_rescued
				decayed_health = (survival.health - survival.decay_rate).clamp(min=0.0)
				total_decay += torch.where(decaying, survival.health - decayed_health, torch.zeros_like(total_decay))
				survival.health = torch.where(decaying, decayed_health, survival.health)

				survival.rescued = survival.rescued | newly_rescued

		self._shared_reward = -self.decay_penalty * total_decay + self.time_penalty

	def reward(self, agent: Agent):
		agent_index = self.world.agents.index(agent)
		return self._agent_rescue_bonus[:, agent_index] + self._shared_reward

	def done(self):
		if not self._survivals:
			return torch.zeros(self.world.batch_dim, dtype=torch.bool, device=self.world.device)
		resolved = [survival.rescued | (survival.health <= 0) for survival in self._survivals]
		return torch.stack(resolved, dim=-1).all(dim=-1)

	def info(self, agent: Agent):
		if not self._survivals:
			return {}
		rescued = torch.stack([survival.rescued for survival in self._survivals], dim=-1)
		return {"survivals_rescued": rescued.sum(dim=-1)}
