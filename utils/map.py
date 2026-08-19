"""
Map utilities.

Coordinate convention: map coordinates use the outermost boundary as the origin.
That is, (0.0, 0.0) is the bottom-left corner of the full map including walls;
interior positions (agents/obstacles) are placed with that origin and avoid
the boundary walls according to `wall_thickness`.
"""

import math
import random

import torch
import yaml

from torch import Tensor
from vmas.simulator.core import Box, Color, Shape, Sphere


class Map:
    class entity_info:
        def __init__(self, name: str = ""):
            self.name = name
            self.shape: Shape

    class AgentInfo(entity_info):
        def __init__(self, radius: float, position: Tensor | None = None, name: str = "", color: Color = Color.RED):
            super().__init__(name)
            self.radius = radius
            self.position = position if position is not None else torch.zeros(2, dtype=torch.float32)
            self.shape: Shape = Sphere(radius=radius)
            self.color: Color = color

    class ObstacleInfo(entity_info):
        def __init__(
            self,
            shape: Shape,
            position: Tensor,
            angle: float,
            is_boundary: bool = False,
            name: str = "",
        ):
            super().__init__(name)
            self.shape = shape
            self.position = position
            self.angle = angle
            self.is_boundary: bool = is_boundary

    class VictimInfo(entity_info):
        def __init__(
                self,weight:float,
                position: torch.Tensor,
                shape: Sphere,
                name:str = "",
                ):
            super().__init__(name)
            self.weight = weight
            self.position: torch.Tensor = position
            self.shape = shape
        def get_reqired_agents_to_rescue(self) -> int: 
            return math.ceil(self.weight / 10)

    def __init__(
        self,
        height: float,
        wall_thickness: float,
        width: float,
        have_boundary: bool = True,
        cell_size: float = 1.0,
    ):
        self.height = height
        self.width = width
        self.have_boundary = have_boundary
        self.wall_thickness = wall_thickness if have_boundary else 0.0

        self.obstacles: list[Map.ObstacleInfo] = []
        self.agents: list[Map.AgentInfo] = []
        self.occupied_entities: dict[int, tuple[Tensor, float]] = {}
        self._entity_cells: dict[int, tuple[int, int]] = {}

        self.max_cell_size: float = 0.0
        self.cell_size: float = cell_size
        self._configure_cells(cell_size)
        self.victims: list[Map.VictimInfo] = []

        if have_boundary:
            self.obstacles.extend(
                [
                    Map.ObstacleInfo(
                        shape=Box(width=self.height, length=self.wall_thickness),
                        position=torch.tensor([ self.wall_thickness / 2.0  , self.height / 2 ], dtype=torch.float32),
                        angle=0.0,
                        is_boundary=True,
                        name="boundary_left",

                    ),
                    Map.ObstacleInfo(
                        shape=Box(width=self.height, length=self.wall_thickness),
                        position=torch.tensor([ self.width - self.wall_thickness / 2  , self.height / 2 ], dtype=torch.float32),
                        angle=0.0,
                        is_boundary=True,
                        name="boundary_right",
                    ),
                    Map.ObstacleInfo(
                        shape=Box(width=self.wall_thickness, length=self.width),
                        position=torch.tensor([self.width / 2, self.wall_thickness / 2 ], dtype=torch.float32),
                        angle=0.0,
                        is_boundary=True,
                        name="boundary_bottom",
                    ),
                    Map.ObstacleInfo(
                        shape=Box(width=self.wall_thickness, length=self.width),
                        position=torch.tensor([self.width / 2, self.height - self.wall_thickness / 2], dtype=torch.float32),
                        angle=0.0,
                        is_boundary=True,
                        name="boundary_top",
                    ),
                ]
            )

    def _configure_cells(self, cell_size: float) -> None:
        self.cell_size = cell_size
        usable_left =  (self.wall_thickness if self.have_boundary else 0.0)  
        usable_bottom = (self.wall_thickness if self.have_boundary else 0.0)  
        usable_width = self.width - max((2 * self.wall_thickness if self.have_boundary else 0.0), 0.0)
        usable_height = self.height - max((2 * self.wall_thickness if self.have_boundary else 0.0), 0.0)

        self.grid_width = max(1, math.ceil(usable_width / cell_size))
        self.grid_height = max(1, math.ceil(usable_height / cell_size))
        self.cell_width = usable_width / self.grid_width if self.grid_width > 0 else cell_size
        self.cell_height = usable_height / self.grid_height if self.grid_height > 0 else cell_size
        self.max_cell_size = min(self.cell_width, self.cell_height) if self.cell_width > 0 and self.cell_height > 0 else max(usable_width, usable_height)
        self.start_x = usable_left + (self.cell_width / 2.0)
        self.start_y = usable_bottom + (self.cell_height / 2.0)

        self.cells_bitmap = torch.zeros((self.grid_height, self.grid_width), dtype=torch.bool)
        self.occupied_entities = {}
        self._entity_cells = {}
        self._refresh_free_cells()

    def _cell_center(self, row: int, col: int) -> Tensor:
        return torch.tensor([self.start_x + col * self.cell_width, self.start_y + row * self.cell_height], dtype=torch.float32)

    def _cell_index_from_position(self, position: Tensor) -> tuple[int, int]:
        col = int(round( (float(position[0]) - self.start_x) / self.cell_width))
        row = int(round((float(position[1]) - self.start_y) / self.cell_height))
        row = max(0, min(self.cells_bitmap.shape[0] - 1, row))
        col = max(0, min(self.cells_bitmap.shape[1] - 1, col))
        return row, col 

    def _refresh_free_cells(self) -> None:
        free_cells: list[Tensor] = []
        for row in range(self.cells_bitmap.shape[0]):
            for col in range(self.cells_bitmap.shape[1]):
                if not bool(self.cells_bitmap[row, col]):
                    free_cells.append(self._cell_center(row, col))
        self.free_cell = free_cells

    def _entity_clearance(self, entity: object) -> float:
        if isinstance(entity, Map.AgentInfo):
            return float(entity.radius)
        if isinstance(entity, Map.ObstacleInfo):
            if isinstance(entity.shape, Sphere):
                return float(entity.shape.radius)
            if isinstance(entity.shape, Box):
                half_width = float(entity.shape.width) / 2.0
                half_length = float(entity.shape.length) / 2.0
                return math.sqrt(half_width**2 + half_length**2)
        return 0.0

    def _cell_is_legal(self, row: int, col: int, clearance: float) -> bool:
        center = self._cell_center(row, col)

        if self.have_boundary:
            min_x = self.wall_thickness + clearance
            max_x = self.width - self.wall_thickness - clearance
            min_y = self.wall_thickness + clearance
            max_y = self.height - self.wall_thickness - clearance
            if float(center[0]) < min_x or float(center[0]) > max_x:
                return False
            if float(center[1]) < min_y or float(center[1]) > max_y:
                return False

        for occupied_center, occupied_clearance in self.occupied_entities.values():
            if torch.dist(center, occupied_center) < clearance + occupied_clearance:
                return False

        return True

    def _mark_cell_occupied(self, row: int, col: int, entity: object) -> Tensor:
        center = self._cell_center(row, col)
        self.cells_bitmap[row, col] = True
        entity_id = id(entity)
        self.occupied_entities[entity_id] = (center, self._entity_clearance(entity))
        self._entity_cells[entity_id] = (row, col)
        self._refresh_free_cells()
        return center

    def _release_entity(self, entity: object) -> None:
        entity_id = id(entity)
        cell = self._entity_cells.pop(entity_id, None)
        self.occupied_entities.pop(entity_id, None)

        if cell is not None:
            row, col = cell
            self.cells_bitmap[row, col] = False
            self._refresh_free_cells()

    def remove_agent(self, agent: object) -> "Map.AgentInfo":
        if isinstance(agent, int):
            agent_index = agent
            try:
                agent_info = self.agents[agent_index]
            except IndexError as error:
                raise IndexError(f"Agent index {agent_index} is out of range") from error
        elif isinstance(agent, str):
            agent_info = next((candidate for candidate in self.agents if candidate.name == agent), None)
            if agent_info is None:
                raise ValueError(f"No agent named {agent!r} exists in the map")
        else:
            agent_info = next((candidate for candidate in self.agents if candidate is agent), None)
            if agent_info is None:
                raise ValueError("The provided agent is not registered in the map")

        self.agents.remove(agent_info)
        self._release_entity(agent_info)
        return agent_info

    def _nearest_legal_cell(self, target_row: int, target_col: int, clearance: float) -> tuple[int, int]:
        target_center = self._cell_center(target_row, target_col)
        candidates: list[tuple[float, int, int]] = []

        for row in range(self.cells_bitmap.shape[0]):
            for col in range(self.cells_bitmap.shape[1]):
                if bool(self.cells_bitmap[row, col]):
                    continue
                if not self._cell_is_legal(row, col, clearance):
                    continue
                center = self._cell_center(row, col)
                distance = torch.dist(center, target_center).item()
                candidates.append((distance, row, col))

        if not candidates:
            raise RuntimeError("No collision-free cell available for the entity")

        _, row, col = min(candidates, key=lambda item: item[0])
        return row, col

    def read_map_from_file(self, path: str) -> "Map":
        with open(path) as f:
            config = yaml.safe_load(f)

        self.height = config["world"].get("height", config["world"].get("heigh"))
        self.width = config["world"]["width"]
        self.wall_thickness = config["world"]["wall_thickness"]
        self.have_boundary = config["world"].get("have_boundary", True)
        self._configure_cells(self.cell_size)

        for obs in config.get("obstacles", []):
            shape_str = obs["shape"]
            if shape_str == "sphere":
                shape = Sphere(radius=obs["radius"])
            elif shape_str == "box":
                shape = Box(width=obs["width"], length=obs["height"])
            else:
                raise AssertionError(f"shape {obs['shape']} not a valid shape")

            obstacle = Map.ObstacleInfo(
                shape=shape,
                position=torch.tensor(obs["position"], dtype=torch.float32),
                angle=obs.get("rotation", obs.get("angle", 0.0)),
            )
            self.obstacles.append(obstacle)
            row, col = self._cell_index_from_position(obstacle.position)
            self._mark_cell_occupied(row, col, obstacle)

        for ag in config.get("agents", []):
            position = torch.tensor(ag.get("position", [0.0, 0.0]), dtype=torch.float32)
            agent = Map.AgentInfo(radius=ag["radius"], position=position, name=ag.get("name", ""))
            self.agents.append(agent)
            row, col = self._cell_index_from_position(position)
            self._mark_cell_occupied(row, col, agent)

        return self

    def make_random_map(
        self,
        num_agents: int = 2,
        max_agent_radius: float = 0.5,
        min_agent_radius: float = 0.2,
        num_obstacles_circle: int = 1,
        num_obstacles_box: int = 1,
        max_obstacle_length: float = 4,
        min_obstacle_length: float = 0.3,
        max_obstacle_width: float = 4,
        min_obstacle_width: float = 0.3,
        cell_size: float = 1.0,
        num_victims: int = 3,
    ) -> "Map":
        self.obstacles = [obstacle for obstacle in self.obstacles if obstacle.is_boundary]
        self.agents = []
        self._configure_cells(cell_size)
        agent_radius_low, agent_radius_high = sorted((min_agent_radius, max_agent_radius))
        obstacle_width_low, obstacle_width_high = sorted((min_obstacle_width, max_obstacle_width))
        obstacle_length_low, obstacle_length_high = sorted((min_obstacle_length, max_obstacle_length))

        for index in range(num_obstacles_circle):
            radius = random.uniform(obstacle_width_low, obstacle_width_high) / 2.0
            obstacle = Map.ObstacleInfo(
                shape=Sphere(radius=radius),
                position=torch.zeros(2, dtype=torch.float32),
                angle=0.0,
                name=f"obstacle_circle_{index}",
            )
            obstacle.position = self.get_free_cell(obstacle)
            self.obstacles.append(obstacle)

        for index in range(num_obstacles_box):
            box_width = random.uniform(obstacle_width_low, obstacle_width_high)
            box_length = random.uniform(obstacle_length_low, obstacle_length_high)
            obstacle = Map.ObstacleInfo(
                shape=Box(width=box_width, length=box_length),
                position=torch.zeros(2, dtype=torch.float32),
                angle=random.uniform(-math.pi, math.pi),
                name=f"obstacle_box_{index}",
            )
            obstacle.position = self.get_free_cell(obstacle)
            self.obstacles.append(obstacle)

        for index in range(num_agents):
            radius = random.uniform(agent_radius_low, agent_radius_high)
            agent = Map.AgentInfo(
                radius=radius,
                position=torch.zeros(2, dtype=torch.float32),
                name=f"agent_{index}",
                color= Color.BLUE,
            )
            agent.position = self.get_free_cell(agent)
            self.agents.append(agent)
        for index in range(num_victims):
            weight = random.uniform(5, 30)
            victim = Map.VictimInfo(
                name = f"victim_{index}",
                weight=weight,
                position=torch.zeros(2, dtype=torch.float32),
                shape = Sphere(radius=min(self.width, self.height) / 100),
            )
            victim.position = self.get_free_cell(victim)
            self.victims.append(victim)
            

        return self
    def reset_agents(self):
        for agent in self.agents:
            self._release_entity(agent)
            agent.position = self.get_free_cell(agent)

    def get_free_cell(self, entity: object = None) -> Tensor:
        if self.cells_bitmap.numel() == 0:
            raise RuntimeError("cells_bitmap is empty")

        clearance = self._entity_clearance(entity)
        row = random.randrange(self.cells_bitmap.shape[0])
        col = random.randrange(self.cells_bitmap.shape[1])

        if bool(self.cells_bitmap[row, col]) or not self._cell_is_legal(row, col, clearance):
            row, col = self._nearest_legal_cell(row, col, clearance)

        return self._mark_cell_occupied(row, col, entity if entity is not None else Map.entity_info())

    def to_dict(self) -> dict:
        obstacles = []
        for obstacle in self.obstacles:
            if obstacle.is_boundary:
                continue
            shape = obstacle.shape
            obstacle_data = {
                "name": obstacle.name,
                "shape": "box" if isinstance(shape, Box) else "sphere",
                "position": obstacle.position.tolist(),
                "rotation": float(obstacle.angle),
            }
            if isinstance(shape, Box):
                obstacle_data["shape"] = "box"
                obstacle_data["width"] = float(shape.width)
                obstacle_data["length"] = float(shape.length)
            elif isinstance(shape, Sphere):
                obstacle_data["shape"] = "sphere"
                obstacle_data["radius"] = float(shape.radius)
            obstacles.append(obstacle_data)

        agents = []
        for agent in self.agents:
            agents.append(
                {
                    "name": agent.name,
                    "radius": float(agent.radius),
                    "position": agent.position.tolist(),
                }
            )

        return {
            "world": {
                "height": float(self.height),
                "width": float(self.width),
                "wall_thickness": float(self.wall_thickness),
                "have_boundary": self.have_boundary,
            },
            "agents": agents,
            "obstacles": obstacles,
        }

    def save_to_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as file:
            yaml.safe_dump(self.to_dict(), file, sort_keys=False)

    def Plot(self, ax=None, show: bool = True, with_grid: bool = True, title: str | None = None):
        from matplotlib import patches, pyplot as plt, transforms

        created_figure = False
        if ax is None:
            figure, ax = plt.subplots(figsize=(8, 8))
            created_figure = True
        else:
            figure = ax.figure

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title or "Map")

        if with_grid:
            ax.set_xticks([i * self.cell_size for i in range(int(math.ceil(self.width / self.cell_size)) + 1)])
            ax.set_yticks([i * self.cell_size for i in range(int(math.ceil(self.height / self.cell_size)) + 1)])
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

        if self.have_boundary and self.wall_thickness > 0:
            boundary = patches.Rectangle(
                (0, 0),
                self.width,
                self.height,
                fill=False,
                linewidth=2.0,
                edgecolor="black",
            )
            ax.add_patch(boundary)

        def add_box(center: Tensor, width: float, length: float, angle: float, color: str, alpha: float, label: str):
            lower_left = (float(center[0]) - width / 2.0, float(center[1]) - length / 2.0)
            box = patches.Rectangle(
                lower_left,
                width,
                length,
                linewidth=1.5,
                edgecolor=color,
                facecolor=color,
                alpha=alpha,
                label=label,
            )
            box.set_transform(transforms.Affine2D().rotate_around(float(center[0]), float(center[1]), angle) + ax.transData)
            ax.add_patch(box)

        def add_circle(center: Tensor, radius: float, color: str, alpha: float, label: str):
            circle = patches.Circle(
                (float(center[0]), float(center[1])),
                radius=radius,
                linewidth=1.5,
                edgecolor=color,
                facecolor=color,
                alpha=alpha,
                label=label,
            )
            ax.add_patch(circle)

        for obstacle in self.obstacles:
            if obstacle.is_boundary:
                continue
            if isinstance(obstacle.shape, Sphere):
                add_circle(obstacle.position, float(obstacle.shape.radius), color="#4A4A4A", alpha=0.45, label="obstacle")
            elif isinstance(obstacle.shape, Box):
                add_box(
                    obstacle.position,
                    float(obstacle.shape.width),
                    float(obstacle.shape.length),
                    float(obstacle.angle),
                    color="#4A4A4A",
                    alpha=0.45,
                    label="obstacle",
                )

        for agent in self.agents:
            add_circle(agent.position, float(agent.radius), color="tab:blue", alpha=0.8, label="agent")

        handles, labels = ax.get_legend_handles_labels()
        legend_items = {}
        for handle, label in zip(handles, labels):
            if label and label not in legend_items:
                legend_items[label] = handle
        if legend_items:
            ax.legend(legend_items.values(), legend_items.keys(), loc="upper right")

        if show:
            if created_figure:
                plt.show()
            else:
                figure.canvas.draw_idle()

        return figure, ax

    def plot(self, *args, **kwargs):
        return self.Plot(*args, **kwargs)


def make_map_from_file(path: str) -> Map:
    with open(path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    world = config["world"]
    map_ = Map(
        height=world.get("height", world.get("heigh")),
        wall_thickness=world.get("wall_thickness", 0.0),
        width=world["width"],
        have_boundary=world.get("have_boundary", True),
    )
    return map_.read_map_from_file(path)


def make_random_map(
    height: float = 10.0,
    width: float = 10.0,
    wall_thickness: float = 0.1,
    have_boundary: bool = True,
    num_agents: int = 2,
    max_agent_radius: float = 0.5,
    min_agent_radius: float = 0.2,
    num_obstacles_circle: int = 1,
    num_obstacles_box: int = 1,
    max_obstacle_length: float = 4.0,
    min_obstacle_length: float = 0.3,
    max_obstacle_width: float = 4.0,
    min_obstacle_width: float = 0.3,
) -> Map:
    map_ = Map(height=height, wall_thickness=wall_thickness, width=width, have_boundary=have_boundary)
    return map_.make_random_map(
        num_agents=num_agents,
        max_agent_radius=max_agent_radius,
        min_agent_radius=min_agent_radius,
        num_obstacles_circle=num_obstacles_circle,
        num_obstacles_box=num_obstacles_box,
        max_obstacle_length=max_obstacle_length,
        min_obstacle_length=min_obstacle_length,
        max_obstacle_width=max_obstacle_width,
        min_obstacle_width=min_obstacle_width,
    )