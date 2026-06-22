import sys
from pathlib import Path
import torch

# ensure imports from workspace
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.map import make_random_map
from utils.scenario import Scenario


def main():
    device = torch.device('cpu')
    map_ = make_random_map(width=10.0, height=10.0, num_agents=3, num_obstacles_circle=1, num_obstacles_box=1, max_obstacle_length=2.0)
    scen = Scenario(map_=map_)
    world = scen.make_world(batch_dim=1, device=device)

    offset = torch.tensor([map_.width / 2.0, map_.height / 2.0], device=device)

    print('\nAgents:')
    for i, agent_info in enumerate(map_.agents):
        p = agent_info.position
        if (p >= 0).all() and (p <= torch.tensor([map_.width, map_.height], device=device)).all():
            pos_world = p - offset
        else:
            pos_world = p
        try:
            actual = world.agents[i].state.pos
        except Exception as e:
            actual = f'<error accessing world.agents[{i}]: {e}>'
        print(f'Agent {i}: map={p.tolist()} pos_world={pos_world.tolist()} world_state={actual}')

    print('\nObstacles / Landmarks:')
    for j, obs in enumerate(map_.obstacles):
        q = obs.position
        if (q >= 0).all() and (q <= torch.tensor([map_.width, map_.height], device=device)).all():
            pos_world = q - offset
        else:
            pos_world = q
        # try to get corresponding landmark in world
        try:
            lm = world.landmarks[j]
            actual = lm.state.pos
        except Exception as e:
            actual = f'<error accessing world.landmarks[{j}]: {e}>'
        print(f'Obs {j}: map={q.tolist()} pos_world={pos_world.tolist()} world_state={actual}')


if __name__ == '__main__':
    main()
