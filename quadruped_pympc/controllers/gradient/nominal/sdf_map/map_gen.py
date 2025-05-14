import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from map_utils import Environment
from map_utils import CustomGroundEnvironment
from map_utils import Obstacle
from map_utils import CircularObstacle
from map_utils import BoxObstacle

# save options
save_dir = 'data/environments/paper_friction_change'

# environment parameters
payload = 0
friction = 1

# obstacle configurations
obstacle_height = 1
obstacle_configurations = [
    [
        ['box', [5.5, -2], np.pi/2, 6, 0.1, obstacle_height],
        ['box', [8.5, 2], np.pi/2, 6, 0.1, obstacle_height],
        # ['circular', [3, 0], 1, obstacle_height],
        # ['box', [5, 3], -np.pi/4, 5.5, 0.1, obstacle_height],
        # ['box', [5, -3], np.pi/4, 5.5, 0.1, obstacle_height],
    ],
    [
        ['box', [5.5, -2], np.pi/2, 6, 0.1, obstacle_height],
        ['box', [8.5, 2], np.pi/2, 6, 0.1, obstacle_height],
        # ['circular', [3, 0], 1, obstacle_height],
        # ['box', [5, 3], -np.pi/4, 5.5, 0.1, obstacle_height],
        # ['box', [5, -3], np.pi/4, 5.5, 0.1, obstacle_height],
    ],
]

# ground parameter dicts list
ground_parameter_dicts_list = [
    [
        {
            'x': 1.5,
            'y': 0,
            'length': 7,
            'width': 10,
            'friction': 1.0,
            'color': (0.5, 0.5, 0.5),
        },
        {
            'x': 7,
            'y': 0,
            'length': 4,
            'width': 10,
            'friction': 0.5,
            'color': (111/255, 123/255, 148/255),
        },
        {
            'x': 10.5,
            'y': 0,
            'length': 3,
            'width': 10,
            'friction': 1.0,
            'color': (0.5, 0.5, 0.5),
        },
    ],
    [
        {
            'x': 1.5,
            'y': 0,
            'length': 7,
            'width': 10,
            'friction': 1.0,
            'color': (0.5, 0.5, 0.5),
        },
        {
            'x': 7,
            'y': 0,
            'length': 4,
            'width': 10,
            'friction': 1.0,
            'color': (0.5, 0.5, 0.5),
        },
        {
            'x': 10.5,
            'y': 0,
            'length': 3,
            'width': 10,
            'friction': 1.0,
            'color': (0.5, 0.5, 0.5),
        },
    ],
]

# visualization options
lower_left_vis_point, upper_right_vis_point = [-2.5, -5.5], [12.5, 5.5]
resolution = 1000
xs = np.linspace(lower_left_vis_point[0], upper_right_vis_point[0], resolution)
ys = np.linspace(lower_left_vis_point[1], upper_right_vis_point[1], resolution)
grid_xs, grid_ys = np.meshgrid(xs, ys, indexing='ij')
grid_xys = np.stack((grid_xs, grid_ys), axis=-1)

# create save directory
start_i = 0
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
else:
    # continue data generation from earlier run
    while os.path.exists(os.path.join(save_dir, str(start_i), 'environment.pickle')):
        start_i += 1

# generate environments
for i in tqdm(range(start_i, len(obstacle_configurations))):
    # create environment directory
    if not os.path.exists(os.path.join(save_dir, str(i))):
        os.makedirs(os.path.join(save_dir, str(i)))
    elif os.path.exists(os.path.join(save_dir, str(i), 'environment.pickle')):
        continue
    # generate obstacles
    obstacles = []
    for obstacle in obstacle_configurations[i]:
        if obstacle[0] == 'circular':
            obstacles.append(CircularObstacle(center=obstacle[1], radius=obstacle[2], height=obstacle[3]))
        elif obstacle[0] == 'box':
            obstacles.append(BoxObstacle(center=obstacle[1], angle=obstacle[2], length=obstacle[3], width=obstacle[4], height=obstacle[5]))
        else:
            raise NotImplementedError
    # add walls
    obstacles.append(BoxObstacle(center=[-2, 0], angle=np.pi/2, length=10, width=0.1, height=obstacle_height))
    obstacles.append(BoxObstacle(center=[12, 0], angle=np.pi/2, length=10, width=0.1, height=obstacle_height))
    obstacles.append(BoxObstacle(center=[5, -5], angle=0, length=14, width=0.1, height=obstacle_height))
    obstacles.append(BoxObstacle(center=[5, 5], angle=0, length=14, width=0.1, height=obstacle_height))
    # save environment
    env = CustomGroundEnvironment(
        obstacles=obstacles,
        payload=payload,
        friction=friction,
        ground_parameter_dicts=ground_parameter_dicts_list[i],
    )
    with open(os.path.join(save_dir, str(i), 'environment.pickle'), 'wb') as f:
        pickle.dump(env, f)
    # save visualization of environment
    grid_distances = env.distances(grid_xys.reshape(resolution*resolution, 2)).reshape(resolution, resolution)
    legend_limit = np.max(np.abs(grid_distances))
    plt.figure()
    plt.pcolormesh(xs, ys, grid_distances.T, cmap='RdBu', vmin=-legend_limit, vmax=legend_limit)
    plt.colorbar()
    plt.contour(xs, ys, grid_distances.T, levels=0, colors='k')
    for gpd in ground_parameter_dicts_list[i]:
        plt.fill_between(
            [gpd['x']-gpd['length']/2, gpd['x']+gpd['length']/2],
            [gpd['y']-gpd['width']/2, gpd['y']-gpd['width']/2],
            [gpd['y']+gpd['width']/2, gpd['y']+gpd['width']/2],
            color=gpd['color'], alpha=0.5,
        )
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(f'Signed Distance to Obstacles\nPayload: {env.payload:2.1f}, Friction: {env.friction:2.1f}')
    plt.gca().set_aspect('equal')
    plt.savefig(os.path.join(save_dir, str(i), 'visualization.jpg'), bbox_inches='tight', dpi=800)
    plt.show()
    plt.close()