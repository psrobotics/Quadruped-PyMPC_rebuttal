import casadi as cs
import numpy as np
import abc

import pickle

import os
from tqdm import tqdm

from .map_utils import Environment
from .map_utils import CustomGroundEnvironment
from .map_utils import Obstacle
from .map_utils import CircularObstacle
from .map_utils import BoxObstacle

import importlib
import sys
import matplotlib.pyplot as plt

# --- User's Provided Environment Classes ---
class Obstacle(metaclass=abc.ABCMeta):
    """An abstract base class for representing an environment obstacle."""

    @abc.abstractmethod
    def distances(
            self,
            positions: "np.ndarray[np.float_]",
    ) -> "np.ndarray[np.float_]":
        """Returns signed distances from the positions to the obstacle.

        Args:
            positions: A numpy array with shape [batch_size, 2].

        Returns:
            distances: A numpy array with shape [batch_size].
        """
        pass # Ensure abstract methods are implemented or pass

class CircularObstacle(Obstacle):
    """A circular obstacle in an environment."""

    def __init__(
            self,
            center: "list[float]",
            radius: float,
            height: float, # Not used in 2D distance calculation
            name: str = "circular_obstacle" # Added for clarity
    ):
        self.center = np.array(center) # Expected [x,y]
        self.radius = radius
        self.height = height
        self.name = name
        if len(self.center) != 2:
            raise ValueError("CircularObstacle center must be 2D [x,y].")

    def distances(
            self,
            positions: "np.ndarray[np.float_]",
    ) -> "np.ndarray[np.float_]":
        return np.linalg.norm(
            positions - self.center,
            axis=-1
        ) - self.radius

class BoxObstacle(Obstacle):
    """A box obstacle in an environment."""

    def __init__(
            self,
            center: "list[float]", # [x,y]
            angle: float, # radians
            length: float, # along box's local x-axis
            width: float,  # along box's local y-axis
            height: float, # Not used in 2D distance calculation
            name: str = "box_obstacle" # Added for clarity
    ):
        self.center = np.array(center)
        self.angle = angle
        self.length = length
        self.width = width
        self.height = height
        self.name = name
        if len(self.center) != 2:
            raise ValueError("BoxObstacle center must be 2D [x,y].")
        self.half_length = length / 2.0
        self.half_width = width / 2.0


    def distances(
            self,
            positions: "np.ndarray[np.float_]",
    ) -> "np.ndarray[np.float_]":
        # compute positions in ego frame
        positions_local = positions.copy()
        cth, sth = np.cos(self.angle), np.sin(self.angle)
        # Rotation matrix for world to box frame
        rot_world_to_box = np.array([
            [cth, sth],
            [-sth, cth],
        ])
        # Translate points so box center is origin, then rotate
        translated_positions = positions_local - self.center
        # Correct batch matrix multiplication
        # positions_in_box_frame = np.einsum('ij,bj->bi', rot_world_to_box, translated_positions)
        positions_in_box_frame = (rot_world_to_box @ translated_positions.T).T


        # Signed distances to box boundaries in its local frame
        # q = abs(local_pos) - half_sizes
        q_x = np.abs(positions_in_box_frame[:, 0]) - self.half_length
        q_y = np.abs(positions_in_box_frame[:, 1]) - self.half_width

        # SDF formula for an axis-aligned box (Inigo Quilez)
        # For a single point q = [qx, qy]:
        #   length(max(q, vec2(0.0))) + min(max(qx, qy), 0.0)
        
        # Vectorized:
        q_stacked = np.stack((q_x, q_y), axis=-1) # Shape: (batch_size, 2)
        
        # Term 1: length(max(q, 0.0))
        # This is for points outside the box. It's the distance to the closest corner/edge.
        term1 = np.linalg.norm(np.maximum(q_stacked, 0.0), axis=-1)
        
        # Term 2: min(max(q.x, q.y), 0.0)
        # This is for points inside the box. It's the negative distance to the closest edge.
        # max(q.x, q.y) finds which axis is "less deep" inside.
        max_of_q_components = np.maximum(q_x, q_y) # Shape: (batch_size,)
        term2 = np.minimum(max_of_q_components, 0.0) # Shape: (batch_size,)
        
        return term1 + term2


class Environment:
    """A base class for representing an environment with obstacles."""
    def __init__(
        self,
        obstacles: "list[Obstacle]",
        payload: float = 0,
        friction: float = 1,
        name: str = "default_env" # Added for clarity
    ):
        self.obstacles = obstacles if obstacles is not None else []
        self.payload = payload
        self.friction = friction
        self.name = name

    def distances(
            self,
            positions, # Expected shape [batch_size, 2]
    ) -> "np.ndarray[np.float_]":
        if not self.obstacles:
            return np.full(positions.shape[0], float('inf')) # No obstacles, infinite distance

        all_obs_distances = []
        for obstacle in self.obstacles:
            all_obs_distances.append(obstacle.distances(positions))
        
        # Stack distances for each obstacle: shape (num_obstacles, batch_size)
        stacked_distances = np.stack(all_obs_distances, axis=0)
        
        # Minimum distance across all obstacles for each position
        min_distances = np.min(stacked_distances, axis=0)
        return min_distances

class CustomGroundEnvironment(Environment):
    """
    Environment with potentially different ground patches.
    For SDF purposes, ground patches with different friction are not obstacles
    unless explicitly defined as such (e.g., a keep-out zone as a BoxObstacle).
    This class will behave like 'Environment' for SDF calculations based on its 'obstacles' list.
    """
    def __init__(self, ground_parameter_dicts, obstacles_list, *args, **kwargs):
        # Pass obstacles_list to the parent Environment constructor
        super().__init__(obstacles=obstacles_list, *args, **kwargs)
        self.ground_parameter_dicts = ground_parameter_dicts
        # ground_parameter_dicts: [{'x', 'y', 'length', 'width', 'friction', 'color'}]
        # If these ground patches should also act as obstacles for SDF,
        # they should be converted into BoxObstacle objects and added to obstacles_list.
        # For example:
        # for patch_params in ground_parameter_dicts:
        #     if patch_params.get("is_obstacle", False): # Add a flag if it's an obstacle
        #         obs = BoxObstacle(center=[patch_params['x'], patch_params['y']],
        #                             angle=patch_params.get('angle', 0.0), # Assume angle if exists
        #                             length=patch_params['length'],
        #                             width=patch_params['width'],
        #                             height=1.0) # Arbitrary height for 2D SDF
        #         self.obstacles.append(obs)

# --- CasADi Symbolic SDF Environment ---
class CasadiSDFEnvironment:
    def __init__(self, python_environment: Environment, name: str = "symbolic_sdf_env"):
        """
        Initializes the symbolic SDF environment from a Python Environment instance.

        Args:
            python_environment: An instance of your Environment or CustomGroundEnvironment.
            name: A base name for the created CasADi functions.
        """
        self.name = name
        self.parsed_obstacles = []
        self._parse_python_environment(python_environment)
        self.casadi_sdf_function = self._build_casadi_sdf_function()

    def _parse_python_environment(self, python_env: Environment):
        """Extracts parameters from Python obstacle objects."""
        if not hasattr(python_env, 'obstacles') or not isinstance(python_env.obstacles, list):
            print(f"Warning: CasadiSDFEnvironment - Python environment '{python_env.name}' has no 'obstacles' list. SDF will be empty.")
            return

        for obs_py in python_env.obstacles:
            if isinstance(obs_py, CircularObstacle):
                self.parsed_obstacles.append({
                    "type": "circle",
                    "center": np.array(obs_py.center), # Should be [x,y]
                    "radius": float(obs_py.radius),
                    "name": getattr(obs_py, 'name', 'unnamed_circle')
                })
            elif isinstance(obs_py, BoxObstacle):
                self.parsed_obstacles.append({
                    "type": "box",
                    "center": np.array(obs_py.center), # Should be [x,y]
                    "angle_rad": float(obs_py.angle),
                    "half_length": float(obs_py.length) / 2.0,
                    "half_width": float(obs_py.width) / 2.0,
                    "name": getattr(obs_py, 'name', 'unnamed_box')
                })
            else:
                print(f"Warning: CasadiSDFEnvironment - Unknown obstacle type {type(obs_py)}. Skipping.")
        print(f"CasadiSDFEnvironment: Parsed {len(self.parsed_obstacles)} obstacles.")

    def _symbolic_sdf_circular_2d(self, pt_sx_2d: cs.SX, center_np_2d: np.ndarray, radius_val: float) -> cs.SX:
        """Symbolic SDF for a 2D circle."""
        center_sx = cs.DM(center_np_2d) # Convert NumPy array to CasADi DM (constant matrix)
        return cs.norm_2(pt_sx_2d - center_sx) - radius_val

    def _symbolic_sdf_box_2d_oriented(self, pt_sx_2d: cs.SX, center_np_2d: np.ndarray,
                                      angle_rad_val: float, half_length_val: float,
                                      half_width_val: float) -> cs.SX:
        """Symbolic SDF for an oriented 2D box."""
        center_sx = cs.DM(center_np_2d)
        half_sizes_sx = cs.DM([half_length_val, half_width_val])

        # Rotation matrix for world to box frame
        c_angle = cs.cos(angle_rad_val)
        s_angle = cs.sin(angle_rad_val)
        # R_world_to_box = [[c, s], [-s, c]]
        # (pt_world - center_world) rotated by R_world_to_box
        # pt_in_box_x = c * (pt_sx_2d[0]-center_sx[0]) + s * (pt_sx_2d[1]-center_sx[1])
        # pt_in_box_y = -s * (pt_sx_2d[0]-center_sx[0]) + c * (pt_sx_2d[1]-center_sx[1])
        # pt_in_box_frame = cs.vertcat(pt_in_box_x, pt_in_box_y)
        
        # Simpler: create rotation matrix and multiply
        rot_world_to_box_sx = cs.SX(2,2)
        rot_world_to_box_sx[0,0] = c_angle
        rot_world_to_box_sx[0,1] = s_angle
        rot_world_to_box_sx[1,0] = -s_angle
        rot_world_to_box_sx[1,1] = c_angle
        
        pt_relative_to_center = pt_sx_2d - center_sx
        pt_in_box_frame = rot_world_to_box_sx @ pt_relative_to_center

        # SDF for axis-aligned box in its own frame
        # q = abs(local_pos) - half_sizes
        q = cs.fabs(pt_in_box_frame) - half_sizes_sx
        
        # SDF_box(q) = ||max(q,0)||_2 + min(max(q_x, q_y), 0)
        term1 = cs.norm_2(cs.fmax(q, 0.0)) # For points outside
        term2 = cs.fmin(cs.mmax(q), 0.0)    # For points inside (mmax gets max element of q)
        
        return term1 + term2

    def _build_casadi_sdf_function(self) -> cs.Function:
        """Builds the combined CasADi SDF function for all parsed obstacles."""
        pt_sx_2d = cs.SX.sym("pt_xy_world", 2) # Symbolic input point [x, y]

        if not self.parsed_obstacles:
            print("CasadiSDFEnvironment: No obstacles parsed, creating a default 'always safe' SDF function.")
            sdf_final_expr = cs.SX(1e6) # Large positive value (very safe)
        else:
            obstacle_sdfs_sx = []
            for obs_params in self.parsed_obstacles:
                if obs_params["type"] == "circle":
                    sdf_obs = self._symbolic_sdf_circular_2d(pt_sx_2d,
                                                             obs_params["center"],
                                                             obs_params["radius"])
                    obstacle_sdfs_sx.append(sdf_obs)
                elif obs_params["type"] == "box":
                    sdf_obs = self._symbolic_sdf_box_2d_oriented(pt_sx_2d,
                                                                 obs_params["center"],
                                                                 obs_params["angle_rad"],
                                                                 obs_params["half_length"],
                                                                 obs_params["half_width"])
                    obstacle_sdfs_sx.append(sdf_obs)
            
            if not obstacle_sdfs_sx: # Should not happen if parsed_obstacles is not empty
                sdf_final_expr = cs.SX(1e6)
            else:
                # Combine SDFs: minimum distance to any obstacle
                current_min_sdf = obstacle_sdfs_sx[0]
                for i in range(1, len(obstacle_sdfs_sx)):
                    current_min_sdf = cs.fmin(current_min_sdf, obstacle_sdfs_sx[i])
                sdf_final_expr = current_min_sdf
        
        sdf_casadi_func = cs.Function(
            self.name + "_fn",       # CasADi function name
            [pt_sx_2d],              # Inputs
            [sdf_final_expr],        # Outputs
            ['point_xy_world'],      # Input names (optional)
            ['sdf_value']            # Output names (optional)
        )
        print(f"CasADi symbolic SDF function '{self.name}_fn' built successfully.")
        return sdf_casadi_func

    def get_casadi_sdf(self) -> cs.Function:
        """Returns the compiled CasADi SDF function."""
        if self.casadi_sdf_function is None:
             # This case should ideally be handled by _build_casadi_function creating a fallback
             print("Error: CasADi SDF function was not built. Returning a dummy function.")
             pt_sx = cs.SX.sym("pt_dummy", 2)
             return cs.Function("dummy_sdf_fn", [pt_sx], [cs.SX(1e6)])
        return self.casadi_sdf_function

# --- Example Usage (Illustrative) ---
if __name__ == '__main__':
    # 1. Create your Python Environment instance
    obstacle_height = 1

    # add walls
    wall_1 = BoxObstacle(center=[-2, 0], angle=np.pi/2, length=10, width=0.1, height=obstacle_height, name="W1")
    wall_2 = BoxObstacle(center=[12, 0], angle=np.pi/2, length=10, width=0.1, height=obstacle_height, name="W2")
    wall_3 = BoxObstacle(center=[5, -5], angle=0, length=14, width=0.1, height=obstacle_height, name="W3")
    wall_4 = BoxObstacle(center=[5, 5], angle=0, length=14, width=0.1, height=obstacle_height, name="W4")
    # add box obst
    obst_1 = BoxObstacle(center=[5.5, -2], angle=np.pi/2, length=6, width=0.1, height=obstacle_height, name = "OBST1")
    obst_2 = BoxObstacle(center=[8.5, 5], angle=np.pi/2, length=6, width=0.1, height=obstacle_height, name = "OBST2")

    # Create a list of obstacles for the environment
    my_obstacles = [wall_1, wall_2, wall_3, wall_4, obst_1, obst_2]

    # Create the Python environment instance
    python_env = Environment(obstacles=my_obstacles, name="TestEnv")
    
    # If using CustomGroundEnvironment:
    # ground_patches = [{'x':0, 'y':0, 'length':5, 'width':5, 'friction':0.8, 'color':'gray'}]
    # custom_python_env = CustomGroundEnvironment(ground_parameter_dicts=ground_patches, obstacles_list=my_obstacles, name="CustomTestEnv")


    # 2. Create the CasADi Symbolic SDF Environment from it
    # casadi_sdf_env = CasadiSDFFromPickle(pickle_path) # If loading from pickle
    # For this example, we initialize directly from the Python environment instance
    symbolic_sdf_wrapper = CasadiSDFEnvironment(python_environment=python_env, name="my_symbolic_env")

    # 3. Get the CasADi Function
    casadi_sdf = symbolic_sdf_wrapper.get_casadi_sdf()

    # 4. Test the CasADi Function (Optional)
    test_point_np = np.array([1.1, 0.9]) # Near circle1
    sdf_value_casadi = casadi_sdf(test_point_np)
    print(f"SDF at {test_point_np} (CasADi): {sdf_value_casadi}")

    # Compare with original Python environment (for the same point)
    sdf_value_python = python_env.distances(test_point_np.reshape(1,2))
    print(f"SDF at {test_point_np} (Python): {sdf_value_python[0]}")
    
    test_point_np_box = np.array([3.0, 0.5]) # Center of box1
    sdf_value_casadi_box = casadi_sdf(test_point_np_box)
    print(f"SDF at {test_point_np_box} (CasADi): {sdf_value_casadi_box}")
    sdf_value_python_box = python_env.distances(test_point_np_box.reshape(1,2))
    print(f"SDF at {test_point_np_box} (Python): {sdf_value_python_box[0]}")

    # Example of using it in your NMPC's _get_casadi_sdf_expression
    # In Acados_NMPC_Nominal:
    # def __init__(self, ..., python_env_instance):
    #     # ...
    #     self.symbolic_sdf_wrapper = CasadiSDFEnvironment(python_env_instance)
    #     self.casadi_sdf_for_nmpc = self.symbolic_sdf_wrapper.get_casadi_sdf()
    #     # ...
    #
    # def _get_casadi_sdf_expression(self, point_sx_2d): # point_sx_2d is SX sym for [x,y]
    #     if self.casadi_sdf_for_nmpc:
    #         return self.casadi_sdf_for_nmpc(point_sx_2d)
    #     return cs.SX(1e6) # Fallback

    # visualization options
    lower_left_vis_point, upper_right_vis_point = [-2.5, -5.5], [12.5, 5.5]
    resolution = 1000
    xs = np.linspace(lower_left_vis_point[0], upper_right_vis_point[0], resolution)
    ys = np.linspace(lower_left_vis_point[1], upper_right_vis_point[1], resolution)
    grid_xs, grid_ys = np.meshgrid(xs, ys, indexing='ij')
    grid_xys = np.stack((grid_xs, grid_ys), axis=-1)
    # save visualization of environment
    grid_distances = python_env.distances(grid_xys.reshape(resolution*resolution, 2)).reshape(resolution, resolution)
    legend_limit = np.max(np.abs(grid_distances))
    plt.figure()
    plt.pcolormesh(xs, ys, grid_distances.T, cmap='RdBu', vmin=-legend_limit, vmax=legend_limit)
    plt.colorbar()
    plt.contour(xs, ys, grid_distances.T, levels=0, colors='k')

    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(f'Signed Distance to Obstacles\nPayload: {python_env.payload:2.1f}, Friction: {python_env.friction:2.1f}')
    plt.gca().set_aspect('equal')
    #plt.savefig(os.path.join(save_dir, str(i), 'visualization.jpg'), bbox_inches='tight', dpi=800)
    plt.show()
    plt.close()