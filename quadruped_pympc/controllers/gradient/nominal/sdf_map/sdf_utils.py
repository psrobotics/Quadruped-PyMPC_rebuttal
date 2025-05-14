import casadi as cs
import numpy as np
import pickle

import os
from tqdm import tqdm

from map_utils import Environment
from map_utils import CustomGroundEnvironment
from map_utils import Obstacle
from map_utils import CircularObstacle
from map_utils import BoxObstacle

import importlib
import sys

def barrier_and_grad(env, xy, eps=1e-3):
    x, y = float(xy[0]), float(xy[1])
    # ∂h/∂x
    h_p = env.distances(np.array([[x + eps, y]]))[0]
    h_m = env.distances(np.array([[x - eps, y]]))[0]
    dh_dx = (h_p - h_m) / (2 * eps)
    # ∂h/∂y
    h_p = env.distances(np.array([[x, y + eps]]))[0]
    h_m = env.distances(np.array([[x, y - eps]]))[0]
    dh_dy = (h_p - h_m) / (2 * eps)
    # h at center
    h0 = env.distances(np.array([[x, y]]))[0]
    return h0, np.array([dh_dx, dh_dy])

class SDFCallback:
    """
    Wrap an environment pickle into a CasADi SXCallback.
    """
    def __init__(self, pickle_path: str, name: str = "sdf_cb"):
        # 1) load your env
        # 1) Redirect the old "map_utils" name to your real module path:
        real_mod = importlib.import_module(
            "quadruped_pympc.controllers.gradient.nominal.sdf_map.map_utils"
        )
        sys.modules["map_utils"] = real_mod

        # 2) Now un-pickle—any references to `map_utils` will resolve
        with open(pickle_path, "rb") as f:
            self.env = pickle.load(f)


        # 2) build the SXCallback
        opts = {
            "n_in":       lambda: 1,
            "n_out":      lambda: 1,
            "spars_in":   lambda i: cs.Sparsity.dense(3,1),
            "spars_out":  lambda i: cs.Sparsity.dense(1,1),
            "has_jac":    lambda: True,
            "spars_jac":  lambda i,j: cs.Sparsity.dense(1,3),
            "eval":       lambda x: [barrier_and_grad(self.env, x.full().flatten())[0]],
            "jac":        lambda x: [ np.append(barrier_and_grad(self.env, x.full().flatten())[1], 0.0 ).reshape(1,3) ],
        }
        self.builder = cs.SXCallback(name, opts)
        self.builder.init()

        # 3) expose as a Function
        pt    = cs.SX.sym("pt", 3)
        h_sym = self.builder(pt)
        self.sdf = cs.Function(name + "_fn", [pt], [h_sym])

    def casadi_function(self):
        return self.sdf


# -------------------------
# Usage in your controller:
# -------------------------
# 1) create the callback and CasADi function once at init
#sdf_cb = SDFCallback("data/environments/paper_friction_change/0/environment.pickle")
#sdf_ext = sdf_cb.casadi_callback()  # this is a CasADi Function

# 2) in your Acados_NMPC_Nominal._get_casadi_sdf_expression:
#def _get_casadi_sdf_expression(self, point_sx):
#    # point_sx: SX(3×1)
#    return sdf_ext(point_sx)  # returns an SX(1×1) signed-distance

# Now self.expr_h_sdf = cs.vertcat(..., self._get_casadi_sdf_expression(com_pos_world), ...)
# will correctly include both value and gradient of your pickled environment’s SDF.
