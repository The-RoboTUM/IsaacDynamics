# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a simple Cartpole robot."""


import math
from scipy.spatial.transform import Rotation as R

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
# from isaaclab.utils.math import quat_from_euler_xyz

##
# Configuration
##

PENDULUM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="source/isaaclab_assets/isaaclab_assets/custom/pendulum_grav_simple/pendulum_grav_simple.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=R.from_euler("xyz", [0.0, math.pi / 2, 0.0]).as_quat().tolist(),
        joint_pos={"base_to_pendulum": 0.0},
    ),
    actuators={
        "base_actuator": ImplicitActuatorCfg(
            joint_names_expr=["base_to_pendulum"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
"""Configuration for a simple gravitational pendulum."""
