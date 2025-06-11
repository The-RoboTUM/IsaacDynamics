# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Simple pendulum balancing environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Pendulum-Simple-Direct-v0",
    entry_point=f"{__name__}.pendulum_grav_simple:PendulumSimpleDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pendulum_grav_simple:PendulumSimpleDirectEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "pid_cfg_entry_point": f"{agents.__name__}:skrl_pid_cfg.yaml",
        "random_cfg_entry_point": f"{agents.__name__}:skrl_random_cfg.yaml",
    },
)

# gym.register(
#     id="Isaac-Cartpole-RGB-Camera-Direct-v0",
#     entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.cartpole_camera_env:CartpoleRGBCameraEnvCfg",
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
#     },
# )
#
# gym.register(
#     id="Isaac-Cartpole-Depth-Camera-Direct-v0",
#     entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.cartpole_camera_env:CartpoleDepthCameraEnvCfg",
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
#     },
# )
