# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to run the RL environment for the pendulum balancing task."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Experiment on running the Pendulum RL environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from isaaclab_dynamics import controllers

from isaaclab.envs import ManagerBasedRLEnv

# from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleEnvCfg
from isaaclab_tasks.manager_based.custom.pendulum_grav_simple.pendulum_grav_simple_env_cfg import PendulumEnvCfg

controller = controllers.Controller()


def main():
    """Main function."""
    # create environment configuration
    env_cfg = PendulumEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():

            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")

            # sample random actions
            # joint_efforts = torch.randn_like(env.action_manager.action)
            joint_efforts = torch.tensor([[controller.step()]])
            # print(f"Joint effort: {value}")

            # step the environment
            obs, rew, terminated, truncated, info = env.step(joint_efforts)
            # print current orientation of pole
            # print("[Env 0]: Pendulum joint: ", math.degrees(obs["policy"][0][0].item()))
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
