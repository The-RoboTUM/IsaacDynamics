# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to launch an Isaac Sim environment and use variable controllers for Reinforcement Learning (RL).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# GPU memory handling
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def setup_parser(applauncher):
    """
    Parse the command-line arguments.

    Args:
        applauncher: AppLauncher instance to add specific CLI arguments.

    Returns:
        tuple: Parsed CLI arguments (args_cli) and Hydra arguments (hydra_args).
    """
    parser = argparse.ArgumentParser(description="Run an RL agent with skrl.")

    # Video-related arguments
    parser.add_argument(
        "--video",
        action="store_true",
        default=False,
        help="Record videos during training.",
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=200,
        help="Length of recorded video (in steps).",
    )
    parser.add_argument(
        "--video_interval",
        type=int,
        default=2000,
        help="Interval between video recordings (in steps).",
    )

    # Environment-related arguments
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
    parser.add_argument(
        "--distributed",
        action="store_true",
        default=False,
        help="Run training with multiple GPUs or nodes.",
    )
    parser.add_argument("--controller", type=str, default=None, help="Name of the controller.")

    # Checkpoint-related arguments
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
    parser.add_argument(
        "--use_pretrained_checkpoint",
        action="store_true",
        help="Use the pretrained checkpoint from Nucleus.",
    )

    # Training-related arguments
    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Specify training or testing mode.",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=None,
        help="Maximum RL policy training iterations.",
    )
    parser.add_argument(
        "--ml_framework",
        type=str,
        default="torch",
        choices=["torch", "jax", "jax-numpy"],
        help="ML framework for training the skrl agent.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="PPO",
        choices=["AMP", "PPO", "IPPO", "MAPPO"],
        help="The RL algorithm used for training the skrl agent.",
    )

    # Experiment-specific arguments
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="",
        help="Name of the experiment for logging purposes.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        default=False,
        help="Record observations and actions.",
    )

    # Debugging and runtime arguments
    parser.add_argument("--debugger", action="store_true", default=False, help="Enable debugging mode.")
    parser.add_argument(
        "--disable_fabric",
        action="store_true",
        default=False,
        help="Disable fabric operations (use USD I/O).",
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        default=False,
        help="Run in real-time mode if supported.",
    )

    # AppLauncher-specific arguments
    applauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()

    return args_cli, hydra_args


args_cli, hydra_args = setup_parser(AppLauncher)

# Clear Hydra-specific arguments from sys.argv
sys.argv = [sys.argv[0]] + hydra_args

# Initialize simulation app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Debugger setup (if enabled)
if args_cli.debugger:
    pydevd_name = "pydevd_pycharm"
    pydevd = __import__(pydevd_name)
    pydevd.settrace(
        host="localhost",
        port=5678,
        stdoutToServer=True,
        stderrToServer=True,  # Optional: waiting for debugger connection
    )

# Enable cameras for video recording if necessary
if args_cli.video:
    args_cli.enable_cameras = True

"""Rest everything follows."""

# Post app-launcher imports
import gymnasium as gym
import random

from isaaclab_dynamics.controllers import base_controllers, wrappers
from isaaclab_dynamics.utilities import configure_logging, save_configuration

# from isaaclab.envs import (
#     DirectMARLEnv,
#     DirectMARLEnvCfg,
#     DirectRLEnvCfg,
#     ManagerBasedRLEnvCfg,
#     multi_agent_to_single_agent,
# )
from isaaclab.utils.dict import print_dict

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


def record_env(env, log_dir):
    """
    Optionally wrap the environment with a video recorder.

    Args:
        env: Gymnasium environment instance.
        log_dir: Path to save videos.

    Returns:
        env: Environment wrapped with a video recorder.
    """
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", args_cli.mode),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Video recording activated.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    return env


def setup_env(env_cfg):
    """
    Prepare and initialize the simulation environment.

    Args:
        env_cfg: Configuration object for the environment.

    Returns:
        tuple: Updated environment configuration, initialized environment, seed, and time step (dt).
    """
    env_cfg.scene.num_envs = args_cli.num_envs or env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Retrieve time step (dt)
    try:
        dt = env.physics_dt
    except AttributeError:
        dt = env.unwrapped.physics_dt

    # Set random seed if not explicitly set
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    env_cfg.seed = args_cli.seed

    return env_cfg, env, args_cli.seed, dt


# Algorithm configuration
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm == "ppo" else f"skrl_{algorithm}_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg, agent_cfg):
    """
    Main function for setting up controllers and running the simulation.

    Args:
        env_cfg: Configuration for the environment.
        agent_cfg: Configuration for the RL agent.
    """
    # Initialize environment
    env_cfg, env, seed, dt = setup_env(env_cfg)

    # Controller setup
    if args_cli.controller == "rl":
        controller = base_controllers.ControllerRL(args_cli=args_cli)
    elif args_cli.controller == "keyboard":
        controller = base_controllers.ControllerKeyboard(args_cli=args_cli)
    elif args_cli.controller == "pid":
        controller = base_controllers.ControllerPID(args_cli=args_cli)
    else:
        raise ValueError("Invalid controller specified.")
    if args_cli.record:
        controller = wrappers.ControllerLogger(controller, args_cli=args_cli)

    # Wrap and configure environment
    env, _ = controller.setup(env, dt, agent_cfg, seed=seed)

    # Logging and video recording setup
    log_dir, resume_path = configure_logging(args_cli, agent_cfg)
    env = record_env(env, log_dir)

    # Save configurations
    save_configuration(args_cli, log_dir, env_cfg, agent_cfg)

    # Run the environment with the controller
    controller.run(env, simulation_app.is_running, controller.iterate, resume_path=resume_path)
    env.close()


if __name__ == "__main__":
    # Launch the main function
    main()

    # Close the simulation app
    simulation_app.close()
