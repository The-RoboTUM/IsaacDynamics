# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to launch an Isaac Sim environment and use variable controllers for Reinforcement Learning (RL).
"""

"""Launch Isaac Sim Simulator first."""

import sys

from isaaclab_dynamics.utils.io import setup_parser

from isaaclab.app import AppLauncher

# Handle user input
parser = setup_parser()

# AppLauncher-specific arguments
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Clear Hydra-specific arguments from sys.argv
sys.argv = [sys.argv[0]] + hydra_args

# Initialize simulation app in headless mode
args_cli.headless = True
args_cli.mode = "test"
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

"""Rest everything follows."""

import matplotlib.pyplot as plt

import seaborn as sns
from isaaclab_dynamics.utils.env_utils import configure_logging
from isaaclab_dynamics.utils.io import episode_counts, format_subset, load_database

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# Algorithm configuration
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm == "ppo" else f"skrl_{algorithm}_cfg_entry_point"


def plot_episode(df_sub, state_dim: int, action_dim: int):
    sns.set_theme()

    # Set up the figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot observations
    for i in range(state_dim):
        sns.lineplot(x="step", y=f"obs_{i}", data=df_sub, ax=axs[0], label=f"obs_{i}")
    axs[0].set_ylabel("Observations")
    axs[0].legend()
    axs[0].grid(True)

    # Plot actions
    for i in range(action_dim):
        sns.lineplot(x="step", y=f"action_{i}", data=df_sub, ax=axs[1], label=f"action_{i}")
    axs[1].set_xlabel("Step")
    axs[1].set_ylabel("Actions")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg, agent_cfg):
    """ """

    # Loading the logging directories
    log_dir, _ = configure_logging(args_cli, agent_cfg)

    # Load the database
    df = load_database(log_dir)

    # Inspect the database
    num_complete_episodes, episode_len, last_episode_len = episode_counts(df)
    print(f"[INFO]: Number of complete episodes: {num_complete_episodes}")
    print(f"[INFO]:Episode length: {episode_len}")
    if last_episode_len > 0:
        print(f"[INFO]:Last episode length: {last_episode_len}")
    else:
        print("[INFO]:No incomplete episodes.")

    # Load the first episode
    df_sub, state_dim, action_dim = format_subset(df, (0, episode_len))

    # Plot results
    plot_episode(df_sub, state_dim, action_dim)


if __name__ == "__main__":
    # Launch the main function
    main()

    # Close the simulation app
    simulation_app.close()
