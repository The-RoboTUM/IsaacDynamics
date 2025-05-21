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
import numpy as np

# Probability magic
from scipy.stats import gaussian_kde

import seaborn as sns
from isaaclab_dynamics.stochastics.metrics import delta_entropy
from isaaclab_dynamics.utils.env_utils import configure_logging
from isaaclab_dynamics.utils.io import episode_counts, format_subset, load_database

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# Algorithm configuration
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm == "ppo" else f"skrl_{algorithm}_cfg_entry_point"


def plot_distribution(df_clean, dists, name: str = "obs_0"):
    sns.set_theme()
    values = df_clean[name].values

    # Compute mean and std for ±3σ bounds
    mean = np.mean(values)
    std = np.std(values)
    x_min = mean - 3 * std
    x_max = mean + 3 * std

    x_plot = np.linspace(x_min, x_max, 500)
    kde_values = dists[name].evaluate(x_plot)

    plt.figure(figsize=(10, 6))
    sns.histplot(
        values,
        bins=600,
        stat="density",
        color="skyblue",
        label="Data (histogram)",
        edgecolor=None,
    )
    sns.lineplot(x=x_plot, y=kde_values, color="red", label="KDE")

    plt.xlim(x_min, x_max)
    plt.xlabel(name)
    plt.ylabel("Density")
    plt.title(f"Probability Density of {name} (±3σ)")
    plt.legend()
    plt.grid(True)
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

    # Prepare the whole dataframe
    df_clean, state_dim, action_dim = format_subset(df, (0, len(df)))

    # Model the KDE per observation
    obs_dists = {}
    for i in range(state_dim):
        obs_values = df_clean[f"obs_{i}"].values  # All values of obs_i
        kde_obs = gaussian_kde(obs_values)
        obs_dists[f"obs_{i}"] = kde_obs

    actions_dists = {}
    for i in range(action_dim):
        action_values = df_clean[f"action_{i}"].values
        kde_actions = gaussian_kde(action_values)
        actions_dists[f"action_{i}"] = kde_actions

    print(f"[INFO]: Built KDEs for {state_dim} observation dimensions.")
    print(f"[INFO]: Built KDEs for {action_dim} action dimensions.")

    # Test any dummy probability
    obs_name = "obs_0"
    action_name = "action_0"
    value_to_query = 0.5
    probability = obs_dists[obs_name].evaluate([value_to_query])[0]
    print(f"Probability density at {value_to_query} for {obs_name}: {probability}")

    # Plot distributions
    plot_distribution(df_clean, obs_dists, obs_name)
    plot_distribution(df_clean, actions_dists, action_name)

    # Calculate the control effort metric per episode

    # observation data
    kde_model_obs = obs_dists[obs_name]
    obs_values_database = df_clean[obs_name].values
    x_min, x_max = obs_values_database.min(), obs_values_database.max()

    # actions data
    kde_model_actions = actions_dists[action_name]
    actions_values_database = df_clean[action_name].values
    u_min, u_max = actions_values_database.min(), actions_values_database.max()

    # calculate entropies
    Delta_H_y, Delta_H_y_u, Delta_H_u = delta_entropy(kde_model_obs, kde_model_actions, x_min, x_max, u_min, u_max)
    Delta_H_y_dt = Delta_H_y / (env_cfg.sim.dt * 2 * episode_len)
    Delta_H_y_u_dt = Delta_H_y_u / (env_cfg.sim.dt * 2 * episode_len)
    Delta_H_u_dt = Delta_H_u / (env_cfg.sim.dt * 2 * episode_len)

    print(f"INFO: Control effort (Sensors): {Delta_H_y_u:.2f} Bits (per episode)")
    print(f"INFO: Control effort (Sensors) per second: {Delta_H_y_u_dt:.2f} Bits/s")
    print(f"INFO: Control effort (Controller): {Delta_H_u:.2f} Bits (per episode)")
    print(f"INFO: Control effort (Controller) per second: {Delta_H_u_dt:.2f} Bits/s")
    print(f"INFO: Control effort (System): {Delta_H_y:.2f} Bits (per episode)")
    print(f"INFO: Control effort (System) per second: {Delta_H_y_dt:.2f} Bits/s")


if __name__ == "__main__":
    # Launch the main function
    main()

    # Close the simulation app
    simulation_app.close()
