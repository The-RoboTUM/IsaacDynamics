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
from isaaclab_dynamics.stochastics.metrics import control_effort
from isaaclab_dynamics.utils.env_utils import configure_logging
from isaaclab_dynamics.utils.io import episode_counts, format_subset, load_database

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# Algorithm configuration
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm == "ppo" else f"skrl_{algorithm}_cfg_entry_point"


def plot_distribution(df_clean, obs_dists, obs_name: str = "obs_0"):
    sns.set_theme()
    obs_values = df_clean[obs_name].values
    x_min, x_max = obs_values.min() - 0.5, obs_values.max() + 0.5
    x_plot = np.linspace(x_min, x_max, 500)
    kde_values = obs_dists[obs_name].evaluate(x_plot)

    plt.figure(figsize=(10, 6))
    sns.histplot(
        obs_values,
        bins=60,
        stat="density",
        color="skyblue",
        label="Data (histogram)",
        edgecolor=None,
    )
    sns.lineplot(x=x_plot, y=kde_values, color="red", label="KDE")

    plt.xlabel(obs_name)
    plt.ylabel("Density")
    plt.title(f"Probability Density of {obs_name}")
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
        kde = gaussian_kde(obs_values)
        obs_dists[f"obs_{i}"] = kde

    print(f"[INFO]: Built KDEs for {state_dim} observation dimensions.")

    # Test any dummy probability
    obs_name = "obs_0"
    value_to_query = 0.5
    probability = obs_dists[obs_name].evaluate([value_to_query])[0]
    print(f"Probability density at {value_to_query} for {obs_name}: {probability}")

    # Plot distributions
    plot_distribution(df_clean, obs_dists, obs_name)

    # Calculate the control effort metric per episode
    kde_model = obs_dists[obs_name]
    obs_values_database = df_clean[obs_name].values
    x_min, x_max = obs_values_database.min(), obs_values_database.max()

    # Run the loop
    # iterations = num_complete_episodes + (
    #     last_episode_len if last_episode_len > 0 else 0
    # )
    I_eff_i = []
    I_eff_s_i = []
    iterations = num_complete_episodes
    for i in range(iterations):
        df_episode, _, _ = format_subset(df, (i * episode_len, (i + 1) * episode_len))
        obs_values_episode = df_episode[obs_name].values
        I_eff = control_effort(kde_model, obs_values_episode, env_cfg.sim.dt * 2, x_min, x_max)
        I_eff_s = I_eff / (env_cfg.sim.dt * 2 * episode_len)
        I_eff_i.append(I_eff)
        I_eff_s_i.append(I_eff_s)

    I_eff = np.average(I_eff_i)
    I_eff_std = np.std(I_eff_i)
    I_eff_s = np.average(I_eff_s_i)
    I_eff_s_std = np.std(I_eff_s_i)
    print(f"INFO: Control effort: {I_eff:.2f} +- {I_eff_std:.2f} Bits (per episode)")
    print(f"INFO: Control effort per second: {I_eff_s:.2f} +- {I_eff_s_std:.2f} Bits/s")


if __name__ == "__main__":
    # Launch the main function
    main()

    # Close the simulation app
    simulation_app.close()
