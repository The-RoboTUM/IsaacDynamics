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

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Probability magic
from scipy.stats import gaussian_kde

import jax.numpy as jnp
import seaborn as sns
from isaaclab_dynamics.stochastics.metrics import delta_entropy
from isaaclab_dynamics.utils.env_utils import configure_logging
from isaaclab_dynamics.utils.io import episode_counts, format_subset, load_database

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# Algorithm configuration
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = f"skrl_{algorithm}_cfg_entry_point"


def plot_distribution_over_time(
    kde_grid,
    data_matrix,
    dim_index=0,
    name="obs",
    save_dir="./plots",
    show=False,
    file_name="unknown",
):
    sns.set_theme()

    # Select timesteps
    _, _, T = data_matrix.shape
    timesteps = [0, T // 2, T - 1]
    labels = ["start", "middle", "end"]

    # Collect all values to compute x range
    all_values = np.concatenate([np.array(data_matrix[dim_index, :, t]) for t in timesteps])
    mean = np.mean(all_values)
    std = np.std(all_values)
    x_min = mean - 3 * std
    x_max = mean + 3 * std
    x_plot = np.linspace(x_min, x_max, 500)

    plt.figure(figsize=(10, 6))

    # Plot single global histogram
    sns.histplot(
        all_values,
        bins=160,
        stat="density",
        color="gray",
        label="All data (histogram)",
        alpha=0.3,
        edgecolor=None,
    )

    for t, label in zip(timesteps, labels):
        kde = kde_grid[dim_index][t]
        kde_values = kde.evaluate(x_plot)
        sns.lineplot(x=x_plot, y=kde_values, label=f"KDE ({label})")

    plt.xlim(x_min, x_max)
    plt.xlabel(f"{name}_{dim_index}")
    plt.ylabel("Density")
    plt.title(f"KDE of {name}_{dim_index} at Start, Middle, End")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{file_name}_{name}_{dim_index}_kde.png")
    plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close()


def build_kde_grid(data_matrix, mode="simplified"):
    dim, num_episodes, timesteps = data_matrix.shape
    kde_grid = []

    for d in range(dim):
        kde_per_timestep = []

        if mode == "simplified":
            values = np.array(data_matrix[d]).reshape(-1)  # shape: (num_episodes * timesteps,)
            if np.allclose(values, 0):
                values += 1e-8 * np.random.randn(values.shape[0])
            kde = gaussian_kde(values)
            kde_per_timestep = [kde] * timesteps  # reuse same KDE for all timesteps

        elif mode == "full":
            for t in range(timesteps):
                values = np.array(data_matrix[d, :, t])  # shape: (num_episodes,)
                if np.allclose(values, 0):
                    values = values.astype(np.float64) + 1e-8 * np.random.randn(values.shape[0])
                kde = gaussian_kde(values)
                kde_per_timestep.append(kde)

        kde_grid.append(kde_per_timestep)

    return kde_grid


def pad(array, max_len):
    return jnp.pad(array, ((0, 0), (0, max_len - array.shape[1])), constant_values=0)


def build_matrices(df_clean, state_dim, action_dim, rewards_dim, episode_num=-1):
    # Collect episodes
    episodes_obs = []
    episodes_actions = []
    episodes_rewards = []
    current_obs = []
    current_actions = []
    current_rewards = []
    episode_count = 0
    for _, row in df_clean.iterrows():
        current_obs.append([row[f"obs_{i}"] for i in range(state_dim)])
        current_actions.append([row[f"action_{i}"] for i in range(action_dim)])
        current_rewards.append([row[f"rewards_{i}"] for i in range(rewards_dim)])

        if row["truncated"]:  # End of episode
            episodes_obs.append(np.array(current_obs).T)  # shape: (state_dim, T)
            episodes_actions.append(np.array(current_actions).T)  # shape: (action_dim, T)
            episodes_rewards.append(np.array(current_rewards).T)
            current_obs, current_actions = [], []
            episode_count += 1
            if episode_num > 0 and episode_count == episode_num:
                break

    # Pad episodes to same length
    max_len = max(e.shape[1] for e in episodes_obs)

    obs_matrix = jnp.stack([pad(e, max_len) for e in episodes_obs], axis=1)  # (state_dim, num_episodes, max_len)
    action_matrix = jnp.stack(
        [pad(e, max_len) for e in episodes_actions], axis=1
    )  # (action_dim, num_episodes, max_len)

    return obs_matrix, action_matrix


def save_control_effort(
    Delta_H_y_u,
    Delta_H_y_u_dt,
    Delta_H_u,
    Delta_H_u_dt,
    Delta_H_y,
    Delta_H_y_dt,
    save_dir="./report_results",
    experiment_name="unknown",
    results_file_name="control_effort.json",
):
    # Ensure save_dir exists
    os.makedirs(save_dir, exist_ok=True)

    # Use Path consistently
    save_path = Path(save_dir) / results_file_name

    # Prepare current entry
    entry = {
        "experiment_name": experiment_name,
        "Delta_H_y_u": Delta_H_y_u,
        "Delta_H_y_u_dt": Delta_H_y_u_dt,
        "Delta_H_u": Delta_H_u,
        "Delta_H_u_dt": Delta_H_u_dt,
        "Delta_H_y": Delta_H_y,
        "Delta_H_y_dt": Delta_H_y_dt,
    }

    # Load existing DB or initialize
    if save_path.exists():
        with save_path.open("r") as f:
            db = json.load(f)
    else:
        db = []

    # Append and save
    db.append(entry)
    with save_path.open("w") as f:
        json.dump(db, f, indent=2)

    print(f"INFO: Control effort (Sensors): {Delta_H_y_u:.2f} Bits (per episode)")
    print(f"INFO: Control effort (Sensors) per second: {Delta_H_y_u_dt:.2f} Bits/s")
    print(f"INFO: Control effort (Controller): {Delta_H_u:.2f} Bits (per episode)")
    print(f"INFO: Control effort (Controller) per second: {Delta_H_u_dt:.2f} Bits/s")
    print(f"INFO: Control effort (System): {Delta_H_y:.2f} Bits (per episode)")
    print(f"INFO: Control effort (System) per second: {Delta_H_y_dt:.2f} Bits/s")
    print(f"Saved control effort entry to {save_path}")


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg, agent_cfg):

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
    df_clean, state_dim, action_dim, rewards_dim = format_subset(df, (0, len(df)))

    # Build the matrices
    obs_matrix, action_matrix = build_matrices(df_clean, state_dim, action_dim, episode_num=args_cli.episode_num)

    # Build the KDEs
    obs_kde_grid = build_kde_grid(obs_matrix, mode=args_cli.stoch_mode)
    action_kde_grid = build_kde_grid(action_matrix, mode=args_cli.stoch_mode)

    print(f"[INFO]: Built KDEs for {state_dim} observation dimensions.")
    print(f"[INFO]: Built KDEs for {action_dim} action dimensions.")

    # Test any dummy probability
    dim = 0
    obs_name = f"obs_{dim}"
    action_name = f"action_{dim}"
    value_to_query = 0.5
    step_to_query = obs_matrix.shape[-1] - 1
    print(step_to_query)
    probability = obs_kde_grid[dim][step_to_query](value_to_query)
    print(f"Probability density at {value_to_query} for {obs_name}: {probability}")

    # Plot obsevation distributions
    plot_distribution_over_time(
        obs_kde_grid,
        obs_matrix,
        name="observation",
        show=args_cli.show_plots,
        save_dir=log_dir + "/plots",
        file_name=args_cli.experiment_name,
    )

    # Plot action distributions
    plot_distribution_over_time(
        action_kde_grid,
        action_matrix,
        name="actions",
        show=args_cli.show_plots,
        save_dir=log_dir + "/plots",
        file_name=args_cli.experiment_name,
    )

    # Calculate the control effort metric per episode

    # observation data
    obs_values_database = df_clean[obs_name].values
    x_min, x_max = obs_values_database.min(), obs_values_database.max()

    # actions data
    actions_values_database = df_clean[action_name].values
    u_min, u_max = actions_values_database.min(), actions_values_database.max()

    # calculate entropies
    Delta_H_y, Delta_H_y_u, Delta_H_u = delta_entropy(dim, obs_kde_grid, action_kde_grid, x_min, x_max, u_min, u_max)
    Delta_H_y_dt = Delta_H_y / (env_cfg.sim.dt * 2 * episode_len)
    Delta_H_y_u_dt = Delta_H_y_u / (env_cfg.sim.dt * 2 * episode_len)
    Delta_H_u_dt = Delta_H_u / (env_cfg.sim.dt * 2 * episode_len)

    # Save the current entry in a JSON file
    save_control_effort(
        Delta_H_y_u,
        Delta_H_y_u_dt,
        Delta_H_u,
        Delta_H_u_dt,
        Delta_H_y,
        Delta_H_y_dt,
        save_dir="./report_results",
        experiment_name=args_cli.experiment_name,
        results_file_name=args_cli.results_file_name,
    )


if __name__ == "__main__":
    # Launch the main function
    main()

    # Close the simulation app
    simulation_app.close()
