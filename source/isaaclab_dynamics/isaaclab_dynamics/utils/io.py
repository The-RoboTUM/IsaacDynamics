# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import ast
import json
import os
import sqlite3

import pandas as pd


def setup_parser():
    """
    Parse the command-line arguments.

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
    parser.add_argument(
        "--spring",
        action="store_true",
        default=False,
        help="Enable a spring in the joints.",
    )
    parser.add_argument(
        "--spring_setpoint",
        type=float,
        default=90,
        help="Setpoint for the spring.",
    )
    parser.add_argument(
        "--range",
        type=float,
        default=5.0,
        help="Range for the random controller.",
    )

    # Checkpoint-related arguments
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
    parser.add_argument(
        "--use_pretrained_checkpoint",
        action="store_true",
        help="Use the pretrained checkpoint from Nucleus.",
    )
    parser.add_argument("--duration", type=float, default=5.0, help="Time in seconds for each run.")

    # Training-related arguments
    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Specify training or testing mode.",
    )
    parser.add_argument(
        "--max_rollouts",
        type=int,
        default=100,
        help="Maximum episodes to run for.",
    )
    parser.add_argument(
        "--ml_framework",
        type=str,
        default="jax",
        choices=["torch", "jax", "jax-numpy"],
        help="ML framework for training the skrl agent.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="PPO",
        # choices=["AMP", "PPO", "IPPO", "MAPPO"],
        help="The algorithm to create the agent.",
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
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=10,
        help="Maximum number of episodes to run.",
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

    # Logging related arguments
    parser.add_argument(
        "--show_plots",
        action="store_true",
        default=False,
        help="Show the probability distribution plots.",
    )

    # Metric related arguments
    parser.add_argument(
        "--stoch_mode",
        type=str,
        default="full",
        help="Mode for the control effort metric calculation.",
    )
    parser.add_argument(
        "--episode_num",
        type=int,
        default=-1,
        help="Number of episodes to calculate the metric.",
    )
    parser.add_argument(
        "--results_file_name",
        type=str,
        default="control_effort.json",
        help="Json file name for storing the results of the metric calculation.",
    )

    return parser


def episode_counts(df):
    steps = df["step"].values
    total_steps = len(steps)

    episode_len = 0
    for i in range(1, len(steps)):
        if steps[i] <= steps[i - 1]:
            episode_len = i
            break
    if episode_len == 0:
        episode_len = total_steps

    last_episode_len = total_steps % episode_len
    num_complete_episodes = total_steps // episode_len

    return num_complete_episodes, episode_len, last_episode_len


def safe_json_bool(x):
    try:
        return json.loads(x.strip())[0]
    except Exception:
        raise ValueError(f"Bad value for truncated: {repr(x)}")


def format_subset(df, ids: tuple[int, int]):
    # Filter by id
    df_sub = df[(df["id"] >= ids[0]) & (df["id"] < ids[1])].copy()
    df_sub.reset_index(drop=True, inplace=True)

    # Decode obs and actions if they are strings
    if isinstance(df_sub["obs"].iloc[0], str):
        df_sub["obs"] = df_sub["obs"].apply(json.loads)
    if isinstance(df_sub["actions"].iloc[0], str):
        df_sub["actions"] = df_sub["actions"].apply(json.loads)

    # Decode the Truncated flag
    df_sub["truncated"] = df_sub["truncated"].apply(lambda x: ast.literal_eval(x)[0])

    # Count the state and action space
    state_dim = len(df_sub["obs"].iloc[0])
    action_dim = len(df_sub["actions"].iloc[0])
    rewards_dim = len(df_sub["rewards"].iloc[0])

    # Expand obs into separate columns
    obs_expanded = pd.DataFrame(df_sub["obs"].tolist(), columns=[f"obs_{i}" for i in range(state_dim)])
    df_sub = pd.concat([df_sub.drop(columns=["obs"]), obs_expanded], axis=1)

    # Expand actions into separate columns (if needed)
    actions_expanded = pd.DataFrame(df_sub["actions"].tolist(), columns=[f"action_{i}" for i in range(action_dim)])
    df_sub = pd.concat([df_sub.drop(columns=["actions"]), actions_expanded], axis=1)

    # Expand rewards into separate columns
    rewards_expanded = pd.DataFrame(df_sub["rewards"].tolist(), columns=[f"rewards_{i}" for i in range(rewards_dim)])
    df_sub = pd.concat([df_sub.drop(columns=["rewards"]), rewards_expanded], axis=1)

    return df_sub, state_dim, action_dim, rewards_dim


def load_database(log_dir):
    # Loading the exact database directory
    db_folder = os.path.join(log_dir, "data_logs")
    db_files = sorted([f for f in os.listdir(db_folder) if f.endswith(".db")])
    if not db_files:
        raise FileNotFoundError(f"No database files found in {db_folder}")
    db_path = os.path.join(db_folder, db_files[-1])  # Take the most recent

    # Connect and load DataFrame
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM logs", conn)
    finally:
        conn.close()

    return df
