# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse


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
    parser.add_argument(
        "--max_runtime_iterations",
        type=int,
        default=10000,
        help="Maximum number of iterations for the runtime (for all episodes).",
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

    return parser
