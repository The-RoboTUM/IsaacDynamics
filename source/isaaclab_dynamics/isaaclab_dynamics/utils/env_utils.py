# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from datetime import datetime

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_tasks.utils import get_checkpoint_path


def save_configuration(args_cli, log_dir, env_cfg, agent_cfg):
    """
    Save the provided environment and agent configurations to the specified logging directory
    during the training phase. The configurations are serialized in both YAML and Pickle formats
    and stored within a subdirectory named 'params'.

    Args:
        args_cli: Parsed command-line arguments that indicate operational mode.
        log_dir: The directory where configuration files are to be saved.
        env_cfg: Dictionary containing environment configuration details.
        agent_cfg: Dictionary containing agent configuration details.
    """
    if args_cli.mode == "train":
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)


def configure_logging(args_cli, agent_cfg):
    """
    Configure and resolve logging directory and checkpoint paths based on the operation mode
    (train or test). It determines the paths for saving experiment logs and checkpoints
    for continuing or testing experiments.

    Args:
        args_cli (Namespace): Parsed command-line arguments with details
                              like algorithm, mode, checkpoint paths, etc.
        agent_cfg (dict): Dictionary containing settings for experiments
                          like log directories and experiment names.

    Returns:
        Tuple[Optional[str], Optional[str]]: Resolved logging directory (str) and checkpoint path (str),
                                             or (None, None) if resolution fails.
    """
    log_dir = None
    resume_path = None
    algorithm = args_cli.algorithm.lower()

    if args_cli.mode == "train":
        # Set up the root log directory path
        log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Logging experiment in directory: {log_root_path}")

        # Generate log directory name based on timestamp and experiment details
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
        print(f"Exact experiment name requested from command line: {log_dir}")

        if args_cli.experiment_name:
            log_dir += f"_{args_cli.experiment_name}"
        elif agent_cfg["agent"]["experiment"]["experiment_name"]:
            log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'

        # Update agent configuration with directory paths
        agent_cfg["agent"]["experiment"]["directory"] = log_root_path
        agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
        log_dir = os.path.join(log_root_path, log_dir)

        # Resolve checkpoint path for resuming training
        resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    elif args_cli.mode == "test":
        # Resolve log directory and checkpoint path for testing
        log_root_path = os.path.abspath(os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"]))
        print(f"[INFO] Loading experiment from directory: {log_root_path}")

        if args_cli.use_pretrained_checkpoint:
            resume_path = get_published_pretrained_checkpoint("skrl", args_cli.task)
            if not resume_path:
                print("[INFO] Unfortunately, a pre-trained checkpoint is unavailable for this task.")
                return None, None
        elif args_cli.checkpoint:
            resume_path = os.path.abspath(args_cli.checkpoint)
        else:
            # Automatically retrieve the latest checkpoint
            resume_path = get_checkpoint_path(
                log_root_path,
                run_dir=f".*_{algorithm}_{args_cli.ml_framework}",
                other_dirs=["checkpoints"],
            )
        log_dir = os.path.dirname(os.path.dirname(resume_path))
    args_cli.log_dir = log_dir

    return log_dir, resume_path
