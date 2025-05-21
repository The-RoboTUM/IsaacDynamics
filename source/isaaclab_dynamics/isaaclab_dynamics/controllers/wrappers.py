# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import atexit
import datetime
import numpy as np
import os
import sqlite3
import torch

import jax
import pandas as pd
from isaaclab_dynamics.controllers.base_controllers import ControllerBase
from skrl.envs.wrappers.jax.isaaclab_envs import IsaacLabWrapper


class ControllerLogger(ControllerBase):
    """
    A controller wrapper that logs observations, actions, and optionally other info
    from another controller.
    """

    def __init__(self, controller: ControllerBase, log_dir, args_cli=None):
        super().__init__(args_cli)
        self.controller = controller
        self.log = {
            "id": [],
            "step": [],
            "obs": [],
            "actions": [],
            "truncated": [],
        }
        self.log_dir = log_dir
        self.saved_logs = False
        atexit.register(self.save_log)

    def setup(self, env, dt, config=None, seed=None):
        env, cfg = self.controller.setup(env, dt, config=config, seed=seed)
        return env, cfg

    def run(self, env, alive_check, iterate, args=None, resume_path=None):
        # Set up the simulation
        # obs = self.reset(env)
        # timestep = 0

        # Run the simulation
        self.controller.run(env, alive_check, iterate, args=args, resume_path=resume_path)

        # Save the logs when done
        self.save_log()

    def reset(self, env):
        obs = self.controller.reset(env)

        # saving data into internal log
        if self.episode_count < self.args_cli.max_episodes:
            self.log["id"].append(0)
            self.log["step"].append(0)
            if isinstance(env, IsaacLabWrapper):
                self.log["obs"].append(str(self.ground(obs)))
            else:
                self.log["obs"].append(str(self.ground(obs["policy"])))
            self.log["actions"].append(str([0]))  # TODO: Modify to handle variable dimensional action spaces
            self.log["truncated"].append(str([True]))

        return obs

    def iterate(self, env, obs, args=None):
        actions, obs, rewards, terminated, truncated, info = self.controller.iterate(env, obs, args=args)

        # saving data into internal log
        if self.episode_count < self.args_cli.max_episodes:
            self.log["id"].append(self.controller.step_runtime)
            self.log["step"].append(self.ground(env.unwrapped.episode_length_buf)[0])
            if isinstance(env, IsaacLabWrapper):
                self.log["obs"].append(str(self.ground(obs)))
            else:
                self.log["obs"].append(str(self.ground(obs["policy"])))
            self.log["actions"].append(str(self.ground(actions)))
            self.log["truncated"].append(str(self.ground(truncated)))

        return actions, obs, rewards, terminated, truncated, info

    @staticmethod
    def ground(tensor):
        if isinstance(tensor, dict):
            return {k: v.detach().cpu().numpy().flatten().tolist() for k, v in tensor.items()}
        elif isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy().flatten().tolist()
        elif isinstance(tensor, jax.Array):
            return np.array(tensor).flatten().tolist()
        else:
            raise TypeError(f"Unsupported type for ground(): {type(tensor)}")

    def step(self, obs, args=None):
        return self.controller.step(obs, args)

    def save_log(self):
        if self.saved_logs is False:
            # Create a new folder inside self.log_dir
            if self.log_dir is None:
                raise ValueError("log_dir is not specified.")

            folder_path = os.path.join(self.log_dir, "data_logs")
            os.makedirs(folder_path, exist_ok=True)

            # Define the database file path
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            db_path = os.path.join(folder_path, f"{timestamp}_logs.db")

            # Save the logs into an SQLite database
            conn = sqlite3.connect(db_path)
            try:
                df = pd.DataFrame(self.log)
                df = df.iloc[:-1]  # drop the last element since it is the start of the next episode
                df.to_sql("logs", conn, if_exists="replace", index=False)
            finally:
                conn.close()

            print(f"[INFO]: Logs saved to {db_path}")
            self.saved_logs = True

    def __del__(self):
        self.save_log()
