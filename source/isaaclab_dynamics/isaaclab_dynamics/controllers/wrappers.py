# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

import pandas as pd
from isaaclab_dynamics.controllers.base_controllers import ControllerBase


class ControllerLogger(ControllerBase):
    """
    A controller wrapper that logs observations, actions, and optionally other info
    from another controller.
    """

    def __init__(self, controller: ControllerBase, args_cli=None):
        super().__init__(args_cli)
        self.controller = controller
        self.log = {
            "id": [],
            "step": [],
            "obs": [],
            "actions": [],
        }
        self.step_runtime = 0
        self.step_count = 0

    def setup(self, env, dt, config=None, seed=None):
        env, cfg = self.controller.setup(env, dt, config=config, seed=seed)
        return env, cfg

    def run(self, env, alive_check, iterate, args=None, resume_path=None):
        self.controller.run(env, alive_check, iterate, args=args, resume_path=resume_path)
        self.save_log()  # Save the logs when done

    def iterate(self, env, obs, args=None):
        actions, obs, rewards, terminated, truncated, info = self.controller.iterate(env, obs, args=args)

        # handling of the data takes place here
        self.log["id"].append(self.step_runtime)
        self.log["step"].append(self.step_count)
        self.log["obs"].append(self.ground(obs["policy"]))
        self.log["actions"].append(self.ground(actions))
        self.step_count += 1
        self.step_runtime += 1

        # handle episode termination
        if terminated:
            self.step_count = 0

        return actions, obs, rewards, terminated, truncated, info

    @staticmethod
    def ground(tensor):
        if isinstance(tensor, dict):
            return {k: v.detach().cpu().numpy().flatten().tolist() for k, v in tensor.items()}
        elif isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy().flatten().tolist()
        else:
            raise TypeError(f"Unsupported type for ground(): {type(tensor)}")

    def step(self, obs, args=None):
        return self.controller.step(obs, args)

    def save_log(self, filename="controller_log.csv"):
        df = pd.DataFrame(self.log)
        print(df)
        # df.to_csv(filename, index=False)
        # print(f"[Logger] Saved log to {filename}")
