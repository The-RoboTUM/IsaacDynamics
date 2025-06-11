# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import atexit
import contextlib
import copy
import datetime
import numpy as np
import os
import sqlite3
import sys
import tqdm

import jax.numpy as jnp
import pandas as pd
from skrl.agents.jax import Agent
from skrl.envs.wrappers.jax import Wrapper
from skrl.trainers.jax import Trainer

EXPOSED_TRAINER_DEFAULT_CONFIG = {
    "timesteps": 100000,  # number of timesteps to train for
    "headless": False,  # whether to use headless mode (no rendering)
    "disable_progressbar": False,  # whether to disable the progressbar. If None, disable on non-TTY
    "close_environment_at_exit": True,  # whether to close the environment on normal program termination
    "environment_info": "episode",  # key used to get and log environment info
    "stochastic_evaluation": False,  # whether to use actions rather than (deterministic) mean actions during evaluation
}


class ExposedTrainer(Trainer):
    def __init__(self, env: Wrapper, agents: Agent | list[Agent], agents_scope=None, cfg=None):
        _cfg = copy.deepcopy(EXPOSED_TRAINER_DEFAULT_CONFIG)
        _cfg.update(cfg or {})
        agents_scope = agents_scope or []
        super().__init__(env=env, agents=agents, agents_scope=agents_scope, cfg=_cfg)

        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.init(trainer_cfg=self.cfg)
        else:
            self.agents.init(trainer_cfg=self.cfg)

        self.record = self.cfg.get("record", False)
        self.log_dir = self.cfg.get("log_dir")
        self.saved_logs = False
        if self.record:
            self.log = {
                "id": [],
                "step": [],
                "obs": [],
                "actions": [],
                "rewards": [],
                "terminated": [],
                "truncated": [],
            }
            atexit.register(self.save_log)

    def train(self):
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.set_running_mode("train")
        else:
            self.agents.set_running_mode("train")

        if self.num_simultaneous_agents == 1:
            if self.env.num_agents == 1:
                self.single_agent_train()
            else:
                self.multi_agent_train()
            if self.record:
                self.save_log()
            return

        states, infos = self.env.reset()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps),
            disable=self.disable_progressbar,
            file=sys.stdout,
        ):
            for agent in self.agents:
                agent.pre_interaction(timestep, self.timesteps)

            actions = jnp.vstack([
                agent.act(states[scope[0] : scope[1]], timestep, self.timesteps)[0]
                for agent, scope in zip(self.agents, self.agents_scope)
            ])
            next_states, rewards, terminated, truncated, infos = self.env.step(actions)

            if not self.headless:
                self.env.render()

            for agent, scope in zip(self.agents, self.agents_scope):
                agent.record_transition(
                    states=states[scope[0] : scope[1]],
                    actions=actions[scope[0] : scope[1]],
                    rewards=rewards[scope[0] : scope[1]],
                    next_states=next_states[scope[0] : scope[1]],
                    terminated=terminated[scope[0] : scope[1]],
                    truncated=truncated[scope[0] : scope[1]],
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

            if self.record:
                self.log["id"].append(timestep)
                self.log["step"].append(int(timestep))
                self.log["obs"].append(str(np.array(states).flatten().tolist()))
                self.log["actions"].append(str(np.array(actions).flatten().tolist()))
                self.log["rewards"].append(str(np.array(rewards).flatten().tolist()))
                self.log["terminated"].append(str(np.array(terminated).flatten().tolist()))
                self.log["truncated"].append(str(np.array(truncated).flatten().tolist()))

            for agent in self.agents:
                agent.post_interaction(timestep, self.timesteps)

            if terminated.any() or truncated.any():
                states, infos = self.env.reset()
            else:
                states = next_states

    def eval(self):
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.set_running_mode("eval")
        else:
            self.agents.set_running_mode("eval")

        if self.num_simultaneous_agents == 1:
            if self.env.num_agents == 1:
                self.single_agent_eval()
            else:
                self.multi_agent_eval()
            if self.record:
                self.save_log()
            return

        states, infos = self.env.reset()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps),
            disable=self.disable_progressbar,
            file=sys.stdout,
        ):
            for agent in self.agents:
                agent.pre_interaction(timestep, self.timesteps)

            outputs = [
                agent.act(states[scope[0] : scope[1]], timestep, self.timesteps)
                for agent, scope in zip(self.agents, self.agents_scope)
            ]
            actions = jnp.vstack([
                (output[0] if self.stochastic_evaluation else output[-1].get("mean_actions", output[0]))
                for output in outputs
            ])
            next_states, rewards, terminated, truncated, infos = self.env.step(actions)

            if not self.headless:
                self.env.render()

            for agent, scope in zip(self.agents, self.agents_scope):
                agent.record_transition(
                    states=states[scope[0] : scope[1]],
                    actions=actions[scope[0] : scope[1]],
                    rewards=rewards[scope[0] : scope[1]],
                    next_states=next_states[scope[0] : scope[1]],
                    terminated=terminated[scope[0] : scope[1]],
                    truncated=truncated[scope[0] : scope[1]],
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

            if self.record:
                self.log["id"].append(timestep)
                self.log["step"].append(int(timestep))
                self.log["obs"].append(str(np.array(states).flatten().tolist()))
                self.log["actions"].append(str(np.array(actions).flatten().tolist()))
                self.log["rewards"].append(str(np.array(rewards).flatten().tolist()))
                self.log["terminated"].append(str(np.array(terminated).flatten().tolist()))
                self.log["truncated"].append(str(np.array(truncated).flatten().tolist()))

            for agent in self.agents:
                super(type(agent), agent).post_interaction(timestep, self.timesteps)

            if terminated.any() or truncated.any():
                states, infos = self.env.reset()
            else:
                states = next_states

    def single_agent_train(self) -> None:
        """Train agent

        This method executes the following steps in loop:

        - Pre-interaction
        - Compute actions
        - Interact with the environments
        - Render scene
        - Record transitions
        - Post-interaction
        - Reset environments
        """
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents == 1, "This method is not allowed for multi-agents"

        # reset env
        states, infos = self.env.reset()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps),
            disable=self.disable_progressbar,
            file=sys.stdout,
        ):

            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with contextlib.nullcontext():
                # compute actions
                actions = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)[0]

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                # render scene
                if not self.headless:
                    self.env.render()

                # record the environments' transitions
                self.agents.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

            if self.record:
                self.log["id"].append(timestep)
                self.log["step"].append(int(timestep))
                self.log["obs"].append(str(np.array(states).flatten().tolist()))
                self.log["actions"].append(str(np.array(actions).flatten().tolist()))
                self.log["rewards"].append(str(np.array(rewards).flatten().tolist()))
                self.log["terminated"].append(str(np.array(terminated).flatten().tolist()))
                self.log["truncated"].append(str(np.array(truncated).flatten().tolist()))

            # post-interaction
            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            if self.env.num_envs > 1:
                states = next_states
            else:
                if terminated.any() or truncated.any():
                    with contextlib.nullcontext():
                        states, infos = self.env.reset()
                else:
                    states = next_states

    def single_agent_eval(self) -> None:
        """Evaluate agent

        This method executes the following steps in loop:

        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Reset environments
        """
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents == 1, "This method is not allowed for multi-agents"

        # reset env
        states, infos = self.env.reset()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps),
            disable=self.disable_progressbar,
            file=sys.stdout,
        ):

            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with contextlib.nullcontext():
                # compute actions
                outputs = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)
                actions = outputs[0] if self.stochastic_evaluation else outputs[-1].get("mean_actions", outputs[0])

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                # render scene
                if not self.headless:
                    self.env.render()

                # write data to TensorBoard
                self.agents.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

            if self.record:
                self.log["id"].append(timestep)
                self.log["step"].append(int(timestep))
                self.log["obs"].append(str(np.array(states).flatten().tolist()))
                self.log["actions"].append(str(np.array(actions).flatten().tolist()))
                self.log["rewards"].append(str(np.array(rewards).flatten().tolist()))
                self.log["terminated"].append(str(np.array(terminated).flatten().tolist()))
                self.log["truncated"].append(str(np.array(truncated).flatten().tolist()))

            # post-interaction
            super(type(self.agents), self.agents).post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            if self.env.num_envs > 1:
                states = next_states
            else:
                if terminated.any() or truncated.any():
                    with contextlib.nullcontext():
                        states, infos = self.env.reset()
                else:
                    states = next_states

    def save_log(self):
        if self.saved_logs or not self.record:
            return
        if self.log_dir is None:
            raise ValueError("log_dir must be specified if record=True")

        os.makedirs(os.path.join(self.log_dir, "data_logs"), exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        db_path = os.path.join(self.log_dir, "data_logs", f"{timestamp}_logs.db")

        conn = sqlite3.connect(db_path)
        try:
            pd.DataFrame(self.log).to_sql("logs", conn, if_exists="replace", index=False)
        finally:
            conn.close()
        print(f"[INFO] Logs saved to {db_path}")
        self.saved_logs = True

    def __del__(self):
        self.save_log()
