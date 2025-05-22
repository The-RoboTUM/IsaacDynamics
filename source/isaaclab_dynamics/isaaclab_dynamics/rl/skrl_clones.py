# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# from collections.abc import Mapping
# from typing import Any, Type, Union

from skrl.utils.runner.jax.runner import Runner


class ExpandedRunner(Runner):
    def _component(self, name: str) -> type:
        """Get skrl component (e.g.: agent, trainer, etc..) from string identifier

        :return: skrl component
        """
        component = None
        name = name.lower()
        # model
        if name == "gaussianmixin":
            from skrl.utils.model_instantiators.jax import gaussian_model as component
        elif name == "categoricalmixin":
            from skrl.utils.model_instantiators.jax import categorical_model as component
        elif name == "multicategoricalmixin":
            from skrl.utils.model_instantiators.jax import multicategorical_model as component
        elif name == "deterministicmixin":
            from skrl.utils.model_instantiators.jax import deterministic_model as component
        # memory
        elif name == "randommemory":
            from skrl.memories.jax import RandomMemory as component
        # agent
        elif name in ["a2c", "a2c_default_config"]:
            from skrl.agents.jax.a2c import A2C, A2C_DEFAULT_CONFIG

            component = A2C_DEFAULT_CONFIG if "default_config" in name else A2C
        elif name in ["cem", "cem_default_config"]:
            from skrl.agents.jax.cem import CEM, CEM_DEFAULT_CONFIG

            component = CEM_DEFAULT_CONFIG if "default_config" in name else CEM
        elif name in ["ddpg", "ddpg_default_config"]:
            from skrl.agents.jax.ddpg import DDPG, DDPG_DEFAULT_CONFIG

            component = DDPG_DEFAULT_CONFIG if "default_config" in name else DDPG
        elif name in ["ddqn", "ddqn_default_config"]:
            from skrl.agents.jax.dqn import DDQN, DDQN_DEFAULT_CONFIG

            component = DDQN_DEFAULT_CONFIG if "default_config" in name else DDQN
        elif name in ["dqn", "dqn_default_config"]:
            from skrl.agents.jax.dqn import DQN, DQN_DEFAULT_CONFIG

            component = DQN_DEFAULT_CONFIG if "default_config" in name else DQN
        elif name in ["ppo", "ppo_default_config"]:
            from skrl.agents.jax.ppo import PPO, PPO_DEFAULT_CONFIG

            component = PPO_DEFAULT_CONFIG if "default_config" in name else PPO
        elif name in ["rpo", "rpo_default_config"]:
            from skrl.agents.jax.rpo import RPO, RPO_DEFAULT_CONFIG

            component = RPO_DEFAULT_CONFIG if "default_config" in name else RPO
        elif name in ["sac", "sac_default_config"]:
            from skrl.agents.jax.sac import SAC, SAC_DEFAULT_CONFIG

            component = SAC_DEFAULT_CONFIG if "default_config" in name else SAC
        elif name in ["td3", "td3_default_config"]:
            from skrl.agents.jax.td3 import TD3, TD3_DEFAULT_CONFIG

            component = TD3_DEFAULT_CONFIG if "default_config" in name else TD3
        elif name in ["PID", "pid_default_config"]:
            from isaaclab_dynamics.rl.control_agent import CONTROLLER_DEFAULT_CONFIG, PIDAgent

            component = CONTROLLER_DEFAULT_CONFIG if "default_config" in name else PIDAgent

        # multi-agent
        elif name in ["ippo", "ippo_default_config"]:
            from skrl.multi_agents.jax.ippo import IPPO, IPPO_DEFAULT_CONFIG

            component = IPPO_DEFAULT_CONFIG if "default_config" in name else IPPO
        elif name in ["mappo", "mappo_default_config"]:
            from skrl.multi_agents.jax.mappo import MAPPO, MAPPO_DEFAULT_CONFIG

            component = MAPPO_DEFAULT_CONFIG if "default_config" in name else MAPPO

        # trainer
        elif name == "sequentialtrainer":
            from skrl.trainers.jax import SequentialTrainer as component
        elif name == "exposedtrainer":
            from isaaclab_dynamics.simulation.manager import ExposedTrainer as component

        if component is None:
            raise ValueError(f"Unknown component '{name}' in runner cfg")
        return component
