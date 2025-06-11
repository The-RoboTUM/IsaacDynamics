# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Mapping
from typing import Any

import skrl
from packaging import version
from skrl import logger
from skrl.agents.jax import Agent
from skrl.envs.wrappers.jax import MultiAgentEnvWrapper, Wrapper
from skrl.models.jax import Model
from skrl.utils.runner.jax.runner import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401

# Validate skrl version
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()


class IsaacRunnerWrapper:
    def __init__(self, args_cli=None):
        self.args_cli = args_cli or {}
        self.is_initialized = False
        self.step_runtime = 0
        self.episode_count = 0
        self.dt = 0

        self.algorithm = self.args_cli.algorithm.lower()
        self.agent_cfg = None

    def setup(self, env, dt, config=None, seed=None):
        self.is_initialized = True
        self.dt = dt
        self.agent_cfg = config

        if self.args_cli.ml_framework.startswith("jax"):
            skrl.config.jax.backend = "jax" if self.args_cli.ml_framework == "jax" else "numpy"

        if self.args_cli.mode == "train":
            if self.args_cli.max_iterations:
                self.agent_cfg["trainer"]["timesteps"] = self.args_cli.max_iterations * config["agent"]["rollouts"]
            self.agent_cfg["trainer"]["close_environment_at_exit"] = False
            self.agent_cfg["seed"] = seed if seed is not None else config["seed"]
        elif self.args_cli.mode == "test":
            config["trainer"]["close_environment_at_exit"] = False
            config["agent"]["experiment"]["write_interval"] = 0
            config["agent"]["experiment"]["checkpoint_interval"] = 0

        if isinstance(env.unwrapped, DirectMARLEnv) and self.algorithm in ["ppo"]:
            env = multi_agent_to_single_agent(env)
        env = SkrlVecEnvWrapper(env, ml_framework=self.args_cli.ml_framework)
        return env, self.agent_cfg

    def run(self, env, args=None, resume_path=None):
        Runner = None
        if self.args_cli.ml_framework.startswith("torch"):
            from skrl.utils.runner.torch import Runner
        elif self.args_cli.ml_framework.startswith("jax"):
            from isaaclab_dynamics.sim.skrl_link import ExpandedRunner as Runner

        self.agent_cfg["trainer"]["record"] = self.args_cli.record
        self.agent_cfg["trainer"]["log_dir"] = self.args_cli.log_dir
        runner = Runner(env, cfg=self.agent_cfg)

        if resume_path:
            print(f"[INFO] Loading model checkpoint from: {resume_path}")
            runner.agent.load(resume_path)

        if self.args_cli.mode == "train":
            runner.run(mode="train")
        elif self.args_cli.mode == "test":
            runner.run(mode="eval")


# Controller class for keyboard-based interaction
# class ControllerKeyboard(ControllerBase):
#     """
#     Represents a keyboard input controller that listens for key presses to control
#     actions. This class provides functionality to set up and manage key-based input
#     for interactive environments, allowing for dynamic control during runtime.
#
#     The class initializes a key-status map to track the current state of directional
#     keys (left, right, up, down). It uses a background thread to handle keyboard
#     input asynchronously via a listener. The class also provides integration with
#     an environment setup and step-by-step processing of actions based on key inputs.
#
#     Attributes:
#         key_status (dict): A map containing the status of directional keys, with
#             boolean values indicating whether a key is pressed (True) or released
#             (False).
#     """
#
#     def __init__(self, args_cli=None):
#         """
#         Initialize class variables, such as the key status map and listeners.
#         """
#         super().__init__(args_cli)
#         self.key_status = {
#             "left": False,
#             "right": False,
#             "up": False,
#             "down": False,
#         }
#
#         # Define keyboard listeners for key press and release
#         def on_press(key):
#             if key == Key.up:
#                 self.key_status["up"] = True
#             elif key == Key.down:
#                 self.key_status["down"] = True
#             elif key == Key.left:
#                 self.key_status["left"] = True
#             elif key == Key.right:
#                 self.key_status["right"] = True
#
#         def on_release(key):
#             if key == Key.up:
#                 self.key_status["up"] = False
#             elif key == Key.down:
#                 self.key_status["down"] = False
#             elif key == Key.left:
#                 self.key_status["left"] = False
#             elif key == Key.right:
#                 self.key_status["right"] = False
#
#         # Start the listener in a background thread
#         listener = Listener(on_press=on_press, on_release=on_release)
#         listener.start()
#
#     def setup(self, env, dt, config=None, seed=None):
#         super().setup(env, dt, config, seed)
#         return env, config
#
#     def run(self, env, alive_check, iterate, args=None, resume_path=None):
#         super().run(env, alive_check, iterate, args=args, resume_path=resume_path)
#
#     def step(self, obs, args=None):
#         if not self.is_initialized:
#             raise RuntimeError(
#                 "Controller is not set up. Call the 'setup()' method before stepping."
#             )
#         # Example action selection based on keyboard press
#         action = 1 if self.key_status["left"] else -1 if self.key_status["right"] else 0
#         return torch.tensor([[action]])


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
        elif name in ["pid", "pid_default_config"]:
            from isaaclab_dynamics.agents.base_agents import CONTROLLER_DEFAULT_CONFIG, PIDAgent

            component = CONTROLLER_DEFAULT_CONFIG if "default_config" in name else PIDAgent
        elif name in ["random", "random_default_config"]:
            from isaaclab_dynamics.agents.base_agents import CONTROLLER_DEFAULT_CONFIG, RandomAgent

            component = CONTROLLER_DEFAULT_CONFIG if "default_config" in name else RandomAgent

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
            from isaaclab_dynamics.sim.trainers import ExposedTrainer as component

        if component is None:
            raise ValueError(f"Unknown component '{name}' in runner cfg")
        return component

    def _generate_agent(
        self,
        env: Wrapper | MultiAgentEnvWrapper,
        cfg: Mapping[str, Any],
        models: Mapping[str, Mapping[str, Model]],
    ) -> Agent:
        """Generate agent instance according to the environment specification and the given config and models

        :param env: Wrapped environment
        :param cfg: A configuration dictionary
        :param models: Agent's model instances

        :return: Agent instances
        """
        multi_agent = isinstance(env, MultiAgentEnvWrapper)
        device = env.device
        num_envs = env.num_envs
        possible_agents = env.possible_agents if multi_agent else ["agent"]
        state_spaces = env.state_spaces if multi_agent else {"agent": env.state_space}
        observation_spaces = env.observation_spaces if multi_agent else {"agent": env.observation_space}
        action_spaces = env.action_spaces if multi_agent else {"agent": env.action_space}

        agent_class = cfg.get("agent", {}).get("class", "").lower()
        if not agent_class:
            raise ValueError("No 'class' field defined in 'agent' cfg")

        # check for memory configuration (backward compatibility)
        if "memory" not in cfg:
            logger.warning(
                "Deprecation warning: No 'memory' field defined in cfg. Using the default generated configuration"
            )
            cfg["memory"] = {"class": "RandomMemory", "memory_size": -1}
        # get memory class and remove 'class' field
        try:
            memory_class = self._component(cfg["memory"]["class"])
            del cfg["memory"]["class"]
        except KeyError:
            memory_class = self._component("RandomMemory")
            logger.warning("No 'class' field defined in 'memory' cfg. 'RandomMemory' will be used as default")
        memories = {}
        # instantiate memory
        if cfg["memory"]["memory_size"] < 0:
            cfg["memory"]["memory_size"] = cfg["agent"]["rollouts"]  # memory_size is the agent's number of rollouts
        for agent_id in possible_agents:
            memories[agent_id] = memory_class(num_envs=num_envs, device=device, **self._process_cfg(cfg["memory"]))

        # single-agent configuration and instantiation
        if agent_class in [
            "a2c",
            "cem",
            "ddpg",
            "ddqn",
            "dqn",
            "ppo",
            "rpo",
            "sac",
            "td3",
        ]:
            agent_id = possible_agents[0]
            agent_cfg = self._component(f"{agent_class}_DEFAULT_CONFIG").copy()
            agent_cfg.update(self._process_cfg(cfg["agent"]))
            agent_cfg.get("state_preprocessor_kwargs", {}).update(
                {"size": observation_spaces[agent_id], "device": device}
            )
            agent_cfg.get("value_preprocessor_kwargs", {}).update({"size": 1, "device": device})
            if agent_cfg.get("exploration", {}).get("noise"):
                agent_cfg["exploration"]["noise"] = agent_cfg["exploration"]["noise"](
                    **agent_cfg["exploration"].get("noise_kwargs", {})
                )
            if agent_cfg.get("smooth_regularization_noise"):
                agent_cfg["smooth_regularization_noise"] = agent_cfg["smooth_regularization_noise"](
                    **agent_cfg.get("smooth_regularization_noise_kwargs", {})
                )
            agent_kwargs = {
                "models": models[agent_id],
                "memory": memories[agent_id],
                "observation_space": observation_spaces[agent_id],
                "action_space": action_spaces[agent_id],
            }
        # multi-agent configuration and instantiation
        elif agent_class in ["ippo"]:
            agent_cfg = self._component(f"{agent_class}_DEFAULT_CONFIG").copy()
            agent_cfg.update(self._process_cfg(cfg["agent"]))
            agent_cfg["state_preprocessor_kwargs"].update(
                {agent_id: {"size": observation_spaces[agent_id], "device": device} for agent_id in possible_agents}
            )
            agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": device})
            agent_kwargs = {
                "models": models,
                "memories": memories,
                "observation_spaces": observation_spaces,
                "action_spaces": action_spaces,
                "possible_agents": possible_agents,
            }
        elif agent_class in ["mappo"]:
            agent_cfg = self._component(f"{agent_class}_DEFAULT_CONFIG").copy()
            agent_cfg.update(self._process_cfg(cfg["agent"]))
            agent_cfg["state_preprocessor_kwargs"].update(
                {agent_id: {"size": observation_spaces[agent_id], "device": device} for agent_id in possible_agents}
            )
            agent_cfg["shared_state_preprocessor_kwargs"].update(
                {agent_id: {"size": state_spaces[agent_id], "device": device} for agent_id in possible_agents}
            )
            agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": device})
            agent_kwargs = {
                "models": models,
                "memories": memories,
                "observation_spaces": observation_spaces,
                "action_spaces": action_spaces,
                "shared_observation_spaces": state_spaces,
                "possible_agents": possible_agents,
            }
        else:
            agent_id = possible_agents[0]
            agent_cfg = self._component(f"{agent_class}_DEFAULT_CONFIG").copy()

            if cfg.get("range") is not None:
                agent_cfg["range"] = cfg["range"]

            agent_cfg.update(self._process_cfg(cfg["agent"]))
            agent_cfg.get("state_preprocessor_kwargs", {}).update(
                {"size": observation_spaces[agent_id], "device": device}
            )
            agent_cfg.get("value_preprocessor_kwargs", {}).update({"size": 1, "device": device})
            if agent_cfg.get("exploration", {}).get("noise"):
                agent_cfg["exploration"]["noise"] = agent_cfg["exploration"]["noise"](
                    **agent_cfg["exploration"].get("noise_kwargs", {})
                )
            if agent_cfg.get("smooth_regularization_noise"):
                agent_cfg["smooth_regularization_noise"] = agent_cfg["smooth_regularization_noise"](
                    **agent_cfg.get("smooth_regularization_noise_kwargs", {})
                )
            agent_kwargs = {
                "models": models[agent_id],
                "memory": memories[agent_id],
                "observation_space": observation_spaces[agent_id],
                "action_space": action_spaces[agent_id],
            }
        return self._component(agent_class)(cfg=agent_cfg, device=device, **agent_kwargs)
