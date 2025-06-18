# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium
import numpy as np
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from skrl.agents.jax import Agent
from skrl.memories.jax import Memory
from skrl.models.jax import Model

# from skrl.resources.optimizers.jax import Adam

CONTROLLER_DEFAULT_CONFIG = {
    # ...
    "experiment": {
        "directory": "",  # experiment's parent directory
        "experiment_name": "",  # experiment name
        "write_interval": "auto",  # TensorBoard writing interval (timesteps)
        "checkpoint_interval": "auto",  # interval for checkpoints (timesteps)
        "store_separately": False,  # whether to store checkpoints separately
        "wandb": False,  # whether to use Weights & Biases
        "wandb_kwargs": {},  # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}


class BasicAgent(Agent):
    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Memory | tuple[Memory] | None = None,
        observation_space: int | tuple[int] | gymnasium.Space | None = None,
        action_space: int | tuple[int] | gymnasium.Space | None = None,
        device: str | jax.Device | None = None,
        cfg: dict | None = None,
    ) -> None:
        """Custom agent

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.jax.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.jax.Memory, list of skrl.memory.jax.Memory or None
        :param observation_space: Observation/state space or shape (default: None)
        :type observation_space: int, tuple or list of integers, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: None)
        :type action_space: int, tuple or list of integers, gymnasium.Space or None, optional
        :param device: Device on which a jax array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda:0"`` if available or ``"cpu"``
        :type device: str or jaxlib.xla_extension.Device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict
        """
        _cfg = CONTROLLER_DEFAULT_CONFIG
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )
        # =======================================================================
        # - get and process models from `self.models`
        # - populate `self.checkpoint_modules` dictionary for storing checkpoints
        # - parse configurations from `self.cfg`
        # - setup optimizers and learning rate scheduler
        # - set up preprocessors
        # =======================================================================

    def init(self, trainer_cfg: dict[str, Any] | None = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")
        # =================================================================
        # - create tensors in memory if required
        # - # create temporary variables needed for storage and computation
        # - set up models for just-in-time compilation with XLA
        # =================================================================

    def act(self, states: jnp.ndarray, timestep: int, timesteps: int) -> jnp.ndarray:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: jnp.ndarray
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: jnp.ndarray
        """
        # ======================================
        # - sample random actions if required or
        #   sample and return agent's actions
        # ======================================

    def record_transition(
        self,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        next_states: jnp.ndarray,
        terminated: jnp.ndarray,
        truncated: jnp.ndarray,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: jnp.ndarray
        :param actions: Actions taken by the agent
        :type actions: jnp.ndarray
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: jnp.ndarray
        :param next_states: Next observations/states of the environment
        :type next_states: jnp.ndarray
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: jnp.ndarray
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: jnp.ndarray
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super().record_transition(
            states,
            actions,
            rewards,
            next_states,
            terminated,
            truncated,
            infos,
            timestep,
            timesteps,
        )
        # ========================================
        # - record agent's specific data in memory
        # ========================================

    # def pre_interaction(self, timestep: int, timesteps: int) -> None:
    #     """Callback called before the interaction with the environment
    #
    #     :param timestep: Current timestep
    #     :type timestep: int
    #     :param timesteps: Number of timesteps
    #     :type timesteps: int
    #     """
    #     # =====================================
    #     # - call `self.update(...)` if required
    #     # =====================================

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # =====================================
        # - call `self.update(...)` if required
        # =====================================
        # call parent's method for checkpointing and TensorBoard writing
        super().post_interaction(timestep, timesteps)

    # def _update(self, timestep: int, timesteps: int) -> None:
    #     """Algorithm's main update step
    #
    #     :param timestep: Current timestep
    #     :type timestep: int
    #     :param timesteps: Number of timesteps
    #     :type timesteps: int
    #     """
    #     # ===================================================
    #     # - implement algorithm's update step
    #     # - record tracking data using `self.track_data(...)`
    #     # ===================================================


class PIDAgent(BasicAgent):
    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Memory | tuple[Memory] | None = None,
        observation_space: int | tuple[int] | gymnasium.Space | None = None,
        action_space: int | tuple[int] | gymnasium.Space | None = None,
        device: str | jax.Device | None = None,
        cfg: dict | None = None,
        kp: float = 1.0,
        ki: float = 0.01,
        kd: float = 0.5,
    ) -> None:
        """
        PID-based Agent

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.jax.Model
        :param observation_space: Observation/state space or shape (default: None)
        :type observation_space: int, tuple or list of integers, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: None)
        :type action_space: int, tuple or list of integers, gymnasium.Space or None, optional
        :param device: Device on which a jax array is or will be allocated (default: ``None``)
        :type device: str or jaxlib.xla_extension.Device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict
        :param kp: Proportional gain
        :type kp: float
        :param ki: Integral gain
        :type ki: float
        :param kd: Derivative gain
        :type kd: float
        """
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=cfg,
        )
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0.0
        self.integral = 0.0
        self.dt = cfg.get("dt", 1 / 120)

    def act(self, states: jnp.ndarray, timestep: int, timesteps: int) -> jnp.ndarray:
        """
        Process the environment's states to compute actions using the PID controller.

        :param states: Environment's states
        :type states: jnp.ndarray
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: jnp.ndarray
        """
        setpoint = jnp.pi / 2  # Desired target value
        error = setpoint - jnp.array(states).ravel()[0]  # Assuming 1D input state
        self.integral += error * self.dt
        derivative = (error - self.previous_error) / self.dt if self.previous_error is not None else 0.0

        action = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error

        return jnp.array([action])


class RandomAgent(BasicAgent):
    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Memory | tuple[Memory] | None = None,
        observation_space: int | tuple[int] | gymnasium.Space | None = None,
        action_space: int | tuple[int] | gymnasium.Space | None = None,
        device: str | jax.Device | None = None,
        cfg: dict | None = None,
    ) -> None:
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=cfg,
        )
        self.effort_range = (
            -cfg.get("range", 5.0),
            cfg.get("range", 5.0),
        )  # Default random effort range
        self.dt = cfg.get("dt", 1 / 120)

    def act(self, states: jnp.ndarray, timestep: int, timesteps: int) -> jnp.ndarray:
        random_effort = np.random.uniform(self.effort_range[0], self.effort_range[1])
        return jnp.array([random_effort])


class DummyAgent(BasicAgent):
    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Memory | tuple[Memory] | None = None,
        observation_space: int | tuple[int] | gymnasium.Space | None = None,
        action_space: int | tuple[int] | gymnasium.Space | None = None,
        device: str | jax.Device | None = None,
        cfg: dict | None = None,
    ) -> None:
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=cfg,
        )
        self.dt = cfg.get("dt", 1 / 120)

    def act(self, states: jnp.ndarray, timestep: int, timesteps: int) -> jnp.ndarray:
        return jnp.array([0.0])
