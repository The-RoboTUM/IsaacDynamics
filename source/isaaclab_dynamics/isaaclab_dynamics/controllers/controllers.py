# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import time
import torch

# Import necessary modules
from abc import ABC, abstractmethod

import skrl
from packaging import version
from pynput.keyboard import Key, Listener

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


# Base class for controllers
class ControllerBase(ABC):
    """
    Abstract base class for controllers. Provides a standard interface
    for setup and execution, as well as shared setup behavior.
    """

    def __init__(self, args_cli=None):
        # Store command line arguments, if provided
        self.args_cli = args_cli or {}
        self.is_initialized = False

    def setup(self, env, dt, config=None, seed=None):
        """
        Perform common setup tasks for the controller.

        :param env: Environment instance
        :param config: Configuration dictionary (optional)
        :param seed: Seed value for reproducibility (optional)
        """
        self.is_initialized = True
        self.dt = dt

    def run(self, env, alive_check, args=None):
        obs, _ = env.reset()
        timestep = 0

        # Main simulation loop
        while alive_check():
            start_time = time.time()
            with torch.inference_mode():

                # Take a step in the environment with the computed action
                actions = self.step(obs, args=args)
                obs, _, _, _, _ = env.step(actions)

            # Handle video-related settings
            if self.args_cli.video:
                timestep += 1
                if timestep == self.args_cli.video_length:
                    break

            # Sleep for real-time simulation if needed
            sleep_time = self.dt - (time.time() - start_time)
            if self.args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

    @abstractmethod
    def step(self, obs, args=None):
        pass


# Controller class for RL-based algorithms
class ControllerRL(ControllerBase):
    """
    Implementation of RL-based controller. Includes setup and runtime logic
    for training and testing reinforcement learning agents.
    """

    def __init__(self, args_cli=None):
        super().__init__(args_cli)
        # Store algorithm name and initialize agent configuration
        self.algorithm = args_cli.algorithm.lower()
        self.agent_cfg = None

    def setup(self, env, dt, config=None, seed=None):
        """
        Sets up the given environment, configuration parameters, and associated settings
        for training or testing. Configures the underlying Machine Learning framework,
        agent parameters, and environment handling based on the provided mode
        ('train' or 'test'). Also adjusts agent-specific configuration
        depending on the runtime parameters.

        Args:
            env: The original environment to be wrapped and configured.
            dt: The time step or time interval for the simulation or environment updates.
            config: Optional configuration dictionary containing parameters for the
                agent, trainer, and experiment setup. Defaults to None.
            seed: An optional value to set the random seed for reproducibility. If not
                provided, the seed is fetched from the provided configuration.

        Returns:
            tuple: A tuple containing the wrapped and configured environment and the
                final agent configuration dictionary.
        """
        super().setup(env, dt, config, seed)
        self.agent_cfg = config

        # Configure ML framework (e.g., JAX or NumPy backend for jax framework)
        if self.args_cli.ml_framework.startswith("jax"):
            skrl.config.jax.backend = "jax" if self.args_cli.ml_framework == "jax" else "numpy"

        # Adjust agent configuration depending on mode (train/test)
        if self.args_cli.mode == "train":
            # Configure training-specific parameters
            if self.args_cli.max_iterations:
                self.agent_cfg["trainer"]["timesteps"] = self.args_cli.max_iterations * config["agent"]["rollouts"]
            self.agent_cfg["trainer"]["close_environment_at_exit"] = False
            self.agent_cfg["seed"] = seed if seed is not None else config["seed"]
        elif self.args_cli.mode == "test":
            # Configure testing-specific parameters
            config["trainer"]["close_environment_at_exit"] = False
            config["agent"]["experiment"]["write_interval"] = 0
            config["agent"]["experiment"]["checkpoint_interval"] = 0

        # Wrap environment based on its type and algorithm
        if isinstance(env.unwrapped, DirectMARLEnv) and self.algorithm in ["ppo"]:
            env = multi_agent_to_single_agent(env)
        env = SkrlVecEnvWrapper(env, ml_framework=self.args_cli.ml_framework)
        return env, self.agent_cfg

    def run(self, env, alive_check, args=None, resume_path=None):
        """
        Run the RL controller in either train or test mode.

        :param env: Environment instance
        :param dt: Timestep duration
        :param alive_check: Function to check if the process should continue
        :param resume_path: Path to load a saved agent checkpoint (optional)
        """
        # Import the appropriate Runner class dynamically
        Runner = None
        if self.args_cli.ml_framework.startswith("torch"):
            from skrl.utils.runner.torch import Runner
        elif self.args_cli.ml_framework.startswith("jax"):
            from skrl.utils.runner.jax import Runner

        # Initialize the runner with the environment and agent configuration
        runner = Runner(env, self.agent_cfg)

        # Optionally load a pre-trained model checkpoint
        if resume_path:
            print(f"[INFO] Loading model checkpoint from: {resume_path}")
            runner.agent.load(resume_path)

        # Run the controller based on the mode (train or test)
        if self.args_cli.mode == "train":
            runner.run()
        elif self.args_cli.mode == "test":
            runner.agent.set_running_mode("eval")
            super().run(env, alive_check, args={"runner": runner, "env": env})

    def step(self, obs, args=None):
        runner = args.get("runner")
        env = args.get("env")

        outputs = runner.agent.act(obs, timestep=0, timesteps=0)
        if hasattr(env, "possible_agents"):
            actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
        else:
            actions = outputs[-1].get("mean_actions", outputs[0])

        return actions


# Controller class for keyboard-based interaction
class ControllerKeyboard(ControllerBase):
    """
    Implementation of a keyboard-based controller. This controller
    listens for keyboard events and maps them to environment actions.
    """

    def __init__(self, args_cli=None):
        """
        Initialize class variables, such as the key status map and listeners.
        """
        super().__init__(args_cli)
        self.key_status = {
            "left": False,
            "right": False,
            "up": False,
            "down": False,
        }

        # Define keyboard listeners for key press and release
        def on_press(key):
            if key == Key.up:
                self.key_status["up"] = True
            elif key == Key.down:
                self.key_status["down"] = True
            elif key == Key.left:
                self.key_status["left"] = True
            elif key == Key.right:
                self.key_status["right"] = True

        def on_release(key):
            if key == Key.up:
                self.key_status["up"] = False
            elif key == Key.down:
                self.key_status["down"] = False
            elif key == Key.left:
                self.key_status["left"] = False
            elif key == Key.right:
                self.key_status["right"] = False

        # Start the listener in a background thread
        listener = Listener(on_press=on_press, on_release=on_release)
        listener.start()

    def setup(self, env, dt, config=None, seed=None):
        """
        Perform setup tasks for the keyboard controller.

        :param env: Environment instance
        :param config: Configuration dictionary (optional)
        :param seed: (Unused) Seed value
        """
        super().setup(env, dt, config, seed)
        return env, config

    def run(self, env, alive_check, args=None, resume_path=None):
        """
        Execute the keyboard controller's main loop.

        :param env: Environment instance
        :param dt: Timestep duration
        :param alive_check: Function to check if the process should continue
        :param resume_path: Unused for keyboard controller
        """
        super().run(env, alive_check, args=args)

    def step(self, obs, args=None):
        """
        Process input state and compute actions based on key presses.

        :param inputs: (Optional) Input values for additional logic
        :return: Action value based on keyboard status
        """
        if not self.is_initialized:
            raise RuntimeError("Controller is not set up. Call the 'setup()' method before stepping.")
        # Example action selection based on keyboard press
        action = 1 if self.key_status["left"] else -1 if self.key_status["right"] else 0
        return action


class ControllerPID(ControllerBase):
    """
    Implementation of a PID controller for controlling environments
    where a proportional-integral-derivative control system is suitable.
    """

    def __init__(self, args_cli=None):
        """
        Initialize the PID controller parameters and state variables.

        :param args_cli: Command line arguments or configuration dictionary
        """
        super().__init__(args_cli)
        self.Kp = None  # Proportional gain
        self.Ki = None  # Integral gain
        self.Kd = None  # Derivative gain
        self.dt = 0

        self.setpoint = None  # Desired target value
        self.integral = 0.0  # Integral term accumulator
        self.prev_error = None  # Previous error term for derivative calculation

    def setup(self, env, dt, config=None, seed=None):
        """
        Setup the PID controller.

        :param env: The environment instance
        :param config: Configuration dictionary (optional)
        :param seed: Seed value for reproducibility (optional)
        """
        super().setup(env, dt, config, seed)
        self.Kp = 1.0  # Proportional gain
        self.Ki = 0.01  # Integral gain
        self.Kd = 0.5  # Derivative gain
        self.setpoint = np.pi / 2  # Desired target value
        return env, config

    def run(self, env, alive_check, resume_path=None):
        """
        Execute the PID controller's control loop.

        :param env: The environment instance
        :param dt: Timestep duration
        :param alive_check: Function to check whether the process should continue
        """
        super().run(env, alive_check)

    def step(self, obs, args=None):
        # Get the current value
        current_value = obs["policy"][0, 0].item()

        # Calculate the current error
        error = self.setpoint - current_value

        # Compute the integral term
        self.integral += error * self.dt

        # Compute the derivative term
        derivative = (error - self.prev_error) / self.dt if self.prev_error is not None else 0.0

        # Compute the control action
        joint_efforts = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        # Update the previous error
        self.prev_error = error

        return joint_efforts
