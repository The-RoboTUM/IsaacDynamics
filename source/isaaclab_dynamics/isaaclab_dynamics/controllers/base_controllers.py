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
    Defines an abstract base class for controllers in simulation environments.

    This class serves as a base for building specific controllers, providing common
    setup, initialization, and execution functionalities. Subclasses must
    implement the `step` method to define specific behavior for each controller.

    Attributes:
        args_cli (dict): Dictionary containing command-line arguments provided
            to the controller.
        is_initialized (bool): Indicates whether the controller has been
            initialized and set up.
        dt (float): Time step duration for the simulation.
    """

    def __init__(self, args_cli=None):
        # Store command line arguments, if provided
        self.args_cli = args_cli or {}
        self.is_initialized = False
        self.step_runtime = 0
        self.episode_count = 0
        self.dt = 0

    def setup(self, env, dt, config=None, seed=None):
        self.is_initialized = True
        self.dt = dt

    def reset(self, env):
        obs, _ = env.reset()
        return obs

    def run(self, env, alive_check, iterate, args=None, resume_path=None):
        # Set up the simulation
        obs = self.reset(env)
        timestep = 0

        # Main simulation loop
        self.loop(env, alive_check, iterate, obs, timestep, args=args, resume_path=resume_path)

    def loop(self, env, alive_check, iterate, obs, timestep, args=None, resume_path=None):
        while alive_check():
            start_time = time.time()
            with torch.inference_mode():

                # Take a step in the environment with the computed action
                actions, obs, _, _, truncated, _ = iterate(env, obs, args=args)

            # Handle video-related settings
            if self.args_cli.video:
                timestep += 1
                if timestep == self.args_cli.video_length:
                    break

            # Sleep for real-time simulation if needed
            sleep_time = self.dt - (time.time() - start_time)
            if self.args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

            # Count episode terminations
            if truncated:
                self.episode_count += 1

            # Handle experiment termination
            if self.episode_count >= self.args_cli.max_episodes:
                break

    @abstractmethod
    def step(self, obs, args=None):
        pass

    def iterate(self, env, obs, args=None):
        actions = self.step(obs, args=args)
        obs, reward, terminated, truncated, info = env.step(actions)
        self.step_runtime += 1
        return actions, obs, reward, terminated, truncated, info


# Controller class for RL-based algorithms
class ControllerRL(ControllerBase):
    """
    A Reinforcement Learning (RL) Controller class that manages the setup, configuration,
    and execution of reinforcement learning algorithms in different environments. It
    facilitates compatibility with multiple machine learning frameworks (e.g., PyTorch,
    JAX), supports both training and testing modes, and allows seamless handling of
    multi-agent and single-agent environments.

    This class serves as an interface for defining the pipeline of RL-based controllers,
    including environment setup, algorithm-specific configurations, and running of
    agents.

    Attributes:
        algorithm (str): The name of the algorithm to be used, set based on command-line
            arguments and converted to lowercase.
        agent_cfg (dict): Configuration dictionary for the RL agent. It stores agent-
            specific parameters such as trainer settings and runtime configurations.
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

    def run(self, env, alive_check, iterate, args=None, resume_path=None):
        """
        Runs the main process for setting up and executing the environment and agent interaction.

        This method dynamically imports the appropriate Runner class based on the machine learning
        framework in use, initializes the runner, optionally loads a pre-trained model checkpoint,
        and executes the controller in either training or testing mode.

        Args:
            env: The environment instance in which the agent will operate.
            alive_check: A function or mechanism used to determine if the process is alive and should
                continue running.
            args: Optional; Additional arguments that might be required for running the process.
                Defaults to None.
            resume_path: Optional; The file path to a pre-trained model checkpoint that can be loaded
                for resuming the process. Defaults to None.
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
            super().run(env, alive_check, iterate, args={"runner": runner, "env": env})

    def step(self, obs, args=None):
        """
        Executes a single step in the environment using the provided observations and additional
        arguments. This method retrieves actions from the associated agent based on the observations
        and returns the computed actions. The implementation supports environments with multiple
        possible agents, where actions are generated for each agent individually.

        Args:
            obs: Observations from the environment, provided to the agent to compute subsequent
                actions.
            args: Optional dictionary containing additional arguments. It is expected to include
                'runner' (agent manager) and 'env' (environment object).

        Returns:
            dict: A dictionary where the keys are agent identifiers (for environments with
                multiple agents) or a single key for non-multi-agent environments.
                The values are the corresponding actions computed by the agent.
        """
        runner = args.get("runner")
        env = args.get("env")

        outputs = runner.agent.act(obs, timestep=0, timesteps=0)  # look into why this has timestep=0
        if hasattr(env, "possible_agents"):
            actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
        else:
            actions = outputs[-1].get("mean_actions", outputs[0])

        return actions


# Controller class for keyboard-based interaction
class ControllerKeyboard(ControllerBase):
    """
    Represents a keyboard input controller that listens for key presses to control
    actions. This class provides functionality to set up and manage key-based input
    for interactive environments, allowing for dynamic control during runtime.

    The class initializes a key-status map to track the current state of directional
    keys (left, right, up, down). It uses a background thread to handle keyboard
    input asynchronously via a listener. The class also provides integration with
    an environment setup and step-by-step processing of actions based on key inputs.

    Attributes:
        key_status (dict): A map containing the status of directional keys, with
            boolean values indicating whether a key is pressed (True) or released
            (False).
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
        super().setup(env, dt, config, seed)
        return env, config

    def run(self, env, alive_check, iterate, args=None, resume_path=None):
        super().run(env, alive_check, iterate, args=args, resume_path=resume_path)

    def step(self, obs, args=None):
        if not self.is_initialized:
            raise RuntimeError("Controller is not set up. Call the 'setup()' method before stepping.")
        # Example action selection based on keyboard press
        action = 1 if self.key_status["left"] else -1 if self.key_status["right"] else 0
        return torch.tensor([[action]])


class ControllerPID(ControllerBase):
    """
    ControllerPID implements a Proportional-Integral-Derivative (PID) control
    strategy for managing system behavior by minimizing the difference
    (error) between a setpoint and a measured process variable.

    The class provides functionality to initialize the PID parameters,
    accommodate custom environment-specific configurations, and perform
    control actions. This makes it suitable for closed-loop feedback systems
    where precise control and adjustments are required.

    Attributes:
        Kp (float): Proportional gain, influences the magnitude of the control
            action proportional to the error.
        Ki (float): Integral gain, accumulates past error to eliminate steady-state
            error.
        Kd (float): Derivative gain, reacts to the rate of change of the error.
        dt (float): Time step or duration between control updates.
        setpoint (float): Desired target value that the controller attempts to reach.
        integral (float): Accumulator for the integral term.
        prev_error (float): Error value from the previous control step, used for
            derivative computation.
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

    def run(self, env, alive_check, iterate, args=None, resume_path=None):
        """
        Execute the PID controller's control loop.

        :param env: The environment instance
        :param dt: Timestep duration
        :param alive_check: Function to check whether the process should continue
        """
        super().run(env, alive_check, iterate, args=args, resume_path=resume_path)

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

        return torch.tensor([[joint_efforts]])


class ControllerRandom(ControllerBase):
    """
    ControllerRandom generates random efforts for each step. Useful for testing
    or exploratory purposes in simulation environments.

    Attributes:
        effort_range (tuple): A tuple defining the (min, max) range for random efforts.
    """

    def __init__(self, args_cli=None):
        """
        Initializes the random controller.

        :param args_cli: Command line arguments or configuration dictionary.
        """
        super().__init__(args_cli)
        self.effort_range = (-5.0, 5.0)  # Default random effort range

    def setup(self, env, dt, config=None, seed=None):
        """
        Configure the random controller with optional environment, timestep, and seed.

        :param env: Environment instance.
        :param dt: Time step duration.
        :param config: Configuration dictionary.
        :param seed: Optional random seed.
        """
        super().setup(env, dt, config, seed)
        if config and "effort_range" in config:
            self.effort_range = config["effort_range"]
        if seed is not None:
            np.random.seed(seed)
        return env, config

    def step(self, obs, args=None):
        """
        Generate random efforts as actions.

        :param obs: Observations from the environment (not used here).
        :param args: Optional arguments.
        :return: A tensor with random efforts.
        """
        random_effort = np.random.uniform(self.effort_range[0], self.effort_range[1])
        return torch.tensor([[random_effort]])
