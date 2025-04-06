# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pynput.keyboard import Key, Listener


class Controller:
    """
    A controller class for managing simulation or robot control logic.

    Methods:
        setup(): Initializes or configures the controller settings.
        step(): Executes the control logic at each step.
    """

    def __init__(self, config=None):
        """Initialize class variables or parameters for the controller."""
        self.is_initialized = False  # Indicates whether the controller is set up
        self.control_state = {}  # Stores control-related state variables
        self.config = config or {}  # Use the provided configuration or set to an empty dict
        self.is_initialized = True
        self.key_status = {
            "left": False,
            "right": False,
            "up": False,
            "down": False,
        }

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

        # Start the listener in the background
        listener = Listener(on_press=on_press, on_release=on_release)
        listener.start()  # This starts the listener in a separate thread

        print("Controller initialized and setup complete with configuration:", self.config)

    def step(self, inputs=None):
        """
        Executes control logic for a single step.

        Args:
            inputs (dict): Input data required for the control logic (e.g., sensor readings, state).

        Returns:
            dict: Outputs or actions based on the control logic.
        """
        if not self.is_initialized:
            raise RuntimeError("Controller is not set up. Call the 'setup()' method before stepping.")

        # Example logic for handling inputs and returning an action
        inputs = inputs or {}
        print("Processing inputs:", inputs)

        # Example action logic (override with your control algorithm)
        action = -1 if self.key_status["left"] else 1 if self.key_status["right"] else 0
        self.control_state.update(inputs)  # Update control state based on inputs

        print("Control action produced:", action)
        return action
