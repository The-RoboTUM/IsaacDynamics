# IsaacDynamics

This project is the first implementation of a co-design framework for the discovery of mechanisms and controllers
through isaac lab.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Params](#params)
- [Information](#information)
- [Example Commands](#example-commands)
- [Useful Commands](#useful-commands)
- [Known Issues](#known-issues)

## Installation

Instructions for installation.

TODO: Add the instructions from Isaac lab here

Example:

```bash
sudo apt install cmake build-essential
./isaaclab.sh --install # or "./isaaclab.sh -i"
```

If you use the jax framework make sure to run the following to get all the dependencies:
```bash
pip install --upgrade "jax[cuda12]"
pip install skrl["all"]
```

## Usage

To train an environments:
```bash
    ./isaaclab.sh -p scripts/run.py \
        --task Isaac-Pendulum-Simple-Direct-v0 \
        --num_envs 100 \
        --ml_framework jax \
        --mode train \
        --algorithm *Controller \
        --duration 3.0 \
        --max_rollouts 100 \
        --headless
```

To visualize some test data:
```bash
    ./isaaclab.sh -p scripts/run.py \
        --task Isaac-Pendulum-Simple-Direct-v0 \
        --num_envs 1 \
        --ml_framework jax \
        --mode test \
        --algorithm *Controller \
        --duration 3.0 \
        --max_rollouts 10 \
        --record
```

## Params TODO:
To see the flags you can use for the `run.py` run `./isaaclab.sh -p scripts/run.py --help`

## Information (TDOD: Add Pedro's Notes)

- To create an environment, a .usd file must be created from a .urdf
- The .usd files from this project are to be found at `source/isaaclab_assets/isaaclab_assets/custom`. The .py files set up the simulation.
- The task definitions (RL-problem) are defined in `source/isaaclab_tasks/isaaclab_tasks`. Inside there are the
  managed tasks and the direct tasks. Managed tasks use higher level abstractions that reduce the difficulty while
  direct tasks allow granular control.
- For now, all the environments are being run as managed tasks

## Example commands

Run to train pendulum environment:
```bash
./isaaclab.sh -p scripts/effort_calcs/run.py --task Isaac-Pendulum-v0 --num_envs 64 --ml_framework jax
--controller 'rl' --headless --mode train
```

Run to record some data on the logs:
```bash
./isaaclab.sh -p scripts/effort_calcs/run.py --task Isaac-Pendulum-v0 --num_envs 1 --ml_framework jax --mode test
--controller pid --record --duration 1.0 --max_runtime_iterations 1000 --headless
```

Run to visualize the firs episode of the latest test:
```bash
./isaaclab.sh -p ./scripts/effort_calcs/visualize.py --task Isaac-Pendulum-v0  --ml_framework jax
```

## Useful commands

If you need to setup wandb again:
```bash
export WANDB_API_KEY=3a8c037b3fd5bda9fb344d61686b81afc661b0cc
pip install wandb
wandb login
```

If you want to debug in pycharm, create a conditional in the script like this and also create a run configuration of
"Python Debugger" with the code mappings pointing at the root folder:
```python
# Handle debugger connection
if args_cli.debugger:
    import pydevd_pycharm

    pydevd_pycharm.settrace(
        host="localhost",
        port=5678,
        stdoutToServer=True,
        stderrToServer=True,  # Optional: waits for debugger to attach before running
    )
```

## Known Issues

If the isaac sim warning that is stuck persists, run:

```bash
gsettings set org.gnome.mutter check-alive-timeout 10000
```

If you run out of GPU memory while using both isaac AND jax on it, then allow dynamic memory allocation to the start of
your script:
```python
# this is required to be able to run both sim and learning in cpu
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
```
