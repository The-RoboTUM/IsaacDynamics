# IsaacDynamics

This project is the first implementation of a co-design framework for the discovery of mechanisms and controllers
through isaac lab.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Params](#params)
- [Information](#information)
- [Useful Commands](#useful-commands)

## Installation

Instructions for installation.
Example:

```bash
sudo apt install cmake build-essential
./isaaclab.sh --install # or "./isaaclab.sh -i"
```

If you use the jax framework make sure to run the following to get all of the dependencies:
```bash
pip install skrl["all"]
```

## Usage

To run the testing environments:
```bash
./isaaclab.sh -p scripts/codesign/<environment_test>
```

To train an environments:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task <environment_entry_point> --num_envs
<parallel_robots>
```

## Params

- `--task`: Which environment are you using. All environments are defined in `source/isaaclab_tasks/isaaclab_tasks`
- `--num_envs`: How many parallel environments to use now
- `--ml_framework`: Either `torch` or `jax`, use jax for better results
- `--experiment_name`: What are you calling the current run
- `--debugger`: Set if want to connect to the pycharm debugger
- `--model_path`: Only relevant for play, how to specify model loading destination, will use latest if not given

## Information

- To create an environment, a .usd file must be created from a .urdf
- The .usd files from this project are to be found at `source/isaaclab_assets/isaaclab_assets/custom`. The .py files set up the simulation.
- The task definitions (RL-problem) are defined in `source/isaaclab_tasks/isaaclab_tasks`. Inside there are the
  managed tasks and the direct tasks. Managed tasks use higher level abstractions that reduce the difficulty while
  direct tasks allow granular control.
- For now, all the environments are being run as managed tasks

## Useful commands

Run if the isaac sim warning persists:

```bash
gsettings set org.gnome.mutter check-alive-timeout 10000
```
Run to train pendulum environment:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Pendulum-v0 --num_envs 64 --ml_framework jax
```

If you want to debug in pycharm, create a conditional in the script like this:
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
