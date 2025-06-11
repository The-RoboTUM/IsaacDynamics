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

To run an environment with a certain controller:
```bash
./isaaclab.sh -p scripts/codesign/run.py --task <environment_handle> --controller <controler_name> --mode test
```
Note: In the future, the controllers will also have access points

To train an environments:
```bash
./isaaclab.sh -p scripts/codesign/run.py --task <environment_entry_point> --num_envs --controller <rl-based controller>
<parallel_robots> --mode 'train'
```

To visualize some test data:
```bash
./isaaclab.sh -p ./scripts/codesign/visualize.py --task <environment_handle>  --ml_framework <dl-backend>
```

## Params

- `--task`: Which environment are you using. All environments are defined in `source/isaaclab_tasks/isaaclab_tasks`
- `--num_envs`: How many parallel environments to use now
- `--ml_framework`: Either `torch` or `jax`, use jax for better results
- `--experiment_name`: What are you calling the current run
- `--debugger`: Set if want to connect to the pycharm debugger
- `--model_path`: Only relevant for play, how to specify model loading destination, will use latest if not given
- `--video`: Set if you want to record videos (required test time) (Extra flags: `--video_length`, `--video_interval`)
- `--duration`: Time in seconds for each episode
- `--controller`: Name of the controller you want to use with your system
- `--mode`: If you want to run the current experiment in test or train mode
- `--record`: If set it will record the observation and actions to databases in the logs

## Information

- To create an environment, a .usd file must be created from a .urdf
- The .usd files from this project are to be found at `source/isaaclab_assets/isaaclab_assets/custom`. The .py files set up the simulation.
- The task definitions (RL-problem) are defined in `source/isaaclab_tasks/isaaclab_tasks`. Inside there are the
  managed tasks and the direct tasks. Managed tasks use higher level abstractions that reduce the difficulty while
  direct tasks allow granular control.
- For now, all the environments are being run as managed tasks

## Example commands

Run to train pendulum environment:
```bash
./isaaclab.sh -p scripts/codesign/run.py --task Isaac-Pendulum-v0 --num_envs 64 --ml_framework jax
--controller 'rl' --headless --mode train
```

Run to record some data on the logs:
```bash
./isaaclab.sh -p scripts/codesign/run.py --task Isaac-Pendulum-v0 --num_envs 1 --ml_framework jax --mode test
--controller pid --record --duration 1.0 --max_runtime_iterations 1000 --headless
```

Run to visualize the firs episode of the latest test:
```bash
./isaaclab.sh -p ./scripts/codesign/visualize.py --task Isaac-Pendulum-v0  --ml_framework jax
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
