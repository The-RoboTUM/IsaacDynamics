# IsaacDynamics

This project is the first implementation of a co-design framework for the discovery of mechanisms and controllers
through isaac lab.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

Instructions for installation.
Example:

```bash
sudo apt install cmake build-essential
./isaaclab.sh --install # or "./isaaclab.sh -i"
```

## Usage

To run the testing environments:
```bash
./isaaclab.sh -p scripts/codesign/<environment_test>
```

To train an environments:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task <environment_entry_point> --num_envs <parallel_robots>
```

## Information
- To create an environment, a .usd file must be created from a .urdf
- The .usd files from this project are to be found at `source/isaaclab_assets/isaaclab_assets/custom`. The .py files set up the simulation.
- The task definitions (RL-problem) are defined in `source/isaaclab_tasks/isaaclab_tasks`. Inside there are the
  managed tasks and the direct tasks. Managed tasks use higher level abstractions that reduce the difficulty while
  direct tasks allow granular control.
- For now, all the environments are being run as managed tasks
