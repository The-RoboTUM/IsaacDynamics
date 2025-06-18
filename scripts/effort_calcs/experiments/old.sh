#!/usr/bin/env bash

# Script to run all the experiments

# Run dummy trainings
./isaaclab.sh -p scripts/codesign/run.py \
    --task Isaac-Pendulum-Simple-Direct-v0 \
    --num_envs 1 \
    --ml_framework jax \
    --mode train \
    --controller pid \
    --algorithm pid \
    --duration 3.0 \
    --max_episodes 1 \
    --headless
./isaaclab.sh -p scripts/codesign/run.py \
    --task Isaac-Pendulum-Simple-Direct-v0 \
    --num_envs 1 \
    --ml_framework jax \
    --mode train \
    --controller random \
    --algorithm random \
    --duration 3.0 \
    --max_episodes 1 \
    --headless

# Train the environment with spring
./isaaclab.sh -p scripts/codesign/run.py \
    --task Isaac-Pendulum-Simple-Direct-v0 \
    --num_envs 100 \
    --ml_framework jax \
    --mode train \
    --controller rl \
    --algorithm ppo \
    --duration 3.0 \
    --max_episodes 100 \
    --spring \
    --headless

# Gather data with each controller
# 1. rl
./isaaclab.sh -p scripts/codesign/run.py \
    --task Isaac-Pendulum-Simple-Direct-v0 \
    --num_envs 1 \
    --ml_framework jax \
    --mode test \
    --controller rl \
    --algorithm ppo \
    --duration 3.0 \
    --max_episodes 100 \
    --spring \
    --record \
    --headless
./isaaclab.sh -p ./scripts/codesign/save_control_effort.py \
 --task Isaac-Pendulum-Simple-Direct-v0 \
 --ml_framework jax \
--algorithm ppo \
 --experiment_name spring_rl

# 2. pid
./isaaclab.sh -p scripts/codesign/run.py \
    --task Isaac-Pendulum-Simple-Direct-v0 \
    --num_envs 1 \
    --ml_framework jax \
    --mode test \
    --controller pid \
    --algorithm pid \
    --duration 3.0 \
    --max_episodes 100 \
    --spring \
    --record \
    --headless
./isaaclab.sh -p ./scripts/codesign/save_control_effort.py \
 --task Isaac-Pendulum-Simple-Direct-v0 \
 --ml_framework jax \
 --algorithm pid \
 --experiment_name spring_pid

 # 3. random
 ./isaaclab.sh -p scripts/codesign/run.py \
    --task Isaac-Pendulum-Simple-Direct-v0 \
    --num_envs 1 \
    --ml_framework jax \
    --mode test \
    --controller random \
    --algorithm random \
    --duration 3.0 \
    --max_episodes 100 \
    --spring \
    --record \
    --headless
./isaaclab.sh -p ./scripts/codesign/save_control_effort.py \
 --task Isaac-Pendulum-Simple-Direct-v0 \
 --ml_framework jax \
 --algorithm random \
 --experiment_name spring_random

 # 4. unactuated
  ./isaaclab.sh -p scripts/codesign/run.py \
    --task Isaac-Pendulum-Simple-Direct-v0 \
    --num_envs 1 \
    --ml_framework jax \
    --mode test \
    --controller random \
    --algorithm random \
    --duration 3.0 \
    --max_rollouts 100 \
    --spring \
    --record \
    --headless \
    --range 0.0
./isaaclab.sh -p ./scripts/codesign/save_control_effort.py \
 --task Isaac-Pendulum-Simple-Direct-v0 \
 --ml_framework jax \
 --algorithm random \
 --experiment_name spring_unactuated

 # 5. unaligned spring
