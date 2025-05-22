#!/usr/bin/env bash

# Script to run all the experiments

# Train the environment with spring
#./isaaclab.sh -p scripts/codesign/run.py \
#    --task Isaac-Pendulum-Simple-Direct-v0 \
#    --num_envs 100 \
#    --ml_framework jax \
#    --mode train \
#    --controller rl \
#    --duration 3.0 \
#    --max_episodes 100 \
#    --spring \
#    --headless

## Gather data with each controller
## 1. rl
#./isaaclab.sh -p scripts/codesign/run.py \
#    --task Isaac-Pendulum-Simple-Direct-v0 \
#    --num_envs 1 \
#    --ml_framework jax \
#    --mode test \
#    --controller rl \
#    --duration 3.0 \
#    --max_episodes 100 \
#    --spring \
#    --record \
#    --headless
#./isaaclab.sh -p ./scripts/codesign/save_control_effort.py \
# --task Isaac-Pendulum-Simple-Direct-v0 \
# --ml_framework jax \
# --experiment_name spring_rl
## 2. pid
#./isaaclab.sh -p scripts/codesign/run.py \
#    --task Isaac-Pendulum-Simple-Direct-v0 \
#    --num_envs 1 \
#    --ml_framework jax \
#    --mode test \
#    --controller pid \
#    --duration 3.0 \
#    --max_episodes 100 \
#    --spring \
#    --record \
#    --headless
#./isaaclab.sh -p ./scripts/codesign/save_control_effort.py \
# --task Isaac-Pendulum-Simple-Direct-v0 \
# --ml_framework jax \
# --experiment_name spring_pid
# # 3. random
# ./isaaclab.sh -p scripts/codesign/run.py \
#    --task Isaac-Pendulum-Simple-Direct-v0 \
#    --num_envs 1 \
#    --ml_framework jax \
#    --mode test \
#    --controller random \
#    --duration 3.0 \
#    --max_episodes 100 \
#    --spring \
#    --record \
#    --headless
#./isaaclab.sh -p ./scripts/codesign/save_control_effort.py \
# --task Isaac-Pendulum-Simple-Direct-v0 \
# --ml_framework jax \
# --experiment_name spring_random

# Train the environment with spring
./isaaclab.sh -p scripts/codesign/run.py \
    --task Isaac-Pendulum-Simple-Direct-v0 \
    --num_envs 100 \
    --ml_framework jax \
    --mode train \
    --controller rl \
    --duration 3.0 \
    --max_episodes 100 \
    --headless

# Gather data with each controller
# 1. rl
./isaaclab.sh -p scripts/codesign/run.py \
    --task Isaac-Pendulum-Simple-Direct-v0 \
    --num_envs 1 \
    --ml_framework jax \
    --mode test \
    --controller rl \
    --duration 3.0 \
    --max_episodes 100 \
    --record \
    --headless
./isaaclab.sh -p ./scripts/codesign/save_control_effort.py \
 --task Isaac-Pendulum-Simple-Direct-v0 \
 --ml_framework jax \
 --experiment_name rl
# 2. pid
./isaaclab.sh -p scripts/codesign/run.py \
    --task Isaac-Pendulum-Simple-Direct-v0 \
    --num_envs 1 \
    --ml_framework jax \
    --mode test \
    --controller pid \
    --duration 3.0 \
    --max_episodes 100 \
    --record \
    --headless
./isaaclab.sh -p ./scripts/codesign/save_control_effort.py \
 --task Isaac-Pendulum-Simple-Direct-v0 \
 --ml_framework jax \
 --experiment_name pid
 # 3. random
 ./isaaclab.sh -p scripts/codesign/run.py \
    --task Isaac-Pendulum-Simple-Direct-v0 \
    --num_envs 1 \
    --ml_framework jax \
    --mode test \
    --controller random \
    --duration 3.0 \
    --max_episodes 100 \
    --record \
    --headless
./isaaclab.sh -p ./scripts/codesign/save_control_effort.py \
 --task Isaac-Pendulum-Simple-Direct-v0 \
 --ml_framework jax \
 --experiment_name random
