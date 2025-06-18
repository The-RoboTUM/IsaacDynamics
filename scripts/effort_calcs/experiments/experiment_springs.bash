#!/usr/bin/env bash

set -e

# === Default config ===
MAX_ROLLOUTS=100

# === Parse arguments ===
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max_rollouts)
            MAX_ROLLOUTS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo ">>> Starting PPO training WITHOUT spring..."
./isaaclab.sh -p scripts/run.py \
    --task Isaac-Pendulum-Simple-Direct-v0 \
    --num_envs 100 \
    --ml_framework jax \
    --mode train \
    --algorithm ppo \
    --duration 3.0 \
    --max_rollouts 100 \
    --headless

echo ">>> Starting controller experiments WITHOUT spring..."
./scripts/effort_calcs/experiments/experiment_controller.bash \
    --max_rollouts "$MAX_ROLLOUTS" \
    --collect

echo ">>> Starting PPO training WITH spring..."
./isaaclab.sh -p scripts/run.py \
    --task Isaac-Pendulum-Simple-Direct-v0 \
    --num_envs 100 \
    --ml_framework jax \
    --mode train \
    --algorithm ppo \
    --duration 3.0 \
    --max_rollouts 100 \
    --spring \
    --headless

echo ">>> Starting controller experiments WITH spring..."
./scripts/effort_calcs/experiments/experiment_controller.bash \
    --max_rollouts "$MAX_ROLLOUTS" \
    --spring \
    --collect

echo ">>> Plotting the results..."
python scripts/effort_calcs/visualizations/plot_controllers.py

echo ">>> Done."
