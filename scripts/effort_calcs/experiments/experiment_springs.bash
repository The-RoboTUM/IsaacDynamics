#!/usr/bin/env bash

set -e

# === Default config ===
MAX_ROLLOUTS=100
SPRING_SETPOINTS=(90 60 45 30 0)  # Add more setpoints here

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
    --headless > /dev/null

echo ">>> Starting controller experiments WITHOUT spring..."
./scripts/effort_calcs/experiments/experiment_controller.bash \
    --max_rollouts "$MAX_ROLLOUTS" \
    --collect

# === Loop over custom spring setpoints ===
for SETPOINT in "${SPRING_SETPOINTS[@]}"; do
    echo ">>> Starting PPO training WITH spring and setpoint $SETPOINT..."
    ./isaaclab.sh -p scripts/run.py \
        --task Isaac-Pendulum-Simple-Direct-v0 \
        --num_envs 100 \
        --ml_framework jax \
        --mode train \
        --algorithm ppo \
        --duration 3.0 \
        --max_rollouts 100 \
        --spring \
        --spring_setpoint "$SETPOINT" \
        --headless > /dev/null

    echo ">>> Starting controller experiments WITH spring and setpoint $SETPOINT..."
    ./scripts/effort_calcs/experiments/experiment_controller.bash \
        --max_rollouts "$MAX_ROLLOUTS" \
        --spring \
        --spring_setpoint "$SETPOINT" \
        --collect
done

echo ">>> Plotting the results..."
python scripts/effort_calcs/visualizations/plot_controllers.py

echo ">>> Done."
