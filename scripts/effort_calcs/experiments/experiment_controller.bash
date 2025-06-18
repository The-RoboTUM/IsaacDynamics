#!/usr/bin/env bash

set -e

# === Parse flags ===
SPRING_ENABLED=false
COLLECT_ENABLED=false
MAX_ROLLOUTS=100

while [[ $# -gt 0 ]]; do
    case $1 in
        --spring)
            SPRING_ENABLED=true
            shift
            ;;
        --collect)
            COLLECT_ENABLED=true
            shift
            ;;
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

echo ">>> SPRING ENABLED: $SPRING_ENABLED"
echo ">>> COLLECT ENABLED: $COLLECT_ENABLED"
echo ">>> MAX ROLLOUTS: $MAX_ROLLOUTS"

SPRING_FLAG=""
EXPERIMENT_SUFFIX="springless"
if $SPRING_ENABLED; then
    SPRING_FLAG="--spring"
    EXPERIMENT_SUFFIX="spring"
fi

# === List of controllers ===
controllers=("ppo" "pid" "random" "unactuated")

collect_rollout() {
    CONTROLLER=$1
    EXTRA_ARGS=""

    if [[ "$CONTROLLER" == "unactuated" ]]; then
        EXTRA_ARGS="--range 0.0"
    fi

    echo ">>> Starting rollout collection for the $CONTROLLER controller..."
    ./isaaclab.sh -p scripts/run.py \
        --task Isaac-Pendulum-Simple-Direct-v0 \
        --num_envs 1 \
        --ml_framework jax \
        --mode test \
        --algorithm "$CONTROLLER" \
        --duration 3.0 \
        --max_rollouts "$MAX_ROLLOUTS" \
        --record \
        $SPRING_FLAG \
        $EXTRA_ARGS \
        --headless
    echo ">>> Finished rollout collection for $CONTROLLER."
}

calculate_metric() {
    CONTROLLER=$1
    echo ">>> Calculating metrics with $CONTROLLER controller..."
    ./isaaclab.sh -p scripts/effort_calcs/metric_experiment.py \
        --task Isaac-Pendulum-Simple-Direct-v0 \
        --ml_framework jax \
        --algorithm "$CONTROLLER" \
        --stoch_mode full \
        --episode_num "$MAX_ROLLOUTS" \
        $SPRING_FLAG \
        --experiment_name controllers_"$CONTROLLER"_"$EXPERIMENT_SUFFIX" \
        --results_file_name experiment_controllers_"$EXPERIMENT_SUFFIX".json
}

# === Step 1: Rollouts ===
if $COLLECT_ENABLED; then
    for CONTROLLER in "${controllers[@]}"; do
        collect_rollout "$CONTROLLER"
    done
fi

# === Step 2: Metric calculations ===
for CONTROLLER in "${controllers[@]}"; do
    calculate_metric "$CONTROLLER"
done
