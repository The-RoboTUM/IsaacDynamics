#!/usr/bin/env bash

echo ">>> Starting rollout collection with 100 rollouts..."
./isaaclab.sh -p scripts/run.py \
    --task Isaac-Pendulum-Simple-Direct-v0 \
    --num_envs 1 \
    --ml_framework jax \
    --mode test \
    --algorithm ppo \
    --duration 3.0 \
    --max_rollouts 100 \
    --record \
    --headless
echo ">>> Finished rollout collection."

echo ">>> Starting metric calculations..."
for N in 2 5 10 50 100; do
    echo ">>> Calculating metrics for rollouts_${N} full metric..."
    ./isaaclab.sh -p scripts/effort_calcs/metric_experiment.py \
        --task Isaac-Pendulum-Simple-Direct-v0 \
        --ml_framework jax \
        --algorithm ppo \
        --stoch_mode full \
        --episode_num $N \
        --experiment_name rollouts_${N}_full \
        --results_file_name experiment_rollout_num.json

    echo ">>> Calculating metrics for rollouts_${N} simplified metric..."
    ./isaaclab.sh -p scripts/effort_calcs/metric_experiment.py \
      --task Isaac-Pendulum-Simple-Direct-v0 \
      --ml_framework jax \
      --algorithm ppo \
      --stoch_mode simplified \
      --episode_num $N \
      --experiment_name rollouts_${N}_simplified \
      --results_file_name experiment_rollout_num.json
done
echo ">>> All metrics calculated successfully."

echo ">>> Save the final plots of the experiment."
python scripts/effort_calcs/visualizations/plot_rollouts.py \
--results_file_name /home/andrew/Documents/RoboTUM_Workspace/IsaacDynamics/logs/skrl/pendulum_simple_direct/2025-06-16_02-27-37_ppo_jax/results/experiment_rollout_num.json
