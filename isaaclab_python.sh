#!/bin/bash

# Activate your Conda environment named "sim"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate sim

# Run isaaclab.sh with all arguments passed from PyCharm
/home/andrew/Documents/RoboTUM_Workspace/IsaacDynamics/isaaclab.sh "$@"
