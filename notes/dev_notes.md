# Dev Notes

# 30.04.25
- Today I have been performing experiments of the control effort metric on the pendulum
- I got this for a random controller
  - INFO: Control effort: 0.93 +- 0.58 Bits (per episode)
  - INFO: Control effort per second: 0.37 +- 0.23 Bits/s
- I got this for a PID controller
  - INFO: Control effort: 53.08 +- 16.88 Bits (per episode)
  - INFO: Control effort per second: 21.23 +- 6.75 Bits/s

# 28.04.25
- Today I am finishing the data gathering mechanisms and the probability modeling classes to calculate the metric
- Will be closing this branch in the meantime
- Today I finally finished the data recording wrapper for controllers, still has some minor TODOs should be usable
  for arbitrary controllers
- Created a basic visualization for observations and actions and tested that saving and loading database recordings
  work like a charm

# 23.04.25
- Today I need to finish something I could demo for the progress of my thesis
- Until this day I finished the control environment, implemented and tested a few controllers

# 16.04.25
- Today I have a lot of time to work on the project, let's see if I manage to record some data
- Managed to graph different performance metrics in tensorflow and wandb
- Choose the next environments to create and created some models
- Will be creating a repo only to handle the python extension to create URDFs

# 15.04.25
- It's been a while since I continued the notes
- I learned that I have to specify the environment to JAX if I want it to actually use it.
- Tested JAX with SKRL
- Updated the documentation on how to use JAX and commands to install skrl when installing isaac
- Modified the scripts for a rudimentary saving and loading models based on my wishes

## 25.03.25
- Starting setting up the project
- Copied the code from my previous project
