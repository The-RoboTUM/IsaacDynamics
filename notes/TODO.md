# TODO List

## 📌 High-Priority Tasks
- Create a managed env for
  - Double pendulum
  - Acrobot
  - Slip walker
  - Articulated walker
- Encapsulate most of the complexity of the environment in own class
- Write a trainer backend for RL controllers
- Write a graphing class
- Write a visualization script
- Set up a nice simple logging directory and setup interface in main script

## 🔄 In Progress
- [X] Save the information per step into a pandas frame and maybe a database

## ✅ Completed
- [X] Make the dummy wrapper print all the info that you want to record
- [X] Write a dummy wrapper around the controller class that is still recognized as a controller
- [x] Unify the run script into one method
- [X] Write a PID controller for the pendulum
- [x] Write a section of the run script handling variable models and controllers etc.
- [x] Document the issue with jax and memory allocation & write an issue in isaac sim
- [x] Document how to use the debugger
- [x] Read the notes from the previous meeting with Daniel, decide on tasks and add them here (Then proceed)
- [x] Write a run script that can handle play/train matters

## 💡 Future Ideas
- Figure out how to make the env look nice with nice cameras
- Look into how to use your local version of skrl, since this one has the fix you made
- Make the way that controllers are accessed the same way that envs are accessed
