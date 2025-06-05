# CPG

## 1. Key Components

### 1.1 RL Policy

* Outputs the CPG parameters:

  * **Amplitude (μ)**: Sets the desired limit-cycle radius.
  * **Frequency (ω)**: Controls oscillation speed.

### 1.2 CPG Class (Hopf Oscillator)

* **Initialization Parameters:**
  * **num_dof**: Number of degrees of freedom (required)
  * **state_size**: Size of the state vector (required)
  * **delta_t**: Integration timestep (optional, defaults to MIN_TIMESTEP)
  * **t_0**: Initial time (optional, defaults to 0.0)
  * **x_0**: Initial state vector (optional, defaults to zeros)

* **State Variables:** 
  * **x**: Current state vector [x₁, x₂, ...] for each degree of freedom
  * **q**: Joint angles derived from first `num_dof` state variables
  * **p**: Cartesian positions of links
  * **E**: Energy values (kinetic + potential)

* **Equations of Motion:** 
  ```
  dx₁/dt = -x₂ω + x₁(μ² - ρ)
  dx₂/dt = x₁ω + x₂(μ² - ρ)
  where ρ = x₁² + x₂²
  ```

* **Integration:** Uses scipy's `solve_ivp` with RK23 method for numerical integration
* **Output Mapping:** Maps the first `num_dof` state variables to joint angles qᵢ
* **Trajectory Logging:** Records complete histories of:
  * Time (t_traj)
  * Oscillator state (x_traj)
  * Joint outputs (q_traj)
  * Cartesian positions (p_traj)
  * Energy values (E_traj)
  * CPG parameters (params_traj: [ω, μ])

* **Coiling Detection:** Monitors phase angle changes to count revolutions around the origin
* **State Management:** Separate tracking of initial, current, and trajectory values

### 1.3 Low-Level Controller

* **Receives:** Joint angles from the CPG via `get_joint_state()` method.
* **Functions:** Applies PID or torque-based control to track desired angles on hardware.


## 2. Detailed Data Flow

```
[RL Policy]
   └──> {μ, ω} parameters
          └──> [CPG Class]
                 ├─ step(params): Integrate Hopf oscillator dynamics
                 ├─ _state_to_joints(): Map state x to q (joint angles)
                 ├─ update_trajectories(): Log (t, x, q, p, E, μ, ω)
                 ├─ detect_coiling(): Monitor oscillator revolutions
                 └─ get_joint_state(): Output q
                          └──> [Low-Level Controller] → Hardware
```

1. **Policy** provides μ and ω parameters in a dictionary
2. **CPG.step()** integrates oscillator dynamics over Δt using scipy's solve_ivp
3. **State Mapping** converts oscillator state to joint angles via `_state_to_joints()`
4. **Trajectory Updates** store complete state history for analysis
5. **Controller** receives joint angles via `get_joint_state()` method
6. **Logging** enables comprehensive analysis of oscillator behavior and parameter evolution


## 3 Core Methods
* **step(params)**: Advance simulation by one timestep
* **update_trajectories(params)**: Update all trajectory arrays
* **restart(params)**: Reset to initial conditions
* **select_initial(params)**: Set new initial conditions


## 4. Usage Highlights

* **Hopf Oscillator:** Guarantees amplitude convergence to μ and smooth sinusoidal behavior through stable limit cycle dynamics
* **Parameter Updates:** RL policy can adjust μ and ω at runtime for adaptive gait patterns
* **Comprehensive Logging:** Internal storage of all state, joint, parameter, and energy histories supports debugging and visualization
* **Modularity:** Clean separation between oscillator dynamics, state mapping, and trajectory management
* **Real-Time Integration:** Designed for control loop integration with configurable timestep and robust error handling
* **Scientific Computing:** Built on scipy/numpy with plans for JAX migration for improved efficiency

## 5. Demo Script (`demo_cpg.py`)

The demo script provides a comprehensive visualization and testing environment for CPG behavior with dynamic parameter changes.

### 5.1 Features
* **Multi-DOF Support:** Configurable number of oscillators via `--num_dof` argument
* **Dynamic Parameter Changes:** Random parameter updates during simulation to test adaptation
* **Parameter Variation:** 
  * Changes occur every 1-5 seconds randomly
  * 70% probability of changing each μ and ω parameter
  * Realistic parameter bounds (μ: 0.3-2.5, ω: 0.5-6.0)
* **Comprehensive Analysis:** Calculates average tracking error against dynamic targets (not at every timestep)
* **Real-time Visualization:** 4-panel plot showing:
  1. **Input Parameters:** μ and ω trajectories with change markers
  2. **Amplitude Adaptation:** Oscillator radius vs. target amplitude
  3. **Frequency Adaptation:** Instantaneous frequency vs. target frequency  
  4. **Joint Outputs:** Final joint angle trajectories

### 5.2 Usage
```bash
# Single oscillator demo
python source/isaaclab_cpg/tests/demo_cpg.py 

# Multi-oscillator demo
python source/isaaclab_cpg/tests/demo_cpg.py --num_dof 3
```

### 5.3 Output Analysis
* **Error Metrics:** Average percentage error against time-varying targets
* **Convergence Assessment:** Visual tracking performance during parameter transitions
* **System Robustness:** Demonstrates CPG stability under dynamic conditions
* **Export:** Show plots and data for further analysis

