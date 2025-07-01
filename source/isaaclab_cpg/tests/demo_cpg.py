import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os

# TODO: Adjust the import path based on your project structure; Preferably with init files
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)
from source.isaaclab_cpg.cpg import CPG

def run_cpg_demo(num_dof=1):
    # Simulation parameters
    state_size = 2 * num_dof  # [x1, x2] for single Hopf oscillator (one oscillator for each DoF)
    delta_t = 0.005
    simulation_time = 10.0  # seconds
    num_steps = int(simulation_time / delta_t)
    
    # Generate different random CPG parameters for each joint
    np.random.seed(20)  # For reproducible results
    mu_targets = np.random.uniform(0.5, 2.0, num_dof)  # Different amplitude for each joint
    omega_targets = np.random.uniform(1.0, 5.0, num_dof)  # Different frequency for each joint
    
    # Parameter change settings
    min_change_interval = 1.0  # Minimum time between changes (seconds)
    max_change_interval = 5.0  # Maximum time between changes (seconds)
    next_change_time = np.random.uniform(min_change_interval, max_change_interval)
    change_times = [] 
    
    print(f"Demo Parameters:")
    print(f"- Number of DOF: {num_dof}")
    for i in range(num_dof):
        print(f"- Joint {i+1} - Initial Amplitude (μ): {mu_targets[i]:.3f}, Initial Frequency (ω): {omega_targets[i]:.3f}")
    print(f"- Simulation time: {simulation_time}s")
    print(f"- Time step: {delta_t}s")
    print(f"- Number of steps: {num_steps}")
    print(f"- Parameter changes: Every {min_change_interval:.1f}-{max_change_interval:.1f}s randomly")

    # Create initial state for all oscillators
    x_0 = []
    for i in range(num_dof):
        x_0.extend([0.5 + 0.1 * i, 0.1 + 0.05 * i])
    
    cpg_params = {
        'num_dof': num_dof,
        'state_size': state_size,
        'delta_t': delta_t,
        't_0': 0.0,
        'x_0': np.array(x_0) 
    }
    
    cpg = CPG(cpg_params)
    
    # Initialize parameters using the update_parameters method
    initial_params = {
        'mu': mu_targets, 
        'omega': omega_targets 
    }
    cpg.update_parameters(initial_params)
    
    # Run simulation
    for i in range(num_steps):
        current_time = i * delta_t
        
        # Check if it's time to change parameters
        if current_time >= next_change_time:
            # Randomly change some parameters
            for j in range(num_dof):
                # 70% chance to change each parameter
                if np.random.random() < 0.7:
                    # Change mu (amplitude) within reasonable bounds
                    mu_targets[j] = np.random.uniform(0.3, 2.5)
                if np.random.random() < 0.7:
                    # Change omega (frequency) within reasonable bounds
                    omega_targets[j] = np.random.uniform(0.5, 6.0)
            
            # Update CPG parameters using the update_parameters method
            new_params = {
                'mu': mu_targets.copy(),
                'omega': omega_targets.copy()
            }
            cpg.update_parameters(new_params)
            
            # Record the change
            change_times.append(current_time)
            
            # Schedule next change
            next_change_time = current_time + np.random.uniform(min_change_interval, max_change_interval)
        
        # Step the CPG forward with current parameters
        step_params = {
            'mu': cpg.mu_current,
            'omega': cpg.omega_current
        }
        cpg.step(step_params)
        
        # Update trajectories (no parameters needed - uses internal state)
        cpg.update_trajectories()
        
        # Detect coiling
        cpg.detect_coiling()
        
        if (i + 1) % 100 == 0:
            current_state = cpg.get_state()
            current_radius = np.sqrt(current_state[0]**2 + current_state[1]**2)
    
    # Extract trajectories for visualization
    time_traj = cpg.get_temporal_traj()
    state_traj = cpg.get_state_traj()
    joint_traj = cpg.get_joint_traj()
    cartesian_traj = cpg.get_cartesian_traj()
    param_traj = cpg.get_parametric_traj()
    
    # Analyze convergence properties first
    analyze_convergence(time_traj, state_traj, param_traj)
    
    # Create visualization after analysis
    create_visualization(time_traj, state_traj, joint_traj, cartesian_traj, 
                        param_traj, mu_targets, omega_targets, num_dof, change_times)

def create_visualization(time_traj, state_traj, joint_traj, cartesian_traj, 
                        param_traj, mu_targets, omega_targets, num_dof, change_times):
    
    # Check if we have enough data points
    if len(time_traj) < 2:
        print("Error: Insufficient trajectory data for visualization")
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('CPG Demo', fontsize=16, fontweight='bold')
    
    # Plot 1: Input parameters over time (now showing actual changes)
    ax1 = axes[0, 0]
    for i in range(num_dof):
        omega_col = i
        mu_col = num_dof + i
        ax1.plot(time_traj, param_traj[:, omega_col], 
                linewidth=2, label=f'ω{i+1}')
        ax1.plot(time_traj, param_traj[:, mu_col], 
                linewidth=2, label=f'μ{i+1}')
    
    for change_time in change_times:
        ax1.axvline(x=change_time, color='red', linestyle=':', alpha=0.7, linewidth=1)
    
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Parameter Value')
    ax1.set_title('CPG Input Parameters')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Radius convergence for all oscillators
    ax2 = axes[0, 1]
    colors = ['purple', 'blue', 'red', 'green', 'orange']
    for i in range(min(num_dof, len(colors))):
        x_idx = 2 * i
        y_idx = 2 * i + 1
        if y_idx < state_traj.shape[1]:
            radius = np.sqrt(state_traj[:, x_idx]**2 + state_traj[:, y_idx]**2)
            ax2.plot(time_traj, radius, colors[i], linewidth=2, label=f'Oscillator {i+1} radius')
            
            
            mu_col = num_dof + i
            ax2.plot(time_traj, param_traj[:, mu_col], colors[i], linestyle='--', alpha=0.7, linewidth=1,
                    label=f'Target μ{i+1}')
    
    
    for change_time in change_times:
        ax2.axvline(x=change_time, color='red', linestyle=':', alpha=0.7, linewidth=1)
    
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Radius')
    ax2.set_title('Amplitude Adaptation')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Frequency convergence for all oscillators
    ax3 = axes[1, 0]
    
    if len(time_traj) > 10:
        dt = time_traj[1] - time_traj[0]
        window_size = min(100, len(time_traj) // 5)
        
        freq_values = []
        
        colors_freq = ['red', 'green', 'blue', 'orange', 'purple'] 
        for i in range(min(num_dof, len(colors_freq))):
            x_idx = 2 * i
            y_idx = 2 * i + 1
            if y_idx < state_traj.shape[1]:
                phase = np.arctan2(state_traj[:, y_idx], state_traj[:, x_idx])
                phase_unwrapped = np.unwrap(phase)
                freq_instant = []
                for j in range(window_size, len(phase_unwrapped)):
                    phase_diff = phase_unwrapped[j] - phase_unwrapped[j-window_size]
                    time_diff = time_traj[j] - time_traj[j-window_size]
                    freq_instant.append(phase_diff / time_diff)
                time_freq = time_traj[window_size:]
                ax3.plot(time_freq, freq_instant, colors_freq[i], linewidth=2, 
                        label=f'Oscillator {i+1} frequency')
                
                
                omega_col = i
                target_freq_window = param_traj[window_size:, omega_col]
                ax3.plot(time_freq, target_freq_window, colors_freq[i], linestyle='--', alpha=0.7, linewidth=1,
                        label=f'Target ω{i+1}')
                
                freq_values.extend(freq_instant)
        
        for change_time in change_times:
            if change_time >= time_freq[0]:
                ax3.axvline(x=change_time, color='red', linestyle=':', alpha=0.7, linewidth=1)
    
        ax3.ticklabel_format(style='plain', axis='y')
    
        if freq_values:
            all_targets = list(omega_targets)
            freq_min = min(min(freq_values), min(all_targets)) - 0.1
            freq_max = max(max(freq_values), max(all_targets)) + 0.1
            ax3.set_ylim([freq_min, freq_max])
    
    else:
        ax3.text(0.5, 0.5, 'Insufficient data\nfor frequency analysis', 
                ha='center', va='center', transform=ax3.transAxes)
        
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Frequency [rad/s]')
    ax3.set_title('Frequency Adaptation')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Joint angle outputs
    ax4 = axes[1, 1]
    colors_joint = ['green', 'blue', 'red', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i in range(min(joint_traj.shape[1], len(colors_joint))):
        ax4.plot(time_traj, joint_traj[:, i], color=colors_joint[i], 
                linewidth=2, label=f'Joint {i+1}')
    
    for change_time in change_times:
        ax4.axvline(x=change_time, color='red', linestyle=':', alpha=0.7, linewidth=1)
    
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Joint Angle [rad]')
    ax4.set_title('Joint Outputs')
    ax4.grid(True, alpha=0.3)
    
    if joint_traj.shape[1] > 1:
        ax4.legend()
    
    plt.tight_layout()
    plt.show()

def analyze_convergence(time_traj, state_traj, param_traj):
    print(f"\n=== Convergence Analysis (Average Error vs Dynamic Targets) ===")
    
    num_oscillators = state_traj.shape[1] // 2
    print(f"Analyzing {num_oscillators} oscillator(s)")
    
    # Amplitude convergence analysis for all oscillators (average error)
    print(f"\n--- Average Amplitude (μ) Error vs Dynamic Targets ---")
    for i in range(num_oscillators):
        x_idx = 2 * i
        y_idx = 2 * i + 1
        if y_idx < state_traj.shape[1]:
            radius = np.sqrt(state_traj[:, x_idx]**2 + state_traj[:, y_idx]**2)
            
            # Get the dynamic target trajectory for this oscillator
            mu_col = num_oscillators + i 
            mu_target_traj = param_traj[:, mu_col]
            
            # Filter out zero or near-zero targets to avoid division by zero
            valid_mask = np.abs(mu_target_traj) > 1e-6
            if np.any(valid_mask):
                radius_valid = radius[valid_mask]
                mu_target_valid = mu_target_traj[valid_mask]
                
                # Calculate average error against the dynamic target (only for valid points)
                average_radius_error = np.mean(np.abs(radius_valid - mu_target_valid) / mu_target_valid * 100)
                
                print(f"  Oscillator {i+1}: Average error = {average_radius_error:.2f}%")
            else:
                print(f"  Oscillator {i+1}: No valid target values found (all targets near zero)")
    
    # Frequency convergence analysis for all oscillators (average error)
    print(f"\n--- Average Frequency (ω) Error vs Dynamic Targets ---")
    
    if len(time_traj) > 10:
        dt = time_traj[1] - time_traj[0]
        
        for i in range(num_oscillators):
            x_idx = 2 * i
            y_idx = 2 * i + 1
            if y_idx < state_traj.shape[1]:
                phase = np.arctan2(state_traj[:, y_idx], state_traj[:, x_idx])
                phase_unwrapped = np.unwrap(phase)
                freq_instant = np.gradient(phase_unwrapped, dt)
                
                # Get the dynamic target trajectory for this oscillator
                omega_col = i  # omega parameters are first in param_traj
                omega_target_traj = param_traj[:, omega_col]
                
                # Filter out zero or near-zero targets to avoid division by zero
                valid_mask = np.abs(omega_target_traj) > 1e-6
                if np.any(valid_mask):
                    freq_valid = freq_instant[valid_mask]
                    omega_target_valid = omega_target_traj[valid_mask]
                    
                    # Calculate average frequency error against the dynamic target (only for valid points)
                    average_freq_error = np.mean(np.abs(freq_valid - omega_target_valid) / omega_target_valid * 100)
                    
                    print(f"  Oscillator {i+1}: Average error = {average_freq_error:.2f}%")
                else:
                    print(f"  Oscillator {i+1}: No valid target values found (all targets near zero)")
    else:
        print("Insufficient data for frequency analysis")

if __name__ == "__main__":
    print("CPG Demo Script")
    print("===============")
    
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='CPG Demo with configurable degrees of freedom')
    parser.add_argument('--num_dof', type=int, default=1, 
                       help='Number of degrees of freedom (default: 1)')
    args = parser.parse_args()
    
    try:
        run_cpg_demo(num_dof=args.num_dof)
    except Exception as e:
        print(f"\nError during demo: {str(e)}")
        raise
