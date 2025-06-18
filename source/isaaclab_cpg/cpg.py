import numpy as np  # TODO: Implement JAX for efficiency, e.g. with jax.numpy; task for group 3
from typing import Dict
from scipy.integrate import solve_ivp
import warnings

# TODO: Import constants from config file or anywherere else appropriate; Fallback definitions:
TIMESTEP = 0.01
INTEGRATION_METHOD = 'RK23'


class CPG:
    def __init__(self, params: Dict = None):
        if params is None:
            params = {}
        
        # Validate and initialize required parameters
        self._validate_required_params(params)
        self.num_dof = params['num_dof']
        self.state_size = params['state_size']
        
        # Set timestep with warning if not provided
        if 'delta_t' not in params:
            warnings.warn(f"'delta_t' not provided, using default TIMESTEP={TIMESTEP}")
        self.delta_t = params.get('delta_t', TIMESTEP)
        
        # Initialize equations of motion
        self.eqs_motion = self._make_eqs_motion()
        
        # Initialize CPG-specific parameters FIRST
        self._initialize_cpg_params()
        
        # Initialize state variables
        self._initialize_states(params)
        
        # Initialize tracking arrays
        self._initialize_trajectories()

    def _validate_required_params(self, params: Dict):
        required_params = ['num_dof', 'state_size']
        for param in required_params:
            if param not in params:
                raise ValueError(f"CPG error: Required parameter '{param}' must be provided.")

    def _initialize_states(self, params: Dict):
        self.t_0 = np.asarray([params.get('t_0', 0.0)]).flatten()
        
        # Use provided x_0 or default to zeros
        if 'x_0' in params:
            self.x_0 = np.asarray(params['x_0'])
        else:
            self.x_0 = np.zeros(self.state_size)
            
        self.q_0 = np.zeros(self.num_dof)
        
        # Current state variables
        self.t_current = self.t_0.copy()
        self.x_current = self.x_0.copy()
        self.q_current = self.q_0.copy()
        
        # Derived variables
        self.p_0 = self.get_link_cartesian_positions()
        self.E_0 = np.asarray([sum(self.get_energies())]).flatten()
        self.p_current = self.p_0.copy()
        self.E_current = self.E_0.copy()

    def _initialize_trajectories(self):
        self.t_traj = self.t_current.copy()
        self.x_traj = np.asarray([self.x_current])
        self.q_traj = np.asarray([self.q_current])
        self.p_traj = np.asarray([self.p_current])
        self.E_traj = self.E_current.copy()

    def _initialize_cpg_params(self):
        self.omega_current = np.zeros(self.num_dof)
        self.mu_current = np.zeros(self.num_dof)
        self.coils = 0
        
        # CPG parameter tracking - initialize with correct size for arrays
        # Format: [omega1, omega2, ..., mu1, mu2, ...]
        params_cpg = np.zeros(2 * self.num_dof)
        self.params_traj = np.asarray([params_cpg])

    def _make_eqs_motion(self):
        def eqs_motion(t, x, params):
            mu = params['mu']
            omega = params['omega']

            # Calculate number of oscillators from state size
            num_oscillators = len(x) // 2
            dx = np.zeros_like(x)
            
            # Process each oscillator pair
            for i in range(num_oscillators):
                x1_idx = 2 * i
                x2_idx = 2 * i + 1
                
                if x2_idx < len(x):
                    x1 = x[x1_idx]
                    x2 = x[x2_idx]
                    
                    # Use per-oscillator parameters
                    mu_i = mu[i] if i < len(mu) else mu[0]
                    omega_i = omega[i] if i < len(omega) else omega[0]
                    
                    rho = x1 ** 2 + x2 ** 2
                    circleDist = mu_i ** 2 - rho

                    dx[x1_idx] = -x2 * omega_i + x1 * circleDist
                    dx[x2_idx] = x1 * omega_i + x2 * circleDist

            return dx

        return eqs_motion

    def update_parameters(self, params: Dict):
        if 'mu' in params:
            self.mu_current = np.asarray(params['mu']) if not np.isscalar(params['mu']) else np.full(self.num_dof, params['mu'])
        if 'omega' in params:
            self.omega_current = np.asarray(params['omega']) if not np.isscalar(params['omega']) else np.full(self.num_dof, params['omega'])

    def step(self, params: Dict = None):
        if params is None:
            raise ValueError("CPG step error: Parameter dictionary must be provided.")
        
        # Set integration time span
        t_final = params.get('t_final', self.t_current + self.delta_t)
        if np.isscalar(t_final):
            t_final = np.asarray([t_final])
        ts = np.asarray([self.t_current, t_final]).flatten()

        # Simulate one step
        sim_params = {
            'eqs': self.eqs_motion,
            'eqs_params': params,
            'ts': ts,
            'x_0': self.x_current
        }
        self.x_current = self._simulate(sim_params)

        # Update derived variables
        self.p_current = self.get_link_cartesian_positions()
        self.q_current = self._state_to_joints()
        self.E_current = np.asarray([sum(self.get_energies())]).flatten()
        self.t_current = t_final

    def _simulate(self, params: Dict):
        required_keys = ['eqs', 'eqs_params', 'ts', 'x_0']
        for key in required_keys:
            if key not in params:
                raise ValueError(f"CPG simulate error: Missing required parameter '{key}'.")

        try:
            output = solve_ivp(
                params['eqs'], 
                t_span=params['ts'], 
                y0=params['x_0'], 
                method=INTEGRATION_METHOD, 
                args=(params['eqs_params'],), 
                rtol=5e-2, 
                atol=1e-5
            )
            return np.asarray(output.y[:, -1])
        except Exception as e:
            raise RuntimeError(f"CPG simulation failed: {str(e)}")

    def update_trajectories(self):
        # Update state trajectories
        self.x_traj = np.append(self.x_traj, [self.x_current], axis=0)
        self.q_traj = np.append(self.q_traj, [self.q_current], axis=0)
        self.p_traj = np.append(self.p_traj, [self.p_current], axis=0)
        self.E_traj = np.append(self.E_traj, self.E_current)
        self.t_traj = np.append(self.t_traj, self.t_current)
        
        # Update CPG parameters using current internal state
        # For parameter trajectory, store flattened arrays: [omega1, omega2, ..., mu1, mu2, ...]
        params_update = np.concatenate([self.omega_current, self.mu_current])
        self.params_traj = np.append(self.params_traj, [params_update], axis=0)

    def detect_coiling(self):
        if len(self.x_traj) < 2:
            return self.coils
            
        x_new = self.x_current
        x_old = self.x_traj[-2]

        new_angle = np.arctan2(x_new[1], x_new[0])
        old_angle = np.arctan2(x_old[1], x_old[0])
        
        # Detect phase wrapping
        if (-np.pi / 2 > new_angle) and (old_angle > np.pi / 2):
            self.coils += 1
        elif (-np.pi / 2 > old_angle) and (new_angle > np.pi / 2):
            self.coils -= 1

        return self.coils

    def select_initial(self, params: Dict = None):
        if params is None:
            params = {}

        self.x_0 = np.asarray(params.get('x_0', self.x_0))
        self.x_current = self.x_0.copy()

        self.q_0 = np.zeros(self.num_dof)
        self.q_current = self.q_0.copy()

        self.p_0 = self.get_link_cartesian_positions()
        self.p_current = self.p_0.copy()

    def restart(self, params: Dict = None):
        if params is None:
            params = {}
            
        # Reset time
        self.t_0 = np.asarray([params.get('t_0', 0.0)]).flatten()
        self.t_current = self.t_0.copy()

        # Reset initial conditions
        self.select_initial(params)

        # Reset derived variables
        self.E_0 = np.asarray([sum(self.get_energies())]).flatten()
        self.E_current = self.E_0.copy()

        # Reset trajectories
        self._initialize_trajectories()
        
        # Reset CPG-specific variables - properly sized for multi-DOF
        # Format: [omega1, omega2, ..., mu1, mu2, ...]
        params_cpg = np.zeros(2 * self.num_dof)
        self.params_traj = np.asarray([params_cpg])
        self.coils = 0

        return self.p_0

    def _state_to_joints(self):
        joints = []
        for i in range(self.num_dof):
            if 2*i < len(self.x_current):
                joints.append(self.x_current[2*i])
            else:
                joints.append(0.0)
        return np.asarray(joints)

    # Public getter methods
    def get_joint_state(self):
        return self.q_current.copy()
    
    def get_state(self):
        return self.x_current.copy()
    
    def get_cartesian_state(self):
        mu = getattr(self, 'mu_current')
        omega = getattr(self, 'omega_current')
        params = {'mu': mu, 'omega': omega}
        
        position = self.x_current
        velocity = self.eqs_motion(0, self.x_current, params)
        
        return [position, velocity]

    def get_link_cartesian_positions(self):
        return self.get_cartesian_state()[0]

    # Trajectory getter methods
    def get_state_traj(self):
        return self.x_traj.copy()

    def get_joint_traj(self):
        return self.q_traj.copy()

    def get_cartesian_traj(self):
        return self.p_traj.copy()

    def get_energy_traj(self):
        return self.E_traj.copy()

    def get_temporal_traj(self):
        return self.t_traj.copy()

    def get_parametric_traj(self):
        return self.params_traj.copy()
    
    def get_time(self):
        return self.t_current.copy()

    # Not implemented methods
    def solve(self, t):
        raise NotImplementedError()

    def inverse_kins(self, params: Dict = None):
        raise NotImplementedError()

    def get_energies(self):
        energies = []
        
        # Calculate energy for each oscillator
        for i in range(self.num_dof):
            x1_idx = 2 * i
            x2_idx = 2 * i + 1
            
            if x2_idx < len(self.x_current):
                x1 = self.x_current[x1_idx]
                x2 = self.x_current[x2_idx]
                
                # Simple energy metric: amplitude squared
                energy = x1**2 + x2**2
                energies.append(energy)
            else:
                energies.append(0.0)
        
        return energies
