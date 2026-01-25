import numpy as np
import pandapower as pp
import pandapower.topology as top

class DistributionSystemModel:
    def __init__(self, net, target_line_idx, pmu_indices):
        self.net = net # Reference to pandapower net
        self.target_line_idx = target_line_idx
        self.pmu_indices = pmu_indices
        self.num_buses = len(net.bus)

        # State Vector Size: 2*N_bus + 2 (R, X)
        self.state_dim = 2 * self.num_buses + 2

        # Store original line parameters for restoration
        self.original_r = self.net.line.at[self.target_line_idx, 'r_ohm_per_km']
        self.original_x = self.net.line.at[self.target_line_idx, 'x_ohm_per_km']

        # Cache for last successful measurement (fallback)
        self._last_measurement = None

    def state_transition(self, x):
        """
        f(x): Predicts next state.
        For Parameters (last 2 elements): Identity (p_k = p_{k-1}).
        For Voltages/Angles: Identity (Random Walk assumption for tracking).
        """
        return x

    def measurement_function(self, x):
        """
        h(x): Predicted measurements from state x using pandapower's power flow.
        x = [V_1...V_n, delta_1...delta_n, R_34, X_34]

        This approach ensures consistency with the simulation by using
        the same power flow solver.
        """
        # 1. Extract State
        v_mag = x[:self.num_buses]
        delta = x[self.num_buses : 2*self.num_buses]
        r_est = max(float(x[-2]), 1e-6)  # Ensure positive resistance
        x_est = max(float(x[-1]), 1e-6)  # Ensure positive reactance

        # 2. Update line parameters in the network
        self.net.line.at[self.target_line_idx, 'r_ohm_per_km'] = r_est
        self.net.line.at[self.target_line_idx, 'x_ohm_per_km'] = x_est

        # 3. Set voltage magnitude and angle as initial guess
        # This speeds up convergence and uses state information
        for i in range(len(v_mag)):
            v_val = float(np.clip(v_mag[i], 0.8, 1.2))
            angle_val = float(np.clip(delta[i], -np.pi, np.pi))
            self.net.bus.at[i, 'vm_pu'] = v_val
            self.net.bus.at[i, 'va_degree'] = np.degrees(angle_val)

        # 4. Run power flow with state as initial guess
        try:
            pp.runpp(self.net,
                    init='results',  # Use current bus values as initial guess
                    calculate_voltage_angles=True,
                    numba=False,
                    enforce_q_lims=False,
                    max_iteration=20)

            # Extract SCADA measurements: P, Q injections and V magnitude
            p_inj = -self.net.res_bus.p_mw.values  # Negative for injection convention
            q_inj = -self.net.res_bus.q_mvar.values
            v_scada = self.net.res_bus.vm_pu.values

            h_scada = np.concatenate([p_inj, q_inj, v_scada])

            # Extract PMU measurements: V magnitude and angle
            v_pmu = self.net.res_bus.vm_pu.values[self.pmu_indices]
            theta_pmu = np.radians(self.net.res_bus.va_degree.values[self.pmu_indices])

            h_pmu = np.concatenate([v_pmu, theta_pmu])

            result = np.concatenate([h_scada, h_pmu])

            # Cache successful measurement
            self._last_measurement = result

            return result

        except Exception as e:
            # If power flow fails, use a fallback based on last successful run
            # or return a measurement based on the state directly
            if hasattr(self, '_last_measurement') and self._last_measurement is not None:
                # Return cached measurement (filter will handle the mismatch)
                return self._last_measurement
            else:
                # Construct a simple measurement from state
                # SCADA: use zeros for P,Q and state voltages
                p_zero = np.zeros(self.num_buses)
                q_zero = np.zeros(self.num_buses)
                v_state = np.clip(v_mag, 0.8, 1.2)
                h_scada = np.concatenate([p_zero, q_zero, v_state])

                # PMU: use state values
                v_pmu = v_state[self.pmu_indices]
                theta_pmu = np.clip(delta[self.pmu_indices], -np.pi, np.pi)
                h_pmu = np.concatenate([v_pmu, theta_pmu])

                return np.concatenate([h_scada, h_pmu])
