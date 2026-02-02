import numpy as np
import pandapower as pp

class DistributionSystemModelHolt:
    """
    Distribution system model with Holt's exponential smoothing.
    Matches the paper's implementation exactly.
    """
    def __init__(self, net, target_line_idx, pmu_indices, alpha_H=0.8, beta_H=0.5):
        self.net = net
        self.target_line_idx = target_line_idx
        self.pmu_indices = pmu_indices
        self.num_buses = len(net.bus)

        # State dimension: voltages + angles + parameters
        self.state_dim = 2 * self.num_buses + 2

        # Holt's smoothing parameters (from paper)
        self.alpha_H = alpha_H
        self.beta_H = beta_H

        # Initialize Holt's smoothing state variables
        self.S_prev = None  # Level (S_{k-2} at call time)
        self.b_prev = None  # Trend (b_{k-2} at call time)
        self.x_pred_prev = None  # Previous one-step prediction (x_{k-1|k-2})

        # Store original parameters
        self.original_r = self.net.line.at[self.target_line_idx, 'r_ohm_per_km']
        self.original_x = self.net.line.at[self.target_line_idx, 'x_ohm_per_km']

    def state_transition(self, x):
        """
        State transition using Holt's dual exponential smoothing (Eq. 19 in paper).

        For voltage states:
        x_{k|k-1} = S_{k-1} + b_{k-1}
        S_{k-1} = α_H * x_{k-1} + (1-α_H) * x_{k-1|k-2}
        b_{k-1} = β_H * (S_{k-1} - S_{k-2}) + (1-β_H) * b_{k-2}

        For parameters: p_k = p_{k-1} (constant)
        """
        # Extract voltage/angle states and parameters
        x_states = x[:2*self.num_buses]  # Voltages and angles
        x_params = x[2*self.num_buses:]   # Parameters

        # First call: initialize with identity prediction
        if self.S_prev is None:
            self.S_prev = x_states.copy()
            self.b_prev = np.zeros_like(x_states)
            # Use current state as the previous one-step prediction
            self.x_pred_prev = x_states.copy()
            return x  # Identity for first step

        # Holt's smoothing for voltage/angle states (Eq. 19)
        # S_{k-1} = alpha * x_{k-1} + (1-alpha) * x_{k-1|k-2}
        S_new = self.alpha_H * x_states + (1 - self.alpha_H) * self.x_pred_prev
        # b_{k-1} = beta * (S_{k-1} - S_{k-2}) + (1-beta) * b_{k-2}
        b_new = self.beta_H * (S_new - self.S_prev) + (1 - self.beta_H) * self.b_prev

        x_states_pred = S_new + b_new

        # Update history
        self.S_prev = S_new
        self.b_prev = b_new
        # Store current prediction for next call
        self.x_pred_prev = x_states_pred.copy()

        # Parameters remain constant
        return np.concatenate([x_states_pred, x_params])

    def measurement_function(self, x):
        """
        Measurement function h(x).
        Same as before - use power flow to compute measurements.
        """
        v_mag = x[:self.num_buses]
        delta = x[self.num_buses:2*self.num_buses]
        r_est = max(float(x[-2]), 1e-6)
        x_est = max(float(x[-1]), 1e-6)

        # Update line parameters
        self.net.line.at[self.target_line_idx, 'r_ohm_per_km'] = r_est
        self.net.line.at[self.target_line_idx, 'x_ohm_per_km'] = x_est

        # Set voltage initial guess for power flow
        for i in range(len(v_mag)):
            self.net.bus.at[i, 'vm_pu'] = float(np.clip(v_mag[i], 0.8, 1.2))
            self.net.bus.at[i, 'va_degree'] = float(np.degrees(np.clip(delta[i], -np.pi, np.pi)))

        # Run power flow
        try:
            pp.runpp(self.net,
                    init='results',
                    calculate_voltage_angles=True,
                    numba=False,
                    enforce_q_lims=False,
                    max_iteration=20)

            # SCADA measurements
            p_inj = -self.net.res_bus.p_mw.values
            q_inj = -self.net.res_bus.q_mvar.values
            v_scada = self.net.res_bus.vm_pu.values
            h_scada = np.concatenate([p_inj, q_inj, v_scada])

            # PMU measurements
            v_pmu = self.net.res_bus.vm_pu.values[self.pmu_indices]
            theta_pmu = np.radians(self.net.res_bus.va_degree.values[self.pmu_indices])
            h_pmu = np.concatenate([v_pmu, theta_pmu])

            return np.concatenate([h_scada, h_pmu])

        except Exception as e:
            print(f"Warning: Power flow failed: {e}")
            # Return zeros as fallback
            n_scada = 3 * self.num_buses
            n_pmu = 2 * len(self.pmu_indices)
            return np.zeros(n_scada + n_pmu)


# Keep the original simple model as well
from model.models import DistributionSystemModel
