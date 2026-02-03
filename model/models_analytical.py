"""
Analytical measurement model for IAUKF - matches paper's Eq. 21 exactly.

The measurement function h(x) computes measurements directly from state x,
without running power flow. This is crucial for proper UKF operation.

State vector: x = [V_1, ..., V_n, δ_1, ..., δ_n, R, X]
Measurements: z = [P_inj, Q_inj, V_mag, V_pmu, θ_pmu]

Reference: Paper's Eq. 21
"""
import numpy as np
import copy
import pandapower as pp
from pandapower.pypower.makeYbus import makeYbus
from pandapower.pd2ppc import _pd2ppc


class AnalyticalMeasurementModel:
    """
    FAST analytical measurement model following paper's Eq. 21.

    Pre-computes Ybus with original parameters, then adjusts target line
    contribution based on estimated parameters. Much faster than full Ybus rebuild.
    """

    def __init__(self, net, target_line_idx, pmu_indices):
        self.target_line_idx = target_line_idx
        self.pmu_indices = np.array(pmu_indices)
        self.num_buses = len(net.bus)
        self.state_dim = 2 * self.num_buses + 2

        # Store target line info
        self.from_bus = int(net.line.at[target_line_idx, 'from_bus'])
        self.to_bus = int(net.line.at[target_line_idx, 'to_bus'])
        self.line_length = net.line.at[target_line_idx, 'length_km']
        self.original_r = net.line.at[target_line_idx, 'r_ohm_per_km']
        self.original_x = net.line.at[target_line_idx, 'x_ohm_per_km']

        # Run power flow to initialize network, then get Ybus
        pp.runpp(net, numba=False)
        ppc, _ = _pd2ppc(net)
        self.baseMVA = ppc['baseMVA']
        Ybus_orig, _, _ = makeYbus(ppc['baseMVA'], ppc['bus'], ppc['branch'])

        # Store original G and B
        self.G_orig = Ybus_orig.real.toarray()
        self.B_orig = Ybus_orig.imag.toarray()

        # Get base impedance for per-unit conversion
        vn_kv = net.bus.at[net.ext_grid.bus.values[0], 'vn_kv']
        self.z_base = (vn_kv ** 2) / self.baseMVA

        # Pre-compute original line admittance for subtraction
        r_ohm_orig = self.original_r * self.line_length
        x_ohm_orig = self.original_x * self.line_length
        r_pu_orig = r_ohm_orig / self.z_base
        x_pu_orig = x_ohm_orig / self.z_base
        z_sq_orig = r_pu_orig**2 + x_pu_orig**2
        self.g_orig = r_pu_orig / z_sq_orig
        self.b_orig = -x_pu_orig / z_sq_orig

        # Cache
        self._cached_r = None
        self._cached_x = None
        self._cached_G = None
        self._cached_B = None

    def _get_ybus(self, r_est, x_est):
        """Get G and B by adjusting target line in original Ybus. FAST!"""
        r_round = round(r_est, 8)
        x_round = round(x_est, 8)

        if self._cached_r == r_round and self._cached_x == x_round:
            return self._cached_G, self._cached_B

        # Start with original Ybus
        G = self.G_orig.copy()
        B = self.B_orig.copy()

        # Remove original line contribution
        i, j = self.from_bus, self.to_bus
        G[i,i] -= self.g_orig; G[j,j] -= self.g_orig; G[i,j] += self.g_orig; G[j,i] += self.g_orig
        B[i,i] -= self.b_orig; B[j,j] -= self.b_orig; B[i,j] += self.b_orig; B[j,i] += self.b_orig

        # Add new line contribution
        r_ohm = r_est * self.line_length
        x_ohm = x_est * self.line_length
        r_pu = r_ohm / self.z_base
        x_pu = x_ohm / self.z_base
        z_sq = r_pu**2 + x_pu**2
        if z_sq > 1e-20:
            g_new = r_pu / z_sq
            b_new = -x_pu / z_sq
        else:
            g_new, b_new = 1e10, 0

        G[i,i] += g_new; G[j,j] += g_new; G[i,j] -= g_new; G[j,i] -= g_new
        B[i,i] += b_new; B[j,j] += b_new; B[i,j] -= b_new; B[j,i] -= b_new

        self._cached_r = r_round
        self._cached_x = x_round
        self._cached_G = G
        self._cached_B = B

        return G, B

    def state_transition(self, x):
        """Identity state transition."""
        return x.copy()

    def measurement_function(self, x):
        """Fast analytical measurement function h(x) - Eq. 21."""
        V = np.clip(x[:self.num_buses], 0.8, 1.2)
        delta = x[self.num_buses:2*self.num_buses]
        r_est = max(float(x[-2]), 1e-6)
        x_est = max(float(x[-1]), 1e-6)

        G, B = self._get_ybus(r_est, x_est)

        # Vectorized power injection calculation
        cos_d = np.cos(delta[:, np.newaxis] - delta[np.newaxis, :])
        sin_d = np.sin(delta[:, np.newaxis] - delta[np.newaxis, :])
        VV = V[:, np.newaxis] * V[np.newaxis, :]

        P_inj = np.sum(VV * (G * cos_d + B * sin_d), axis=1) * self.baseMVA
        Q_inj = np.sum(VV * (G * sin_d - B * cos_d), axis=1) * self.baseMVA

        # Combine measurements
        h_scada = np.concatenate([P_inj, Q_inj, V])
        h_pmu = np.concatenate([V[self.pmu_indices], delta[self.pmu_indices]])

        return np.concatenate([h_scada, h_pmu])


class AnalyticalMeasurementModelHolt(AnalyticalMeasurementModel):
    """
    Analytical model with Holt's exponential smoothing for state transition.
    Matches paper's Eq. 19.
    """

    def __init__(self, net, target_line_idx, pmu_indices, alpha_H=0.8, beta_H=0.5):
        super().__init__(net, target_line_idx, pmu_indices)

        # Holt's smoothing parameters
        self.alpha_H = alpha_H
        self.beta_H = beta_H

        # State history for Holt's smoothing
        self.S_prev = None  # Level
        self.b_prev = None  # Trend
        self.x_pred_prev = None  # Previous prediction

    def state_transition(self, x):
        """
        State transition using Holt's exponential smoothing (Eq. 19).

        For voltages/angles:
        x_{k|k-1} = S_{k-1} + b_{k-1}
        S_{k-1} = α_H * x_{k-1} + (1-α_H) * x_{k-1|k-2}
        b_{k-1} = β_H * (S_{k-1} - S_{k-2}) + (1-β_H) * b_{k-2}

        For parameters: p_k = p_{k-1}
        """
        x_states = x[:2*self.num_buses]
        x_params = x[2*self.num_buses:]

        # Initialize on first call
        if self.S_prev is None:
            self.S_prev = x_states.copy()
            self.b_prev = np.zeros_like(x_states)
            self.x_pred_prev = x_states.copy()
            return x.copy()

        # Holt's smoothing (Eq. 19)
        S_new = self.alpha_H * x_states + (1 - self.alpha_H) * self.x_pred_prev
        b_new = self.beta_H * (S_new - self.S_prev) + (1 - self.beta_H) * self.b_prev

        x_states_pred = S_new + b_new

        # Update history
        self.S_prev = S_new
        self.b_prev = b_new
        self.x_pred_prev = x_states_pred.copy()

        return np.concatenate([x_states_pred, x_params])
