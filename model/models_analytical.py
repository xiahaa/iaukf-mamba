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
    Analytical measurement model following paper's Eq. 21.

    This model computes measurements directly from the state vector
    using power system equations, without running power flow.

    Uses pandapower's internal Ybus calculation for correctness.
    """

    def __init__(self, net, target_line_idx, pmu_indices):
        """
        Initialize the model.

        Args:
            net: pandapower network (used to get topology and base parameters)
            target_line_idx: Index of the line whose parameters are being estimated
            pmu_indices: List of bus indices with PMU measurements
        """
        self.net = copy.deepcopy(net)  # Make a copy to avoid modifying original
        self.target_line_idx = target_line_idx
        self.pmu_indices = pmu_indices
        self.num_buses = len(net.bus)

        # State dimension: voltages + angles + 2 parameters
        self.state_dim = 2 * self.num_buses + 2

        # Store original line parameters
        self.original_r = net.line.at[target_line_idx, 'r_ohm_per_km']
        self.original_x = net.line.at[target_line_idx, 'x_ohm_per_km']

        # Get base MVA from pandapower conversion
        ppc, _ = _pd2ppc(net)
        self.baseMVA = ppc['baseMVA']

        # Cache last Ybus to avoid recomputation if parameters unchanged
        self._cached_r = None
        self._cached_x = None
        self._cached_G = None
        self._cached_B = None

    def _get_ybus(self, r_est, x_est):
        """Get G and B matrices with estimated line parameters using pandapower."""
        # Check cache
        if self._cached_r == r_est and self._cached_x == x_est:
            return self._cached_G, self._cached_B

        # Update line parameters in the network copy
        self.net.line.at[self.target_line_idx, 'r_ohm_per_km'] = r_est
        self.net.line.at[self.target_line_idx, 'x_ohm_per_km'] = x_est

        # Use pandapower's Ybus calculation
        ppc, _ = _pd2ppc(self.net)
        Ybus, _, _ = makeYbus(ppc['baseMVA'], ppc['bus'], ppc['branch'])

        # Convert to dense arrays
        G = Ybus.real.toarray()
        B = Ybus.imag.toarray()

        # Cache results
        self._cached_r = r_est
        self._cached_x = x_est
        self._cached_G = G
        self._cached_B = B

        return G, B

    def state_transition(self, x):
        """
        State transition function f(x).
        Identity for parameters, identity for states (random walk).
        """
        return x.copy()

    def measurement_function(self, x):
        """
        Analytical measurement function h(x) following paper's Eq. 21.

        Computes:
        - P_i: Active power injection at bus i
        - Q_i: Reactive power injection at bus i
        - V_i: Voltage magnitude at bus i
        - V_pmu, θ_pmu: PMU voltage magnitude and angle

        Args:
            x: State vector [V_1...V_n, δ_1...δ_n, R, X]

        Returns:
            h: Measurement vector [P_inj, Q_inj, V_scada, V_pmu, θ_pmu]
        """
        # Extract state components
        V = np.array(x[:self.num_buses], dtype=float)
        delta = np.array(x[self.num_buses:2*self.num_buses], dtype=float)
        r_est = max(float(x[-2]), 1e-6)
        x_est = max(float(x[-1]), 1e-6)

        # Clip voltage magnitudes to reasonable range
        V = np.clip(V, 0.8, 1.2)

        # Get G and B matrices with estimated parameters
        G, B = self._get_ybus(r_est, x_est)

        # Compute power injections (Eq. 21 in paper)
        P_inj = np.zeros(self.num_buses)
        Q_inj = np.zeros(self.num_buses)

        for i in range(self.num_buses):
            for j in range(self.num_buses):
                delta_ij = delta[i] - delta[j]
                P_inj[i] += V[i] * V[j] * (G[i, j] * np.cos(delta_ij) + B[i, j] * np.sin(delta_ij))
                Q_inj[i] += V[i] * V[j] * (G[i, j] * np.sin(delta_ij) - B[i, j] * np.cos(delta_ij))

        # Scale to MW (from per-unit)
        P_inj_mw = P_inj * self.baseMVA
        Q_inj_mvar = Q_inj * self.baseMVA

        # SCADA measurements: P_inj, Q_inj, V_mag
        h_scada = np.concatenate([P_inj_mw, Q_inj_mvar, V])

        # PMU measurements: V_mag and θ at PMU buses
        V_pmu = V[self.pmu_indices]
        theta_pmu = delta[self.pmu_indices]
        h_pmu = np.concatenate([V_pmu, theta_pmu])

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
