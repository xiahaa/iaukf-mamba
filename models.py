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

        # Holt's Smoothing parameters
        self.alpha_h = 0.8
        self.beta_h = 0.5

        # Initialize trend (b) and level (S) if we were strictly implementing Eq 19 recursion.
        # However, for the UKF sigma-point transition f(x), we typically use a
        # Markovian assumption x_k = x_{k-1} or x_k = x_{k-1} + trend.
        # Here we use Identity transition for robust tracking.
        # The adaptive Q in IAUKF handles the dynamics.

    def state_transition(self, x):
        """
        f(x): Predicts next state.
        For Parameters (last 2 elements): Identity (p_k = p_{k-1}).
        For Voltages/Angles: Identity (Random Walk assumption for tracking).
        """
        return x

    def measurement_function(self, x):
        """
        h(x): Predicted measurements from state x.
        x = [V_1...V_n, delta_1...delta_n, R_34, X_34]
        """
        # 1. Extract State
        v_mag = x[:self.num_buses]
        delta = x[self.num_buses : 2*self.num_buses]
        r_est = x[-2]
        x_est = x[-1]

        # 2. Update Grid Parameter in the Net copy
        # We need to construct Ybus with the dynamic R/X.
        # Modifying net.line in a loop is slow, but necessary for the generic solver.
        # To optimize, we could perform local Ybus update.
        # For IEEE 33 (small), full Ybus rebuild is acceptable (~few ms).

        self.net.line.at[self.target_line_idx, 'r_ohm_per_km'] = r_est
        self.net.line.at[self.target_line_idx, 'x_ohm_per_km'] = x_est

        # Recalculate Ybus (pp.makeYbus is internally cached, need to trigger update)
        # We force update by recalculating.
        Ybus, _ = pp.makeYbus(self.net, calc_line_parameter=True, check_connectivity=False)

        # 3. Calculate Power Flow (S = V * conj(Y * V))
        # Construct complex voltage
        V_complex = v_mag * np.exp(1j * delta)

        # Current Injection I = Y * V
        I_inj = Ybus.dot(V_complex)

        # Power Injection S = V * conj(I)
        S_inj = V_complex * np.conj(I_inj)

        P_calc = np.real(S_inj)
        Q_calc = np.imag(S_inj) # Pandapower uses Generation - Load convention or Load convention?
        # In simulation.py we used -p_mw (Injection).
        # S_inj here is Injection into the bus. So it matches.

        # 4. Assemble Predicted Measurements

        # SCADA: P_all, Q_all, V_all
        h_scada = np.concatenate([P_calc, Q_calc, v_mag])

        # PMU: V_pmu, Theta_pmu
        v_pmu = v_mag[self.pmu_indices]
        theta_pmu = delta[self.pmu_indices]
        h_pmu = np.concatenate([v_pmu, theta_pmu])

        return np.concatenate([h_scada, h_pmu])
