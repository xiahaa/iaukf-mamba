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

        # Cache for Ybus to avoid repeated recalculation
        self._ybus_cache = {}
        self._last_params = None

        # Store original line parameters for restoration
        self.original_r = self.net.line.at[self.target_line_idx, 'r_ohm_per_km']
        self.original_x = self.net.line.at[self.target_line_idx, 'x_ohm_per_km']

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
        r_est = max(x[-2], 1e-6)  # Ensure positive resistance
        x_est = max(x[-1], 1e-6)  # Ensure positive reactance

        # 2. Update Grid Parameter in the Net copy (with caching)
        params_key = (round(r_est, 8), round(x_est, 8))

        # Only recalculate Ybus if parameters changed significantly
        if self._last_params is None or \
           abs(params_key[0] - self._last_params[0]) > 1e-7 or \
           abs(params_key[1] - self._last_params[1]) > 1e-7:

            self.net.line.at[self.target_line_idx, 'r_ohm_per_km'] = r_est
            self.net.line.at[self.target_line_idx, 'x_ohm_per_km'] = x_est

            # Build Ybus manually from line parameters (most reliable method)
            try:
                n_bus = len(self.net.bus)
                Ybus = np.zeros((n_bus, n_bus), dtype=complex)

                # Add line admittances
                for _, line in self.net.line.iterrows():
                    fb = int(line.from_bus)
                    tb = int(line.to_bus)
                    r = line.r_ohm_per_km * line.length_km
                    x = line.x_ohm_per_km * line.length_km
                    z = r + 1j * x
                    y = 1.0 / z if abs(z) > 1e-10 else 1e-10
                    # Add shunt capacitance if present
                    if 'c_nf_per_km' in line and not np.isnan(line.c_nf_per_km):
                        b_sh = line.c_nf_per_km * line.length_km * 1e-9 * 2 * np.pi * 50  # Assume 50Hz
                        y_sh = 1j * b_sh / 2  # Split between buses
                        Ybus[fb, fb] += y_sh
                        Ybus[tb, tb] += y_sh
                    # Add series admittance
                    Ybus[fb, fb] += y
                    Ybus[tb, tb] += y
                    Ybus[fb, tb] -= y
                    Ybus[tb, fb] -= y

                # Add transformer admittances if present
                if len(self.net.trafo) > 0:
                    for _, trafo in self.net.trafo.iterrows():
                        hv_bus = int(trafo.hv_bus)
                        lv_bus = int(trafo.lv_bus)
                        # Simplified transformer model
                        vk_percent = trafo.vk_percent if 'vk_percent' in trafo else 6.0
                        sn_mva = trafo.sn_mva if 'sn_mva' in trafo else 0.4
                        z_pu = (vk_percent / 100) * 1j  # Simplified: mostly reactive
                        y_trafo = 1.0 / z_pu if abs(z_pu) > 1e-10 else 1e-10
                        Ybus[hv_bus, hv_bus] += y_trafo
                        Ybus[lv_bus, lv_bus] += y_trafo
                        Ybus[hv_bus, lv_bus] -= y_trafo
                        Ybus[lv_bus, hv_bus] -= y_trafo

                self._ybus_cache[params_key] = Ybus
                self._last_params = params_key

            except Exception as e:
                # Final fallback: use cached version or create identity
                if self._last_params is not None and self._last_params in self._ybus_cache:
                    Ybus = self._ybus_cache[self._last_params]
                else:
                    print(f"Warning: Ybus construction failed: {e}. Using identity matrix.")
                    n_bus = len(self.net.bus)
                    Ybus = np.eye(n_bus, dtype=complex) * 0.01
                    self._ybus_cache[params_key] = Ybus
                    self._last_params = params_key
        else:
            Ybus = self._ybus_cache[params_key]

        # 3. Calculate Power Flow (S = V * conj(Y * V))
        # Construct complex voltage with bounds checking
        v_mag_bounded = np.clip(v_mag, 0.8, 1.2)  # Prevent unrealistic voltages
        delta_bounded = np.clip(delta, -np.pi, np.pi)  # Wrap angles
        V_complex = v_mag_bounded * np.exp(1j * delta_bounded)

        # Current Injection I = Y * V
        I_inj = Ybus.dot(V_complex)

        # Power Injection S = V * conj(I)
        S_inj = V_complex * np.conj(I_inj)

        P_calc = np.real(S_inj)
        Q_calc = np.imag(S_inj)

        # 4. Assemble Predicted Measurements
        # SCADA: P_all, Q_all, V_all
        h_scada = np.concatenate([P_calc, Q_calc, v_mag_bounded])

        # PMU: V_pmu, Theta_pmu
        v_pmu = v_mag_bounded[self.pmu_indices]
        theta_pmu = delta_bounded[self.pmu_indices]
        h_pmu = np.concatenate([v_pmu, theta_pmu])

        return np.concatenate([h_scada, h_pmu])
