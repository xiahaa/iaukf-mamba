import pandapower as pp
import pandapower.networks as nw
import numpy as np
import copy

class PowerSystemSimulation:
    def __init__(self, steps=200):
        self.steps = steps
        # Load IEEE 33-bus system
        self.net = nw.case33bw()

        # Target Branch 3-4 (Internal index usually 2, but we find it dynamically)
        # Buses in pandapower are 0-indexed. Bus 3 is index 3, Bus 4 is index 4.
        self.line_idx = self.net.line[(self.net.line.from_bus == 3) & (self.net.line.to_bus == 4)].index[0]

        # Store TRUE parameters
        self.r_true = self.net.line.at[self.line_idx, 'r_ohm_per_km']
        self.x_true = self.net.line.at[self.line_idx, 'x_ohm_per_km']
        self.length = self.net.line.at[self.line_idx, 'length_km']

        # We estimate the TOTAL R and X (R_ohm, X_ohm), not per km,
        # but pandapower uses per_km * length.
        # For the state vector, we will track the raw ohm/km value for simplicity
        # or the total. Let's track the parameter as defined in the dataframe (ohm_per_km).

        print(f"Target Line 3-4 (Index {self.line_idx}): R_true={self.r_true:.4f}, X_true={self.x_true:.4f}")

        # PMU locations (0-indexed): 3, 6, 9, 11, 14, 17, 19, 22, 24, 26, 29, 32
        self.pmu_buses = [3, 6, 9, 11, 14, 17, 19, 22, 24, 26, 29, 32]

    def run_simulation(self):
        """Generates ground truth states and noisy measurements."""
        np.random.seed(42)

        z_scada_list = []
        z_pmu_list = []
        true_states_list = []

        # Base loads
        p_load_base = self.net.load.p_mw.values.copy()
        q_load_base = self.net.load.q_mvar.values.copy()

        for t in range(self.steps):
            # 1. Perturb loads (Random fluctuation +/- 10%)
            fluctuation = np.random.uniform(0.9, 1.1, size=len(p_load_base))
            self.net.load.p_mw = p_load_base * fluctuation
            self.net.load.q_mvar = q_load_base * fluctuation

            # 2. Run Power Flow
            try:
                pp.runpp(self.net, algorithm='nr')
            except:
                print(f"Power flow failed at step {t}")
                continue

            # 3. Collect True States [V_mag (33), Delta (33), R, X]
            # Convert delta from degrees to radians for internal logic if needed,
            # but usually power systems use degrees in output. Let's stick to PU and Radians for calc.
            v_true = self.net.res_bus.vm_pu.values
            delta_true = np.radians(self.net.res_bus.va_degree.values) # Convert to radians

            true_states_list.append(np.concatenate([v_true, delta_true, [self.r_true, self.x_true]]))

            # 4. Generate Measurements

            # --- SCADA (All Buses) ---
            # P, Q injections + V magnitude
            # Noise: 0.02 pu/mw
            p_inj = -self.net.res_bus.p_mw.values # Injection = Generation - Load
            q_inj = -self.net.res_bus.q_mvar.values
            v_scada = self.net.res_bus.vm_pu.values

            z_scada = np.concatenate([p_inj, q_inj, v_scada])
            noise_scada = np.random.normal(0, 0.02, size=len(z_scada))
            z_scada_noisy = z_scada + noise_scada
            z_scada_list.append(z_scada_noisy)

            # --- PMU (Specific Buses) ---
            # V magnitude, Theta angle
            # Noise: V=0.005, Theta=0.002 rad
            v_pmu = self.net.res_bus.vm_pu.values[self.pmu_buses]
            theta_pmu = np.radians(self.net.res_bus.va_degree.values[self.pmu_buses])

            z_pmu = np.concatenate([v_pmu, theta_pmu])
            # Construct noise vector
            noise_pmu_v = np.random.normal(0, 0.005, size=len(v_pmu))
            noise_pmu_theta = np.random.normal(0, 0.002, size=len(theta_pmu))
            z_pmu_noisy = z_pmu + np.concatenate([noise_pmu_v, noise_pmu_theta])
            z_pmu_list.append(z_pmu_noisy)

        return {
            "z_scada": np.array(z_scada_list),
            "z_pmu": np.array(z_pmu_list),
            "true_states": np.array(true_states_list),
            "net": self.net,
            "target_line_idx": self.line_idx,
            "pmu_indices": self.pmu_buses,
            "r_true": self.r_true,
            "x_true": self.x_true
        }