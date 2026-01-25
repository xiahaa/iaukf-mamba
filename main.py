import numpy as np
import matplotlib.pyplot as plt
from simulation import PowerSystemSimulation
from .model.models import DistributionSystemModel
from .model.iaukf import IAUKF

def main():
    print("--- Starting Augmented State Estimation (IAUKF) ---")

    # 1. Phase 1: Simulation
    steps = 150  # Reduced for faster testing
    sim = PowerSystemSimulation(steps=steps)
    data = sim.run_simulation()

    # 2. Phase 3: Setup Models
    model = DistributionSystemModel(data['net'], data['target_line_idx'], data['pmu_indices'])

    # 3. Initialization
    # Combine measurements
    Z_all = []
    for t in range(steps):
        z_comb = np.concatenate([data['z_scada'][t], data['z_pmu'][t]])
        Z_all.append(z_comb)
    Z_all = np.array(Z_all)

    # Initial State Guess
    # Voltages: Flat start (1.0 pu, 0.0 rad)
    # Parameters: Distorted Guess (0.5 * True)
    x0_v = np.ones(sim.net.bus.shape[0])
    x0_d = np.zeros(sim.net.bus.shape[0])
    x0_r = data['r_true'] * 0.5
    x0_x = data['x_true'] * 0.5

    x0 = np.concatenate([x0_v, x0_d, [x0_r, x0_x]])

    print(f"True Params: R={data['r_true']:.4f}, X={data['x_true']:.4f}")
    print(f"Initial Guess: R={x0_r:.4f}, X={x0_x:.4f}")

    # Covariance Init
    P0 = np.eye(len(x0)) * 0.01

    # Process Noise Init (Small initially, will adapt)
    Q0 = np.eye(len(x0)) * 1e-6
    # Give parameters slightly more noise initially to allow movement
    Q0[-2, -2] = 1e-4
    Q0[-1, -1] = 1e-4

    # Measurement Noise R
    # Construct R matrix based on noise stds
    # Z vector structure: [P_inj(33), Q_inj(33), V_scada(33), V_pmu(12), Theta_pmu(12)]
    n_scada = 33 * 3
    n_pmu = 12 * 2
    R_diag = np.concatenate([
        np.full(33, 0.02**2), # P
        np.full(33, 0.02**2), # Q
        np.full(33, 0.02**2), # V_scada
        np.full(12, 0.005**2), # V_pmu
        np.full(12, 0.002**2)  # Theta_pmu
    ])
    R_cov = np.diag(R_diag)

    # 4. Phase 2: IAUKF Execution
    iaukf = IAUKF(model, x0, P0, Q0, R_cov)

    history_r = []
    history_x = []

    print("Running Filter...")
    for t in range(steps):
        print(f"  Step {t}/{steps}...", end='\r')
        iaukf.predict()
        x_est = iaukf.update(Z_all[t])

        history_r.append(x_est[-2])
        history_x.append(x_est[-1])

        if t % 5 == 0:
            print(f"Step {t:3d}: R_est={x_est[-2]:.4f}, X_est={x_est[-1]:.4f}")

    print("\nFilter completed!")
    print(f"Final: R_est={x_est[-2]:.4f} (true={data['r_true']:.4f}), X_est={x_est[-1]:.4f} (true={data['x_true']:.4f})")

    # 5. Visualization
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history_r, label='Estimated R', linewidth=2)
    plt.axhline(data['r_true'], color='r', linestyle='--', label='True R')
    plt.title('Line Resistance Estimation')
    plt.xlabel('Time Step')
    plt.ylabel('Resistance (Ohm/km)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history_x, label='Estimated X', linewidth=2)
    plt.axhline(data['x_true'], color='r', linestyle='--', label='True X')
    plt.title('Line Reactance Estimation')
    plt.xlabel('Time Step')
    plt.ylabel('Reactance (Ohm/km)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('tmp/iaukf_results.png', dpi=150)
    print("Plot saved as 'iaukf_results.png'")
    plt.close()

    print("Done!")

if __name__ == "__main__":
    main()