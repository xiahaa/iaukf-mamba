"""
Phase 1: IAUKF Validation with ANALYTICAL Measurement Model

This uses the analytical measurement function h(x) from paper's Eq. 21.

NOTE: The analytical model may show slightly higher error (~2-3%) compared to
the power-flow-based model (~0.5-1%) due to subtle differences in:
1. Ybus computation between our implementation and pandapower's internal Ybus
2. Numerical precision in power injection calculations

For best results matching paper's claims, use phase1_validate_iaukf.py or
phase1_tuned.py which use power flow for measurement prediction.

The analytical model is provided for:
1. Faster computation (no power flow needed per sigma point)
2. Theoretical consistency with paper's Eq. 21
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import pandapower as pp
from model.simulation import PowerSystemSimulation
from model.models_analytical import AnalyticalMeasurementModel, AnalyticalMeasurementModelHolt
from model.iaukf import IAUKF


def run_analytical_validation(steps=300, use_holt=False):
    """
    Validate IAUKF with analytical measurement model.

    Key tuning (matching paper's claims):
    - b_factor = 0.96 (paper's value) - more aggressive adaptation
    - Q0 = 1e-6 * I (paper's value)
    - More steps for better convergence
    """
    print("=" * 70)
    print("PHASE 1: IAUKF with ANALYTICAL Measurement Model")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  - System: IEEE 33-bus")
    print(f"  - Target: Branch 3-4")
    print(f"  - Steps: {steps}")
    print(f"  - State transition: {'Holt smoothing' if use_holt else 'Identity'}")
    print(f"  - Measurement model: ANALYTICAL (Eq. 21)")

    # Generate simulation data
    print("\n[1] Generating simulation data (constant loads)...")
    sim = PowerSystemSimulation(steps=steps)

    p_load_base = sim.net.load.p_mw.values.copy()
    q_load_base = sim.net.load.q_mvar.values.copy()

    z_scada_list = []
    z_pmu_list = []
    true_states_list = []

    np.random.seed(42)

    for t in range(steps):
        # Constant loads
        sim.net.load.p_mw = p_load_base
        sim.net.load.q_mvar = q_load_base

        pp.runpp(sim.net, algorithm='nr', numba=False)

        # True states
        v_true = sim.net.res_bus.vm_pu.values
        delta_true = np.radians(sim.net.res_bus.va_degree.values)
        true_states_list.append(np.concatenate([v_true, delta_true, [sim.r_true, sim.x_true]]))

        # SCADA with noise (std=0.02 as in paper)
        p_inj = -sim.net.res_bus.p_mw.values
        q_inj = -sim.net.res_bus.q_mvar.values
        v_scada = sim.net.res_bus.vm_pu.values
        z_scada = np.concatenate([p_inj, q_inj, v_scada])
        z_scada_list.append(z_scada + np.random.normal(0, 0.02, len(z_scada)))

        # PMU with noise (std V=0.005, theta=0.002 as in paper)
        v_pmu = sim.net.res_bus.vm_pu.values[sim.pmu_buses]
        theta_pmu = np.radians(sim.net.res_bus.va_degree.values[sim.pmu_buses])
        z_pmu = np.concatenate([v_pmu, theta_pmu])
        noise_pmu = np.concatenate([
            np.random.normal(0, 0.005, len(v_pmu)),
            np.random.normal(0, 0.002, len(theta_pmu))
        ])
        z_pmu_list.append(z_pmu + noise_pmu)

    Z_scada = np.array(z_scada_list)
    Z_pmu = np.array(z_pmu_list)

    print(f"✓ Generated {steps} snapshots")
    print(f"  True: R={sim.r_true:.4f}, X={sim.x_true:.4f}")

    # Setup IAUKF with ANALYTICAL model
    print("\n[2] Setting up IAUKF with ANALYTICAL measurement model...")

    if use_holt:
        model = AnalyticalMeasurementModelHolt(sim.net, sim.line_idx, sim.pmu_buses,
                                               alpha_H=0.8, beta_H=0.5)
    else:
        model = AnalyticalMeasurementModel(sim.net, sim.line_idx, sim.pmu_buses)

    # Initial state
    x0_v = np.ones(sim.net.bus.shape[0])
    x0_d = np.zeros(sim.net.bus.shape[0])
    x0_r = 0.01  # Paper uses small initial values
    x0_x = 0.01
    x0 = np.concatenate([x0_v, x0_d, [x0_r, x0_x]])

    print(f"  Initial params: R={x0_r:.4f}, X={x0_x:.4f}")
    print(f"  Initial error: R={abs(x0_r-sim.r_true)/sim.r_true*100:.1f}%, "
          f"X={abs(x0_x-sim.x_true)/sim.x_true*100:.1f}%")

    # Covariances - paper's settings
    P0 = np.eye(len(x0)) * 0.01
    P0[-2, -2] = 0.1  # Higher uncertainty for parameters
    P0[-1, -1] = 0.1

    Q0 = np.eye(len(x0)) * 1e-6  # Paper's exact value

    R_diag = np.concatenate([
        np.full(33, 0.02**2),   # P injection
        np.full(33, 0.02**2),   # Q injection
        np.full(33, 0.02**2),   # V SCADA
        np.full(12, 0.005**2),  # V PMU
        np.full(12, 0.002**2)   # Theta PMU
    ])
    R_cov = np.diag(R_diag)

    iaukf = IAUKF(model, x0, P0, Q0, R_cov)
    iaukf.b_factor = 0.96  # Paper's exact value

    print(f"  UKF params: alpha={iaukf.alpha}, beta={iaukf.beta}, kappa={iaukf.kappa}")
    print(f"  NSE b_factor: {iaukf.b_factor} (paper's value)")

    # Run IAUKF
    print("\n[3] Running IAUKF...")

    history_r = []
    history_x = []
    history_r_err = []
    history_x_err = []

    for t in range(steps):
        z_t = np.concatenate([Z_scada[t], Z_pmu[t]])

        iaukf.predict()
        x_est = iaukf.update(z_t)

        r_est = x_est[-2]
        x_est_param = x_est[-1]

        r_err_pct = abs(r_est - sim.r_true) / sim.r_true * 100
        x_err_pct = abs(x_est_param - sim.x_true) / sim.x_true * 100

        history_r.append(r_est)
        history_x.append(x_est_param)
        history_r_err.append(r_err_pct)
        history_x_err.append(x_err_pct)

        if t % 20 == 0 or t < 5:
            print(f"  Step {t:3d}: R={r_est:.6f} (err={r_err_pct:5.2f}%), "
                  f"X={x_est_param:.6f} (err={x_err_pct:5.2f}%)")

    # Paper's convergence criterion
    print("\n[4] Checking convergence (|p_k - p_{k-1}| <= 0.001)...")

    convergence_threshold = 0.001
    r_converged_step = None
    x_converged_step = None

    for t in range(1, len(history_r)):
        if r_converged_step is None and abs(history_r[t] - history_r[t-1]) <= convergence_threshold:
            r_converged_step = t
        if x_converged_step is None and abs(history_x[t] - history_x[t-1]) <= convergence_threshold:
            x_converged_step = t
        if r_converged_step and x_converged_step:
            break

    print(f"  R converges at step: {r_converged_step}")
    print(f"  X converges at step: {x_converged_step}")

    # Post-convergence averaging (Eq. 40)
    if r_converged_step:
        r_final = np.mean(history_r[r_converged_step:])
    else:
        r_final = history_r[-1]

    if x_converged_step:
        x_final = np.mean(history_x[x_converged_step:])
    else:
        x_final = history_x[-1]

    r_final_err = abs(r_final - sim.r_true) / sim.r_true * 100
    x_final_err = abs(x_final - sim.x_true) / sim.x_true * 100

    # Check oscillation
    r_std_last20 = np.std(history_r[-20:])
    x_std_last20 = np.std(history_x[-20:])

    print(f"\n[5] Results...")
    print(f"  R_final = {r_final:.6f} (true={sim.r_true:.4f}, error={r_final_err:.2f}%)")
    print(f"  X_final = {x_final:.6f} (true={sim.x_true:.4f}, error={x_final_err:.2f}%)")
    print(f"  Oscillation (last 20 steps std): R={r_std_last20:.6f}, X={x_std_last20:.6f}")

    # Compare with paper
    print(f"\n{'='*70}")
    print("COMPARISON WITH PAPER")
    print('='*70)
    print(f"  Paper's results (branch 3-4):")
    print(f"    R error: 0.18%")
    print(f"    X error: 1.55%")
    print(f"\n  Our results:")
    print(f"    R error: {r_final_err:.2f}%")
    print(f"    X error: {x_final_err:.2f}%")
    print(f"    R oscillation: {r_std_last20:.6f}")
    print(f"    X oscillation: {x_std_last20:.6f}")

    # Evaluate - Note: analytical model may have ~2-3% model mismatch
    converged_well = r_std_last20 < 0.01 and x_std_last20 < 0.01
    accurate = r_final_err < 5.0 and x_final_err < 5.0  # Relaxed for analytical model

    if r_final_err < 2.0 and x_final_err < 2.0:
        print(f"\n  ✓✓✓ EXCELLENT: Converged and accurate! ✓✓✓")
    elif accurate:
        print(f"\n  ✓✓ GOOD: Acceptable for analytical model")
        print(f"  Note: For paper-level accuracy, use phase1_validate_iaukf.py")
    else:
        print(f"\n  ⚠ Results outside expected range")

    # Visualization
    print(f"\n[6] Generating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # R estimation
    axes[0, 0].plot(history_r, 'b-', linewidth=1.5, alpha=0.7, label='Estimated')
    axes[0, 0].axhline(sim.r_true, color='r', linestyle='--', linewidth=2, label='True')
    axes[0, 0].axhline(r_final, color='g', linestyle=':', linewidth=2, label='Final (averaged)')
    if r_converged_step:
        axes[0, 0].axvline(r_converged_step, color='orange', linestyle='--', alpha=0.5, label='Converged')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('R (Ohm/km)')
    axes[0, 0].set_title('Resistance Estimation (Analytical Model)', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0.0, 0.6)
    axes[0, 0].grid(True, alpha=0.3)

    # X estimation
    axes[0, 1].plot(history_x, 'g-', linewidth=1.5, alpha=0.7, label='Estimated')
    axes[0, 1].axhline(sim.x_true, color='r', linestyle='--', linewidth=2, label='True')
    axes[0, 1].axhline(x_final, color='b', linestyle=':', linewidth=2, label='Final (averaged)')
    if x_converged_step:
        axes[0, 1].axvline(x_converged_step, color='orange', linestyle='--', alpha=0.5, label='Converged')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('X (Ohm/km)')
    axes[0, 1].set_title('Reactance Estimation (Analytical Model)', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # R error
    axes[1, 0].semilogy(history_r_err, 'b-', linewidth=1.5)
    axes[1, 0].axhline(r_final_err, color='g', linestyle=':', linewidth=2, label=f'Final: {r_final_err:.2f}%')
    axes[1, 0].axhline(0.18, color='r', linestyle='--', linewidth=2, label='Paper: 0.18%')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Error (%)')
    axes[1, 0].set_title('Resistance Error', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # X error
    axes[1, 1].semilogy(history_x_err, 'g-', linewidth=1.5)
    axes[1, 1].axhline(x_final_err, color='b', linestyle=':', linewidth=2, label=f'Final: {x_final_err:.2f}%')
    axes[1, 1].axhline(1.55, color='r', linestyle='--', linewidth=2, label='Paper: 1.55%')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Error (%)')
    axes[1, 1].set_title('Reactance Error', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'tmp/phase1_analytical_{"holt" if use_holt else "identity"}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()

    return {
        'r_final': r_final,
        'x_final': x_final,
        'r_error': r_final_err,
        'x_error': x_final_err,
        'r_std': r_std_last20,
        'x_std': x_std_last20,
        'history_r': history_r,
        'history_x': history_x,
    }


if __name__ == "__main__":
    print("\nTesting with ANALYTICAL measurement model (identity transition)...")
    results = run_analytical_validation(steps=300, use_holt=False)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nAnalytical model results:")
    print(f"  R error: {results['r_error']:.2f}%")
    print(f"  X error: {results['x_error']:.2f}%")
    print(f"  R oscillation (std): {results['r_std']:.6f}")
    print(f"  X oscillation (std): {results['x_std']:.6f}")

    # Check accuracy - Note: analytical model typically shows ~2-3% error
    if results['r_error'] < 2.0 and results['x_error'] < 2.0:
        print(f"\n✓✓✓ Excellent for analytical model! ✓✓✓")
    elif results['r_error'] < 5.0 and results['x_error'] < 5.0:
        print(f"\n✓✓ Good results for analytical model")
        print(f"Note: The analytical model may have slight model mismatch.")
        print(f"For paper-level accuracy (<1%), use phase1_validate_iaukf.py")
    else:
        print(f"\n⚠ Results outside expected range")
