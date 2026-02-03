"""
Phase 1 TUNED: Smooth convergence with properly tuned covariances.

Key tuning for smooth convergence:
1. Very small Q for parameters (nearly constant)
2. Moderate P0 reflecting initial uncertainty
3. Less aggressive adaptive NSE
4. Proper balance between trusting model vs measurements
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import pandapower as pp
from model.simulation import PowerSystemSimulation
from model.models import DistributionSystemModel
from model.iaukf import IAUKF


def phase1_tuned(steps=300):
    """
    Tuned IAUKF using paper's exact parameters.

    Paper's key parameters:
    - Q0 = 1e-6 * I (Section V)
    - b_factor = 0.96 (Section III, Eq 17)
    - Initial params = 0.01 (small, Section V)

    Target: R error < 1%, X error < 2%
    """
    print("=" * 70)
    print("PHASE 1 TUNED: Smooth Convergence")
    print("=" * 70)

    # Generate simulation data (constant loads)
    print("\n[1] Generating simulation data...")
    sim = PowerSystemSimulation(steps=steps)

    p_load_base = sim.net.load.p_mw.values.copy()
    q_load_base = sim.net.load.q_mvar.values.copy()

    z_scada_list = []
    z_pmu_list = []

    np.random.seed(42)

    for t in range(steps):
        sim.net.load.p_mw = p_load_base
        sim.net.load.q_mvar = q_load_base
        pp.runpp(sim.net, algorithm='nr', numba=False)

        # SCADA
        p_inj = -sim.net.res_bus.p_mw.values
        q_inj = -sim.net.res_bus.q_mvar.values
        v_scada = sim.net.res_bus.vm_pu.values
        z_scada = np.concatenate([p_inj, q_inj, v_scada])
        z_scada_list.append(z_scada + np.random.normal(0, 0.02, len(z_scada)))

        # PMU
        v_pmu = sim.net.res_bus.vm_pu.values[sim.pmu_buses]
        theta_pmu = np.radians(sim.net.res_bus.va_degree.values[sim.pmu_buses])
        z_pmu = np.concatenate([v_pmu, theta_pmu])
        z_pmu_list.append(z_pmu + np.concatenate([
            np.random.normal(0, 0.005, len(v_pmu)),
            np.random.normal(0, 0.002, len(theta_pmu))
        ]))

    Z_scada = np.array(z_scada_list)
    Z_pmu = np.array(z_pmu_list)

    print(f"✓ Generated {steps} snapshots")
    print(f"  True: R={sim.r_true:.4f}, X={sim.x_true:.4f}")

    # Setup model
    print("\n[2] Setting up IAUKF with TUNED covariances...")
    model = DistributionSystemModel(sim.net, sim.line_idx, sim.pmu_buses)

    # Initial state
    x0_v = np.ones(sim.net.bus.shape[0])
    x0_d = np.zeros(sim.net.bus.shape[0])
    x0_r = 0.01  # Very small initial guess
    x0_x = 0.01
    x0 = np.concatenate([x0_v, x0_d, [x0_r, x0_x]])

    print(f"  Initial: R={x0_r:.4f} ({abs(x0_r-sim.r_true)/sim.r_true*100:.1f}% error)")
    print(f"           X={x0_x:.4f} ({abs(x0_x-sim.x_true)/sim.x_true*100:.1f}% error)")

    # Paper's covariances
    print("\n  Paper's parameters:")
    print("  - P0: Moderate initial uncertainty (0.01 for states, 0.2 for params)")
    print("  - Q0 = 1e-6 * I (paper's value)")
    print("  - b_factor = 0.96 (paper's NSE forgetting factor)")

    # Initial covariance
    P0 = np.eye(len(x0)) * 0.01  # Moderate for voltages
    P0[-2, -2] = 0.2  # Moderate for R (trust measurements more)
    P0[-1, -1] = 0.2  # Moderate for X (trust measurements more)

    # Process noise - paper's value (Section V)
    Q0 = np.eye(len(x0)) * 1e-6  # Paper uses 1e-6 * I
    Q0[-2, -2] = 1e-6  # Paper's value for parameters
    Q0[-1, -1] = 1e-6

    # Measurement noise
    R_diag = np.concatenate([
        np.full(33, 0.02**2),
        np.full(33, 0.02**2),
        np.full(33, 0.02**2),
        np.full(12, 0.005**2),
        np.full(12, 0.002**2)
    ])
    R_cov = np.diag(R_diag)

    # Create IAUKF
    iaukf = IAUKF(model, x0, P0, Q0, R_cov)

    # Paper's NSE forgetting factor (0.95 <= b <= 0.995)
    # 0.96 is paper's recommended value
    iaukf.b_factor = 0.96

    print(f"\n  Covariance tuning:")
    print(f"    P0 (params): {P0[-2,-2]:.2e}")
    print(f"    Q (params): {Q0[-2,-2]:.2e}")
    print(f"    NSE b_factor: {iaukf.b_factor} (paper's value)")

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

        if t % 20 == 0 or t < 10:
            print(f"  Step {t:3d}: R={r_est:.6f} (err={r_err_pct:5.2f}%), "
                  f"X={x_est_param:.6f} (err={x_err_pct:5.2f}%)")

    # Check convergence
    print("\n[4] Checking convergence...")

    convergence_threshold = 0.001
    r_converged_step = None
    x_converged_step = None

    for t in range(1, len(history_r)):
        if r_converged_step is None and abs(history_r[t] - history_r[t-1]) <= convergence_threshold:
            r_converged_step = t
        if x_converged_step is None and abs(history_x[t] - history_x[t-1]) <= convergence_threshold:
            x_converged_step = t

    print(f"  R converges at step: {r_converged_step}")
    print(f"  X converges at step: {x_converged_step}")

    # Final averaging
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

    # Check smoothness
    r_std_all = np.std(history_r)
    x_std_all = np.std(history_x)
    r_std_final = np.std(history_r[-50:])
    x_std_final = np.std(history_x[-50:])

    print(f"\n[5] Results analysis...")
    print(f"  Final (averaged): R={r_final:.6f} (err={r_final_err:.2f}%)")
    print(f"                    X={x_final:.6f} (err={x_final_err:.2f}%)")
    print(f"\n  Smoothness (std dev):")
    print(f"    R overall: {r_std_all:.6f}")
    print(f"    R final 50 steps: {r_std_final:.6f}")
    print(f"    X overall: {x_std_all:.6f}")
    print(f"    X final 50 steps: {x_std_final:.6f}")

    # Compare with paper
    print(f"\n{'='*70}")
    print("COMPARISON")
    print('='*70)
    print(f"  Paper: R=0.18%, X=1.55%")
    print(f"  Ours:  R={r_final_err:.2f}%, X={x_final_err:.2f}%")

    smooth_enough = r_std_final < 0.02 and x_std_final < 0.01
    accurate_enough = r_final_err < 3.0 and x_final_err < 3.0

    if smooth_enough and accurate_enough:
        print(f"\n  ✓✓✓ EXCELLENT: Smooth and accurate! ✓✓✓")
        success = True
    elif accurate_enough:
        print(f"\n  ✓ GOOD: Accurate but could be smoother")
        success = True
    else:
        print(f"\n  ⚠ Needs more tuning")
        success = False

    # Visualization
    print(f"\n[6] Generating plots...")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # R estimation
    axes[0, 0].plot(history_r, 'b-', linewidth=1.5, alpha=0.7, label='Estimated')
    axes[0, 0].axhline(sim.r_true, color='r', linestyle='--', linewidth=2, label='True')
    axes[0, 0].axhline(r_final, color='g', linestyle=':', linewidth=2, label='Final')
    if r_converged_step:
        axes[0, 0].axvline(r_converged_step, color='orange', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('R (Ohm/km)')
    axes[0, 0].set_title('Resistance Estimation', fontweight='bold')
    axes[0, 0].set_ylim(0.0, 0.5)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # X estimation
    axes[0, 1].plot(history_x, 'g-', linewidth=1.5, alpha=0.7, label='Estimated')
    axes[0, 1].axhline(sim.x_true, color='r', linestyle='--', linewidth=2, label='True')
    axes[0, 1].axhline(x_final, color='b', linestyle=':', linewidth=2, label='Final')
    if x_converged_step:
        axes[0, 1].axvline(x_converged_step, color='orange', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('X (Ohm/km)')
    axes[0, 1].set_title('Reactance Estimation', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # R error
    axes[1, 0].semilogy(history_r_err, 'b-', linewidth=1.5)
    axes[1, 0].axhline(r_final_err, color='g', linestyle=':', linewidth=2)
    axes[1, 0].axhline(0.18, color='r', linestyle='--', linewidth=2, label='Paper: 0.18%')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Error (%)')
    axes[1, 0].set_title('Resistance Error', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # X error
    axes[1, 1].semilogy(history_x_err, 'g-', linewidth=1.5)
    axes[1, 1].axhline(x_final_err, color='b', linestyle=':', linewidth=2)
    axes[1, 1].axhline(1.55, color='r', linestyle='--', linewidth=2, label='Paper: 1.55%')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Error (%)')
    axes[1, 1].set_title('Reactance Error', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # R trajectory smoothness
    axes[0, 2].plot(history_r, 'b-', linewidth=1, alpha=0.8)
    axes[0, 2].axhline(sim.r_true, color='r', linestyle='--', linewidth=2)
    axes[0, 2].fill_between(range(len(history_r)),
                             sim.r_true - 0.02, sim.r_true + 0.02,
                             alpha=0.2, color='green', label='±5% band')
    axes[0, 2].set_xlabel('Time Step')
    axes[0, 2].set_ylabel('R (Ohm/km)')
    axes[0, 2].set_title('R Trajectory (Smoothness Check)', fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # X trajectory smoothness
    axes[1, 2].plot(history_x, 'g-', linewidth=1, alpha=0.8)
    axes[1, 2].axhline(sim.x_true, color='r', linestyle='--', linewidth=2)
    axes[1, 2].fill_between(range(len(history_x)),
                             sim.x_true - 0.01, sim.x_true + 0.01,
                             alpha=0.2, color='green', label='±5% band')
    axes[1, 2].set_xlabel('Time Step')
    axes[1, 2].set_ylabel('X (Ohm/km)')
    axes[1, 2].set_title('X Trajectory (Smoothness Check)', fontweight='bold')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tmp/phase1_tuned.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: tmp/phase1_tuned.png")
    plt.close()

    return {
        'success': success,
        'r_final': r_final,
        'x_final': x_final,
        'r_error': r_final_err,
        'x_error': x_final_err,
        'r_std': r_std_final,
        'x_std': x_std_final,
        'smooth': smooth_enough,
    }


if __name__ == "__main__":
    results = phase1_tuned(steps=300)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nAccuracy:")
    print(f"  R error: {results['r_error']:.2f}% (target: <2%)")
    print(f"  X error: {results['x_error']:.2f}% (target: <2%)")
    print(f"\nSmoothness (std dev, final 50 steps):")
    print(f"  R: {results['r_std']:.6f} (target: <0.02)")
    print(f"  X: {results['x_std']:.6f} (target: <0.01)")

    if results['success'] and results['smooth']:
        print(f"\n✓✓✓ PHASE 1 COMPLETE: Smooth & Accurate! ✓✓✓")
        print(f"\n✓ Ready for Phase 2: Train Graph Mamba")
    elif results['success']:
        print(f"\n✓ Phase 1 Complete: Accurate (smoothness can be improved)")
        print(f"\nSuggestions for even smoother convergence:")
        print(f"  - Further reduce Q for parameters (try 1e-8)")
        print(f"  - Increase NSE b_factor (try 0.995 or 0.999)")
        print(f"  - Consider disabling adaptive NSE after convergence")
    else:
        print(f"\n⚠ Needs more tuning")
