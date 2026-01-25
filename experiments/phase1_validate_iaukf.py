"""
Phase 1: Validate IAUKF Implementation
Reproduce the paper's results with constant loads.
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


def phase1_validate_iaukf(steps=150, save_results=True):
    """
    Phase 1: Reproduce paper's IAUKF results.

    Scenario:
    - Constant loads (steady-state)
    - 50% initial parameter error
    - IEEE 33-bus system
    - SCADA + PMU measurements
    """
    print("=" * 70)
    print("PHASE 1: VALIDATE IAUKF (Reproduce Paper Results)")
    print("=" * 70)

    # ========================================
    # 1. Generate Simulation Data (Constant Loads)
    # ========================================
    print("\n[Step 1] Generating simulation data with CONSTANT loads...")

    # Modify simulation for constant loads
    sim = PowerSystemSimulation(steps=steps)

    # Get base loads
    p_load_base = sim.net.load.p_mw.values.copy()
    q_load_base = sim.net.load.q_mvar.values.copy()

    # Generate data with CONSTANT loads
    z_scada_list = []
    z_pmu_list = []
    true_states_list = []

    np.random.seed(42)  # Reproducibility

    for t in range(steps):
        # NO load fluctuation - constant!
        sim.net.load.p_mw = p_load_base
        sim.net.load.q_mvar = q_load_base

        # Run power flow
        try:
            pp.runpp(sim.net, algorithm='nr', numba=False)
        except Exception as e:
            print(f"Warning: Power flow failed at step {t}: {e}")
            continue

        # Extract true states
        v_true = sim.net.res_bus.vm_pu.values
        delta_true = np.radians(sim.net.res_bus.va_degree.values)
        true_states_list.append(np.concatenate([v_true, delta_true, [sim.r_true, sim.x_true]]))

        # Generate SCADA measurements (with noise)
        p_inj = -sim.net.res_bus.p_mw.values
        q_inj = -sim.net.res_bus.q_mvar.values
        v_scada = sim.net.res_bus.vm_pu.values

        z_scada = np.concatenate([p_inj, q_inj, v_scada])
        noise_scada = np.random.normal(0, 0.02, size=len(z_scada))
        z_scada_list.append(z_scada + noise_scada)

        # Generate PMU measurements (with noise)
        v_pmu = sim.net.res_bus.vm_pu.values[sim.pmu_buses]
        theta_pmu = np.radians(sim.net.res_bus.va_degree.values[sim.pmu_buses])
        z_pmu = np.concatenate([v_pmu, theta_pmu])

        noise_pmu_v = np.random.normal(0, 0.005, size=len(v_pmu))
        noise_pmu_theta = np.random.normal(0, 0.002, size=len(theta_pmu))
        noise_pmu = np.concatenate([noise_pmu_v, noise_pmu_theta])
        z_pmu_list.append(z_pmu + noise_pmu)

    Z_scada = np.array(z_scada_list)
    Z_pmu = np.array(z_pmu_list)

    print(f"✓ Generated {steps} steady-state snapshots")
    print(f"  True parameters: R={sim.r_true:.4f}, X={sim.x_true:.4f}")

    # ========================================
    # 2. Setup IAUKF (Augmented State)
    # ========================================
    print("\n[Step 2] Setting up IAUKF with augmented state...")

    model = DistributionSystemModel(sim.net, sim.line_idx, sim.pmu_buses)

    # Initial state: [V, delta, R, X]
    x0_v = np.ones(sim.net.bus.shape[0])  # Flat start for voltages
    x0_d = np.zeros(sim.net.bus.shape[0])  # Zero angles
    x0_r = sim.r_true * 0.5  # 50% initial error (as in paper)
    x0_x = sim.x_true * 0.5  # 50% initial error

    x0 = np.concatenate([x0_v, x0_d, [x0_r, x0_x]])

    print(f"  State dimension: {len(x0)}")
    print(f"  Initial guess: R={x0_r:.4f} ({abs(x0_r-sim.r_true)/sim.r_true*100:.1f}% error)")
    print(f"                 X={x0_x:.4f} ({abs(x0_x-sim.x_true)/sim.x_true*100:.1f}% error)")

    # Covariance matrices (tuning parameters)
    P0 = np.eye(len(x0)) * 0.01
    P0[-2, -2] = 0.1  # Larger uncertainty for R
    P0[-1, -1] = 0.1  # Larger uncertainty for X

    Q0 = np.eye(len(x0)) * 1e-8  # Very small for steady-state
    Q0[-2, -2] = 1e-6  # Slightly larger for parameters
    Q0[-1, -1] = 1e-6

    # Measurement noise covariance
    R_diag = np.concatenate([
        np.full(33, 0.02**2),    # P injection
        np.full(33, 0.02**2),    # Q injection
        np.full(33, 0.02**2),    # V SCADA
        np.full(12, 0.005**2),   # V PMU
        np.full(12, 0.002**2)    # Theta PMU
    ])
    R_cov = np.diag(R_diag)

    print(f"  Measurement dimension: {len(R_diag)}")

    # ========================================
    # 3. Run IAUKF
    # ========================================
    print("\n[Step 3] Running IAUKF...")

    iaukf = IAUKF(model, x0, P0, Q0, R_cov)

    # History tracking
    history_r = []
    history_x = []
    history_r_err = []
    history_x_err = []
    history_v = []  # Track voltage estimates

    for t in range(steps):
        # Combine measurements
        z_t = np.concatenate([Z_scada[t], Z_pmu[t]])

        # IAUKF predict and update
        iaukf.predict()
        x_est = iaukf.update(z_t)

        # Extract parameter estimates
        r_est = x_est[-2]
        x_est_param = x_est[-1]

        # Calculate errors
        r_err_pct = abs(r_est - sim.r_true) / sim.r_true * 100
        x_err_pct = abs(x_est_param - sim.x_true) / sim.x_true * 100

        history_r.append(r_est)
        history_x.append(x_est_param)
        history_r_err.append(r_err_pct)
        history_x_err.append(x_err_pct)

        # Track voltage RMSE
        v_est = x_est[:33]
        v_true = true_states_list[t][:33]
        v_rmse = np.sqrt(np.mean((v_est - v_true)**2))
        history_v.append(v_rmse)

        # Progress output
        if t % 10 == 0 or t < 5:
            print(f"  Step {t:3d}: R={r_est:.4f} (err={r_err_pct:5.1f}%), "
                  f"X={x_est_param:.4f} (err={x_err_pct:5.1f}%), "
                  f"V_RMSE={v_rmse:.6f}")

    # ========================================
    # 4. Analyze Results
    # ========================================
    print("\n[Step 4] Analyzing results...")

    final_r_err = history_r_err[-1]
    final_x_err = history_x_err[-1]

    # Find convergence point (error < 5%)
    r_converged = next((i for i, e in enumerate(history_r_err) if e < 5), None)
    x_converged = next((i for i, e in enumerate(history_x_err) if e < 5), None)

    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print('='*70)
    print(f"Final estimates:")
    print(f"  R = {history_r[-1]:.4f} (true: {sim.r_true:.4f}, error: {final_r_err:.2f}%)")
    print(f"  X = {history_x[-1]:.4f} (true: {sim.x_true:.4f}, error: {final_x_err:.2f}%)")
    print(f"\nConvergence (error < 5%):")
    print(f"  R: {'Step ' + str(r_converged) if r_converged else 'Did not converge'}")
    print(f"  X: {'Step ' + str(x_converged) if x_converged else 'Did not converge'}")
    print(f"\nFinal voltage RMSE: {history_v[-1]:.6f} pu")

    # Last 20 steps statistics
    r_std_final = np.std(history_r[-20:])
    x_std_final = np.std(history_x[-20:])
    print(f"\nOscillation (last 20 steps std):")
    print(f"  R: {r_std_final:.6f}")
    print(f"  X: {x_std_final:.6f}")

    # ========================================
    # 5. Visualization
    # ========================================
    print("\n[Step 5] Generating plots...")

    fig = plt.figure(figsize=(15, 10))

    # Plot 1: R estimation
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(history_r, 'b-', linewidth=2, label='Estimated')
    ax1.axhline(sim.r_true, color='r', linestyle='--', linewidth=2, label='True')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('R (Ohm/km)')
    ax1.set_title('Resistance Estimation', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: X estimation
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(history_x, 'g-', linewidth=2, label='Estimated')
    ax2.axhline(sim.x_true, color='r', linestyle='--', linewidth=2, label='True')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('X (Ohm/km)')
    ax2.set_title('Reactance Estimation', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: R error percentage
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(history_r_err, 'b-', linewidth=2)
    ax3.axhline(5, color='orange', linestyle='--', label='5% threshold')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Error (%)')
    ax3.set_title('Resistance Error', fontweight='bold')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: X error percentage
    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(history_x_err, 'g-', linewidth=2)
    ax4.axhline(5, color='orange', linestyle='--', label='5% threshold')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Error (%)')
    ax4.set_title('Reactance Error', fontweight='bold')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Voltage RMSE
    ax5 = plt.subplot(3, 2, 5)
    ax5.plot(history_v, 'm-', linewidth=2)
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('RMSE (pu)')
    ax5.set_title('Voltage Estimation RMSE', fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # Plot 6: Final statistics
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')
    stats_text = f"""
    VALIDATION RESULTS

    True Parameters:
      R = {sim.r_true:.4f} Ohm/km
      X = {sim.x_true:.4f} Ohm/km

    Final Estimates:
      R = {history_r[-1]:.4f} ({final_r_err:.2f}% error)
      X = {history_x[-1]:.4f} ({final_x_err:.2f}% error)

    Convergence:
      R < 5%: Step {r_converged if r_converged else 'N/A'}
      X < 5%: Step {x_converged if x_converged else 'N/A'}

    Stability (std, last 20 steps):
      R: {r_std_final:.6f}
      X: {x_std_final:.6f}
    """
    ax6.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center')

    plt.tight_layout()

    if save_results:
        plt.savefig('tmp/phase1_validation_results.png', dpi=150, bbox_inches='tight')
        print("✓ Plot saved: tmp/phase1_validation_results.png")

    plt.close()

    # ========================================
    # 6. Success Check
    # ========================================
    print(f"\n{'='*70}")
    print("VALIDATION STATUS")
    print('='*70)

    success_criteria = {
        'Converges (R < 5%)': r_converged is not None,
        'Converges (X < 5%)': x_converged is not None,
        'Final error (R < 10%)': final_r_err < 10,
        'Final error (X < 10%)': final_x_err < 10,
        'Stable (R std < 0.05)': r_std_final < 0.05,
        'Stable (X std < 0.05)': x_std_final < 0.05,
    }

    for criterion, passed in success_criteria.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {criterion}")

    all_passed = all(success_criteria.values())

    if all_passed:
        print(f"\n{'='*70}")
        print("✓✓✓ PHASE 1 SUCCESS: IAUKF Implementation Validated! ✓✓✓")
        print('='*70)
        print("\nNext: Run Phase 2 to train Graph Mamba in same scenario")
    else:
        print(f"\n{'='*70}")
        print("⚠ PHASE 1 INCOMPLETE: Some criteria not met")
        print('='*70)
        print("\nTuning suggestions:")
        print("  - Adjust Q matrix (process noise)")
        print("  - Tune NSE parameters (b_factor)")
        print("  - Increase simulation steps")

    return {
        'success': all_passed,
        'final_r_err': final_r_err,
        'final_x_err': final_x_err,
        'r_converged_step': r_converged,
        'x_converged_step': x_converged,
        'history_r': history_r,
        'history_x': history_x,
        'history_r_err': history_r_err,
        'history_x_err': history_x_err,
    }


if __name__ == "__main__":
    results = phase1_validate_iaukf(steps=150)

    if results['success']:
        print("\n✓ Ready for Phase 2: python experiments/phase2_train_mamba.py")
