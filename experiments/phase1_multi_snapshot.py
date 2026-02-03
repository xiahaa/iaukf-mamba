"""
Phase 1: IAUKF Validation with Multiple Measurement Snapshots

This tests the augmented state-space model under multiple measurement snapshots
as described in Section IV.C of the paper (Eq 32-38).

According to Table II, multi-snapshot achieves:
- Branch 3-4: R error 0.13% (vs 0.18% single), X error 0.09% (vs 1.55% single)

The multi-snapshot model improves measurement redundancy (Eq 31):
  mt / (nt + n_p) > m / (n + n_p)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import pandapower as pp
from model.simulation import PowerSystemSimulation
from model.models_analytical import AnalyticalMeasurementModel
from model.iaukf import IAUKFMultiSnapshot


def run_multi_snapshot_validation(steps=200, num_snapshots=5):
    """
    Validate IAUKF with multiple measurement snapshots.

    Args:
        steps: Total simulation steps
        num_snapshots: Number of snapshots per update (paper uses 5)
    """
    print("=" * 70)
    print("PHASE 1: IAUKF with Multiple Measurement Snapshots")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  - System: IEEE 33-bus")
    print(f"  - Target: Branch 3-4")
    print(f"  - Steps: {steps}")
    print(f"  - Snapshots per update: {num_snapshots}")
    print(f"  - Reference: Paper Table II (0.13% R, 0.09% X)")

    # Generate simulation data
    print("\n[1] Generating simulation data (constant loads)...")
    sim = PowerSystemSimulation(steps=steps)

    p_load_base = sim.net.load.p_mw.values.copy()
    q_load_base = sim.net.load.q_mvar.values.copy()

    z_list = []  # Combined measurements
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

        # SCADA with noise
        p_inj = -sim.net.res_bus.p_mw.values
        q_inj = -sim.net.res_bus.q_mvar.values
        v_scada = sim.net.res_bus.vm_pu.values
        z_scada = np.concatenate([p_inj, q_inj, v_scada])
        z_scada_noisy = z_scada + np.random.normal(0, 0.02, len(z_scada))

        # PMU with noise
        v_pmu = sim.net.res_bus.vm_pu.values[sim.pmu_buses]
        theta_pmu = np.radians(sim.net.res_bus.va_degree.values[sim.pmu_buses])
        z_pmu = np.concatenate([v_pmu, theta_pmu])
        noise_pmu = np.concatenate([
            np.random.normal(0, 0.005, len(v_pmu)),
            np.random.normal(0, 0.002, len(theta_pmu))
        ])
        z_pmu_noisy = z_pmu + noise_pmu

        z_list.append(np.concatenate([z_scada_noisy, z_pmu_noisy]))

    print(f"✓ Generated {steps} snapshots")
    print(f"  True: R={sim.r_true:.4f}, X={sim.x_true:.4f}")

    # Setup IAUKF with multi-snapshot
    print("\n[2] Setting up IAUKF Multi-Snapshot...")

    model = AnalyticalMeasurementModel(sim.net, sim.line_idx, sim.pmu_buses)

    # Initial state (single snapshot)
    x0_v = np.ones(sim.net.bus.shape[0])
    x0_d = np.zeros(sim.net.bus.shape[0])
    x0_r = 0.01
    x0_x = 0.01
    x0 = np.concatenate([x0_v, x0_d, [x0_r, x0_x]])

    print(f"  Initial params: R={x0_r:.4f}, X={x0_x:.4f}")
    print(f"  Single state dim: {len(x0)}")

    # Covariances (single snapshot)
    P0 = np.eye(len(x0)) * 0.01
    P0[-2, -2] = 0.1
    P0[-1, -1] = 0.1

    Q0 = np.eye(len(x0)) * 1e-6

    R_diag = np.concatenate([
        np.full(33, 0.02**2),
        np.full(33, 0.02**2),
        np.full(33, 0.02**2),
        np.full(12, 0.005**2),
        np.full(12, 0.002**2)
    ])
    R_cov = np.diag(R_diag)

    # Create multi-snapshot IAUKF
    iaukf = IAUKFMultiSnapshot(model, x0, P0, Q0, R_cov, num_snapshots=num_snapshots)
    iaukf.b_factor = 0.98

    print(f"  Augmented state dim: {iaukf.n}")
    print(f"  Num snapshots: {num_snapshots}")
    print(f"  NSE b_factor: {iaukf.b_factor}")

    # Run IAUKF
    print("\n[3] Running IAUKF Multi-Snapshot...")

    history_r = []
    history_x = []
    history_r_err = []
    history_x_err = []

    # Collect measurements into batches
    effective_steps = steps // num_snapshots

    for batch_idx in range(effective_steps):
        # Collect num_snapshots measurements
        batch_measurements = []
        for i in range(num_snapshots):
            t = batch_idx * num_snapshots + i
            if t < steps:
                batch_measurements.append(z_list[t])

        if len(batch_measurements) < num_snapshots:
            break

        # Predict
        iaukf.predict()

        # Update with batch
        x_est = iaukf.update(batch_measurements)

        # Extract parameters
        params = iaukf.get_parameters()
        r_est = params[0]
        x_est_param = params[1]

        r_err_pct = abs(r_est - sim.r_true) / sim.r_true * 100
        x_err_pct = abs(x_est_param - sim.x_true) / sim.x_true * 100

        history_r.append(r_est)
        history_x.append(x_est_param)
        history_r_err.append(r_err_pct)
        history_x_err.append(x_err_pct)

        if batch_idx % 5 == 0 or batch_idx < 3:
            print(f"  Batch {batch_idx:3d}: R={r_est:.6f} (err={r_err_pct:5.2f}%), "
                  f"X={x_est_param:.6f} (err={x_err_pct:5.2f}%)")

    # Convergence check
    print("\n[4] Checking convergence...")

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

    print(f"  R converges at batch: {r_converged_step}")
    print(f"  X converges at batch: {x_converged_step}")

    # Final averaging
    if r_converged_step and r_converged_step < len(history_r):
        r_final = np.mean(history_r[r_converged_step:])
    else:
        r_final = np.mean(history_r[-10:]) if len(history_r) >= 10 else history_r[-1]

    if x_converged_step and x_converged_step < len(history_x):
        x_final = np.mean(history_x[x_converged_step:])
    else:
        x_final = np.mean(history_x[-10:]) if len(history_x) >= 10 else history_x[-1]

    r_final_err = abs(r_final - sim.r_true) / sim.r_true * 100
    x_final_err = abs(x_final - sim.x_true) / sim.x_true * 100

    r_std_last = np.std(history_r[-10:]) if len(history_r) >= 10 else np.std(history_r)
    x_std_last = np.std(history_x[-10:]) if len(history_x) >= 10 else np.std(history_x)

    print(f"\n[5] Results...")
    print(f"  R_final = {r_final:.6f} (true={sim.r_true:.4f}, error={r_final_err:.2f}%)")
    print(f"  X_final = {x_final:.6f} (true={sim.x_true:.4f}, error={x_final_err:.2f}%)")
    print(f"  Oscillation (last 10 batches std): R={r_std_last:.6f}, X={x_std_last:.6f}")

    # Compare with paper
    print(f"\n{'='*70}")
    print("COMPARISON WITH PAPER (Table II - Multi-Snapshot)")
    print('='*70)
    print(f"  Paper's results (branch 3-4, multi-snapshot):")
    print(f"    R error: 0.13%")
    print(f"    X error: 0.09%")
    print(f"\n  Our results:")
    print(f"    R error: {r_final_err:.2f}%")
    print(f"    X error: {x_final_err:.2f}%")
    print(f"    R oscillation: {r_std_last:.6f}")
    print(f"    X oscillation: {x_std_last:.6f}")

    # Evaluate
    if r_final_err < 1.0 and x_final_err < 1.0:
        print(f"\n  ✓✓✓ EXCELLENT: Paper-level accuracy achieved! ✓✓✓")
    elif r_final_err < 3.0 and x_final_err < 3.0:
        print(f"\n  ✓✓ VERY GOOD: Close to paper's results")
    elif r_final_err < 5.0 and x_final_err < 5.0:
        print(f"\n  ✓ GOOD: Acceptable accuracy")
    else:
        print(f"\n  ⚠ Needs investigation")

    # Visualization
    print(f"\n[6] Generating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    batches = range(len(history_r))

    # R estimation
    axes[0, 0].plot(batches, history_r, 'b-', linewidth=1.5, alpha=0.7, label='Estimated')
    axes[0, 0].axhline(sim.r_true, color='r', linestyle='--', linewidth=2, label='True')
    axes[0, 0].axhline(r_final, color='g', linestyle=':', linewidth=2, label='Final (averaged)')
    if r_converged_step:
        axes[0, 0].axvline(r_converged_step, color='orange', linestyle='--', alpha=0.5, label='Converged')
    axes[0, 0].set_xlabel('Batch')
    axes[0, 0].set_ylabel('R (Ohm/km)')
    axes[0, 0].set_title(f'Resistance Estimation (Multi-Snapshot, t={num_snapshots})', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0.0, 0.6)
    axes[0, 0].grid(True, alpha=0.3)

    # X estimation
    axes[0, 1].plot(batches, history_x, 'g-', linewidth=1.5, alpha=0.7, label='Estimated')
    axes[0, 1].axhline(sim.x_true, color='r', linestyle='--', linewidth=2, label='True')
    axes[0, 1].axhline(x_final, color='b', linestyle=':', linewidth=2, label='Final (averaged)')
    if x_converged_step:
        axes[0, 1].axvline(x_converged_step, color='orange', linestyle='--', alpha=0.5, label='Converged')
    axes[0, 1].set_xlabel('Batch')
    axes[0, 1].set_ylabel('X (Ohm/km)')
    axes[0, 1].set_title(f'Reactance Estimation (Multi-Snapshot, t={num_snapshots})', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # R error
    axes[1, 0].semilogy(batches, history_r_err, 'b-', linewidth=1.5)
    axes[1, 0].axhline(r_final_err, color='g', linestyle=':', linewidth=2, label=f'Final: {r_final_err:.2f}%')
    axes[1, 0].axhline(0.13, color='r', linestyle='--', linewidth=2, label='Paper: 0.13%')
    axes[1, 0].set_xlabel('Batch')
    axes[1, 0].set_ylabel('Error (%)')
    axes[1, 0].set_title('Resistance Error', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # X error
    axes[1, 1].semilogy(batches, history_x_err, 'g-', linewidth=1.5)
    axes[1, 1].axhline(x_final_err, color='b', linestyle=':', linewidth=2, label=f'Final: {x_final_err:.2f}%')
    axes[1, 1].axhline(0.09, color='r', linestyle='--', linewidth=2, label='Paper: 0.09%')
    axes[1, 1].set_xlabel('Batch')
    axes[1, 1].set_ylabel('Error (%)')
    axes[1, 1].set_title('Reactance Error', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'tmp/phase1_multi_snapshot_t{num_snapshots}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()

    return {
        'r_final': r_final,
        'x_final': x_final,
        'r_error': r_final_err,
        'x_error': x_final_err,
        'r_std': r_std_last,
        'x_std': x_std_last,
        'history_r': history_r,
        'history_x': history_x,
    }


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Testing Multi-Snapshot IAUKF (t=5 snapshots)")
    print("="*70)

    results = run_multi_snapshot_validation(steps=300, num_snapshots=5)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nMulti-snapshot (t=5) results:")
    print(f"  R error: {results['r_error']:.2f}%")
    print(f"  X error: {results['x_error']:.2f}%")
    print(f"  R oscillation (std): {results['r_std']:.6f}")
    print(f"  X oscillation (std): {results['x_std']:.6f}")

    print(f"\nPaper's multi-snapshot results (Table II):")
    print(f"  R error: 0.13%")
    print(f"  X error: 0.09%")
