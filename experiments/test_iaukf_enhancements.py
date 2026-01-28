"""
Test IAUKF Enhancements
=======================

This script validates the enhancements made to IAUKF:
1. Exact NSE implementation (Eq 17 & 18)
2. Multi-snapshot support (Eq 32-38)

Expected outcomes based on ref_core.md Table II:
- Single snapshot: R error ~0.18%, X error ~1.55% (branch 3-4)
- Multi-snapshot (5 snapshots): R error ~0.13%, X error ~0.09% (branch 3-4)
- End branch (21-22) with multi-snapshot: R error ~0.52%, X error ~2.03%
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import pandapower as pp
from model.simulation import PowerSystemSimulation
from model.models_holt import DistributionSystemModelHolt
from model.iaukf import IAUKF, IAUKFMultiSnapshot


def test_single_snapshot_nse():
    """
    Test 1: Single-snapshot IAUKF with fixed NSE on branch 3-4.
    Expected: R error ~0.18%, X error ~1.55%
    """
    print("=" * 80)
    print("TEST 1: Single-Snapshot IAUKF with Exact NSE (Branch 3-4)")
    print("=" * 80)
    
    steps = 200
    target_branch = 3  # Branch 3-4 (index in pandapower)
    
    # Setup simulation
    print("\n[1] Setting up simulation...")
    sim = PowerSystemSimulation(steps=steps)
    sim.target_line_idx = target_branch
    
    # Get true parameters
    r_true = sim.net.line.at[target_branch, 'r_ohm_per_km']
    x_true = sim.net.line.at[target_branch, 'x_ohm_per_km']
    print(f"  True R = {r_true:.4f}, X = {x_true:.4f}")
    
    # Generate measurements with constant loads
    print("\n[2] Generating measurements...")
    p_load_base = sim.net.load.p_mw.values.copy()
    q_load_base = sim.net.load.q_mvar.values.copy()
    
    measurements = []
    np.random.seed(42)
    
    for t in range(steps):
        sim.net.load.p_mw = p_load_base
        sim.net.load.q_mvar = q_load_base
        pp.runpp(sim.net, algorithm='nr', numba=False)
        
        # SCADA measurements
        p_inj = -sim.net.res_bus.p_mw.values
        q_inj = -sim.net.res_bus.q_mvar.values
        v_scada = sim.net.res_bus.vm_pu.values
        z_scada = np.concatenate([p_inj, q_inj, v_scada])
        z_scada += np.random.normal(0, 0.02, len(z_scada))
        
        # PMU measurements
        v_pmu = sim.net.res_bus.vm_pu.values[sim.pmu_buses]
        theta_pmu = np.radians(sim.net.res_bus.va_degree.values[sim.pmu_buses])
        z_pmu = np.concatenate([v_pmu, theta_pmu])
        noise_pmu = np.concatenate([
            np.random.normal(0, 0.005, len(v_pmu)),
            np.random.normal(0, 0.002, len(theta_pmu))
        ])
        z_pmu += noise_pmu
        
        measurements.append(np.concatenate([z_scada, z_pmu]))
    
    # Initialize IAUKF
    print("\n[3] Initializing IAUKF...")
    num_buses = len(sim.net.bus)
    state_dim = 2 * num_buses + 2
    
    # Initial state
    x0 = np.ones(state_dim)
    x0[:num_buses] = 1.0  # Voltage magnitudes
    x0[num_buses:2*num_buses] = 0.0  # Angles
    x0[-2] = 0.01  # Small initial R (as in paper)
    x0[-1] = 0.01  # Small initial X (as in paper)
    
    P0 = np.eye(state_dim) * 1e-6
    Q0 = np.eye(state_dim) * 1e-6
    
    # Measurement covariance
    n_scada = 3 * num_buses
    n_pmu = 2 * len(sim.pmu_buses)
    R = np.eye(n_scada + n_pmu)
    R[:n_scada, :n_scada] *= 0.02**2
    R[n_scada:n_scada+len(sim.pmu_buses), n_scada:n_scada+len(sim.pmu_buses)] *= 0.005**2
    R[n_scada+len(sim.pmu_buses):, n_scada+len(sim.pmu_buses):] *= 0.002**2
    
    # Create model
    model = DistributionSystemModelHolt(
        sim.net.copy(), target_branch, sim.pmu_buses,
        alpha_H=0.8, beta_H=0.5
    )
    
    iaukf = IAUKF(model, x0, P0, Q0, R)
    iaukf.b_factor = 0.96  # As in paper
    
    # Run filter
    print("\n[4] Running IAUKF...")
    r_estimates = []
    x_estimates = []
    
    for t, z in enumerate(measurements):
        iaukf.predict()
        x_est = iaukf.update(z)
        r_estimates.append(x_est[-2])
        x_estimates.append(x_est[-1])
        
        if (t+1) % 50 == 0:
            r_err = abs(x_est[-2] - r_true) / r_true * 100
            x_err = abs(x_est[-1] - x_true) / x_true * 100
            print(f"  Step {t+1}: R={x_est[-2]:.4f} ({r_err:.2f}%), X={x_est[-1]:.4f} ({x_err:.2f}%)")
    
    # Calculate final error (average over converged steps)
    # Use convergence criterion: |p_k - p_{k-1}| <= 0.001
    converged_idx = steps // 2  # Start from halfway
    for i in range(10, steps-1):
        if abs(r_estimates[i+1] - r_estimates[i]) <= 0.001 and \
           abs(x_estimates[i+1] - x_estimates[i]) <= 0.001:
            converged_idx = i
            break
    
    r_final = np.mean(r_estimates[converged_idx:])
    x_final = np.mean(x_estimates[converged_idx:])
    
    r_error = abs(r_final - r_true) / r_true * 100
    x_error = abs(x_final - x_true) / x_true * 100
    
    print(f"\n[5] Results:")
    print(f"  Converged at step: {converged_idx}")
    print(f"  Final R estimate: {r_final:.6f} (True: {r_true:.6f})")
    print(f"  Final X estimate: {x_final:.6f} (True: {x_true:.6f})")
    print(f"  R error: {r_error:.2f}% (Target: ~0.18%)")
    print(f"  X error: {x_error:.2f}% (Target: ~1.55%)")
    
    # Check if within reasonable bounds (allow 2x tolerance)
    assert r_error < 0.5, f"R error too high: {r_error:.2f}%"
    assert x_error < 3.0, f"X error too high: {x_error:.2f}%"
    print(f"  ✓ Test PASSED!")
    
    return r_estimates, x_estimates, r_error, x_error


def test_multi_snapshot():
    """
    Test 2: Multi-snapshot IAUKF on branch 3-4.
    Expected: Better accuracy than single-snapshot (R ~0.13%, X ~0.09%)
    """
    print("\n" + "=" * 80)
    print("TEST 2: Multi-Snapshot IAUKF (5 snapshots) - Branch 3-4")
    print("=" * 80)
    
    steps = 200
    num_snapshots = 5
    target_branch = 3
    
    # Setup simulation
    print("\n[1] Setting up simulation...")
    sim = PowerSystemSimulation(steps=steps)
    sim.target_line_idx = target_branch
    
    r_true = sim.net.line.at[target_branch, 'r_ohm_per_km']
    x_true = sim.net.line.at[target_branch, 'x_ohm_per_km']
    print(f"  True R = {r_true:.4f}, X = {x_true:.4f}")
    print(f"  Using {num_snapshots} snapshots")
    
    # Generate measurements
    print("\n[2] Generating measurements...")
    p_load_base = sim.net.load.p_mw.values.copy()
    q_load_base = sim.net.load.q_mvar.values.copy()
    
    measurements = []
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
        z_scada += np.random.normal(0, 0.02, len(z_scada))
        
        # PMU
        v_pmu = sim.net.res_bus.vm_pu.values[sim.pmu_buses]
        theta_pmu = np.radians(sim.net.res_bus.va_degree.values[sim.pmu_buses])
        z_pmu = np.concatenate([v_pmu, theta_pmu])
        noise_pmu = np.concatenate([
            np.random.normal(0, 0.005, len(v_pmu)),
            np.random.normal(0, 0.002, len(theta_pmu))
        ])
        z_pmu += noise_pmu
        
        measurements.append(np.concatenate([z_scada, z_pmu]))
    
    # Initialize multi-snapshot IAUKF
    print("\n[3] Initializing Multi-Snapshot IAUKF...")
    num_buses = len(sim.net.bus)
    state_dim = 2 * num_buses + 2
    
    x0 = np.ones(state_dim)
    x0[:num_buses] = 1.0
    x0[num_buses:2*num_buses] = 0.0
    x0[-2] = 0.01
    x0[-1] = 0.01
    
    P0 = np.eye(state_dim) * 1e-6
    Q0 = np.eye(state_dim) * 1e-6
    
    n_scada = 3 * num_buses
    n_pmu = 2 * len(sim.pmu_buses)
    R = np.eye(n_scada + n_pmu)
    R[:n_scada, :n_scada] *= 0.02**2
    R[n_scada:n_scada+len(sim.pmu_buses), n_scada:n_scada+len(sim.pmu_buses)] *= 0.005**2
    R[n_scada+len(sim.pmu_buses):, n_scada+len(sim.pmu_buses):] *= 0.002**2
    
    model = DistributionSystemModelHolt(
        sim.net.copy(), target_branch, sim.pmu_buses,
        alpha_H=0.8, beta_H=0.5
    )
    
    iaukf_ms = IAUKFMultiSnapshot(model, x0, P0, Q0, R, num_snapshots=num_snapshots)
    iaukf_ms.b_factor = 0.96
    
    # Run filter
    print("\n[4] Running Multi-Snapshot IAUKF...")
    r_estimates = []
    x_estimates = []
    
    for t, z in enumerate(measurements):
        iaukf_ms.predict()
        x_est = iaukf_ms.update(z)
        params = iaukf_ms.get_parameters()
        r_estimates.append(params[0])
        x_estimates.append(params[1])
        
        if (t+1) % 50 == 0:
            r_err = abs(params[0] - r_true) / r_true * 100
            x_err = abs(params[1] - x_true) / x_true * 100
            print(f"  Step {t+1}: R={params[0]:.4f} ({r_err:.2f}%), X={params[1]:.4f} ({x_err:.2f}%)")
    
    # Calculate final error
    converged_idx = steps // 2
    for i in range(num_snapshots+10, steps-1):
        if abs(r_estimates[i+1] - r_estimates[i]) <= 0.001 and \
           abs(x_estimates[i+1] - x_estimates[i]) <= 0.001:
            converged_idx = i
            break
    
    r_final = np.mean(r_estimates[converged_idx:])
    x_final = np.mean(x_estimates[converged_idx:])
    
    r_error = abs(r_final - r_true) / r_true * 100
    x_error = abs(x_final - x_true) / x_true * 100
    
    print(f"\n[5] Results:")
    print(f"  Converged at step: {converged_idx}")
    print(f"  Final R estimate: {r_final:.6f} (True: {r_true:.6f})")
    print(f"  Final X estimate: {x_final:.6f} (True: {x_true:.6f})")
    print(f"  R error: {r_error:.2f}% (Target: ~0.13%)")
    print(f"  X error: {x_error:.2f}% (Target: ~0.09%)")
    
    # Multi-snapshot should be more accurate
    assert r_error < 0.5, f"R error too high: {r_error:.2f}%"
    assert x_error < 1.0, f"X error too high: {x_error:.2f}%"
    print(f"  ✓ Test PASSED!")
    
    return r_estimates, x_estimates, r_error, x_error


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("IAUKF ENHANCEMENTS VALIDATION")
    print("="*80)
    print("\nTesting:")
    print("  1. Exact NSE implementation (Eq 17 & 18)")
    print("  2. Multi-snapshot support (Eq 32-38)")
    print("\nExpected outcomes from ref_core.md Table II:")
    print("  - Single snapshot: R ~0.18%, X ~1.55%")
    print("  - Multi-snapshot: R ~0.13%, X ~0.09% (better accuracy)")
    
    try:
        # Test 1: Single snapshot with exact NSE
        r1, x1, r_err1, x_err1 = test_single_snapshot_nse()
        
        # Test 2: Multi-snapshot
        r2, x2, r_err2, x_err2 = test_multi_snapshot()
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"\nSingle-Snapshot IAUKF:")
        print(f"  R error: {r_err1:.2f}% (Target: ~0.18%)")
        print(f"  X error: {x_err1:.2f}% (Target: ~1.55%)")
        print(f"  Status: {'✓ PASS' if r_err1 < 0.5 and x_err1 < 3.0 else '✗ FAIL'}")
        
        print(f"\nMulti-Snapshot IAUKF (5 snapshots):")
        print(f"  R error: {r_err2:.2f}% (Target: ~0.13%)")
        print(f"  X error: {x_err2:.2f}% (Target: ~0.09%)")
        print(f"  Status: {'✓ PASS' if r_err2 < 0.5 and x_err2 < 1.0 else '✗ FAIL'}")
        
        print(f"\nImprovement from multi-snapshot:")
        print(f"  R error reduction: {r_err1 - r_err2:.2f}%")
        print(f"  X error reduction: {x_err1 - x_err2:.2f}%")
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(r1, label='Single-snapshot', alpha=0.7)
        ax1.plot(r2, label='Multi-snapshot', alpha=0.7)
        ax1.axhline(y=r_true, color='k', linestyle='--', label='True')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('R estimate')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Resistance Estimation')
        
        ax2.plot(x1, label='Single-snapshot', alpha=0.7)
        ax2.plot(x2, label='Multi-snapshot', alpha=0.7)
        ax2.axhline(y=x_true, color='k', linestyle='--', label='True')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('X estimate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Reactance Estimation')
        
        plt.tight_layout()
        plt.savefig('/tmp/iaukf_enhancements_test.png', dpi=150)
        print(f"\n  Plot saved to: /tmp/iaukf_enhancements_test.png")
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    # Need to get r_true, x_true from simulation
    sim = PowerSystemSimulation(steps=1)
    r_true = sim.net.line.at[3, 'r_ohm_per_km']
    x_true = sim.net.line.at[3, 'x_ohm_per_km']
    
    exit(main())
