"""
Phase 1: Single vs Multi-Snapshot IAUKF Comparison
====================================================

This experiment validates the enhanced IAUKF implementation by comparing:
1. Single-snapshot IAUKF (Eq 1-18)
2. Multi-snapshot IAUKF (Eq 32-38)

Reference paper Table II results:
- Single snapshot (Branch 3-4): R=0.18%, X=1.55%
- Multi-snapshot t=5 (Branch 3-4): R=0.13%, X=0.09%
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandapower as pp
from model.simulation import PowerSystemSimulation
from model.models_analytical import AnalyticalMeasurementModel
from model.iaukf import IAUKF, IAUKFMultiSnapshot

# Configuration
STEPS = 300
NUM_SNAPSHOTS_LIST = [1, 3, 5, 7]  # Test different snapshot counts
SEED = 42

print("=" * 70)
print("PHASE 1: SINGLE vs MULTI-SNAPSHOT IAUKF COMPARISON")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  Steps: {STEPS}")
print(f"  Snapshot counts to test: {NUM_SNAPSHOTS_LIST}")
print(f"  Target: Branch 3-4 (IEEE 33-bus)")

# ========================================
# Generate Simulation Data
# ========================================

print("\n[1] Generating simulation data (constant loads)...")

sim = PowerSystemSimulation(steps=STEPS)
num_buses = len(sim.net.bus)
pmu_buses = sim.pmu_buses

p_load_base = sim.net.load.p_mw.values.copy()
q_load_base = sim.net.load.q_mvar.values.copy()

# True parameters
r_true = sim.r_true
x_true = sim.x_true
print(f"  True: R={r_true:.4f}, X={x_true:.4f}")

# Generate measurements
measurements = []
np.random.seed(SEED)

for t in range(STEPS):
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
    v_pmu = sim.net.res_bus.vm_pu.values[pmu_buses]
    theta_pmu = np.radians(sim.net.res_bus.va_degree.values[pmu_buses])
    z_pmu = np.concatenate([v_pmu, theta_pmu])
    z_pmu += np.concatenate([
        np.random.normal(0, 0.005, len(v_pmu)),
        np.random.normal(0, 0.002, len(theta_pmu))
    ])

    measurements.append(np.concatenate([z_scada, z_pmu]))

print(f"  ✓ Generated {STEPS} measurements")

# ========================================
# Helper: Run Single-Snapshot IAUKF
# ========================================

def run_single_snapshot_iaukf(measurements, sim, verbose=False):
    """Run standard single-snapshot IAUKF."""
    model = AnalyticalMeasurementModel(sim.net, sim.line_idx, sim.pmu_buses)

    # Initial state
    x0 = np.ones(2 * num_buses + 2)
    x0[:num_buses] = 1.0
    x0[num_buses:2*num_buses] = 0.0
    x0[-2] = 0.01  # Small initial R
    x0[-1] = 0.01  # Small initial X

    # Covariances (tuned)
    P0 = np.eye(len(x0)) * 0.01
    P0[-2, -2] = 0.1
    P0[-1, -1] = 0.1

    Q0 = np.eye(len(x0)) * 1e-6
    Q0[-2, -2] = 1e-6
    Q0[-1, -1] = 1e-6

    # Measurement covariance
    n_scada = 3 * num_buses
    n_pmu = 2 * len(sim.pmu_buses)
    R_diag = np.concatenate([
        np.full(n_scada, 0.02**2),
        np.full(len(sim.pmu_buses), 0.005**2),
        np.full(len(sim.pmu_buses), 0.002**2)
    ])
    R = np.diag(R_diag)

    # Create IAUKF
    iaukf = IAUKF(model, x0, P0, Q0, R)
    iaukf.b_factor = 0.96

    # Run filter
    r_history = []
    x_history = []

    for t, z in enumerate(measurements):
        iaukf.predict()
        iaukf.update(z)
        r_history.append(iaukf.x[-2])
        x_history.append(iaukf.x[-1])

    return np.array(r_history), np.array(x_history)


# ========================================
# Helper: Run Multi-Snapshot IAUKF
# ========================================

def run_multi_snapshot_iaukf(measurements, sim, num_snapshots, verbose=False):
    """Run multi-snapshot IAUKF."""
    model = AnalyticalMeasurementModel(sim.net, sim.line_idx, sim.pmu_buses)

    # Initial state for single snapshot
    x0_single = np.ones(2 * num_buses + 2)
    x0_single[:num_buses] = 1.0
    x0_single[num_buses:2*num_buses] = 0.0
    x0_single[-2] = 0.01
    x0_single[-1] = 0.01

    # Covariances
    P0 = np.eye(len(x0_single)) * 0.01
    P0[-2, -2] = 0.1
    P0[-1, -1] = 0.1

    Q0 = np.eye(len(x0_single)) * 1e-6
    Q0[-2, -2] = 1e-6
    Q0[-1, -1] = 1e-6

    # Measurement covariance
    n_scada = 3 * num_buses
    R_diag = np.concatenate([
        np.full(n_scada, 0.02**2),
        np.full(len(sim.pmu_buses), 0.005**2),
        np.full(len(sim.pmu_buses), 0.002**2)
    ])
    R = np.diag(R_diag)

    # Create multi-snapshot IAUKF
    iaukf_ms = IAUKFMultiSnapshot(model, x0_single, P0, Q0, R, num_snapshots=num_snapshots)
    iaukf_ms.b_factor = 0.96

    # Run filter
    r_history = []
    x_history = []

    for t, z in enumerate(measurements):
        iaukf_ms.predict()
        iaukf_ms.update(z)
        params = iaukf_ms.get_parameters()
        r_history.append(params[0])
        x_history.append(params[1])

    return np.array(r_history), np.array(x_history)


# ========================================
# Run Experiments
# ========================================

print("\n[2] Running experiments...")

results = {}

for num_snapshots in NUM_SNAPSHOTS_LIST:
    print(f"\n  Testing t={num_snapshots} snapshots...")

    if num_snapshots == 1:
        r_hist, x_hist = run_single_snapshot_iaukf(measurements, sim)
    else:
        r_hist, x_hist = run_multi_snapshot_iaukf(measurements, sim, num_snapshots)

    # Compute final error (post-convergence averaging, Eq. 40)
    # Find convergence point
    convergence_threshold = 0.001
    r_converged = STEPS
    x_converged = STEPS

    for k in range(1, len(r_hist)):
        if abs(r_hist[k] - r_hist[k-1]) <= convergence_threshold and r_converged == STEPS:
            r_converged = k
        if abs(x_hist[k] - x_hist[k-1]) <= convergence_threshold and x_converged == STEPS:
            x_converged = k

    # Average post-convergence
    start_avg = max(r_converged, x_converged, STEPS // 2)
    r_final = np.mean(r_hist[start_avg:])
    x_final = np.mean(x_hist[start_avg:])

    r_error = abs(r_final - r_true) / r_true * 100
    x_error = abs(x_final - x_true) / x_true * 100

    # Oscillation (std of last 20 steps)
    r_std = np.std(r_hist[-20:])
    x_std = np.std(x_hist[-20:])

    results[num_snapshots] = {
        'r_history': r_hist,
        'x_history': x_hist,
        'r_final': r_final,
        'x_final': x_final,
        'r_error': r_error,
        'x_error': x_error,
        'r_std': r_std,
        'x_std': x_std,
        'r_converged': r_converged,
        'x_converged': x_converged
    }

    print(f"    R error: {r_error:.2f}%, X error: {x_error:.2f}%")

# ========================================
# Generate Summary Table
# ========================================

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

# Create comparison table
print("\n╔═══════════════════════════════════════════════════════════════════╗")
print("║                 SINGLE vs MULTI-SNAPSHOT COMPARISON               ║")
print("╠═══════════════════════════════════════════════════════════════════╣")
print("║ Snapshots │  R Error  │  X Error  │ R Converge │ X Converge │ Std ║")
print("╠═══════════════════════════════════════════════════════════════════╣")

for t in NUM_SNAPSHOTS_LIST:
    r = results[t]
    label = "Single" if t == 1 else f"Multi-{t}"
    print(f"║ {label:9s} │ {r['r_error']:7.2f}% │ {r['x_error']:7.2f}% │ "
          f"{r['r_converged']:10d} │ {r['x_converged']:10d} │ {r['r_std']:.4f} ║")

print("╚═══════════════════════════════════════════════════════════════════╝")

# Paper comparison
print("\n" + "-" * 70)
print("PAPER REFERENCE (Table II, Branch 3-4):")
print("-" * 70)
print("  Single snapshot:     R=0.18%, X=1.55%")
print("  Multi-snapshot (5):  R=0.13%, X=0.09%")
print("-" * 70)
print("OUR RESULTS:")
print(f"  Single snapshot:     R={results[1]['r_error']:.2f}%, X={results[1]['x_error']:.2f}%")
print(f"  Multi-snapshot (5):  R={results[5]['r_error']:.2f}%, X={results[5]['x_error']:.2f}%")
print("-" * 70)

# Validation status
print("\nVALIDATION STATUS:")
if results[1]['r_error'] < 5 and results[1]['x_error'] < 5:
    print("  ✓ Single-snapshot: PASS (< 5% error)")
else:
    print("  ✗ Single-snapshot: FAIL")

if results[5]['r_error'] < 2 and results[5]['x_error'] < 2:
    print("  ✓ Multi-snapshot:  PASS (< 2% error, close to paper)")
else:
    print("  ✗ Multi-snapshot:  FAIL")

# ========================================
# Generate Plot
# ========================================

print("\n[3] Generating comparison plot...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot R convergence
ax = axes[0, 0]
for t in NUM_SNAPSHOTS_LIST:
    label = "Single (t=1)" if t == 1 else f"Multi (t={t})"
    ax.plot(results[t]['r_history'], label=label, alpha=0.8)
ax.axhline(y=r_true, color='k', linestyle='--', label='True R')
ax.set_xlabel('Step')
ax.set_ylabel('R (Ohm/km)')
ax.set_title('R Convergence: Single vs Multi-Snapshot')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot X convergence
ax = axes[0, 1]
for t in NUM_SNAPSHOTS_LIST:
    label = "Single (t=1)" if t == 1 else f"Multi (t={t})"
    ax.plot(results[t]['x_history'], label=label, alpha=0.8)
ax.axhline(y=x_true, color='k', linestyle='--', label='True X')
ax.set_xlabel('Step')
ax.set_ylabel('X (Ohm/km)')
ax.set_title('X Convergence: Single vs Multi-Snapshot')
ax.legend()
ax.grid(True, alpha=0.3)

# Error comparison bar chart
ax = axes[1, 0]
x_pos = np.arange(len(NUM_SNAPSHOTS_LIST))
width = 0.35
labels = [f"t={t}" for t in NUM_SNAPSHOTS_LIST]
r_errors = [results[t]['r_error'] for t in NUM_SNAPSHOTS_LIST]
x_errors = [results[t]['x_error'] for t in NUM_SNAPSHOTS_LIST]

bars1 = ax.bar(x_pos - width/2, r_errors, width, label='R Error', color='steelblue')
bars2 = ax.bar(x_pos + width/2, x_errors, width, label='X Error', color='coral')

# Add paper reference lines
ax.axhline(y=0.18, color='steelblue', linestyle=':', alpha=0.7, label='Paper R (single)')
ax.axhline(y=1.55, color='coral', linestyle=':', alpha=0.7, label='Paper X (single)')

ax.set_xlabel('Number of Snapshots')
ax.set_ylabel('Error (%)')
ax.set_title('Error Comparison by Snapshot Count')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars1, r_errors):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=8)
for bar, val in zip(bars2, x_errors):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=8)

# Oscillation comparison
ax = axes[1, 1]
r_stds = [results[t]['r_std'] for t in NUM_SNAPSHOTS_LIST]
x_stds = [results[t]['x_std'] for t in NUM_SNAPSHOTS_LIST]

ax.bar(x_pos - width/2, r_stds, width, label='R Std', color='steelblue')
ax.bar(x_pos + width/2, x_stds, width, label='X Std', color='coral')
ax.set_xlabel('Number of Snapshots')
ax.set_ylabel('Standard Deviation')
ax.set_title('Estimation Stability (Last 20 Steps)')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('tmp/phase1_single_vs_multi_snapshot.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: tmp/phase1_single_vs_multi_snapshot.png")

# ========================================
# Save Results
# ========================================

import pickle

results_summary = {
    'single_snapshot': {
        'r_error': results[1]['r_error'],
        'x_error': results[1]['x_error'],
        'r_std': results[1]['r_std'],
        'x_std': results[1]['x_std']
    },
    'multi_snapshot_5': {
        'r_error': results[5]['r_error'],
        'x_error': results[5]['x_error'],
        'r_std': results[5]['r_std'],
        'x_std': results[5]['x_std']
    },
    'paper_reference': {
        'single': {'r_error': 0.18, 'x_error': 1.55},
        'multi_5': {'r_error': 0.13, 'x_error': 0.09}
    },
    'all_results': results
}

with open('tmp/phase1_single_vs_multi_results.pkl', 'wb') as f:
    pickle.dump(results_summary, f)
print("  ✓ Saved: tmp/phase1_single_vs_multi_results.pkl")

print("\n" + "=" * 70)
print("✓ PHASE 1 VALIDATION COMPLETE")
print("=" * 70)
