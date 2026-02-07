"""
Phase 1: End Branch Estimation Test
=====================================

According to the reference paper (Section V.A):
- End branches (17-18, 21-22, 24-25, 32-33) cannot converge under single snapshot
- Multi-snapshot (t=5) is required for these branches
- Paper shows branch 21-22 with multi-snapshot: R=0.52%, X=2.03%

This experiment validates our multi-snapshot IAUKF implementation on end branches.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandapower as pp
from tqdm import tqdm
from model.simulation import PowerSystemSimulation
from model.models_analytical import AnalyticalMeasurementModel
from model.iaukf import IAUKF, IAUKFMultiSnapshot

# ========================================
# Configuration
# ========================================

# End branches identified in paper
END_BRANCHES = {
    'branch_17_18': 16,  # pandapower index
    'branch_21_22': 20,
    'branch_24_25': 23,
    'branch_32_33': 31
}

# Also test a regular branch for comparison
REGULAR_BRANCH = {'branch_3_4': 3}

ALL_BRANCHES = {**REGULAR_BRANCH, **END_BRANCHES}

STEPS = 300
NUM_SNAPSHOTS_OPTIONS = [1, 5]  # Compare single vs 5 snapshots
SEED = 42

print("=" * 70)
print("PHASE 1: END BRANCH ESTIMATION TEST")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  Steps: {STEPS}")
print(f"  End branches: {list(END_BRANCHES.keys())}")
print(f"  Regular branch: {list(REGULAR_BRANCH.keys())}")
print(f"  Snapshot options: {NUM_SNAPSHOTS_OPTIONS}")

# ========================================
# Helper: Run IAUKF on specific branch
# ========================================

def run_iaukf_on_branch(branch_idx, num_snapshots=1, steps=300, verbose=False):
    """
    Run IAUKF on a specific branch.

    Args:
        branch_idx: Index of the branch in pandapower
        num_snapshots: Number of snapshots (1 for single, >1 for multi)
        steps: Number of time steps
        verbose: Print progress

    Returns:
        dict with results
    """
    # Create simulation
    sim = PowerSystemSimulation(steps=steps)
    sim.target_line_idx = branch_idx
    sim.line_idx = branch_idx

    # Get true parameters
    r_true = sim.net.line.at[branch_idx, 'r_ohm_per_km']
    x_true = sim.net.line.at[branch_idx, 'x_ohm_per_km']

    if verbose:
        print(f"  Branch {branch_idx}: R_true={r_true:.4f}, X_true={x_true:.4f}")

    # Create model
    model = AnalyticalMeasurementModel(sim.net, branch_idx, sim.pmu_buses)

    # Get base loads
    p_load_base = sim.net.load.p_mw.values.copy()
    q_load_base = sim.net.load.q_mvar.values.copy()

    # Generate measurements
    measurements = []
    np.random.seed(SEED)

    for t in range(steps):
        sim.net.load.p_mw = p_load_base
        sim.net.load.q_mvar = q_load_base

        try:
            pp.runpp(sim.net, algorithm='nr', numba=False)
        except:
            continue

        # SCADA
        num_buses = len(sim.net.bus)
        p_inj = -sim.net.res_bus.p_mw.values
        q_inj = -sim.net.res_bus.q_mvar.values
        v_scada = sim.net.res_bus.vm_pu.values
        z_scada = np.concatenate([p_inj, q_inj, v_scada])
        z_scada += np.random.normal(0, 0.02, len(z_scada))

        # PMU
        v_pmu = sim.net.res_bus.vm_pu.values[sim.pmu_buses]
        theta_pmu = np.radians(sim.net.res_bus.va_degree.values[sim.pmu_buses])
        z_pmu = np.concatenate([v_pmu, theta_pmu])
        z_pmu += np.concatenate([
            np.random.normal(0, 0.005, len(v_pmu)),
            np.random.normal(0, 0.002, len(theta_pmu))
        ])

        measurements.append(np.concatenate([z_scada, z_pmu]))

    # Initial state
    num_buses = len(sim.net.bus)
    x0 = np.ones(2 * num_buses + 2)
    x0[:num_buses] = 1.0
    x0[num_buses:2*num_buses] = 0.0
    x0[-2] = 0.01  # Small initial R
    x0[-1] = 0.01  # Small initial X

    # Covariances
    P0 = np.eye(len(x0)) * 0.01
    P0[-2, -2] = 0.1
    P0[-1, -1] = 0.1

    Q0 = np.eye(len(x0)) * 1e-6
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

    # Create IAUKF
    if num_snapshots == 1:
        iaukf = IAUKF(model, x0, P0, Q0, R)
    else:
        iaukf = IAUKFMultiSnapshot(model, x0, P0, Q0, R, num_snapshots=num_snapshots)
    iaukf.b_factor = 0.96

    # Run filter
    r_history = []
    x_history = []

    for t, z in enumerate(measurements):
        iaukf.predict()
        iaukf.update(z)

        if num_snapshots == 1:
            r_history.append(iaukf.x[-2])
            x_history.append(iaukf.x[-1])
        else:
            params = iaukf.get_parameters()
            r_history.append(params[0])
            x_history.append(params[1])

    r_history = np.array(r_history)
    x_history = np.array(x_history)

    # Check convergence (paper's criterion: |p_k - p_{k-1}| <= 0.001)
    convergence_threshold = 0.001
    r_converged = False
    x_converged = False
    r_conv_step = steps
    x_conv_step = steps

    for k in range(10, len(r_history)):
        if abs(r_history[k] - r_history[k-1]) <= convergence_threshold:
            if not r_converged:
                r_converged = True
                r_conv_step = k
        if abs(x_history[k] - x_history[k-1]) <= convergence_threshold:
            if not x_converged:
                x_converged = True
                x_conv_step = k

    # Final estimate (post-convergence average)
    if r_converged and x_converged:
        start_avg = max(r_conv_step, x_conv_step)
        r_final = np.mean(r_history[start_avg:])
        x_final = np.mean(x_history[start_avg:])
    else:
        # Not converged - use last 20% anyway
        start_avg = int(0.8 * len(r_history))
        r_final = np.mean(r_history[start_avg:])
        x_final = np.mean(x_history[start_avg:])

    r_error = abs(r_final - r_true) / r_true * 100
    x_error = abs(x_final - x_true) / x_true * 100

    # Check oscillation (high std = not converged)
    r_std = np.std(r_history[-20:])
    x_std = np.std(x_history[-20:])

    return {
        'r_true': r_true,
        'x_true': x_true,
        'r_final': r_final,
        'x_final': x_final,
        'r_error': r_error,
        'x_error': x_error,
        'r_converged': r_converged,
        'x_converged': x_converged,
        'r_conv_step': r_conv_step,
        'x_conv_step': x_conv_step,
        'r_std': r_std,
        'x_std': x_std,
        'r_history': r_history,
        'x_history': x_history
    }


# ========================================
# Run Experiments
# ========================================

print("\n[1] Running experiments on all branches...")

results = {}

for branch_name, branch_idx in tqdm(ALL_BRANCHES.items(), desc="Branches"):
    results[branch_name] = {}

    for num_snap in NUM_SNAPSHOTS_OPTIONS:
        snap_key = f"t={num_snap}"
        result = run_iaukf_on_branch(branch_idx, num_snapshots=num_snap, steps=STEPS)
        results[branch_name][snap_key] = result

# ========================================
# Results Summary
# ========================================

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

# Create summary table
print("\n╔═════════════════════════════════════════════════════════════════════════════╗")
print("║                    END BRANCH ESTIMATION RESULTS                            ║")
print("╠═════════════════════════════════════════════════════════════════════════════╣")
print("║ Branch      │ Snapshots │ R Error  │ X Error  │ Converged │ Oscillation   ║")
print("╠═════════════════════════════════════════════════════════════════════════════╣")

for branch_name, branch_results in results.items():
    is_end_branch = branch_name in END_BRANCHES
    marker = "⚠ END" if is_end_branch else "  REG"

    for snap_key, result in branch_results.items():
        conv_str = "✓" if (result['r_converged'] and result['x_converged']) else "✗"
        osc_str = f"R:{result['r_std']:.4f} X:{result['x_std']:.4f}"

        print(f"║ {branch_name:11s} │ {snap_key:9s} │ {result['r_error']:7.2f}% │ {result['x_error']:7.2f}% │     {conv_str}     │ {osc_str} ║")
    print("╠═════════════════════════════════════════════════════════════════════════════╣")

print("╚═════════════════════════════════════════════════════════════════════════════╝")

# Paper comparison
print("\n" + "-" * 70)
print("PAPER REFERENCE (Table II - End Branch 21-22 with multi-snapshot):")
print("-" * 70)
print("  R error: 0.52%")
print("  X error: 2.03%")
print("-" * 70)
print("OUR RESULTS (Branch 21-22 with t=5):")
r21_22 = results['branch_21_22']['t=5']
print(f"  R error: {r21_22['r_error']:.2f}%")
print(f"  X error: {r21_22['x_error']:.2f}%")
print("-" * 70)

# Key findings
print("\n" + "=" * 70)
print("KEY FINDINGS")
print("=" * 70)

# Check if single snapshot fails on end branches
single_snap_failures = []
for branch_name in END_BRANCHES.keys():
    result = results[branch_name]['t=1']
    if not (result['r_converged'] and result['x_converged']):
        single_snap_failures.append(branch_name)
    elif result['r_error'] > 10 or result['x_error'] > 10:
        single_snap_failures.append(branch_name)

if single_snap_failures:
    print(f"\n  ✓ Single snapshot fails on end branches: {single_snap_failures}")
    print("    (Consistent with paper's observation)")
else:
    print("\n  ⚠ Single snapshot didn't clearly fail on end branches")
    print("    (May need more steps or different initialization)")

# Check if multi-snapshot helps
print("\n  Multi-snapshot improvement:")
for branch_name in END_BRANCHES.keys():
    r_single = results[branch_name]['t=1']
    r_multi = results[branch_name]['t=5']

    r_improve = r_single['r_error'] - r_multi['r_error']
    x_improve = r_single['x_error'] - r_multi['x_error']

    print(f"    {branch_name}: R {r_improve:+.2f}%, X {x_improve:+.2f}%")

# ========================================
# Generate Visualization
# ========================================

print("\n[2] Generating visualization...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot convergence for each end branch
for idx, (branch_name, branch_idx) in enumerate(END_BRANCHES.items()):
    if idx >= 4:
        break

    ax = axes[idx // 2, idx % 2]

    r_single = results[branch_name]['t=1']
    r_multi = results[branch_name]['t=5']

    # Plot R convergence
    ax.plot(r_single['r_history'], label='Single (R)', color='steelblue', alpha=0.7)
    ax.plot(r_multi['r_history'], label='Multi-5 (R)', color='coral', alpha=0.7)
    ax.axhline(y=r_single['r_true'], color='k', linestyle='--', alpha=0.5, label='True R')

    ax.set_xlabel('Step')
    ax.set_ylabel('R (Ohm/km)')
    ax.set_title(f'{branch_name}\nSingle: {r_single["r_error"]:.1f}% | Multi: {r_multi["r_error"]:.1f}%')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Summary bar chart
ax = axes[1, 2]

branch_names = list(END_BRANCHES.keys())
x_pos = np.arange(len(branch_names))
width = 0.35

single_errors = [results[b]['t=1']['r_error'] for b in branch_names]
multi_errors = [results[b]['t=5']['r_error'] for b in branch_names]

bars1 = ax.bar(x_pos - width/2, single_errors, width, label='Single (t=1)', color='steelblue')
bars2 = ax.bar(x_pos + width/2, multi_errors, width, label='Multi (t=5)', color='coral')

ax.set_ylabel('R Error (%)')
ax.set_title('End Branch: Single vs Multi-Snapshot')
ax.set_xticks(x_pos)
ax.set_xticklabels([b.replace('branch_', '') for b in branch_names], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add reference line from paper
ax.axhline(y=0.52, color='green', linestyle=':', alpha=0.7, label='Paper (21-22)')

# Regular branch comparison
ax = axes[0, 2]
reg_single = results['branch_3_4']['t=1']
reg_multi = results['branch_3_4']['t=5']

ax.plot(reg_single['r_history'], label='Single (R)', color='steelblue', alpha=0.7)
ax.plot(reg_multi['r_history'], label='Multi-5 (R)', color='coral', alpha=0.7)
ax.axhline(y=reg_single['r_true'], color='k', linestyle='--', alpha=0.5, label='True R')

ax.set_xlabel('Step')
ax.set_ylabel('R (Ohm/km)')
ax.set_title(f'Regular Branch 3-4\nSingle: {reg_single["r_error"]:.1f}% | Multi: {reg_multi["r_error"]:.1f}%')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tmp/phase1_end_branch_estimation.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: tmp/phase1_end_branch_estimation.png")

# Save results
import pickle
with open('tmp/phase1_end_branch_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("  ✓ Saved: tmp/phase1_end_branch_results.pkl")

print("\n" + "=" * 70)
print("✓ END BRANCH ESTIMATION TEST COMPLETE")
print("=" * 70)
