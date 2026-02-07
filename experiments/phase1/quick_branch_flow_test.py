"""
Quick test of branch flow measurement impact on IAUKF for end branches.
Simplified version for faster testing.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib.pyplot as plt
import pandapower as pp
from model.simulation import PowerSystemSimulation
from model.models_analytical import AnalyticalMeasurementModel, AnalyticalMeasurementModelWithBranchFlow
from model.iaukf import IAUKF
import time

# ========================================
# Configuration
# ========================================

BRANCHES = {
    'branch_3_4': 3,      # Regular branch
    'branch_21_22': 20,   # End branch (paper claims 0.52%, 2.03%)
}

STEPS = 50  # Fewer steps for quick test
SEED = 42

print("=" * 70)
print("QUICK BRANCH FLOW MEASUREMENT TEST")
print("=" * 70)

def run_iaukf_quick(branch_idx, use_branch_flow=True, steps=STEPS):
    """Run IAUKF with pre-generated measurements."""
    np.random.seed(SEED)
    
    # Create simulation
    sim = PowerSystemSimulation(steps=steps)
    sim.target_line_idx = branch_idx
    
    # Get true parameters
    r_true = sim.net.line.at[branch_idx, 'r_ohm_per_km']
    x_true = sim.net.line.at[branch_idx, 'x_ohm_per_km']
    from_bus = int(sim.net.line.at[branch_idx, 'from_bus'])
    to_bus = int(sim.net.line.at[branch_idx, 'to_bus'])
    
    print(f"  Branch {from_bus}-{to_bus}: R_true={r_true:.4f}, X_true={x_true:.4f}")
    
    # Create model
    if use_branch_flow:
        model = AnalyticalMeasurementModelWithBranchFlow(sim.net, branch_idx, sim.pmu_buses)
    else:
        model = AnalyticalMeasurementModel(sim.net, branch_idx, sim.pmu_buses)
    
    num_buses = len(sim.net.bus)
    
    # Run power flow once for base measurements
    pp.runpp(sim.net, algorithm='nr', numba=False)
    
    # Generate all measurements at once (with slight random variations)
    measurements = []
    for t in range(steps):
        # Small random load variations
        load_factor = 1.0 + np.random.normal(0, 0.05)
        sim.net.load.p_mw = sim.net.load.p_mw * load_factor
        sim.net.load.q_mvar = sim.net.load.q_mvar * load_factor
        
        try:
            pp.runpp(sim.net, algorithm='nr', numba=False)
        except:
            continue
        
        # SCADA measurements
        p_inj = -sim.net.res_bus.p_mw.values
        q_inj = -sim.net.res_bus.q_mvar.values
        v_scada = sim.net.res_bus.vm_pu.values
        z_scada = np.concatenate([p_inj, q_inj, v_scada])
        z_scada += np.random.normal(0, 0.02, len(z_scada))
        
        # Branch power flow
        if use_branch_flow:
            P_ij = sim.net.res_line.at[branch_idx, 'p_from_mw']
            Q_ij = sim.net.res_line.at[branch_idx, 'q_from_mvar']
            z_branch = np.array([P_ij + np.random.normal(0, 0.01),
                                 Q_ij + np.random.normal(0, 0.01)])
        
        # PMU measurements
        v_pmu = sim.net.res_bus.vm_pu.values[sim.pmu_buses]
        theta_pmu = np.radians(sim.net.res_bus.va_degree.values[sim.pmu_buses])
        z_pmu = np.concatenate([v_pmu + np.random.normal(0, 0.005, len(v_pmu)),
                                theta_pmu + np.random.normal(0, 0.002, len(theta_pmu))])
        
        if use_branch_flow:
            measurements.append(np.concatenate([z_scada, z_branch, z_pmu]))
        else:
            measurements.append(np.concatenate([z_scada, z_pmu]))
    
    # Reset loads
    sim.net.load.p_mw = sim.net.load.p_mw.values.copy()
    sim.net.load.q_mvar = sim.net.load.q_mvar.values.copy()
    
    # Initial state - use better initial guess
    x0 = np.ones(2 * num_buses + 2)
    x0[:num_buses] = 1.0
    x0[num_buses:2*num_buses] = 0.0
    x0[-2] = 0.1  # Better initial R guess
    x0[-1] = 0.1  # Better initial X guess
    
    # Covariances
    P0 = np.eye(len(x0)) * 0.01
    P0[-2, -2] = 0.1
    P0[-1, -1] = 0.1
    
    Q0 = np.eye(len(x0)) * 1e-6
    Q0[-2, -2] = 1e-6
    Q0[-1, -1] = 1e-6
    
    # Measurement covariance
    n_scada = 3 * num_buses
    if use_branch_flow:
        R_diag = np.concatenate([
            np.full(n_scada, 0.02**2),
            np.full(2, 0.01**2),  # Branch flow
            np.full(len(sim.pmu_buses), 0.005**2),
            np.full(len(sim.pmu_buses), 0.002**2)
        ])
    else:
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
    
    t0 = time.time()
    for t, z in enumerate(measurements):
        iaukf.predict()
        iaukf.update(z)
        r_history.append(iaukf.x[-2])
        x_history.append(iaukf.x[-1])
    elapsed = time.time() - t0
    
    r_history = np.array(r_history)
    x_history = np.array(x_history)
    
    # Post-convergence averaging
    start_avg = len(r_history) // 2
    r_final = np.mean(r_history[start_avg:])
    x_final = np.mean(x_history[start_avg:])
    
    r_error = abs(r_final - r_true) / r_true * 100
    x_error = abs(x_final - x_true) / x_true * 100
    
    return {
        'r_true': r_true, 'x_true': x_true,
        'r_final': r_final, 'x_final': x_final,
        'r_error': r_error, 'x_error': x_error,
        'r_history': r_history, 'x_history': x_history,
        'from_bus': from_bus, 'to_bus': to_bus,
        'elapsed': elapsed
    }

# ========================================
# Run Experiments
# ========================================

print("\n[1] Running experiments...")

results = {}

for branch_name, branch_idx in BRANCHES.items():
    results[branch_name] = {}
    print(f"\n{branch_name}:")
    
    # Without branch flow
    print("  Without branch flow...")
    result_no_bf = run_iaukf_quick(branch_idx, use_branch_flow=False)
    results[branch_name]['no_branch_flow'] = result_no_bf
    print(f"    R error: {result_no_bf['r_error']:.2f}%, X error: {result_no_bf['x_error']:.2f}% ({result_no_bf['elapsed']:.1f}s)")
    
    # With branch flow
    print("  With branch flow...")
    result_bf = run_iaukf_quick(branch_idx, use_branch_flow=True)
    results[branch_name]['with_branch_flow'] = result_bf
    print(f"    R error: {result_bf['r_error']:.2f}%, X error: {result_bf['x_error']:.2f}% ({result_bf['elapsed']:.1f}s)")

# ========================================
# Results Summary
# ========================================

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

print("\n{:<15} {:<20} {:>10} {:>10}".format("Branch", "Model", "R Error", "X Error"))
print("-" * 55)

for branch_name in BRANCHES.keys():
    r_no = results[branch_name]['no_branch_flow']
    r_bf = results[branch_name]['with_branch_flow']
    
    print("{:<15} {:<20} {:>9.2f}% {:>9.2f}%".format(
        branch_name, "No Branch Flow", r_no['r_error'], r_no['x_error']))
    print("{:<15} {:<20} {:>9.2f}% {:>9.2f}%".format(
        "", "+ Branch Flow", r_bf['r_error'], r_bf['x_error']))
    
    r_improve = r_no['r_error'] - r_bf['r_error']
    x_improve = r_no['x_error'] - r_bf['x_error']
    print("{:<15} {:<20} {:>+9.2f}% {:>+9.2f}%".format(
        "", "Improvement", r_improve, x_improve))
    print()

# Paper comparison
print("-" * 70)
print("PAPER REFERENCE (Branch 21-22):")
print("  R error: 0.52%")
print("  X error: 2.03%")
print("-" * 70)

# ========================================
# Visualization
# ========================================

print("\n[2] Generating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, (branch_name, branch_idx) in enumerate(BRANCHES.items()):
    ax = axes[0, idx]
    
    r_no = results[branch_name]['no_branch_flow']
    r_bf = results[branch_name]['with_branch_flow']
    
    ax.plot(r_no['r_history'], label='No BF (R)', color='blue', alpha=0.5)
    ax.plot(r_bf['r_history'], label='+ BF (R)', color='blue', linewidth=2)
    ax.axhline(y=r_no['r_true'], color='red', linestyle='--', alpha=0.7, label='True R')
    
    ax.set_xlabel('Step')
    ax.set_ylabel('R (Ohm/km)')
    ax.set_title(f'{branch_name}\nNo BF: {r_no["r_error"]:.1f}% → +BF: {r_bf["r_error"]:.1f}%')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Bar chart
ax = axes[1, 0]
branch_names = list(BRANCHES.keys())
x_pos = np.arange(len(branch_names))
width = 0.35

no_bf_errors = [results[b]['no_branch_flow']['r_error'] for b in branch_names]
bf_errors = [results[b]['with_branch_flow']['r_error'] for b in branch_names]

bars1 = ax.bar(x_pos - width/2, no_bf_errors, width, label='No Branch Flow', color='steelblue')
bars2 = ax.bar(x_pos + width/2, bf_errors, width, label='+ Branch Flow', color='coral')

ax.set_ylabel('R Error (%)')
ax.set_title('R Error Comparison')
ax.set_xticks(x_pos)
ax.set_xticklabels([b.replace('branch_', '') for b in branch_names])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0.52, color='green', linestyle=':', linewidth=2, label='Paper (0.52%)')

# X error
ax = axes[1, 1]
no_bf_x_errors = [results[b]['no_branch_flow']['x_error'] for b in branch_names]
bf_x_errors = [results[b]['with_branch_flow']['x_error'] for b in branch_names]

bars1 = ax.bar(x_pos - width/2, no_bf_x_errors, width, label='No Branch Flow', color='steelblue')
bars2 = ax.bar(x_pos + width/2, bf_x_errors, width, label='+ Branch Flow', color='coral')

ax.set_ylabel('X Error (%)')
ax.set_title('X Error Comparison')
ax.set_xticks(x_pos)
ax.set_xticklabels([b.replace('branch_', '') for b in branch_names])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=2.03, color='green', linestyle=':', linewidth=2, label='Paper (2.03%)')

plt.tight_layout()
plt.savefig('tmp/quick_branch_flow_test.png', dpi=150, bbox_inches='tight')
print("  Saved: tmp/quick_branch_flow_test.png")

print("\n" + "=" * 70)
print("QUICK TEST COMPLETE")
print("=" * 70)
