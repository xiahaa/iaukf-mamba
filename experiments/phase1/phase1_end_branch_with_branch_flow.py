"""
Phase 1: End Branch Estimation with Branch Power Flow Measurements
===================================================================

This experiment tests whether adding branch power flow measurements (P_ij, Q_ij)
improves end branch estimation, as suggested by the paper's Eq. 21.

Reference paper claims for branch 21-22 with multi-snapshot: R=0.52%, X=2.03%
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib.pyplot as plt
import pandapower as pp
from tqdm import tqdm
from model.simulation import PowerSystemSimulation
from model.models_analytical import AnalyticalMeasurementModel, AnalyticalMeasurementModelWithBranchFlow
from model.iaukf import IAUKF, IAUKFMultiSnapshot

# ========================================
# Configuration
# ========================================

BRANCHES = {
    'branch_3_4': 3,      # Regular branch (should work)
    'branch_21_22': 20,   # End branch (paper claims 0.52%, 2.03%)
    'branch_32_33': 31,   # End branch
}

STEPS = 100  # Reduced for faster testing
NUM_SNAPSHOTS = 5
SEED = 42

print("=" * 70)
print("END BRANCH ESTIMATION WITH BRANCH POWER FLOW MEASUREMENTS")
print("=" * 70)

# ========================================
# Helper: Run IAUKF with Branch Flow Model
# ========================================

def run_iaukf_with_branch_flow(branch_idx, num_snapshots=1, steps=300, use_branch_flow=True):
    """
    Run IAUKF with enhanced model including branch power flow measurements.
    """
    # Create simulation
    sim = PowerSystemSimulation(steps=steps)
    sim.target_line_idx = branch_idx
    sim.line_idx = branch_idx
    
    # Get true parameters
    r_true = sim.net.line.at[branch_idx, 'r_ohm_per_km']
    x_true = sim.net.line.at[branch_idx, 'x_ohm_per_km']
    from_bus = int(sim.net.line.at[branch_idx, 'from_bus'])
    to_bus = int(sim.net.line.at[branch_idx, 'to_bus'])
    
    # Create model (with or without branch flow)
    if use_branch_flow:
        model = AnalyticalMeasurementModelWithBranchFlow(sim.net, branch_idx, sim.pmu_buses)
    else:
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
        
        # SCADA measurements
        num_buses = len(sim.net.bus)
        p_inj = -sim.net.res_bus.p_mw.values
        q_inj = -sim.net.res_bus.q_mvar.values
        v_scada = sim.net.res_bus.vm_pu.values
        z_scada = np.concatenate([p_inj, q_inj, v_scada])
        z_scada += np.random.normal(0, 0.02, len(z_scada))
        
        # Branch power flow measurements (if using enhanced model)
        if use_branch_flow:
            P_ij = sim.net.res_line.at[branch_idx, 'p_from_mw']
            Q_ij = sim.net.res_line.at[branch_idx, 'q_from_mvar']
            # Add noise to branch measurements
            P_ij += np.random.normal(0, 0.01)  # Lower noise for branch flow
            Q_ij += np.random.normal(0, 0.01)
            z_branch = np.array([P_ij, Q_ij])
        
        # PMU measurements
        v_pmu = sim.net.res_bus.vm_pu.values[sim.pmu_buses]
        theta_pmu = np.radians(sim.net.res_bus.va_degree.values[sim.pmu_buses])
        z_pmu = np.concatenate([v_pmu, theta_pmu])
        z_pmu += np.concatenate([
            np.random.normal(0, 0.005, len(v_pmu)),
            np.random.normal(0, 0.002, len(theta_pmu))
        ])
        
        # Combine measurements
        if use_branch_flow:
            measurements.append(np.concatenate([z_scada, z_branch, z_pmu]))
        else:
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
    if use_branch_flow:
        R_diag = np.concatenate([
            np.full(n_scada, 0.02**2),            # SCADA
            np.full(2, 0.01**2),                  # Branch flow (lower noise)
            np.full(len(sim.pmu_buses), 0.005**2), # PMU V
            np.full(len(sim.pmu_buses), 0.002**2)  # PMU θ
        ])
    else:
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
    
    # Post-convergence averaging
    start_avg = len(r_history) // 2
    r_final = np.mean(r_history[start_avg:])
    x_final = np.mean(x_history[start_avg:])
    
    r_error = abs(r_final - r_true) / r_true * 100
    x_error = abs(x_final - x_true) / x_true * 100
    
    return {
        'r_true': r_true,
        'x_true': x_true,
        'r_final': r_final,
        'x_final': x_final,
        'r_error': r_error,
        'x_error': x_error,
        'r_history': r_history,
        'x_history': x_history,
        'from_bus': from_bus,
        'to_bus': to_bus
    }


# ========================================
# Run Experiments
# ========================================

print("\n[1] Running experiments...")

results = {}

for branch_name, branch_idx in tqdm(BRANCHES.items(), desc="Branches"):
    results[branch_name] = {}
    
    # Without branch flow (original)
    print(f"\n  {branch_name}: Testing without branch flow...")
    result_no_bf = run_iaukf_with_branch_flow(branch_idx, num_snapshots=NUM_SNAPSHOTS, 
                                               steps=STEPS, use_branch_flow=False)
    results[branch_name]['no_branch_flow'] = result_no_bf
    
    # With branch flow (enhanced)
    print(f"  {branch_name}: Testing WITH branch flow...")
    result_bf = run_iaukf_with_branch_flow(branch_idx, num_snapshots=NUM_SNAPSHOTS, 
                                           steps=STEPS, use_branch_flow=True)
    results[branch_name]['with_branch_flow'] = result_bf

# ========================================
# Results Summary
# ========================================

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

print("\n╔═══════════════════════════════════════════════════════════════════════════╗")
print("║          EFFECT OF ADDING BRANCH POWER FLOW MEASUREMENTS                  ║")
print("╠═══════════════════════════════════════════════════════════════════════════╣")
print("║ Branch       │ Model           │ R Error  │ X Error  │ Improvement       ║")
print("╠═══════════════════════════════════════════════════════════════════════════╣")

for branch_name in BRANCHES.keys():
    r_no = results[branch_name]['no_branch_flow']
    r_bf = results[branch_name]['with_branch_flow']
    
    r_improve = r_no['r_error'] - r_bf['r_error']
    x_improve = r_no['x_error'] - r_bf['x_error']
    
    print(f"║ {branch_name:12s} │ No Branch Flow  │ {r_no['r_error']:7.2f}% │ {r_no['x_error']:7.2f}% │                   ║")
    print(f"║              │ + Branch Flow   │ {r_bf['r_error']:7.2f}% │ {r_bf['x_error']:7.2f}% │ R:{r_improve:+.2f}% X:{x_improve:+.2f}% ║")
    print("╠═══════════════════════════════════════════════════════════════════════════╣")

print("╚═══════════════════════════════════════════════════════════════════════════╝")

# Paper comparison for branch 21-22
print("\n" + "-" * 70)
print("PAPER REFERENCE (Branch 21-22 with multi-snapshot t=5):")
print("-" * 70)
print("  R error: 0.52%")
print("  X error: 2.03%")
print("-" * 70)
print("OUR RESULTS (Branch 21-22 with branch flow + multi-snapshot t=5):")
r_21_22 = results['branch_21_22']['with_branch_flow']
print(f"  R error: {r_21_22['r_error']:.2f}%")
print(f"  X error: {r_21_22['x_error']:.2f}%")
print("-" * 70)

# ========================================
# Generate Visualization
# ========================================

print("\n[2] Generating visualization...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, (branch_name, branch_idx) in enumerate(BRANCHES.items()):
    ax = axes[0, idx]
    
    r_no = results[branch_name]['no_branch_flow']
    r_bf = results[branch_name]['with_branch_flow']
    
    ax.plot(r_no['r_history'], label='No BF (R)', color='blue', alpha=0.5)
    ax.plot(r_bf['r_history'], label='+ BF (R)', color='blue', linewidth=2)
    ax.axhline(y=r_no['r_true'], color='k', linestyle='--', alpha=0.7, label='True R')
    
    ax.set_xlabel('Step')
    ax.set_ylabel('R (Ohm/km)')
    ax.set_title(f'{branch_name}\nNo BF: {r_no["r_error"]:.1f}% → +BF: {r_bf["r_error"]:.1f}%')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Bar chart comparison
ax = axes[1, 0]
branch_names = list(BRANCHES.keys())
x_pos = np.arange(len(branch_names))
width = 0.35

no_bf_errors = [results[b]['no_branch_flow']['r_error'] for b in branch_names]
bf_errors = [results[b]['with_branch_flow']['r_error'] for b in branch_names]

bars1 = ax.bar(x_pos - width/2, no_bf_errors, width, label='No Branch Flow', color='steelblue')
bars2 = ax.bar(x_pos + width/2, bf_errors, width, label='+ Branch Flow', color='coral')

ax.set_ylabel('R Error (%)')
ax.set_title('R Error: Effect of Branch Flow Measurements')
ax.set_xticks(x_pos)
ax.set_xticklabels([b.replace('branch_', '') for b in branch_names])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add paper reference line
ax.axhline(y=0.52, color='green', linestyle=':', linewidth=2, label='Paper (0.52%)')

# X error comparison
ax = axes[1, 1]
no_bf_x_errors = [results[b]['no_branch_flow']['x_error'] for b in branch_names]
bf_x_errors = [results[b]['with_branch_flow']['x_error'] for b in branch_names]

bars1 = ax.bar(x_pos - width/2, no_bf_x_errors, width, label='No Branch Flow', color='steelblue')
bars2 = ax.bar(x_pos + width/2, bf_x_errors, width, label='+ Branch Flow', color='coral')

ax.set_ylabel('X Error (%)')
ax.set_title('X Error: Effect of Branch Flow Measurements')
ax.set_xticks(x_pos)
ax.set_xticklabels([b.replace('branch_', '') for b in branch_names])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

ax.axhline(y=2.03, color='green', linestyle=':', linewidth=2, label='Paper (2.03%)')

# Improvement summary
ax = axes[1, 2]
improvements_r = [results[b]['no_branch_flow']['r_error'] - results[b]['with_branch_flow']['r_error'] for b in branch_names]
improvements_x = [results[b]['no_branch_flow']['x_error'] - results[b]['with_branch_flow']['x_error'] for b in branch_names]

bars1 = ax.bar(x_pos - width/2, improvements_r, width, label='R Improvement', color='steelblue')
bars2 = ax.bar(x_pos + width/2, improvements_x, width, label='X Improvement', color='coral')

ax.set_ylabel('Error Reduction (%)')
ax.set_title('Improvement from Adding Branch Flow')
ax.set_xticks(x_pos)
ax.set_xticklabels([b.replace('branch_', '') for b in branch_names])
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('tmp/phase1_end_branch_with_branch_flow.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: tmp/phase1_end_branch_with_branch_flow.png")

# Save results
import pickle
with open('tmp/phase1_end_branch_with_branch_flow.pkl', 'wb') as f:
    pickle.dump(results, f)
print("  ✓ Saved: tmp/phase1_end_branch_with_branch_flow.pkl")

print("\n" + "=" * 70)
print("✓ END BRANCH TEST WITH BRANCH FLOW COMPLETE")
print("=" * 70)
