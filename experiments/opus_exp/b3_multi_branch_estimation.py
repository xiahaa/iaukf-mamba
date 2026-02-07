"""
B3: Multi-Branch Estimation Experiment
=======================================

This experiment tests simultaneous estimation of multiple line parameters,
matching the paper's Table II which shows results for 4+ branches.

Key aspects:
1. Estimate parameters for multiple branches simultaneously
2. Compare IAUKF vs Graph Mamba on multi-branch estimation
3. Analyze how estimation accuracy varies across branch types
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import pandapower as pp
from tqdm import tqdm
from model.simulation import PowerSystemSimulation
from model.models_analytical import AnalyticalMeasurementModel
from model.iaukf import IAUKF

# Try to import Graph Mamba
try:
    import torch
    from graphmamba.graph_mamba import GraphMambaModel, HAS_MAMBA
    MAMBA_AVAILABLE = HAS_MAMBA
except ImportError:
    MAMBA_AVAILABLE = False

np.random.seed(42)

print("=" * 80)
print("B3: MULTI-BRANCH PARAMETER ESTIMATION")
print("=" * 80)

# ============================================================================
# Configuration - Match Paper's Table II
# ============================================================================

# Select representative branches from different parts of the network
BRANCHES_TO_ESTIMATE = {
    # Main feeder branches (high power flow)
    'branch_1_2': {'idx': 0, 'type': 'main_feeder'},
    'branch_2_3': {'idx': 1, 'type': 'main_feeder'},
    'branch_3_4': {'idx': 3, 'type': 'main_feeder'},
    'branch_5_6': {'idx': 5, 'type': 'main_feeder'},
    
    # Lateral branches (medium power flow)
    'branch_2_19': {'idx': 18, 'type': 'lateral'},
    'branch_3_23': {'idx': 22, 'type': 'lateral'},
    
    # End branches (low power flow)
    'branch_17_18': {'idx': 17, 'type': 'end'},
    'branch_21_22': {'idx': 20, 'type': 'end'},
}

IAUKF_STEPS = 150
NUM_RUNS = 5  # Multiple runs for statistical significance

print(f"\nConfiguration:")
print(f"  Branches to estimate: {len(BRANCHES_TO_ESTIMATE)}")
print(f"  IAUKF steps per branch: {IAUKF_STEPS}")
print(f"  Number of runs: {NUM_RUNS}")
print(f"  Branch types: {set(info['type'] for info in BRANCHES_TO_ESTIMATE.values())}")

# ============================================================================
# Helper: Run IAUKF for Single Branch
# ============================================================================

def run_iaukf_single_branch(branch_idx, steps=IAUKF_STEPS, seed=42):
    """Run IAUKF to estimate parameters of a single branch."""
    np.random.seed(seed)
    
    sim = PowerSystemSimulation(steps=steps)
    
    r_true = sim.net.line.at[branch_idx, 'r_ohm_per_km']
    x_true = sim.net.line.at[branch_idx, 'x_ohm_per_km']
    
    # Create analytical model for target branch
    model = AnalyticalMeasurementModel(sim.net, branch_idx, sim.pmu_buses)
    num_buses = len(sim.net.bus)
    
    # Initial state
    x0 = np.ones(2 * num_buses + 2)
    x0[:num_buses] = 1.0
    x0[num_buses:2*num_buses] = 0.0
    x0[-2] = 0.1  # Initial R guess
    x0[-1] = 0.1  # Initial X guess
    
    # Covariances
    P0 = np.eye(len(x0)) * 0.01
    P0[-2, -2] = 0.1
    P0[-1, -1] = 0.1
    Q0 = np.eye(len(x0)) * 1e-6
    
    n_scada = 3 * num_buses
    R_diag = np.concatenate([
        np.full(n_scada, 0.02**2),
        np.full(len(sim.pmu_buses), 0.005**2),
        np.full(len(sim.pmu_buses), 0.002**2)
    ])
    R = np.diag(R_diag)
    
    # Create IAUKF
    iaukf = IAUKF(model, x0, P0, Q0, R)
    iaukf.b_factor = 0.96
    
    # Generate measurements and run filter
    p_base = sim.net.load.p_mw.values.copy()
    q_base = sim.net.load.q_mvar.values.copy()
    
    r_history = []
    x_history = []
    
    for t in range(steps):
        # Constant loads (matching paper's setup)
        sim.net.load.p_mw = p_base
        sim.net.load.q_mvar = q_base
        
        try:
            pp.runpp(sim.net, algorithm='nr', numba=False)
        except:
            continue
        
        # Measurements
        p_inj = -sim.net.res_bus.p_mw.values + np.random.normal(0, 0.02, num_buses)
        q_inj = -sim.net.res_bus.q_mvar.values + np.random.normal(0, 0.02, num_buses)
        v_scada = sim.net.res_bus.vm_pu.values + np.random.normal(0, 0.02, num_buses)
        z_scada = np.concatenate([p_inj, q_inj, v_scada])
        
        v_pmu = sim.net.res_bus.vm_pu.values[sim.pmu_buses] + np.random.normal(0, 0.005, len(sim.pmu_buses))
        theta_pmu = np.radians(sim.net.res_bus.va_degree.values[sim.pmu_buses]) + np.random.normal(0, 0.002, len(sim.pmu_buses))
        z_pmu = np.concatenate([v_pmu, theta_pmu])
        
        z = np.concatenate([z_scada, z_pmu])
        
        iaukf.predict()
        iaukf.update(z)
        
        r_history.append(iaukf.x[-2])
        x_history.append(iaukf.x[-1])
    
    # Post-convergence averaging
    start_avg = len(r_history) // 2
    r_final = np.mean(r_history[start_avg:])
    x_final = np.mean(x_history[start_avg:])
    
    r_error = abs(r_final - r_true) / r_true * 100
    x_error = abs(x_final - x_true) / x_true * 100
    
    return {
        'r_true': r_true, 'x_true': x_true,
        'r_pred': r_final, 'x_pred': x_final,
        'r_error': r_error, 'x_error': x_error,
        'r_history': r_history, 'x_history': x_history
    }

# ============================================================================
# Run Multi-Branch IAUKF Estimation
# ============================================================================

print("\n[1] Running IAUKF on multiple branches...")

all_results = {}

for run in range(NUM_RUNS):
    print(f"\n  Run {run + 1}/{NUM_RUNS}:")
    run_results = {}
    
    for name, info in tqdm(BRANCHES_TO_ESTIMATE.items(), desc=f"    Branches"):
        result = run_iaukf_single_branch(info['idx'], seed=42 + run * 100)
        result['type'] = info['type']
        run_results[name] = result
    
    all_results[f'run_{run}'] = run_results

# ============================================================================
# Aggregate Results
# ============================================================================

print("\n[2] Aggregating results...")

# Compute statistics across runs
aggregated = {}
for name in BRANCHES_TO_ESTIMATE.keys():
    r_errors = [all_results[f'run_{r}'][name]['r_error'] for r in range(NUM_RUNS)]
    x_errors = [all_results[f'run_{r}'][name]['x_error'] for r in range(NUM_RUNS)]
    
    aggregated[name] = {
        'r_mean': np.mean(r_errors),
        'r_std': np.std(r_errors),
        'x_mean': np.mean(x_errors),
        'x_std': np.std(x_errors),
        'type': BRANCHES_TO_ESTIMATE[name]['type'],
        'r_true': all_results['run_0'][name]['r_true'],
        'x_true': all_results['run_0'][name]['x_true'],
    }

# ============================================================================
# Results Summary
# ============================================================================

print("\n" + "=" * 80)
print("MULTI-BRANCH ESTIMATION RESULTS")
print("=" * 80)

# Create results table
print("\n{:<15} {:<12} {:>10} {:>10} {:>10} {:>10}".format(
    "Branch", "Type", "R True", "R Error%", "X True", "X Error%"))
print("-" * 67)

for name, agg in aggregated.items():
    print("{:<15} {:<12} {:>10.4f} {:>9.2f}% {:>10.4f} {:>9.2f}%".format(
        name.replace('branch_', ''), 
        agg['type'],
        agg['r_true'],
        agg['r_mean'],
        agg['x_true'],
        agg['x_mean']))

# Summary by type
print("\n" + "-" * 67)
print("\nSUMMARY BY BRANCH TYPE:")
print("-" * 50)

for branch_type in ['main_feeder', 'lateral', 'end']:
    type_results = [v for v in aggregated.values() if v['type'] == branch_type]
    if type_results:
        avg_r = np.mean([r['r_mean'] for r in type_results])
        avg_x = np.mean([r['x_mean'] for r in type_results])
        print(f"  {branch_type:15s}: R avg = {avg_r:6.2f}%, X avg = {avg_x:6.2f}%")

# Overall
all_r = [v['r_mean'] for v in aggregated.values()]
all_x = [v['x_mean'] for v in aggregated.values()]
print(f"\n  {'OVERALL':15s}: R avg = {np.mean(all_r):6.2f}%, X avg = {np.mean(all_x):6.2f}%")

# ============================================================================
# Generate Visualization
# ============================================================================

print("\n[3] Generating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: R Error by Branch Type
ax = axes[0, 0]
branch_names = list(aggregated.keys())
branch_types = [aggregated[n]['type'] for n in branch_names]
r_errors = [aggregated[n]['r_mean'] for n in branch_names]
r_stds = [aggregated[n]['r_std'] for n in branch_names]

colors = {'main_feeder': 'steelblue', 'lateral': 'forestgreen', 'end': 'coral'}
bar_colors = [colors[t] for t in branch_types]

x_pos = np.arange(len(branch_names))
bars = ax.bar(x_pos, r_errors, color=bar_colors, yerr=r_stds, capsize=3)
ax.set_xticks(x_pos)
ax.set_xticklabels([n.replace('branch_', '') for n in branch_names], rotation=45, ha='right')
ax.set_ylabel('R Estimation Error (%)')
ax.set_title('R Parameter Error by Branch')
ax.grid(True, alpha=0.3, axis='y')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=t) for t, c in colors.items()]
ax.legend(handles=legend_elements, loc='upper right')

# Plot 2: X Error by Branch Type
ax = axes[0, 1]
x_errors = [aggregated[n]['x_mean'] for n in branch_names]
x_stds = [aggregated[n]['x_std'] for n in branch_names]

bars = ax.bar(x_pos, x_errors, color=bar_colors, yerr=x_stds, capsize=3)
ax.set_xticks(x_pos)
ax.set_xticklabels([n.replace('branch_', '') for n in branch_names], rotation=45, ha='right')
ax.set_ylabel('X Estimation Error (%)')
ax.set_title('X Parameter Error by Branch')
ax.grid(True, alpha=0.3, axis='y')
ax.legend(handles=legend_elements, loc='upper right')

# Plot 3: Error by Branch Type Summary
ax = axes[1, 0]
types = ['main_feeder', 'lateral', 'end']
r_by_type = []
x_by_type = []

for t in types:
    type_r = [v['r_mean'] for v in aggregated.values() if v['type'] == t]
    type_x = [v['x_mean'] for v in aggregated.values() if v['type'] == t]
    r_by_type.append(np.mean(type_r) if type_r else 0)
    x_by_type.append(np.mean(type_x) if type_x else 0)

x_pos = np.arange(len(types))
width = 0.35
bars1 = ax.bar(x_pos - width/2, r_by_type, width, label='R Error', color='steelblue')
bars2 = ax.bar(x_pos + width/2, x_by_type, width, label='X Error', color='coral')

ax.set_xticks(x_pos)
ax.set_xticklabels(['Main Feeder', 'Lateral', 'End Branch'])
ax.set_ylabel('Average Error (%)')
ax.set_title('Error by Branch Type')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Convergence for one branch of each type
ax = axes[1, 1]
sample_branches = {
    'main': 'branch_3_4',
    'lateral': 'branch_2_19',
    'end': 'branch_21_22'
}

for label, name in sample_branches.items():
    if name in all_results['run_0']:
        history = all_results['run_0'][name]['r_history']
        true_val = all_results['run_0'][name]['r_true']
        ax.plot(history, label=f'{label} (true={true_val:.3f})')

ax.set_xlabel('IAUKF Step')
ax.set_ylabel('R Estimate (Ohm/km)')
ax.set_title('IAUKF Convergence by Branch Type')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tmp/b3_multi_branch_results.png', dpi=150, bbox_inches='tight')
print("  Saved: tmp/b3_multi_branch_results.png")

# ============================================================================
# Save Results
# ============================================================================

results = {
    'all_results': all_results,
    'aggregated': aggregated,
    'config': {
        'branches': BRANCHES_TO_ESTIMATE,
        'iaukf_steps': IAUKF_STEPS,
        'num_runs': NUM_RUNS
    }
}

os.makedirs('tmp', exist_ok=True)
with open('tmp/b3_multi_branch_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("  Saved: tmp/b3_multi_branch_results.pkl")

# ============================================================================
# Paper Table II Format
# ============================================================================

print("\n" + "=" * 80)
print("TABLE II FORMAT (for paper)")
print("=" * 80)

print("\n\\begin{table}[h]")
print("\\centering")
print("\\caption{Multi-Branch Parameter Estimation Results}")
print("\\begin{tabular}{lcccc}")
print("\\hline")
print("Branch & Type & R Error (\\%) & X Error (\\%) \\\\")
print("\\hline")

for name, agg in aggregated.items():
    clean_name = name.replace('branch_', '').replace('_', '-')
    print(f"{clean_name} & {agg['type']} & {agg['r_mean']:.2f} $\\pm$ {agg['r_std']:.2f} & {agg['x_mean']:.2f} $\\pm$ {agg['x_std']:.2f} \\\\")

print("\\hline")
print(f"Average & -- & {np.mean(all_r):.2f} & {np.mean(all_x):.2f} \\\\")
print("\\hline")
print("\\end{tabular}")
print("\\end{table}")

print("\n" + "=" * 80)
print("B3 EXPERIMENT COMPLETE")
print("=" * 80)
