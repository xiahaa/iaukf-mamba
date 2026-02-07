"""
B4: Bad Data Robustness Experiment
===================================

This experiment tests the robustness of IAUKF and Graph Mamba to bad data,
as mentioned in the paper (10% bad data: 5% missing + 5% erroneous).

Key scenarios:
1. Missing data (5%): Some measurements unavailable
2. Erroneous data (5%): Some measurements have gross errors
3. Combined (10%): Both missing and erroneous data

This is critical for real-world deployment where data quality varies.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import pickle
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
print("B4: BAD DATA ROBUSTNESS TEST")
print("=" * 80)

# ============================================================================
# Configuration
# ============================================================================

BAD_DATA_SCENARIOS = {
    'clean': {'missing_rate': 0.0, 'error_rate': 0.0},
    'missing_5pct': {'missing_rate': 0.05, 'error_rate': 0.0},
    'error_5pct': {'missing_rate': 0.0, 'error_rate': 0.05},
    'combined_10pct': {'missing_rate': 0.05, 'error_rate': 0.05},
    'severe_20pct': {'missing_rate': 0.10, 'error_rate': 0.10},
}

IAUKF_STEPS = 150
NUM_RUNS = 3
TARGET_BRANCH = 3  # Branch 3-4

print(f"\nConfiguration:")
print(f"  Target branch: {TARGET_BRANCH}")
print(f"  IAUKF steps: {IAUKF_STEPS}")
print(f"  Number of runs: {NUM_RUNS}")
print(f"  Bad data scenarios: {list(BAD_DATA_SCENARIOS.keys())}")

# ============================================================================
# Helper: Apply Bad Data
# ============================================================================

def apply_bad_data(z, missing_rate=0.0, error_rate=0.0, error_magnitude=5.0):
    """
    Apply bad data corruption to measurement vector.
    
    Args:
        z: Clean measurement vector
        missing_rate: Fraction of measurements to set as missing (replaced with NaN or zero)
        error_rate: Fraction of measurements with gross errors
        error_magnitude: Multiplier for gross error standard deviation
    
    Returns:
        Corrupted measurement vector, mask of valid measurements
    """
    z_corrupted = z.copy()
    n = len(z)
    
    # Track valid measurements
    valid_mask = np.ones(n, dtype=bool)
    
    # Apply missing data (replace with last known value or zero)
    if missing_rate > 0:
        n_missing = int(n * missing_rate)
        missing_idx = np.random.choice(n, n_missing, replace=False)
        z_corrupted[missing_idx] = 0  # Replace with zero (or could use interpolation)
        valid_mask[missing_idx] = False
    
    # Apply gross errors (outliers)
    if error_rate > 0:
        n_errors = int(n * error_rate)
        error_idx = np.random.choice(n, n_errors, replace=False)
        # Gross errors: significantly larger than normal noise
        for idx in error_idx:
            error_sign = np.random.choice([-1, 1])
            gross_error = error_sign * error_magnitude * np.abs(z[idx]) * np.random.uniform(0.5, 1.5)
            z_corrupted[idx] += gross_error
    
    return z_corrupted, valid_mask

# ============================================================================
# Helper: Run IAUKF with Bad Data
# ============================================================================

def run_iaukf_with_bad_data(branch_idx, missing_rate=0.0, error_rate=0.0, steps=IAUKF_STEPS, seed=42):
    """Run IAUKF with corrupted measurements."""
    np.random.seed(seed)
    
    sim = PowerSystemSimulation(steps=steps)
    
    r_true = sim.net.line.at[branch_idx, 'r_ohm_per_km']
    x_true = sim.net.line.at[branch_idx, 'x_ohm_per_km']
    
    model = AnalyticalMeasurementModel(sim.net, branch_idx, sim.pmu_buses)
    num_buses = len(sim.net.bus)
    
    # Initial state
    x0 = np.ones(2 * num_buses + 2)
    x0[:num_buses] = 1.0
    x0[num_buses:2*num_buses] = 0.0
    x0[-2] = 0.1
    x0[-1] = 0.1
    
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
    
    # Increase R for scenarios with bad data (robustness measure)
    if missing_rate > 0 or error_rate > 0:
        R = R * (1 + 2 * (missing_rate + error_rate))  # Inflate noise assumption
    
    iaukf = IAUKF(model, x0, P0, Q0, R)
    iaukf.b_factor = 0.96
    
    p_base = sim.net.load.p_mw.values.copy()
    q_base = sim.net.load.q_mvar.values.copy()
    
    r_history = []
    x_history = []
    num_bad_measurements = 0
    
    for t in range(steps):
        sim.net.load.p_mw = p_base
        sim.net.load.q_mvar = q_base
        
        try:
            pp.runpp(sim.net, algorithm='nr', numba=False)
        except:
            continue
        
        # Generate clean measurements
        p_inj = -sim.net.res_bus.p_mw.values + np.random.normal(0, 0.02, num_buses)
        q_inj = -sim.net.res_bus.q_mvar.values + np.random.normal(0, 0.02, num_buses)
        v_scada = sim.net.res_bus.vm_pu.values + np.random.normal(0, 0.02, num_buses)
        z_scada = np.concatenate([p_inj, q_inj, v_scada])
        
        v_pmu = sim.net.res_bus.vm_pu.values[sim.pmu_buses] + np.random.normal(0, 0.005, len(sim.pmu_buses))
        theta_pmu = np.radians(sim.net.res_bus.va_degree.values[sim.pmu_buses]) + np.random.normal(0, 0.002, len(sim.pmu_buses))
        z_pmu = np.concatenate([v_pmu, theta_pmu])
        
        z_clean = np.concatenate([z_scada, z_pmu])
        
        # Apply bad data corruption
        z_corrupted, valid_mask = apply_bad_data(z_clean, missing_rate, error_rate)
        num_bad_measurements += np.sum(~valid_mask)
        
        iaukf.predict()
        iaukf.update(z_corrupted)
        
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
        'r_history': r_history, 'x_history': x_history,
        'num_bad_measurements': num_bad_measurements
    }

# ============================================================================
# Run Experiments
# ============================================================================

print("\n[1] Running IAUKF bad data experiments...")

all_results = {}

for scenario_name, params in BAD_DATA_SCENARIOS.items():
    print(f"\n  Scenario: {scenario_name}")
    scenario_results = []
    
    for run in tqdm(range(NUM_RUNS), desc=f"    Runs"):
        result = run_iaukf_with_bad_data(
            TARGET_BRANCH,
            missing_rate=params['missing_rate'],
            error_rate=params['error_rate'],
            seed=42 + run * 100
        )
        scenario_results.append(result)
    
    # Aggregate
    r_errors = [r['r_error'] for r in scenario_results]
    x_errors = [r['x_error'] for r in scenario_results]
    
    all_results[scenario_name] = {
        'params': params,
        'r_mean': np.mean(r_errors),
        'r_std': np.std(r_errors),
        'x_mean': np.mean(x_errors),
        'x_std': np.std(x_errors),
        'runs': scenario_results
    }
    
    print(f"      R error: {np.mean(r_errors):.2f}% ± {np.std(r_errors):.2f}%")
    print(f"      X error: {np.mean(x_errors):.2f}% ± {np.std(x_errors):.2f}%")

# ============================================================================
# Results Summary
# ============================================================================

print("\n" + "=" * 80)
print("BAD DATA ROBUSTNESS RESULTS")
print("=" * 80)

print("\n{:<20} {:>12} {:>12} {:>15} {:>15}".format(
    "Scenario", "Missing%", "Error%", "R Error%", "X Error%"))
print("-" * 74)

for name, result in all_results.items():
    missing = result['params']['missing_rate'] * 100
    error = result['params']['error_rate'] * 100
    print("{:<20} {:>12.0f} {:>12.0f} {:>14.2f}% {:>14.2f}%".format(
        name, missing, error, result['r_mean'], result['x_mean']))

# Compute degradation from clean baseline
clean_r = all_results['clean']['r_mean']
clean_x = all_results['clean']['x_mean']

print("\n" + "-" * 74)
print("DEGRADATION FROM CLEAN BASELINE:")
print("-" * 50)

for name, result in all_results.items():
    if name != 'clean':
        r_degrad = (result['r_mean'] / clean_r - 1) * 100 if clean_r > 0 else 0
        x_degrad = (result['x_mean'] / clean_x - 1) * 100 if clean_x > 0 else 0
        print(f"  {name:20s}: R +{r_degrad:5.1f}%, X +{x_degrad:5.1f}%")

# ============================================================================
# Generate Visualization
# ============================================================================

print("\n[2] Generating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Error vs Bad Data Level
ax = axes[0, 0]
scenarios = list(all_results.keys())
r_means = [all_results[s]['r_mean'] for s in scenarios]
r_stds = [all_results[s]['r_std'] for s in scenarios]
x_means = [all_results[s]['x_mean'] for s in scenarios]
x_stds = [all_results[s]['x_std'] for s in scenarios]

x_pos = np.arange(len(scenarios))
width = 0.35

bars1 = ax.bar(x_pos - width/2, r_means, width, label='R Error', color='steelblue', yerr=r_stds, capsize=3)
bars2 = ax.bar(x_pos + width/2, x_means, width, label='X Error', color='coral', yerr=x_stds, capsize=3)

ax.set_xticks(x_pos)
ax.set_xticklabels([s.replace('_', '\n') for s in scenarios], fontsize=9)
ax.set_ylabel('Estimation Error (%)')
ax.set_title('IAUKF Performance vs Bad Data Level')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: R Error Degradation
ax = axes[0, 1]
degrad_r = [(all_results[s]['r_mean'] / clean_r - 1) * 100 if s != 'clean' else 0 for s in scenarios]
colors = ['green' if d == 0 else 'orange' if d < 50 else 'red' for d in degrad_r]

ax.bar(scenarios, degrad_r, color=colors)
ax.set_ylabel('R Error Degradation (%)')
ax.set_title('Performance Degradation from Clean Baseline')
ax.set_xticklabels([s.replace('_', '\n') for s in scenarios], fontsize=9)
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Convergence comparison
ax = axes[1, 0]
for scenario in ['clean', 'missing_5pct', 'error_5pct', 'combined_10pct']:
    if scenario in all_results and len(all_results[scenario]['runs']) > 0:
        history = all_results[scenario]['runs'][0]['r_history']
        ax.plot(history, label=scenario.replace('_', ' '))

r_true = all_results['clean']['runs'][0]['r_true']
ax.axhline(y=r_true, color='black', linestyle='--', alpha=0.7, label='True R')
ax.set_xlabel('IAUKF Step')
ax.set_ylabel('R Estimate (Ohm/km)')
ax.set_title('IAUKF Convergence under Different Bad Data Scenarios')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Summary Table
ax = axes[1, 1]
ax.axis('off')

summary_text = f"""
BAD DATA ROBUSTNESS SUMMARY
===========================

Clean Baseline:
  R Error: {clean_r:.2f}%
  X Error: {clean_x:.2f}%

5% Missing Data:
  R Error: {all_results['missing_5pct']['r_mean']:.2f}% (+{(all_results['missing_5pct']['r_mean']/clean_r-1)*100:.1f}%)
  X Error: {all_results['missing_5pct']['x_mean']:.2f}% (+{(all_results['missing_5pct']['x_mean']/clean_x-1)*100:.1f}%)

5% Gross Errors:
  R Error: {all_results['error_5pct']['r_mean']:.2f}% (+{(all_results['error_5pct']['r_mean']/clean_r-1)*100:.1f}%)
  X Error: {all_results['error_5pct']['x_mean']:.2f}% (+{(all_results['error_5pct']['x_mean']/clean_x-1)*100:.1f}%)

10% Combined (5% missing + 5% error):
  R Error: {all_results['combined_10pct']['r_mean']:.2f}% (+{(all_results['combined_10pct']['r_mean']/clean_r-1)*100:.1f}%)
  X Error: {all_results['combined_10pct']['x_mean']:.2f}% (+{(all_results['combined_10pct']['x_mean']/clean_x-1)*100:.1f}%)

KEY FINDING:
IAUKF shows {'moderate' if all_results['combined_10pct']['r_mean'] < clean_r * 2 else 'significant'} 
degradation under 10% bad data.
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('tmp/b4_bad_data_results.png', dpi=150, bbox_inches='tight')
print("  Saved: tmp/b4_bad_data_results.png")

# ============================================================================
# Save Results
# ============================================================================

os.makedirs('tmp', exist_ok=True)
with open('tmp/b4_bad_data_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)
print("  Saved: tmp/b4_bad_data_results.pkl")

print("\n" + "=" * 80)
print("B4 EXPERIMENT COMPLETE")
print("=" * 80)
