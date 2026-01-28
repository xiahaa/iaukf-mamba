"""
Quick validation test for IAUKF enhancements.
This is a simplified version to quickly verify the implementation works.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import copy
import pandapower as pp
from model.simulation import PowerSystemSimulation
from model.models_holt import DistributionSystemModelHolt
from model.iaukf import IAUKF, IAUKFMultiSnapshot

print("="*80)
print("QUICK IAUKF ENHANCEMENT VALIDATION")
print("="*80)

# Setup
print("\n[1] Setting up simulation...")
steps = 50  # Reduced for quick test
sim = PowerSystemSimulation(steps=steps)
target_branch = sim.line_idx  # Use the actual index from simulation

r_true = sim.r_true
x_true = sim.x_true
print(f"  Target line: {target_branch}")
print(f"  True R = {r_true:.4f}, X = {x_true:.4f}")

# Generate simple measurements
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

# Test 1: Single-snapshot IAUKF with exact NSE
print("\n[3] Testing Single-Snapshot IAUKF with Exact NSE...")
num_buses = len(sim.net.bus)
state_dim = 2 * num_buses + 2

x0 = np.ones(state_dim)
x0[:num_buses] = 1.0
x0[num_buses:2*num_buses] = 0.0
x0[-2] = 0.01
x0[-1] = 0.01

# Use tuned covariances as in phase1_exact_paper.py
P0 = np.eye(state_dim) * 0.01  # Moderate for voltages
P0[-2, -2] = 0.2  # Moderate for R
P0[-1, -1] = 0.2  # Moderate for X

Q0 = np.eye(state_dim) * 1e-9  # Extremely small for voltages
Q0[-2, -2] = 1e-8  # Extremely small for parameters
Q0[-1, -1] = 1e-8

n_scada = 3 * num_buses
n_pmu = 2 * len(sim.pmu_buses)
R = np.eye(n_scada + n_pmu)
R[:n_scada, :n_scada] *= 0.02**2
R[n_scada:n_scada+len(sim.pmu_buses), n_scada:n_scada+len(sim.pmu_buses)] *= 0.005**2
R[n_scada+len(sim.pmu_buses):, n_scada+len(sim.pmu_buses):] *= 0.002**2

model = DistributionSystemModelHolt(
    copy.deepcopy(sim.net), target_branch, sim.pmu_buses,
    alpha_H=0.8, beta_H=0.5
)

iaukf = IAUKF(model, x0, P0, Q0, R)
iaukf.b_factor = 0.995  # Less aggressive as in phase1_exact_paper.py

r_est_single = []
x_est_single = []

for t, z in enumerate(measurements):
    iaukf.predict()
    x_est = iaukf.update(z)
    r_est_single.append(x_est[-2])
    x_est_single.append(x_est[-1])

r_final_single = np.mean(r_est_single[-10:])
x_final_single = np.mean(x_est_single[-10:])
r_err_single = abs(r_final_single - r_true) / r_true * 100
x_err_single = abs(x_final_single - x_true) / x_true * 100

print(f"  Final R: {r_final_single:.4f} (Error: {r_err_single:.2f}%)")
print(f"  Final X: {x_final_single:.4f} (Error: {x_err_single:.2f}%)")
print(f"  Status: {'✓ PASS' if r_err_single < 5.0 and x_err_single < 5.0 else '⚠ WARN'}")

# Test 2: Multi-snapshot IAUKF
print("\n[4] Testing Multi-Snapshot IAUKF (3 snapshots)...")
model_ms = DistributionSystemModelHolt(
    copy.deepcopy(sim.net), target_branch, sim.pmu_buses,
    alpha_H=0.8, beta_H=0.5
)

iaukf_ms = IAUKFMultiSnapshot(model_ms, x0, P0, Q0, R, num_snapshots=3)
iaukf_ms.b_factor = 0.995

r_est_multi = []
x_est_multi = []

for t, z in enumerate(measurements):
    iaukf_ms.predict()
    x_est = iaukf_ms.update(z)
    params = iaukf_ms.get_parameters()
    r_est_multi.append(params[0])
    x_est_multi.append(params[1])

r_final_multi = np.mean(r_est_multi[-10:])
x_final_multi = np.mean(x_est_multi[-10:])
r_err_multi = abs(r_final_multi - r_true) / r_true * 100
x_err_multi = abs(x_final_multi - x_true) / x_true * 100

print(f"  Final R: {r_final_multi:.4f} (Error: {r_err_multi:.2f}%)")
print(f"  Final X: {x_final_multi:.4f} (Error: {x_err_multi:.2f}%)")
print(f"  Status: {'✓ PASS' if r_err_multi < 5.0 and x_err_multi < 5.0 else '⚠ WARN'}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Single-snapshot: R={r_err_single:.2f}%, X={x_err_single:.2f}%")
print(f"Multi-snapshot:  R={r_err_multi:.2f}%, X={x_err_multi:.2f}%")

if r_err_single < 5.0 and x_err_single < 5.0 and r_err_multi < 5.0 and x_err_multi < 5.0:
    print("\n✓ VALIDATION PASSED - Both implementations work correctly!")
    print("\nNote: Run with more steps (200+) for paper-level accuracy.")
else:
    print("\n⚠ WARNING - Errors higher than expected. May need more steps or tuning.")

print("="*80)
