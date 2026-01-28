"""
Simple test focusing on b_factor and Q0 parameters.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandapower as pp
from model.simulation import PowerSystemSimulation
from model.models import DistributionSystemModel
from model.iaukf import IAUKF


def test_params(b_factor, q0_scale, seed):
    """Test with specific parameters."""
    steps = 200

    # Generate data
    sim = PowerSystemSimulation(steps=steps)
    p_load_base = sim.net.load.p_mw.values.copy()
    q_load_base = sim.net.load.q_mvar.values.copy()

    z_scada_list = []
    z_pmu_list = []

    np.random.seed(seed)

    for t in range(steps):
        sim.net.load.p_mw = p_load_base
        sim.net.load.q_mvar = q_load_base
        pp.runpp(sim.net, algorithm='nr', numba=False)

        p_inj = -sim.net.res_bus.p_mw.values
        q_inj = -sim.net.res_bus.q_mvar.values
        v_scada = sim.net.res_bus.vm_pu.values
        z_scada = np.concatenate([p_inj, q_inj, v_scada])
        z_scada_list.append(z_scada + np.random.normal(0, 0.02, len(z_scada)))

        v_pmu = sim.net.res_bus.vm_pu.values[sim.pmu_buses]
        theta_pmu = np.radians(sim.net.res_bus.va_degree.values[sim.pmu_buses])
        z_pmu = np.concatenate([v_pmu, theta_pmu])
        noise_pmu = np.concatenate([
            np.random.normal(0, 0.005, len(v_pmu)),
            np.random.normal(0, 0.002, len(theta_pmu))
        ])
        z_pmu_list.append(z_pmu + noise_pmu)

    Z_scada = np.array(z_scada_list)
    Z_pmu = np.array(z_pmu_list)

    # Setup model (Identity transition for stability)
    model = DistributionSystemModel(sim.net, sim.line_idx, sim.pmu_buses)

    # Initial state
    x0_v = np.ones(sim.net.bus.shape[0])
    x0_d = np.zeros(sim.net.bus.shape[0])
    x0 = np.concatenate([x0_v, x0_d, [0.01, 0.01]])

    # Parameters
    P0 = np.eye(len(x0)) * 0.01
    P0[-2, -2] = 0.1
    P0[-1, -1] = 0.1

    Q0 = np.eye(len(x0)) * q0_scale
    Q0[-2, -2] = q0_scale
    Q0[-1, -1] = q0_scale

    R_diag = np.concatenate([
        np.full(33, 0.02**2),
        np.full(33, 0.02**2),
        np.full(33, 0.02**2),
        np.full(12, 0.005**2),
        np.full(12, 0.002**2)
    ])
    R_cov = np.diag(R_diag)

    # Create IAUKF
    iaukf = IAUKF(model, x0, P0, Q0, R_cov)
    iaukf.b_factor = b_factor

    # Run
    history_r = []
    history_x = []

    for t in range(steps):
        z_t = np.concatenate([Z_scada[t], Z_pmu[t]])
        iaukf.predict()
        x_est = iaukf.update(z_t)
        history_r.append(x_est[-2])
        history_x.append(x_est[-1])

    # Find convergence
    r_converged = None
    x_converged = None

    for t in range(1, len(history_r)):
        if r_converged is None and abs(history_r[t] - history_r[t-1]) <= 0.001:
            r_converged = t
        if x_converged is None and abs(history_x[t] - history_x[t-1]) <= 0.001:
            x_converged = t
        if r_converged and x_converged:
            break

    # Average after convergence
    r_final = np.mean(history_r[r_converged:]) if r_converged else history_r[-1]
    x_final = np.mean(history_x[x_converged:]) if x_converged else history_x[-1]

    r_err = abs(r_final - sim.r_true) / sim.r_true * 100
    x_err = abs(x_final - sim.x_true) / sim.x_true * 100

    return r_err, x_err, r_final, x_final


if __name__ == "__main__":
    print("Testing different parameter combinations:\n")
    print(f"{'b_factor':<10} {'Q0':<10} {'Seed':<8} {'R err %':<10} {'X err %':<10}")
    print("="*60)

    # Test different combinations
    tests = [
        (0.96, 1e-6, 42),    # Paper's exact values
        (0.96, 1e-6, 0),     # Different seed
        (0.96, 1e-6, 123),   # Different seed
        (0.96, 1e-6, 999),   # Different seed
        (0.96, 1e-8, 42),    # Smaller Q0
        (0.96, 1e-7, 42),    # In-between Q0
        (0.98, 1e-6, 42),    # Different b_factor
        (0.95, 1e-6, 42),    # Different b_factor
    ]

    results = []
    for b, q, seed in tests:
        try:
            r_err, x_err, r_final, x_final = test_params(b, q, seed)
            results.append((b, q, seed, r_err, x_err))
            print(f"{b:<10.3f} {q:<10.0e} {seed:<8} {r_err:<10.2f} {x_err:<10.2f}")
        except Exception as e:
            print(f"{b:<10.3f} {q:<10.0e} {seed:<8} ERROR: {str(e)[:20]}")

    print("\n" + "="*60)
    print(f"Paper target: R=0.18%, X=1.55%")

    # Find best R error
    best_r = min(results, key=lambda x: x[3])
    print(f"\nBest R error: {best_r[3]:.2f}% (b={best_r[0]:.3f}, Q0={best_r[1]:.0e}, seed={best_r[2]})")

    # Find best overall (sum of errors)
    best_overall = min(results, key=lambda x: x[3] + x[4])
    print(f"Best overall: R={best_overall[3]:.2f}%, X={best_overall[4]:.2f}% (b={best_overall[0]:.3f}, Q0={best_overall[1]:.0e}, seed={best_overall[2]})")
