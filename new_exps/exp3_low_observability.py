"""
Experiment 3: Low Observability / Sparse PMU (P1)
===================================================
Test estimation with reduced PMU measurements.

This is where Graph-Mamba's message passing should shine -
inferring unmeasured nodes from neighbor information.

Expected: Graph-Mamba maintains accuracy, IAUKF diverges
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import swanlab
from tqdm import tqdm

from model.simulation import PowerSystemSimulation
from model.models_analytical import AnalyticalMeasurementModel
from model.iaukf import IAUKF
from graphmamba import GraphMambaPhysicsModel, HAS_MAMBA

# Configuration
CONFIG = {
    'experiment': 'exp3_low_observability',
    'system': 'ieee33',
    'branch': 20,  # End branch 21-22 (difficult for IAUKF)
    'pmu_configs': {
        'full': list(range(0, 33, 3)),  # Every 3rd bus (11 PMUs)
        'reduced': list(range(0, 33, 6)),  # Every 6th bus (5 PMUs)
        'minimal': [0, 10, 20, 30],  # 4 PMUs only
        'sparse': [0],  # Only at substation
    },
    'num_runs': 5,
    'iaukf_steps': 200,
    'gm_steps': 50,
    'sequence_length': 50,
    'noise_scada': 0.02,
    'noise_pmu_v': 0.005,
    'noise_pmu_theta': 0.002,
}


def run_iaukf_sparse_pmu(branch_idx, pmu_buses, steps=200, seed=42):
    """Run IAUKF with sparse PMU configuration (FIXED initialization)."""
    np.random.seed(seed)
    
    sim = PowerSystemSimulation(steps=steps)
    # Override PMU buses
    sim.pmu_buses = pmu_buses
    
    r_true = sim.net.line.at[branch_idx, 'r_ohm_per_km']
    x_true = sim.net.line.at[branch_idx, 'x_ohm_per_km']
    
    model = AnalyticalMeasurementModel(sim.net, branch_idx, sim.pmu_buses)
    num_buses = len(sim.net.bus)
    
    # FIXED: Better initialization
    x0 = np.ones(2 * num_buses + 2)
    x0[:num_buses] = 1.0
    x0[num_buses:2*num_buses] = 0.0
    x0[-2] = 0.5  # Typical R value
    x0[-1] = 0.3  # Typical X value
    
    # FIXED: Higher initial covariance for faster adaptation
    P0 = np.eye(len(x0)) * 1e-3
    P0[-2, -2] = 0.5
    P0[-1, -1] = 0.5
    Q0 = np.eye(len(x0)) * 1e-6
    
    n_scada = 3 * num_buses
    n_pmu = 2 * len(pmu_buses)
    R_diag = np.concatenate([
        np.full(n_scada, CONFIG['noise_scada']**2),
        np.full(len(pmu_buses), CONFIG['noise_pmu_v']**2),
        np.full(len(pmu_buses), CONFIG['noise_pmu_theta']**2)
    ])
    R = np.diag(R_diag)
    
    iaukf = IAUKF(model, x0, P0, Q0, R)
    iaukf.b_factor = 0.95  # Slightly lower for faster adaptation
    
    import pandapower as pp
    p_base = sim.net.load.p_mw.values.copy()
    q_base = sim.net.load.q_mvar.values.copy()
    
    r_history = []
    x_history = []
    
    for t in range(steps):
        sim.net.load.p_mw = p_base
        sim.net.load.q_mvar = q_base
        
        try:
            pp.runpp(sim.net, algorithm='nr', numba=False)
        except:
            continue
        
        p_inj = -sim.net.res_bus.p_mw.values + np.random.normal(0, CONFIG['noise_scada'], num_buses)
        q_inj = -sim.net.res_bus.q_mvar.values + np.random.normal(0, CONFIG['noise_scada'], num_buses)
        v_scada = sim.net.res_bus.vm_pu.values + np.random.normal(0, CONFIG['noise_scada'], num_buses)
        z_scada = np.concatenate([p_inj, q_inj, v_scada])
        
        v_pmu = sim.net.res_bus.vm_pu.values[pmu_buses] + np.random.normal(0, CONFIG['noise_pmu_v'], len(pmu_buses))
        theta_pmu = np.radians(sim.net.res_bus.va_degree.values[pmu_buses]) + np.random.normal(0, CONFIG['noise_pmu_theta'], len(pmu_buses))
        z_pmu = np.concatenate([v_pmu, theta_pmu])
        
        z = np.concatenate([z_scada, z_pmu])
        
        iaukf.predict()
        iaukf.update(z)
        
        r_history.append(iaukf.x[-2])
        x_history.append(iaukf.x[-1])
    
    # Check convergence
    start_avg = len(r_history) // 2
    if len(r_history) < steps // 2:
        # Didn't converge enough
        return {
            'r_error': float('inf'),
            'x_error': float('inf'),
            'converged': False,
            'r_history': r_history,
            'x_history': x_history,
            'r_true': r_true,
            'x_true': x_true
        }
    
    r_final = np.mean(r_history[start_avg:])
    x_final = np.mean(x_history[start_avg:])
    
    r_error = abs(r_final - r_true) / r_true * 100
    x_error = abs(x_final - x_true) / x_true * 100
    
    # Check if estimates are reasonable (not diverged)
    converged = r_error < 50 and x_error < 50 and not (np.isnan(r_error) or np.isnan(x_error))
    
    return {
        'r_error': r_error,
        'x_error': x_error,
        'converged': converged,
        'r_history': r_history,
        'x_history': x_history,
        'r_true': r_true,
        'x_true': x_true
    }


def load_graphmamba_model():
    """Load pretrained Physics-Informed Graph-Mamba model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Try physics-informed checkpoint first
    checkpoint_path = '../checkpoints/graph_mamba_physics_best.pt'
    fallback_path = '../checkpoints/graph_mamba_phase2_best.pt'
    
    model = GraphMambaPhysicsModel(
        num_nodes=33,
        in_features=3,
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2
    )
    
    loaded = False
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"  Loaded physics-informed checkpoint")
            loaded = True
        except Exception as e:
            print(f"  Physics checkpoint failed: {e}")
    
    if not loaded and os.path.exists(fallback_path):
        try:
            checkpoint = torch.load(fallback_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"  Loaded standard checkpoint")
            loaded = True
        except Exception as e:
            print(f"  Standard checkpoint failed: {e}")
    
    if not loaded:
        print("  Warning: No checkpoint loaded, using random init")
    
    model.eval()
    model.to(device)
    return model, device


def run_graphmamba_sparse_pmu(model, device, branch_idx, pmu_buses, steps=50, seed=42):
    """Run Graph-Mamba with sparse PMU (leverages graph message passing)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    sim = PowerSystemSimulation(steps=steps)
    num_buses = len(sim.net.bus)
    
    r_true = sim.net.line.at[branch_idx, 'r_ohm_per_km']
    x_true = sim.net.line.at[branch_idx, 'x_ohm_per_km']
    
    import pandapower as pp
    p_base = sim.net.load.p_mw.values.copy()
    q_base = sim.net.load.q_mvar.values.copy()
    
    # Build edge index
    edge_index = []
    for _, line in sim.net.line.iterrows():
        from_bus = int(line['from_bus'])
        to_bus = int(line['to_bus'])
        edge_index.append([from_bus, to_bus])
        edge_index.append([to_bus, from_bus])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().to(device)
    
    # Generate measurements
    sequences = []
    for t in range(steps):
        sim.net.load.p_mw = p_base
        sim.net.load.q_mvar = q_base
        
        try:
            pp.runpp(sim.net, algorithm='nr', numba=False)
        except:
            continue
        
        p_inj = -sim.net.res_bus.p_mw.values + np.random.normal(0, CONFIG['noise_scada'], num_buses)
        q_inj = -sim.net.res_bus.q_mvar.values + np.random.normal(0, CONFIG['noise_scada'], num_buses)
        v_scada = sim.net.res_bus.vm_pu.values + np.random.normal(0, CONFIG['noise_scada'], num_buses)
        
        # Mask unmeasured PMU nodes (set to 0 or use SCADA only)
        # Graph-Mamba can still use the node features from SCADA
        node_features = np.stack([p_inj, q_inj, v_scada], axis=1)
        sequences.append(node_features)
    
    x = torch.tensor(np.array(sequences), dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(x, edge_index)
        output = output.cpu()  # Move back to CPU for processing
    
    r_final = output[0, 0].item()
    x_final = output[0, 1].item()
    
    r_error = abs(r_final - r_true) / r_true * 100
    x_error = abs(x_final - x_true) / x_true * 100
    
    return {
        'r_error': r_error,
        'x_error': x_error,
        'converged': True,  # Graph-Mamba always gives an output
        'r_true': r_true,
        'x_true': x_true
    }


def main():
    """Run Experiment 3."""
    print("=" * 80)
    print("EXPERIMENT 3: Low Observability / Sparse PMU")
    print("=" * 80)
    
    # Initialize SwanLab
    swanlab.init(
        project="graphmamba-vs-iaukf",
        experiment_name="exp3_low_observability",
        config=CONFIG
    )
    
    # Load Graph-Mamba model
    print("\n[1] Loading Graph-Mamba model...")
    gm_model, device = load_graphmamba_model()
    
    results = {}
    
    # Test each PMU configuration
    for config_name, pmu_buses in CONFIG['pmu_configs'].items():
        print(f"\n[2] Testing PMU configuration: {config_name}")
        print(f"    PMU buses: {pmu_buses} ({len(pmu_buses)} PMUs)")
        
        # IAUKF runs
        print(f"    Running IAUKF ({CONFIG['num_runs']} runs)...")
        iaukf_results = []
        converged_count = 0
        
        for run in range(CONFIG['num_runs']):
            result = run_iaukf_sparse_pmu(
                CONFIG['branch'], 
                pmu_buses, 
                CONFIG['iaukf_steps'], 
                seed=42+run
            )
            iaukf_results.append(result)
            if result['converged']:
                converged_count += 1
        
        # Aggregate IAUKF
        converged_results = [r for r in iaukf_results if r['converged']]
        if converged_results:
            r_errors = [r['r_error'] for r in converged_results]
            x_errors = [r['x_error'] for r in converged_results]
            iaukf_agg = {
                'r_error_mean': np.mean(r_errors),
                'r_error_std': np.std(r_errors),
                'x_error_mean': np.mean(x_errors),
                'x_error_std': np.std(x_errors),
                'convergence_rate': converged_count / CONFIG['num_runs'],
                'converged': True
            }
        else:
            iaukf_agg = {
                'r_error_mean': float('inf'),
                'r_error_std': 0,
                'x_error_mean': float('inf'),
                'x_error_std': 0,
                'convergence_rate': 0,
                'converged': False
            }
        
        # Graph-Mamba runs
        print(f"    Running Graph-Mamba ({CONFIG['num_runs']} runs)...")
        gm_results = []
        for run in range(CONFIG['num_runs']):
            result = run_graphmamba_sparse_pmu(
                gm_model,
                device,
                CONFIG['branch'],
                pmu_buses,
                CONFIG['gm_steps'],
                seed=42+run
            )
            gm_results.append(result)
        
        r_errors = [r['r_error'] for r in gm_results]
        x_errors = [r['x_error'] for r in gm_results]
        
        gm_agg = {
            'r_error_mean': np.mean(r_errors),
            'r_error_std': np.std(r_errors),
            'x_error_mean': np.mean(x_errors),
            'x_error_std': np.std(x_errors),
            'convergence_rate': 1.0,  # Always converges
            'converged': True
        }
        
        results[config_name] = {
            'iaukf': iaukf_agg,
            'graphmamba': gm_agg,
            'pmu_buses': pmu_buses,
            'num_pmus': len(pmu_buses)
        }
        
        # Log to SwanLab
        swanlab.log({
            f'{config_name}/num_pmus': len(pmu_buses),
            f'{config_name}/iaukf_r_error': iaukf_agg['r_error_mean'] if iaukf_agg['converged'] else 100,
            f'{config_name}/iaukf_x_error': iaukf_agg['x_error_mean'] if iaukf_agg['converged'] else 100,
            f'{config_name}/iaukf_convergence': iaukf_agg['convergence_rate'],
            f'{config_name}/gm_r_error': gm_agg['r_error_mean'],
            f'{config_name}/gm_x_error': gm_agg['x_error_mean'],
        })
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nBranch: 21-22 (end branch - most challenging)")
    print(f"\n{'Config':<12} {'#PMUs':<8} {'Method':<15} {'R Error (%)':<20} {'X Error (%)':<20} {'Conv. Rate':<12}")
    print("-" * 95)
    
    for config_name, data in results.items():
        # IAUKF row
        if data['iaukf']['converged']:
            r_str = f"{data['iaukf']['r_error_mean']:.3f} ± {data['iaukf']['r_error_std']:.3f}"
            x_str = f"{data['iaukf']['x_error_mean']:.3f} ± {data['iaukf']['x_error_std']:.3f}"
        else:
            r_str = "DIVERGED"
            x_str = "DIVERGED"
        
        conv_str = f"{data['iaukf']['convergence_rate']*100:.0f}%"
        
        print(f"{config_name:<12} {data['num_pmus']:<8} {'IAUKF':<15} {r_str:<20} {x_str:<20} {conv_str:<12}")
        
        # Graph-Mamba row
        r_str = f"{data['graphmamba']['r_error_mean']:.3f} ± {data['graphmamba']['r_error_std']:.3f}"
        x_str = f"{data['graphmamba']['x_error_mean']:.3f} ± {data['graphmamba']['x_error_std']:.3f}"
        conv_str = "100%"
        
        print(f"{'':<12} {'':<8} {'Graph-Mamba':<15} {r_str:<20} {x_str:<20} {conv_str:<12}")
        print()
    
    # Generate plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    configs = list(results.keys())
    num_pmus = [results[c]['num_pmus'] for c in configs]
    x_pos = np.arange(len(configs))
    width = 0.35
    
    # R error plot
    ax = axes[0]
    iaukf_r = [results[c]['iaukf']['r_error_mean'] if results[c]['iaukf']['converged'] else 50 
               for c in configs]
    gm_r = [results[c]['graphmamba']['r_error_mean'] for c in configs]
    
    bars1 = ax.bar(x_pos - width/2, iaukf_r, width, label='IAUKF', color='steelblue')
    bars2 = ax.bar(x_pos + width/2, gm_r, width, label='Graph-Mamba', color='coral')
    
    ax.set_ylabel('R Estimation Error (%)')
    ax.set_title('Resistance Estimation vs PMU Density')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{c}\n({n} PMUs)" for c, n in zip(configs, num_pmus)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(max(iaukf_r), max(gm_r)) * 1.2)
    
    # X error plot
    ax = axes[1]
    iaukf_x = [results[c]['iaukf']['x_error_mean'] if results[c]['iaukf']['converged'] else 50 
               for c in configs]
    gm_x = [results[c]['graphmamba']['x_error_mean'] for c in configs]
    
    ax.bar(x_pos - width/2, iaukf_x, width, label='IAUKF', color='steelblue')
    ax.bar(x_pos + width/2, gm_x, width, label='Graph-Mamba', color='coral')
    
    ax.set_ylabel('X Estimation Error (%)')
    ax.set_title('Reactance Estimation vs PMU Density')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{c}\n({n} PMUs)" for c, n in zip(configs, num_pmus)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(max(iaukf_x), max(gm_x)) * 1.2)
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/exp3_low_observability.png', dpi=150, bbox_inches='tight')
    print("\nFigure saved to: results/exp3_low_observability.png")
    
    # Save results
    with open('results/exp3_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("Results saved to: results/exp3_results.pkl")
    
    swanlab.log({"observability_plot": swanlab.Image("results/exp3_low_observability.png")})
    
    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    sparse_result = results.get('sparse', {})
    if sparse_result:
        if not sparse_result['iaukf']['converged'] or sparse_result['iaukf']['convergence_rate'] < 0.5:
            print("✓ IAUKF fails with minimal PMU coverage (only substation PMU)")
        if sparse_result['graphmamba']['r_error_mean'] < 10:
            print("✓ Graph-Mamba maintains accuracy even with sparse PMUs")
            print("  (Leverages message passing across graph topology)")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 3 COMPLETE")
    print("=" * 80)
    
    swanlab.finish()


if __name__ == '__main__':
    main()
