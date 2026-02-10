"""
Experiment 1: Basic Performance Comparison (P0)
================================================
Compare Graph-Mamba vs IAUKF on standard parameter estimation.

Metrics:
- RMSE/MAPE for R and X estimation
- Convergence speed
- Stability across multiple runs

Expected: Graph-Mamba < 0.1% error vs IAUKF ~0.18%
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import time
import swanlab

# Import models
from model.simulation import PowerSystemSimulation
from model.models_analytical import AnalyticalMeasurementModel
from model.iaukf import IAUKF
from graphmamba import GraphMambaPhysicsModel, HAS_MAMBA

# Configuration
CONFIG = {
    'experiment': 'exp1_basic_performance',
    'systems': ['ieee33'],  # Can add 'ieee118' later
    'branches': [3, 7, 20],  # 3-4 (main), 7-8 (lateral), 21-22 (end)
    'num_runs': 5,
    'iaukf_steps': 200,
    'gm_steps': 50,  # Graph-Mamba uses sequences
    'sequence_length': 50,
    'noise_scada': 0.02,
    'noise_pmu_v': 0.005,
    'noise_pmu_theta': 0.002,
}

def run_iaukf_benchmark(branch_idx, steps=200, seed=42):
    """Run IAUKF on a single branch."""
    np.random.seed(seed)
    
    sim = PowerSystemSimulation(steps=steps)
    r_true = sim.net.line.at[branch_idx, 'r_ohm_per_km']
    x_true = sim.net.line.at[branch_idx, 'x_ohm_per_km']
    
    # Create model
    model = AnalyticalMeasurementModel(sim.net, branch_idx, sim.pmu_buses)
    num_buses = len(sim.net.bus)
    
    # Initialize IAUKF with better parameter initial guess
    # Use typical line parameter values as initial guess
    x0 = np.ones(2 * num_buses + 2)
    x0[:num_buses] = 1.0
    x0[num_buses:2*num_buses] = 0.0
    # Better initial guess: typical distribution line values
    x0[-2] = 0.5  # R initial guess (typical value)
    x0[-1] = 0.3  # X initial guess (typical value)
    
    # Higher initial covariance for parameters to allow faster adaptation
    P0 = np.eye(len(x0)) * 1e-3
    P0[-2, -2] = 0.5  # Higher uncertainty for R
    P0[-1, -1] = 0.5  # Higher uncertainty for X
    
    Q0 = np.eye(len(x0)) * 1e-6
    
    n_scada = 3 * num_buses
    R_diag = np.concatenate([
        np.full(n_scada, CONFIG['noise_scada']**2),
        np.full(len(sim.pmu_buses), CONFIG['noise_pmu_v']**2),
        np.full(len(sim.pmu_buses), CONFIG['noise_pmu_theta']**2)
    ])
    R = np.diag(R_diag)
    
    iaukf = IAUKF(model, x0, P0, Q0, R)
    iaukf.b_factor = 0.95  # Slightly lower for faster adaptation
    
    # Generate constant load measurements
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
        
        # Generate noisy measurements
        p_inj = -sim.net.res_bus.p_mw.values + np.random.normal(0, CONFIG['noise_scada'], num_buses)
        q_inj = -sim.net.res_bus.q_mvar.values + np.random.normal(0, CONFIG['noise_scada'], num_buses)
        v_scada = sim.net.res_bus.vm_pu.values + np.random.normal(0, CONFIG['noise_scada'], num_buses)
        z_scada = np.concatenate([p_inj, q_inj, v_scada])
        
        v_pmu = sim.net.res_bus.vm_pu.values[sim.pmu_buses] + np.random.normal(0, CONFIG['noise_pmu_v'], len(sim.pmu_buses))
        theta_pmu = np.radians(sim.net.res_bus.va_degree.values[sim.pmu_buses]) + np.random.normal(0, CONFIG['noise_pmu_theta'], len(sim.pmu_buses))
        z_pmu = np.concatenate([v_pmu, theta_pmu])
        
        z = np.concatenate([z_scada, z_pmu])
        
        iaukf.predict()
        iaukf.update(z)
        
        r_history.append(iaukf.x[-2])
        x_history.append(iaukf.x[-1])
    
    # Post-convergence averaging (last 50% of steps)
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


def load_or_train_graphmamba(system='ieee33'):
    """Load pretrained Graph-Mamba model or train if not exists."""
    # Try physics-informed checkpoint first
    checkpoint_path = f'../checkpoints/graph_mamba_physics_best.pt'
    fallback_path = f'../checkpoints/graph_mamba_phase2_best.pt'
    
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    if not torch.cuda.is_available():
        print("  WARNING: CUDA not available. Graph-Mamba trained with Mamba requires CUDA.")
        print("  Graph-Mamba tests will be skipped or use random initialization.")
        return None, device
    
    model = GraphMambaPhysicsModel(
        num_nodes=33 if system == 'ieee33' else 118,
        in_features=3,  # P, Q, V
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2
    )
    
    # Try physics-informed checkpoint first
    loaded = False
    if os.path.exists(checkpoint_path):
        print(f"  Loading physics-informed checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"  Physics-informed checkpoint loaded successfully!")
            loaded = True
        except RuntimeError as e:
            print(f"  Warning: Physics checkpoint mismatch - {e}")
    
    # Fall back to standard checkpoint
    if not loaded and os.path.exists(fallback_path):
        print(f"  Loading standard checkpoint: {fallback_path}")
        try:
            checkpoint = torch.load(fallback_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"  Standard checkpoint loaded")
            loaded = True
        except RuntimeError as e:
            print(f"  Warning: Standard checkpoint mismatch - {e}")
    
    if not loaded:
        print(f"  Warning: No checkpoint found")
        print(f"  Using randomly initialized model (for testing only)")
    
    model.eval()
    model.to(device)
    return model, device


def run_graphmamba_benchmark(model, device, branch_idx, steps=50, seed=42):
    """Run Graph-Mamba on a single branch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    sim = PowerSystemSimulation(steps=steps)
    r_true = sim.net.line.at[branch_idx, 'r_ohm_per_km']
    x_true = sim.net.line.at[branch_idx, 'x_ohm_per_km']
    
    # Generate sequence of measurements
    import pandapower as pp
    p_base = sim.net.load.p_mw.values.copy()
    q_base = sim.net.load.q_mvar.values.copy()
    num_buses = len(sim.net.bus)
    
    # Build edge index from network topology
    edge_index = []
    for _, line in sim.net.line.iterrows():
        from_bus = int(line['from_bus'])
        to_bus = int(line['to_bus'])
        edge_index.append([from_bus, to_bus])
        edge_index.append([to_bus, from_bus])  # Undirected
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().to(device)
    
    sequences = []
    
    for t in range(steps):
        sim.net.load.p_mw = p_base
        sim.net.load.q_mvar = q_base
        
        try:
            pp.runpp(sim.net, algorithm='nr', numba=False)
        except:
            continue
        
        # Generate noisy measurements
        p_inj = -sim.net.res_bus.p_mw.values + np.random.normal(0, CONFIG['noise_scada'], num_buses)
        q_inj = -sim.net.res_bus.q_mvar.values + np.random.normal(0, CONFIG['noise_scada'], num_buses)
        v_scada = sim.net.res_bus.vm_pu.values + np.random.normal(0, CONFIG['noise_scada'], num_buses)
        
        # Node features: [P, Q, V] for each bus
        node_features = np.stack([p_inj, q_inj, v_scada], axis=1)
        sequences.append(node_features)
    
    # Convert to tensor [batch=1, time, nodes, features]
    x = torch.tensor(np.array(sequences), dtype=torch.float32).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(x, edge_index)
        output = output.cpu()  # Move back to CPU for processing
    
    r_final = output[0, 0].item()
    x_final = output[0, 1].item()
    
    r_error = abs(r_final - r_true) / r_true * 100
    x_error = abs(x_final - x_true) / x_true * 100
    
    return {
        'r_true': r_true, 'x_true': x_true,
        'r_pred': r_final, 'x_pred': x_final,
        'r_error': r_error, 'x_error': x_error
    }


def main():
    """Run Experiment 1."""
    print("=" * 80)
    print("EXPERIMENT 1: Basic Performance Comparison")
    print("=" * 80)
    
    # Initialize SwanLab
    swanlab.init(
        project="graphmamba-vs-iaukf",
        experiment_name="exp1_basic_performance",
        config=CONFIG
    )
    
    results = {
        'iaukf': {},
        'graphmamba': {},
        'config': CONFIG
    }
    
    # Load Graph-Mamba model once
    print("\n[1] Loading Graph-Mamba model...")
    gm_model, device = load_or_train_graphmamba('ieee33')
    
    # Run benchmarks for each branch
    for branch_idx in CONFIG['branches']:
        branch_name = f"branch_{branch_idx}-{branch_idx+1}"
        print(f"\n[2] Testing {branch_name}...")
        
        # IAUKF runs
        print(f"  Running IAUKF ({CONFIG['num_runs']} runs)...")
        iaukf_results = []
        for run in range(CONFIG['num_runs']):
            result = run_iaukf_benchmark(branch_idx, CONFIG['iaukf_steps'], seed=42+run)
            iaukf_results.append(result)
        
        # Aggregate IAUKF results
        r_errors = [r['r_error'] for r in iaukf_results]
        x_errors = [r['x_error'] for r in iaukf_results]
        
        results['iaukf'][branch_name] = {
            'r_error_mean': np.mean(r_errors),
            'r_error_std': np.std(r_errors),
            'x_error_mean': np.mean(x_errors),
            'x_error_std': np.std(x_errors),
            'r_true': iaukf_results[0]['r_true'],
            'x_true': iaukf_results[0]['x_true'],
            'all_runs': iaukf_results
        }
        
        # Graph-Mamba runs (skip if model not loaded)
        if gm_model is not None:
            print(f"  Running Graph-Mamba ({CONFIG['num_runs']} runs)...")
            gm_results = []
            for run in range(CONFIG['num_runs']):
                result = run_graphmamba_benchmark(gm_model, device, branch_idx, CONFIG['gm_steps'], seed=42+run)
                gm_results.append(result)
            
            # Aggregate Graph-Mamba results
            r_errors = [r['r_error'] for r in gm_results]
            x_errors = [r['x_error'] for r in gm_results]
            
            results['graphmamba'][branch_name] = {
                'r_error_mean': np.mean(r_errors),
                'r_error_std': np.std(r_errors),
                'x_error_mean': np.mean(x_errors),
                'x_error_std': np.std(x_errors),
                'r_true': gm_results[0]['r_true'],
                'x_true': gm_results[0]['x_true'],
                'all_runs': gm_results
            }
            
            # Log to SwanLab
            swanlab.log({
                f'{branch_name}/iaukf_r_error': results['iaukf'][branch_name]['r_error_mean'],
                f'{branch_name}/iaukf_x_error': results['iaukf'][branch_name]['x_error_mean'],
                f'{branch_name}/gm_r_error': results['graphmamba'][branch_name]['r_error_mean'],
                f'{branch_name}/gm_x_error': results['graphmamba'][branch_name]['x_error_mean'],
            })
        else:
            print(f"  Skipping Graph-Mamba (CUDA not available)")
            results['graphmamba'][branch_name] = {
                'r_error_mean': float('nan'),
                'r_error_std': 0,
                'x_error_mean': float('nan'),
                'x_error_std': 0,
                'r_true': iaukf_results[0]['r_true'],
                'x_true': iaukf_results[0]['x_true'],
                'all_runs': []
            }
    
    # Print results table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Branch':<15} {'Method':<12} {'R Error (%)':<20} {'X Error (%)':<20}")
    print("-" * 70)
    
    for branch_name in results['iaukf'].keys():
        iaukf = results['iaukf'][branch_name]
        gm = results['graphmamba'][branch_name]
        
        print(f"{branch_name:<15} {'IAUKF':<12} "
              f"{iaukf['r_error_mean']:>6.3f} ± {iaukf['r_error_std']:<6.3f}   "
              f"{iaukf['x_error_mean']:>6.3f} ± {iaukf['x_error_std']:<6.3f}")
        
        if not np.isnan(gm['r_error_mean']):
            print(f"{'':<15} {'Graph-Mamba':<12} "
                  f"{gm['r_error_mean']:>6.3f} ± {gm['r_error_std']:<6.3f}   "
                  f"{gm['x_error_mean']:>6.3f} ± {gm['x_error_std']:<6.3f}")
            
            # Calculate improvement
            r_improvement = (iaukf['r_error_mean'] - gm['r_error_mean']) / iaukf['r_error_mean'] * 100
            x_improvement = (iaukf['x_error_mean'] - gm['x_error_mean']) / iaukf['x_error_mean'] * 100
            print(f"{'':<15} {'Improvement':<12} {r_improvement:>6.1f}%{'':<12} {x_improvement:>6.1f}%")
        else:
            print(f"{'':<15} {'Graph-Mamba':<12} N/A (CUDA not available)")
        print()
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/exp1_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("Results saved to: results/exp1_results.pkl")
    
    # Generate plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    branches = list(results['iaukf'].keys())
    x_pos = np.arange(len(branches))
    width = 0.35
    
    # R error plot
    iaukf_r = [results['iaukf'][b]['r_error_mean'] for b in branches]
    iaukf_r_std = [results['iaukf'][b]['r_error_std'] for b in branches]
    axes[0].bar(x_pos - width/2, iaukf_r, width, label='IAUKF', yerr=iaukf_r_std, capsize=3)
    
    gm_r = [results['graphmamba'][b]['r_error_mean'] for b in branches]
    if not all(np.isnan(gm_r)):
        gm_r_std = [results['graphmamba'][b]['r_error_std'] for b in branches]
        axes[0].bar(x_pos + width/2, gm_r, width, label='Graph-Mamba', yerr=gm_r_std, capsize=3)
    
    axes[0].set_ylabel('R Estimation Error (%)')
    axes[0].set_title('Resistance Estimation Accuracy')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(branches, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # X error plot
    iaukf_x = [results['iaukf'][b]['x_error_mean'] for b in branches]
    iaukf_x_std = [results['iaukf'][b]['x_error_std'] for b in branches]
    axes[1].bar(x_pos - width/2, iaukf_x, width, label='IAUKF', yerr=iaukf_x_std, capsize=3)
    
    gm_x = [results['graphmamba'][b]['x_error_mean'] for b in branches]
    if not all(np.isnan(gm_x)):
        gm_x_std = [results['graphmamba'][b]['x_error_std'] for b in branches]
        axes[1].bar(x_pos + width/2, gm_x, width, label='Graph-Mamba', yerr=gm_x_std, capsize=3)
    
    axes[1].set_ylabel('X Estimation Error (%)')
    axes[1].set_title('Reactance Estimation Accuracy')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(branches, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/exp1_basic_performance.png', dpi=150, bbox_inches='tight')
    print("Figure saved to: results/exp1_basic_performance.png")
    
    # Log figure to SwanLab
    swanlab.log({"results_plot": swanlab.Image("results/exp1_basic_performance.png")})
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 1 COMPLETE")
    print("=" * 80)
    
    swanlab.finish()


if __name__ == '__main__':
    main()
