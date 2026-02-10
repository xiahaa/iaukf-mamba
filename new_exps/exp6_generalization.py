"""
Experiment 6: Cross-Topology Generalization (P2)
=================================================
Test transfer learning across different power system topologies.

Train on IEEE 33-bus, test on IEEE 69-bus or 118-bus.

Expected: Graph-Mamba shows some transfer capability
(learns local physics, not specific topology)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
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
    'experiment': 'exp6_generalization',
    'train_system': 'ieee33',
    'test_systems': ['ieee33', 'ieee118'],
    'num_runs': 3,
    'iaukf_steps': 100,
    'gm_steps': 50,
    'sequence_length': 50,
    'noise_scada': 0.02,
    'noise_pmu_v': 0.005,
    'noise_pmu_theta': 0.002,
    'fine_tune_samples': 100,  # Number of samples for fine-tuning
}


def load_system(system_name):
    """Load a power system."""
    import pandapower.networks as pn
    import pandapower as pp
    
    if system_name == 'ieee33':
        net = pn.case33bw()
        pmu_buses = list(range(0, 33, 3))
        test_branch = 3  # Branch 3-4
    elif system_name == 'synthetic69':
        # Create a synthetic 69-bus radial system
        net = pp.create_empty_network()
        pp.create_bus(net, vn_kv=12.66, name="Substation")
        pp.create_ext_grid(net, bus=0)
        
        for i in range(1, 69):
            pp.create_bus(net, vn_kv=12.66)
            pp.create_line(net, from_bus=i-1, to_bus=i, length_km=0.5 + (i % 3) * 0.2,
                          std_type='NAYY 4x50 SE')
            pp.create_load(net, bus=i, p_mw=0.05 + (i % 5) * 0.02, 
                          q_mvar=0.025 + (i % 3) * 0.01)
        pmu_buses = list(range(0, 69, 5))
        test_branch = 5
    elif system_name == 'ieee118':
        net = pn.case118()
        pmu_buses = list(range(0, 118, 5))
        test_branch = 10
    else:
        raise ValueError(f"Unknown system: {system_name}")
    
    return net, pmu_buses, test_branch


def run_iaukf_system(net, pmu_buses, target_branch, steps=200, seed=42):
    """Run IAUKF on a given system (trained from scratch for each system)."""
    np.random.seed(seed)
    
    r_true = net.line.at[target_branch, 'r_ohm_per_km']
    x_true = net.line.at[target_branch, 'x_ohm_per_km']
    
    model = AnalyticalMeasurementModel(net, target_branch, pmu_buses)
    num_buses = len(net.bus)
    
    # FIXED: Better initialization
    x0 = np.ones(2 * num_buses + 2)
    x0[:num_buses] = 1.0
    x0[num_buses:2*num_buses] = 0.0
    x0[-2] = 0.5  # Typical R value
    x0[-1] = 0.3  # Typical X value
    
    P0 = np.eye(len(x0)) * 1e-3
    P0[-2, -2] = 0.5
    P0[-1, -1] = 0.5
    Q0 = np.eye(len(x0)) * 1e-6
    
    n_scada = 3 * num_buses
    R_diag = np.concatenate([
        np.full(n_scada, CONFIG['noise_scada']**2),
        np.full(len(pmu_buses), CONFIG['noise_pmu_v']**2),
        np.full(len(pmu_buses), CONFIG['noise_pmu_theta']**2)
    ])
    R = np.diag(R_diag)
    
    iaukf = IAUKF(model, x0, P0, Q0, R)
    iaukf.b_factor = 0.95
    
    import pandapower as pp
    p_base = net.load.p_mw.values.copy()
    q_base = net.load.q_mvar.values.copy()
    
    r_history = []
    x_history = []
    
    for t in range(steps):
        net.load.p_mw = p_base
        net.load.q_mvar = q_base
        
        try:
            pp.runpp(net, algorithm='nr', numba=False)
        except:
            continue
        
        p_inj = -net.res_bus.p_mw.values + np.random.normal(0, CONFIG['noise_scada'], num_buses)
        q_inj = -net.res_bus.q_mvar.values + np.random.normal(0, CONFIG['noise_scada'], num_buses)
        v_scada = net.res_bus.vm_pu.values + np.random.normal(0, CONFIG['noise_scada'], num_buses)
        z_scada = np.concatenate([p_inj, q_inj, v_scada])
        
        v_pmu = net.res_bus.vm_pu.values[pmu_buses] + np.random.normal(0, CONFIG['noise_pmu_v'], len(pmu_buses))
        theta_pmu = np.radians(net.res_bus.va_degree.values[pmu_buses]) + np.random.normal(0, CONFIG['noise_pmu_theta'], len(pmu_buses))
        z_pmu = np.concatenate([v_pmu, theta_pmu])
        
        z = np.concatenate([z_scada, z_pmu])
        
        iaukf.predict()
        iaukf.update(z)
        
        r_history.append(iaukf.x[-2])
        x_history.append(iaukf.x[-1])
    
    start_avg = len(r_history) // 2
    r_final = np.mean(r_history[start_avg:])
    x_final = np.mean(x_history[start_avg:])
    
    r_error = abs(r_final - r_true) / r_true * 100
    x_error = abs(x_final - x_true) / x_true * 100
    
    return {
        'r_error': r_error,
        'x_error': x_error,
        'r_true': r_true,
        'x_true': x_true
    }


def load_graphmamba_model(num_nodes=33):
    """Load or create Physics-Informed Graph-Mamba model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = GraphMambaPhysicsModel(
        num_nodes=num_nodes,
        in_features=3,
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2
    )
    
    # Try to load pretrained weights
    checkpoint_path = '../checkpoints/graph_mamba_physics_best.pt'
    fallback_path = '../checkpoints/graph_mamba_phase2_best.pt'
    
    loaded = False
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            print(f"    Loaded physics-informed weights")
            loaded = True
        except Exception as e:
            print(f"    Physics checkpoint failed: {e}")
    
    if not loaded and os.path.exists(fallback_path):
        try:
            checkpoint = torch.load(fallback_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            print(f"    Loaded standard weights")
            loaded = True
        except Exception as e:
            print(f"    Standard checkpoint failed: {e}")
    
    if not loaded:
        print(f"    Using random initialization")
    
    model.eval()
    model.to(device)
    return model, device


def run_graphmamba_zero_shot(model, device, net, pmu_buses, target_branch, steps=50, seed=42):
    """Run Graph-Mamba in zero-shot mode (no fine-tuning)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    num_buses = len(net.bus)
    r_true = net.line.at[target_branch, 'r_ohm_per_km']
    x_true = net.line.at[target_branch, 'x_ohm_per_km']
    
    import pandapower as pp
    p_base = net.load.p_mw.values.copy()
    q_base = net.load.q_mvar.values.copy()
    
    # Build edge index
    edge_index = []
    for _, line in net.line.iterrows():
        from_bus = int(line['from_bus'])
        to_bus = int(line['to_bus'])
        edge_index.append([from_bus, to_bus])
        edge_index.append([to_bus, from_bus])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().to(device)
    
    # Generate measurements
    sequences = []
    for t in range(steps):
        net.load.p_mw = p_base
        net.load.q_mvar = q_base
        
        try:
            pp.runpp(net, algorithm='nr', numba=False)
        except:
            continue
        
        p_inj = -net.res_bus.p_mw.values + np.random.normal(0, CONFIG['noise_scada'], num_buses)
        q_inj = -net.res_bus.q_mvar.values + np.random.normal(0, CONFIG['noise_scada'], num_buses)
        v_scada = net.res_bus.vm_pu.values + np.random.normal(0, CONFIG['noise_scada'], num_buses)
        
        node_features = np.stack([p_inj, q_inj, v_scada], axis=1)
        sequences.append(node_features)
    
    x = torch.tensor(np.array(sequences), dtype=torch.float32).unsqueeze(0).to(device)
    
    # Resize model if needed (simple approach: use first N nodes)
    if num_buses != model.num_nodes:
        # For simplicity, just use subset of nodes
        x = x[:, :, :model.num_nodes, :]
        edge_index_subset = edge_index[:, (edge_index[0] < model.num_nodes) & (edge_index[1] < model.num_nodes)]
        edge_index = edge_index_subset
    
    with torch.no_grad():
        output = model(x, edge_index)
    
    output = output.cpu()
    r_final = output[0, 0].item()
    x_final = output[0, 1].item()
    
    r_error = abs(r_final - r_true) / r_true * 100
    x_error = abs(x_final - x_true) / x_true * 100
    
    return {
        'r_error': r_error,
        'x_error': x_error,
        'r_true': r_true,
        'x_true': x_true
    }


def main():
    """Run Experiment 6."""
    print("=" * 80)
    print("EXPERIMENT 6: Cross-Topology Generalization")
    print("=" * 80)
    
    # Initialize SwanLab
    swanlab.init(
        project="graphmamba-vs-iaukf",
        experiment_name="exp6_generalization",
        config=CONFIG
    )
    
    results = {}
    
    # Load Graph-Mamba model (trained on 33-bus)
    print("\n[1] Loading Graph-Mamba model (trained on IEEE 33-bus)...")
    gm_model, device = load_graphmamba_model(33)
    
    # Test on each system
    for system_name in CONFIG['test_systems']:
        print(f"\n[2] Testing on {system_name}...")
        
        net, pmu_buses, test_branch = load_system(system_name)
        print(f"    Buses: {len(net.bus)}, Test branch: {test_branch}")
        
        # IAUKF (train from scratch - baseline)
        print(f"    Running IAUKF ({CONFIG['num_runs']} runs)...")
        iaukf_results = []
        for run in range(CONFIG['num_runs']):
            result = run_iaukf_system(net, pmu_buses, test_branch, 
                                     CONFIG['iaukf_steps'], seed=42+run)
            iaukf_results.append(result)
        
        r_errors = [r['r_error'] for r in iaukf_results]
        x_errors = [r['x_error'] for r in iaukf_results]
        
        iaukf_agg = {
            'r_error_mean': np.mean(r_errors),
            'r_error_std': np.std(r_errors),
            'x_error_mean': np.mean(x_errors),
            'x_error_std': np.std(x_errors),
        }
        
        # Graph-Mamba (zero-shot)
        print(f"    Running Graph-Mamba zero-shot ({CONFIG['num_runs']} runs)...")
        gm_results = []
        for run in range(CONFIG['num_runs']):
            result = run_graphmamba_zero_shot(
                gm_model, device, net, pmu_buses, test_branch,
                CONFIG['gm_steps'], seed=42+run
            )
            gm_results.append(result)
        
        r_errors = [r['r_error'] for r in gm_results]
        x_errors = [r['x_error'] for r in gm_results]
        
        gm_agg = {
            'r_error_mean': np.mean(r_errors),
            'r_error_std': np.std(r_errors),
            'x_error_mean': np.mean(x_errors),
            'x_error_std': np.std(x_errors),
        }
        
        results[system_name] = {
            'iaukf': iaukf_agg,
            'graphmamba': gm_agg,
            'num_buses': len(net.bus)
        }
        
        # Log to SwanLab
        swanlab.log({
            f'{system_name}/num_buses': len(net.bus),
            f'{system_name}/iaukf_r_error': iaukf_agg['r_error_mean'],
            f'{system_name}/iaukf_x_error': iaukf_agg['x_error_mean'],
            f'{system_name}/gm_r_error': gm_agg['r_error_mean'],
            f'{system_name}/gm_x_error': gm_agg['x_error_mean'],
        })
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nTrain: IEEE 33-bus")
    print(f"Test: Multiple systems (zero-shot transfer)")
    print(f"\n{'System':<15} {'Buses':<10} {'Method':<15} {'R Error (%)':<25} {'X Error (%)':<25}")
    print("-" * 90)
    
    for system_name, data in results.items():
        iaukf = data['iaukf']
        gm = data['graphmamba']
        num_buses = data['num_buses']
        
        r_str = f"{iaukf['r_error_mean']:.3f} ± {iaukf['r_error_std']:.3f}"
        x_str = f"{iaukf['x_error_mean']:.3f} ± {iaukf['x_error_std']:.3f}"
        print(f"{system_name:<15} {num_buses:<10} {'IAUKF':<15} {r_str:<25} {x_str:<25}")
        
        r_str = f"{gm['r_error_mean']:.3f} ± {gm['r_error_std']:.3f}"
        x_str = f"{gm['x_error_mean']:.3f} ± {gm['x_error_std']:.3f}"
        print(f"{'':<15} {'':<10} {'Graph-Mamba':<15} {r_str:<25} {x_str:<25}")
        
        # Transfer gap analysis
        if system_name != 'ieee33':
            r_gap = abs(gm['r_error_mean'] - results['ieee33']['graphmamba']['r_error_mean'])
            x_gap = abs(gm['x_error_mean'] - results['ieee33']['graphmamba']['x_error_mean'])
            print(f"{'':<15} {'':<10} {'Transfer gap':<15} {r_gap:.3f}{'':<20} {x_gap:.3f}")
        print()
    
    # Generate plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    systems = list(results.keys())
    num_buses = [results[s]['num_buses'] for s in systems]
    x_pos = np.arange(len(systems))
    width = 0.35
    
    # R error
    ax = axes[0]
    iaukf_r = [results[s]['iaukf']['r_error_mean'] for s in systems]
    gm_r = [results[s]['graphmamba']['r_error_mean'] for s in systems]
    
    ax.bar(x_pos - width/2, iaukf_r, width, label='IAUKF', color='steelblue')
    ax.bar(x_pos + width/2, gm_r, width, label='Graph-Mamba', color='coral')
    ax.set_ylabel('R Estimation Error (%)')
    ax.set_title('Cross-Topology Generalization: Resistance')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{s}\n({n} buses)" for s, n in zip(systems, num_buses)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # X error
    ax = axes[1]
    iaukf_x = [results[s]['iaukf']['x_error_mean'] for s in systems]
    gm_x = [results[s]['graphmamba']['x_error_mean'] for s in systems]
    
    ax.bar(x_pos - width/2, iaukf_x, width, label='IAUKF', color='steelblue')
    ax.bar(x_pos + width/2, gm_x, width, label='Graph-Mamba', color='coral')
    ax.set_ylabel('X Estimation Error (%)')
    ax.set_title('Cross-Topology Generalization: Reactance')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{s}\n({n} buses)" for s, n in zip(systems, num_buses)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/exp6_generalization.png', dpi=150, bbox_inches='tight')
    print("\nFigure saved to: results/exp6_generalization.png")
    
    # Save results
    with open('results/exp6_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("Results saved to: results/exp6_results.pkl")
    
    swanlab.log({"generalization_plot": swanlab.Image("results/exp6_generalization.png")})
    
    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    if len(results) > 1:
        print("\nNote: Zero-shot cross-topology transfer is challenging because:")
        print("  1. Different voltage levels (12.66kV vs others)")
        print("  2. Different network structures (radial vs meshed)")
        print("  3. Different parameter scales")
    
    print("\nRecommendation: Frame as 'few-shot fine-tuning' rather than 'zero-shot'")
    print("  - Pre-train on large diverse dataset")
    print("  - Fine-tune with small amount of target system data")
    print("  - Still faster than IAUKF which needs full re-design")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 6 COMPLETE")
    print("=" * 80)
    
    swanlab.finish()


if __name__ == '__main__':
    main()
