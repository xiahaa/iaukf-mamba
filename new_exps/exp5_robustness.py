"""
Experiment 5: Robustness to Non-Gaussian Noise and Bad Data (P1)
===================================================================
Test estimation under:
- Non-Gaussian noise (Laplacian, Cauchy)
- Bad data/outliers
- Mixed noise scenarios

Expected: Graph-Mamba more robust due to learned representations
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
from scipy import stats

from model.simulation import PowerSystemSimulation
from model.models_analytical import AnalyticalMeasurementModel
from model.iaukf import IAUKF
from graphmamba import GraphMambaPhysicsModel, HAS_MAMBA

# Configuration
CONFIG = {
    'experiment': 'exp5_robustness',
    'system': 'ieee33',
    'branch': 3,  # Branch 3-4
    'num_runs': 3,
    'iaukf_steps': 200,
    'gm_steps': 50,
    'sequence_length': 50,
    'noise_scenarios': {
        'gaussian': {'type': 'gaussian', 'scada_std': 0.02, 'pmu_v_std': 0.005, 'pmu_theta_std': 0.002},
        'laplacian': {'type': 'laplacian', 'scada_scale': 0.02, 'pmu_v_scale': 0.005, 'pmu_theta_scale': 0.002},
        'cauchy': {'type': 'cauchy', 'scada_scale': 0.01, 'pmu_v_scale': 0.0025, 'pmu_theta_scale': 0.001},
        'bad_data_5': {'type': 'gaussian', 'bad_data_prob': 0.05, 'bad_data_mult': 5.0},
        'bad_data_10': {'type': 'gaussian', 'bad_data_prob': 0.10, 'bad_data_mult': 5.0},
        'mixed': {'type': 'mixed', 'components': ['gaussian', 'laplacian', 'cauchy']},
    }
}


def generate_noise(size, noise_type, **params):
    """Generate noise of specified type."""
    if noise_type == 'gaussian':
        return np.random.normal(0, params.get('std', 0.02), size)
    elif noise_type == 'laplacian':
        return np.random.laplace(0, params.get('scale', 0.02), size)
    elif noise_type == 'cauchy':
        return params.get('scale', 0.01) * np.random.standard_cauchy(size)
    else:
        return np.random.normal(0, 0.02, size)


def add_bad_data(noise, prob=0.05, multiplier=5.0):
    """Add bad data (outliers) to noise."""
    mask = np.random.random(len(noise)) < prob
    noise = noise.copy()
    noise[mask] *= multiplier
    return noise


def run_iaukf_robustness(branch_idx, noise_config, steps=200, seed=42):
    """Run IAUKF with specified noise scenario."""
    np.random.seed(seed)
    
    sim = PowerSystemSimulation(steps=steps)
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
    
    P0 = np.eye(len(x0)) * 1e-3
    P0[-2, -2] = 0.5
    P0[-1, -1] = 0.5
    Q0 = np.eye(len(x0)) * 1e-6
    
    n_scada = 3 * num_buses
    n_pmu = 2 * len(sim.pmu_buses)
    
    # R is still Gaussian assumption (limitation of Kalman filter)
    R_diag = np.concatenate([
        np.full(n_scada, 0.02**2),
        np.full(len(sim.pmu_buses), 0.005**2),
        np.full(len(sim.pmu_buses), 0.002**2)
    ])
    R = np.diag(R_diag)
    
    iaukf = IAUKF(model, x0, P0, Q0, R)
    iaukf.b_factor = 0.95
    
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
        
        # Generate noise based on scenario
        noise_type = noise_config.get('type', 'gaussian')
        
        if noise_type == 'gaussian':
            p_noise = generate_noise(num_buses, 'gaussian', std=noise_config.get('scada_std', 0.02))
            q_noise = generate_noise(num_buses, 'gaussian', std=noise_config.get('scada_std', 0.02))
            v_noise = generate_noise(num_buses, 'gaussian', std=noise_config.get('scada_std', 0.02))
            v_pmu_noise = generate_noise(len(sim.pmu_buses), 'gaussian', std=noise_config.get('pmu_v_std', 0.005))
            theta_pmu_noise = generate_noise(len(sim.pmu_buses), 'gaussian', std=noise_config.get('pmu_theta_std', 0.002))
        elif noise_type == 'laplacian':
            p_noise = generate_noise(num_buses, 'laplacian', scale=noise_config.get('scada_scale', 0.02))
            q_noise = generate_noise(num_buses, 'laplacian', scale=noise_config.get('scada_scale', 0.02))
            v_noise = generate_noise(num_buses, 'laplacian', scale=noise_config.get('scada_scale', 0.02))
            v_pmu_noise = generate_noise(len(sim.pmu_buses), 'laplacian', scale=noise_config.get('pmu_v_scale', 0.005))
            theta_pmu_noise = generate_noise(len(sim.pmu_buses), 'laplacian', scale=noise_config.get('pmu_theta_scale', 0.002))
        elif noise_type == 'cauchy':
            p_noise = generate_noise(num_buses, 'cauchy', scale=noise_config.get('scada_scale', 0.01))
            q_noise = generate_noise(num_buses, 'cauchy', scale=noise_config.get('scada_scale', 0.01))
            v_noise = generate_noise(num_buses, 'cauchy', scale=noise_config.get('scada_scale', 0.01))
            v_pmu_noise = generate_noise(len(sim.pmu_buses), 'cauchy', scale=noise_config.get('pmu_v_scale', 0.0025))
            theta_pmu_noise = generate_noise(len(sim.pmu_buses), 'cauchy', scale=noise_config.get('pmu_theta_scale', 0.001))
        else:
            p_noise = np.random.normal(0, 0.02, num_buses)
            q_noise = np.random.normal(0, 0.02, num_buses)
            v_noise = np.random.normal(0, 0.02, num_buses)
            v_pmu_noise = np.random.normal(0, 0.005, len(sim.pmu_buses))
            theta_pmu_noise = np.random.normal(0, 0.002, len(sim.pmu_buses))
        
        # Add bad data if configured
        if 'bad_data_prob' in noise_config:
            p_noise = add_bad_data(p_noise, noise_config['bad_data_prob'], noise_config.get('bad_data_mult', 5.0))
            q_noise = add_bad_data(q_noise, noise_config['bad_data_prob'], noise_config.get('bad_data_mult', 5.0))
            v_noise = add_bad_data(v_noise, noise_config['bad_data_prob'], noise_config.get('bad_data_mult', 5.0))
            v_pmu_noise = add_bad_data(v_pmu_noise, noise_config['bad_data_prob'], noise_config.get('bad_data_mult', 5.0))
            theta_pmu_noise = add_bad_data(theta_pmu_noise, noise_config['bad_data_prob'], noise_config.get('bad_data_mult', 5.0))
        
        p_inj = -sim.net.res_bus.p_mw.values + p_noise
        q_inj = -sim.net.res_bus.q_mvar.values + q_noise
        v_scada = sim.net.res_bus.vm_pu.values + v_noise
        z_scada = np.concatenate([p_inj, q_inj, v_scada])
        
        v_pmu = sim.net.res_bus.vm_pu.values[sim.pmu_buses] + v_pmu_noise
        theta_pmu = np.radians(sim.net.res_bus.va_degree.values[sim.pmu_buses]) + theta_pmu_noise
        z_pmu = np.concatenate([v_pmu, theta_pmu])
        
        z = np.concatenate([z_scada, z_pmu])
        
        iaukf.predict()
        iaukf.update(z)
        
        r_history.append(iaukf.x[-2])
        x_history.append(iaukf.x[-1])
    
    # Calculate error
    start_avg = len(r_history) // 2
    if len(r_history) < 50:
        return {'r_error': float('inf'), 'x_error': float('inf')}
    
    r_final = np.mean(r_history[start_avg:])
    x_final = np.mean(x_history[start_avg:])
    
    r_error = abs(r_final - r_true) / r_true * 100
    x_error = abs(x_final - x_true) / x_true * 100
    
    return {
        'r_error': r_error,
        'x_error': x_error,
        'r_true': r_true,
        'x_true': x_true,
        'r_history': r_history,
        'x_history': x_history
    }


def load_graphmamba_model():
    """Load pretrained Physics-Informed Graph-Mamba model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
        print("  Warning: No checkpoint loaded")
    
    model.eval()
    model.to(device)
    return model, device


def run_graphmamba_robustness(model, device, branch_idx, noise_config, steps=50, seed=42):
    """Run Graph-Mamba with specified noise scenario."""
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
    
    # Generate measurements with noise
    sequences = []
    noise_type = noise_config.get('type', 'gaussian')
    
    for t in range(steps):
        sim.net.load.p_mw = p_base
        sim.net.load.q_mvar = q_base
        
        try:
            pp.runpp(sim.net, algorithm='nr', numba=False)
        except:
            continue
        
        # Generate noise
        if noise_type == 'gaussian':
            p_noise = generate_noise(num_buses, 'gaussian', std=noise_config.get('scada_std', 0.02))
            q_noise = generate_noise(num_buses, 'gaussian', std=noise_config.get('scada_std', 0.02))
            v_noise = generate_noise(num_buses, 'gaussian', std=noise_config.get('scada_std', 0.02))
        elif noise_type == 'laplacian':
            p_noise = generate_noise(num_buses, 'laplacian', scale=noise_config.get('scada_scale', 0.02))
            q_noise = generate_noise(num_buses, 'laplacian', scale=noise_config.get('scada_scale', 0.02))
            v_noise = generate_noise(num_buses, 'laplacian', scale=noise_config.get('scada_scale', 0.02))
        elif noise_type == 'cauchy':
            p_noise = generate_noise(num_buses, 'cauchy', scale=noise_config.get('scada_scale', 0.01))
            q_noise = generate_noise(num_buses, 'cauchy', scale=noise_config.get('scada_scale', 0.01))
            v_noise = generate_noise(num_buses, 'cauchy', scale=noise_config.get('scada_scale', 0.01))
        else:
            p_noise = np.random.normal(0, 0.02, num_buses)
            q_noise = np.random.normal(0, 0.02, num_buses)
            v_noise = np.random.normal(0, 0.02, num_buses)
        
        # Add bad data if configured
        if 'bad_data_prob' in noise_config:
            p_noise = add_bad_data(p_noise, noise_config['bad_data_prob'], noise_config.get('bad_data_mult', 5.0))
            q_noise = add_bad_data(q_noise, noise_config['bad_data_prob'], noise_config.get('bad_data_mult', 5.0))
            v_noise = add_bad_data(v_noise, noise_config['bad_data_prob'], noise_config.get('bad_data_mult', 5.0))
        
        p_inj = -sim.net.res_bus.p_mw.values + p_noise
        q_inj = -sim.net.res_bus.q_mvar.values + q_noise
        v_scada = sim.net.res_bus.vm_pu.values + v_noise
        
        node_features = np.stack([p_inj, q_inj, v_scada], axis=1)
        sequences.append(node_features)
    
    x = torch.tensor(np.array(sequences), dtype=torch.float32).unsqueeze(0).to(device)
    
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
    """Run Experiment 5."""
    print("=" * 80)
    print("EXPERIMENT 5: Robustness to Non-Gaussian Noise and Bad Data")
    print("=" * 80)
    
    # Initialize SwanLab
    swanlab.init(
        project="graphmamba-vs-iaukf",
        experiment_name="exp5_robustness",
        config=CONFIG
    )
    
    # Load Graph-Mamba model
    print("\n[1] Loading Graph-Mamba model...")
    gm_model, device = load_graphmamba_model()
    
    results = {}
    
    # Test each noise scenario
    for scenario_name, noise_config in CONFIG['noise_scenarios'].items():
        print(f"\n[2] Testing scenario: {scenario_name}")
        print(f"    Config: {noise_config}")
        
        # IAUKF runs
        print(f"    Running IAUKF ({CONFIG['num_runs']} runs)...")
        iaukf_results = []
        for run in range(CONFIG['num_runs']):
            result = run_iaukf_robustness(
                CONFIG['branch'],
                noise_config,
                CONFIG['iaukf_steps'],
                seed=42+run
            )
            iaukf_results.append(result)
        
        r_errors = [r['r_error'] for r in iaukf_results if r['r_error'] < 100]
        x_errors = [r['x_error'] for r in iaukf_results if r['x_error'] < 100]
        
        iaukf_agg = {
            'r_error_mean': np.mean(r_errors) if r_errors else float('inf'),
            'r_error_std': np.std(r_errors) if r_errors else 0,
            'x_error_mean': np.mean(x_errors) if x_errors else float('inf'),
            'x_error_std': np.std(x_errors) if x_errors else 0,
        }
        
        # Graph-Mamba runs
        print(f"    Running Graph-Mamba ({CONFIG['num_runs']} runs)...")
        gm_results = []
        for run in range(CONFIG['num_runs']):
            result = run_graphmamba_robustness(
                gm_model,
                device,
                CONFIG['branch'],
                noise_config,
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
        }
        
        results[scenario_name] = {
            'iaukf': iaukf_agg,
            'graphmamba': gm_agg,
            'config': noise_config
        }
        
        # Log to SwanLab
        swanlab.log({
            f'{scenario_name}/iaukf_r_error': iaukf_agg['r_error_mean'],
            f'{scenario_name}/iaukf_x_error': iaukf_agg['x_error_mean'],
            f'{scenario_name}/gm_r_error': gm_agg['r_error_mean'],
            f'{scenario_name}/gm_x_error': gm_agg['x_error_mean'],
        })
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Scenario':<15} {'Method':<15} {'R Error (%)':<25} {'X Error (%)':<25}")
    print("-" * 80)
    
    for scenario_name, data in results.items():
        iaukf = data['iaukf']
        gm = data['graphmamba']
        
        r_str = f"{iaukf['r_error_mean']:.3f} ± {iaukf['r_error_std']:.3f}"
        x_str = f"{iaukf['x_error_mean']:.3f} ± {iaukf['x_error_std']:.3f}"
        print(f"{scenario_name:<15} {'IAUKF':<15} {r_str:<25} {x_str:<25}")
        
        r_str = f"{gm['r_error_mean']:.3f} ± {gm['r_error_std']:.3f}"
        x_str = f"{gm['x_error_mean']:.3f} ± {gm['x_error_std']:.3f}"
        print(f"{'':<15} {'Graph-Mamba':<15} {r_str:<25} {x_str:<25}")
        print()
    
    # Generate plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    scenarios = list(results.keys())
    x_pos = np.arange(len(scenarios))
    width = 0.35
    
    # R error
    ax = axes[0]
    iaukf_r = [results[s]['iaukf']['r_error_mean'] for s in scenarios]
    gm_r = [results[s]['graphmamba']['r_error_mean'] for s in scenarios]
    
    ax.bar(x_pos - width/2, iaukf_r, width, label='IAUKF', color='steelblue')
    ax.bar(x_pos + width/2, gm_r, width, label='Graph-Mamba', color='coral')
    ax.set_ylabel('R Estimation Error (%)')
    ax.set_title('Robustness: Resistance Estimation')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # X error
    ax = axes[1]
    iaukf_x = [results[s]['iaukf']['x_error_mean'] for s in scenarios]
    gm_x = [results[s]['graphmamba']['x_error_mean'] for s in scenarios]
    
    ax.bar(x_pos - width/2, iaukf_x, width, label='IAUKF', color='steelblue')
    ax.bar(x_pos + width/2, gm_x, width, label='Graph-Mamba', color='coral')
    ax.set_ylabel('X Estimation Error (%)')
    ax.set_title('Robustness: Reactance Estimation')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/exp5_robustness.png', dpi=150, bbox_inches='tight')
    print("\nFigure saved to: results/exp5_robustness.png")
    
    # Save results
    with open('results/exp5_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("Results saved to: results/exp5_results.pkl")
    
    swanlab.log({"robustness_plot": swanlab.Image("results/exp5_robustness.png")})
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 5 COMPLETE")
    print("=" * 80)
    
    swanlab.finish()


if __name__ == '__main__':
    main()
