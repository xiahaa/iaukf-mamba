"""
Experiment 2: Dynamic Parameter Tracking (P0)
===============================================
Test tracking capability for time-varying parameters.

Scenarios:
A. Linear drift (aging)
B. Step mutation (fault/reconfiguration)
C. Periodic fluctuation (temperature/daily cycle)

Expected: Graph-Mamba tracks faster with less lag than IAUKF
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
import swanlab

from model.simulation import PowerSystemSimulation
from model.models_analytical import AnalyticalMeasurementModel
from model.iaukf import IAUKF
from graphmamba.graph_mamba import GraphMambaModel, HAS_MAMBA

# Configuration
CONFIG = {
    'experiment': 'exp2_dynamic_tracking',
    'system': 'ieee33',
    'branch': 3,  # Branch 3-4
    'steps': 300,
    'sequence_length': 50,
    'noise_scada': 0.02,
    'noise_pmu_v': 0.005,
    'noise_pmu_theta': 0.002,
}


def generate_linear_drift(base_value, steps, total_drift_percent=20):
    """Generate linear parameter drift (aging simulation)."""
    drift = np.linspace(0, base_value * total_drift_percent / 100, steps)
    return base_value + drift


def generate_step_mutation(base_value, steps, mutation_step=100, mutation_percent=30):
    """Generate step mutation (fault/reconfiguration)."""
    values = np.ones(steps) * base_value
    values[mutation_step:] = base_value * (1 + mutation_percent / 100)
    return values


def generate_periodic_fluctuation(base_value, steps, amplitude_percent=10, period=50):
    """Generate periodic fluctuation (daily temperature cycle)."""
    t = np.arange(steps)
    fluctuation = base_value * (amplitude_percent / 100) * np.sin(2 * np.pi * t / period)
    return base_value + fluctuation


def run_iaukf_tracking(branch_idx, r_true_series, x_true_series, seed=42):
    """Run IAUKF tracking on time-varying parameters."""
    np.random.seed(seed)
    
    steps = len(r_true_series)
    sim = PowerSystemSimulation(steps=steps)
    
    # Create model
    model = AnalyticalMeasurementModel(sim.net, branch_idx, sim.pmu_buses)
    num_buses = len(sim.net.bus)
    
    # Initialize IAUKF
    x0 = np.ones(2 * num_buses + 2)
    x0[:num_buses] = 1.0
    x0[num_buses:2*num_buses] = 0.0
    x0[-2] = r_true_series[0]  # Start with true initial value
    x0[-1] = x_true_series[0]
    
    P0 = np.eye(len(x0)) * 1e-6
    Q0 = np.eye(len(x0)) * 1e-4  # Higher Q for tracking
    
    n_scada = 3 * num_buses
    R_diag = np.concatenate([
        np.full(n_scada, CONFIG['noise_scada']**2),
        np.full(len(sim.pmu_buses), CONFIG['noise_pmu_v']**2),
        np.full(len(sim.pmu_buses), CONFIG['noise_pmu_theta']**2)
    ])
    R = np.diag(R_diag)
    
    iaukf = IAUKF(model, x0, P0, Q0, R)
    iaukf.b_factor = 0.95  # Lower forgetting factor for faster adaptation
    
    import pandapower as pp
    p_base = sim.net.load.p_mw.values.copy()
    q_base = sim.net.load.q_mvar.values.copy()
    
    r_estimates = []
    x_estimates = []
    
    for t in range(steps):
        # Update true parameters in network
        sim.net.line.at[branch_idx, 'r_ohm_per_km'] = r_true_series[t]
        sim.net.line.at[branch_idx, 'x_ohm_per_km'] = x_true_series[t]
        
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
        
        r_estimates.append(iaukf.x[-2])
        x_estimates.append(iaukf.x[-1])
    
    return {
        'r_estimates': np.array(r_estimates),
        'x_estimates': np.array(x_estimates),
        'r_true': r_true_series,
        'x_true': x_true_series
    }


def load_graphmamba_model():
    """Load pretrained Graph-Mamba model."""
    checkpoint_path = '../checkpoints/graph_mamba_phase3_best.pt'  # Use phase 3 (time-varying) if available
    
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    model = GraphMambaModel(
        num_nodes=33,
        in_features=3,
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2
    )
    
    if os.path.exists(checkpoint_path):
        print(f"  Loading checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"  Checkpoint loaded successfully")
        except RuntimeError as e:
            print(f"  Warning: Checkpoint mismatch, trying phase 2...")
            checkpoint_path = '../checkpoints/graph_mamba_phase2_best.pt'
            if os.path.exists(checkpoint_path):
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    print(f"  Phase 2 checkpoint loaded successfully")
                except RuntimeError as e2:
                    print(f"  Warning: Checkpoint mismatch - {e2}")
                    print(f"  Using randomly initialized model")
            else:
                print("  Warning: No checkpoint found, using random init")
    else:
        # Fall back to phase 2
        checkpoint_path = '../checkpoints/graph_mamba_phase2_best.pt'
        if os.path.exists(checkpoint_path):
            print(f"  Loading checkpoint: {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print(f"  Checkpoint loaded successfully")
            except RuntimeError as e:
                print(f"  Warning: Checkpoint mismatch - {e}")
                print(f"  Using randomly initialized model")
        else:
            print("  Warning: No checkpoint found, using random init")
    
    model.eval()
    model.to(device)
    return model, device


def run_graphmamba_tracking(model, device, branch_idx, r_true_series, x_true_series, seed=42):
    """Run Graph-Mamba tracking with sliding window."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    steps = len(r_true_series)
    seq_len = CONFIG['sequence_length']
    
    sim = PowerSystemSimulation(steps=steps)
    num_buses = len(sim.net.bus)
    
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
    
    # Generate all measurements first
    all_measurements = []
    for t in range(steps):
        sim.net.line.at[branch_idx, 'r_ohm_per_km'] = r_true_series[t]
        sim.net.line.at[branch_idx, 'x_ohm_per_km'] = x_true_series[t]
        
        sim.net.load.p_mw = p_base
        sim.net.load.q_mvar = q_base
        
        try:
            pp.runpp(sim.net, algorithm='nr', numba=False)
        except:
            # Use previous measurement if power flow fails
            if all_measurements:
                all_measurements.append(all_measurements[-1])
            continue
        
        p_inj = -sim.net.res_bus.p_mw.values + np.random.normal(0, CONFIG['noise_scada'], num_buses)
        q_inj = -sim.net.res_bus.q_mvar.values + np.random.normal(0, CONFIG['noise_scada'], num_buses)
        v_scada = sim.net.res_bus.vm_pu.values + np.random.normal(0, CONFIG['noise_scada'], num_buses)
        
        node_features = np.stack([p_inj, q_inj, v_scada], axis=1)
        all_measurements.append(node_features)
    
    # Sliding window prediction
    r_estimates = []
    x_estimates = []
    
    for t in range(seq_len, steps):
        # Get window of measurements
        window = np.array(all_measurements[t-seq_len:t])
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(x, edge_index)
            r_pred = output[0, 0]
            x_pred = output[0, 1]
        
        r_estimates.append(r_pred.item())
        x_estimates.append(x_pred.item())
    
    return {
        'r_estimates': np.array(r_estimates),
        'x_estimates': np.array(x_estimates),
        'r_true': r_true_series[seq_len:],
        'x_true': x_true_series[seq_len:]
    }


def calculate_tracking_metrics(estimates, true_values, scenario_name):
    """Calculate tracking performance metrics."""
    # RMSE
    rmse = np.sqrt(np.mean((estimates - true_values) ** 2))
    
    # MAPE
    mape = np.mean(np.abs((estimates - true_values) / true_values)) * 100
    
    # Maximum error
    max_error = np.max(np.abs(estimates - true_values))
    
    # Lag (for step mutation scenario)
    if 'step' in scenario_name.lower():
        # Find where true value changes
        change_idx = np.where(np.diff(true_values) != 0)[0]
        if len(change_idx) > 0:
            change_idx = change_idx[0]
            # Find where estimate reaches 90% of new value
            new_value = true_values[change_idx + 1]
            old_value = true_values[change_idx]
            threshold = old_value + 0.9 * (new_value - old_value)
            
            settling_idx = change_idx + 1
            while settling_idx < len(estimates) and abs(estimates[settling_idx]) < abs(threshold):
                settling_idx += 1
            
            lag = settling_idx - change_idx
        else:
            lag = None
    else:
        lag = None
    
    return {
        'rmse': rmse,
        'mape': mape,
        'max_error': max_error,
        'lag': lag
    }


def main():
    """Run Experiment 2."""
    print("=" * 80)
    print("EXPERIMENT 2: Dynamic Parameter Tracking")
    print("=" * 80)
    
    # Initialize SwanLab
    swanlab.init(
        project="graphmamba-vs-iaukf",
        experiment_name="exp2_dynamic_tracking",
        config=CONFIG
    )
    
    # Get base parameter values
    sim = PowerSystemSimulation(steps=1)
    r_base = sim.net.line.at[CONFIG['branch'], 'r_ohm_per_km']
    x_base = sim.net.line.at[CONFIG['branch'], 'x_ohm_per_km']
    
    # Define scenarios
    scenarios = {
        'linear_drift': {
            'r_series': generate_linear_drift(r_base, CONFIG['steps'], 20),
            'x_series': generate_linear_drift(x_base, CONFIG['steps'], 20),
        },
        'step_mutation': {
            'r_series': generate_step_mutation(r_base, CONFIG['steps'], 100, 30),
            'x_series': generate_step_mutation(x_base, CONFIG['steps'], 100, 30),
        },
        'periodic_fluctuation': {
            'r_series': generate_periodic_fluctuation(r_base, CONFIG['steps'], 10, 50),
            'x_series': generate_periodic_fluctuation(x_base, CONFIG['steps'], 10, 50),
        }
    }
    
    results = {}
    
    # Load Graph-Mamba model
    print("\n[1] Loading Graph-Mamba model...")
    gm_model, device = load_graphmamba_model()
    
    # Run each scenario
    for scenario_name, series in scenarios.items():
        print(f"\n[2] Running scenario: {scenario_name}")
        
        # IAUKF
        print(f"  Running IAUKF...")
        iaukf_result = run_iaukf_tracking(
            CONFIG['branch'],
            series['r_series'],
            series['x_series'],
            seed=42
        )
        
        # Graph-Mamba
        print(f"  Running Graph-Mamba...")
        gm_result = run_graphmamba_tracking(
            gm_model,
            device,
            CONFIG['branch'],
            series['r_series'],
            series['x_series'],
            seed=42
        )
        
        # Calculate metrics
        iaukf_metrics_r = calculate_tracking_metrics(
            iaukf_result['r_estimates'],
            iaukf_result['r_true'],
            scenario_name
        )
        iaukf_metrics_x = calculate_tracking_metrics(
            iaukf_result['x_estimates'],
            iaukf_result['x_true'],
            scenario_name
        )
        
        gm_metrics_r = calculate_tracking_metrics(
            gm_result['r_estimates'],
            gm_result['r_true'],
            scenario_name
        )
        gm_metrics_x = calculate_tracking_metrics(
            gm_result['x_estimates'],
            gm_result['x_true'],
            scenario_name
        )
        
        results[scenario_name] = {
            'iaukf': iaukf_result,
            'graphmamba': gm_result,
            'metrics': {
                'iaukf': {'r': iaukf_metrics_r, 'x': iaukf_metrics_x},
                'graphmamba': {'r': gm_metrics_r, 'x': gm_metrics_x}
            }
        }
        
        # Log metrics
        swanlab.log({
            f'{scenario_name}/iaukf_rmse_r': iaukf_metrics_r['rmse'],
            f'{scenario_name}/iaukf_rmse_x': iaukf_metrics_x['rmse'],
            f'{scenario_name}/gm_rmse_r': gm_metrics_r['rmse'],
            f'{scenario_name}/gm_rmse_x': gm_metrics_x['rmse'],
        })
    
    # Print metrics summary
    print("\n" + "=" * 80)
    print("METRICS SUMMARY")
    print("=" * 80)
    
    for scenario_name, data in results.items():
        print(f"\n{scenario_name.upper()}:")
        print(f"  {'Metric':<15} {'IAUKF (R/X)':<25} {'Graph-Mamba (R/X)':<25}")
        print(f"  {'-'*65}")
        
        metrics = data['metrics']
        print(f"  {'RMSE':<15} "
              f"{metrics['iaukf']['r']['rmse']:.6f} / {metrics['iaukf']['x']['rmse']:.6f}   "
              f"{metrics['graphmamba']['r']['rmse']:.6f} / {metrics['graphmamba']['x']['rmse']:.6f}")
        print(f"  {'MAPE (%)':<15} "
              f"{metrics['iaukf']['r']['mape']:.3f} / {metrics['iaukf']['x']['mape']:.3f}   "
              f"{metrics['graphmamba']['r']['mape']:.3f} / {metrics['graphmamba']['x']['mape']:.3f}")
        print(f"  {'Max Error':<15} "
              f"{metrics['iaukf']['r']['max_error']:.6f} / {metrics['iaukf']['x']['max_error']:.6f}   "
              f"{metrics['graphmamba']['r']['max_error']:.6f} / {metrics['graphmamba']['x']['max_error']:.6f}")
        
        if metrics['iaukf']['r']['lag'] is not None:
            print(f"  {'Lag (steps)':<15} "
                  f"{metrics['iaukf']['r']['lag']}   "
                  f"{metrics['graphmamba']['r']['lag']}")
    
    # Generate plots
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    for idx, (scenario_name, data) in enumerate(results.items()):
        # R tracking
        ax = axes[idx, 0]
        iaukf_data = data['iaukf']
        gm_data = data['graphmamba']
        
        ax.plot(iaukf_data['r_true'], 'k-', label='True', linewidth=2)
        ax.plot(iaukf_data['r_estimates'], 'b--', label='IAUKF', alpha=0.7)
        ax.plot(range(CONFIG['sequence_length'], CONFIG['sequence_length'] + len(gm_data['r_estimates'])),
                gm_data['r_estimates'], 'r-', label='Graph-Mamba', alpha=0.7)
        ax.set_ylabel('R (Ω/km)')
        ax.set_title(f'{scenario_name.replace("_", " ").title()} - Resistance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # X tracking
        ax = axes[idx, 1]
        ax.plot(iaukf_data['x_true'], 'k-', label='True', linewidth=2)
        ax.plot(iaukf_data['x_estimates'], 'b--', label='IAUKF', alpha=0.7)
        ax.plot(range(CONFIG['sequence_length'], CONFIG['sequence_length'] + len(gm_data['x_estimates'])),
                gm_data['x_estimates'], 'r-', label='Graph-Mamba', alpha=0.7)
        ax.set_ylabel('X (Ω/km)')
        ax.set_title(f'{scenario_name.replace("_", " ").title()} - Reactance')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/exp2_dynamic_tracking.png', dpi=150, bbox_inches='tight')
    print("\nFigure saved to: results/exp2_dynamic_tracking.png")
    
    # Save results
    with open('results/exp2_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("Results saved to: results/exp2_results.pkl")
    
    swanlab.log({"tracking_plot": swanlab.Image("results/exp2_dynamic_tracking.png")})
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 2 COMPLETE")
    print("=" * 80)
    
    swanlab.finish()


if __name__ == '__main__':
    main()
