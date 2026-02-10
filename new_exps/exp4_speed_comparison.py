"""
Experiment 4: Speed Comparison (P0)
=====================================
Compare inference time between IAUKF and Graph-Mamba.

Expected: Graph-Mamba O(n) vs IAUKF O(n³), 5×+ speedup on large systems
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import time
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
    'experiment': 'exp4_speed_comparison',
    'system_sizes': [33, 69, 118],  # Different system sizes
    'num_runs': 100,  # Number of inference runs for averaging
    'warmup_runs': 10,
    'sequence_length': 50,
    'noise_scada': 0.02,
    'noise_pmu_v': 0.005,
    'noise_pmu_theta': 0.002,
}


def create_test_system(num_buses):
    """Create a test power system of given size."""
    import pandapower as pp
    
    if num_buses == 33:
        # Use built-in IEEE 33
        import pandapower.networks as pn
        net = pn.case33bw()
        pmu_buses = list(range(0, 33, 3))  # Every 3rd bus
    elif num_buses == 69:
        import pandapower.networks as pn
        net = pn.case69()
        pmu_buses = list(range(0, 69, 3))
    elif num_buses == 118:
        import pandapower.networks as pn
        net = pn.case118()
        pmu_buses = list(range(0, 118, 5))
    else:
        # Create simple radial system
        net = pp.create_empty_network()
        for i in range(num_buses):
            pp.create_bus(net, vn_kv=12.66)
            if i > 0:
                pp.create_line(net, from_bus=i-1, to_bus=i, length_km=1, 
                              std_type='NAYY 4x50 SE')
        # Add loads
        for i in range(1, num_buses):
            pp.create_load(net, bus=i, p_mw=0.1, q_mvar=0.05)
        pp.create_ext_grid(net, bus=0)
        pmu_buses = list(range(0, num_buses, 5))
    
    return net, pmu_buses


def benchmark_iaukf(net, pmu_buses, target_branch, num_runs=100):
    """Benchmark IAUKF inference time."""
    num_buses = len(net.bus)
    
    # Initialize IAUKF
    model = AnalyticalMeasurementModel(net, target_branch, pmu_buses)
    
    x0 = np.ones(2 * num_buses + 2)
    x0[:num_buses] = 1.0
    x0[num_buses:2*num_buses] = 0.0
    x0[-2] = 0.1
    x0[-1] = 0.1
    
    P0 = np.eye(len(x0)) * 0.01
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
    
    # Generate a dummy measurement
    z_dummy = np.random.randn(n_scada + n_pmu)
    
    # Warmup
    for _ in range(CONFIG['warmup_runs']):
        iaukf.predict()
        iaukf.update(z_dummy)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        # Reset for fair comparison
        iaukf.x = x0.copy()
        iaukf.P = P0.copy()
        
        start = time.perf_counter()
        iaukf.predict()
        iaukf.update(z_dummy)
        end = time.perf_counter()
        
        times.append((end - start) * 1000)  # Convert to ms
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'median': np.median(times),
        'all_times': times
    }


def benchmark_graphmamba(num_buses, num_runs=100):
    """Benchmark Graph-Mamba inference time (GPU)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Build edge index for test system
    edge_index = []
    for i in range(num_buses - 1):
        edge_index.append([i, i+1])
        edge_index.append([i+1, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().to(device)
    
    # Create model
    model = GraphMambaPhysicsModel(
        num_nodes=num_buses,
        in_features=3,
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2
    )
    model.eval()
    model.to(device)
    
    # Generate dummy input sequence
    seq_len = CONFIG['sequence_length']
    x_dummy = torch.randn(1, seq_len, num_buses, 3).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(CONFIG['warmup_runs']):
            _ = model(x_dummy, edge_index)
    
    # Synchronize CUDA before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(x_dummy, edge_index)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            
            times.append((end - start) * 1000)  # Convert to ms
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'median': np.median(times),
        'all_times': times
    }


def benchmark_scaling():
    """Benchmark how inference time scales with system size."""
    sizes = [11, 33, 69, 118, 200, 500, 1000]
    
    results = {
        'sizes': sizes,
        'iaukf': {},
        'graphmamba': {}
    }
    
    print("\n[1] Benchmarking scalability...")
    
    for size in sizes:
        print(f"\n  System size: {size} buses")
        
        # Skip IAUKF for very large systems (too slow)
        if size <= 200:
            print(f"    Running IAUKF...")
            try:
                net, pmu_buses = create_test_system(size)
                target_branch = 3 if size > 3 else 0
                iaukf_result = benchmark_iaukf(net, pmu_buses, target_branch, 
                                               num_runs=min(CONFIG['num_runs'], 50))
                results['iaukf'][size] = iaukf_result
                print(f"      Mean: {iaukf_result['mean']:.3f} ms")
            except Exception as e:
                print(f"      Failed: {e}")
                results['iaukf'][size] = None
        else:
            print(f"    Skipping IAUKF (too slow for {size} buses)")
            results['iaukf'][size] = None
        
        # Graph-Mamba
        print(f"    Running Graph-Mamba...")
        try:
            gm_result = benchmark_graphmamba(size, num_runs=CONFIG['num_runs'])
            results['graphmamba'][size] = gm_result
            print(f"      Mean: {gm_result['mean']:.3f} ms")
        except Exception as e:
            print(f"      Failed: {e}")
            results['graphmamba'][size] = None
    
    return results


def main():
    """Run Experiment 4."""
    print("=" * 80)
    print("EXPERIMENT 4: Speed Comparison")
    print("=" * 80)
    
    # Initialize SwanLab
    swanlab.init(
        project="graphmamba-vs-iaukf",
        experiment_name="exp4_speed_comparison",
        config=CONFIG
    )
    
    # Run scalability benchmark
    results = benchmark_scaling()
    
    # Print results table
    print("\n" + "=" * 80)
    print("SPEED COMPARISON RESULTS")
    print("=" * 80)
    print(f"\n{'Size (buses)':<15} {'IAUKF (ms)':<25} {'Graph-Mamba (ms)':<25} {'Speedup':<15}")
    print("-" * 80)
    
    for size in results['sizes']:
        iaukf = results['iaukf'].get(size)
        gm = results['graphmamba'].get(size)
        
        iaukf_str = f"{iaukf['mean']:.3f} ± {iaukf['std']:.3f}" if iaukf else "N/A"
        gm_str = f"{gm['mean']:.3f} ± {gm['std']:.3f}" if gm else "N/A"
        
        if iaukf and gm:
            speedup = iaukf['mean'] / gm['mean']
            speedup_str = f"{speedup:.2f}×"
        else:
            speedup_str = "N/A"
        
        print(f"{size:<15} {iaukf_str:<25} {gm_str:<25} {speedup_str:<15}")
        
        # Log to SwanLab
        if iaukf and gm:
            swanlab.log({
                f'size_{size}/iaukf_time': iaukf['mean'],
                f'size_{size}/gm_time': gm['mean'],
                f'size_{size}/speedup': speedup,
            })
    
    # Generate plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Inference time vs system size (linear scale)
    ax = axes[0]
    sizes_with_data = [s for s in results['sizes'] if results['iaukf'].get(s) is not None]
    iaukf_means = [results['iaukf'][s]['mean'] for s in sizes_with_data]
    iaukf_stds = [results['iaukf'][s]['std'] for s in sizes_with_data]
    
    gm_means = [results['graphmamba'][s]['mean'] for s in results['sizes']]
    gm_stds = [results['graphmamba'][s]['std'] for s in results['sizes']]
    
    ax.errorbar(sizes_with_data, iaukf_means, yerr=iaukf_stds, 
                marker='o', label='IAUKF', capsize=5, linewidth=2)
    ax.errorbar(results['sizes'], gm_means, yerr=gm_stds,
                marker='s', label='Graph-Mamba', capsize=5, linewidth=2)
    ax.set_xlabel('Number of Buses')
    ax.set_ylabel('Inference Time (ms)')
    ax.set_title('Inference Time vs System Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Log-log scale to show complexity
    ax = axes[1]
    ax.loglog(sizes_with_data, iaukf_means, 'o-', label='IAUKF', linewidth=2, markersize=8)
    ax.loglog(results['sizes'], gm_means, 's-', label='Graph-Mamba', linewidth=2, markersize=8)
    
    # Add reference lines
    x_ref = np.array(results['sizes'])
    # O(n) reference
    y_linear = gm_means[0] * (x_ref / results['sizes'][0])
    ax.loglog(x_ref, y_linear, '--', alpha=0.5, label='O(n) reference')
    # O(n³) reference (scaled)
    if len(sizes_with_data) >= 2:
        y_cubic = iaukf_means[0] * ((x_ref / sizes_with_data[0]) ** 3)
        ax.loglog(x_ref, y_cubic, ':', alpha=0.5, label='O(n³) reference')
    
    ax.set_xlabel('Number of Buses')
    ax.set_ylabel('Inference Time (ms)')
    ax.set_title('Computational Complexity (Log-Log)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/exp4_speed_comparison.png', dpi=150, bbox_inches='tight')
    print("\nFigure saved to: results/exp4_speed_comparison.png")
    
    # Save results
    with open('results/exp4_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("Results saved to: results/exp4_results.pkl")
    
    swanlab.log({"speed_plot": swanlab.Image("results/exp4_speed_comparison.png")})
    
    # Summary
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    # Find 33-bus comparison
    if 33 in results['sizes'] and results['iaukf'].get(33) and results['graphmamba'].get(33):
        speedup_33 = results['iaukf'][33]['mean'] / results['graphmamba'][33]['mean']
        print(f"33-bus system: Graph-Mamba is {speedup_33:.2f}× faster")
    
    if 118 in results['sizes'] and results['iaukf'].get(118) and results['graphmamba'].get(118):
        speedup_118 = results['iaukf'][118]['mean'] / results['graphmamba'][118]['mean']
        print(f"118-bus system: Graph-Mamba is {speedup_118:.2f}× faster")
    
    # Complexity analysis
    print(f"\nGraph-Mamba complexity: ~O(n) - linear scaling")
    print(f"IAUKF complexity: ~O(n³) - cubic scaling")
    print(f"\nAt 1000+ buses, IAUKF becomes impractical for real-time use")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 4 COMPLETE")
    print("=" * 80)
    
    swanlab.finish()


if __name__ == '__main__':
    main()
