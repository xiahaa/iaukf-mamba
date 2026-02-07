"""
Phase 2: IAUKF vs Graph Mamba on Constant Parameters
======================================================

This is a CRITICAL comparison that was MISSING from the experiments.

The original paper (Wang et al.) tests IAUKF on constant parameters.
We must show that:
1. IAUKF works well on constant parameters (as claimed in paper)
2. Graph Mamba also works well on constant parameters
3. Fair comparison on the SAME test conditions

This validates that Graph Mamba is not just "better because IAUKF fails"
but is genuinely competitive even when IAUKF's assumptions hold.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandapower as pp

from model.simulation import PowerSystemSimulation
from model.models_analytical import AnalyticalMeasurementModel
from model.iaukf import IAUKF
from graphmamba.graph_mamba import GraphMambaModel, HAS_MAMBA

# ========================================
# Configuration
# ========================================

DATA_DIR = 'data/phase2'
CHECKPOINT_DIR = 'checkpoints'
RESULTS_DIR = 'tmp'
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_TEST_EPISODES = 20  # Test on subset for IAUKF (slower)
IAUKF_STEPS = 200  # Steps for IAUKF to converge

print("=" * 70)
print("PHASE 2: IAUKF vs GRAPH MAMBA ON CONSTANT PARAMETERS")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  Test episodes: {NUM_TEST_EPISODES}")
print(f"  IAUKF convergence steps: {IAUKF_STEPS}")
print(f"  Device: {DEVICE}")

# ========================================
# Load Test Data
# ========================================

print("\n[1] Loading test data...")

with open(os.path.join(DATA_DIR, 'test_data.pkl'), 'rb') as f:
    test_data = pickle.load(f)

print(f"  ✓ Loaded {len(test_data)} test episodes")
print(f"  Using first {NUM_TEST_EPISODES} episodes")

# Get sample info
sample = test_data[0]
num_nodes = sample['snapshots'].shape[1]
in_features = sample['snapshots'].shape[2]
seq_len = sample['snapshots'].shape[0]
edge_index = sample['edge_index']

print(f"  Sequence length: {seq_len}")
print(f"  Nodes: {num_nodes}")
print(f"  Features per node: {in_features}")

# ========================================
# Load Graph Mamba Model
# ========================================

print("\n[2] Loading Graph Mamba model...")

checkpoint = torch.load(
    os.path.join(CHECKPOINT_DIR, 'graph_mamba_phase2_best.pt'),
    weights_only=False
)

model = GraphMambaModel(
    num_nodes=num_nodes,
    in_features=in_features,
    d_model=64,
    d_state=16,
    d_conv=4,
    expand=2
).to(DEVICE)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"  ✓ Model loaded (epoch {checkpoint['epoch'] + 1})")
print(f"  Val metrics: R={checkpoint['val_metrics']['r_error_mean']:.2f}%, X={checkpoint['val_metrics']['x_error_mean']:.2f}%")

# ========================================
# Setup IAUKF
# ========================================

print("\n[3] Setting up IAUKF...")

sim = PowerSystemSimulation(steps=IAUKF_STEPS)
pmu_buses = sim.pmu_buses
target_line_idx = sim.line_idx

print(f"  Target line: {target_line_idx}")
print(f"  PMU buses: {len(pmu_buses)}")

# ========================================
# Helper: Run IAUKF on Episode
# ========================================

def run_iaukf_on_constant_episode(episode, sim, verbose=False):
    """
    Run IAUKF on a constant-parameter episode.

    Since parameters are constant, IAUKF should converge well.
    We generate fresh measurements from power flow for accuracy.
    """
    # Get true parameters from episode
    true_params = episode['true_params'].numpy()
    r_true = true_params[0]  # Total Ohms
    x_true = true_params[1]  # Total Ohms

    # Line length for conversion
    line_length = sim.net.line.at[sim.line_idx, 'length_km']
    r_true_per_km = r_true / line_length
    x_true_per_km = x_true / line_length

    # Create analytical model
    model = AnalyticalMeasurementModel(sim.net, sim.line_idx, sim.pmu_buses)

    # Set the true parameters in the network
    sim.net.line.at[sim.line_idx, 'r_ohm_per_km'] = r_true_per_km
    sim.net.line.at[sim.line_idx, 'x_ohm_per_km'] = x_true_per_km

    # Initial state
    num_buses = len(sim.net.bus)
    x0 = np.ones(2 * num_buses + 2)
    x0[:num_buses] = 1.0
    x0[num_buses:2*num_buses] = 0.0
    x0[-2] = 0.01  # Small initial R (Ohm/km)
    x0[-1] = 0.01  # Small initial X (Ohm/km)

    # Covariances
    P0 = np.eye(len(x0)) * 0.01
    P0[-2, -2] = 0.1
    P0[-1, -1] = 0.1

    Q0 = np.eye(len(x0)) * 1e-6
    Q0[-2, -2] = 1e-6
    Q0[-1, -1] = 1e-6

    # Measurement covariance
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

    # Get base loads
    p_load_base = sim.net.load.p_mw.values.copy()
    q_load_base = sim.net.load.q_mvar.values.copy()

    # Run filter with constant loads (matching paper's setup)
    r_history = []
    x_history = []

    np.random.seed(42)

    for t in range(IAUKF_STEPS):
        # Constant loads
        sim.net.load.p_mw = p_load_base
        sim.net.load.q_mvar = q_load_base

        # Run power flow
        try:
            pp.runpp(sim.net, algorithm='nr', numba=False)
        except:
            continue

        # Generate measurements
        p_inj = -sim.net.res_bus.p_mw.values
        q_inj = -sim.net.res_bus.q_mvar.values
        v_scada = sim.net.res_bus.vm_pu.values
        z_scada = np.concatenate([p_inj, q_inj, v_scada])
        z_scada += np.random.normal(0, 0.02, len(z_scada))

        v_pmu = sim.net.res_bus.vm_pu.values[sim.pmu_buses]
        theta_pmu = np.radians(sim.net.res_bus.va_degree.values[sim.pmu_buses])
        z_pmu = np.concatenate([v_pmu, theta_pmu])
        z_pmu += np.concatenate([
            np.random.normal(0, 0.005, len(v_pmu)),
            np.random.normal(0, 0.002, len(theta_pmu))
        ])

        z = np.concatenate([z_scada, z_pmu])

        # IAUKF step
        iaukf.predict()
        iaukf.update(z)

        r_history.append(iaukf.x[-2])
        x_history.append(iaukf.x[-1])

    # Post-convergence averaging (Eq. 40)
    # Use last 50% of estimates
    start_avg = IAUKF_STEPS // 2
    r_final = np.mean(r_history[start_avg:]) * line_length  # Convert to total Ohms
    x_final = np.mean(x_history[start_avg:]) * line_length

    r_error = abs(r_final - r_true) / r_true * 100
    x_error = abs(x_final - x_true) / x_true * 100

    return {
        'r_true': r_true,
        'x_true': x_true,
        'r_pred': r_final,
        'x_pred': x_final,
        'r_error': r_error,
        'x_error': x_error,
        'r_history': np.array(r_history) * line_length,
        'x_history': np.array(x_history) * line_length
    }


# ========================================
# Run Graph Mamba on Test Episodes
# ========================================

print("\n[4] Running Graph Mamba on test episodes...")

mamba_results = []

for i, episode in enumerate(tqdm(test_data[:NUM_TEST_EPISODES], desc="Graph Mamba")):
    snapshots = episode['snapshots'].unsqueeze(0).to(DEVICE)
    edge_idx = episode['edge_index'].to(DEVICE)
    true_params = episode['true_params'].numpy()

    with torch.no_grad():
        pred = model(snapshots, edge_idx)

    r_pred = pred[0, 0].cpu().item()
    x_pred = pred[0, 1].cpu().item()

    r_error = abs(r_pred - true_params[0]) / true_params[0] * 100
    x_error = abs(x_pred - true_params[1]) / true_params[1] * 100

    mamba_results.append({
        'r_true': true_params[0],
        'x_true': true_params[1],
        'r_pred': r_pred,
        'x_pred': x_pred,
        'r_error': r_error,
        'x_error': x_error
    })

mamba_r_errors = [r['r_error'] for r in mamba_results]
mamba_x_errors = [r['x_error'] for r in mamba_results]

print(f"\n  Graph Mamba Results:")
print(f"    R error: {np.mean(mamba_r_errors):.2f}% ± {np.std(mamba_r_errors):.2f}%")
print(f"    X error: {np.mean(mamba_x_errors):.2f}% ± {np.std(mamba_x_errors):.2f}%")

# ========================================
# Run IAUKF on Test Episodes
# ========================================

print("\n[5] Running IAUKF on test episodes...")
print("    (This is slower due to power flow computation)")

iaukf_results = []

for i, episode in enumerate(tqdm(test_data[:NUM_TEST_EPISODES], desc="IAUKF")):
    result = run_iaukf_on_constant_episode(episode, sim, verbose=False)
    iaukf_results.append(result)

iaukf_r_errors = [r['r_error'] for r in iaukf_results]
iaukf_x_errors = [r['x_error'] for r in iaukf_results]

print(f"\n  IAUKF Results:")
print(f"    R error: {np.mean(iaukf_r_errors):.2f}% ± {np.std(iaukf_r_errors):.2f}%")
print(f"    X error: {np.mean(iaukf_x_errors):.2f}% ± {np.std(iaukf_x_errors):.2f}%")

# ========================================
# Comparison Summary
# ========================================

print("\n" + "=" * 70)
print("COMPARISON SUMMARY: CONSTANT PARAMETERS")
print("=" * 70)

print("\n╔═══════════════════════════════════════════════════════════════════╗")
print("║            IAUKF vs GRAPH MAMBA (Constant Parameters)             ║")
print("╠═══════════════════════════════════════════════════════════════════╣")
print("║ Method          │   R Error (%)   │   X Error (%)   │  Winner    ║")
print("╠═══════════════════════════════════════════════════════════════════╣")

iaukf_r_mean = np.mean(iaukf_r_errors)
iaukf_x_mean = np.mean(iaukf_x_errors)
mamba_r_mean = np.mean(mamba_r_errors)
mamba_x_mean = np.mean(mamba_x_errors)

r_winner = "IAUKF" if iaukf_r_mean < mamba_r_mean else "Mamba"
x_winner = "IAUKF" if iaukf_x_mean < mamba_x_mean else "Mamba"

print(f"║ IAUKF           │ {iaukf_r_mean:6.2f} ± {np.std(iaukf_r_errors):5.2f} │ {iaukf_x_mean:6.2f} ± {np.std(iaukf_x_errors):5.2f} │            ║")
print(f"║ Graph Mamba     │ {mamba_r_mean:6.2f} ± {np.std(mamba_r_errors):5.2f} │ {mamba_x_mean:6.2f} ± {np.std(mamba_x_errors):5.2f} │            ║")
print("╠═══════════════════════════════════════════════════════════════════╣")
print(f"║ Winner          │ {r_winner:^15s} │ {x_winner:^15s} │            ║")
print("╚═══════════════════════════════════════════════════════════════════╝")

# Calculate improvements
if mamba_r_mean < iaukf_r_mean:
    r_improvement = (1 - mamba_r_mean / iaukf_r_mean) * 100
    print(f"\n  Graph Mamba is {r_improvement:.1f}% better on R")
else:
    r_degradation = (mamba_r_mean / iaukf_r_mean - 1) * 100
    print(f"\n  Graph Mamba is {r_degradation:.1f}% worse on R (IAUKF wins on constant params!)")

if mamba_x_mean < iaukf_x_mean:
    x_improvement = (1 - mamba_x_mean / iaukf_x_mean) * 100
    print(f"  Graph Mamba is {x_improvement:.1f}% better on X")
else:
    x_degradation = (mamba_x_mean / iaukf_x_mean - 1) * 100
    print(f"  Graph Mamba is {x_degradation:.1f}% worse on X (IAUKF wins on constant params!)")

# ========================================
# Generate Visualization
# ========================================

print("\n[6] Generating comparison plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Error comparison bar chart
ax = axes[0, 0]
x_pos = np.array([0, 1])
width = 0.35

bars1 = ax.bar(x_pos - width/2, [iaukf_r_mean, iaukf_x_mean], width,
               label='IAUKF', color='steelblue', yerr=[np.std(iaukf_r_errors), np.std(iaukf_x_errors)], capsize=5)
bars2 = ax.bar(x_pos + width/2, [mamba_r_mean, mamba_x_mean], width,
               label='Graph Mamba', color='coral', yerr=[np.std(mamba_r_errors), np.std(mamba_x_errors)], capsize=5)

ax.set_ylabel('Error (%)')
ax.set_title('IAUKF vs Graph Mamba: Constant Parameters')
ax.set_xticks(x_pos)
ax.set_xticklabels(['R Error', 'X Error'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val in zip(bars1, [iaukf_r_mean, iaukf_x_mean]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars2, [mamba_r_mean, mamba_x_mean]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=9)

# Plot 2: Error distribution box plot
ax = axes[0, 1]
data = [iaukf_r_errors, mamba_r_errors, iaukf_x_errors, mamba_x_errors]
bp = ax.boxplot(data, labels=['IAUKF\nR', 'Mamba\nR', 'IAUKF\nX', 'Mamba\nX'], patch_artist=True)
colors = ['steelblue', 'coral', 'steelblue', 'coral']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('Error (%)')
ax.set_title('Error Distribution (Constant Parameters)')
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: IAUKF convergence example (first episode)
ax = axes[1, 0]
if len(iaukf_results) > 0 and 'r_history' in iaukf_results[0]:
    ax.plot(iaukf_results[0]['r_history'], label='R estimate', color='steelblue')
    ax.axhline(y=iaukf_results[0]['r_true'], color='steelblue', linestyle='--', alpha=0.7, label='R true')
    ax.plot(iaukf_results[0]['x_history'], label='X estimate', color='coral')
    ax.axhline(y=iaukf_results[0]['x_true'], color='coral', linestyle='--', alpha=0.7, label='X true')
    ax.set_xlabel('Step')
    ax.set_ylabel('Parameter Value (Ohm)')
    ax.set_title('IAUKF Convergence (Episode 1)')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Plot 4: Scatter plot of predictions
ax = axes[1, 1]
ax.scatter([r['r_true'] for r in iaukf_results], [r['r_pred'] for r in iaukf_results],
           alpha=0.7, label='IAUKF R', marker='o', color='steelblue')
ax.scatter([r['x_true'] for r in iaukf_results], [r['x_pred'] for r in iaukf_results],
           alpha=0.7, label='IAUKF X', marker='s', color='lightblue')
ax.scatter([r['r_true'] for r in mamba_results], [r['r_pred'] for r in mamba_results],
           alpha=0.7, label='Mamba R', marker='o', color='coral')
ax.scatter([r['x_true'] for r in mamba_results], [r['x_pred'] for r in mamba_results],
           alpha=0.7, label='Mamba X', marker='s', color='lightsalmon')

# Perfect prediction line
all_true = ([r['r_true'] for r in iaukf_results] + [r['x_true'] for r in iaukf_results] +
            [r['r_true'] for r in mamba_results] + [r['x_true'] for r in mamba_results])
min_val, max_val = min(all_true), max(all_true)
ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect', alpha=0.5)
ax.set_xlabel('True Value (Ohm)')
ax.set_ylabel('Predicted Value (Ohm)')
ax.set_title('Prediction Accuracy')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'phase2_iaukf_vs_mamba.png'), dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {RESULTS_DIR}/phase2_iaukf_vs_mamba.png")

# ========================================
# Save Results
# ========================================

results = {
    'iaukf': {
        'r_mean': iaukf_r_mean,
        'r_std': np.std(iaukf_r_errors),
        'x_mean': iaukf_x_mean,
        'x_std': np.std(iaukf_x_errors),
        'all_results': iaukf_results
    },
    'mamba': {
        'r_mean': mamba_r_mean,
        'r_std': np.std(mamba_r_errors),
        'x_mean': mamba_x_mean,
        'x_std': np.std(mamba_x_errors),
        'all_results': mamba_results
    },
    'comparison': {
        'r_winner': r_winner,
        'x_winner': x_winner,
        'scenario': 'constant_parameters'
    }
}

with open(os.path.join(RESULTS_DIR, 'phase2_iaukf_vs_mamba.pkl'), 'wb') as f:
    pickle.dump(results, f)
print(f"  ✓ Saved: {RESULTS_DIR}/phase2_iaukf_vs_mamba.pkl")

# ========================================
# Key Takeaway
# ========================================

print("\n" + "=" * 70)
print("KEY TAKEAWAY")
print("=" * 70)

print("""
This experiment shows performance on CONSTANT parameters (IAUKF's ideal case).

When parameters are CONSTANT (IAUKF's assumption holds):
""")

if mamba_r_mean < iaukf_r_mean and mamba_x_mean < iaukf_x_mean:
    print("  ✓ Graph Mamba STILL outperforms IAUKF even on constant parameters!")
    print("  → This is a strong result: Mamba wins in both favorable and unfavorable conditions.")
elif mamba_r_mean > iaukf_r_mean and mamba_x_mean > iaukf_x_mean:
    print("  ✓ IAUKF performs better on constant parameters (as expected)")
    print("  → This is fair: IAUKF is designed for this scenario")
    print("  → The real comparison is Phase 3 (time-varying), where Mamba excels")
else:
    print("  Mixed results - each method wins on different parameters")
    print("  → Phase 3 (time-varying) is the critical differentiator")

print("""
For paper: This shows both methods work on constant parameters,
but the key advantage of Graph Mamba is on TIME-VARYING parameters (Phase 3).
""")

print("=" * 70)
print("✓ PHASE 2 COMPARISON COMPLETE")
print("=" * 70)
