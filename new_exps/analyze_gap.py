"""
Analyze the accuracy gap between Graph-Mamba and Multi-snapshot IAUKF.
"""

import sys
sys.path.insert(0, '..')

import numpy as np
import torch
import pandapower as pp

from model.simulation import PowerSystemSimulation
from model.models_analytical import AnalyticalMeasurementModel
from model.iaukf import IAUKF, IAUKFMultiSnapshot
from graphmamba import GraphMambaPhysicsModel

print("=" * 70)
print("Gap Analysis: Graph-Mamba vs Multi-snapshot IAUKF")
print("=" * 70)

# Common setup
STEPS = 300
sim = PowerSystemSimulation(steps=STEPS)
num_buses = len(sim.net.bus)
target_branch = 3
pmu_buses = sim.pmu_buses

r_true = sim.net.line.at[target_branch, 'r_ohm_per_km']
x_true = sim.net.line.at[target_branch, 'x_ohm_per_km']

print(f"\nTrue: R={r_true:.4f}, X={x_true:.4f}")

# Generate measurements (constant loads)
p_load_base = sim.net.load.p_mw.values.copy()
q_load_base = sim.net.load.q_mvar.values.copy()

measurements = []
np.random.seed(42)

for t in range(STEPS):
    sim.net.load.p_mw = p_load_base
    sim.net.load.q_mvar = q_load_base
    pp.runpp(sim.net, algorithm='nr', numba=False, verbose=False)
    
    p_inj = -sim.net.res_bus.p_mw.values
    q_inj = -sim.net.res_bus.q_mvar.values
    v_scada = sim.net.res_bus.vm_pu.values
    z_scada = np.concatenate([p_inj, q_inj, v_scada])
    z_scada += np.random.normal(0, 0.02, len(z_scada))
    
    v_pmu = sim.net.res_bus.vm_pu.values[pmu_buses]
    theta_pmu = np.radians(sim.net.res_bus.va_degree.values[pmu_buses])
    z_pmu = np.concatenate([v_pmu, theta_pmu])
    z_pmu += np.concatenate([
        np.random.normal(0, 0.005, len(v_pmu)),
        np.random.normal(0, 0.002, len(theta_pmu))
    ])
    measurements.append(np.concatenate([z_scada, z_pmu]))

# Test 1: Multi-snapshot IAUKF (t=3) - the gold standard
print("\n[1] Multi-snapshot IAUKF (t=3)...")
model = AnalyticalMeasurementModel(sim.net, target_branch, pmu_buses)
x0 = np.ones(2 * num_buses + 2)
x0[:num_buses] = 1.0
x0[num_buses:2*num_buses] = 0.0
x0[-2] = 0.01
x0[-1] = 0.01

P0 = np.eye(len(x0)) * 0.01
P0[-2, -2] = 0.1
P0[-1, -1] = 0.1

Q0 = np.eye(len(x0)) * 1e-6
Q0[-2, -2] = 1e-6
Q0[-1, -1] = 1e-6

n_scada = 3 * num_buses
R_diag = np.concatenate([
    np.full(n_scada, 0.02**2),
    np.full(len(pmu_buses), 0.005**2),
    np.full(len(pmu_buses), 0.002**2)
])
R = np.diag(R_diag)

iaukf_ms = IAUKFMultiSnapshot(model, x0, P0, Q0, R, num_snapshots=3)
iaukf_ms.b_factor = 0.96

r_hist_ms = []
for z in measurements:
    iaukf_ms.predict()
    iaukf_ms.update(z)
    params = iaukf_ms.get_parameters()
    r_hist_ms.append(params[0])

r_final_ms = np.mean(r_hist_ms[STEPS//2:])
error_ms = abs(r_final_ms - r_true) / r_true * 100
print(f"    Final R: {r_final_ms:.4f} (Error: {error_ms:.2f}%)")

# Analyze what multi-snapshot uses that Graph-Mamba doesn't
print("\n[2] Key differences between methods:")
print("    Multi-snapshot advantages:")
print("    - Uses 3 snapshots simultaneously (more data per update)")
print("    - Iterative refinement over 300 steps")
print("    - Explicit power flow model (h function)")
print("    - Adaptive noise estimation (NSE)")
print()
print("    Graph-Mamba limitations:")
print("    - Single forward pass (no iteration)")
print("    - Fixed learned weights (no adaptation during inference)")
print("    - Trained on varied data (not just constant params)")

# Test 2: What if Graph-Mamba had perfect information?
print("\n[3] Theoretical analysis:")
print(f"    Multi-snapshot uses: 300 steps × 3 snapshots = 900 measurements")
print(f"    Graph-Mamba uses: 50 timesteps (typical sequence length)")
print(f"    Data ratio: 900/50 = 18× more data for multi-snapshot")

# Test 3: Test with longer sequence
print("\n[4] Testing Graph-Mamba with longer sequences...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    checkpoint = torch.load('../checkpoints/graph_mamba_physics_best.pt', 
                           map_location=device, weights_only=False)
    gm_model = GraphMambaPhysicsModel(
        num_nodes=33,
        in_features=3,
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2
    ).to(device)
    gm_model.load_state_dict(checkpoint['model_state_dict'])
    gm_model.eval()
    
    # Build edge index
    edge_index = []
    for _, line in sim.net.line.iterrows():
        from_bus = int(line.from_bus)
        to_bus = int(line.to_bus)
        edge_index.append([from_bus, to_bus])
        edge_index.append([to_bus, from_bus])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Test with different sequence lengths
    for seq_len in [50, 100, 200, 300]:
        if seq_len > len(measurements):
            break
            
        data = np.zeros((seq_len, num_buses, 3))
        n_scada = 3 * num_buses
        
        for t, m in enumerate(measurements[:seq_len]):
            z_scada = m[:n_scada]
            data[t, :, 0] = z_scada[:num_buses]
            data[t, :, 1] = z_scada[num_buses:2*num_buses]
            data[t, :, 2] = z_scada[2*num_buses:]
        
        x_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
        edge_tensor = edge_index.to(device)
        
        with torch.no_grad():
            pred = gm_model(x_tensor, edge_tensor)
        
        pred_np = pred.squeeze(0).cpu().numpy()
        r_pred, x_pred = pred_np[0], pred_np[1]
        r_error = abs(r_pred - r_true) / r_true * 100
        
        print(f"    Seq length {seq_len}: R error = {r_error:.2f}%")
        
except Exception as e:
    print(f"    Could not load model: {e}")

print("\n[5] Recommendations to close the gap:")
print("    1. Train specifically on constant-parameter episodes")
print("    2. Use longer sequences (200-300 timesteps)")
print("    3. Larger model capacity (128+ dim)")
print("    4. Ensemble of models")
print("    5. Test-time adaptation (iterative refinement)")
print("    6. Multi-scale temporal modeling")

print("\n" + "=" * 70)
