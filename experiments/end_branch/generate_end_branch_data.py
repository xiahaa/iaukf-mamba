"""
End Branch Data Generation for Graph Mamba
============================================

Generates training data for end branch (21-22) which has:
- Low power flow (28x less current than main feeder)
- Poor observability for IAUKF (~83% error)

Goal: Show Graph Mamba can handle low-observability scenarios better
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
import pickle
from tqdm import tqdm
import pandapower as pp
import pandapower.networks as nw

# Configuration
NUM_TRAIN_EPISODES = 800
NUM_VAL_EPISODES = 100
NUM_TEST_EPISODES = 100
STEPS_PER_EPISODE = 200

# Target END BRANCH 21-22
TARGET_FROM_BUS = 21
TARGET_TO_BUS = 22

DATA_DIR = 'data/end_branch'
os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 70)
print("END BRANCH DATA GENERATION")
print("=" * 70)
print(f"\nTarget: Branch {TARGET_FROM_BUS}-{TARGET_TO_BUS} (END BRANCH)")
print(f"\nConfiguration:")
print(f"  Train episodes: {NUM_TRAIN_EPISODES}")
print(f"  Val episodes: {NUM_VAL_EPISODES}")
print(f"  Test episodes: {NUM_TEST_EPISODES}")
print(f"  Steps per episode: {STEPS_PER_EPISODE}")
print(f"  Output directory: {DATA_DIR}")


def generate_dataset(num_episodes, steps_per_episode, start_seed=42, desc="Dataset"):
    """Generate dataset for end branch estimation."""
    print(f"\n[Generating {desc}]")
    
    dataset = []
    
    # Create network
    net = nw.case33bw()
    
    # Find target line
    line_mask = (net.line.from_bus == TARGET_FROM_BUS) & (net.line.to_bus == TARGET_TO_BUS)
    if not line_mask.any():
        # Try reverse direction
        line_mask = (net.line.from_bus == TARGET_TO_BUS) & (net.line.to_bus == TARGET_FROM_BUS)
    
    if not line_mask.any():
        raise ValueError(f"Line {TARGET_FROM_BUS}-{TARGET_TO_BUS} not found!")
    
    line_idx = net.line[line_mask].index[0]
    r_true = net.line.at[line_idx, 'r_ohm_per_km']
    x_true = net.line.at[line_idx, 'x_ohm_per_km']
    
    print(f"  Target line index: {line_idx}")
    print(f"  True R: {r_true:.4f} Ω/km")
    print(f"  True X: {x_true:.4f} Ω/km")
    
    # Edge index
    from_bus = np.array(net.line.from_bus.values)
    to_bus = np.array(net.line.to_bus.values)
    edge_index_np = np.stack([from_bus, to_bus])
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    num_buses = len(net.bus)
    
    # PMU locations
    pmu_buses = [3, 6, 9, 11, 14, 17, 19, 22, 24, 26, 29, 32]
    
    pbar = tqdm(range(num_episodes), desc=f"  {desc}", ncols=100)
    
    for i in pbar:
        # Fresh network for each episode
        net = nw.case33bw()
        
        # Get base loads
        p_load_base = net.load.p_mw.values.copy()
        q_load_base = net.load.q_mvar.values.copy()
        
        z_scada_list = []
        z_pmu_list = []
        
        np.random.seed(start_seed + i)
        
        for t in range(steps_per_episode):
            # Constant loads (for fair comparison)
            net.load.p_mw = p_load_base
            net.load.q_mvar = q_load_base
            
            # Power flow
            try:
                pp.runpp(net, algorithm='nr', numba=False)
            except:
                continue
            
            # SCADA measurements
            p_inj = -net.res_bus.p_mw.values
            q_inj = -net.res_bus.q_mvar.values
            v_scada = net.res_bus.vm_pu.values
            
            z_scada = np.concatenate([p_inj, q_inj, v_scada])
            z_scada_noisy = z_scada + np.random.normal(0, 0.02, len(z_scada))
            z_scada_list.append(z_scada_noisy)
            
            # PMU measurements
            v_pmu = net.res_bus.vm_pu.values[pmu_buses]
            theta_pmu = np.radians(net.res_bus.va_degree.values[pmu_buses])
            z_pmu = np.concatenate([v_pmu, theta_pmu])
            noise_pmu = np.concatenate([
                np.random.normal(0, 0.005, len(v_pmu)),
                np.random.normal(0, 0.002, len(theta_pmu))
            ])
            z_pmu_list.append(z_pmu + noise_pmu)
        
        # Process
        z_scada = np.array(z_scada_list)
        
        p_part = z_scada[:, :num_buses]
        q_part = z_scada[:, num_buses:2*num_buses]
        v_part = z_scada[:, 2*num_buses:]
        snapshot_features = np.stack([p_part, q_part, v_part], axis=2)
        
        episode_data = {
            'snapshots': torch.from_numpy(snapshot_features).float(),
            'edge_index': edge_index,
            'true_params': torch.tensor([r_true, x_true], dtype=torch.float32),
            'z_pmu': torch.from_numpy(np.array(z_pmu_list)).float(),
            'episode_id': i,
            'seed': start_seed + i,
            'target_line': f"{TARGET_FROM_BUS}-{TARGET_TO_BUS}"
        }
        dataset.append(episode_data)
        
        if (i + 1) % 100 == 0:
            pbar.set_postfix({'R': f"{r_true:.4f}", 'X': f"{x_true:.4f}"})
    
    pbar.close()
    print(f"  ✓ Generated {num_episodes} episodes")
    
    return dataset


def main():
    print("\n" + "=" * 70)
    print("GENERATING TRAINING SET")
    print("=" * 70)
    train_data = generate_dataset(
        NUM_TRAIN_EPISODES, STEPS_PER_EPISODE, start_seed=42, desc="Training"
    )
    
    print("\n" + "=" * 70)
    print("GENERATING VALIDATION SET")
    print("=" * 70)
    val_data = generate_dataset(
        NUM_VAL_EPISODES, STEPS_PER_EPISODE, start_seed=10000, desc="Validation"
    )
    
    print("\n" + "=" * 70)
    print("GENERATING TEST SET")
    print("=" * 70)
    test_data = generate_dataset(
        NUM_TEST_EPISODES, STEPS_PER_EPISODE, start_seed=20000, desc="Test"
    )
    
    # Save
    print("\n" + "=" * 70)
    print("SAVING DATASETS")
    print("=" * 70)
    
    with open(os.path.join(DATA_DIR, 'train_data.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    print(f"  ✓ Saved: {DATA_DIR}/train_data.pkl")
    
    with open(os.path.join(DATA_DIR, 'val_data.pkl'), 'wb') as f:
        pickle.dump(val_data, f)
    print(f"  ✓ Saved: {DATA_DIR}/val_data.pkl")
    
    with open(os.path.join(DATA_DIR, 'test_data.pkl'), 'wb') as f:
        pickle.dump(test_data, f)
    print(f"  ✓ Saved: {DATA_DIR}/test_data.pkl")
    
    # Stats
    r_true = train_data[0]['true_params'][0].item()
    x_true = train_data[0]['true_params'][1].item()
    
    print(f"\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"\nTarget: End Branch {TARGET_FROM_BUS}-{TARGET_TO_BUS}")
    print(f"  R_true: {r_true:.4f} Ω/km")
    print(f"  X_true: {x_true:.4f} Ω/km")
    print(f"\nIAUKF baseline error on this branch: ~83% (R), ~87% (X)")
    print(f"Goal: Show Graph Mamba can do better!")
    
    print(f"\n" + "=" * 70)
    print("✓ DATA GENERATION COMPLETE!")
    print("=" * 70)
    print(f"\nNext: python experiments/end_branch/train_end_branch_mamba.py")


if __name__ == "__main__":
    main()
