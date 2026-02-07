"""
Multi-Branch Data Generation for Graph Mamba
==============================================

Generates training data for estimating MULTIPLE branch parameters simultaneously.
This tests Graph Mamba's ability to learn spatial correlations across the network.

Target branches (4 branches, 8 parameters total):
- Branch 3-4 (main feeder)
- Branch 5-6 (main feeder)  
- Branch 2-19 (lateral)
- Branch 21-22 (end branch)
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

# Target MULTIPLE branches
TARGET_BRANCHES = [
    (3, 4),    # Main feeder
    (5, 6),    # Main feeder
    (2, 19),   # Lateral
    (21, 22),  # End branch
]

DATA_DIR = 'data/multi_branch'
os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 70)
print("MULTI-BRANCH DATA GENERATION")
print("=" * 70)
print(f"\nTarget branches: {TARGET_BRANCHES}")
print(f"Output parameters: {len(TARGET_BRANCHES) * 2} (R and X for each branch)")
print(f"\nConfiguration:")
print(f"  Train episodes: {NUM_TRAIN_EPISODES}")
print(f"  Val episodes: {NUM_VAL_EPISODES}")
print(f"  Test episodes: {NUM_TEST_EPISODES}")
print(f"  Steps per episode: {STEPS_PER_EPISODE}")


def get_line_params(net, from_bus, to_bus):
    """Get line index and true parameters."""
    mask = (net.line.from_bus == from_bus) & (net.line.to_bus == to_bus)
    if not mask.any():
        mask = (net.line.from_bus == to_bus) & (net.line.to_bus == from_bus)
    if not mask.any():
        raise ValueError(f"Line {from_bus}-{to_bus} not found!")
    idx = net.line[mask].index[0]
    r = net.line.at[idx, 'r_ohm_per_km']
    x = net.line.at[idx, 'x_ohm_per_km']
    return idx, r, x


def generate_dataset(num_episodes, steps_per_episode, start_seed=42, desc="Dataset"):
    """Generate dataset for multi-branch estimation."""
    print(f"\n[Generating {desc}]")
    
    dataset = []
    net = nw.case33bw()
    
    # Get all target line info
    line_info = []
    true_params = []
    for from_bus, to_bus in TARGET_BRANCHES:
        idx, r, x = get_line_params(net, from_bus, to_bus)
        line_info.append({'from': from_bus, 'to': to_bus, 'idx': idx, 'r': r, 'x': x})
        true_params.extend([r, x])
        print(f"  Branch {from_bus}-{to_bus}: R={r:.4f}, X={x:.4f}")
    
    true_params = torch.tensor(true_params, dtype=torch.float32)
    
    # Edge index
    from_bus = np.array(net.line.from_bus.values)
    to_bus = np.array(net.line.to_bus.values)
    edge_index_np = np.stack([from_bus, to_bus])
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    num_buses = len(net.bus)
    pmu_buses = [3, 6, 9, 11, 14, 17, 19, 22, 24, 26, 29, 32]
    
    pbar = tqdm(range(num_episodes), desc=f"  {desc}", ncols=100)
    
    for i in pbar:
        net = nw.case33bw()
        p_load_base = net.load.p_mw.values.copy()
        q_load_base = net.load.q_mvar.values.copy()
        
        z_scada_list = []
        np.random.seed(start_seed + i)
        
        for t in range(steps_per_episode):
            net.load.p_mw = p_load_base
            net.load.q_mvar = q_load_base
            
            try:
                pp.runpp(net, algorithm='nr', numba=False)
            except:
                continue
            
            p_inj = -net.res_bus.p_mw.values
            q_inj = -net.res_bus.q_mvar.values
            v_scada = net.res_bus.vm_pu.values
            
            z_scada = np.concatenate([p_inj, q_inj, v_scada])
            z_scada_noisy = z_scada + np.random.normal(0, 0.02, len(z_scada))
            z_scada_list.append(z_scada_noisy)
        
        z_scada = np.array(z_scada_list)
        p_part = z_scada[:, :num_buses]
        q_part = z_scada[:, num_buses:2*num_buses]
        v_part = z_scada[:, 2*num_buses:]
        snapshot_features = np.stack([p_part, q_part, v_part], axis=2)
        
        episode_data = {
            'snapshots': torch.from_numpy(snapshot_features).float(),
            'edge_index': edge_index,
            'true_params': true_params,  # [R1, X1, R2, X2, R3, X3, R4, X4]
            'episode_id': i,
            'target_branches': TARGET_BRANCHES
        }
        dataset.append(episode_data)
    
    pbar.close()
    print(f"  Generated {num_episodes} episodes")
    return dataset


def main():
    print("\n" + "=" * 70)
    print("GENERATING TRAINING SET")
    print("=" * 70)
    train_data = generate_dataset(NUM_TRAIN_EPISODES, STEPS_PER_EPISODE, start_seed=42, desc="Training")
    
    print("\n" + "=" * 70)
    print("GENERATING VALIDATION SET")
    print("=" * 70)
    val_data = generate_dataset(NUM_VAL_EPISODES, STEPS_PER_EPISODE, start_seed=10000, desc="Validation")
    
    print("\n" + "=" * 70)
    print("GENERATING TEST SET")
    print("=" * 70)
    test_data = generate_dataset(NUM_TEST_EPISODES, STEPS_PER_EPISODE, start_seed=20000, desc="Test")
    
    # Save
    print("\n" + "=" * 70)
    print("SAVING DATASETS")
    print("=" * 70)
    
    with open(f'{DATA_DIR}/train_data.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    with open(f'{DATA_DIR}/val_data.pkl', 'wb') as f:
        pickle.dump(val_data, f)
    with open(f'{DATA_DIR}/test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    
    print(f"  Saved to {DATA_DIR}/")
    
    print(f"\n" + "=" * 70)
    print("DATA GENERATION COMPLETE!")
    print("=" * 70)
    print(f"\nOutput: 8 parameters (R, X for each of 4 branches)")
    print(f"Next: python experiments/multi_branch/train_multi_branch_mamba.py")


if __name__ == "__main__":
    main()
