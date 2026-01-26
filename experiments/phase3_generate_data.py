"""
Phase 3 Data Generation: Time-Varying Parameters
==================================================

Based on Phase 2 generation, but with time-varying line parameters.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pickle
from tqdm import tqdm
import pandapower as pp

from model.simulation import PowerSystemSimulation

# Configuration
NUM_TRAIN_EPISODES = 800
NUM_VAL_EPISODES = 100
NUM_TEST_EPISODES = 100
STEPS_PER_EPISODE = 200
CHANGE_INTERVAL = 50  # Parameters change every 50 timesteps
PARAM_VARIATION = 0.08  # ±8% variation

DATA_DIR = 'data/phase3'
os.makedirs(DATA_DIR, exist_ok=True)

SEED = 42

print("=" * 70)
print("PHASE 3: GENERATE DATA WITH TIME-VARYING PARAMETERS")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  Train episodes: {NUM_TRAIN_EPISODES}")
print(f"  Val episodes: {NUM_VAL_EPISODES}")
print(f"  Test episodes: {NUM_TEST_EPISODES}")
print(f"  Steps per episode: {STEPS_PER_EPISODE}")
print(f"  Parameter change interval: {CHANGE_INTERVAL}")
print(f"  Parameter variation: ±{PARAM_VARIATION*100:.0f}%")
print(f"  Output directory: {DATA_DIR}")


def generate_timevarying_parameters(r_base, x_base, steps, change_interval, variation, seed=None):
    """Generate time-varying R and X profiles"""
    if seed is not None:
        np.random.seed(seed)

    r_profile = np.zeros(steps)
    x_profile = np.zeros(steps)

    # Start with base values
    r_current = r_base
    x_current = x_base

    for t in range(steps):
        # Change parameters at intervals
        if t > 0 and t % change_interval == 0:
            # Random change within ±variation
            r_change = np.random.uniform(-variation, variation)
            x_change = np.random.uniform(-variation, variation)

            r_current = r_base * (1 + r_change)
            x_current = x_base * (1 + x_change)

            # Ensure positive
            r_current = max(r_current, 1e-6)
            x_current = max(x_current, 1e-6)

        r_profile[t] = r_current
        x_profile[t] = x_current

    return r_profile, x_profile


def generate_dataset_phase3(num_episodes, steps_per_episode, start_seed=42, desc="Dataset"):
    """
    Generate dataset with TIME-VARYING parameters.
    """
    print(f"\n[Generating {desc}]")

    dataset = []

    # Get static topology (only once)
    dummy_sim = PowerSystemSimulation(steps=1)
    net = dummy_sim.net

    # Edge index (undirected graph)
    from_bus = np.array(net.line.from_bus.values)
    to_bus = np.array(net.line.to_bus.values)
    edge_index_np = np.stack([from_bus, to_bus])
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    num_buses = len(net.bus)

    # Progress bar
    pbar = tqdm(range(num_episodes), desc=f"  {desc}", ncols=100)

    for i in pbar:
        # Create simulation
        sim = PowerSystemSimulation(steps=steps_per_episode)

        # Get base line parameters
        line_idx = sim.line_idx
        r_base = sim.net.line.at[line_idx, 'r_ohm_per_km']
        x_base = sim.net.line.at[line_idx, 'x_ohm_per_km']
        length = sim.net.line.at[line_idx, 'length_km']

        # Generate time-varying parameter profiles
        r_profile, x_profile = generate_timevarying_parameters(
            r_base, x_base, steps_per_episode,
            CHANGE_INTERVAL, PARAM_VARIATION, seed=start_seed+i
        )

        # Get base loads (constant within episode)
        p_load_base = sim.net.load.p_mw.values.copy()
        q_load_base = sim.net.load.q_mvar.values.copy()

        # Generate time series
        snapshot_features = []

        np.random.seed(start_seed + i)

        for t in range(steps_per_episode):
            # CONSTANT LOADS (same as Phase 2)
            sim.net.load.p_mw = p_load_base
            sim.net.load.q_mvar = q_load_base

            # UPDATE LINE PARAMETERS (different from Phase 2!)
            sim.net.line.at[line_idx, 'r_ohm_per_km'] = r_profile[t]
            sim.net.line.at[line_idx, 'x_ohm_per_km'] = x_profile[t]

            # Run power flow
            try:
                pp.runpp(sim.net, algorithm='nr', max_iteration=100, tolerance_mva=1e-6)

                # Extract measurements
                # P, Q, V at each bus
                p_meas = sim.net.res_bus.p_mw.values
                q_meas = sim.net.res_bus.q_mvar.values
                v_meas = sim.net.res_bus.vm_pu.values

                # Add noise
                p_meas += np.random.randn(num_buses) * 0.02 * np.abs(p_meas)
                q_meas += np.random.randn(num_buses) * 0.02 * np.abs(q_meas)
                v_meas += np.random.randn(num_buses) * 0.02 * v_meas

                # Stack features: [num_buses, 3]
                features = np.stack([p_meas, q_meas, v_meas], axis=-1)
                snapshot_features.append(features)

            except Exception as e:
                # If power flow fails, use previous data or skip
                if len(snapshot_features) > 0:
                    snapshot_features.append(snapshot_features[-1])
                else:
                    # Skip this episode if first step fails
                    break

        # Only add episode if we got all timesteps
        if len(snapshot_features) == steps_per_episode:
            snapshot_features = np.array(snapshot_features)  # [T, N, 3]

            # Convert to tensors
            episode_data = {
                'snapshots': torch.from_numpy(snapshot_features).float(),
                'edge_index': edge_index,
                'r_profile': torch.from_numpy(r_profile * length).float(),  # Total ohms
                'x_profile': torch.from_numpy(x_profile * length).float(),  # Total ohms
                'r_base': float(r_base * length),
                'x_base': float(x_base * length),
            }

            dataset.append(episode_data)

    return dataset


# Generate datasets
print("\n" + "=" * 70)
print("GENERATING DATASETS")
print("=" * 70)

train_data = generate_dataset_phase3(
    NUM_TRAIN_EPISODES,
    STEPS_PER_EPISODE,
    start_seed=SEED,
    desc="Train"
)

val_data = generate_dataset_phase3(
    NUM_VAL_EPISODES,
    STEPS_PER_EPISODE,
    start_seed=SEED + NUM_TRAIN_EPISODES,
    desc="Val"
)

test_data = generate_dataset_phase3(
    NUM_TEST_EPISODES,
    STEPS_PER_EPISODE,
    start_seed=SEED + NUM_TRAIN_EPISODES + NUM_VAL_EPISODES,
    desc="Test"
)

# Save datasets
print("\n" + "=" * 70)
print("SAVING DATASETS")
print("=" * 70)

with open(os.path.join(DATA_DIR, 'train_data.pkl'), 'wb') as f:
    pickle.dump(train_data, f)
print(f"  ✓ Saved training data: {len(train_data)} episodes")

with open(os.path.join(DATA_DIR, 'val_data.pkl'), 'wb') as f:
    pickle.dump(val_data, f)
print(f"  ✓ Saved validation data: {len(val_data)} episodes")

with open(os.path.join(DATA_DIR, 'test_data.pkl'), 'wb') as f:
    pickle.dump(test_data, f)
print(f"  ✓ Saved test data: {len(test_data)} episodes")

# Save config
config = {
    'steps_per_episode': STEPS_PER_EPISODE,
    'change_interval': CHANGE_INTERVAL,
    'param_variation': PARAM_VARIATION,
}

with open(os.path.join(DATA_DIR, 'config.pkl'), 'wb') as f:
    pickle.dump(config, f)
print(f"  ✓ Saved configuration")

# Data analysis
print("\n" + "=" * 70)
print("DATA ANALYSIS")
print("=" * 70)

sample = train_data[0]
r_profile = sample['r_profile'].numpy()
x_profile = sample['x_profile'].numpy()

print(f"\nSample parameter profile (first episode):")
print(f"  R base: {sample['r_base']:.4f} Ω")
print(f"  R range: [{r_profile.min():.4f}, {r_profile.max():.4f}] Ω")
print(f"  R variation: {(r_profile.max() - r_profile.min()) / sample['r_base'] * 100:.1f}%")
print(f"  X base: {sample['x_base']:.4f} Ω")
print(f"  X range: [{x_profile.min():.4f}, {x_profile.max():.4f}] Ω")
print(f"  X variation: {(x_profile.max() - x_profile.min()) / sample['x_base'] * 100:.1f}%")

# Count parameter changes
num_changes = 0
for t in range(1, len(r_profile)):
    if r_profile[t] != r_profile[t-1]:
        num_changes += 1

print(f"\n  Number of parameter changes: {num_changes}")
print(f"  Change points: {[t for t in range(1, len(r_profile)) if r_profile[t] != r_profile[t-1]]}")

print(f"\nData shapes:")
print(f"  Snapshots: {sample['snapshots'].shape} [Time, Nodes, Features]")
print(f"  Edge index: {sample['edge_index'].shape}")
print(f"  R profile: {sample['r_profile'].shape}")

print("\n" + "=" * 70)
print("✓ PHASE 3 DATA GENERATION COMPLETE!")
print("=" * 70)

print(f"\nKey Differences from Phase 2:")
print(f"  ✓ Parameters vary over time (every {CHANGE_INTERVAL} steps)")
print(f"  ✓ ±{PARAM_VARIATION*100:.0f}% variation (realistic changes)")
print(f"  ✓ Tests adaptability and robustness")

print(f"\nNext Steps:")
print(f"  1. Test IAUKF: python experiments/phase3_test_iaukf.py")
print(f"  2. Train Mamba: python experiments/phase3_train_mamba.py")

print(f"\n✓ Data ready at: {DATA_DIR}/")
