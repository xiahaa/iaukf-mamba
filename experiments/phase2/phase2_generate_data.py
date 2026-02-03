"""
Phase 2 Step 1: Generate and save dataset
This script generates the dataset once and saves it to disk.
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

DATA_DIR = 'data/phase2'
os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 70)
print("PHASE 2 STEP 1: GENERATE DATASET")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  Train episodes: {NUM_TRAIN_EPISODES}")
print(f"  Val episodes: {NUM_VAL_EPISODES}")
print(f"  Test episodes: {NUM_TEST_EPISODES}")
print(f"  Steps per episode: {STEPS_PER_EPISODE}")
print(f"  Total episodes: {NUM_TRAIN_EPISODES + NUM_VAL_EPISODES + NUM_TEST_EPISODES}")
print(f"  Output directory: {DATA_DIR}")


def generate_dataset_phase2(num_episodes, steps_per_episode, start_seed=42, desc="Dataset"):
    """
    Generate dataset with constant loads (matching Phase 1 IAUKF).
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

        # Get base loads (constant)
        p_load_base = sim.net.load.p_mw.values.copy()
        q_load_base = sim.net.load.q_mvar.values.copy()

        # Generate time series
        z_scada_list = []
        z_pmu_list = []

        np.random.seed(start_seed + i)

        for t in range(steps_per_episode):
            # CONSTANT LOADS (key for Phase 2)
            sim.net.load.p_mw = p_load_base
            sim.net.load.q_mvar = q_load_base

            # Power flow
            try:
                pp.runpp(sim.net, algorithm='nr', numba=False)
            except:
                pbar.write(f"    Warning: Power flow failed for episode {i}, step {t}")
                continue

            # SCADA measurements (with noise, std=0.02)
            p_inj = -sim.net.res_bus.p_mw.values
            q_inj = -sim.net.res_bus.q_mvar.values
            v_scada = sim.net.res_bus.vm_pu.values

            z_scada = np.concatenate([p_inj, q_inj, v_scada])
            z_scada_noisy = z_scada + np.random.normal(0, 0.02, len(z_scada))
            z_scada_list.append(z_scada_noisy)

            # PMU measurements (with noise)
            v_pmu = sim.net.res_bus.vm_pu.values[sim.pmu_buses]
            theta_pmu = np.radians(sim.net.res_bus.va_degree.values[sim.pmu_buses])
            z_pmu = np.concatenate([v_pmu, theta_pmu])
            noise_pmu = np.concatenate([
                np.random.normal(0, 0.005, len(v_pmu)),
                np.random.normal(0, 0.002, len(theta_pmu))
            ])
            z_pmu_list.append(z_pmu + noise_pmu)

        # Process measurements
        z_scada = np.array(z_scada_list)  # [T, 99]
        z_pmu = np.array(z_pmu_list)      # [T, 24]

        # Reshape SCADA to [T, N, 3] (P, Q, V per node)
        p_part = z_scada[:, :num_buses]
        q_part = z_scada[:, num_buses:2*num_buses]
        v_part = z_scada[:, 2*num_buses:]
        snapshot_features = np.stack([p_part, q_part, v_part], axis=2)

        # Convert numpy arrays to tensors efficiently
        # Store episode
        episode_data = {
            'snapshots': torch.from_numpy(snapshot_features).float(),
            'edge_index': edge_index,
            'true_params': torch.tensor([sim.r_true, sim.x_true], dtype=torch.float32),
            'z_pmu': torch.from_numpy(z_pmu).float(),
            'episode_id': i,
            'seed': start_seed + i
        }
        dataset.append(episode_data)

        # Update progress bar with current stats
        if (i + 1) % 100 == 0:
            pbar.set_postfix({
                'R_mean': f"{np.mean([d['true_params'][0].item() for d in dataset[-100:]]):.4f}",
                'X_mean': f"{np.mean([d['true_params'][1].item() for d in dataset[-100:]]):.4f}"
            })

    pbar.close()
    print(f"  ✓ Generated {num_episodes} episodes")

    return dataset


def main():
    # Generate datasets
    print("\n" + "=" * 70)
    print("GENERATING TRAINING SET")
    print("=" * 70)
    train_data = generate_dataset_phase2(
        NUM_TRAIN_EPISODES,
        STEPS_PER_EPISODE,
        start_seed=42,
        desc="Training"
    )

    print("\n" + "=" * 70)
    print("GENERATING VALIDATION SET")
    print("=" * 70)
    val_data = generate_dataset_phase2(
        NUM_VAL_EPISODES,
        STEPS_PER_EPISODE,
        start_seed=10000,
        desc="Validation"
    )

    print("\n" + "=" * 70)
    print("GENERATING TEST SET")
    print("=" * 70)
    test_data = generate_dataset_phase2(
        NUM_TEST_EPISODES,
        STEPS_PER_EPISODE,
        start_seed=20000,
        desc="Test"
    )

    # Save datasets
    print("\n" + "=" * 70)
    print("SAVING DATASETS")
    print("=" * 70)

    print("\nSaving to disk...")

    with open(os.path.join(DATA_DIR, 'train_data.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    print(f"  ✓ Saved: {DATA_DIR}/train_data.pkl ({len(train_data)} episodes)")

    with open(os.path.join(DATA_DIR, 'val_data.pkl'), 'wb') as f:
        pickle.dump(val_data, f)
    print(f"  ✓ Saved: {DATA_DIR}/val_data.pkl ({len(val_data)} episodes)")

    with open(os.path.join(DATA_DIR, 'test_data.pkl'), 'wb') as f:
        pickle.dump(test_data, f)
    print(f"  ✓ Saved: {DATA_DIR}/test_data.pkl ({len(test_data)} episodes)")

    # Calculate statistics
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)

    all_data = train_data + val_data + test_data

    r_values = [d['true_params'][0].item() for d in all_data]
    x_values = [d['true_params'][1].item() for d in all_data]

    print(f"\nParameter distribution:")
    print(f"  R: mean={np.mean(r_values):.4f}, std={np.std(r_values):.4f}, "
          f"min={np.min(r_values):.4f}, max={np.max(r_values):.4f}")
    print(f"  X: mean={np.mean(x_values):.4f}, std={np.std(x_values):.4f}, "
          f"min={np.min(x_values):.4f}, max={np.max(x_values):.4f}")

    # Size on disk
    total_size = 0
    for fname in ['train_data.pkl', 'val_data.pkl', 'test_data.pkl']:
        fpath = os.path.join(DATA_DIR, fname)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        total_size += size_mb
        print(f"\n  {fname}: {size_mb:.1f} MB")

    print(f"\n  Total: {total_size:.1f} MB")

    print("\n" + "=" * 70)
    print("✓ DATA GENERATION COMPLETE!")
    print("=" * 70)
    print(f"\nNext step: Run training")
    print(f"  python experiments/phase2_train_mamba.py")


if __name__ == "__main__":
    main()
