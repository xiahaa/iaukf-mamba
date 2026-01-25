import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Batch
from simulation import PowerSystemSimulation
from graph_mamba import GraphMambaModel, PhysicsInformedLoss
import pandapower as pp
import os
from torch.utils.data import Dataset, DataLoader

# --- Configuration ---
NUM_TRAIN_EPISODES = 50      # Number of simulation runs for training
NUM_VAL_EPISODES = 10        # Number of validation episodes
STEPS_PER_EPISODE = 200      # Time steps per episode
BATCH_SIZE = 8               # Number of episodes per training batch
EPOCHS = 30                  # Training epochs
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = 'graph_mamba_checkpoint.pt'

print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Available GPUs: {torch.cuda.device_count()}")


class PowerGridDataset(Dataset):
    """PyTorch Dataset for power grid episodes."""
    def __init__(self, episodes):
        self.episodes = episodes

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        return self.episodes[idx]


def generate_dataset(num_episodes, steps_per_episode, desc="Dataset"):
    """
    Generates a dataset using the simulation environment.
    Returns a list of episodes, where each episode contains:
    - snapshots: [Time, Num_Nodes, Features]
    - edge_index: [2, Num_Edges]
    - true_params: [R_true, X_true]
    """
    print(f"Generating {desc} ({num_episodes} episodes)...")
    dataset = []

    # We need to run the simulation once to get static topology data
    dummy_sim = PowerSystemSimulation(steps=1)
    net = dummy_sim.net

    # Extract Edge Index (Static Topology)
    # Pandapower 'line' DataFrame has from_bus and to_bus columns
    edge_index = torch.tensor([
        net.line.from_bus.values,
        net.line.to_bus.values
    ], dtype=torch.long)

    # Make undirected graph (add reverse edges) for GNN
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    for i in range(num_episodes):
        # Use different random seeds for diversity
        sim = PowerSystemSimulation(steps=steps_per_episode)
        data = sim.run_simulation(seed=42 + i)

        # Extract Features X: [P_inj, Q_inj, V_mag]
        # In simulation.py, we generated z_scada which has [P(33), Q(33), V(33)] flattened
        # We need to reshape this back to [Time, Nodes, Features]

        # Note: z_scada in simulation.py is noisy. Perfect for training inputs.
        # Structure of z_scada in simulation.py is: np.concatenate([p_inj, q_inj, v_scada])
        # Total length = 33 * 3 = 99

        z_scada = data['z_scada'] # [Time, 99]
        num_buses = len(net.bus)

        # Reshape to [Time, Nodes, 3]
        # p_inj is first 33, q_inj is next 33, v_scada is last 33
        p_part = z_scada[:, :num_buses]
        q_part = z_scada[:, num_buses:2*num_buses]
        v_part = z_scada[:, 2*num_buses:]

        # Stack features: [Time, Nodes, 3]
        snapshot_features = np.stack([p_part, q_part, v_part], axis=2)

        episode_data = {
            'snapshots': torch.tensor(snapshot_features, dtype=torch.float32),
            'edge_index': edge_index, # Shared topology
            'true_params': torch.tensor([data['r_true'], data['x_true']], dtype=torch.float32)
        }
        dataset.append(episode_data)

        if (i+1) % 10 == 0:
            print(f"  Generated {i+1}/{num_episodes} episodes")

    return dataset


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    snapshots = torch.stack([item['snapshots'] for item in batch])
    true_params = torch.stack([item['true_params'] for item in batch])
    edge_index = batch[0]['edge_index']  # Same for all
    return {
        'snapshots': snapshots,
        'edge_index': edge_index,
        'true_params': true_params
    }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_phy = 0.0
    num_batches = 0

    for batch in dataloader:
        snapshots = batch['snapshots'].to(device)  # [B, T, N, F]
        edge_index = batch['edge_index'].to(device)
        targets = batch['true_params'].to(device)  # [B, 2]

        # Forward pass
        optimizer.zero_grad()
        preds = model(snapshots, edge_index)  # [B, 2]

        # Loss calculation
        loss, loss_mse, loss_phy = criterion(preds, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_phy += loss_phy.item() if isinstance(loss_phy, torch.Tensor) else 0.0
        num_batches += 1

    return total_loss / num_batches, total_mse / num_batches, total_phy / num_batches


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            snapshots = batch['snapshots'].to(device)
            edge_index = batch['edge_index'].to(device)
            targets = batch['true_params'].to(device)

            preds = model(snapshots, edge_index)
            loss, loss_mse, _ = criterion(preds, targets)

            total_loss += loss.item()
            total_mse += loss_mse.item()
            num_batches += 1

    return total_loss / num_batches, total_mse / num_batches


def train():
    """Main training function."""
    # 1. Prepare Data
    print("=" * 60)
    print("GRAPH MAMBA TRAINING PIPELINE")
    print("=" * 60)

    train_dataset = generate_dataset(NUM_TRAIN_EPISODES, STEPS_PER_EPISODE, desc="Training Set")
    val_dataset = generate_dataset(NUM_VAL_EPISODES, STEPS_PER_EPISODE, desc="Validation Set")

    # Create DataLoaders
    train_loader = DataLoader(
        PowerGridDataset(train_dataset),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )

    val_loader = DataLoader(
        PowerGridDataset(val_dataset),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Get dimensions from first episode
    sample_snapshot = train_dataset[0]['snapshots'] # [T, N, F]
    num_nodes = sample_snapshot.shape[1]
    in_features = sample_snapshot.shape[2]

    print(f"\nDataset Info:")
    print(f"  - Num nodes: {num_nodes}")
    print(f"  - Input features: {in_features}")
    print(f"  - Train episodes: {NUM_TRAIN_EPISODES}")
    print(f"  - Val episodes: {NUM_VAL_EPISODES}")
    print(f"  - Steps per episode: {STEPS_PER_EPISODE}")

    # 2. Initialize Model
    model = GraphMambaModel(
        num_nodes=num_nodes,
        in_features=in_features,
        d_model=64  # Increased hidden dimension for better capacity
    ).to(DEVICE)

    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    criterion = PhysicsInformedLoss(lambda_phy=0.0) # Start with pure supervised learning

    # 3. Training Loop
    print("\n" + "=" * 60)
    print("TRAINING START")
    print("=" * 60)

    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # Train
        train_loss, train_mse, train_phy = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )
        train_loss_history.append(train_loss)

        # Validate
        val_loss, val_mse = validate(model, val_loader, criterion, DEVICE)
        val_loss_history.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Print progress
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.6f} (MSE: {train_mse:.6f}) | "
              f"Val Loss: {val_loss:.6f} (MSE: {val_mse:.6f})")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'num_nodes': num_nodes,
                'in_features': in_features,
            }, CHECKPOINT_PATH)
            print(f"  → Checkpoint saved (Val Loss: {val_loss:.6f})")

    # 4. Final Evaluation
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    # Load best model
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Best model from epoch {checkpoint['epoch']+1} loaded.")
    print(f"Best validation loss: {checkpoint['val_loss']:.6f}")

    # Test on a few validation samples
    model.eval()
    print("\nSample Predictions on Validation Set:")
    print("-" * 60)
    with torch.no_grad():
        for i in range(min(5, len(val_dataset))):
            test_ep = val_dataset[i]
            snapshots = test_ep['snapshots'].unsqueeze(0).to(DEVICE) # [1, T, N, F]
            edge_index = test_ep['edge_index'].to(DEVICE)
            target = test_ep['true_params'].to(DEVICE) # [2]

            pred = model(snapshots, edge_index)[0] # [2]

            r_true, x_true = target[0].item(), target[1].item()
            r_pred, x_pred = pred[0].item(), pred[1].item()
            r_err = abs(r_pred - r_true) / r_true * 100
            x_err = abs(x_pred - x_true) / x_true * 100

            print(f"Sample {i+1}:")
            print(f"  R: True={r_true:.5f}, Pred={r_pred:.5f}, Error={r_err:.2f}%")
            print(f"  X: True={x_true:.5f}, Pred={x_pred:.5f}, Error={x_err:.2f}%")

    # 5. Plot Loss Curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Training Loss', linewidth=2)
    plt.plot(val_loss_history, label='Validation Loss', linewidth=2)
    plt.axhline(y=best_val_loss, color='r', linestyle='--', alpha=0.5, label=f'Best Val Loss: {best_val_loss:.6f}')
    plt.title("Graph Mamba Training Progress", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_loss.png", dpi=150)
    print("\n✓ Loss curve saved as 'training_loss.png'")
    plt.close()

    print("\n" + "=" * 60)
    print(f"✓ Training checkpoint saved as '{CHECKPOINT_PATH}'")
    print("=" * 60)


if __name__ == "__main__":
    train()
