import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Batch
from simulation import PowerSystemSimulation
from graph_mamba import GraphMambaModel, PhysicsInformedLoss
import pandapower as pp

# --- Configuration ---
NUM_EPISODES = 50        # Number of simulation runs (episodes)
STEPS_PER_EPISODE = 200  # Time steps per episode
BATCH_SIZE = 10          # Number of episodes per training batch
EPOCHS = 20              # Training epochs
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_dataset(num_episodes, steps_per_episode):
    """
    Generates a dataset using the simulation environment.
    Returns a list of episodes, where each episode contains:
    - snapshots: [Time, Num_Nodes, Features]
    - edge_index: [2, Num_Edges]
    - true_params: [R_true, X_true]
    """
    print(f"Generating dataset ({num_episodes} episodes)...")
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
        sim = PowerSystemSimulation(steps=steps_per_episode)
        data = sim.run_simulation()

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
            print(f"Generated {i+1}/{num_episodes} episodes")

    return dataset

def train():
    # 1. Prepare Data
    dataset = generate_dataset(NUM_EPISODES, STEPS_PER_EPISODE)

    # Get dimensions from first episode
    sample_snapshot = dataset[0]['snapshots'] # [T, N, F]
    num_nodes = sample_snapshot.shape[1]
    in_features = sample_snapshot.shape[2]

    # 2. Initialize Model
    model = GraphMambaModel(
        num_nodes=num_nodes,
        in_features=in_features,
        d_model=32 # Hidden dimension
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = PhysicsInformedLoss(lambda_phy=0.0) # Start with pure supervised learning

    # 3. Training Loop
    print("Starting Training...")
    loss_history = []

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        num_batches = 0

        # Simple Batching (Shuffling could be added)
        for i in range(0, len(dataset), BATCH_SIZE):
            batch_episodes = dataset[i : i+BATCH_SIZE]

            # Prepare Batch Tensors
            # snapshots: [Batch, Time, Nodes, Features]
            batch_snapshots = torch.stack([ep['snapshots'] for ep in batch_episodes]).to(DEVICE)

            # true_params: [Batch, 2]
            batch_targets = torch.stack([ep['true_params'] for ep in batch_episodes]).to(DEVICE)

            # edge_index is static, just take from first episode
            edge_index = batch_episodes[0]['edge_index'].to(DEVICE)

            # Forward Pass
            optimizer.zero_grad()
            preds = model(batch_snapshots, edge_index) # [Batch, 2]

            # Loss Calculation
            loss = criterion(preds, batch_targets)

            # Backward Pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}")

    # 4. Evaluation / Visualization
    print("Training Complete. Evaluating on last episode...")
    model.eval()
    with torch.no_grad():
        test_ep = dataset[-1]
        snapshots = test_ep['snapshots'].unsqueeze(0).to(DEVICE) # [1, T, N, F]
        edge_index = test_ep['edge_index'].to(DEVICE)
        target = test_ep['true_params'].to(DEVICE) # [2]

        pred = model(snapshots, edge_index)[0] # [2]

        print(f"True R: {target[0]:.4f}, Est R: {pred[0]:.4f}")
        print(f"True X: {target[1]:.4f}, Est X: {pred[1]:.4f}")

    # Plot Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, marker='o')
    plt.title("Graph Mamba Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.savefig("training_loss.png")
    plt.show()

if __name__ == "__main__":
    train()
