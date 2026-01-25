import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Batch

# Attempt to import Mamba. If not available, we will fallback to a standard SSM/LSTM
# to ensure the code remains runnable for prototyping.
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print("Warning: 'mamba_ssm' library not found. Falling back to LSTM as a proxy for the State Space Model.")


class FeatureNormalizer(nn.Module):
    """Learnable feature normalization layer."""
    def __init__(self, num_features):
        super(FeatureNormalizer, self).__init__()
        self.scale = nn.Parameter(torch.ones(num_features))
        self.shift = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # x: [Batch, Time, Nodes, Features] or [Nodes, Features]
        return x * self.scale + self.shift

class GraphEncoder(nn.Module):
    """
    Encodes the grid snapshot at time t into a latent vector h_t.
    Uses a Graph Neural Network (GNN) to capture spatial topology.
    Includes edge features (R, X initial values).
    """
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=None):
        super(GraphEncoder, self).__init__()
        # GCN Layers (edge_attr not directly supported by GCNConv, using edge_weight only)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.act = nn.SiLU() # SiLU (Swish) is often used in modern architectures like Mamba
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        # x: [Num_Nodes, Features] (P, Q, V)
        # edge_index: [2, Num_Edges]
        # edge_weight: [Num_Edges] optional edge weights

        x = self.act(self.conv1(x, edge_index, edge_weight))
        x = self.dropout(x)
        x = self.act(self.conv2(x, edge_index, edge_weight))
        x = self.dropout(x)
        x = self.act(self.conv3(x, edge_index, edge_weight))

        # Aggregate node features into a single graph embedding vector
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h_graph = global_mean_pool(x, batch) # [Batch, out_channels]

        return h_graph

class GraphMambaModel(nn.Module):
    """
    The Core Architecture:
    1. Spatial: GraphEncoder processes each time step's grid snapshot.
    2. Temporal: Mamba (SSM) processes the sequence of graph embeddings.
    3. Head: Decodes the final state into R and X parameter estimates.

    Improvements:
    - Efficient batch processing using PyG batching
    - Feature normalization
    - Supports online inference (expanding window)
    """
    def __init__(self, num_nodes, in_features, d_model, d_state=16, d_conv=4, expand=2):
        super(GraphMambaModel, self).__init__()

        self.d_model = d_model
        self.num_nodes = num_nodes

        # 0. Feature Normalization
        self.normalizer = FeatureNormalizer(in_features)

        # 1. Spatial Encoder (GNN)
        self.graph_encoder = GraphEncoder(in_features, d_model, d_model)

        # 2. Temporal Encoder (Mamba / SSM)
        if HAS_MAMBA:
            self.temporal_layer = Mamba(
                d_model=d_model, # Model dimension
                d_state=d_state, # SSM state expansion factor
                d_conv=d_conv,   # Local convolution width
                expand=expand    # Block expansion factor
            )
        else:
            # Fallback: LSTM (A traditional Recurrent State Space Model)
            self.temporal_layer = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=2,
                batch_first=True,
                dropout=0.1
            )

        # 3. Prediction Head
        # Predicts R and X for the target branch
        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 2) # Outputs: [R_est, X_est]
        )

    def forward(self, snapshot_sequence, edge_index, mode='batch'):
        """
        snapshot_sequence: [Batch, Time, Num_Nodes, Features]
        edge_index: [2, Num_Edges] (Assumed static topology for now)
        mode: 'batch' for training (full sequence), 'online' for inference (expanding window)
        """
        batch_size, seq_len, num_nodes, num_features = snapshot_sequence.shape

        # Normalize features
        snapshot_sequence = self.normalizer(snapshot_sequence)

        # --- Spatial Pass (Efficient Batch Processing) ---
        # Reshape to [B*T, N, F] for parallel GNN processing
        flat_snapshots = snapshot_sequence.view(batch_size * seq_len, num_nodes, num_features)

        # Create batch indices for PyG batching
        # Each snapshot is a separate graph in the batch
        batch_idx = torch.arange(batch_size * seq_len, device=snapshot_sequence.device)
        batch_idx = batch_idx.repeat_interleave(num_nodes)

        # Flatten node features: [B*T*N, F]
        flat_node_features = flat_snapshots.view(-1, num_features)

        # Expand edge_index for all graphs in batch
        edge_index_batch = edge_index.repeat(1, batch_size * seq_len)
        # Offset node indices for each graph in batch
        offsets = torch.arange(batch_size * seq_len, device=edge_index.device) * num_nodes
        offsets = offsets.repeat_interleave(edge_index.size(1))
        edge_index_batch = edge_index_batch + offsets.unsqueeze(0)

        # Process all snapshots through GNN in parallel
        embeddings = self.graph_encoder(flat_node_features, edge_index_batch, batch=batch_idx)

        # Reshape back to [B, T, D]
        embeddings = embeddings.view(batch_size, seq_len, self.d_model)

        # --- Temporal Pass (Mamba) ---
        if HAS_MAMBA:
            temporal_out = self.temporal_layer(embeddings)
        else:
            temporal_out, _ = self.temporal_layer(embeddings)

        # Take the last time step's output for prediction
        final_state = temporal_out[:, -1, :] # [B, D]

        # --- Prediction ---
        params_est = self.head(final_state) # [B, 2]

        # Ensure positive parameters
        params_est = F.softplus(params_est) + 1e-6

        return params_est

    def forward_online(self, snapshot_sequence, edge_index):
        """
        Online inference mode: Returns predictions for each timestep.
        Useful for time-series tracking and fair comparison with IAUKF.

        snapshot_sequence: [1, Time, Num_Nodes, Features] (single episode)
        Returns: [Time, 2] - parameter predictions at each timestep
        """
        seq_len = snapshot_sequence.size(1)
        predictions = []

        for t in range(1, seq_len + 1):
            # Use expanding window: from start to current timestep
            window = snapshot_sequence[:, :t, :, :]
            pred = self.forward(window, edge_index, mode='online')
            predictions.append(pred[0])  # [2]

        return torch.stack(predictions)  # [Time, 2]

class PhysicsInformedLoss(nn.Module):
    """
    Hybrid Loss: MSE + Physics Residual

    The physics loss computes the power flow residual using predicted parameters.
    This encourages the model to learn physically consistent parameters.
    """
    def __init__(self, lambda_phy=0.1):
        super(PhysicsInformedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.lambda_phy = lambda_phy

    def forward(self, pred_params, true_params, physics_residual=None):
        # 1. Data Loss (Supervised)
        loss_mse = self.mse(pred_params, true_params)

        # 2. Physics Loss
        # If physics_residual is provided (computed externally), use it
        # Otherwise, set to 0 (pure supervised learning)
        if physics_residual is not None:
            loss_phy = torch.mean(physics_residual ** 2)
        else:
            loss_phy = 0.0

        total_loss = loss_mse + self.lambda_phy * loss_phy
        return total_loss, loss_mse, loss_phy if isinstance(loss_phy, torch.Tensor) else torch.tensor(0.0)