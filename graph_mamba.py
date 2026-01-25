import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

# Attempt to import Mamba. If not available, we will fallback to a standard SSM/LSTM
# to ensure the code remains runnable for prototyping.
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print("Warning: 'mamba_ssm' library not found. Falling back to LSTM as a proxy for the State Space Model.")

class GraphEncoder(nn.Module):
    """
    Encodes the grid snapshot at time t into a latent vector h_t.
    Uses a Graph Neural Network (GNN) to capture spatial topology.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphEncoder, self).__init__()
        # GCN Layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.act = nn.SiLU() # SiLU (Swish) is often used in modern architectures like Mamba

    def forward(self, x, edge_index, edge_weight=None):
        # x: [Num_Nodes, Features] (P, Q, V)
        # edge_index: [2, Num_Edges]

        x = self.act(self.conv1(x, edge_index, edge_weight))
        x = self.act(self.conv2(x, edge_index, edge_weight))
        x = self.act(self.conv3(x, edge_index, edge_weight))

        # Aggregate node features into a single graph embedding vector
        # batch vector is zeros because we process one snapshot at a time or handle batching externally
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h_graph = global_mean_pool(x, batch) # [1, out_channels]

        return h_graph

class GraphMambaModel(nn.Module):
    """
    The Core Architecture:
    1. Spatial: GraphEncoder processes each time step's grid snapshot.
    2. Temporal: Mamba (SSM) processes the sequence of graph embeddings.
    3. Head: Decodes the final state into R and X parameter estimates.
    """
    def __init__(self, num_nodes, in_features, d_model, d_state=16, d_conv=4, expand=2):
        super(GraphMambaModel, self).__init__()

        self.d_model = d_model

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
                batch_first=True
            )

        # 3. Prediction Head
        # Predicts R and X for the target branch
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.SiLU(),
            nn.Linear(64, 2) # Outputs: [R_est, X_est]
        )

    def forward(self, snapshot_sequence, edge_index):
        """
        snapshot_sequence: [Batch, Time, Num_Nodes, Features]
        edge_index: [2, Num_Edges] (Assumed static topology for now)
        """
        batch_size, seq_len, num_nodes, num_features = snapshot_sequence.shape

        # --- Spatial Pass ---
        # Process each time step through the GNN
        # Flatten batch and time to process all snapshots in parallel
        flat_snapshots = snapshot_sequence.view(-1, num_nodes, num_features) # [B*T, N, F]

        # Run GNN for each snapshot
        # Note: For prototype simplicity we iterate. In production, use PyG Batching.
        embeddings = []
        for i in range(flat_snapshots.size(0)):
            x_t = flat_snapshots[i]
            # Assumes standard edge_index on same device
            h_t = self.graph_encoder(x_t, edge_index) # [1, d_model]
            embeddings.append(h_t)

        embeddings = torch.cat(embeddings, dim=0) # [B*T, d_model]
        embeddings = embeddings.view(batch_size, seq_len, self.d_model) # [B, T, D]

        # --- Temporal Pass (Mamba) ---
        if HAS_MAMBA:
            temporal_out = self.temporal_layer(embeddings)
        else:
            temporal_out, _ = self.temporal_layer(embeddings)

        # Take the last time step's output for prediction
        final_state = temporal_out[:, -1, :] # [B, D]

        # --- Prediction ---
        params_est = self.head(final_state) # [B, 2]

        return params_est

class PhysicsInformedLoss(nn.Module):
    """
    Hybrid Loss: MSE + Physics Residual
    """
    def __init__(self, lambda_phy=0.1):
        super(PhysicsInformedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.lambda_phy = lambda_phy

    def forward(self, pred_params, true_params):
        # 1. Data Loss (Supervised)
        loss_mse = self.mse(pred_params, true_params)

        # 2. Physics Loss (Placeholder)
        # In a full implementation, we would calculate Power Flow here
        # using pred_params and penalize mismatch.
        loss_phy = 0.0

        total_loss = loss_mse + self.lambda_phy * loss_phy
        return total_loss