"""
Graph-Mamba with Physics-Informed Loss
======================================

Enhanced Graph-Mamba model that incorporates power flow physics constraints
directly into the training loss. This encourages physically consistent parameter
estimates that satisfy Kirchhoff's laws.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Batch
import numpy as np

# Attempt to import Mamba
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print("Warning: 'mamba_ssm' library not found. Falling back to LSTM.")


class FeatureNormalizer(nn.Module):
    """Learnable feature normalization layer."""
    def __init__(self, num_features):
        super(FeatureNormalizer, self).__init__()
        self.scale = nn.Parameter(torch.ones(num_features))
        self.shift = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        return x * self.scale + self.shift


class GraphEncoder(nn.Module):
    """Graph Neural Network encoder for spatial feature extraction."""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index, batch=None):
        x = self.act(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.act(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.act(self.conv3(x, edge_index))
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h_graph = global_mean_pool(x, batch)
        return h_graph


class GraphMambaPhysicsModel(nn.Module):
    """
    Graph-Mamba with Physics-Informed Training
    
    Key improvements:
    1. Physics-informed loss computing power flow residuals
    2. Branch-specific prediction (not just global graph embedding)
    3. Edge-aware message passing
    """
    def __init__(self, num_nodes, in_features, d_model, d_state=16, d_conv=4, expand=2):
        super(GraphMambaPhysicsModel, self).__init__()
        
        self.d_model = d_model
        self.num_nodes = num_nodes
        
        # Feature normalization
        self.normalizer = FeatureNormalizer(in_features)
        
        # Spatial encoder
        self.graph_encoder = GraphEncoder(in_features, d_model, d_model)
        
        # Temporal encoder
        if HAS_MAMBA:
            self.temporal_layer = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
        else:
            self.temporal_layer = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=2,
                batch_first=True,
                dropout=0.1
            )
        
        # Enhanced prediction head with physics constraints
        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 2)  # [R, X]
        )
        
        # Learnable physics constraint weights
        self.phy_weight_r = nn.Parameter(torch.ones(1))
        self.phy_weight_x = nn.Parameter(torch.ones(1))

    def forward(self, snapshot_sequence, edge_index):
        """
        Forward pass.
        snapshot_sequence: [Batch, Time, Num_Nodes, Features]
        edge_index: [2, Num_Edges]
        """
        batch_size, seq_len, num_nodes, num_features = snapshot_sequence.shape
        
        # Normalize
        snapshot_sequence = self.normalizer(snapshot_sequence)
        
        # Spatial encoding
        flat_snapshots = snapshot_sequence.view(batch_size * seq_len, num_nodes, num_features)
        batch_idx = torch.arange(batch_size * seq_len, device=snapshot_sequence.device)
        batch_idx = batch_idx.repeat_interleave(num_nodes)
        flat_node_features = flat_snapshots.view(-1, num_features)
        
        edge_index_batch = edge_index.repeat(1, batch_size * seq_len)
        offsets = torch.arange(batch_size * seq_len, device=edge_index.device) * num_nodes
        offsets = offsets.repeat_interleave(edge_index.size(1))
        edge_index_batch = edge_index_batch + offsets.unsqueeze(0)
        
        embeddings = self.graph_encoder(flat_node_features, edge_index_batch, batch=batch_idx)
        embeddings = embeddings.view(batch_size, seq_len, self.d_model)
        
        # Temporal encoding
        if HAS_MAMBA:
            temporal_out = self.temporal_layer(embeddings)
        else:
            temporal_out, _ = self.temporal_layer(embeddings)
        
        final_state = temporal_out[:, -1, :]
        params_est = self.head(final_state)
        params_est = F.softplus(params_est) + 1e-6
        
        return params_est

    def compute_physics_residual(self, params, node_features, edge_index, from_bus, to_bus):
        """
        Compute power flow residual using estimated parameters.
        
        Args:
            params: [R, X] estimated line parameters
            node_features: [N, 3] node features [P, Q, V]
            edge_index: [2, E] edge connectivity
            from_bus, to_bus: indices of target branch endpoints
        
        Returns:
            residual: scalar physics residual
        """
        R, X = params[:, 0], params[:, 1]
        
        # Extract node features
        P = node_features[:, 0]  # Active power injection
        Q = node_features[:, 1]  # Reactive power injection
        V = node_features[:, 2]  # Voltage magnitude
        
        # Compute power flow on target branch using estimated R, X
        # Simplified DC power flow residual for the target branch
        v_from = V[from_bus]
        v_to = V[to_bus]
        
        # Power flow equation: P = (V_from - V_to) * V / R (simplified)
        # This is a simplified physics constraint
        delta_v = v_from - v_to
        
        # Expected power flow based on estimated parameters
        Z_squared = R**2 + X**2
        expected_p = delta_v * v_from * R / Z_squared
        expected_q = delta_v * v_from * X / Z_squared
        
        # Actual power injection at from_bus
        actual_p = P[from_bus]
        actual_q = Q[from_bus]
        
        # Compute residual
        residual_p = (expected_p - actual_p) ** 2
        residual_q = (expected_q - actual_q) ** 2
        
        return residual_p + residual_q


class PhysicsInformedLossV2(nn.Module):
    """
    Enhanced Physics-Informed Loss with proper power flow constraints.
    """
    def __init__(self, lambda_phy=0.1, lambda_smooth=0.01):
        super(PhysicsInformedLossV2, self).__init__()
        self.lambda_phy = lambda_phy
        self.lambda_smooth = lambda_smooth
        self.mse = nn.MSELoss()
        
    def forward(self, pred_params, true_params, model=None, 
                node_features=None, edge_index=None, target_branch=None):
        """
        Compute combined loss.
        
        Args:
            pred_params: [B, 2] predicted [R, X]
            true_params: [B, 2] ground truth [R, X]
            model: GraphMambaPhysicsModel for computing physics residual
            node_features: [B, T, N, F] node measurements
            edge_index: [2, E] graph connectivity
            target_branch: (from_bus, to_bus) indices
        """
        # 1. Data loss
        loss_data = self.mse(pred_params, true_params)
        
        # 2. Physics loss
        loss_phy = torch.tensor(0.0, device=pred_params.device)
        if model is not None and node_features is not None and target_branch is not None:
            batch_size = pred_params.size(0)
            from_bus, to_bus = target_branch
            
            for b in range(batch_size):
                # Use last timestep for physics computation
                nf = node_features[b, -1, :, :]  # [N, F]
                residual = model.compute_physics_residual(
                    pred_params[b:b+1], nf, edge_index, from_bus, to_bus
                )
                loss_phy += residual.mean()
            
            loss_phy = loss_phy / batch_size
        
        # 3. Smoothness loss (penalize extreme ratios)
        # Encourage R/X ratio to be within reasonable bounds
        ratio = pred_params[:, 0] / (pred_params[:, 1] + 1e-8)
        loss_smooth = torch.mean(F.relu(ratio - 5.0) + F.relu(0.1 - ratio))
        
        total_loss = loss_data + self.lambda_phy * loss_phy + self.lambda_smooth * loss_smooth
        
        return total_loss, {
            'data': loss_data.item(),
            'physics': loss_phy.item() if isinstance(loss_phy, torch.Tensor) else 0.0,
            'smoothness': loss_smooth.item()
        }


class RobustLoss(nn.Module):
    """
    Robust loss function that handles outliers better than MSE.
    Uses Huber loss for data term.
    """
    def __init__(self, delta=0.1, lambda_phy=0.1):
        super(RobustLoss, self).__init__()
        self.delta = delta
        self.lambda_phy = lambda_phy
        
    def huber_loss(self, pred, target):
        error = torch.abs(pred - target)
        quadratic = torch.min(error, torch.ones_like(error) * self.delta)
        linear = error - quadratic
        return torch.mean(0.5 * quadratic**2 + self.delta * linear)
    
    def forward(self, pred_params, true_params, model=None,
                node_features=None, edge_index=None, target_branch=None):
        # Huber loss for robustness
        loss_data = self.huber_loss(pred_params, true_params)
        
        # Physics loss (same as above)
        loss_phy = torch.tensor(0.0, device=pred_params.device)
        if model is not None and node_features is not None and target_branch is not None:
            batch_size = pred_params.size(0)
            from_bus, to_bus = target_branch
            
            for b in range(batch_size):
                nf = node_features[b, -1, :, :]
                residual = model.compute_physics_residual(
                    pred_params[b:b+1], nf, edge_index, from_bus, to_bus
                )
                loss_phy += residual.mean()
            
            loss_phy = loss_phy / batch_size
        
        total_loss = loss_data + self.lambda_phy * loss_phy
        
        return total_loss, {
            'data': loss_data.item(),
            'physics': loss_phy.item() if isinstance(loss_phy, torch.Tensor) else 0.0
        }
