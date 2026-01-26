"""
Enhanced Graph Mamba Model with Advanced Techniques
- Residual connections
- Layer normalization
- Temporal attention
- Uncertainty estimation
- Stochastic depth
- Better regularization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Batch

# Attempt to import Mamba
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print("Warning: 'mamba_ssm' library not found. Falling back to LSTM.")


class DropPath(nn.Module):
    """
    Stochastic Depth (Drop Path) for regularization.
    Randomly drops entire residual branches during training.
    """
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class FeatureNormalizer(nn.Module):
    """Learnable feature normalization with instance norm."""
    def __init__(self, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=True)

    def forward(self, x):
        # x: [Batch, Time, Nodes, Features]
        if x.dim() == 4:
            b, t, n, f = x.shape
            x = x.view(b * t * n, f)
            x = self.norm(x)
            x = x.view(b, t, n, f)
        return x


class EnhancedGraphEncoder(nn.Module):
    """
    Enhanced GNN encoder with:
    - Residual connections
    - Layer normalization
    - Stochastic depth
    - Better activation functions
    """
    def __init__(self, in_channels, hidden_channels, out_channels, drop_path=0.1):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

        # Residual projections
        self.res_proj1 = nn.Linear(in_channels, hidden_channels)
        self.res_proj2 = nn.Linear(hidden_channels, hidden_channels)
        self.res_proj3 = nn.Linear(hidden_channels, out_channels)

        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_channels)
        self.ln2 = nn.LayerNorm(hidden_channels)
        self.ln3 = nn.LayerNorm(out_channels)

        # Stochastic depth
        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)
        self.drop_path3 = DropPath(drop_path)

        self.act = nn.SiLU()
        self.dropout = nn.Dropout(0.15)

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        # Layer 1 with residual
        identity = self.res_proj1(x)
        out = self.conv1(x, edge_index, edge_weight)
        out = self.drop_path1(out)
        x = self.ln1(out + identity)
        x = self.act(x)
        x = self.dropout(x)

        # Layer 2 with residual
        identity = self.res_proj2(x)
        out = self.conv2(x, edge_index, edge_weight)
        out = self.drop_path2(out)
        x = self.ln2(out + identity)
        x = self.act(x)
        x = self.dropout(x)

        # Layer 3 with residual
        identity = self.res_proj3(x)
        out = self.conv3(x, edge_index, edge_weight)
        out = self.drop_path3(out)
        x = self.ln3(out + identity)

        # Aggregate
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h_graph = global_mean_pool(x, batch)

        return h_graph


class TemporalAttention(nn.Module):
    """
    Multi-head attention for temporal features.
    Helps model focus on important timesteps.
    """
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [Batch, Time, D]
        attn_out, _ = self.attention(x, x, x)
        x = self.layer_norm(x + self.dropout(attn_out))
        return x


class ProbabilisticHead(nn.Module):
    """
    Probabilistic prediction head that outputs:
    - Mean: point estimate
    - Log variance: uncertainty estimate

    Useful for:
    - Uncertainty quantification
    - Out-of-distribution detection
    - Confidence-aware predictions
    """
    def __init__(self, d_model, dropout=0.2):
        super().__init__()

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        # Mean prediction
        self.fc_mean = nn.Linear(64, 2)  # R, X

        # Log variance prediction
        self.fc_logvar = nn.Linear(64, 2)  # R, X uncertainty

    def forward(self, x, return_uncertainty=False):
        features = self.backbone(x)
        mean = self.fc_mean(features)

        # Ensure positive parameters
        mean = F.softplus(mean) + 1e-6

        if return_uncertainty:
            logvar = self.fc_logvar(features)
            # Clamp for numerical stability
            logvar = torch.clamp(logvar, min=-10, max=2)
            std = torch.exp(0.5 * logvar)
            return mean, std

        return mean

    def sample(self, x, n_samples=10):
        """Monte Carlo sampling for uncertainty estimation"""
        mean, std = self.forward(x, return_uncertainty=True)
        samples = []
        for _ in range(n_samples):
            eps = torch.randn_like(std)
            sample = mean + eps * std
            samples.append(sample)
        return torch.stack(samples), mean, std


class EnhancedGraphMambaModel(nn.Module):
    """
    Enhanced Graph Mamba with advanced techniques:
    - Residual GNN encoder
    - Temporal attention
    - Probabilistic head
    - Better regularization
    - Stochastic depth
    """
    def __init__(
        self,
        num_nodes,
        in_features,
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2,
        use_attention=True,
        use_probabilistic=False,
        drop_path=0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.num_nodes = num_nodes
        self.use_attention = use_attention
        self.use_probabilistic = use_probabilistic

        # 0. Feature Normalization
        self.normalizer = FeatureNormalizer(in_features)

        # 1. Enhanced Spatial Encoder (GNN with residuals)
        self.graph_encoder = EnhancedGraphEncoder(
            in_features, d_model, d_model, drop_path=drop_path
        )

        # 2. Temporal Attention (optional)
        if use_attention:
            self.temporal_attention = TemporalAttention(d_model, num_heads=4)

        # 3. Temporal Encoder (Mamba / LSTM)
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
                dropout=0.15
            )

        # 4. Prediction Head
        if use_probabilistic:
            self.head = ProbabilisticHead(d_model, dropout=0.2)
        else:
            self.head = nn.Sequential(
                nn.Linear(d_model, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Dropout(0.15),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.SiLU(),
                nn.Dropout(0.15),
                nn.Linear(64, 2)
            )

        # Training noise (optional, for robustness)
        self.training_noise_std = 0.0

    def set_training_noise(self, std):
        """Add noise during training for robustness"""
        self.training_noise_std = std

    def forward(self, snapshot_sequence, edge_index, return_uncertainty=False):
        """
        snapshot_sequence: [Batch, Time, Num_Nodes, Features]
        edge_index: [2, Num_Edges]
        return_uncertainty: bool, return uncertainty estimates
        """
        batch_size, seq_len, num_nodes, num_features = snapshot_sequence.shape

        # Add training noise for robustness
        if self.training and self.training_noise_std > 0:
            noise = torch.randn_like(snapshot_sequence) * self.training_noise_std
            snapshot_sequence = snapshot_sequence + noise

        # Normalize features
        snapshot_sequence = self.normalizer(snapshot_sequence)

        # --- Spatial Pass (Efficient Batch Processing) ---
        flat_snapshots = snapshot_sequence.view(batch_size * seq_len, num_nodes, num_features)

        # Create batch indices
        batch_idx = torch.arange(batch_size * seq_len, device=snapshot_sequence.device)
        batch_idx = batch_idx.repeat_interleave(num_nodes)

        # Flatten node features
        flat_node_features = flat_snapshots.view(-1, num_features)

        # Expand edge_index
        edge_index_batch = edge_index.repeat(1, batch_size * seq_len)
        offsets = torch.arange(batch_size * seq_len, device=edge_index.device) * num_nodes
        offsets = offsets.repeat_interleave(edge_index.size(1))
        edge_index_batch = edge_index_batch + offsets.unsqueeze(0)

        # Process through GNN
        embeddings = self.graph_encoder(flat_node_features, edge_index_batch, batch=batch_idx)
        embeddings = embeddings.view(batch_size, seq_len, self.d_model)

        # --- Temporal Attention (optional) ---
        if self.use_attention:
            embeddings = self.temporal_attention(embeddings)

        # --- Temporal Pass (Mamba/LSTM) ---
        if HAS_MAMBA:
            temporal_out = self.temporal_layer(embeddings)
        else:
            temporal_out, _ = self.temporal_layer(embeddings)

        # Take last timestep
        final_state = temporal_out[:, -1, :]

        # --- Prediction ---
        if self.use_probabilistic and return_uncertainty:
            params_est, std = self.head(final_state, return_uncertainty=True)
            return params_est, std
        elif self.use_probabilistic:
            params_est = self.head(final_state, return_uncertainty=False)
        else:
            params_est = self.head(final_state)
            params_est = F.softplus(params_est) + 1e-6

        return params_est

    def forward_online(self, snapshot_sequence, edge_index):
        """Online inference with expanding window"""
        seq_len = snapshot_sequence.size(1)
        predictions = []

        for t in range(1, seq_len + 1):
            window = snapshot_sequence[:, :t, :, :]
            pred = self.forward(window, edge_index, return_uncertainty=False)
            predictions.append(pred[0])

        return torch.stack(predictions)


class RobustLoss(nn.Module):
    """
    Robust loss with multiple components:
    - MSE loss with optional label smoothing
    - Physics residual loss
    - Uncertainty-aware loss (if probabilistic head)
    """
    def __init__(self, lambda_phy=0.1, label_smoothing=0.0, use_uncertainty=False):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_phy = lambda_phy
        self.label_smoothing = label_smoothing
        self.use_uncertainty = use_uncertainty

    def forward(self, pred_params, true_params, physics_residual=None, uncertainty=None):
        # Apply label smoothing to targets
        if self.label_smoothing > 0 and self.training:
            noise = torch.randn_like(true_params) * self.label_smoothing * true_params
            smooth_target = true_params + noise
        else:
            smooth_target = true_params

        # Data loss
        if uncertainty is not None and self.use_uncertainty:
            # Negative log-likelihood loss (uncertainty-aware)
            loss_mse = torch.mean(
                0.5 * torch.log(2 * np.pi * uncertainty**2) +
                0.5 * ((pred_params - smooth_target) ** 2) / (uncertainty**2)
            )
        else:
            loss_mse = self.mse(pred_params, smooth_target)

        # Physics loss
        if physics_residual is not None:
            loss_phy = torch.mean(physics_residual ** 2)
        else:
            loss_phy = torch.tensor(0.0, device=pred_params.device)

        total_loss = loss_mse + self.lambda_phy * loss_phy

        return total_loss, loss_mse, loss_phy


# Legacy alias for backward compatibility
GraphMambaModel = EnhancedGraphMambaModel
PhysicsInformedLoss = RobustLoss
