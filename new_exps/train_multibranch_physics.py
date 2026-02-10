"""
Multi-Branch Physics-Informed Training
======================================

Train Graph-Mamba on multiple branches simultaneously to improve generalization.
Uses branch-specific prediction heads with shared spatial/temporal encoders.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
import json

from graphmamba.graph_mamba_physics import GraphEncoder, FeatureNormalizer

# Attempt to import Mamba
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print("Warning: 'mamba_ssm' library not found. Falling back to LSTM.")


class MultiBranchGraphMamba(nn.Module):
    """
    Multi-Branch Graph-Mamba with branch-specific heads.
    
    Architecture:
    - Shared Graph Encoder (spatial)
    - Shared Temporal Encoder (Mamba/LSTM)
    - Branch-specific prediction heads
    """
    
    def __init__(self, num_nodes, in_features, d_model, branches, 
                 d_state=16, d_conv=4, expand=2):
        """
        Args:
            branches: dict of {branch_idx: (from_bus, to_bus)}
        """
        super(MultiBranchGraphMamba, self).__init__()
        
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.branches = branches
        
        # Shared encoders
        self.normalizer = FeatureNormalizer(in_features)
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
        
        # Branch-specific heads
        self.heads = nn.ModuleDict({
            str(bidx): nn.Sequential(
                nn.Linear(d_model, 128),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.SiLU(),
                nn.Linear(64, 2)
            ) for bidx in branches.keys()
        })
        
        # Learnable branch embeddings (for branch type awareness)
        self.branch_embeddings = nn.Embedding(len(branches), d_model)
        
    def forward(self, snapshot_sequence, edge_index, branch_idx):
        """
        Args:
            snapshot_sequence: [Batch, Time, Nodes, Features]
            edge_index: [2, Num_Edges]
            branch_idx: int or list of branch indices
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
        
        final_state = temporal_out[:, -1, :]  # [B, D]
        
        # Add branch embedding
        if isinstance(branch_idx, int):
            branch_idx = [branch_idx] * batch_size
        branch_idx_tensor = torch.tensor(branch_idx, device=final_state.device)
        branch_emb = self.branch_embeddings(branch_idx_tensor)
        
        combined = final_state + branch_emb
        
        # Branch-specific prediction
        outputs = []
        for i, bidx in enumerate(branch_idx):
            pred = self.heads[str(bidx)](combined[i:i+1])
            pred = F.softplus(pred) + 1e-6
            outputs.append(pred)
        
        return torch.cat(outputs, dim=0)
    
    def compute_physics_residual(self, params, node_features, edge_index, from_bus, to_bus):
        """Compute power flow residual."""
        R, X = params[:, 0], params[:, 1]
        
        P = node_features[:, 0]
        Q = node_features[:, 1]
        V = node_features[:, 2]
        
        v_from = V[from_bus]
        v_to = V[to_bus]
        
        delta_v = v_from - v_to
        Z_squared = R**2 + X**2
        
        expected_p = delta_v * v_from * R / Z_squared
        expected_q = delta_v * v_from * X / Z_squared
        
        actual_p = P[from_bus]
        actual_q = Q[from_bus]
        
        residual_p = (expected_p - actual_p) ** 2
        residual_q = (expected_q - actual_q) ** 2
        
        return residual_p + residual_q


class MultiBranchDataset(Dataset):
    """Dataset that samples from multiple branches."""
    
    def __init__(self, data_paths, branches, samples_per_branch=1000):
        """
        Args:
            data_paths: dict of {branch_idx: path_to_data}
            branches: dict of {branch_idx: (from_bus, to_bus)}
            samples_per_branch: number of samples to use per branch
        """
        self.data = {}
        self.branches = branches
        self.branch_list = list(branches.keys())
        
        for bidx, path in data_paths.items():
            with open(path, 'rb') as f:
                episodes = pickle.load(f)
            self.data[bidx] = episodes[:samples_per_branch]
        
        # Flatten all episodes with branch labels
        self.samples = []
        for bidx, episodes in self.data.items():
            for ep in episodes:
                self.samples.append((bidx, ep))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        bidx, ep = self.samples[idx]
        
        if isinstance(ep['snapshots'], torch.Tensor):
            x = ep['snapshots'].float()
        else:
            x = torch.tensor(ep['snapshots'], dtype=torch.float32)
        
        if isinstance(ep['true_params'], torch.Tensor):
            y = ep['true_params'].float()
        else:
            y = torch.tensor([ep['true_params']['r'], ep['true_params']['x']], dtype=torch.float32)
        
        return x, y, bidx


def train_epoch(model, dataloader, optimizer, criterion, device, branches):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_data_loss = 0
    total_phy_loss = 0
    
    for x, y, bidx in dataloader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # Get edge index (same for all in batch)
        edge_index = dataloader.dataset.data[bidx[0].item()][0]['edge_index'].to(device)
        
        # Forward pass with branch indices
        pred = model(x, edge_index, bidx.tolist())
        
        # Data loss
        loss_data = F.mse_loss(pred, y)
        
        # Physics loss (compute for each sample)
        loss_phy = 0
        for i in range(len(bidx)):
            from_bus, to_bus = branches[bidx[i].item()]
            nf = x[i, -1, :, :]  # Last timestep
            residual = model.compute_physics_residual(
                pred[i:i+1], nf, edge_index, from_bus, to_bus
            )
            loss_phy += residual.mean()
        
        loss_phy = loss_phy / len(bidx)
        
        # Total loss
        loss = loss_data + 0.1 * loss_phy
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_data_loss += loss_data.item()
        total_phy_loss += loss_phy.item()
    
    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'data_loss': total_data_loss / n,
        'phy_loss': total_phy_loss / n
    }


def validate(model, dataloader, device, branches):
    """Validate model."""
    model.eval()
    
    # Track errors per branch
    errors = {bidx: {'r': [], 'x': []} for bidx in branches.keys()}
    
    with torch.no_grad():
        for x, y, bidx in dataloader:
            x, y = x.to(device), y.to(device)
            edge_index = dataloader.dataset.data[bidx[0].item()][0]['edge_index'].to(device)
            
            pred = model(x, edge_index, bidx.tolist())
            
            # Compute errors per sample
            for i in range(len(bidx)):
                b = bidx[i].item()
                r_error = abs(pred[i, 0].item() - y[i, 0].item()) / y[i, 0].item() * 100
                x_error = abs(pred[i, 1].item() - y[i, 1].item()) / y[i, 1].item() * 100
                errors[b]['r'].append(r_error)
                errors[b]['x'].append(x_error)
    
    # Average errors per branch
    avg_errors = {}
    for bidx, errs in errors.items():
        avg_errors[bidx] = {
            'r': np.mean(errs['r']),
            'x': np.mean(errs['x']),
            'avg': (np.mean(errs['r']) + np.mean(errs['x'])) / 2
        }
    
    return avg_errors


def main():
    """Main training loop."""
    print("=" * 80)
    print("Multi-Branch Physics-Informed Training")
    print("=" * 80)
    
    # Configuration
    CONFIG = {
        'num_nodes': 33,
        'in_features': 3,
        'd_model': 64,
        'd_state': 16,
        'd_conv': 4,
        'expand': 2,
        'batch_size': 32,
        'epochs': 100,
        'lr': 1e-3,
        'samples_per_branch': 500,  # Reduced for faster training
    }
    
    # Define branches to train on
    BRANCHES = {
        3: (3, 4),    # Main branch
        7: (7, 8),    # Lateral branch  
        20: (20, 21), # End branch
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available.")
        return
    
    # Create data paths (need to generate data for each branch)
    print("\n[1] Preparing datasets...")
    
    # Check if multi-branch data exists, if not create it
    data_dir = '../data/multibranch'
    os.makedirs(data_dir, exist_ok=True)
    
    data_paths = {}
    for bidx in BRANCHES.keys():
        path = f'{data_dir}/branch_{bidx}_train.pkl'
        if not os.path.exists(path):
            print(f"  Generating data for branch {bidx}...")
            generate_branch_data(bidx, path, CONFIG['samples_per_branch'])
        data_paths[bidx] = path
    
    # Create datasets
    train_dataset = MultiBranchDataset(data_paths, BRANCHES, CONFIG['samples_per_branch'])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    # Use same data for validation (in practice, use separate data)
    val_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'])
    
    print(f"  Total training samples: {len(train_dataset)}")
    
    # Create model
    print("\n[2] Creating model...")
    model = MultiBranchGraphMamba(
        num_nodes=CONFIG['num_nodes'],
        in_features=CONFIG['in_features'],
        d_model=CONFIG['d_model'],
        branches=BRANCHES,
        d_state=CONFIG['d_state'],
        d_conv=CONFIG['d_conv'],
        expand=CONFIG['expand']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    
    # Training loop
    print("\n[3] Training...")
    best_avg_error = float('inf')
    
    for epoch in range(CONFIG['epochs']):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, None, device, BRANCHES
        )
        
        # Validate every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            val_errors = validate(model, val_loader, device, BRANCHES)
            
            avg_error = np.mean([e['avg'] for e in val_errors.values()])
            
            print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
            print(f"  Train: loss={train_metrics['loss']:.4f}")
            for bidx, errs in val_errors.items():
                print(f"  Branch {bidx}: R={errs['r']:.2f}%, X={errs['x']:.2f}%")
            print(f"  Avg error: {avg_error:.2f}%")
            
            # Save best model
            if avg_error < best_avg_error:
                best_avg_error = avg_error
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'branches': BRANCHES,
                    'config': CONFIG,
                    'val_errors': val_errors
                }, '../checkpoints/graph_mamba_multibranch_best.pt')
                print(f"  -> Saved best model")
        
        scheduler.step()
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best validation error: {best_avg_error:.2f}%")
    print("=" * 80)


def generate_branch_data(branch_idx, output_path, n_samples=500):
    """Generate training data for a specific branch."""
    import sys
    sys.path.insert(0, '..')
    from model.simulation import PowerSystemSimulation
    import pandapower as pp
    
    episodes = []
    
    for seed in range(n_samples):
        np.random.seed(seed)
        
        sim = PowerSystemSimulation(steps=50)
        
        # Get true parameters
        r_true = sim.net.line.at[branch_idx, 'r_ohm_per_km']
        x_true = sim.net.line.at[branch_idx, 'x_ohm_per_km']
        
        # Generate measurements
        snapshots = []
        p_base = sim.net.load.p_mw.values.copy()
        q_base = sim.net.load.q_mvar.values.copy()
        
        for t in range(50):
            # Add load variation
            sim.net.load.p_mw = p_base * (1 + np.random.normal(0, 0.05))
            sim.net.load.q_mvar = q_base * (1 + np.random.normal(0, 0.05))
            
            try:
                pp.runpp(sim.net, algorithm='nr', numba=False)
            except:
                continue
            
            # Get measurements
            num_buses = len(sim.net.bus)
            p_inj = -sim.net.res_bus.p_mw.values + np.random.normal(0, 0.02, num_buses)
            q_inj = -sim.net.res_bus.q_mvar.values + np.random.normal(0, 0.02, num_buses)
            v_mag = sim.net.res_bus.vm_pu.values + np.random.normal(0, 0.02, num_buses)
            
            snapshots.append(np.stack([p_inj, q_inj, v_mag], axis=1))
        
        # Build edge index
        edge_index = []
        for _, line in sim.net.line.iterrows():
            from_bus = int(line['from_bus'])
            to_bus = int(line['to_bus'])
            edge_index.append([from_bus, to_bus])
            edge_index.append([to_bus, from_bus])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        episodes.append({
            'snapshots': torch.tensor(np.array(snapshots), dtype=torch.float32),
            'edge_index': edge_index,
            'true_params': torch.tensor([r_true, x_true], dtype=torch.float32),
            'target_branch': branch_idx
        })
    
    with open(output_path, 'wb') as f:
        pickle.dump(episodes, f)
    
    print(f"  Saved {len(episodes)} episodes to {output_path}")


if __name__ == '__main__':
    main()
