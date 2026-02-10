"""
Improved Graph-Mamba Training
==============================

Strategies to close the gap with multi-snapshot IAUKF (0.12% R error):
1. More training data with constant parameters (matching IAUKF test conditions)
2. Larger model capacity (deeper GNN, wider Mamba)
3. Longer training sequences
4. Better physics-informed loss weighting
5. Data augmentation (noise injection)
6. Separate prediction heads for R and X
"""

import sys
sys.path.insert(0, '..')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandapower as pp
from tqdm import tqdm
import json
import os

from graphmamba import GraphMambaPhysicsModel
from model.simulation import PowerSystemSimulation

# Configuration
CONFIG = {
    'num_episodes': 2000,  # More training data
    'episode_length': 100,
    'batch_size': 32,
    'num_epochs': 100,
    'lr': 1e-3,
    'weight_decay': 1e-5,
    
    # Model architecture - larger capacity
    'd_model': 128,  # Was 64
    'd_state': 32,   # Was 16
    'num_layers': 3, # Was 2
    
    # Loss weights - tuned for accuracy
    'lambda_physics': 0.2,  # Was 0.1
    'lambda_smoothness': 0.01,
    
    # Data augmentation
    'noise_augmentation': 0.01,  # Add noise during training
    
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
}


class PowerGridDatasetImproved(Dataset):
    """Dataset with constant parameters (matching IAUKF test conditions)."""
    
    def __init__(self, num_episodes=1000, episode_length=100, noise_scada=0.02, 
                 noise_pmu_v=0.005, noise_pmu_theta=0.002, seed=42):
        self.num_episodes = num_episodes
        self.episode_length = episode_length
        self.noise_scada = noise_scada
        self.noise_pmu_v = noise_pmu_v
        self.noise_pmu_theta = noise_pmu_theta
        
        np.random.seed(seed)
        self.episodes = []
        
        print(f"Generating {num_episodes} episodes with CONSTANT parameters...")
        
        for ep in tqdm(range(num_episodes)):
            # Randomly select a branch for this episode
            branch_idx = np.random.randint(0, 32)
            
            sim = PowerSystemSimulation(steps=episode_length)
            num_buses = len(sim.net.bus)
            
            # Get base loads (constant)
            p_base = sim.net.load.p_mw.values.copy()
            q_base = sim.net.load.q_mvar.values.copy()
            
            # Generate measurements with constant parameters
            snapshots = []
            for t in range(episode_length):
                sim.net.load.p_mw = p_base
                sim.net.load.q_mvar = q_base
                
                try:
                    pp.runpp(sim.net, algorithm='nr', numba=False, verbose=False)
                except:
                    continue
                
                # Features: [P, Q, V] for each bus
                features = np.zeros((num_buses, 3))
                features[:, 0] = -sim.net.res_bus.p_mw.values  # P
                features[:, 1] = -sim.net.res_bus.q_mvar.values  # Q
                features[:, 2] = sim.net.res_bus.vm_pu.values  # V
                
                # Add noise
                features += np.random.normal(0, noise_scada, features.shape)
                
                snapshots.append(features)
            
            if len(snapshots) == episode_length:
                # Get target parameters
                r_true = sim.net.line.at[branch_idx, 'r_ohm_per_km']
                x_true = sim.net.line.at[branch_idx, 'x_ohm_per_km']
                
                # Build edge index
                edge_index = []
                for _, line in sim.net.line.iterrows():
                    from_bus = int(line.from_bus)
                    to_bus = int(line.to_bus)
                    edge_index.append([from_bus, to_bus])
                    edge_index.append([to_bus, from_bus])
                edge_index = torch.tensor(edge_index, dtype=torch.long).t()
                
                self.episodes.append({
                    'snapshots': np.array(snapshots),
                    'edge_index': edge_index,
                    'r_true': r_true,
                    'x_true': x_true,
                    'branch_idx': branch_idx
                })
        
        print(f"Generated {len(self.episodes)} valid episodes")
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        ep = self.episodes[idx]
        
        x = torch.tensor(ep['snapshots'], dtype=torch.float32)
        y = torch.tensor([ep['r_true'], ep['x_true']], dtype=torch.float32)
        
        # Data augmentation during training
        if np.random.random() < 0.5:
            noise_scale = CONFIG['noise_augmentation']
            x += torch.randn_like(x) * noise_scale
        
        return x, ep['edge_index'], y, ep['branch_idx']


class ImprovedGraphMamba(nn.Module):
    """Graph-Mamba with improvements for accuracy."""
    
    def __init__(self, num_nodes=33, in_features=3, d_model=128, d_state=32, 
                 d_conv=4, expand=2, num_layers=3):
        super().__init__()
        
        # Use the physics model as base
        self.base_model = GraphMambaPhysicsModel(
            num_nodes=num_nodes,
            in_features=in_features,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # Replace prediction head with separate heads for R and X
        # This allows each parameter to have its own learning capacity
        self.r_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1)
        )
        
        self.x_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x, edge_index, branch_idx=None):
        # Get embeddings from base model (without final prediction)
        batch_size, seq_len, num_nodes, num_features = x.shape
        
        # Normalize
        x_flat = x.view(batch_size * seq_len * num_nodes, num_features)
        x_flat = self.base_model.normalizer(x_flat)
        x_norm = x_flat.view(batch_size, seq_len, num_nodes, num_features)
        
        # Graph encoding
        x_reshaped = x_norm.view(batch_size * seq_len, num_nodes, num_features)
        snapshots_flat = x_reshaped.view(batch_size * seq_len * num_nodes, num_features)
        
        # Batch edge index
        edge_index_batch = edge_index.repeat(1, batch_size * seq_len)
        offsets = torch.arange(batch_size * seq_len, device=edge_index.device) * num_nodes
        offsets = offsets.repeat_interleave(edge_index.size(1))
        edge_index_batch = edge_index_batch + offsets.unsqueeze(0)
        
        # Graph encoder
        from torch_geometric.nn import GCNConv
        embeddings = self.base_model.graph_encoder(snapshots_flat, edge_index_batch)
        embeddings = embeddings.view(batch_size, seq_len, self.base_model.d_model)
        
        # Temporal encoding with Mamba
        temporal_out = self.base_model.temporal_encoder(embeddings)
        final_state = temporal_out[:, -1, :]  # Last timestep
        
        # Separate predictions for R and X
        r_pred = self.r_head(final_state).squeeze(-1)
        x_pred = self.x_head(final_state).squeeze(-1)
        
        return torch.stack([r_pred, x_pred], dim=1)


def improved_physics_loss(pred, target, snapshots, edge_index, net_template):
    """Enhanced physics-informed loss with better numerical stability."""
    r_pred, x_pred = pred[:, 0], pred[:, 1]
    r_true, x_true = target[:, 0], target[:, 1]
    
    # MSE loss
    mse_loss = torch.mean((r_pred - r_true)**2 + (x_pred - x_true)**2)
    
    # Physics loss - simplified for stability
    # For each batch, compute power flow residual
    batch_size = pred.size(0)
    physics_loss = 0.0
    
    for i in range(batch_size):
        r, x = r_pred[i], x_pred[i]
        
        # Get average P, Q from snapshots
        avg_p = torch.mean(snapshots[i, :, :, 0])
        avg_q = torch.mean(snapshots[i, :, :, 1])
        avg_v = torch.mean(snapshots[i, :, :, 2])
        
        # Simplified physics: R*P + X*Q should be small relative to V^2
        # This approximates voltage drop equation
        residual = (r * avg_p + x * avg_q) / (avg_v**2 + 1e-6)
        physics_loss += residual**2
    
    physics_loss = physics_loss / batch_size
    
    # Smoothness loss - R/X ratio constraints
    ratio = r_pred / (x_pred + 1e-6)
    smoothness_loss = torch.mean(torch.relu(ratio - 5.0) + torch.relu(0.2 - ratio))
    
    total_loss = (mse_loss + 
                  CONFIG['lambda_physics'] * physics_loss + 
                  CONFIG['lambda_smoothness'] * smoothness_loss)
    
    return total_loss, mse_loss, physics_loss, smoothness_loss


def train():
    print("=" * 70)
    print("Improved Graph-Mamba Training")
    print("=" * 70)
    print(f"Config: {CONFIG}")
    print()
    
    device = torch.device(CONFIG['device'])
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    # Create datasets
    train_dataset = PowerGridDatasetImproved(
        num_episodes=CONFIG['num_episodes'],
        episode_length=CONFIG['episode_length']
    )
    
    val_dataset = PowerGridDatasetImproved(
        num_episodes=200,
        episode_length=CONFIG['episode_length'],
        seed=CONFIG['seed'] + 1000
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    
    # Create model
    model = ImprovedGraphMamba(
        d_model=CONFIG['d_model'],
        d_state=CONFIG['d_state'],
        num_layers=CONFIG['num_layers']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], 
                           weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    best_val_error = float('inf')
    
    for epoch in range(CONFIG['num_epochs']):
        # Training
        model.train()
        train_losses = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}"):
            x, edge_index, y, branch_idx = batch
            x = x.to(device)
            edge_index = edge_index[0].to(device)  # Assume same graph
            y = y.to(device)
            
            optimizer.zero_grad()
            pred = model(x, edge_index)
            
            # Use improved loss
            loss, mse_loss, phy_loss, smooth_loss = improved_physics_loss(
                pred, y, x, edge_index, None
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_errors = []
        
        with torch.no_grad():
            for batch in val_loader:
                x, edge_index, y, branch_idx = batch
                x = x.to(device)
                edge_index = edge_index[0].to(device)
                y = y.to(device)
                
                pred = model(x, edge_index)
                
                # Compute percentage error
                r_error = torch.abs(pred[:, 0] - y[:, 0]) / y[:, 0] * 100
                x_error = torch.abs(pred[:, 1] - y[:, 1]) / y[:, 1] * 100
                
                val_errors.extend((r_error + x_error).cpu().numpy() / 2)
        
        mean_val_error = np.mean(val_errors)
        
        print(f"Epoch {epoch+1}: Train Loss={np.mean(train_losses):.4f}, "
              f"Val Error={mean_val_error:.2f}%")
        
        # Save best model
        if mean_val_error < best_val_error:
            best_val_error = mean_val_error
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_error': mean_val_error,
                'config': CONFIG
            }, '../checkpoints/graph_mamba_improved_best.pt')
            print(f"  ✓ Saved best model (val error: {mean_val_error:.2f}%)")
    
    print()
    print(f"Training complete! Best validation error: {best_val_error:.2f}%")
    print(f"Target (multi-snapshot IAUKF): 0.12%")
    print(f"Gap: {best_val_error / 0.12:.1f}x")


if __name__ == '__main__':
    os.makedirs('../checkpoints', exist_ok=True)
    train()
