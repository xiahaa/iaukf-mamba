"""
Train Graph-Mamba with Physics-Informed Loss
============================================

This script trains the Graph-Mamba model with physics-informed constraints
to improve parameter estimation accuracy.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
import json

from graphmamba.graph_mamba_physics import GraphMambaPhysicsModel, PhysicsInformedLossV2


class PowerSystemDataset(Dataset):
    """Dataset for power system parameter estimation."""
    
    def __init__(self, data_path, target_branch_idx=3):
        """
        Args:
            data_path: path to episode data file
            target_branch_idx: which branch to estimate parameters for
        """
        with open(data_path, 'rb') as f:
            self.episodes = pickle.load(f)
        
        self.target_branch_idx = target_branch_idx
        
        # Get network info from first episode
        first_ep = self.episodes[0]
        self.edge_index = first_ep['edge_index']
        
        # Get target branch from/to buses
        import sys
        sys.path.insert(0, '..')
        from model.simulation import PowerSystemSimulation
        sim = PowerSystemSimulation(steps=1)
        self.from_bus = int(sim.net.line.at[target_branch_idx, 'from_bus'])
        self.to_bus = int(sim.net.line.at[target_branch_idx, 'to_bus'])
        
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        ep = self.episodes[idx]
        
        # Get measurements [Time, Nodes, Features]
        # Data may already be tensors
        if isinstance(ep['snapshots'], torch.Tensor):
            x = ep['snapshots'].float()
        else:
            x = torch.tensor(ep['snapshots'], dtype=torch.float32)
        
        # Get true parameters [R, X]
        if isinstance(ep['true_params'], torch.Tensor):
            y = ep['true_params'].float()
        else:
            y = torch.tensor([ep['true_params']['r'], ep['true_params']['x']], dtype=torch.float32)
        
        return x, y


def train_epoch(model, dataloader, optimizer, criterion, device, target_branch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_data_loss = 0
    total_phy_loss = 0
    
    from_bus, to_bus = target_branch
    
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(x, train_dataset.edge_index.to(device))
        
        # Compute loss with physics constraints
        loss, loss_dict = criterion(
            pred, y,
            model=model,
            node_features=x,
            edge_index=train_dataset.edge_index.to(device),
            target_branch=(from_bus, to_bus)
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_data_loss += loss_dict['data']
        total_phy_loss += loss_dict['physics']
    
    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'data_loss': total_data_loss / n,
        'phy_loss': total_phy_loss / n
    }


def validate(model, dataloader, criterion, device, target_branch):
    """Validate model."""
    model.eval()
    total_loss = 0
    total_r_error = 0
    total_x_error = 0
    
    from_bus, to_bus = target_branch
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            pred = model(x, val_dataset.edge_index.to(device))
            
            loss, _ = criterion(
                pred, y,
                model=model,
                node_features=x,
                edge_index=val_dataset.edge_index.to(device),
                target_branch=(from_bus, to_bus)
            )
            
            total_loss += loss.item()
            
            # Compute errors
            r_error = torch.abs(pred[:, 0] - y[:, 0]) / y[:, 0] * 100
            x_error = torch.abs(pred[:, 1] - y[:, 1]) / y[:, 1] * 100
            
            total_r_error += r_error.mean().item()
            total_x_error += x_error.mean().item()
    
    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'r_error': total_r_error / n,
        'x_error': total_x_error / n
    }


def main():
    """Main training loop."""
    print("=" * 80)
    print("Training Graph-Mamba with Physics-Informed Loss")
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
        'lambda_phy': 0.1,  # Physics loss weight
        'target_branch': 3,  # Branch 3-4
        'train_data': '../data/phase2/train_data.pkl',
        'val_data': '../data/phase2/val_data.pkl',
        'checkpoint_dir': '../checkpoints'
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Physics-informed training requires GPU.")
        return
    
    # Load datasets
    print("\n[1] Loading datasets...")
    global train_dataset, val_dataset
    train_dataset = PowerSystemDataset(CONFIG['train_data'], CONFIG['target_branch'])
    val_dataset = PowerSystemDataset(CONFIG['val_data'], CONFIG['target_branch'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # Create model
    print("\n[2] Creating model...")
    model = GraphMambaPhysicsModel(
        num_nodes=CONFIG['num_nodes'],
        in_features=CONFIG['in_features'],
        d_model=CONFIG['d_model'],
        d_state=CONFIG['d_state'],
        d_conv=CONFIG['d_conv'],
        expand=CONFIG['expand']
    ).to(device)
    
    # Loss and optimizer
    criterion = PhysicsInformedLossV2(lambda_phy=CONFIG['lambda_phy'])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    
    # Training loop
    print("\n[3] Training...")
    best_val_error = float('inf')
    target_branch = (train_dataset.from_bus, train_dataset.to_bus)
    
    for epoch in range(CONFIG['epochs']):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, target_branch
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, target_branch)
        
        scheduler.step()
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
            print(f"  Train: loss={train_metrics['loss']:.4f}, "
                  f"data={train_metrics['data_loss']:.4f}, "
                  f"phy={train_metrics['phy_loss']:.4f}")
            print(f"  Val: loss={val_metrics['loss']:.4f}, "
                  f"R_err={val_metrics['r_error']:.2f}%, "
                  f"X_err={val_metrics['x_error']:.2f}%")
        
        # Save best model
        avg_val_error = (val_metrics['r_error'] + val_metrics['x_error']) / 2
        if avg_val_error < best_val_error:
            best_val_error = avg_val_error
            os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_error': avg_val_error,
                'config': CONFIG
            }, os.path.join(CONFIG['checkpoint_dir'], 'graph_mamba_physics_best.pt'))
            print(f"  -> Saved best model (val error: {avg_val_error:.2f}%)")
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best validation error: {best_val_error:.2f}%")
    print(f"Checkpoint saved to: {CONFIG['checkpoint_dir']}/graph_mamba_physics_best.pt")
    print("=" * 80)


if __name__ == '__main__':
    main()
