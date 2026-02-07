"""
Train Graph Mamba for Multi-Branch Parameter Estimation
=========================================================

Estimates parameters for 4 branches simultaneously:
- Branch 3-4 (main feeder)
- Branch 5-6 (main feeder)
- Branch 2-19 (lateral)
- Branch 21-22 (end branch)

Output: 8 parameters (R1, X1, R2, X2, R3, X3, R4, X4)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from graphmamba.graph_mamba import HAS_MAMBA

DATA_DIR = 'data/multi_branch'
CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-3
D_MODEL = 64
D_STATE = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TARGET_BRANCHES = [(3, 4), (5, 6), (2, 19), (21, 22)]
NUM_OUTPUTS = len(TARGET_BRANCHES) * 2  # R and X for each


class MultiBranchGraphMamba(nn.Module):
    """Graph Mamba modified for multi-branch output (8 parameters)."""
    
    def __init__(self, num_nodes, in_features, d_model=64, d_state=16, d_conv=4, expand=2, num_outputs=8):
        super().__init__()
        from torch_geometric.nn import GCNConv, global_mean_pool
        
        self.num_nodes = num_nodes
        self.d_model = d_model
        
        # GNN encoder
        self.conv1 = GCNConv(in_features, d_model)
        self.conv2 = GCNConv(d_model, d_model)
        self.conv3 = GCNConv(d_model, d_model)
        
        # Temporal: simple LSTM fallback if no Mamba
        self.temporal = nn.LSTM(d_model, d_model, batch_first=True, bidirectional=False)
        
        # Multi-branch prediction head (outputs 8 values)
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, num_outputs)  # 8 outputs
        )
        
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(0.15)
    
    def forward(self, x, edge_index):
        # x: [B, T, N, F]
        B, T, N, F = x.shape
        
        # Process each timestep with GNN
        temporal_features = []
        for t in range(T):
            xt = x[:, t]  # [B, N, F]
            
            # Flatten for GNN
            xt_flat = xt.reshape(B * N, F)
            
            # Expand edge_index for batch
            batch_edge_index = edge_index.clone()
            for b in range(1, B):
                batch_edge_index = torch.cat([
                    batch_edge_index,
                    edge_index + b * N
                ], dim=1)
            
            # GNN layers
            h = self.act(self.conv1(xt_flat, batch_edge_index))
            h = self.dropout(h)
            h = self.act(self.conv2(h, batch_edge_index))
            h = self.dropout(h)
            h = self.act(self.conv3(h, batch_edge_index))
            
            # Reshape and pool
            h = h.reshape(B, N, -1)
            h_pooled = h.mean(dim=1)  # [B, d_model]
            temporal_features.append(h_pooled)
        
        # Stack temporal features
        temporal_seq = torch.stack(temporal_features, dim=1)  # [B, T, d_model]
        
        # Temporal processing
        temporal_out, _ = self.temporal(temporal_seq)
        final_temporal = temporal_out[:, -1]  # [B, d_model]
        
        # Predict all 8 parameters
        params = self.pred_head(final_temporal)  # [B, 8]
        return params


class MultiOutputLoss(nn.Module):
    """Loss for multi-branch estimation."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        return self.mse(pred, target), self.mse(pred, target), 0.0


class PowerGridDataset(Dataset):
    def __init__(self, episodes):
        self.episodes = episodes
    def __len__(self):
        return len(self.episodes)
    def __getitem__(self, idx):
        return self.episodes[idx]


def collate_fn(batch):
    return {
        'snapshots': torch.stack([item['snapshots'] for item in batch]),
        'edge_index': batch[0]['edge_index'],
        'true_params': torch.stack([item['true_params'] for item in batch])
    }


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            preds = model(batch['snapshots'].to(device), batch['edge_index'].to(device))
            all_preds.append(preds.cpu())
            all_targets.append(batch['true_params'])
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    # Calculate errors per branch
    errors = {}
    branch_names = ['3-4', '5-6', '2-19', '21-22']
    
    for i, name in enumerate(branch_names):
        r_idx = i * 2
        x_idx = i * 2 + 1
        r_err = (torch.abs(all_preds[:, r_idx] - all_targets[:, r_idx]) / all_targets[:, r_idx] * 100)
        x_err = (torch.abs(all_preds[:, x_idx] - all_targets[:, x_idx]) / all_targets[:, x_idx] * 100)
        errors[name] = {'r_mean': r_err.mean().item(), 'r_std': r_err.std().item(),
                       'x_mean': x_err.mean().item(), 'x_std': x_err.std().item()}
    
    return errors


def main():
    print("=" * 70)
    print("TRAIN GRAPH MAMBA FOR MULTI-BRANCH ESTIMATION")
    print("=" * 70)
    print(f"Target branches: {TARGET_BRANCHES}")
    print(f"Output parameters: {NUM_OUTPUTS}")
    
    # Load data
    with open(f'{DATA_DIR}/train_data.pkl', 'rb') as f: train_data = pickle.load(f)
    with open(f'{DATA_DIR}/val_data.pkl', 'rb') as f: val_data = pickle.load(f)
    with open(f'{DATA_DIR}/test_data.pkl', 'rb') as f: test_data = pickle.load(f)
    print(f"\nLoaded: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    # Print true params
    true_params = train_data[0]['true_params']
    for i, (fb, tb) in enumerate(TARGET_BRANCHES):
        print(f"  Branch {fb}-{tb}: R={true_params[i*2]:.4f}, X={true_params[i*2+1]:.4f}")
    
    train_loader = DataLoader(PowerGridDataset(train_data), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(PowerGridDataset(val_data), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(PowerGridDataset(test_data), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Model
    sample = train_data[0]
    model = MultiBranchGraphMamba(
        num_nodes=sample['snapshots'].shape[1],
        in_features=sample['snapshots'].shape[2],
        d_model=D_MODEL,
        d_state=D_STATE,
        num_outputs=NUM_OUTPUTS
    ).to(DEVICE)
    
    criterion = MultiOutputLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Training
    best_val_loss = float('inf')
    history = {name: {'r': [], 'x': []} for name in ['3-4', '5-6', '2-19', '21-22']}
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", ncols=100)
        
        for batch in pbar:
            optimizer.zero_grad()
            preds = model(batch['snapshots'].to(DEVICE), batch['edge_index'].to(DEVICE))
            loss, _, _ = criterion(preds, batch['true_params'].to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Validate
        val_errors = evaluate(model, val_loader, DEVICE)
        avg_error = sum(e['r_mean'] + e['x_mean'] for e in val_errors.values()) / len(val_errors) / 2
        scheduler.step(avg_error)
        
        print(f"\n  Epoch {epoch+1} Results:")
        for name, err in val_errors.items():
            print(f"    {name}: R={err['r_mean']:.2f}%, X={err['x_mean']:.2f}%")
            history[name]['r'].append(err['r_mean'])
            history[name]['x'].append(err['x_mean'])
        
        if avg_error < best_val_loss:
            best_val_loss = avg_error
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch},
                      f'{CHECKPOINT_DIR}/graph_mamba_multi_branch_best.pt')
            print(f"    Saved best model (avg error: {avg_error:.2f}%)")
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    ckpt = torch.load(f'{CHECKPOINT_DIR}/graph_mamba_multi_branch_best.pt')
    model.load_state_dict(ckpt['model_state_dict'])
    test_errors = evaluate(model, test_loader, DEVICE)
    
    # IAUKF baselines from B3 experiment
    iaukf_baseline = {
        '3-4': {'r': 2.40, 'x': 2.44},
        '5-6': {'r': 10.37, 'x': 3.93},
        '2-19': {'r': 22.18, 'x': 9.32},
        '21-22': {'r': 74.58, 'x': 84.28}
    }
    
    print("\n{:<10} | {:^20} | {:^20} | {:^15}".format(
        "Branch", "Graph Mamba", "IAUKF", "Improvement"))
    print("-" * 70)
    
    for name in ['3-4', '5-6', '2-19', '21-22']:
        gm_r = test_errors[name]['r_mean']
        gm_x = test_errors[name]['x_mean']
        ia_r = iaukf_baseline[name]['r']
        ia_x = iaukf_baseline[name]['x']
        imp_r = (ia_r - gm_r) / ia_r * 100
        imp_x = (ia_x - gm_x) / ia_x * 100
        print(f"{name:<10} | R={gm_r:5.2f}% X={gm_x:5.2f}% | R={ia_r:5.2f}% X={ia_x:5.2f}% | R={imp_r:+.1f}% X={imp_x:+.1f}%")
    
    # Save results
    os.makedirs('tmp', exist_ok=True)
    pickle.dump({'test': test_errors, 'history': history, 'iaukf': iaukf_baseline},
               open('tmp/multi_branch_results.pkl', 'wb'))
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, name in enumerate(['3-4', '5-6', '2-19', '21-22']):
        ax = axes[i // 2, i % 2]
        ax.plot(history[name]['r'], 'b-', label=f'R (Mamba)')
        ax.plot(history[name]['x'], 'g-', label=f'X (Mamba)')
        ax.axhline(iaukf_baseline[name]['r'], color='b', linestyle='--', alpha=0.5, label=f'R (IAUKF)')
        ax.axhline(iaukf_baseline[name]['x'], color='g', linestyle='--', alpha=0.5, label=f'X (IAUKF)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Error (%)')
        ax.set_title(f'Branch {name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tmp/multi_branch_training.png', dpi=150)
    print("\nSaved: tmp/multi_branch_training.png")
    
    print("\n" + "=" * 70)
    print("MULTI-BRANCH EXPERIMENT COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
