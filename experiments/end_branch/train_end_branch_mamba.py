"""
Train Graph Mamba on End Branch Data
=====================================
Tests if Graph Mamba can handle low-observability scenarios
where IAUKF fails (~83% error).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from graphmamba.graph_mamba import GraphMambaModel, PhysicsInformedLoss, HAS_MAMBA

DATA_DIR = 'data/end_branch'
CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-3
D_MODEL = 64
D_STATE = 16
D_CONV = 4
EXPAND = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PowerGridDataset(Dataset):
    def __init__(self, episodes):
        self.episodes = episodes
    def __len__(self):
        return len(self.episodes)
    def __getitem__(self, idx):
        return self.episodes[idx]

def collate_fn(batch):
    snapshots = torch.stack([item['snapshots'] for item in batch])
    true_params = torch.stack([item['true_params'] for item in batch])
    edge_index = batch[0]['edge_index']
    return {'snapshots': snapshots, 'edge_index': edge_index, 'true_params': true_params}

def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", ncols=100)
    for batch in pbar:
        snapshots = batch['snapshots'].to(device)
        edge_index = batch['edge_index'].to(device)
        targets = batch['true_params'].to(device)
        optimizer.zero_grad()
        preds = model(snapshots, edge_index)
        loss, loss_mse, _ = criterion(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in dataloader:
            preds = model(batch['snapshots'].to(device), batch['edge_index'].to(device))
            all_preds.append(preds.cpu())
            all_targets.append(batch['true_params'])
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    r_err = (torch.abs(all_preds[:,0] - all_targets[:,0]) / all_targets[:,0] * 100)
    x_err = (torch.abs(all_preds[:,1] - all_targets[:,1]) / all_targets[:,1] * 100)
    return {'r_mean': r_err.mean().item(), 'r_std': r_err.std().item(),
            'x_mean': x_err.mean().item(), 'x_std': x_err.std().item()}

def main():
    print("=" * 70)
    print("TRAIN GRAPH MAMBA ON END BRANCH (21-22)")
    print("=" * 70)
    
    with open(f'{DATA_DIR}/train_data.pkl', 'rb') as f: train_data = pickle.load(f)
    with open(f'{DATA_DIR}/val_data.pkl', 'rb') as f: val_data = pickle.load(f)
    with open(f'{DATA_DIR}/test_data.pkl', 'rb') as f: test_data = pickle.load(f)
    print(f"Loaded: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    r_true = train_data[0]['true_params'][0].item()
    x_true = train_data[0]['true_params'][1].item()
    print(f"Target: R={r_true:.4f}, X={x_true:.4f}")
    
    train_loader = DataLoader(PowerGridDataset(train_data), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(PowerGridDataset(val_data), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(PowerGridDataset(test_data), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    sample = train_data[0]
    model = GraphMambaModel(num_nodes=sample['snapshots'].shape[1], in_features=sample['snapshots'].shape[2],
                           d_model=D_MODEL, d_state=D_STATE, d_conv=D_CONV, expand=EXPAND).to(DEVICE)
    criterion = PhysicsInformedLoss(lambda_phy=0.01)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_val_loss = float('inf')
    history = {'r_error': [], 'x_error': []}
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE, epoch)
        val_m = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_m['r_mean'] + val_m['x_mean'])
        
        history['r_error'].append(val_m['r_mean'])
        history['x_error'].append(val_m['x_mean'])
        
        print(f"  Val: R={val_m['r_mean']:.2f}+/-{val_m['r_std']:.2f}%, X={val_m['x_mean']:.2f}+/-{val_m['x_std']:.2f}%")
        
        if val_m['r_mean'] + val_m['x_mean'] < best_val_loss:
            best_val_loss = val_m['r_mean'] + val_m['x_mean']
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'metrics': val_m},
                      f'{CHECKPOINT_DIR}/graph_mamba_end_branch_best.pt')
            print(f"  Saved best model")
    
    # Final evaluation
    ckpt = torch.load(f'{CHECKPOINT_DIR}/graph_mamba_end_branch_best.pt')
    model.load_state_dict(ckpt['model_state_dict'])
    test_m = evaluate(model, test_loader, criterion, DEVICE)
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"\nGraph Mamba on End Branch 21-22:")
    print(f"  R error: {test_m['r_mean']:.2f}% +/- {test_m['r_std']:.2f}%")
    print(f"  X error: {test_m['x_mean']:.2f}% +/- {test_m['x_std']:.2f}%")
    print(f"\nIAUKF baseline: R~83%, X~87%")
    print(f"\nImprovement: R={((83-test_m['r_mean'])/83*100):.1f}%, X={((87-test_m['x_mean'])/87*100):.1f}%")
    
    os.makedirs('tmp', exist_ok=True)
    pickle.dump({'test': test_m, 'history': history}, open('tmp/end_branch_results.pkl', 'wb'))
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['r_error'], 'b-', label='Graph Mamba')
    plt.axhline(83, color='r', linestyle='--', label='IAUKF')
    plt.xlabel('Epoch'); plt.ylabel('R Error (%)'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(history['x_error'], 'g-', label='Graph Mamba')
    plt.axhline(87, color='r', linestyle='--', label='IAUKF')
    plt.xlabel('Epoch'); plt.ylabel('X Error (%)'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tmp/end_branch_training.png', dpi=150)
    print("\nSaved: tmp/end_branch_training.png")

if __name__ == "__main__":
    main()
