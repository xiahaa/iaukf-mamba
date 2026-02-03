"""
Phase 2 Step 2: Train Graph Mamba
Loads pre-generated dataset and trains the model.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from tqdm import tqdm

from graph_mamba import GraphMambaModel, PhysicsInformedLoss, HAS_MAMBA

# SwanLab
try:
    import swanlab
    HAS_SWANLAB = True
except ImportError:
    HAS_SWANLAB = False
    print("Warning: swanlab not installed. Continuing without logging...")

# Configuration
DATA_DIR = 'data/phase2'
CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Training hyperparameters
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
LAMBDA_PHY = 0.01

# Model architecture
D_MODEL = 64
D_STATE = 16
D_CONV = 4
EXPAND = 2

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 70)
print("PHASE 2 STEP 2: TRAIN GRAPH MAMBA")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  Data directory: {DATA_DIR}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Device: {DEVICE}")
print(f"  Mamba available: {HAS_MAMBA}")
if DEVICE.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")


# Dataset
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
    return {
        'snapshots': snapshots,
        'edge_index': edge_index,
        'true_params': true_params
    }


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train one epoch with progress bar."""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_phy = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]", ncols=120)

    for batch in pbar:
        snapshots = batch['snapshots'].to(device)
        edge_index = batch['edge_index'].to(device)
        targets = batch['true_params'].to(device)

        optimizer.zero_grad()
        preds = model(snapshots, edge_index)

        loss, loss_mse, loss_phy = criterion(preds, targets)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_phy += loss_phy.item() if isinstance(loss_phy, torch.Tensor) else 0.0

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'mse': f"{loss_mse.item():.4f}"
        })

    n = len(dataloader)
    return total_loss / n, total_mse / n, total_phy / n


def evaluate(model, dataloader, criterion, device, desc="Eval"):
    """Evaluate with progress bar."""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(dataloader, desc=f"  [{desc}]", ncols=120, leave=False)

    with torch.no_grad():
        for batch in pbar:
            snapshots = batch['snapshots'].to(device)
            edge_index = batch['edge_index'].to(device)
            targets = batch['true_params'].to(device)

            preds = model(snapshots, edge_index)
            loss, loss_mse, _ = criterion(preds, targets)

            total_loss += loss.item()
            total_mse += loss_mse.item()
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    n = len(dataloader)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Calculate percentage errors
    r_error = torch.abs(all_preds[:, 0] - all_targets[:, 0]) / all_targets[:, 0] * 100
    x_error = torch.abs(all_preds[:, 1] - all_targets[:, 1]) / all_targets[:, 1] * 100

    return {
        'loss': total_loss / n,
        'mse': total_mse / n,
        'r_error_mean': r_error.mean().item(),
        'r_error_std': r_error.std().item(),
        'r_error_median': r_error.median().item(),
        'x_error_mean': x_error.mean().item(),
        'x_error_std': x_error.std().item(),
        'x_error_median': x_error.median().item(),
    }


def main():
    # Load datasets
    print("\n" + "=" * 70)
    print("STEP 1: LOAD DATASET")
    print("=" * 70)

    print("\nLoading saved datasets...")

    with open(os.path.join(DATA_DIR, 'train_data.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    print(f"  ✓ Loaded training: {len(train_data)} episodes")

    with open(os.path.join(DATA_DIR, 'val_data.pkl'), 'rb') as f:
        val_data = pickle.load(f)
    print(f"  ✓ Loaded validation: {len(val_data)} episodes")

    with open(os.path.join(DATA_DIR, 'test_data.pkl'), 'rb') as f:
        test_data = pickle.load(f)
    print(f"  ✓ Loaded test: {len(test_data)} episodes")

    # Create dataloaders
    train_loader = DataLoader(
        PowerGridDataset(train_data),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 for compatibility
        pin_memory=True if DEVICE.type == 'cuda' else False
    )

    val_loader = DataLoader(
        PowerGridDataset(val_data),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )

    test_loader = DataLoader(
        PowerGridDataset(test_data),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )

    print(f"\n✓ Dataloaders created:")
    print(f"  Train: {len(train_loader)} batches ({len(train_data)} episodes)")
    print(f"  Val: {len(val_loader)} batches ({len(val_data)} episodes)")
    print(f"  Test: {len(test_loader)} batches ({len(test_data)} episodes)")

    # Create model
    print("\n" + "=" * 70)
    print("STEP 2: CREATE MODEL")
    print("=" * 70)

    sample = train_data[0]
    num_nodes = sample['snapshots'].shape[1]
    in_features = sample['snapshots'].shape[2]

    print(f"\n  Input shape: [Time={sample['snapshots'].shape[0]}, "
          f"Nodes={num_nodes}, Features={in_features}]")
    print(f"  Model config: d_model={D_MODEL}, d_state={D_STATE}")

    model = GraphMambaModel(
        num_nodes=num_nodes,
        in_features=in_features,
        d_model=D_MODEL,
        d_state=D_STATE,
        d_conv=D_CONV,
        expand=EXPAND
    ).to(DEVICE)

    criterion = PhysicsInformedLoss(lambda_phy=LAMBDA_PHY)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✓ Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Initialize SwanLab
    if HAS_SWANLAB:
        swanlab.init(
            project="power-grid-iaukf",
            experiment_name=f"Phase2_GraphMamba_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                'num_train': len(train_data),
                'num_val': len(val_data),
                'batch_size': BATCH_SIZE,
                'epochs': EPOCHS,
                'learning_rate': LEARNING_RATE,
                'd_model': D_MODEL,
                'lambda_phy': LAMBDA_PHY,
                'has_mamba': HAS_MAMBA,
            }
        )

    # Training loop
    print("\n" + "=" * 70)
    print("STEP 3: TRAINING")
    print("=" * 70)

    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'val_loss': [],
        'val_r_error': [], 'val_x_error': [],
        'lr': []
    }

    for epoch in range(EPOCHS):
        # Train
        train_loss, train_mse, train_phy = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE, epoch
        )

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, DEVICE, desc="Val")

        # Scheduler step
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']

        # Print LR reduction if it changed
        if current_lr != old_lr:
            print(f"    → Learning rate reduced: {old_lr:.2e} → {current_lr:.2e}")

        # Print epoch summary
        print(f"\n  Epoch {epoch+1}/{EPOCHS} Summary:")
        print(f"    Train: loss={train_loss:.6f}, mse={train_mse:.6f}")
        print(f"    Val: loss={val_metrics['loss']:.6f}, "
              f"R={val_metrics['r_error_mean']:.2f}±{val_metrics['r_error_std']:.2f}%, "
              f"X={val_metrics['x_error_mean']:.2f}±{val_metrics['x_error_std']:.2f}%")
        print(f"    LR: {current_lr:.2e}")

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_r_error'].append(val_metrics['r_error_mean'])
        history['val_x_error'].append(val_metrics['x_error_mean'])
        history['lr'].append(current_lr)

        # Log to SwanLab
        if HAS_SWANLAB:
            swanlab.log({
                'train_loss': train_loss,
                'train_mse': train_mse,
                'val_loss': val_metrics['loss'],
                'val_r_error_mean': val_metrics['r_error_mean'],
                'val_r_error_std': val_metrics['r_error_std'],
                'val_x_error_mean': val_metrics['x_error_mean'],
                'val_x_error_std': val_metrics['x_error_std'],
                'learning_rate': current_lr
            }, step=epoch)

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint_path = os.path.join(CHECKPOINT_DIR, 'graph_mamba_phase2_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': best_val_loss,
                'val_metrics': val_metrics,
                'history': history,
            }, checkpoint_path)
            print(f"    ✓ Saved best model (val_loss={best_val_loss:.6f})")

    # Final evaluation
    print("\n" + "=" * 70)
    print("STEP 4: FINAL EVALUATION")
    print("=" * 70)

    # Load best model
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'graph_mamba_phase2_best.pt')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n✓ Loaded best model from epoch {checkpoint['epoch']+1}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, criterion, DEVICE, desc="Test")

    print(f"\nTest Results:")
    print(f"  R error: {test_metrics['r_error_mean']:.2f}% ± {test_metrics['r_error_std']:.2f}% "
          f"(median: {test_metrics['r_error_median']:.2f}%)")
    print(f"  X error: {test_metrics['x_error_mean']:.2f}% ± {test_metrics['x_error_std']:.2f}% "
          f"(median: {test_metrics['x_error_median']:.2f}%)")

    # Compare with IAUKF
    print(f"\n" + "=" * 70)
    print("COMPARISON WITH IAUKF (PHASE 1)")
    print("=" * 70)
    print(f"\nIAUKF (Phase 1):")
    print(f"  R error: 1.60%")
    print(f"  X error: 2.00%")
    print(f"\nGraph Mamba (Phase 2):")
    print(f"  R error: {test_metrics['r_error_mean']:.2f}%")
    print(f"  X error: {test_metrics['x_error_mean']:.2f}%")

    if test_metrics['r_error_mean'] < 3.0 and test_metrics['x_error_mean'] < 3.0:
        print(f"\n✓✓✓ SUCCESS: Graph Mamba matches IAUKF performance! ✓✓✓")
        success = True
    elif test_metrics['r_error_mean'] < 5.0 and test_metrics['x_error_mean'] < 5.0:
        print(f"\n✓ GOOD: Graph Mamba achieves reasonable performance")
        success = True
    else:
        print(f"\n⚠ Needs improvement")
        success = False

    # Plot training history
    print("\nGenerating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # R error
    axes[0, 1].plot(history['val_r_error'], 'b-', linewidth=2, label='Graph Mamba')
    axes[0, 1].axhline(1.60, color='r', linestyle='--', linewidth=2, label='IAUKF (1.60%)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Error (%)')
    axes[0, 1].set_title('Resistance Error', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # X error
    axes[1, 0].plot(history['val_x_error'], 'g-', linewidth=2, label='Graph Mamba')
    axes[1, 0].axhline(2.00, color='r', linestyle='--', linewidth=2, label='IAUKF (2.00%)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Error (%)')
    axes[1, 0].set_title('Reactance Error', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Learning rate
    axes[1, 1].plot(history['lr'], 'orange', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule', fontweight='bold')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tmp/phase2_training_history.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: tmp/phase2_training_history.png")

    if HAS_SWANLAB:
        swanlab.log({
            "training_history": swanlab.Image("tmp/phase2_training_history.png"),
            "test_r_error": test_metrics['r_error_mean'],
            "test_x_error": test_metrics['x_error_mean']
        })
        swanlab.finish()

    plt.close()

    # Save final results
    results = {
        'test_metrics': test_metrics,
        'history': history,
        'best_epoch': checkpoint['epoch'],
        'comparison': {
            'iaukf': {'r_error': 1.60, 'x_error': 2.00},
            'graph_mamba': {
                'r_error': test_metrics['r_error_mean'],
                'x_error': test_metrics['x_error_mean']
            }
        }
    }

    with open('tmp/phase2_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"✓ Saved results: tmp/phase2_results.pkl")

    print(f"\n" + "=" * 70)
    print("✓ PHASE 2 COMPLETE!")
    print("=" * 70)

    if success:
        print(f"\n✓ Ready for Phase 3: Time-varying parameters")
    else:
        print(f"\nSuggestions for improvement:")
        print(f"  - Train for more epochs")
        print(f"  - Tune learning rate")
        print(f"  - Adjust model architecture (d_model, layers)")
        print(f"  - Add data augmentation")


if __name__ == "__main__":
    main()
