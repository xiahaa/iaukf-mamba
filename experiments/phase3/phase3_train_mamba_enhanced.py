"""
Phase 3: Train Enhanced Graph Mamba on Time-Varying Parameters
================================================================

Uses graph_mamba_enhanced.py with:
- Residual connections
- Layer normalization
- Temporal attention
- Stochastic depth
- Better regularization
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime

try:
    import swanlab
    HAS_SWANLAB = True
except ImportError:
    HAS_SWANLAB = False
    print("Warning: swanlab not available")

from graph_mamba_enhanced import EnhancedGraphMambaModel, HAS_MAMBA

# ========================================
# Configuration
# ========================================

DATA_DIR = 'data/phase3'
CHECKPOINT_DIR = 'checkpoints'
RESULTS_DIR = 'tmp'

for d in [CHECKPOINT_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Training hyperparameters
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4  # Increased from 1e-5

# Model hyperparameters (Enhanced)
D_MODEL = 64
D_STATE = 16
USE_ATTENTION = True
USE_PROBABILISTIC = False  # Set True if you want uncertainty estimates
DROP_PATH = 0.1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

print("=" * 70)
print("PHASE 3 ENHANCED: TRAIN ADVANCED GRAPH MAMBA")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  Data directory: {DATA_DIR}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Weight decay: {WEIGHT_DECAY}")
print(f"  Device: {DEVICE}")
print(f"  Mamba available: {HAS_MAMBA}")
print(f"\nEnhanced Features:")
print(f"  Temporal attention: {USE_ATTENTION}")
print(f"  Probabilistic head: {USE_PROBABILISTIC}")
print(f"  Stochastic depth: {DROP_PATH}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# ========================================
# Dataset (same as standard version)
# ========================================

class TimeVaryingDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
        self.edge_index = data_list[0]['edge_index']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'snapshots': self.data[idx]['snapshots'],
            'edge_index': self.edge_index,
            'r_profile': self.data[idx]['r_profile'],
            'x_profile': self.data[idx]['x_profile']
        }


def collate_fn(batch):
    """Custom collate to handle edge_index"""
    snapshots = torch.stack([item['snapshots'] for item in batch])
    r_profile = torch.stack([item['r_profile'] for item in batch])
    x_profile = torch.stack([item['x_profile'] for item in batch])
    edge_index = batch[0]['edge_index']

    return {
        'snapshots': snapshots,
        'edge_index': edge_index,
        'r_profile': r_profile,
        'x_profile': x_profile
    }


# ========================================
# Enhanced Model Wrapper
# ========================================

class EnhancedGraphMambaTimeSeriesModel(nn.Module):
    """
    Wrapper for EnhancedGraphMambaModel to predict at each timestep.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

        # Per-timestep prediction head
        self.per_step_head = nn.Sequential(
            nn.Linear(base_model.d_model, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.2),  # Increased dropout
            nn.Linear(64, 2)
        )

    def forward(self, snapshot_sequence, edge_index, return_all_steps=True):
        """
        Forward pass through enhanced model.

        Returns:
            [Batch, Time, 2] if return_all_steps=True
            [Batch, 2] otherwise
        """
        batch_size, seq_len, num_nodes, num_features = snapshot_sequence.shape

        # Get embeddings from enhanced base model
        # The base model's forward returns just predictions, but we need embeddings
        # So we'll call its internal methods

        # Normalize
        snapshot_sequence = self.base_model.normalizer(snapshot_sequence)

        # Spatial encoding
        flat_snapshots = snapshot_sequence.view(batch_size * seq_len, num_nodes, num_features)
        batch_idx = torch.arange(batch_size * seq_len, device=snapshot_sequence.device)
        batch_idx = batch_idx.repeat_interleave(num_nodes)
        flat_node_features = flat_snapshots.view(-1, num_features)

        # Expand edge_index
        num_edges = edge_index.size(1)
        edge_index_batch = edge_index.repeat(1, batch_size * seq_len)
        offsets = torch.arange(batch_size * seq_len, device=edge_index.device) * num_nodes
        offsets = offsets.repeat_interleave(num_edges)
        edge_index_batch = edge_index_batch + offsets.view(1, -1)

        # Process through enhanced GNN
        embeddings = self.base_model.graph_encoder(flat_node_features, edge_index_batch, batch=batch_idx)
        embeddings = embeddings.view(batch_size, seq_len, self.base_model.d_model)

        # Temporal attention (if enabled)
        if self.base_model.use_attention:
            embeddings = self.base_model.temporal_attention(embeddings)

        # Temporal processing (Mamba/LSTM)
        if HAS_MAMBA:
            temporal_out = self.base_model.temporal_layer(embeddings)
        else:
            temporal_out, _ = self.base_model.temporal_layer(embeddings)

        # Predict at each timestep
        if return_all_steps:
            flat_temporal = temporal_out.view(batch_size * seq_len, self.base_model.d_model)
            predictions = self.per_step_head(flat_temporal)
            predictions = predictions.view(batch_size, seq_len, 2)
            predictions = torch.nn.functional.softplus(predictions) + 1e-6
            return predictions
        else:
            final_state = temporal_out[:, -1, :]
            predictions = self.per_step_head(final_state)
            predictions = torch.nn.functional.softplus(predictions) + 1e-6
            return predictions


# ========================================
# Loss Function (same as standard)
# ========================================

class TimeSeriesLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        loss = self.mse(pred, target)
        r_loss = self.mse(pred[:, :, 0], target[:, :, 0])
        x_loss = self.mse(pred[:, :, 1], target[:, :, 1])
        return loss, r_loss, x_loss


# ========================================
# Training Functions
# ========================================

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_r_loss = 0
    total_x_loss = 0

    pbar = tqdm(loader, desc="Train")
    for batch in pbar:
        snapshots = batch['snapshots'].to(device)
        edge_index = batch['edge_index'].to(device)
        r_profile = batch['r_profile'].to(device)
        x_profile = batch['x_profile'].to(device)

        target = torch.stack([r_profile, x_profile], dim=-1)

        # Forward
        pred = model(snapshots, edge_index, return_all_steps=True)

        # Loss
        loss, r_loss, x_loss = criterion(pred, target)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Log
        total_loss += loss.item()
        total_r_loss += r_loss.item()
        total_x_loss += x_loss.item()

        pbar.set_postfix({'loss': loss.item()})

    return {
        'loss': total_loss / len(loader),
        'r_loss': total_r_loss / len(loader),
        'x_loss': total_x_loss / len(loader)
    }


def evaluate(model, loader, criterion, device, desc="Val"):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    total_r_loss = 0
    total_x_loss = 0

    all_r_errors = []
    all_x_errors = []

    with torch.no_grad():
        pbar = tqdm(loader, desc=desc, leave=False)
        for batch in pbar:
            snapshots = batch['snapshots'].to(device)
            edge_index = batch['edge_index'].to(device)
            r_profile = batch['r_profile'].to(device)
            x_profile = batch['x_profile'].to(device)

            target = torch.stack([r_profile, x_profile], dim=-1)

            pred = model(snapshots, edge_index, return_all_steps=True)

            loss, r_loss, x_loss = criterion(pred, target)

            total_loss += loss.item()
            total_r_loss += r_loss.item()
            total_x_loss += x_loss.item()

            # Compute percentage errors
            r_errors = torch.abs(pred[:, :, 0] - target[:, :, 0]) / target[:, :, 0] * 100
            x_errors = torch.abs(pred[:, :, 1] - target[:, :, 1]) / target[:, :, 1] * 100

            all_r_errors.extend(r_errors.flatten().cpu().numpy())
            all_x_errors.extend(x_errors.flatten().cpu().numpy())

    all_r_errors = np.array(all_r_errors)
    all_x_errors = np.array(all_x_errors)

    return {
        'loss': total_loss / len(loader),
        'r_loss': total_r_loss / len(loader),
        'x_loss': total_x_loss / len(loader),
        'r_error_mean': all_r_errors.mean(),
        'r_error_std': all_r_errors.std(),
        'x_error_mean': all_x_errors.mean(),
        'x_error_std': all_x_errors.std()
    }


# ========================================
# Main Training Loop
# ========================================

def main():
    # Load data
    print("\n" + "=" * 70)
    print("STEP 1: LOAD DATASET")
    print("=" * 70)

    with open(os.path.join(DATA_DIR, 'train_data.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'val_data.pkl'), 'rb') as f:
        val_data = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'test_data.pkl'), 'rb') as f:
        test_data = pickle.load(f)

    print(f"\n✓ Loaded datasets:")
    print(f"  Train: {len(train_data)} episodes")
    print(f"  Val: {len(val_data)} episodes")
    print(f"  Test: {len(test_data)} episodes")

    # Create datasets and loaders
    train_dataset = TimeVaryingDataset(train_data)
    val_dataset = TimeVaryingDataset(val_data)
    test_dataset = TimeVaryingDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True, collate_fn=collate_fn)

    print(f"\n✓ Dataloaders created:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")

    # Create enhanced model
    print("\n" + "=" * 70)
    print("STEP 2: CREATE ENHANCED MODEL")
    print("=" * 70)

    sample = train_data[0]
    num_nodes = sample['snapshots'].shape[1]
    in_features = sample['snapshots'].shape[2]

    print(f"\n  Input shape: [Time={sample['snapshots'].shape[0]}, Nodes={num_nodes}, Features={in_features}]")
    print(f"  Model config: d_model={D_MODEL}, d_state={D_STATE}")
    print(f"  Enhanced features: attention={USE_ATTENTION}, probabilistic={USE_PROBABILISTIC}, drop_path={DROP_PATH}")

    # Create enhanced base model
    base_model = EnhancedGraphMambaModel(
        num_nodes=num_nodes,
        in_features=in_features,
        d_model=D_MODEL,
        d_state=D_STATE,
        d_conv=4,
        expand=2,
        use_attention=USE_ATTENTION,
        use_probabilistic=USE_PROBABILISTIC,
        drop_path=DROP_PATH
    )

    # Wrap with time-series model
    model = EnhancedGraphMambaTimeSeriesModel(base_model).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n✓ Enhanced model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Training setup
    criterion = TimeSeriesLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # Swanlab logging
    if HAS_SWANLAB:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        swanlab.init(
            project="power-grid-iaukf",
            experiment_name=f"Phase3_Enhanced_{timestamp}",
            config={
                'phase': 3,
                'model': 'enhanced',
                'batch_size': BATCH_SIZE,
                'epochs': EPOCHS,
                'lr': LEARNING_RATE,
                'weight_decay': WEIGHT_DECAY,
                'd_model': D_MODEL,
                'd_state': D_STATE,
                'attention': USE_ATTENTION,
                'probabilistic': USE_PROBABILISTIC,
                'drop_path': DROP_PATH,
                'device': str(DEVICE),
                'mamba': HAS_MAMBA
            }
        )

    # Training loop
    print("\n" + "=" * 70)
    print("STEP 3: TRAINING ENHANCED MODEL")
    print("=" * 70)

    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_r_error': [],
        'val_x_error': []
    }

    for epoch in range(1, EPOCHS + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, DEVICE)

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, DEVICE, desc="Val")

        # Scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']

        if current_lr != old_lr:
            print(f"    → Learning rate reduced: {old_lr:.2e} → {current_lr:.2e}")

        # Log
        print(f"\n  Epoch {epoch}/{EPOCHS} Summary:")
        print(f"    Train: loss={train_metrics['loss']:.6f}")
        print(f"    Val: loss={val_metrics['loss']:.6f}, R={val_metrics['r_error_mean']:.2f}±{val_metrics['r_error_std']:.2f}%, X={val_metrics['x_error_mean']:.2f}±{val_metrics['x_error_std']:.2f}%")
        print(f"    LR: {current_lr:.2e}")

        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_r_error'].append(val_metrics['r_error_mean'])
        history['val_x_error'].append(val_metrics['x_error_mean'])

        # Swanlab logging
        if HAS_SWANLAB:
            swanlab.log({
                'train/loss': train_metrics['loss'],
                'val/loss': val_metrics['loss'],
                'val/r_error': val_metrics['r_error_mean'],
                'val/x_error': val_metrics['x_error_mean'],
                'lr': current_lr
            }, step=epoch)

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_metrics': val_metrics
            }, os.path.join(CHECKPOINT_DIR, 'graph_mamba_phase3_enhanced_best.pt'))
            print(f"    ✓ Saved best enhanced model (val_loss={best_val_loss:.6f})")

    # Final evaluation
    print("\n" + "=" * 70)
    print("STEP 4: FINAL EVALUATION")
    print("=" * 70)

    # Load best model
    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, 'graph_mamba_phase3_enhanced_best.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n✓ Loaded best enhanced model from epoch {checkpoint['epoch']}")

    # Evaluate on test set
    test_metrics = evaluate(model, test_loader, criterion, DEVICE, desc="Test")

    print(f"\nTest Results (Enhanced Model):")
    print(f"  R error: {test_metrics['r_error_mean']:.2f}% ± {test_metrics['r_error_std']:.2f}%")
    print(f"  X error: {test_metrics['x_error_mean']:.2f}% ± {test_metrics['x_error_std']:.2f}%")

    # Comparison with standard model
    print("\n" + "=" * 70)
    print("COMPARISON: STANDARD vs ENHANCED")
    print("=" * 70)

    # Try to load standard model results
    try:
        standard_checkpoint = torch.load('checkpoints/graph_mamba_phase3_best.pt', weights_only=False)
        standard_metrics = standard_checkpoint['val_metrics']

        print(f"\nStandard Model (Phase 3):")
        print(f"  R error: {standard_metrics['r_error_mean']:.2f}%")
        print(f"  X error: {standard_metrics['x_error_mean']:.2f}%")

        print(f"\nEnhanced Model (Phase 3):")
        print(f"  R error: {test_metrics['r_error_mean']:.2f}%")
        print(f"  X error: {test_metrics['x_error_mean']:.2f}%")

        improvement_r = (standard_metrics['r_error_mean'] - test_metrics['r_error_mean']) / standard_metrics['r_error_mean'] * 100
        improvement_x = (standard_metrics['x_error_mean'] - test_metrics['x_error_mean']) / standard_metrics['x_error_mean'] * 100

        print(f"\nImprovement:")
        print(f"  R: {improvement_r:+.1f}%")
        print(f"  X: {improvement_x:+.1f}%")

        if improvement_r > 0 and improvement_x > 0:
            print(f"\n✓✓✓ Enhanced model is better! ✓✓✓")
        elif improvement_r > 0 or improvement_x > 0:
            print(f"\n✓ Enhanced model shows improvement on one metric")
        else:
            print(f"\n→ Standard model performed similarly (enhanced features may not be necessary)")

    except FileNotFoundError:
        print("\n⚠ Standard model results not found for comparison.")

    # Save results
    results = {
        'test_metrics': test_metrics,
        'history': history,
        'config': {
            'd_model': D_MODEL,
            'd_state': D_STATE,
            'attention': USE_ATTENTION,
            'probabilistic': USE_PROBABILISTIC,
            'drop_path': DROP_PATH,
            'epochs': EPOCHS,
            'best_epoch': checkpoint['epoch']
        }
    }

    with open(os.path.join(RESULTS_DIR, 'phase3_enhanced_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    print(f"\n✓ Saved results: {RESULTS_DIR}/phase3_enhanced_results.pkl")

    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(history['train_loss'], label='Train')
    ax.plot(history['val_loss'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Enhanced Model Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(history['val_r_error'], label='R Error')
    ax.plot(history['val_x_error'], label='X Error')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error (%)')
    ax.set_title('Enhanced Model Validation Errors')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'phase3_enhanced_training.png'), dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot: {RESULTS_DIR}/phase3_enhanced_training.png")

    print("\n" + "=" * 70)
    print("✓ PHASE 3 ENHANCED TRAINING COMPLETE!")
    print("=" * 70)

    if HAS_SWANLAB:
        swanlab.finish()


if __name__ == '__main__':
    main()
