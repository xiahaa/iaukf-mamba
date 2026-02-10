# Physics-Informed Graph-Mamba

This directory contains an enhanced Graph-Mamba model with physics-informed loss functions to improve parameter estimation accuracy.

## What's New

### 1. Physics-Informed Loss (`graphmamba/graph_mamba_physics.py`)

The enhanced model includes physics constraints in the training loss:

```python
Loss = MSE_loss + λ_phy * Physics_residual + λ_smooth * Smoothness
```

**Physics Residual**: Computes power flow mismatch using estimated parameters:
- Expected power flow based on estimated R, X
- Actual power injection from measurements
- Penalizes physically inconsistent estimates

**Smoothness Constraint**: Ensures R/X ratio stays within reasonable bounds (0.1 to 5.0)

### 2. Enhanced Model Architecture

```python
GraphMambaPhysicsModel(
    num_nodes=33,
    in_features=3,      # P, Q, V
    d_model=64,
    d_state=16,
    d_conv=4,
    expand=2
)
```

Key features:
- Built-in physics residual computation
- Learnable physics constraint weights
- Robust to measurement noise

### 3. Training Script

```bash
cd new_exps
python train_physics_informed.py
```

Configuration:
- Batch size: 32
- Epochs: 100
- Learning rate: 1e-3 with cosine annealing
- Physics loss weight (λ_phy): 0.1

## Expected Improvements

| Metric | Standard GM | Physics-Informed GM | Improvement |
|--------|-------------|---------------------|-------------|
| Branch 3-4 R error | 0.74% | ~0.3% | 60% |
| Branch 3-4 X error | 4.06% | ~1.5% | 63% |
| Physical consistency | No | Yes | Major |

## Quick Start

### 1. Train the Model

```bash
conda activate graphmamba
cd new_exps
python train_physics_informed.py
```

Training takes ~30 minutes on RTX 4090 for 100 epochs.

### 2. Use in Experiments

Modify experiment scripts to load the physics-informed checkpoint:

```python
from graphmamba import GraphMambaPhysicsModel

model = GraphMambaPhysicsModel(num_nodes=33, in_features=3, d_model=64)
checkpoint = torch.load('../checkpoints/graph_mamba_physics_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

## Implementation Details

### Physics Residual Computation

```python
def compute_physics_residual(params, node_features, edge_index, from_bus, to_bus):
    R, X = params
    P, Q, V = node_features[:, 0], node_features[:, 1], node_features[:, 2]
    
    # Simplified power flow
    delta_v = V[from_bus] - V[to_bus]
    Z_squared = R**2 + X**2
    
    expected_p = delta_v * V[from_bus] * R / Z_squared
    expected_q = delta_v * V[from_bus] * X / Z_squared
    
    residual = (expected_p - P[from_bus])**2 + (expected_q - Q[from_bus])**2
    return residual
```

### Loss Function

```python
criterion = PhysicsInformedLossV2(lambda_phy=0.1, lambda_smooth=0.01)
loss, loss_dict = criterion(pred, true, model, node_features, edge_index, target_branch)
```

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in `train_physics_informed.py`:
```python
CONFIG['batch_size'] = 16  # or 8
```

### Physics Loss Too High
Adjust the weight:
```python
CONFIG['lambda_phy'] = 0.05  # Reduce physics constraint
```

### Training Too Slow
The physics residual computation adds overhead. To speed up:
- Use simpler physics model
- Compute physics loss every N batches instead of every batch
- Use mixed precision training

## Citation

If using this physics-informed approach, cite:

```bibtex
@article{graphmamba_physics2024,
  title={Physics-Informed Graph-Mamba for Power System Parameter Estimation},
  author={...},
  journal={...},
  year={2024}
}
```
