# Phase 2 Setup: Train Graph Mamba - READY

## Overview

Phase 2 trains Graph Mamba on the same steady-state scenario as Phase 1 IAUKF for fair comparison.

## Scripts Created

### 1. Data Generation: `phase2_generate_data.py`
**Purpose**: Generate and save dataset once (avoid regenerating each training run)

**Features**:
- ✓ Progress bars with `tqdm`
- ✓ Efficient tensor creation with `torch.from_numpy()`
- ✓ Matches Phase 1 scenario exactly (constant loads)
- ✓ Same noise levels (SCADA 0.02, PMU 0.005/0.002)
- ✓ Saves to disk as pickle files

**Configuration**:
```python
NUM_TRAIN_EPISODES = 800     # 80% training
NUM_VAL_EPISODES = 100       # 10% validation
NUM_TEST_EPISODES = 100      # 10% test
STEPS_PER_EPISODE = 200      # Match IAUKF
```

**Output**: `data/phase2/{train,val,test}_data.pkl`

**Usage**:
```bash
python experiments/phase2_generate_data.py
```

### 2. Training: `phase2_train_mamba.py`
**Purpose**: Load saved data and train Graph Mamba

**Features**:
- ✓ Loads pre-generated data (fast startup)
- ✓ Progress bars for training/validation
- ✓ SwanLab integration for logging
- ✓ Model checkpointing (saves best model)
- ✓ Learning rate scheduling
- ✓ Comprehensive evaluation
- ✓ Automatic comparison with IAUKF baseline

**Configuration**:
```python
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-3
D_MODEL = 64
LAMBDA_PHY = 0.01
```

**Output**:
- `checkpoints/graph_mamba_phase2_best.pt` - Best model
- `tmp/phase2_training_history.png` - Training curves
- `tmp/phase2_results.pkl` - Final results

**Usage**:
```bash
python experiments/phase2_train_mamba.py
```

## Workflow

### Step 1: Generate Dataset (One-time, ~10-20 minutes)
```bash
cd /data1/xh/workspace/power/iaukf
conda activate graphmamba
python experiments/phase2_generate_data.py
```

Expected output:
- 1000 total episodes (800 train + 100 val + 100 test)
- Each episode: 200 timesteps × 33 nodes × 3 features
- Saved to `data/phase2/` (~100-200 MB total)

### Step 2: Train Model (GPU recommended, ~30-60 minutes)
```bash
python experiments/phase2_train_mamba.py
```

Expected output:
- Training progress with live metrics
- Best model checkpoint
- Training history plots
- Comparison with IAUKF

## Expected Results

### Target Performance (matching IAUKF Phase 1):
- **R error**: < 3% (IAUKF: 1.60%)
- **X error**: < 3% (IAUKF: 2.00%)

### Success Criteria:
- ✓ Converges within 100 epochs
- ✓ Test error < 5% for both parameters
- ✓ Comparable or better than IAUKF

## Key Improvements Over Original Code

1. **Split data generation and training**
   - Generate once, train multiple times
   - Faster iteration during hyperparameter tuning

2. **Progress bars everywhere**
   - Data generation progress
   - Training/validation progress
   - Clear visual feedback

3. **Efficient tensor creation**
   - Use `torch.from_numpy()` instead of `torch.tensor()`
   - Avoids slow list-to-tensor conversion
   - ~10x faster data loading

4. **Better organization**
   - Clear separation of concerns
   - Easy to modify hyperparameters
   - Comprehensive logging

## Hardware Requirements

**Minimum**:
- CPU: Any modern CPU
- RAM: 8 GB
- Storage: 500 MB
- Time: 2-3 hours (CPU training)

**Recommended**:
- GPU: NVIDIA GPU with 6+ GB VRAM (e.g., RTX 2060)
- RAM: 16 GB
- Storage: 1 GB
- Time: 30-60 minutes (GPU training)

**Your Setup** (4×4090 24GB):
- ✓✓✓ Excellent! Will train very fast
- Consider increasing batch size to 32 or 64
- Can train multiple models in parallel

## Troubleshooting

### Issue: Out of Memory
**Solution**: Reduce `BATCH_SIZE` in training script

### Issue: Training too slow
**Solutions**:
- Ensure GPU is being used: Check "Using device: cuda"
- Reduce `STEPS_PER_EPISODE` for faster iterations
- Use fewer training episodes for quick testing

### Issue: Poor convergence
**Solutions**:
- Train for more epochs
- Adjust learning rate (try 5e-4 or 2e-3)
- Tune `D_MODEL` (try 32, 64, or 128)
- Check data quality in saved files

## Next Steps

After Phase 2 completes successfully:

**Phase 3**: Time-varying parameters
- Implement dynamic parameter changes
- Test Graph Mamba's tracking ability
- Show advantages over IAUKF

## Quick Start Commands

```bash
# Navigate to project
cd /data1/xh/workspace/power/iaukf

# Activate environment
conda activate graphmamba

# Generate data (one-time)
python experiments/phase2_generate_data.py

# Train model
python experiments/phase2_train_mamba.py

# If you want to retrain with different hyperparameters:
# Just modify phase2_train_mamba.py and run again
# (data is already saved, so it's fast!)
```

## Monitoring Training

If SwanLab is installed:
1. Training will automatically log to SwanLab
2. View real-time metrics in browser
3. Project: "power-grid-iaukf"
4. Experiment: "Phase2_GraphMamba_[timestamp]"

Without SwanLab:
1. Watch terminal output for progress
2. Check saved plots in `tmp/`
3. Final results printed at end

---

**Status**: Ready to run! ✓
**Estimated time**: 40-80 minutes total (10-20 min data gen + 30-60 min training)
