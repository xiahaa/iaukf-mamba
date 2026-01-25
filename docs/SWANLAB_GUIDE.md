# SwanLab Integration Guide for Power Grid Estimation

## Installation

```bash
pip install swanlab
```

## Usage

SwanLab is now integrated into `train_mamba.py`. It will automatically log:

- Training and validation losses
- Learning rate changes
- Model hyperparameters
- Sample predictions
- Training curves

## Configuration

In `train_mamba.py`, you can configure SwanLab:

```python
# SwanLab Configuration
USE_SWANLAB = True  # Set to False to disable
SWANLAB_PROJECT = "power-grid-estimation"
SWANLAB_EXPERIMENT = "graph-mamba-training"
```

## Running with SwanLab

```bash
python train_mamba.py
```

The logs will be saved to `./swanlog/` directory.

## View Results

```bash
# Start SwanLab web UI
swanlab watch

# Or view online at https://swanlab.cn/
```

## What Gets Logged

### Every Epoch:
- `train/loss`: Training loss
- `train/mse`: Training MSE
- `train/physics_loss`: Physics-informed loss component
- `val/loss`: Validation loss
- `val/mse`: Validation MSE
- `train/learning_rate`: Current learning rate
- `epoch`: Epoch number

### Best Model:
- `best_val_loss`: Best validation loss achieved

### Final Summary:
- `training_curve`: Plot of train/val losses
- `sample_results/mean_r_error`: Mean R parameter error
- `sample_results/mean_x_error`: Mean X parameter error

## Example Dashboard

After training, you'll see:
1. Loss curves (train vs val)
2. Hyperparameter tracking
3. Sample prediction errors
4. Learning rate schedule
5. Model performance metrics

## Advanced Usage

### Compare Multiple Runs

```bash
# Run with different hyperparameters
python train_mamba.py  # Run 1
# Modify hyperparameters in code
python train_mamba.py  # Run 2

# View comparisons in SwanLab dashboard
swanlab watch
```

### Custom Logging

Add your own metrics in `train_mamba.py`:

```python
if USE_SWANLAB:
    swanlab.log({
        "custom_metric": value,
        "custom_plot": swanlab.Image("plot.png"),
    })
```

## Disabling SwanLab

If you don't want to use SwanLab, set:

```python
USE_SWANLAB = False
```

Or if swanlab is not installed, the code will automatically skip logging.
