# Graph-Mamba vs IAUKF Experimental Suite

This folder contains comprehensive experiments comparing Graph-Mamba (deep learning) against IAUKF (Improved Adaptive Unscented Kalman Filter) for power system parameter estimation.

## Experiment Priority

### P0: Must-Have (Core Results)

| # | Experiment | Purpose | Expected Result |
|---|-----------|---------|-----------------|
| 1 | `exp1_basic_performance.py` | Basic accuracy comparison | Graph-Mamba < 0.1% vs IAUKF ~0.18% |
| 2 | `exp2_dynamic_tracking.py` | Time-varying parameter tracking | Faster tracking, less lag |
| 4 | `exp4_speed_comparison.py` | Computational efficiency | 5×+ speedup on large systems |

### P1: Important (Key Differentiators)

| # | Experiment | Purpose | Expected Result |
|---|-----------|---------|-----------------|
| 3 | `exp3_low_observability.py` | Sparse PMU measurements | Graph-Mamba works, IAUKF fails |
| 5 | `exp5_robustness.py` | Non-Gaussian noise & bad data | Better robustness |

### P2: Optional (Nice-to-Have)

| # | Experiment | Purpose | Expected Result |
|---|-----------|---------|-----------------|
| 6 | `exp6_generalization.py` | Cross-topology transfer | Some transfer capability |

## Quick Start

### Prerequisites

```bash
# Activate conda environment
conda activate graphmamba

# Ensure SwanLab is configured
swanlab login  # Already done
```

### Run All Experiments

```bash
# Run all experiments (P0, P1, P2)
python run_all_experiments.py

# Run only P0 (must-have) experiments
python run_all_experiments.py --p0-only

# Skip P2 experiments
python run_all_experiments.py --skip P2

# Run specific experiment
python run_all_experiments.py --exp exp1
```

### Run Individual Experiments

```bash
# Experiment 1: Basic Performance
python exp1_basic_performance.py

# Experiment 2: Dynamic Tracking
python exp2_dynamic_tracking.py

# Experiment 3: Low Observability
python exp3_low_observability.py

# Experiment 4: Speed Comparison
python exp4_speed_comparison.py

# Experiment 5: Robustness
python exp5_robustness.py

# Experiment 6: Generalization
python exp6_generalization.py
```

## Results

Results are saved in `results/`:
- `exp1_results.pkl` - Raw data
- `exp1_basic_performance.png` - Figures
- Logs tracked in SwanLab

## Expected Timeline

| Experiment | Estimated Time |
|-----------|----------------|
| exp1 | ~5 minutes |
| exp2 | ~10 minutes |
| exp3 | ~8 minutes |
| exp4 | ~3 minutes |
| exp5 | ~10 minutes |
| exp6 | ~5 minutes |
| **Total** | **~40 minutes** |

## Key Metrics

### Accuracy
- **R Error**: Percentage error in resistance estimation
- **X Error**: Percentage error in reactance estimation
- Target: < 0.1% for Graph-Mamba vs ~0.18% for IAUKF

### Speed
- **Inference Time**: ms per estimation
- **Scaling**: O(n) vs O(n³)
- Target: 5×+ speedup at 118 buses

### Robustness
- **Convergence Rate**: % of successful runs
- **Bad Data Tolerance**: Error under outlier corruption

## Paper Narrative

Organize results to tell this story:

1. **"Better"** (Exp 1): Graph-Mamba is more accurate
2. **"Faster"** (Exp 4): Graph-Mamba is more scalable
3. **"Stronger"** (Exp 2, 3, 5): Graph-Mamba is more robust

## Troubleshooting

### Experiment fails to run
- Check conda environment: `conda activate graphmamba`
- Check dependencies: `pip list | grep -E "torch|pandapower"`
- Check model checkpoints exist in `../checkpoints/`

### Poor Graph-Mamba performance
- Model may need retraining on your data distribution
- Check checkpoint compatibility
- Try using phase3 checkpoint for time-varying scenarios

### SwanLab not logging
- Check login: `swanlab login`
- Check project name matches

## Citation

If using these experiments, cite:

```bibtex
@article{graphmamba2024,
  title={Graph-Mamba for Power System Parameter Estimation},
  author={...},
  journal={...},
  year={2024}
}
```

## Contact

For questions about the experimental framework, refer to the main project AGENTS.md.
