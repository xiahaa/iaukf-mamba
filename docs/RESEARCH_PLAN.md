# Research Implementation Roadmap

## Overview

Three-phase approach to validate IAUKF and demonstrate Graph Mamba superiority:

**Phase 1**: Reproduce paper's IAUKF (validate implementation)
**Phase 2**: Train Graph Mamba in same scenario (fair comparison)
**Phase 3**: Time-varying parameters (show Graph Mamba advantages)

---

## ~~Phase 1: Reproduce Paper's IAUKF Work~~

### Goal
Validate our IAUKF implementation by reproducing paper's results.

### Scenario Setup (Match Paper)
- **System**: IEEE 33-bus
- **Loads**: Constant (steady-state)
- **Target**: Single line parameter estimation (line 3-4)
- **Measurements**: SCADA (P, Q, V) + PMU (V, θ) at 12 buses
- **Initial error**: 50% (R₀=0.5×R_true, X₀=0.5×X_true)
- **Processing**: Off-line (multiple snapshots if needed)

### Implementation Tasks

#### 1.1 Update Simulation
```python
# model/simulation.py
- Set constant loads (no fluctuations)
- Generate steady-state measurements
- Add multiple snapshot collection option
```

#### 1.2 Keep Augmented State Model
```python
# model/models.py
- State: [V₁...V₃₃, δ₁...δ₃₃, R, X] (68 dims)
- Use Holt's exponential smoothing (optional, can compare)
- Keep current implementation first, add Holt's later
```

#### 1.3 Tune IAUKF Parameters
```python
# main.py
- P₀: Initial covariance (paper might specify)
- Q: Process noise (small for parameters)
- R: Measurement noise (from sensor specs)
- NSE parameters: b_factor, alpha, beta, kappa
```

#### 1.4 Metrics & Validation
- Convergence rate (steps to <5% error)
- Final accuracy (RMSE, MAE, MAPE)
- Compare with paper's reported results
- Generate plots matching paper's figures

### Expected Results
- Convergence in 50-100 steps
- Final error: 2-5% (noise-limited)
- Validates our IAUKF is correct ✓

---

## Phase 2: Train Graph Mamba (Same Scenario)

### Goal
Fair comparison: Graph Mamba vs IAUKF under paper's conditions.

### Dataset Generation
```python
# Use same scenario as Phase 1
- Constant loads (steady-state)
- Same measurement noise levels
- Generate large dataset: 10,000 episodes
- Each episode: 100 timesteps
- Train/Val/Test split: 70/15/15
```

### Graph Mamba Training
```python
# train_mamba.py
- Input: Measurement sequences + graph structure
- Output: Parameter estimates
- Loss: MSE + Physics-informed loss
- Training: 50-100 epochs
- Use SwanLab for logging
```

### Evaluation Metrics (Same as IAUKF)
- Convergence rate
- Final accuracy (RMSE, MAE, MAPE)
- Inference time
- Robustness to noise

### Expected Results
- Similar or better accuracy than IAUKF
- Faster inference (no power flow iterations)
- More robust to noise (learned from data)

---

## Phase 3: Time-Varying Parameters (Graph Mamba Advantage)

### Goal
Show Graph Mamba excels when parameters change dynamically.

### Scenario: Occasional Parameter Changes

**Realistic scenario**: Line aging, temperature effects, conductor sag

```python
# New simulation mode
def generate_time_varying_parameters():
    """
    Parameters change occasionally, not constantly.

    Example pattern:
    - Steps 0-50: R=0.38, X=0.19 (baseline)
    - Steps 51-100: R=0.42, X=0.21 (heating effect, +10%)
    - Steps 101-150: R=0.36, X=0.18 (cooling, -5%)
    - Steps 151-200: R=0.40, X=0.20 (aging, +5%)
    """
    changes = [
        (0, 1.0),      # Baseline
        (50, 1.1),     # +10% heating
        (100, 0.95),   # -5% cooling
        (150, 1.05),   # +5% aging
    ]
    return changes
```

### Implementation

#### 3.1 Enhanced Simulation
```python
# model/simulation_dynamic.py
class DynamicParameterSimulation:
    def __init__(self, parameter_change_schedule):
        self.change_schedule = parameter_change_schedule

    def run_simulation(self):
        # Apply parameter changes at scheduled times
        # Generate measurements with dynamic parameters
```

#### 3.2 IAUKF with Dynamic Parameters
```python
# Test IAUKF ability to track changes
- Challenge: Q tuning becomes critical
- Too small Q: Slow to track changes
- Too large Q: Oscillations
- Adaptive Q helps but has lag
```

#### 3.3 Graph Mamba with Dynamic Parameters
```python
# Train on time-varying data
- Dataset includes parameter change episodes
- Model learns to detect and track changes
- No Q tuning needed!
```

### Evaluation Scenarios

**Scenario 3A: Slow Changes** (realistic)
- Parameter changes every 50 timesteps
- Change magnitude: ±5-10%
- Metric: Tracking lag, steady-state error

**Scenario 3B: Fast Changes** (stress test)
- Parameter changes every 10 timesteps
- Change magnitude: ±10-20%
- Metric: Adaptation speed, overshoot

**Scenario 3C: With Load Variations** (most realistic)
- Parameter changes occasionally
- Loads vary ±10% continuously
- Metric: Overall RMSE, robustness

### Expected Results

| Metric | IAUKF | Graph Mamba |
|--------|-------|-------------|
| Steady-state error | 2-5% | 1-3% |
| Tracking lag | 10-20 steps | 2-5 steps |
| Sensitivity to Q | High ⚠ | None ✓ |
| Works with load variations | Limited ⚠ | Yes ✓ |
| Computational time | Slow (power flow) | Fast (forward pass) |

---

## Implementation Timeline

### Week 1: Phase 1 - IAUKF Validation
- [ ] Day 1-2: Update simulation for constant loads
- [ ] Day 3-4: Tune IAUKF, reproduce paper results
- [ ] Day 5: Documentation and validation plots

### Week 2: Phase 2 - Graph Mamba Training
- [ ] Day 1-2: Generate large dataset (steady-state)
- [ ] Day 3-5: Train Graph Mamba, hyperparameter tuning
- [ ] Day 6-7: Evaluation and comparison with IAUKF

### Week 3: Phase 3 - Dynamic Scenarios
- [ ] Day 1-2: Implement time-varying parameter simulation
- [ ] Day 3-4: Test IAUKF on dynamic scenarios
- [ ] Day 5-6: Generate dynamic dataset and train Graph Mamba
- [ ] Day 7: Comprehensive comparison and plots

---

## Code Structure

```
iaukf/
├── model/
│   ├── simulation.py              # Phase 1: Constant loads
│   ├── simulation_dynamic.py      # Phase 3: Time-varying params
│   ├── models.py                  # IAUKF model (augmented state)
│   └── iaukf.py                   # IAUKF algorithm
├── graphmamba/
│   ├── graph_mamba.py            # Graph Mamba model
│   └── train_mamba.py            # Training script
├── experiments/
│   ├── phase1_validate_iaukf.py  # Reproduce paper
│   ├── phase2_train_mamba.py     # Fair comparison
│   └── phase3_dynamic_params.py  # Dynamic scenarios
├── benchmark.py                   # Unified comparison
└── docs/
    ├── PAPER_ANALYSIS.md          # Paper review
    ├── SOLUTION.md                # IAUKF fixes
    └── RESEARCH_PLAN.md           # This document
```

---

## Key Deliverables

### 1. Validation Report
- "Our IAUKF reproduces paper's results within X% accuracy"
- Comparison plots (convergence, accuracy)

### 2. Fair Comparison Results
- Table comparing IAUKF vs Graph Mamba (steady-state)
- Show Graph Mamba achieves comparable or better performance

### 3. Dynamic Scenario Results
- Plots showing tracking performance
- Highlight Graph Mamba's advantages:
  - Faster adaptation
  - No tuning needed
  - Works with load variations

### 4. Research Paper Sections

**Section IV: Baseline Validation**
- Reproduce IAUKF results
- Establish credibility

**Section V: Proposed Method**
- Graph Mamba architecture
- Training methodology

**Section VI: Comparative Analysis**
- Phase 2 results (fair comparison)
- Show performance parity/superiority

**Section VII: Dynamic Scenarios**
- Phase 3 results (key contribution!)
- Show Graph Mamba advantages

---

## Success Criteria

### Phase 1 Success ✓
- [ ] IAUKF converges in steady-state scenario
- [ ] Results match paper (within 10% of reported accuracy)
- [ ] Validates implementation correctness

### Phase 2 Success ✓
- [ ] Graph Mamba achieves ≤5% error (similar to IAUKF)
- [ ] Faster inference time than IAUKF
- [ ] Shows ML approach is viable

### Phase 3 Success ✓
- [ ] Graph Mamba outperforms IAUKF in dynamic scenarios
- [ ] Faster tracking (2-3× improvement)
- [ ] More robust (no Q tuning needed)
- [ ] Works with load variations (IAUKF struggles)

---

## Research Contribution Statement

**This work demonstrates that while traditional IAUKF works well for off-line parameter calibration under steady-state conditions, Graph Mamba excels in realistic scenarios with:**

1. **Time-varying parameters** (occasional changes)
2. **Unknown load dynamics** (continuous variations)
3. **No manual tuning** (end-to-end learning)
4. **Real-time capability** (fast inference)

**This bridges the gap between theoretical state estimation and practical deployment in modern smart grids with high dynamics and uncertainties.**

---

## Next Steps

Ready to start Phase 1? I can:

1. **Update simulation.py** for constant loads
2. **Create phase1_validate_iaukf.py** with proper evaluation
3. **Tune parameters** to match paper's results
4. **Generate comparison plots**

Let me know when you're ready to proceed! 🚀
