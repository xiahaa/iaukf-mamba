# Physics-Informed Graph Mamba for Power Grid State Estimation

A comprehensive implementation and research framework for augmented state estimation in active distribution networks, combining the traditional IAUKF pipeline with a data-driven Graph Mamba counterpart.

---

## Table of Contents

- [Reproduce: IAUKF Implementation](#reproduce-iaukf-implementation)
- [Improvement: Research Plan](#improvement-research-plan)
- [Graph Mamba Implementation](#graph-mamba-implementation)
- [Training Pipeline](#training-pipeline)
- [Benchmark Experiments](#benchmark-experiments)
- [Expected Results](#expected-results)
- [Additional Merit Features](#additional-merit-features)

---

## Reproduce: IAUKF Implementation

### Solution Components

- **`simulation.py`**: Simulates the IEEE 33-bus system and generates noisy SCADA/PMU data.
- **`models.py`**: Defines the Augmented State Vector and the Physics models (Power Flow & Transition).
- **`iaukf.py`**: Implements the Improved Adaptive Unscented Kalman Filter with the Noise Statistic Estimator (NSE).
- **`main.py`**: Orchestrates the estimation loop and plots states and parameter estimates.

### Implementation Summary

1. **Simulation**
   - Generated synthetic data on the IEEE 33-bus system using `pandapower`.
   - Target line (Bus 3-4) parameters are tracked, with the filter seeded at a 50% error to showcase convergence.

2. **Physics Models**
   - `DistributionSystemModel` updates the pandapower net's line parameters inside `measurement_function` so the filter detects power-flow mismatch due to wrong parameters.
   - This dynamic update lets the estimator correct the physics when residuals spike.

3. **IAUKF with NSE**
   - Implements Adaptive Noise Statistic Estimator (Eq. 17/18) to infer the process covariance.
   - Dynamically inflates \(Q\) to nudge the static parameter toward the truth when residuals are high, then tightens it once the system stabilizes.
   - Addresses the paradox that static parameters theoretically have zero process noise.

### How to Run

```bash
python main.py
```

**Expected Behavior**: The Estimated R and Estimated X curves start from distorted initials and converge toward the red dashed True lines within ~50–100 steps, proving the effectiveness of the augmented state filter.

---

## Improvement: Research Plan

A comprehensive plan to elevate the "Augmented State Estimation" paper into a high-impact scientific publication leveraging Graph Mamba.

### The "Hitting Point" (Scientific Breakthrough)

The baseline IAUKF is effective but limited by:

1. **Linearization/Sigma-point assumptions** that lose fidelity in volatile grids.
2. **Heuristic adaptation** (NSE) that does not learn complex drift patterns or sensor bias.
3. **Short memory**; Kalman filters are Markovian and cannot distinguish slow aging vs. sudden anomalies.

**Breakthrough Idea:** *Physics-Informed Graph Mamba (PI-GraphMamba)*

- **Theoretical bridge:** Kalman Filters stem from State Space Models, and Mamba is essentially a learned, discretized SSM with linear-time complexity.
- **Efficiency:** Mamba runs in \(O(N)\) time with an infinite context window, unlike quadratic Transformers.
- **Spatial awareness:** A GNN encoder handles the grid’s non-Euclidean topology, enabling instant adaptation to topology perturbations.

### Research Plan: Physics-Informed Graph Mamba for Parameter Calibration

**Title Proposal:** *Physics-Informed Graph Mamba: A Linear-Complexity State Space Model for Robust Joint Parameter and State Estimation in Active Distribution Grids*

#### Phase 1 – Architecture (Graph Mamba)

- **Spatial Layer:** Replace the UKF’s vector input with a Graph Neural Network or Graph Transformer. Inputs are node features \((P, Q, |V|)\) and edge features \((R_{initial}, X_{initial})\).
- **Temporal Layer:** Feed the node embeddings through a Mamba block (or LSTM fallback) that learns transitions \(f(x)\) and uses Selection to ignore transient noise while remembering persistent trends.

#### Phase 2 – Physics-Informed Training

- **Hybrid Loss:** \(Loss = L_{MSE} + \lambda \cdot L_{Physics}\).
- **Physics term:** Compare measured SCADA power \(S_{measured}\) against the power flow computed with \(\hat{V}, \hat{\theta}, \hat{R}, \hat{X}\) to enforce explainability.

#### Phase 3 – Robustness Stress Tests

- Add topology-awareness experiments (line trips) to show IAUKF divergence vs. Graph Mamba’s fast adaptation via its Selection mechanism.

#### Phase 4 – Benchmarking Evidence

- **Baselines:** IAUKF, an LSTM-GNN, and a Transformer.
- **Metrics:** RMSE of \(R\) and \(X\), inference efficiency, and long-term stability over 10,000+ steps.

---

## Graph Mamba Implementation

- `graph_mamba.py` fuses a GNN encoder with a selective Mamba (or LSTM) temporal block and a physics-aware loss.
- Nodes embed local power states, edges carry the prior line parameters, and the Selection gate decides which historical context to retain.
- The module gracefully falls back to LSTM when `mamba_ssm` is unavailable, keeping the repository runnable on CPU-only machines.

## Training Pipeline

- `train_mamba.py` generates episodes (e.g., 50 × 200 steps), converts the pandapower net into PyTorch Geometric graphs, and trains `GraphMambaModel` to regress true \(R, X\) from noisy SCADA data.
- Validation RMSE is monitored, and the checkpoint is saved for benchmarking.

## Benchmark Experiments

`benchmark.py` compares IAUKF and Graph Mamba on held-out episodes:

1. Load IAUKF and the trained Graph Mamba checkpoint (or retrain quickly).
2. Evaluate on test episodes with varying noise.
3. Record RMSE for \(R\) and \(X\), convergence time to 95% accuracy, and robustness under noise.
4. Generate plots: time-series tracking, error box plots, and RMSE vs. noise.
5. Print a LaTeX-ready table summarizing mean error and standard deviation per method.

## Expected Results

- **Numerical table:** IAUKF gradually improves, while Graph Mamba stays tight around the truth with lower variance.
- **Tracking plot (`benchmark_tracking.png`):** IAUKF decays from its initial bias; Graph Mamba stays near truth from step 0.
- **Error box plot (`benchmark_boxplot.png`):** IAUKF’s distribution is wider than Graph Mamba’s compact band.
- **Inference notes:** Mamba is \(O(N)\) and GPU-friendly; IAUKF remains sequential and CPU-bound.

### Example Output

```text
Method       Parameter   Mean Error   Std Dev
---------------------------------------------
IAUKF        R           0.01524      0.04210
IAUKF        X           0.02105      0.05100
GraphMamba   R           0.00450      0.00120  <-- Lower Error & Std Dev
GraphMamba   X           0.00510      0.00150
```

### Troubleshooting

- **IAUKF diverges:** Increase the NSE-driven \(Q\) update or inflate \(R_{cov}\).
- **Graph Mamba collapses:** Train longer or diversify \(R_{true}\) values so the network learns the full range rather than constant means.

These outputs support the “Traditional vs. AI” narrative: Graph Mamba is faster, more robust, and smoother, while IAUKF remains theoretically grounded but noisier.

---

## Additional Merit Features

Stress tests that highlight Graph Mamba’s resilience:

1. **Context Shift (Topology Change):** Trip line 6-26 at \(t=100\) and observe IAUKF spike while Graph Mamba recovers instantly thanks to the GNN reading the new adjacency.
2. **Non-Gaussian / Heavy-Tail Noise:** Inject 5× impulse spikes into 10% of measurements; Graph Mamba’s Selection gate ignores spikes while IAUKF oscillates.
3. **Observability Limits (Missing Data):** Mask 30–50% of SCADA entries; Graph Mamba imputes missing pieces from neighbors/history, while IAUKF may diverge.

### Implementing the Stress Tests

- Add a `topology_change=True` flag to `simulation.py` that modifies the pandapower net mid-episode.
- Extend `benchmark.py` with a `run_stress_test()` helper that injects spikes and masks parts of `z_scada`.
- Plot these stress tests to show Graph Mamba staying flat while IAUKF oscillates.

These experiments become the “Scientific Breakthrough” section: **Superior resilience to topology shifts and non-Gaussian noise.**
