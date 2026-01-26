# IAUKF Convergence Issue - Root Cause Analysis

## Executive Summary

The IAUKF does not converge because of a **fundamental observability problem**: we're trying to estimate states (voltages, angles, parameters) from measurements, but the problem is **under-determined**.

## Diagnostic Results

### Test 1: Noise-Free Measurements
- ✗ Filter diverges (parameters go negative)
- Conclusion: NOT a noise/tuning issue

### Test 2: Measurement Function Consistency
- ✗ ~3% error even with true parameters
- Conclusion: Model mismatch

### Test 3: Parameters-Only Estimation
- Still oscillates
- Conclusion: Load uncertainty is the issue

### Test 4: With Load Inference
- Still ~2% measurement error
- Still oscillates
- Conclusion: Nonlinear power flow creates ambiguity

## Root Cause: The Power Flow Equations

The power flow equations are:
```
P_i = V_i * Σ V_j * (G_ij * cos(θ_i - θ_j) + B_ij * sin(θ_i - θ_j))
Q_i = V_i * Σ V_j * (G_ij * sin(θ_i - θ_j) - B_ij * cos(θ_i - θ_j))
```

Where `G_ij` and `B_ij` depend on line parameters (R, X).

### The Problem

Given measurements {P, Q, V, θ} and trying to estimate {R, X}:

1. **Multiple solutions exist**: Different combinations of {V, θ, R, X, P_load, Q_load} can produce the same measurements
2. **Slack absorption**: The slack bus absorbs errors, making the system under-determined
3. **Nonlinearity**: Small errors compound through the nonlinear equations

## Why Standard IAUKF Won't Work

The standard formulation tries to estimate:
- State: [V₁...V_n, θ₁...θ_n, R, X] (68 variables)
- From: [P₁...P_n, Q₁...Q_n, V₁...V_n, θ_pmu] (123 measurements)

But:
- Voltages and angles are **outputs** of power flow, not independent states
- They're determined by loads and line parameters
- Estimating them as states creates circular dependencies

## The Correct Approach

### Option 1: Parameter-Only with Known Loads (Simplest)

**Assumption**: Loads are measured or known

```python
State: [R, X]  # Only 2 variables
Measurements: [P, Q, V, θ]  # 123 values

Process:
1. Measure or estimate loads at each timestep
2. Run power flow with current loads and estimated parameters
3. Compare predicted measurements to actual
4. Update parameter estimates
```

**Pros:**
- Observability is clear
- Only 2 parameters to estimate
- Standard UKF works

**Cons:**
- Requires load measurements/estimation
- In your simulation, loads change randomly

### Option 2: Augmented State with Load Dynamics (Complex)

**Assumption**: Loads have slow dynamics

```python
State: [R, X, P_load_1...P_load_n, Q_load_1...Q_load_n]

Process model:
- R_k+1 = R_k  (constant)
- X_k+1 = X_k  (constant)
- P_load_k+1 = P_load_k + process_noise  (random walk)
```

**Pros:**
- Jointly estimates parameters and loads
- Theoretically complete

**Cons:**
- 66 state variables
- Observability issues
- Computationally expensive
- Requires careful tuning

### Option 3: Two-Stage Estimation (Practical)

**Stage 1**: Estimate loads from measurements
```python
# At each timestep
P_load_i = -P_inj_i  # For load buses
Q_load_i = -Q_inj_i
```

**Stage 2**: Estimate parameters given loads
```python
State: [R, X]
Run UKF with inferred loads
```

**Pros:**
- Separates concerns
- Reduces state dimension
- Easier to debug

**Cons:**
- Load estimation errors propagate
- Still has ~2% error we observed

## Why the 2-3% Error Persists

Even with load inference, we have error because:

1. **Slack bus**: In power flow, one bus (usually bus 0) is the slack bus that balances power. Its injection is NOT measured directly but computed residually.

2. **Network losses**: `Σ P_load ≠ Σ P_inj` because of line losses. The difference is absorbed by slack bus.

3. **Reactive power**: Q flow is complex and depends on voltage profile, which changes with line parameters.

4. **Numerical tolerances**: Power flow solver has convergence tolerance (~1e-8), creating small differences.

## Recommended Solutions

### Solution A: Use Quasi-Static State Estimation (Recommended)

Instead of UKF, use classical power system state estimation:

```python
# Weighted Least Squares State Estimation
1. Fix loads (known or estimated)
2. Estimate voltage state [V, θ] from measurements
3. Given voltage state, estimate parameters [R, X]
```

This is the standard approach in power systems!

### Solution B: Simplify Simulation (For Testing)

Make simulation match assumptions:

```python
# In simulation.py
# Option 1: No load fluctuations
fluctuation = np.ones(len(p_load_base))  # Constant loads

# Option 2: Known load fluctuations
# Save fluctuation values and pass to filter
```

Then filter knows exact loads to use.

### Solution C: Steady-State Snapshots (Practical)

Only estimate parameters during steady-state:

```python
# Wait for N timesteps
# Average measurements over window
# Estimate parameters from averaged data
# This reduces noise and dynamic effects
```

## My Recommendation

**For your research:**

1. **Short-term (Get it working)**:
   - Modify simulation to use constant loads (no fluctuations)
   - Use parameters-only state [R, X]
   - This should converge!

2. **Medium-term (More realistic)**:
   - Implement classical WLS state estimation first
   - Then add UKF for time-varying parameters
   - This is standard practice

3. **Long-term (Research contribution)**:
   - Use Graph Mamba to learn the mapping directly:
     `Measurements → Parameters`
   - Skip the power flow modeling entirely
   - This is where your ML approach shines!

## Code Changes Needed

### For Quick Fix:

In `model/simulation.py`, line ~47:
```python
# Change this:
fluctuation = np.random.uniform(0.9, 1.1, size=len(p_load_base))

# To this:
fluctuation = np.ones(len(p_load_base))  # Constant loads
```

Then IAUKF should converge because:
- Loads are constant and known
- No model mismatch
- Parameters are observable

### Test

Run diagnostic again with constant loads - should see <0.1% measurement error and convergence!

## References

1. Abur & Exposito, "Power System State Estimation" (2004)
2. Schweppe et al., "Power System Static State Estimation" (1970)
3. Wood & Wollenberg, "Power Generation, Operation, and Control" (2014)

---

**Bottom Line**: The UKF is fighting against the nonlinear power flow equations. Either simplify the problem (constant loads) or use methods designed for power systems (WLS state estimation).
