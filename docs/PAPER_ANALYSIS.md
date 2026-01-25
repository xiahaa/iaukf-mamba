# Analysis: IAUKF Paper Approach

## Your Question: "Do they use joint estimation?"

**YES - They use augmented state (joint) estimation!**

## Their State Vector

From the paper (equations 23-25):

```
Augmented state: x̄ₖ = [xₖ, pₖ]ᵀ

Where:
- xₖ = [Vᵢ, δᵢ]ᵀ     (voltage amplitude and phase angle)
- pₖ = [R, X]ᵀ       (line parameters)
```

**Total dimensions**:
- For IEEE 33-bus: xₖ has 66 dimensions (33 voltages + 33 angles)
- For p suspicious lines: pₖ has 2p dimensions
- **Total: 66 + 2p dimensions** (e.g., 68 for 1 line)

This is **exactly what we implemented** initially!

## Their Key Innovation: Holt's Exponential Smoothing

For voltage dynamics (equation 19):

```
xₖ|ₖ₋₁ = Sₖ₋₁ + bₖ₋₁           (prediction)
Sₖ₋₁ = αₕxₖ₋₁ + (1-αₕ)xₖ₋₁|ₖ₋₂  (level)
bₖ₋₁ = βₕ(Sₖ₋₁ - Sₖ₋₂) + (1-βₕ)bₖ₋₂  (trend)
```

**We used**: Identity transition `xₖ = xₖ₋₁` (simpler)

**They used**: Holt's smoothing with trend tracking

## Their Parameter Model

For line parameters (equation 26):

```
pₖ = pₖ₋₁ + wₚ|ₖ
```

Where `wₚ|ₖ ~ N(0, Qₚ|ₖ)` is adaptive process noise.

**Key insight**: They use **Noise Statistic Estimator (NSE)** to adaptively estimate Qₖ during filtering!

## Why They Can Make It Work

### 1. **Multiple Measurement Snapshots**

From the paper:
> "Since the line parameters are constant over a certain period and can be estimated **off-line**, the augmented state-space model can be extended to the form under **multiple measurement snapshots** to avoid the failure of estimation due to insufficient measurement redundancy."

They process multiple timesteps together, increasing observability!

### 2. **Adaptive Noise Estimation**

Their IAUKF includes NSE (equations 15-18):

```
εₖ = zₖ - ẑₖ|ₖ₋₁                           (innovation)
dₖ = (1-b)/(1-bᵏ)                          (adaptive factor)
Qₖ = (1-dₖ)Qₖ₋₁ + dₖ[...]                 (adaptive Q update)
```

This dynamically adjusts process noise when residuals are high!

### 3. **Off-line Estimation**

Key difference from our approach:
- **Their assumption**: Parameters are estimated **off-line** with steady-state data
- **Our assumption**: Real-time estimation with time-varying loads

## Why Our Implementation Struggled

### The Load Problem

In the paper, they don't explicitly discuss **how loads are handled**, but the implicit assumption for off-line estimation is:

1. **Steady-state operation**: System reaches steady state between snapshots
2. **Known or measured loads**: Loads are either constant or measured
3. **Multiple snapshots**: Aggregate data reduces noise and uncertainties

**Our simulation** had:
- Random load fluctuations every timestep: `±10%`
- No steady-state period
- Unknown load values in filter
- Single-snapshot processing

This created the ~2-3% measurement error we observed!

## Their Test Conditions (From Paper)

Looking at Section V (simulation results):

1. **IEEE 33-bus system**: Same as ours ✓
2. **PMU placement**: 12 PMUs ✓
3. **SCADA measurements**: P, Q, V at all buses ✓
4. **Parameter initialization**: They start with 50% error ✓

But critically:
- They likely use **steady-state snapshots**
- **Multiple snapshots** processed together
- Parameters estimated **off-line** (not real-time)

## Our Diagnosis Confirms Their Approach

Our systematic debugging showed:

1. **With constant loads** → 2-3% error (noise-limited) ✓
2. **With varying loads** → Oscillation (model mismatch) ✗
3. **Parameters-only state** → Better but still issues
4. **Load uncertainty** → Root cause

## The Missing Piece

The paper doesn't explicitly state it, but for their approach to work, they must have:

**Option A**: Steady-state data collection
- Wait for system to settle
- Collect measurements at steady state
- Estimate parameters off-line

**Option B**: Known loads
- Loads are measured or estimated separately
- Use known loads in power flow calculations

**Option C**: Joint load-parameter estimation
- Augment state further: x̄ = [V, δ, R, X, P_load, Q_load]
- But this increases dimensions significantly

## Recommendations for Your Implementation

### To Match Paper's Approach:

1. **Use Holt's smoothing** for state transition:
```python
def state_transition(self, x, S_prev, b_prev):
    alpha_H = 0.8
    beta_H = 0.5

    S = alpha_H * x + (1 - alpha_H) * x_prev
    b = beta_H * (S - S_prev) + (1 - beta_H) * b_prev

    return S + b, S, b
```

2. **Use constant loads** (or measure them):
```python
# In simulation
fluctuation = np.ones(len(p_load_base))  # Steady state
```

3. **Keep augmented state** (voltages + parameters):
```python
state = [V₁...Vₙ, δ₁...δₙ, R, X]  # 68 dimensions
```

4. **Implement adaptive NSE** (already in your IAUKF):
```python
# Your adaptive_noise_update() already implements this!
d_k = (1 - self.b_factor) / (1 - self.b_factor**(self.k_step + 1))
Q_next = (1 - d_k) * self.Q + d_k * update_term
```

5. **Process multiple snapshots** (optional for better results):
```python
# Collect N snapshots at steady state
# Process them together to increase redundancy
```

### The Key Difference:

**Paper's assumption**: Off-line estimation with steady-state data
**Your implementation**: Real-time estimation with dynamic loads

For **real-time parameter tracking** (your scenario), you need either:
- Constant/known loads, OR
- Joint load-parameter estimation (much harder), OR
- **Graph Mamba** (your contribution!) which learns the mapping without explicit power flow modeling

## Conclusion

**YES, they use joint (augmented) estimation** with:
- State: [V, δ, R, X] (68 dimensions for 33-bus, 1 line)
- Holt's smoothing for voltage dynamics
- Adaptive NSE for process noise
- **Implicit assumption**: Steady-state or known loads

**Your implementation** was correct in structure but struggled with:
- Time-varying unknown loads (not addressed in paper)
- Real-time vs off-line estimation difference

**Your Graph Mamba approach** actually solves the problem they didn't address: **parameter estimation with unknown time-varying loads**!

This strengthens your research contribution! 🎯
