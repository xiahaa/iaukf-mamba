# IAUKF Enhancement Documentation

## Overview

This document describes the enhancements made to the IAUKF (Improved Adaptive Unscented Kalman Filter) implementation to match the specifications in the reference paper (docs/ref_core.md).

## Enhancements Made

### 1. Exact NSE Implementation (Equations 17 & 18)

**Previous Implementation:**
- Used an approximation for the sigma point covariance term
- Approximated: `sigma_cov ≈ P_pred - Q_k`
- Comment in code: "Let's approximate the Bracket term..."

**New Implementation:**
- Computes sigma point covariance **exactly** as specified in Eq. 17
- Formula: `sigma_cov = Σ W_i^(c) * (X_{k|k-1}^(i) - x_{k|k-1})(X_{k|k-1}^(i) - x_{k|k-1})^T`
- Uses stored sigma points from prediction step (`self.sigmas_f`)

**Code Changes in `model/iaukf.py`:**

```python
# Term 3: Sigma point covariance (Eq 17 - exact computation)
sigma_cov = np.zeros((self.n, self.n))
for i in range(2*self.n + 1):
    diff = self.sigmas_f[i] - self.x_pred
    sigma_cov += self.Wc[i] * np.outer(diff, diff)

# Exact update from Eq 17
Q_next = (1 - d_k) * self.Q + d_k * (term1 + term2 - sigma_cov)
```

**Biased Estimator Fallback (Eq. 18):**
- When unbiased estimator produces non-positive definite Q
- Uses: `Q = (1-d_k)*Q_k + d_k*[diag(K*ε*ε^T*K^T) + K*P_zz*K^T]`
- Previously used simple diagonal extraction, now uses correct formula

### 2. Multiple Measurement Snapshots Support (Equations 32-38)

**Implementation:**
- New class: `IAUKFMultiSnapshot`
- Implements augmented state-space model from Section IV.C of paper

**Key Features:**

1. **Augmented State Vector (Eq. 32-33):**
   ```
   X_k = [x_1, x_2, ..., x_t, p_k]^T
   ```
   - Combines system states from multiple time steps
   - Parameters (R, X) shared across snapshots

2. **Augmented Measurement Vector (Eq. 34):**
   ```
   Z_k = [z_1, z_2, ..., z_t]^T
   ```
   - Stacks measurements from multiple snapshots

3. **State Transition (Eq. 35):**
   - System states: shift window (oldest dropped, newest predicted)
   - Parameters: remain constant

4. **Measurement Function (Eq. 36):**
   - Each snapshot measurement computed independently
   - All use same parameter estimate

5. **Automatic Buffering:**
   - Can accept single measurements
   - Automatically buffers until `num_snapshots` collected
   - Then performs update with all snapshots

**Usage Example:**

```python
from model.iaukf import IAUKFMultiSnapshot

# Initialize with 5 snapshots (as in paper Table II)
iaukf_ms = IAUKFMultiSnapshot(
    model, x0, P0, Q0, R, 
    num_snapshots=5
)

# Run filter
for measurement in measurements:
    iaukf_ms.predict()
    iaukf_ms.update(measurement)  # Auto-buffers snapshots
    params = iaukf_ms.get_parameters()  # Get [R, X]
```

### 3. Improved Numerical Stability

**Enhanced Sigma Point Generation:**

```python
def sigma_points(self, x, P):
    # Ensure P is symmetric
    P = (P + P.T) / 2
    
    # Multi-level fallback for non-positive definite matrices
    try:
        L = cholesky((self.n + self.lam) * P, lower=True)
    except np.linalg.LinAlgError:
        # Add regularization
        min_eig = np.min(np.linalg.eigvalsh(P))
        if min_eig < 0:
            P = P + np.eye(self.n) * (abs(min_eig) + 1e-6)
        else:
            P = P + np.eye(self.n) * 1e-6
        # ... (additional fallbacks)
```

## Expected Performance

Based on ref_core.md Table II, for IEEE 33-bus system branch 3-4:

| Configuration | R Error | X Error |
|--------------|---------|---------|
| Single snapshot | ~0.18% | ~1.55% |
| Multi-snapshot (5 snapshots) | ~0.13% | ~0.09% |

For end branches (e.g., 21-22) with multi-snapshot:
- R Error: ~0.52%
- X Error: ~2.03%

## Files Modified

1. **model/iaukf.py**
   - Lines 30-57: Enhanced `sigma_points()` with better robustness
   - Lines 167-207: Exact NSE implementation (Eq. 17 & 18)
   - Lines 217-505: New `IAUKFMultiSnapshot` class

2. **New Test Files**
   - `experiments/test_code_structure.py`: Validates code structure (PASSED ✓)
   - `experiments/quick_test_enhancements.py`: Quick functional test
   - `experiments/test_iaukf_enhancements.py`: Comprehensive test suite

## Testing and Validation

### Code Structure Validation (PASSED ✓)

Run: `python experiments/test_code_structure.py`

Results:
```
✓ IAUKFMultiSnapshot class implemented with all required methods
✓ Exact NSE formula (Eq 17) implemented - computes sigma_cov directly
✓ Biased estimator fallback (Eq 18) implemented
✓ Multi-snapshot initialization works
✓ Improved robustness for non-positive definite covariance
```

### Functional Testing

The enhancements are structurally correct. Full functional testing requires:
1. Running existing experiments (e.g., `phase1_exact_paper.py`)
2. Comparing results with paper benchmarks
3. Testing multi-snapshot on end branches

## Backward Compatibility

- Original `IAUKF` class maintains same interface
- Existing code using `IAUKF` will benefit from exact NSE
- `IAUKFMultiSnapshot` is an additional class, doesn't break existing code

## Implementation Notes

### Key Differences from Paper

1. **Snapshot Management:**
   - Paper assumes batch processing of snapshots
   - Implementation supports both batch and incremental (auto-buffering)

2. **Numerical Stability:**
   - Added extra safeguards for covariance matrix handling
   - Multiple fallback strategies for Cholesky decomposition

3. **State Augmentation:**
   - Assumes last 2 elements of state are parameters (R, X)
   - Configurable via `n_params` parameter

## Future Work

1. **Benchmark Testing:**
   - Run comprehensive tests with IEEE 33-bus and 118-bus systems
   - Validate against Table II results from paper

2. **End Branch Testing:**
   - Verify multi-snapshot improves end branch estimation
   - Test with branches 17-18, 21-22, 24-25, 32-33

3. **Integration:**
   - Integrate with existing experiment scripts
   - Add multi-snapshot option to phase1/phase2/phase3 experiments

## References

- **Paper**: "Augmented State Estimation of Line Parameters in Active Power Distribution Systems With Phasor Measurement Units"
- **Local Reference**: `docs/ref_core.md`
- **Key Equations**: 
  - NSE: Equations 15-18
  - Multi-snapshot: Equations 32-38
  - Full model: Equation 30

## Contact

For questions or issues related to these enhancements, refer to the PR discussion or open an issue.
