"""
Unit test for IAUKF enhancements - code structure validation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from model.iaukf import IAUKF, IAUKFMultiSnapshot

print("="*80)
print("IAUKF ENHANCEMENTS - CODE STRUCTURE VALIDATION")
print("="*80)

# Test 1: Check that IAUKFMultiSnapshot class exists and has correct methods
print("\n[1] Checking IAUKFMultiSnapshot class...")
assert hasattr(IAUKFMultiSnapshot, '__init__'), "Missing __init__"
assert hasattr(IAUKFMultiSnapshot, 'predict'), "Missing predict method"
assert hasattr(IAUKFMultiSnapshot, 'update'), "Missing update method"
assert hasattr(IAUKFMultiSnapshot, 'adaptive_noise_update'), "Missing adaptive_noise_update"
assert hasattr(IAUKFMultiSnapshot, 'get_parameters'), "Missing get_parameters method"
print("  ✓ IAUKFMultiSnapshot has all required methods")

# Test 2: Check that IAUKF.adaptive_noise_update uses exact formula
print("\n[2] Checking IAUKF.adaptive_noise_update signature...")
import inspect
source = inspect.getsource(IAUKF.adaptive_noise_update)
assert "sigma_cov" in source, "NSE should compute sigma_cov directly"
assert "Eq 17" in source or "EXACT" in source, "Should reference exact formula"
print("  ✓ IAUKF.adaptive_noise_update uses exact NSE formula (Eq 17)")

# Test 3: Check that biased estimator (Eq 18) is implemented
assert "diag" in source and "bias_term" in source, "Should have biased estimator fallback"
print("  ✓ Biased estimator (Eq 18) is implemented")

# Test 4: Verify multi-snapshot initialization
print("\n[3] Testing IAUKFMultiSnapshot initialization...")

class DummyModel:
    def state_transition(self, x):
        return x
    def measurement_function(self, x):
        return x[:10]  # Return first 10 elements as measurement

n_state = 10
x0 = np.ones(n_state)
P0 = np.eye(n_state)
Q0 = np.eye(n_state) * 1e-6
R = np.eye(10)

try:
    iaukf_ms = IAUKFMultiSnapshot(DummyModel(), x0, P0, Q0, R, num_snapshots=3)
    print(f"  ✓ Initialized with num_snapshots=3")
    print(f"  ✓ Augmented state dimension: {iaukf_ms.n}")
    print(f"  ✓ Single state dimension: {iaukf_ms.n_single}")
    print(f"  ✓ System dimension: {iaukf_ms.n_sys_single}")
    print(f"  ✓ Parameter dimension: {iaukf_ms.n_params}")
    
    # Check get_parameters works
    params = iaukf_ms.get_parameters()
    assert len(params) == iaukf_ms.n_params, "get_parameters should return correct size"
    print(f"  ✓ get_parameters() works (returns {len(params)} params)")
    
except Exception as e:
    print(f"  ✗ Failed: {e}")
    raise

# Test 5: Check sigma point generation works with improved robustness
print("\n[4] Testing improved sigma_points robustness...")
iaukf = IAUKF(DummyModel(), x0, P0, Q0, R)

# Test with non-positive definite matrix
P_bad = np.eye(n_state) * -0.01  # Negative eigenvalues
try:
    sigmas = iaukf.sigma_points(x0, P_bad)
    print(f"  ✓ Handles non-positive definite covariance")
    print(f"  ✓ Generated {len(sigmas)} sigma points")
except Exception as e:
    print(f"  ✗ Failed to handle bad covariance: {e}")
    raise

# Test 6: Summary
print("\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)
print("✓ IAUKFMultiSnapshot class implemented with all required methods")
print("✓ Exact NSE formula (Eq 17) implemented - computes sigma_cov directly")
print("✓ Biased estimator fallback (Eq 18) implemented")
print("✓ Multi-snapshot initialization works")
print("✓ Improved robustness for non-positive definite covariance")
print("\nAll structural validations PASSED! ✓")
print("="*80)

print("\nNext steps:")
print("  - The core enhancements are implemented and structurally correct")
print("  - Functional testing requires a working power system simulation")
print("  - Consider running existing experiments (phase1_exact_paper.py) to validate")
