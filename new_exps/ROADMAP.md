# Graph-Mamba Improvement Roadmap

## Current Status (300 steps)
- Multi-snapshot IAUKF: **0.12%** R error, 92s runtime
- Graph-Mamba: **0.30%** R error, 0.1s runtime
- Gap: 2.5× (down from 14× with 50 steps)

## Proposed Improvements

### 1. Train on Constant-Parameter Episodes ⭐ HIGH PRIORITY
**Current issue**: Model trained on time-varying parameters
**Solution**: Generate training data with constant parameters (like IAUKF test)
**Expected gain**: 0.30% → 0.20% (1.5× improvement)

### 2. Longer Training Sequences ⭐ HIGH PRIORITY
**Current**: Trained on 50-step sequences
**Solution**: Train on 200-300 step sequences
**Expected gain**: Better temporal modeling, 0.30% → 0.25%

### 3. Larger Model Capacity ⭐ MEDIUM PRIORITY
**Current**: d_model=64, d_state=16
**Solution**: d_model=128, d_state=32, num_layers=3
**Expected gain**: Better feature extraction, 0.30% → 0.22%

### 4. Ensemble of Models ⭐ MEDIUM PRIORITY
**Solution**: Train 5 models with different seeds, average predictions
**Expected gain**: Reduced variance, 0.30% → 0.20%

### 5. Test-Time Adaptation ⭐ HIGH PRIORITY (Research)
**Solution**: Fine-tune on test sequence for 10-20 iterations
**Expected gain**: Could match multi-snapshot (0.30% → 0.12%)

### 6. Multi-Scale Temporal Modeling
**Solution**: Process sequence at multiple time resolutions
**Expected gain**: Capture both fast and slow dynamics

### 7. Physics-Informed Architecture
**Solution**: Embed power flow equations as neural network layers
**Expected gain**: Better physical consistency

## Implementation Priority

| Priority | Improvement | Effort | Expected Gain |
|----------|-------------|--------|---------------|
| 1 | Constant-param training | Low | 0.30% → 0.20% |
| 2 | Longer sequences | Medium | 0.30% → 0.25% |
| 3 | Test-time adaptation | High | 0.30% → 0.12% |
| 4 | Model ensemble | Low | 0.30% → 0.20% |
| 5 | Larger capacity | Medium | 0.30% → 0.22% |

## Target: Match Multi-Snapshot Accuracy

To reach 0.12% R error (matching multi-snapshot):

**Path A - Training improvements only**:
- Constant-param training: 0.30% → 0.20%
- Longer sequences: 0.20% → 0.15%
- Ensemble (5 models): 0.15% → 0.12% ✓

**Path B - With test-time adaptation**:
- Base model: 0.30%
- 20-step adaptation: 0.30% → 0.12% ✓

## Conclusion

Graph-Mamba can likely **match or beat multi-snapshot IAUKF accuracy** with:
1. Better training data (constant parameters)
2. Longer sequences
3. Optional: Test-time adaptation

**Key advantage**: Still 100×+ faster than multi-snapshot even with adaptation!
