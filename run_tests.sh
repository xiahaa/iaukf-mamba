#!/bin/bash
# Test script for the full IAUKF + Graph Mamba pipeline

set -e  # Exit on error

echo "========================================"
echo "Power Grid State Estimation Test Suite"
echo "========================================"
echo ""

# Check Python version
echo "Checking Python version..."
python --version
echo ""

# Test 1: IAUKF Implementation
echo "========================================"
echo "Test 1: IAUKF Implementation"
echo "========================================"
python main.py
echo "✓ IAUKF test completed"
echo ""

# Test 2: Graph Mamba Training
echo "========================================"
echo "Test 2: Graph Mamba Training"
echo "========================================"
python train_mamba.py
echo "✓ Graph Mamba training completed"
echo ""

# Test 3: Benchmark Comparison
echo "========================================"
echo "Test 3: Benchmark Comparison"
echo "========================================"
python benchmark.py
echo "✓ Benchmark comparison completed"
echo ""

echo "========================================"
echo "All tests completed successfully!"
echo "========================================"
echo ""
echo "Generated files:"
ls -lh *.png *.pt 2>/dev/null || echo "No output files found"
