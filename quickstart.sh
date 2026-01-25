#!/bin/bash
# Quick Start Script for Power Grid Estimation

echo "=========================================="
echo "Power Grid Estimation - Quick Start"
echo "=========================================="
echo ""

# Check environment
echo "1. Checking conda environment..."
if conda env list | grep -q "graphmamba"; then
    echo "✓ graphmamba environment found"
else
    echo "✗ graphmamba environment not found"
    echo "Please activate it: conda activate graphmamba"
    exit 1
fi

# Check if in correct directory
if [ ! -f "main.py" ]; then
    echo "✗ Please run this script from /data1/xh/workspace/power/iaukf"
    exit 1
fi

echo ""
echo "2. Installing swanlab (if needed)..."
pip install swanlab --quiet || echo "SwanLab already installed"

echo ""
echo "=========================================="
echo "Choose an option:"
echo "=========================================="
echo "1. Test IAUKF (20 steps, ~10 seconds)"
echo "2. Train Graph Mamba (50 episodes, ~10 minutes)"
echo "3. Run Benchmark (20 episodes, ~5 minutes)"
echo "4. Full Pipeline (all of the above)"
echo "5. Validate Installation Only"
echo ""
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo ""
        echo "Running IAUKF..."
        python main.py
        echo ""
        echo "✓ Check 'iaukf_results.png' for convergence plot"
        ;;
    2)
        echo ""
        echo "Training Graph Mamba..."
        python train_mamba.py
        echo ""
        echo "✓ Model saved to 'graph_mamba_checkpoint.pt'"
        echo "✓ Logs saved to './swanlog/'"
        echo "✓ Run 'swanlab watch' to view dashboard"
        ;;
    3)
        echo ""
        echo "Running Benchmark..."
        python benchmark.py
        echo ""
        echo "✓ Check 'benchmark_tracking.png' and 'benchmark_boxplot.png'"
        ;;
    4)
        echo ""
        echo "Running Full Pipeline..."
        echo ""
        echo "Step 1/3: Testing IAUKF..."
        python main.py
        echo ""
        echo "Step 2/3: Training Graph Mamba..."
        python train_mamba.py
        echo ""
        echo "Step 3/3: Running Benchmark..."
        python benchmark.py
        echo ""
        echo "=========================================="
        echo "✓ Full pipeline complete!"
        echo "=========================================="
        echo "Generated files:"
        ls -lh *.png *.pt 2>/dev/null | awk '{print "  -", $9, "("$5")"}'
        echo ""
        echo "View SwanLab dashboard: swanlab watch"
        ;;
    5)
        echo ""
        echo "Running validation..."
        python validate.py
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "Done!"
