"""
Master Runner for All Experiments
==================================
Run all experiments from highest to lowest priority.

Priority Order:
P0 (Must Have):
  1. exp1_basic_performance.py - Basic accuracy comparison
  2. exp2_dynamic_tracking.py - Time-varying parameters
  4. exp4_speed_comparison.py - Computational efficiency

P1 (Important):
  3. exp3_low_observability.py - Sparse PMU scenario
  5. exp5_robustness.py - Non-Gaussian noise

P2 (Optional):
  6. exp6_generalization.py - Cross-topology transfer

Usage:
  conda activate graphmamba
  python run_all_experiments.py [--p0-only] [--skip P1,P2]
"""

import sys
import os
import argparse
import subprocess
import time
from datetime import datetime

def run_experiment(script_name, exp_name):
    """Run a single experiment script."""
    print("\n" + "=" * 80)
    print(f"Running: {exp_name}")
    print(f"Script: {script_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            check=True,
            capture_output=False
        )
        elapsed = time.time() - start_time
        print(f"\n✓ {exp_name} completed in {elapsed:.1f}s")
        return True, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {exp_name} failed after {elapsed:.1f}s")
        print(f"Error: {e}")
        return False, elapsed


def main():
    parser = argparse.ArgumentParser(description='Run all Graph-Mamba experiments')
    parser.add_argument('--p0-only', action='store_true', 
                       help='Run only P0 (must-have) experiments')
    parser.add_argument('--skip', type=str, default='',
                       help='Comma-separated list of priorities to skip (e.g., P2)')
    parser.add_argument('--exp', type=str, default='',
                       help='Run specific experiment only (e.g., exp1)')
    args = parser.parse_args()
    
    skip_priorities = [p.strip().upper() for p in args.skip.split(',') if p.strip()]
    
    # Define experiment order
    experiments = [
        ('exp1_basic_performance.py', 'Experiment 1: Basic Performance', 'P0'),
        ('exp2_dynamic_tracking.py', 'Experiment 2: Dynamic Tracking', 'P0'),
        ('exp4_speed_comparison.py', 'Experiment 4: Speed Comparison', 'P0'),
        ('exp3_low_observability.py', 'Experiment 3: Low Observability', 'P1'),
        ('exp5_robustness.py', 'Experiment 5: Robustness', 'P1'),
        ('exp6_generalization.py', 'Experiment 6: Generalization', 'P2'),
    ]
    
    # Filter experiments
    if args.exp:
        target = args.exp if args.exp.endswith('.py') else f"{args.exp}.py"
        experiments = [(s, n, p) for s, n, p in experiments if s == target]
        if not experiments:
            print(f"Error: Experiment '{args.exp}' not found")
            return 1
    elif args.p0_only:
        experiments = [(s, n, p) for s, n, p in experiments if p == 'P0']
    else:
        experiments = [(s, n, p) for s, n, p in experiments if p not in skip_priorities]
    
    print("=" * 80)
    print("GRAPH-MAMBA EXPERIMENTAL SUITE")
    print("=" * 80)
    print(f"\nTotal experiments to run: {len(experiments)}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if experiments:
        print("\nExperiment queue:")
        for i, (script, name, priority) in enumerate(experiments, 1):
            print(f"  {i}. [{priority}] {name}")
    
    print()
    
    # Run experiments
    results = []
    total_start = time.time()
    
    for script, name, priority in experiments:
        success, elapsed = run_experiment(script, name)
        results.append({
            'name': name,
            'script': script,
            'priority': priority,
            'success': success,
            'elapsed': elapsed
        })
    
    total_elapsed = time.time() - total_start
    
    # Summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUITE SUMMARY")
    print("=" * 80)
    
    success_count = sum(1 for r in results if r['success'])
    fail_count = len(results) - success_count
    
    print(f"\nTotal: {len(results)} experiments")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    
    print(f"\n{'Experiment':<40} {'Priority':<8} {'Status':<10} {'Time':<10}")
    print("-" * 70)
    
    for r in results:
        status = "✓ PASS" if r['success'] else "✗ FAIL"
        print(f"{r['name']:<40} {r['priority']:<8} {status:<10} {r['elapsed']:.1f}s")
    
    if fail_count == 0:
        print("\n🎉 All experiments completed successfully!")
    else:
        print(f"\n⚠️  {fail_count} experiment(s) failed. Check logs above.")
    
    print("\nResults saved in: new_exps/results/")
    print("SwanLab logs: https://swanlab.cn/...")
    
    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
