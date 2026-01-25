#!/usr/bin/env python
"""
Quick validation script to test basic functionality without full training.
"""
import sys
import numpy as np

def test_imports():
    """Test that all modules can be imported."""
    print("=" * 60)
    print("Testing Imports...")
    print("=" * 60)

    try:
        import pandapower as pp
        print("✓ pandapower imported successfully")
    except ImportError as e:
        print(f"✗ pandapower import failed: {e}")
        return False

    try:
        import torch
        print(f"✓ torch imported successfully (version: {torch.__version__})")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    - GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError as e:
        print(f"✗ torch import failed: {e}")
        return False

    try:
        import torch_geometric
        print(f"✓ torch_geometric imported successfully")
    except ImportError as e:
        print(f"✗ torch_geometric import failed: {e}")
        return False

    try:
        from simulation import PowerSystemSimulation
        print("✓ simulation module imported")
    except ImportError as e:
        print(f"✗ simulation import failed: {e}")
        return False

    try:
        from models import DistributionSystemModel
        print("✓ models module imported")
    except ImportError as e:
        print(f"✗ models import failed: {e}")
        return False

    try:
        from iaukf import IAUKF
        print("✓ iaukf module imported")
    except ImportError as e:
        print(f"✗ iaukf import failed: {e}")
        return False

    try:
        from graph_mamba import GraphMambaModel, PhysicsInformedLoss
        print("✓ graph_mamba module imported")
    except ImportError as e:
        print(f"✗ graph_mamba import failed: {e}")
        return False

    print("\n✓ All imports successful!\n")
    return True


def test_simulation():
    """Test basic simulation functionality."""
    print("=" * 60)
    print("Testing Simulation...")
    print("=" * 60)

    try:
        from simulation import PowerSystemSimulation

        sim = PowerSystemSimulation(steps=10)
        data = sim.run_simulation()

        assert 'z_scada' in data, "Missing z_scada"
        assert 'z_pmu' in data, "Missing z_pmu"
        assert 'true_states' in data, "Missing true_states"
        assert 'r_true' in data, "Missing r_true"
        assert 'x_true' in data, "Missing x_true"

        print(f"✓ Simulation successful")
        print(f"  - SCADA shape: {data['z_scada'].shape}")
        print(f"  - PMU shape: {data['z_pmu'].shape}")
        print(f"  - True R: {data['r_true']:.5f}")
        print(f"  - True X: {data['x_true']:.5f}")
        print()
        return True
    except Exception as e:
        print(f"✗ Simulation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_iaukf():
    """Test IAUKF basic functionality."""
    print("=" * 60)
    print("Testing IAUKF...")
    print("=" * 60)

    try:
        from simulation import PowerSystemSimulation
        from models import DistributionSystemModel
        from iaukf import IAUKF

        # Generate data
        sim = PowerSystemSimulation(steps=10)
        data = sim.run_simulation()

        # Setup model
        model = DistributionSystemModel(data['net'], data['target_line_idx'], data['pmu_indices'])

        # Initialize filter
        x0_v = np.ones(33)
        x0_d = np.zeros(33)
        x0_r = data['r_true'] * 0.5
        x0_x = data['x_true'] * 0.5
        x0 = np.concatenate([x0_v, x0_d, [x0_r, x0_x]])

        P0 = np.eye(len(x0)) * 0.01
        Q0 = np.eye(len(x0)) * 1e-6
        Q0[-2, -2] = 1e-4
        Q0[-1, -1] = 1e-4

        R_diag = np.concatenate([
            np.full(33, 0.02**2), np.full(33, 0.02**2), np.full(33, 0.02**2),
            np.full(12, 0.005**2), np.full(12, 0.002**2)
        ])
        R_cov = np.diag(R_diag)

        iaukf = IAUKF(model, x0, P0, Q0, R_cov)

        # Run a few steps
        Z_comb = np.hstack([data['z_scada'], data['z_pmu']])

        for t in range(min(5, len(Z_comb))):
            iaukf.predict()
            x_est = iaukf.update(Z_comb[t])

        print(f"✓ IAUKF test successful")
        print(f"  - Initial R: {x0_r:.5f}")
        print(f"  - Estimated R: {x_est[-2]:.5f}")
        print(f"  - True R: {data['r_true']:.5f}")
        print()
        return True
    except Exception as e:
        print(f"✗ IAUKF test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graph_mamba():
    """Test Graph Mamba model instantiation."""
    print("=" * 60)
    print("Testing Graph Mamba Model...")
    print("=" * 60)

    try:
        import torch
        from graph_mamba import GraphMambaModel

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create model
        model = GraphMambaModel(
            num_nodes=33,
            in_features=3,
            d_model=32
        ).to(device)

        # Create dummy input
        batch_size = 2
        seq_len = 10
        num_nodes = 33
        num_features = 3

        dummy_input = torch.randn(batch_size, seq_len, num_nodes, num_features).to(device)

        # Create dummy edge_index
        from simulation import PowerSystemSimulation
        sim = PowerSystemSimulation(steps=1)
        edge_index = torch.tensor([
            sim.net.line.from_bus.values,
            sim.net.line.to_bus.values
        ], dtype=torch.long)
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1).to(device)

        # Forward pass
        output = model(dummy_input, edge_index)

        print(f"✓ Graph Mamba test successful")
        print(f"  - Model device: {device}")
        print(f"  - Input shape: {dummy_input.shape}")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Expected output shape: [{batch_size}, 2]")
        assert output.shape == (batch_size, 2), f"Unexpected output shape: {output.shape}"
        print()
        return True
    except Exception as e:
        print(f"✗ Graph Mamba test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "POWER GRID ESTIMATION VALIDATION" + " " * 15 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Simulation", test_simulation()))
    results.append(("IAUKF", test_iaukf()))
    results.append(("Graph Mamba", test_graph_mamba()))

    # Summary
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:<20} {status}")

    print("=" * 60)

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n✓ All validation tests passed!")
        print("\nYou can now run:")
        print("  - python main.py          # Test IAUKF")
        print("  - python train_mamba.py   # Train Graph Mamba")
        print("  - python benchmark.py     # Compare methods")
        print()
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
