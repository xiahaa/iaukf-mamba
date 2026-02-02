import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from simulation import PowerSystemSimulation
from model.models_holt import DistributionSystemModelHolt as DistributionSystemModel
from model.iaukf import IAUKF
from graph_mamba import GraphMambaModel, PhysicsInformedLoss
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STEPS_PER_EPISODE = 200
TEST_EPISODES = 20  # Number of test scenarios
CHECKPOINT_PATH = 'graph_mamba_checkpoint.pt'

print(f"Using device: {DEVICE}")


def load_trained_mamba_model():
    """Load pre-trained Graph Mamba model or train a quick one."""
    if os.path.exists(CHECKPOINT_PATH):
        print(f">>> Loading trained Graph Mamba model from '{CHECKPOINT_PATH}'...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

        # Reconstruct model
        model = GraphMambaModel(
            num_nodes=checkpoint['num_nodes'],
            in_features=checkpoint['in_features'],
            d_model=64
        ).to(DEVICE)

        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"    Model loaded from epoch {checkpoint['epoch']+1}")
        print(f"    Validation loss: {checkpoint['val_loss']:.6f}")

        # Get edge_index
        dummy_sim = PowerSystemSimulation(steps=1)
        net = dummy_sim.net
        edge_index = torch.tensor([net.line.from_bus.values, net.line.to_bus.values], dtype=torch.long)
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1).to(DEVICE)

        return model, edge_index
    else:
        print(f">>> Checkpoint not found. Training a quick model for benchmarking...")
        return train_mamba_model_for_benchmarking()


def train_mamba_model_for_benchmarking():
    """
    Quickly trains a Mamba model to be used in the benchmark.
    Used as fallback if no checkpoint exists.
    """
    print(">>> Pre-training Graph Mamba Model for Benchmarking...")
    # Generate small training set
    train_sims = 30
    steps = 200

    # Static topology setup
    dummy_sim = PowerSystemSimulation(steps=1)
    net = dummy_sim.net
    edge_index = torch.tensor([net.line.from_bus.values, net.line.to_bus.values], dtype=torch.long)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1).to(DEVICE)

    # Initialize Model
    model = GraphMambaModel(num_nodes=len(net.bus), in_features=3, d_model=64).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    # Training Loop
    model.train()
    for epoch in range(15):  # Short training for demo
        epoch_loss = 0
        for _ in range(train_sims):
            sim = PowerSystemSimulation(steps=steps)
            data = sim.run_simulation()

            # Prepare Data
            z_scada = data['z_scada']  # [T, 99]
            # Reshape [T, N, 3]
            p = z_scada[:, :33]
            q = z_scada[:, 33:66]
            v = z_scada[:, 66:]
            x = np.stack([p, q, v], axis=2)
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # [1, T, N, 3]

            target = torch.tensor([data['r_true'], data['x_true']], dtype=torch.float32).unsqueeze(0).to(DEVICE)

            optimizer.zero_grad()
            pred = model(x_tensor, edge_index)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1}/15 Loss: {epoch_loss/train_sims:.5f}")

    print(">>> Training Complete.\n")
    return model, edge_index


def run_iaukf_benchmark(data, steps=200):
    """Runs IAUKF on a single simulation instance."""
    sim_net = data['net']
    model = DistributionSystemModel(sim_net, data['target_line_idx'], data['pmu_indices'])

    # Initialize State
    x0_v = np.ones(33)
    x0_d = np.zeros(33)
    x0_r = data['r_true'] * 0.5  # Distorted Guess
    x0_x = data['x_true'] * 0.5
    x0 = np.concatenate([x0_v, x0_d, [x0_r, x0_x]])

    P0 = np.eye(len(x0)) * 0.01
    Q0 = np.eye(len(x0)) * 1e-6
    Q0[-2, -2] = 1e-4  # Allow parameter movement
    Q0[-1, -1] = 1e-4

    # Construct R matrix (Measurement Noise)
    # 33 P, 33 Q, 33 V_scada, 12 V_pmu, 12 Th_pmu = 123 measurements
    R_diag = np.concatenate([
        np.full(33, 0.02**2), np.full(33, 0.02**2), np.full(33, 0.02**2),
        np.full(12, 0.005**2), np.full(12, 0.002**2)
    ])
    R_cov = np.diag(R_diag)

    iaukf = IAUKF(model, x0, P0, Q0, R_cov)

    preds = []
    Z_comb = np.hstack([data['z_scada'], data['z_pmu']])

    for t in range(steps):
        iaukf.predict()
        x_est = iaukf.update(Z_comb[t])
        preds.append([x_est[-2], x_est[-1]])  # Store R, X

    return np.array(preds)


def run_mamba_benchmark_online(model, edge_index, data):
    """
    Runs Graph Mamba in ONLINE mode for fair comparison with IAUKF.
    Returns time-series predictions.
    """
    model.eval()
    with torch.no_grad():
        z_scada = data['z_scada']
        p = z_scada[:, :33]
        q = z_scada[:, 33:66]
        v = z_scada[:, 66:]
        x = np.stack([p, q, v], axis=2)
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # [1, T, N, 3]

        # Use online inference mode for time-series predictions
        preds_online = model.forward_online(x_tensor, edge_index)  # [T, 2]
        preds_online = preds_online.cpu().numpy()

    return preds_online


def run_comparative_experiment():
    """Main benchmark comparing IAUKF vs Graph Mamba."""
    # 1. Setup Models
    mamba_model, edge_index = load_trained_mamba_model()

    results = {
        'Method': [],
        'Metric': [],
        'Value': [],
        'Parameter': []
    }

    ts_data = {'Time': [], 'Value': [], 'Type': [], 'Parameter': []}

    print("\n" + "=" * 60)
    print("COMPARATIVE BENCHMARK: IAUKF vs Graph Mamba")
    print("=" * 60)
    print(f"Running {TEST_EPISODES} test episodes...\n")

    for i in range(TEST_EPISODES):
        # Generate test episode with different seed
        sim = PowerSystemSimulation(steps=STEPS_PER_EPISODE)
        data = sim.run_simulation(seed=1000 + i)
        true_r, true_x = data['r_true'], data['x_true']

        print(f"Episode {i+1}/{TEST_EPISODES}: R_true={true_r:.5f}, X_true={true_x:.5f}")

        # --- Run IAUKF ---
        iaukf_preds = run_iaukf_benchmark(data, steps=STEPS_PER_EPISODE)
        iaukf_final = iaukf_preds[-1]

        # --- Run Mamba (Online Mode) ---
        mamba_preds = run_mamba_benchmark_online(mamba_model, edge_index, data)
        mamba_final = mamba_preds[-1]

        # --- Store Metrics ---
        # Error Calculation (Final values)
        err_r_iaukf = np.abs(iaukf_final[0] - true_r)
        err_x_iaukf = np.abs(iaukf_final[1] - true_x)
        err_r_mamba = np.abs(mamba_final[0] - true_r)
        err_x_mamba = np.abs(mamba_final[1] - true_x)

        results['Method'].extend(['IAUKF', 'IAUKF', 'GraphMamba', 'GraphMamba'])
        results['Metric'].extend(['AbsError', 'AbsError', 'AbsError', 'AbsError'])
        results['Value'].extend([err_r_iaukf, err_x_iaukf, err_r_mamba, err_x_mamba])
        results['Parameter'].extend(['R', 'X', 'R', 'X'])

        # --- Store Time Series for the FIRST episode only (for plotting) ---
        if i == 0:
            t_steps = np.arange(STEPS_PER_EPISODE)

            # Ground Truth
            for param_name, true_val, iaukf_vals, mamba_vals in [
                ('Resistance', true_r, iaukf_preds[:, 0], mamba_preds[:, 0]),
                ('Reactance', true_x, iaukf_preds[:, 1], mamba_preds[:, 1])
            ]:
                ts_data['Time'].extend(t_steps)
                ts_data['Value'].extend([true_val] * len(t_steps))
                ts_data['Type'].extend(['Ground Truth'] * len(t_steps))
                ts_data['Parameter'].extend([param_name] * len(t_steps))

                ts_data['Time'].extend(t_steps)
                ts_data['Value'].extend(iaukf_vals)
                ts_data['Type'].extend(['IAUKF (Model-Based)'] * len(t_steps))
                ts_data['Parameter'].extend([param_name] * len(t_steps))

                ts_data['Time'].extend(t_steps)
                ts_data['Value'].extend(mamba_vals)
                ts_data['Type'].extend(['Graph Mamba (DL)'] * len(t_steps))
                ts_data['Parameter'].extend([param_name] * len(t_steps))

    # --- Analysis & Plotting ---
    print("\n" + "=" * 60)
    print("GENERATING RESULTS")
    print("=" * 60)

    df_res = pd.DataFrame(results)
    df_ts = pd.DataFrame(ts_data)

    # 1. Box Plot of Errors
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_res, x='Parameter', y='Value', hue='Method', palette='Set2')
    plt.title(f"Estimation Error Distribution ({TEST_EPISODES} Test Episodes)", fontsize=14, fontweight='bold')
    plt.ylabel("Absolute Error (Ohm/km)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("benchmark_boxplot.png", dpi=150)
    print("✓ Box plot saved as 'benchmark_boxplot.png'")
    plt.close()

    # 2. Time Series Tracking Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, param in enumerate(['Resistance', 'Reactance']):
        ax = axes[idx]
        subset = df_ts[df_ts['Parameter'] == param]

        for method_type in ['Ground Truth', 'IAUKF (Model-Based)', 'Graph Mamba (DL)']:
            method_data = subset[subset['Type'] == method_type]
            if method_type == 'Ground Truth':
                ax.plot(method_data['Time'], method_data['Value'],
                       linestyle='--', linewidth=2.5, label=method_type, color='red', alpha=0.8)
            else:
                ax.plot(method_data['Time'], method_data['Value'],
                       linewidth=2, label=method_type, alpha=0.9)

        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel(f'{param} (Ohm/km)', fontsize=11)
        ax.set_title(f'{param} Tracking', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Parameter Tracking Trajectory (Episode 1)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("benchmark_tracking.png", dpi=150)
    print("✓ Tracking plot saved as 'benchmark_tracking.png'")
    plt.close()

    # 3. Summary Statistics Table
    summary = df_res.groupby(['Method', 'Parameter'])['Value'].agg(['mean', 'std']).reset_index()

    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Method':<15} {'Parameter':<12} {'Mean Error':<15} {'Std Dev':<15}")
    print("-" * 60)
    for _, row in summary.iterrows():
        print(f"{row['Method']:<15} {row['Parameter']:<12} {row['mean']:<15.6f} {row['std']:<15.6f}")
    print("=" * 60)

    # 4. LaTeX Table Generation
    print("\n>>> LaTeX Format:")
    print("-" * 60)
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{lccc}")
    print(r"\hline")
    print(r"Method & Parameter & Mean Error & Std Dev \\")
    print(r"\hline")
    for _, row in summary.iterrows():
        print(f"{row['Method']} & {row['Parameter']} & {row['mean']:.5f} & {row['std']:.5f} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{Comparison of Model-Based IAUKF vs. Data-Driven Graph Mamba}")
    print(r"\label{tab:results}")
    print(r"\end{table}")
    print("-" * 60)

    # 5. Performance Comparison
    print("\n" + "=" * 60)
    print("PERFORMANCE ANALYSIS")
    print("=" * 60)

    # Calculate percentage improvements
    for param in ['R', 'X']:
        iaukf_mean = summary[(summary['Method'] == 'IAUKF') & (summary['Parameter'] == param)]['mean'].values[0]
        mamba_mean = summary[(summary['Method'] == 'GraphMamba') & (summary['Parameter'] == param)]['mean'].values[0]
        improvement = (iaukf_mean - mamba_mean) / iaukf_mean * 100

        print(f"\nParameter {param}:")
        print(f"  IAUKF Mean Error:     {iaukf_mean:.6f}")
        print(f"  GraphMamba Mean Error: {mamba_mean:.6f}")
        print(f"  Improvement:          {improvement:+.2f}%")

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_comparative_experiment()
