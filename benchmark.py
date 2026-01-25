import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from simulation import PowerSystemSimulation
from models import DistributionSystemModel
from iaukf import IAUKF
from graph_mamba import GraphMambaModel, PhysicsInformedLoss
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STEPS_PER_EPISODE = 200
TEST_EPISODES = 20 # Number of test scenarios
NOISE_LEVELS = [0.01, 0.02, 0.05] # SCADA Noise levels to test robustness

def train_mamba_model_for_benchmarking():
    """
    Quickly trains a Mamba model to be used in the benchmark.
    In a real scenario, you would load a saved checkpoint.
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
    model = GraphMambaModel(num_nodes=len(net.bus), in_features=3, d_model=32).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    # Training Loop
    model.train()
    for epoch in range(15): # Short training for demo
        epoch_loss = 0
        for _ in range(train_sims):
            sim = PowerSystemSimulation(steps=steps)
            data = sim.run_simulation()

            # Prepare Data
            z_scada = data['z_scada'] # [T, 99]
            # Reshape [T, N, 3]
            p = z_scada[:, :33]
            q = z_scada[:, 33:66]
            v = z_scada[:, 66:]
            x = np.stack([p, q, v], axis=2)
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE) # [1, T, N, 3]

            target = torch.tensor([data['r_true'], data['x_true']], dtype=torch.float32).unsqueeze(0).to(DEVICE)

            optimizer.zero_grad()
            pred = model(x_tensor, edge_index)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch+1) % 5 == 0:
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
    x0_r = data['r_true'] * 0.5 # Distorted Guess
    x0_x = data['x_true'] * 0.5
    x0 = np.concatenate([x0_v, x0_d, [x0_r, x0_x]])

    P0 = np.eye(len(x0)) * 0.01
    Q0 = np.eye(len(x0)) * 1e-6
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
        preds.append([x_est[-2], x_est[-1]]) # Store R, X

    return np.array(preds)

def run_mamba_benchmark(model, edge_index, data):
    """Runs Graph Mamba on a single simulation instance."""
    model.eval()
    with torch.no_grad():
        z_scada = data['z_scada']
        p = z_scada[:, :33]
        q = z_scada[:, 33:66]
        v = z_scada[:, 66:]
        x = np.stack([p, q, v], axis=2)
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE) # [1, T, N, 3]

        # Mamba predicts the FINAL parameter (static estimate for the whole sequence)
        # To make it comparable time-series wise, we could run it on expanding windows,
        # but standard Mamba takes the whole sequence.
        # For visualization, we can just plot the final converged value as a line
        # or simulate 'online' inference by passing subsequences.
        # For this benchmark, let's assume Mamba gives a "one-shot" calibration after observing T steps.
        pred_final = model(x_tensor, edge_index)[0].cpu().numpy()

    return pred_final

def run_comparative_experiment():
    # 1. Setup Models
    mamba_model, edge_index = train_mamba_model_for_benchmarking()

    results = {
        'Method': [],
        'Metric': [],
        'Value': [],
        'Parameter': []
    }

    ts_data = {'Time': [], 'Value': [], 'Type': [], 'Parameter': []}

    print(">>> Running Comparative Benchmark on Test Episodes...")

    for i in range(TEST_EPISODES):
        sim = PowerSystemSimulation(steps=STEPS_PER_EPISODE)
        data = sim.run_simulation()
        true_r, true_x = data['r_true'], data['x_true']

        # --- Run IAUKF ---
        iaukf_preds = run_iaukf_benchmark(data, steps=STEPS_PER_EPISODE)
        iaukf_final = iaukf_preds[-1]

        # --- Run Mamba ---
        mamba_final = run_mamba_benchmark(mamba_model, edge_index, data)

        # --- Store Metrics ---
        # Error Calculation
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
            # True Line
            ts_data['Time'].extend(t_steps)
            ts_data['Value'].extend([true_r]*len(t_steps))
            ts_data['Type'].extend(['Ground Truth']*len(t_steps))
            ts_data['Parameter'].extend(['Resistance']*len(t_steps))

            ts_data['Time'].extend(t_steps)
            ts_data['Value'].extend([true_x]*len(t_steps))
            ts_data['Type'].extend(['Ground Truth']*len(t_steps))
            ts_data['Parameter'].extend(['Reactance']*len(t_steps))

            # IAUKF Line
            ts_data['Time'].extend(t_steps)
            ts_data['Value'].extend(iaukf_preds[:, 0])
            ts_data['Type'].extend(['IAUKF (Model-Based)']*len(t_steps))
            ts_data['Parameter'].extend(['Resistance']*len(t_steps))

            ts_data['Time'].extend(t_steps)
            ts_data['Value'].extend(iaukf_preds[:, 1])
            ts_data['Type'].extend(['IAUKF (Model-Based)']*len(t_steps))
            ts_data['Parameter'].extend(['Reactance']*len(t_steps))

            # Mamba Line (Constant estimate for visualization)
            ts_data['Time'].extend(t_steps)
            ts_data['Value'].extend([mamba_final[0]]*len(t_steps))
            ts_data['Type'].extend(['Graph Mamba (DL)']*len(t_steps))
            ts_data['Parameter'].extend(['Resistance']*len(t_steps))

            ts_data['Time'].extend(t_steps)
            ts_data['Value'].extend([mamba_final[1]]*len(t_steps))
            ts_data['Type'].extend(['Graph Mamba (DL)']*len(t_steps))
            ts_data['Parameter'].extend(['Reactance']*len(t_steps))

    # --- Analysis & Plotting ---
    df_res = pd.DataFrame(results)
    df_ts = pd.DataFrame(ts_data)

    # 1. Box Plot of Errors
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df_res, x='Parameter', y='Value', hue='Method')
    plt.title("Estimation Error Distribution (20 Test Episodes)")
    plt.ylabel("Absolute Error (Ohm/km)")
    plt.grid(True, alpha=0.3)
    plt.savefig("benchmark_boxplot.png")
    plt.show()

    # 2. Time Series Tracking Plot
    g = sns.FacetGrid(df_ts, col="Parameter", hue="Type", height=5, aspect=1.5, sharey=False)
    g.map(sns.lineplot, "Time", "Value", linewidth=2.5)
    g.add_legend(title="Method")
    g.fig.suptitle("Parameter Tracking Trajectory (Episode 1)", y=1.02)
    plt.savefig("benchmark_tracking.png")
    plt.show()

    # 3. LaTeX Table Generation
    summary = df_res.groupby(['Method', 'Parameter'])['Value'].agg(['mean', 'std']).reset_index()
    print("\n>>> Experiment Summary Table (LaTeX Format):")
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

if __name__ == "__main__":
    run_comparative_experiment()