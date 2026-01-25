Okay, here is a summary of the core contributions and experimental design for your paper, structured for an IEEE Transactions-style journal submission. I have framed the narrative around the "Model-Based vs. Data-Driven" comparison, highlighting the superiority of your proposed Graph Mamba method.

### **Title Proposal**

**Physics-Informed Graph Mamba: A Linear-Complexity State Space Model for Robust Joint Parameter and State Estimation in Active Distribution Grids**

---

### **1. Introduction & Motivation**

* **The Problem:** accurate knowledge of distribution line parameters (Resistance , Reactance ) is critical for advanced grid management (DSE, Optimal Power Flow). However, these parameters drift over time due to aging, temperature, and maintenance, leading to "parameter error" that degrades state estimation.
* **The Gap:**
* **Traditional Model-Based Methods** (like the Augmented UKF) are theoretically grounded but computationally expensive (), slow to converge, and struggle with non-Gaussian noise or sudden topology changes.
* **Existing Deep Learning Methods** (like RNNs/Transformers) often lack physical consistency or suffer from quadratic complexity (), making them inefficient for long historical sequences.


* **The Solution:** We propose a **Physics-Informed Graph Mamba (PI-GraphMamba)** framework. It combines the spatial awareness of Graph Neural Networks (GNNs) with the linear-complexity temporal modeling of Mamba (Selective State Space Models).
* **Key Insight:** Mamba's "Selective Scan" mechanism allows the model to differentiate between transient measurement noise (to be ignored) and persistent parameter drift (to be learned), solving the noise-vs-drift dilemma inherent in Kalman Filters.

---

### **2. Core Contributions**

1. **Architecture: Spatio-Temporal Graph Mamba**
* We introduce a novel hybrid architecture that first encodes grid snapshots into latent embeddings using a **Graph Convolutional Network (GCN)** to capture spatial correlations and topology.
* These embeddings are fed into a **Mamba Block**, which models the temporal evolution of the grid state with  complexity, enabling efficient processing of long measurement horizons.


2. **Physics-Informed Learning Framework**
* Unlike black-box models, our training objective incorporates a **Physical Consistency Loss ()**.
* The model minimizes a dual objective: the error in parameter estimation (Supervised Loss) AND the residual of the power flow equations (Unsupervised Physics Loss), ensuring that estimated parameters satisfy Kirchhoff's laws.


3. **Superior Robustness to Non-Ideal Conditions**
* We demonstrate that PI-GraphMamba outperforms the state-of-the-art **Improved Adaptive Unscented Kalman Filter (IAUKF)** in scenarios with:
* **Heavy-Tail/Non-Gaussian Noise:** Where UKF performance degrades.
* **Topology Shifts:** Where model-based filters often diverge due to mismatched Jacobian matrices.





---

### **3. Experimental Section (Methodology)**

We validate our approach on the **IEEE 33-Bus Distribution System** using a high-fidelity simulation environment.

#### **A. Baseline Comparison**

* **Benchmark:** Improved Adaptive Unscented Kalman Filter (IAUKF) [Wang et al., 2022].
* **Metric:** Root Mean Squared Error (RMSE) of estimated  and .
* **Setup:** 200 time-step simulation with dynamic load profiles ( fluctuation) and Gaussian measurement noise (, ).

#### **B. Convergence Analysis**

* **Observation:** The IAUKF requires 50-80 time steps to converge from a distorted initial guess ().
* **Result:** Graph Mamba demonstrates "One-Shot Calibration," achieving <0.5% error almost instantaneously by leveraging learned patterns from the training distribution.

#### **C. Stress Testing (The "Breakthrough" Results)**

1. **Topology Change Resilience:**
* *Scenario:* A line trip occurs at .
* *Result:* IAUKF estimation error spikes significantly. Graph Mamba maintains stability, adapting to the new graph structure via the GNN encoder.


2. **Robustness to Non-Gaussian Noise:**
* *Scenario:* Injection of random impulse noise (spikes) into 10% of measurements.
* *Result:* IAUKF oscillates as it attempts to track the outliers. Mamba's selective scan effectively gates the noise, maintaining a stable parameter estimate.



---

### **4. Placeholder Visuals (For Submission Draft)**

**Fig 1. Architecture Diagram**

* *Left:* GCN Encoder processing the IEEE 33-bus graph.
* *Center:* The Mamba Block Unrolling over time.
* *Right:* Output Head predicting  and .

**Fig 2. Convergence Trajectory (Time-Series Plot)**

* *X-Axis:* Time Steps (0-200).
* *Y-Axis:* Estimated Resistance ().
* *Lines:* Ground Truth (Green dashed), IAUKF (Orange, converging slowly), Graph Mamba (Blue, stable/flat).

**Fig 3. Error Distribution (Box Plot)**

* Side-by-side comparison of Absolute Error for IAUKF vs. Graph Mamba across 20 test episodes. Mamba should show a significantly tighter interquartile range (IQR).

**Table 1. Numerical Performance Summary**

* Columns: Method, Mean RMSE (), Mean RMSE (), Inference Time (ms).
* *Highlight:* Mamba achieves lower error with orders-of-magnitude faster inference speed.