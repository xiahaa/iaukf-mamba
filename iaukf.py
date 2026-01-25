import numpy as np
from scipy.linalg import cholesky

class IAUKF:
    def __init__(self, system_model, x0, P0, Q0, R):
        self.sys = system_model
        self.x = x0 # State estimate
        self.P = P0 # Covariance
        self.Q = Q0 # Process Noise (Adaptive)
        self.R = R  # Measurement Noise

        self.n = len(x0)

        # Merwe Scaled Sigma Point Parameters
        self.alpha = 1e-3
        self.beta = 2
        self.kappa = 0
        self.lam = self.alpha**2 * (self.n + self.kappa) - self.n

        # Weights
        self.Wm = np.full(2*self.n + 1, 0.5 / (self.n + self.lam))
        self.Wc = np.full(2*self.n + 1, 0.5 / (self.n + self.lam))
        self.Wm[0] = self.lam / (self.n + self.lam)
        self.Wc[0] = self.lam / (self.n + self.lam) + (1 - self.alpha**2 + self.beta)

        # Adaptive NSE parameters
        self.b_factor = 0.96
        self.k_step = 0

    def sigma_points(self, x, P):
        """Generates Sigma Points."""
        # Ensure P is positive definite for Cholesky
        try:
            L = cholesky((self.n + self.lam) * P, lower=True)
        except np.linalg.LinAlgError:
            # Fallback for numerical stability
            L = cholesky((self.n + self.lam) * (P + np.eye(self.n)*1e-6), lower=True)

        sigmas = np.zeros((2*self.n + 1, self.n))
        sigmas[0] = x
        for i in range(self.n):
            sigmas[i+1] = x + L[:, i]
            sigmas[self.n + i + 1] = x - L[:, i]
        return sigmas

    def predict(self):
        # 1. Generate Sigma Points
        sigmas = self.sigma_points(self.x, self.P)

        # 2. Propagate through State Transition
        self.sigmas_f = np.array([self.sys.state_transition(s) for s in sigmas])

        # 3. Predict Mean and Covariance
        self.x_pred = np.dot(self.Wm, self.sigmas_f)
        self.P_pred = np.zeros((self.n, self.n))
        for i in range(2*self.n + 1):
            diff = self.sigmas_f[i] - self.x_pred
            self.P_pred += self.Wc[i] * np.outer(diff, diff)
        self.P_pred += self.Q

    def update(self, z):
        # 1. Propagate Sigma Points through Measurement Function
        # Note: We regenerate sigmas from P_pred or reuse propagated ones?
        # Standard UKF usually regenerates from x_pred, P_pred.
        sigmas_pred = self.sigma_points(self.x_pred, self.P_pred)

        Z_sigmas = np.array([self.sys.measurement_function(s) for s in sigmas_pred])

        # 2. Predict Measurement Mean
        z_pred = np.dot(self.Wm, Z_sigmas)

        # 3. Innovation Covariance
        S = np.zeros((len(z), len(z)))
        for i in range(2*self.n + 1):
            diff = Z_sigmas[i] - z_pred
            S += self.Wc[i] * np.outer(diff, diff)
        S += self.R

        # 4. Cross Covariance
        Pxz = np.zeros((self.n, len(z)))
        for i in range(2*self.n + 1):
            diff_x = sigmas_pred[i] - self.x_pred
            diff_z = Z_sigmas[i] - z_pred
            Pxz += self.Wc[i] * np.outer(diff_x, diff_z)

        # 5. Kalman Gain and Update
        K = np.dot(Pxz, np.linalg.inv(S))
        residual = z - z_pred

        self.x = self.x_pred + np.dot(K, residual)
        self.P = self.P_pred - np.dot(K, np.dot(S, K.T))

        # --- Adaptive Noise Statistic Estimator (NSE) ---
        self.adaptive_noise_update(residual, K, S)

        self.k_step += 1
        return self.x

    def adaptive_noise_update(self, residual, K, S):
        """
        Implements Eq 17 & 18 from Wang et al. (2022)
        Q_{k+1} = (1-d_k)Q_k + d_k * [ ... ]
        """
        d_k = (1 - self.b_factor) / (1 - self.b_factor**(self.k_step + 1))

        # Calculate the correction term
        # Term = K * epsilon * epsilon^T * K^T
        term1 = np.dot(K, np.dot(np.outer(residual, residual), K.T))

        # Simplified update logic typically used in adaptive filtering:
        # Q_new ~ K * residual * residual' * K' + P_posterior ...
        # Based on the paper's specific Equation 17:
        # It approximates Q based on the consistency of residuals.

        # To strictly follow the logic without the complex sigma summation term for Q:
        # We use the simplified form: Q_new = K * (eps*eps^T) * K^T + P - P_pred_minus_Q
        # Or simpler: Q_new = alpha * Q_old + (1-alpha) * (K*eps*eps^T*K^T)

        # Implementing Eq 17 exactly requires the sigma term summation which is P_pred - Q_k.
        # Let's approximate the Bracket term as: K*e*e^T*K^T + P_posterior - (P_pred - Q_old)
        # This simplifies to Q_new = Q_old + d_k * (K*e*e^T*K^T + P_posterior - P_pred)

        # Let's use the explicit update:
        update_term = term1 + self.P - (self.P_pred - self.Q)

        Q_next = (1 - d_k) * self.Q + d_k * update_term

        # Constraint (Eq 18): Check Positive Definiteness
        # If not, use diagonal
        try:
            cholesky(Q_next)
            self.Q = Q_next
        except np.linalg.LinAlgError:
            # Fallback to diagonal
            self.Q = np.diag(np.diag(Q_next))
            # Ensure non-negative diagonal
            self.Q[self.Q < 0] = 1e-6