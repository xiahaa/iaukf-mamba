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

        # Ensure symmetry and positive definiteness
        self.P_pred = (self.P_pred + self.P_pred.T) / 2
        # Add small regularization to maintain numerical stability
        self.P_pred += np.eye(self.n) * 1e-9

    def update(self, z):
        # 1. Propagate Sigma Points through Measurement Function
        # Reuse predicted sigma points instead of regenerating
        sigmas_pred = self.sigmas_f  # Use already propagated sigma points

        Z_sigmas = []
        failed_count = 0
        for idx, s in enumerate(sigmas_pred):
            try:
                h = self.sys.measurement_function(s)
                if not isinstance(h, np.ndarray):
                    print(f"Warning: measurement_function returned {type(h)}: {h}")
                    h = np.array(h)
                Z_sigmas.append(h)
            except Exception as e:
                failed_count += 1
                if failed_count <= 2:  # Only print first few failures
                    print(f"Warning: Measurement failed for sigma {idx}: {type(e).__name__}: {str(e)[:100]}")
                # Use previous successful measurement or skip
                if len(Z_sigmas) > 0:
                    Z_sigmas.append(Z_sigmas[-1])  # Reuse last good measurement
                else:
                    # First sigma point failed, return prediction
                    self.x = self.x_pred
                    self.P = self.P_pred
                    return self.x

        if failed_count > 0 and failed_count < len(sigmas_pred):
            print(f"  ({failed_count} total measurement failures)")

        Z_sigmas = np.array(Z_sigmas)

        # 2. Predict Measurement Mean
        z_pred = np.dot(self.Wm, Z_sigmas)

        # 3. Innovation Covariance
        S = np.zeros((len(z), len(z)))
        for i in range(2*self.n + 1):
            diff = Z_sigmas[i] - z_pred
            S += self.Wc[i] * np.outer(diff, diff)
        S += self.R

        # Ensure S is symmetric
        S = (S + S.T) / 2

        # 4. Cross Covariance
        Pxz = np.zeros((self.n, len(z)))
        for i in range(2*self.n + 1):
            diff_x = sigmas_pred[i] - self.x_pred
            diff_z = Z_sigmas[i] - z_pred
            Pxz += self.Wc[i] * np.outer(diff_x, diff_z)

        # 5. Kalman Gain and Update
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            S_inv = np.linalg.pinv(S)

        K = np.dot(Pxz, S_inv)
        residual = z - z_pred

        # Outlier detection: clip large residuals
        residual_norm = np.linalg.norm(residual)
        if residual_norm > 100:  # Threshold for outlier
            print(f"Warning: Large residual detected: {residual_norm:.2f}, clipping...")
            residual = residual / residual_norm * 100

        self.x = self.x_pred + np.dot(K, residual)
        self.P = self.P_pred - np.dot(K, np.dot(S, K.T))

        # Ensure P is symmetric and positive definite
        self.P = (self.P + self.P.T) / 2
        min_eig = np.min(np.linalg.eigvalsh(self.P))
        if min_eig < 0:
            self.P += np.eye(self.n) * (abs(min_eig) + 1e-6)

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
        # term1 = K * epsilon * epsilon^T * K^T
        term1 = np.dot(K, np.dot(np.outer(residual, residual), K.T))

        # Implementing Eq 17 exactly:
        # Bracket = K*e*e^T*K^T + P_posterior - (P_pred - Q_old)
        # So Q_next = (1-d)*Q + d * (term1 + P - P_pred + Q)
        #           = Q + d * (term1 + P - P_pred)
        update_term = term1 + self.P - self.P_pred
        Q_next = self.Q + d_k * update_term

        # Make symmetric
        Q_next = (Q_next + Q_next.T) / 2

        # Constraint (Eq 18): Check Positive Definiteness
        try:
            cholesky(Q_next)
            self.Q = Q_next
        except np.linalg.LinAlgError:
            # Fallback (Eq 18 "otherwise" case):
            # Q_{k+1} = (1-d_k)Q_k + d_k [ diag(term1) + K * S * K^T ]

            # term2 = K * S * K^T (where S is P_zz)
            term2 = np.dot(K, np.dot(S, K.T))

            bracket_fallback = np.diag(np.diag(term1)) + term2

            Q_fallback = (1 - d_k) * self.Q + d_k * bracket_fallback

            # Ensure symmetry
            Q_fallback = (Q_fallback + Q_fallback.T) / 2

            # If still not positive definite, use diagonal trick or inflation
            try:
                cholesky(Q_fallback)
                self.Q = Q_fallback
            except np.linalg.LinAlgError:
                 # Last resort: ensure diagonal is positive
                Q_diag = np.diag(np.abs(np.diag(Q_fallback)))
                self.Q = Q_diag + np.eye(self.n) * 1e-6

        # Additional safeguard: clip Q values to prevent explosion
        self.Q = np.clip(self.Q, 1e-8, 1.0)