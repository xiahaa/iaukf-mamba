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
        Implements Eq 17 & 18 from Wang et al. (2022) - EXACT formula
        Q_{k+1} = (1-d_k)Q_k + d_k * [K*epsilon*epsilon^T*K^T + P_k - sigma_cov]
        
        where sigma_cov = sum(W_i^c * (X_{k|k-1}^(i) - x_{k|k-1})(X_{k|k-1}^(i) - x_{k|k-1})^T)
        """
        d_k = (1 - self.b_factor) / (1 - self.b_factor**(self.k_step + 1))

        # Term 1: K * epsilon * epsilon^T * K^T
        term1 = np.dot(K, np.dot(np.outer(residual, residual), K.T))

        # Term 2: P_k (posterior covariance)
        term2 = self.P

        # Term 3: Sigma point covariance (Eq 17 - exact computation)
        # This is the sum: sum_{i=0}^{2n} W_i^(c) * (X_{k|k-1}^(i) - x_{k|k-1})(X_{k|k-1}^(i) - x_{k|k-1})^T
        # Note: self.sigmas_f contains the propagated sigma points X_{k|k-1}^(i)
        # and self.x_pred is x_{k|k-1}
        sigma_cov = np.zeros((self.n, self.n))
        for i in range(2*self.n + 1):
            diff = self.sigmas_f[i] - self.x_pred
            sigma_cov += self.Wc[i] * np.outer(diff, diff)

        # Exact update from Eq 17
        Q_next = (1 - d_k) * self.Q + d_k * (term1 + term2 - sigma_cov)

        # Make symmetric
        Q_next = (Q_next + Q_next.T) / 2

        # Constraint (Eq 18): Check Positive Definiteness
        try:
            cholesky(Q_next)
            self.Q = Q_next
        except np.linalg.LinAlgError:
            # Fallback to biased estimator (Eq 18)
            # Q_{k+1} = (1-d_k)*Q_k + d_k * [diag(K*epsilon*epsilon^T*K^T) + K*P_zz*K^T]
            diag_term = np.diag(np.diag(term1))  # Extract diagonal of K*epsilon*epsilon^T*K^T
            bias_term = np.dot(K, np.dot(S, K.T))  # K*P_zz*K^T where S is P_zz
            Q_next_biased = (1 - d_k) * self.Q + d_k * (diag_term + bias_term)
            
            # Make symmetric
            Q_next_biased = (Q_next_biased + Q_next_biased.T) / 2
            
            try:
                cholesky(Q_next_biased)
                self.Q = Q_next_biased
            except np.linalg.LinAlgError:
                # Final fallback: keep old Q with small inflation
                self.Q = self.Q * 1.01 + np.eye(self.n) * 1e-6

        # Additional safeguard: clip Q values to prevent explosion
        self.Q = np.clip(self.Q, 1e-8, 1.0)


class IAUKFMultiSnapshot:
    """
    IAUKF with Multiple Measurement Snapshots support.
    
    Implements the augmented state-space model under multiple measurement snapshots
    as described in Section IV.C of the paper (Eq 32-38).
    
    The augmented state vector includes system states from multiple time steps
    plus the parameter vector:
    X_k = [x_1, x_2, ..., x_t, p_k]^T
    
    The measurement vector is similarly augmented:
    Z_k = [z_1, z_2, ..., z_t]^T
    """
    
    def __init__(self, system_model, x0, P0, Q0, R, num_snapshots=5):
        """
        Initialize IAUKF with multi-snapshot support.
        
        Args:
            system_model: System model with state_transition and measurement_function
            x0: Initial state estimate (single snapshot)
            P0: Initial covariance (single snapshot)
            Q0: Initial process noise covariance (single snapshot)
            R: Measurement noise covariance (single snapshot)
            num_snapshots: Number of measurement snapshots to use (default: 5)
        """
        self.sys = system_model
        self.num_snapshots = num_snapshots
        
        # Determine state dimensions
        self.n_single = len(x0)  # Single snapshot state dimension
        
        # Augmented state: [x_1, x_2, ..., x_t, p]
        # For simplicity, we assume x contains both system states and parameters
        # The last 2 elements are parameters (R, X of line)
        self.n_params = 2  # Number of parameters (R, X)
        self.n_sys_single = self.n_single - self.n_params  # System state dimension
        
        # Augmented state dimension: t * n_sys + n_params
        self.n = num_snapshots * self.n_sys_single + self.n_params
        
        # Initialize augmented state: replicate initial system state
        self.x = np.zeros(self.n)
        for i in range(num_snapshots):
            self.x[i*self.n_sys_single:(i+1)*self.n_sys_single] = x0[:self.n_sys_single]
        self.x[-self.n_params:] = x0[-self.n_params:]  # Parameters
        
        # Initialize augmented covariance
        self.P = np.zeros((self.n, self.n))
        for i in range(num_snapshots):
            idx_start = i * self.n_sys_single
            idx_end = (i+1) * self.n_sys_single
            self.P[idx_start:idx_end, idx_start:idx_end] = P0[:self.n_sys_single, :self.n_sys_single]
        self.P[-self.n_params:, -self.n_params:] = P0[-self.n_params:, -self.n_params:]
        
        # Initialize augmented process noise
        self.Q = np.zeros((self.n, self.n))
        for i in range(num_snapshots):
            idx_start = i * self.n_sys_single
            idx_end = (i+1) * self.n_sys_single
            self.Q[idx_start:idx_end, idx_start:idx_end] = Q0[:self.n_sys_single, :self.n_sys_single]
        self.Q[-self.n_params:, -self.n_params:] = Q0[-self.n_params:, -self.n_params:]
        
        # Measurement noise (replicated for each snapshot)
        self.R_single = R
        self.R = np.kron(np.eye(num_snapshots), R)  # Block diagonal
        
        # UKF parameters
        self.alpha = 1e-3
        self.beta = 2
        self.kappa = 0
        self.lam = self.alpha**2 * (self.n + self.kappa) - self.n
        
        # Weights
        self.Wm = np.full(2*self.n + 1, 0.5 / (self.n + self.lam))
        self.Wc = np.full(2*self.n + 1, 0.5 / (self.n + self.lam))
        self.Wm[0] = self.lam / (self.n + self.lam)
        self.Wc[0] = self.lam / (self.n + self.lam) + (1 - self.alpha**2 + self.beta)
        
        # NSE parameters
        self.b_factor = 0.96
        self.k_step = 0
        
        # Buffer for storing snapshot measurements
        self.measurement_buffer = []
    
    def sigma_points(self, x, P):
        """Generate sigma points for augmented state."""
        try:
            L = cholesky((self.n + self.lam) * P, lower=True)
        except np.linalg.LinAlgError:
            L = cholesky((self.n + self.lam) * (P + np.eye(self.n)*1e-6), lower=True)
        
        sigmas = np.zeros((2*self.n + 1, self.n))
        sigmas[0] = x
        for i in range(self.n):
            sigmas[i+1] = x + L[:, i]
            sigmas[self.n + i + 1] = x - L[:, i]
        return sigmas
    
    def predict(self):
        """Prediction step for multi-snapshot model."""
        # Generate sigma points
        sigmas = self.sigma_points(self.x, self.P)
        
        # Propagate through state transition (Eq 35)
        # State transition: system states shift, parameters stay constant
        self.sigmas_f = np.zeros_like(sigmas)
        for idx, s in enumerate(sigmas):
            # Extract snapshots and parameters
            snapshots = []
            for i in range(self.num_snapshots):
                snap = s[i*self.n_sys_single:(i+1)*self.n_sys_single]
                snapshots.append(snap)
            params = s[-self.n_params:]
            
            # State transition: shift snapshots (new snapshot predicted from last)
            # Create augmented single-snapshot state for prediction
            x_single = np.concatenate([snapshots[-1], params])
            x_next_sys = self.sys.state_transition(x_single)[:self.n_sys_single]
            
            # Shift: drop oldest, add newest prediction
            new_snapshots = snapshots[1:] + [x_next_sys]
            
            # Reconstruct augmented state
            self.sigmas_f[idx, :] = np.concatenate(new_snapshots + [params])
        
        # Predict mean and covariance
        self.x_pred = np.dot(self.Wm, self.sigmas_f)
        self.P_pred = np.zeros((self.n, self.n))
        for i in range(2*self.n + 1):
            diff = self.sigmas_f[i] - self.x_pred
            self.P_pred += self.Wc[i] * np.outer(diff, diff)
        self.P_pred += self.Q
        
        # Ensure symmetry
        self.P_pred = (self.P_pred + self.P_pred.T) / 2
        self.P_pred += np.eye(self.n) * 1e-9
    
    def update(self, measurements):
        """
        Update step for multi-snapshot model.
        
        Args:
            measurements: List of measurement vectors [z_1, z_2, ..., z_t]
                         or single measurement (will be buffered)
        """
        # Handle single vs multi measurements
        if not isinstance(measurements, list):
            # Single measurement - add to buffer
            self.measurement_buffer.append(measurements)
            if len(self.measurement_buffer) < self.num_snapshots:
                # Not enough snapshots yet - just update state
                self.x = self.x_pred
                self.P = self.P_pred
                return self.x
            # Keep only last num_snapshots measurements
            self.measurement_buffer = self.measurement_buffer[-self.num_snapshots:]
            measurements = self.measurement_buffer
        
        # Create augmented measurement vector (Eq 34)
        z = np.concatenate(measurements)
        
        # Propagate sigma points through measurement function (Eq 36)
        Z_sigmas = []
        for s in self.sigmas_f:
            # Extract snapshots and parameters
            h_list = []
            params = s[-self.n_params:]
            
            for i in range(self.num_snapshots):
                snap = s[i*self.n_sys_single:(i+1)*self.n_sys_single]
                x_single = np.concatenate([snap, params])
                h_i = self.sys.measurement_function(x_single)
                h_list.append(h_i)
            
            Z_sigmas.append(np.concatenate(h_list))
        
        Z_sigmas = np.array(Z_sigmas)
        
        # Predict measurement mean
        z_pred = np.dot(self.Wm, Z_sigmas)
        
        # Innovation covariance
        S = np.zeros((len(z), len(z)))
        for i in range(2*self.n + 1):
            diff = Z_sigmas[i] - z_pred
            S += self.Wc[i] * np.outer(diff, diff)
        S += self.R
        S = (S + S.T) / 2
        
        # Cross covariance
        Pxz = np.zeros((self.n, len(z)))
        for i in range(2*self.n + 1):
            diff_x = self.sigmas_f[i] - self.x_pred
            diff_z = Z_sigmas[i] - z_pred
            Pxz += self.Wc[i] * np.outer(diff_x, diff_z)
        
        # Kalman gain and update
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        
        K = np.dot(Pxz, S_inv)
        residual = z - z_pred
        
        self.x = self.x_pred + np.dot(K, residual)
        self.P = self.P_pred - np.dot(K, np.dot(S, K.T))
        
        # Ensure P is symmetric and positive definite
        self.P = (self.P + self.P.T) / 2
        min_eig = np.min(np.linalg.eigvalsh(self.P))
        if min_eig < 0:
            self.P += np.eye(self.n) * (abs(min_eig) + 1e-6)
        
        # Adaptive noise update
        self.adaptive_noise_update(residual, K, S)
        
        self.k_step += 1
        return self.x
    
    def adaptive_noise_update(self, residual, K, S):
        """NSE update for multi-snapshot (same as single-snapshot IAUKF)."""
        d_k = (1 - self.b_factor) / (1 - self.b_factor**(self.k_step + 1))
        
        # Term 1: K * epsilon * epsilon^T * K^T
        term1 = np.dot(K, np.dot(np.outer(residual, residual), K.T))
        
        # Term 2: P_k (posterior covariance)
        term2 = self.P
        
        # Term 3: Sigma point covariance (exact computation)
        sigma_cov = np.zeros((self.n, self.n))
        for i in range(2*self.n + 1):
            diff = self.sigmas_f[i] - self.x_pred
            sigma_cov += self.Wc[i] * np.outer(diff, diff)
        
        # Exact update from Eq 17
        Q_next = (1 - d_k) * self.Q + d_k * (term1 + term2 - sigma_cov)
        Q_next = (Q_next + Q_next.T) / 2
        
        # Check positive definiteness (Eq 18)
        try:
            cholesky(Q_next)
            self.Q = Q_next
        except np.linalg.LinAlgError:
            # Fallback to biased estimator
            diag_term = np.diag(np.diag(term1))
            bias_term = np.dot(K, np.dot(S, K.T))
            Q_next_biased = (1 - d_k) * self.Q + d_k * (diag_term + bias_term)
            Q_next_biased = (Q_next_biased + Q_next_biased.T) / 2
            
            try:
                cholesky(Q_next_biased)
                self.Q = Q_next_biased
            except np.linalg.LinAlgError:
                self.Q = self.Q * 1.01 + np.eye(self.n) * 1e-6
        
        self.Q = np.clip(self.Q, 1e-8, 1.0)
    
    def get_parameters(self):
        """Extract estimated parameters from augmented state."""
        return self.x[-self.n_params:]