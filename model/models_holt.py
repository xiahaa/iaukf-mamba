import numpy as np
import pandapower as pp
from model.models import DistributionSystemModel

class DistributionSystemModelHolt(DistributionSystemModel):
    """
    Distribution system model with Holt's exponential smoothing.

    Note: The paper describes using Holt's smoothing for state prediction.
    However, standard UKF sigma-point propagation requires the state vector
    to contain all memory (Markov property). Since the state vector here
    only contains [V, delta, p], we cannot implement the full recursive Holt's
    logic (which requires Level and Trend history) inside the state transition
    function correctly for each sigma point without augmenting the state.

    To avoid divergence, we default to Identity transition (Random Walk)
    which is robust for tracking time-varying states. The 'Holt' aspect
    is thus theoretical or applied to the mean externally (not implemented here).
    """
    def __init__(self, net, target_line_idx, pmu_indices, alpha_H=0.8, beta_H=0.5):
        super().__init__(net, target_line_idx, pmu_indices)
        self.alpha_H = alpha_H
        self.beta_H = beta_H

    def state_transition(self, x):
        # Default to Identity (same as parent)
        return super().state_transition(x)


class DistributionSystemModelMultiSnapshot(DistributionSystemModel):
    """
    Augmented State-Space Model under Multiple Measurement Snapshots.
    State vector: [x_{k-t+1}, ..., x_k, p]
    Measurement vector: [z_{k-t+1}, ..., z_k]
    """
    def __init__(self, net, target_line_idx, pmu_indices, window_size=5):
        # Initialize parent
        super().__init__(net, target_line_idx, pmu_indices)

        self.window_size = window_size
        self.single_state_dim = 2 * self.num_buses

        # Override state dimension
        # t * (2*N) + 2 parameters
        self.state_dim = self.window_size * self.single_state_dim + 2

    def state_transition(self, x):
        """
        Predicts next window of states.
        Shifts the window left and predicts the new latest state using linear extrapolation
        based on the trend within the window.
        """
        # x has shape (state_dim,)
        p = x[-2:]

        # Extract states: shape (window_size, single_state_dim)
        states = x[:-2].reshape(self.window_size, self.single_state_dim)

        # Shift states left
        new_states = np.zeros_like(states)
        # [x2, x3, ..., xt, ???]
        new_states[:-1] = states[1:]

        # Predict new latest state
        if self.window_size >= 2:
            x_last = states[-1]
            x_prev = states[-2]
            # Linear trend extrapolation: x_{new} = x_{last} + (x_{last} - x_{prev})
            # Add slight damping (0.9) to prevent explosion in non-linear regions
            x_new = x_last + 0.9 * (x_last - x_prev)
        else:
            x_new = states[-1]

        new_states[-1] = x_new

        return np.concatenate([new_states.flatten(), p])

    def measurement_function(self, x):
        """
        Computes predicted measurements for the entire window.
        z = [h(x1, p), h(x2, p), ..., h(xt, p)]
        """
        p = x[-2:]
        states = x[:-2].reshape(self.window_size, self.single_state_dim)

        z_all = []
        for i in range(self.window_size):
            # Construct single state vector [states[i], p] for the parent function
            x_single = np.concatenate([states[i], p])

            # Call parent's single-snapshot measurement function
            z_i = super().measurement_function(x_single)
            z_all.append(z_i)

        return np.concatenate(z_all)
