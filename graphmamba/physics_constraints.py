"""
Advanced Physics Constraints for Power System Parameter Estimation
==================================================================

Implements accurate AC power flow constraints for physics-informed training.
"""

import torch
import torch.nn as nn
import numpy as np


class ACPowerFlowResidual(nn.Module):
    """
    Computes AC power flow residuals using estimated line parameters.
    
    For a line between buses i and j with estimated R and X:
    - Computes power flow using estimated parameters
    - Compares with measured power injections
    - Returns residual that should be minimized
    """
    
    def __init__(self, base_mva=1.0):
        super(ACPowerFlowResidual, self).__init__()
        self.base_mva = base_mva
        
    def forward(self, r_est, x_est, v_i, v_j, theta_i, theta_j, p_meas, q_meas):
        """
        Compute power flow residual.
        
        Args:
            r_est, x_est: Estimated line parameters [batch]
            v_i, v_j: Voltage magnitudes at from/to buses [batch]
            theta_i, theta_j: Voltage angles at from/to buses [batch]
            p_meas, q_meas: Measured power injections at from bus [batch]
        
        Returns:
            residual: Combined P and Q residual [batch]
        """
        # Compute admittance
        g = r_est / (r_est**2 + x_est**2)  # Conductance
        b = -x_est / (r_est**2 + x_est**2)  # Susceptance
        
        # Angle difference
        delta_theta = theta_i - theta_j
        
        # Compute power flow using estimated parameters
        # P_ij = V_i^2 * G - V_i * V_j * (G * cos(delta) + B * sin(delta))
        # Q_ij = -V_i^2 * B - V_i * V_j * (G * sin(delta) - B * cos(delta))
        
        p_calc = v_i**2 * g - v_i * v_j * (g * torch.cos(delta_theta) + b * torch.sin(delta_theta))
        q_calc = -v_i**2 * b - v_i * v_j * (g * torch.sin(delta_theta) - b * torch.cos(delta_theta))
        
        # Residual
        p_residual = (p_calc - p_meas) ** 2
        q_residual = (q_calc - q_meas) ** 2
        
        return p_residual + q_residual


class EnhancedPhysicsLoss(nn.Module):
    """
    Enhanced physics-informed loss with multiple constraints.
    """
    
    def __init__(self, lambda_phy=0.1, lambda_smooth=0.01, lambda_range=0.01):
        super(EnhancedPhysicsLoss, self).__init__()
        self.lambda_phy = lambda_phy
        self.lambda_smooth = lambda_smooth
        self.lambda_range = lambda_range
        self.ac_residual = ACPowerFlowResidual()
        self.mse = nn.MSELoss()
        
    def forward(self, pred_params, true_params, 
                measurements=None, branch_info=None):
        """
        Compute enhanced physics loss.
        
        Args:
            pred_params: [batch, 2] predicted [R, X]
            true_params: [batch, 2] ground truth [R, X]
            measurements: dict with 'v', 'theta', 'p', 'q' [batch, nodes]
            branch_info: dict with 'from_bus', 'to_bus'
        """
        batch_size = pred_params.size(0)
        
        # 1. Data loss
        loss_data = self.mse(pred_params, true_params)
        
        # 2. Physics loss (AC power flow)
        loss_phy = torch.tensor(0.0, device=pred_params.device)
        if measurements is not None and branch_info is not None:
            from_bus = branch_info['from_bus']
            to_bus = branch_info['to_bus']
            
            for b in range(batch_size):
                r = pred_params[b, 0]
                x = pred_params[b, 1]
                
                v_i = measurements['v'][b, from_bus]
                v_j = measurements['v'][b, to_bus]
                theta_i = measurements['theta'][b, from_bus]
                theta_j = measurements['theta'][b, to_bus]
                p_meas = measurements['p'][b, from_bus]
                q_meas = measurements['q'][b, from_bus]
                
                residual = self.ac_residual(
                    r.unsqueeze(0), x.unsqueeze(0),
                    v_i.unsqueeze(0), v_j.unsqueeze(0),
                    theta_i.unsqueeze(0), theta_j.unsqueeze(0),
                    p_meas.unsqueeze(0), q_meas.unsqueeze(0)
                )
                
                loss_phy += residual.mean()
            
            loss_phy = loss_phy / batch_size
        
        # 3. Smoothness loss (R/X ratio)
        ratio = pred_params[:, 0] / (pred_params[:, 1] + 1e-8)
        # Typical R/X for distribution lines: 0.2 to 2.0
        loss_smooth = torch.mean(
            torch.relu(ratio - 2.0) + torch.relu(0.2 - ratio)
        )
        
        # 4. Range loss (keep parameters in reasonable bounds)
        # R typically 0.1 to 2.0 ohm/km
        # X typically 0.1 to 1.0 ohm/km
        r_min, r_max = 0.1, 2.0
        x_min, x_max = 0.1, 1.0
        
        loss_range = torch.mean(
            torch.relu(r_min - pred_params[:, 0]) + 
            torch.relu(pred_params[:, 0] - r_max) +
            torch.relu(x_min - pred_params[:, 1]) +
            torch.relu(pred_params[:, 1] - x_max)
        )
        
        # Total loss
        total_loss = (
            loss_data + 
            self.lambda_phy * loss_phy + 
            self.lambda_smooth * loss_smooth +
            self.lambda_range * loss_range
        )
        
        return total_loss, {
            'data': loss_data.item(),
            'physics': loss_phy.item() if isinstance(loss_phy, torch.Tensor) else 0.0,
            'smoothness': loss_smooth.item(),
            'range': loss_range.item()
        }


class ConsistencyLoss(nn.Module):
    """
    Enforces temporal and spatial consistency in predictions.
    """
    
    def __init__(self, lambda_temporal=0.01, lambda_spatial=0.01):
        super(ConsistencyLoss, self).__init__()
        self.lambda_temporal = lambda_temporal
        self.lambda_spatial = lambda_spatial
        
    def forward(self, predictions, sequence_data=None):
        """
        Args:
            predictions: [batch, time, 2] sequence of predictions
            sequence_data: optional context for spatial consistency
        """
        loss = 0
        
        # Temporal consistency: adjacent predictions should be similar
        if predictions.size(1) > 1:
            temporal_diff = torch.diff(predictions, dim=1)
            loss_temporal = torch.mean(temporal_diff ** 2)
            loss += self.lambda_temporal * loss_temporal
        
        # Spatial consistency: predictions should vary smoothly across similar conditions
        # (simplified version - in practice, use graph structure)
        
        return loss


def compute_line_flows(network, r_mat, x_mat):
    """
    Compute power flows through all lines given parameters.
    
    Args:
        network: pandapower network
        r_mat, x_mat: [n_branches] estimated parameters
    
    Returns:
        p_from, q_from, p_to, q_to: [n_branches] power flows
    """
    import pandapower as pp
    
    # Update line parameters
    for i, (r, x) in enumerate(zip(r_mat, x_mat)):
        network.line.at[i, 'r_ohm_per_km'] = r
        network.line.at[i, 'x_ohm_per_km'] = x
    
    # Run power flow
    try:
        pp.runpp(network, algorithm='nr', numba=False, init='results')
    except:
        return None, None, None, None
    
    # Get flows
    p_from = network.res_line['p_from_mw'].values
    q_from = network.res_line['q_from_mvar'].values
    p_to = network.res_line['p_to_mw'].values
    q_to = network.res_line['q_to_mvar'].values
    
    return p_from, q_from, p_to, q_to


def validate_physics_consistency(pred_r, pred_x, network, measurements):
    """
    Validate that predicted parameters produce consistent power flows.
    
    Returns consistency score (lower is better).
    """
    p_from, q_from, p_to, q_to = compute_line_flows(network, [pred_r], [pred_x])
    
    if p_from is None:
        return float('inf')
    
    # Compare with measurements
    p_error = abs(p_from[0] - measurements['p_from'])
    q_error = abs(q_from[0] - measurements['q_from'])
    
    return p_error + q_error
