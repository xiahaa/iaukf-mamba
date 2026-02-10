"""
Generate Graph-Mamba Architecture Schematic Diagram
====================================================

Visual diagram showing the method architecture:
Input -> Feature Norm -> GNN Encoder -> Mamba -> Prediction Head -> Output
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['font.size'] = 10

def draw_architecture():
    """Draw Graph-Mamba architecture schematic."""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    colors = {
        'input': '#E8F4F8',
        'gnn': '#B8E0F0',
        'mamba': '#90E0B8',
        'output': '#F8D7DA',
        'physics': '#FFE5B4',
        'arrow': '#457B9D',
        'text': '#1D3557'
    }
    
    # Title
    ax.text(7, 9.5, 'Graph-Mamba Architecture', fontsize=18, fontweight='bold', 
            ha='center', va='top', color=colors['text'])
    ax.text(7, 9.1, 'Physics-Informed Parameter Estimation for Power Systems', 
            fontsize=12, ha='center', va='top', color='#666666')
    
    # ===================== INPUT BLOCK =====================
    # Input measurements
    input_box = FancyBboxPatch((0.3, 6.5), 2.2, 2, boxstyle="round,pad=0.05",
                                facecolor=colors['input'], edgecolor='#2E86AB', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.4, 7.9, 'Input', fontsize=12, fontweight='bold', ha='center', color=colors['text'])
    ax.text(1.4, 7.4, 'P, Q, V', fontsize=10, ha='center', color=colors['text'])
    ax.text(1.4, 7.0, 'Measurements', fontsize=9, ha='center', color='#666666')
    ax.text(1.4, 6.7, '(T × N × 3)', fontsize=8, ha='center', color='#888888')
    
    # Arrow 1
    ax.annotate('', xy=(2.8, 7.5), xytext=(2.6, 7.5),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
    
    # ===================== FEATURE NORM =====================
    norm_box = FancyBboxPatch((2.9, 6.8), 1.8, 1.4, boxstyle="round,pad=0.05",
                               facecolor='#F0F0F0', edgecolor='#666666', linewidth=1.5)
    ax.add_patch(norm_box)
    ax.text(3.8, 7.7, 'Feature', fontsize=10, fontweight='bold', ha='center', color=colors['text'])
    ax.text(3.8, 7.4, 'Normalization', fontsize=10, fontweight='bold', ha='center', color=colors['text'])
    ax.text(3.8, 7.0, 'Learnable μ, σ', fontsize=8, ha='center', color='#666666')
    
    # Arrow 2
    ax.annotate('', xy=(4.9, 7.5), xytext=(4.7, 7.5),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
    
    # ===================== GNN ENCODER =====================
    gnn_box = FancyBboxPatch((5, 6.2), 2.5, 2.6, boxstyle="round,pad=0.05",
                              facecolor=colors['gnn'], edgecolor='#2E86AB', linewidth=2.5)
    ax.add_patch(gnn_box)
    ax.text(6.25, 8.4, 'GNN Encoder', fontsize=12, fontweight='bold', ha='center', color=colors['text'])
    ax.text(6.25, 8.0, '(Spatial)', fontsize=10, ha='center', color='#444444')
    ax.text(6.25, 7.4, 'GCNConv(3→d)', fontsize=9, ha='center', color=colors['text'])
    ax.text(6.25, 7.0, 'GCNConv(d→d)', fontsize=9, ha='center', color=colors['text'])
    ax.text(6.25, 6.6, 'Edge-aware message passing', fontsize=8, ha='center', color='#666666', style='italic')
    
    # Graph topology annotation
    graph_note = FancyBboxPatch((5.1, 5.5), 2.3, 0.5, boxstyle="round,pad=0.02",
                                 facecolor='white', edgecolor='#2E86AB', linewidth=1, linestyle='--')
    ax.add_patch(graph_note)
    ax.text(6.25, 5.75, 'Graph Topology: IEEE 33-bus', fontsize=8, ha='center', color='#2E86AB')
    
    # Arrow 3
    ax.annotate('', xy=(7.7, 7.5), xytext=(7.5, 7.5),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
    
    # ===================== MAMBA ENCODER =====================
    mamba_box = FancyBboxPatch((7.8, 6.2), 2.8, 2.6, boxstyle="round,pad=0.05",
                                facecolor=colors['mamba'], edgecolor='#2A9D8F', linewidth=2.5)
    ax.add_patch(mamba_box)
    ax.text(9.2, 8.4, 'Mamba Encoder', fontsize=12, fontweight='bold', ha='center', color=colors['text'])
    ax.text(9.2, 8.0, '(Temporal)', fontsize=10, ha='center', color='#444444')
    ax.text(9.2, 7.4, 'Selective SSM', fontsize=9, ha='center', color=colors['text'])
    ax.text(9.2, 7.0, 'd_model=64, d_state=16', fontsize=8, ha='center', color='#666666')
    ax.text(9.2, 6.6, 'Linear complexity O(T)', fontsize=8, ha='center', color='#666666', style='italic')
    
    # Arrow 4
    ax.annotate('', xy=(10.8, 7.5), xytext=(10.6, 7.5),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
    
    # ===================== PREDICTION HEAD =====================
    head_box = FancyBboxPatch((10.9, 6.8), 1.8, 1.4, boxstyle="round,pad=0.05",
                               facecolor='#E8E8E8', edgecolor='#666666', linewidth=1.5)
    ax.add_patch(head_box)
    ax.text(11.8, 7.7, 'Prediction', fontsize=10, fontweight='bold', ha='center', color=colors['text'])
    ax.text(11.8, 7.4, 'Head', fontsize=10, fontweight='bold', ha='center', color=colors['text'])
    ax.text(11.8, 7.0, 'MLP(d→2)', fontsize=9, ha='center', color='#666666')
    
    # Arrow 5
    ax.annotate('', xy=(12.9, 7.5), xytext=(12.7, 7.5),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
    
    # ===================== OUTPUT =====================
    output_box = FancyBboxPatch((13, 6.5), 1, 2, boxstyle="round,pad=0.05",
                                 facecolor=colors['output'], edgecolor='#C73E1D', linewidth=2)
    ax.add_patch(output_box)
    ax.text(13.5, 7.9, 'Output', fontsize=11, fontweight='bold', ha='center', color=colors['text'])
    ax.text(13.5, 7.4, 'R̂, X̂', fontsize=12, ha='center', color=colors['text'])
    ax.text(13.5, 7.0, '(Ω/km)', fontsize=9, ha='center', color='#666666')
    
    # ===================== PHYSICS-INFORMED LOSS =====================
    # Side box for physics loss
    physics_box = FancyBboxPatch((5, 3.5), 6, 2, boxstyle="round,pad=0.05",
                                  facecolor=colors['physics'], edgecolor='#F18F01', linewidth=2,
                                  linestyle='--')
    ax.add_patch(physics_box)
    ax.text(8, 5.1, 'Physics-Informed Loss (Training Only)', fontsize=11, fontweight='bold', 
            ha='center', color=colors['text'])
    
    # Loss components
    ax.text(8, 4.6, r'$\mathcal{L} = \mathcal{L}_{MSE} + \lambda_{phy}\mathcal{L}_{physics} + \lambda_{smooth}\mathcal{L}_{smooth}$', 
            fontsize=10, ha='center', color=colors['text'])
    
    # Loss details
    loss_details = [
        (5.5, 4.15, r'$\mathcal{L}_{MSE} = \|\hat{R}-R\|^2 + \|\hat{X}-X\|^2$'),
        (5.5, 3.85, r'$\mathcal{L}_{physics} = $Power flow residual'),
        (5.5, 3.55, r'$\mathcal{L}_{smooth} = $R/X ratio bounds'),
    ]
    for x, y, text in loss_details:
        ax.text(x, y, text, fontsize=8, ha='left', color='#555555')
    
    # Dashed arrows from output to physics loss
    ax.annotate('', xy=(9.5, 5.5), xytext=(13.5, 6.5),
                arrowprops=dict(arrowstyle='->', color='#F18F01', lw=1.5, linestyle='--'))
    
    # ===================== KEY FEATURES =====================
    features_box = FancyBboxPatch((0.3, 0.8), 6.5, 2.2, boxstyle="round,pad=0.05",
                                   facecolor='#F8F9FA', edgecolor='#2E86AB', linewidth=1.5)
    ax.add_patch(features_box)
    ax.text(3.55, 2.65, 'Key Features', fontsize=11, fontweight='bold', ha='center', color=colors['text'])
    
    features = [
        '✓ Spatial-temporal modeling (GNN + Mamba)',
        '✓ Linear complexity O(T) for sequence length',
        '✓ Physics-informed training (power flow constraints)',
        '✓ Single forward pass (10ms inference)',
    ]
    for i, feat in enumerate(features):
        ax.text(0.6, 2.25 - i*0.35, feat, fontsize=9, ha='left', color='#333333')
    
    # ===================== COMPARISON =====================
    compare_box = FancyBboxPatch((7.2, 0.8), 6.3, 2.2, boxstyle="round,pad=0.05",
                                  facecolor='#F8F9FA', edgecolor='#2A9D8F', linewidth=1.5)
    ax.add_patch(compare_box)
    ax.text(10.35, 2.65, 'vs Traditional Methods', fontsize=11, fontweight='bold', ha='center', color=colors['text'])
    
    comparisons = [
        '• IAUKF: Sequential O(T×n³), 50-100ms/step',
        '• Graph-Mamba: Parallel O(T), 10ms total',
        '• 934× faster than multi-snapshot IAUKF',
        '• Comparable accuracy (0.30% vs 0.12% R error)',
    ]
    for i, comp in enumerate(comparisons):
        ax.text(7.5, 2.25 - i*0.35, comp, fontsize=9, ha='left', color='#333333')
    
    # Bottom labels
    ax.text(7, 0.3, 'Input: P=active power, Q=reactive power, V=voltage magnitude | T=timesteps, N=nodes, d=hidden dim',
            fontsize=8, ha='center', color='#888888', style='italic')
    
    plt.tight_layout()
    plt.savefig('../tmp/fig_architecture_schematic.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('../tmp/fig_architecture_schematic.pdf', bbox_inches='tight', facecolor='white')
    print('  ✓ Saved: fig_architecture_schematic.png/pdf')
    plt.close()


if __name__ == '__main__':
    print('=' * 60)
    print('Generating Graph-Mamba Architecture Schematic')
    print('=' * 60)
    draw_architecture()
    print()
    print('Done! Saved to ../tmp/')
