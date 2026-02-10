"""
Generate Ablation Study Figure for Graph-Mamba
===============================================

Publication-quality figure showing component contributions.
Style: Light colors, Arial font, comprehensive labels.
"""

import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pickle

# Set publication style
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
rcParams['font.size'] = 12
rcParams['axes.titlesize'] = 14
rcParams['axes.labelsize'] = 13
rcParams['legend.fontsize'] = 11
rcParams['xtick.labelsize'] = 11
rcParams['ytick.labelsize'] = 11

# Beautiful color palette
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#C73E1D',
    'light_blue': '#A8DADC',
    'light_gray': '#E5E5E5',
    'dark_gray': '#457B9D',
    'green': '#2A9D8F',
    'purple': '#9B5DE5',
    'yellow': '#F4A261',
    'mamba_green': '#06D6A0',  # Highlight for Mamba
}

def save_figure(fig, name, dpi=300):
    """Save figure with tight layout."""
    fig.tight_layout()
    fig.savefig(f'../tmp/{name}.png', dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(f'../tmp/{name}.pdf', bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: {name}.png/pdf")


def load_ablation_data():
    """Load ablation study results."""
    with open('../tmp/ablation_results.pkl', 'rb') as f:
        data = pickle.load(f)
    return data


def generate_ablation_figure():
    """Generate comprehensive ablation study figure."""
    print("[Ablation] Generating ablation study figure...")
    
    # Load data
    data = load_ablation_data()
    
    # Sort by R error (ascending - best first)
    data_sorted = sorted(data, key=lambda x: x['r_error'])
    
    names = [d['name'] for d in data_sorted]
    r_errors = [d['r_error'] for d in data_sorted]
    x_errors = [d['x_error'] for d in data_sorted]
    params = [d['params'] for d in data_sorted]
    
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    
    # Color scheme - highlight best model
    bar_colors = []
    for name in names:
        if 'Mamba (Full)' in name:
            bar_colors.append(COLORS['mamba_green'])  # Highlight best
        elif 'Mamba' in name:
            bar_colors.append(COLORS['green'])
        elif 'LSTM' in name:
            bar_colors.append(COLORS['accent'])
        elif 'GNN' in name and 'Only' in name:
            bar_colors.append(COLORS['primary'])
        else:
            bar_colors.append(COLORS['light_gray'])
    
    # (a) R Error comparison (horizontal bar)
    ax1 = fig.add_subplot(gs[0, 0])
    y_pos = np.arange(len(names))
    bars1 = ax1.barh(y_pos, r_errors, color=bar_colors, edgecolor='white', linewidth=1.5, height=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=10)
    ax1.set_xlabel('R Estimation Error (%)', fontweight='bold')
    ax1.set_title('(a) Ablation Study: R Parameter Error', fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    ax1.invert_yaxis()  # Best at top
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, r_errors)):
        width = bar.get_width()
        ax1.text(width + 0.05, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f}%', ha='left', va='center', fontsize=10, fontweight='bold')
    
    # (b) X Error comparison (horizontal bar)
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.barh(y_pos, x_errors, color=bar_colors, edgecolor='white', linewidth=1.5, height=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=10)
    ax2.set_xlabel('X Estimation Error (%)', fontweight='bold')
    ax2.set_title('(b) Ablation Study: X Parameter Error', fontweight='bold', pad=15)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    ax2.invert_yaxis()
    
    for i, (bar, val) in enumerate(zip(bars2, x_errors)):
        width = bar.get_width()
        ax2.text(width + 0.05, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f}%', ha='left', va='center', fontsize=10, fontweight='bold')
    
    # (c) Model Complexity vs Performance
    ax3 = fig.add_subplot(gs[1, 0])
    avg_errors = [(r + x) / 2 for r, x in zip(r_errors, x_errors)]
    
    # Size proportional to inverse error (better = bigger)
    sizes = [3000 / (e + 0.1) for e in avg_errors]
    
    scatter = ax3.scatter(params, avg_errors, s=sizes, c=bar_colors, 
                         edgecolors='white', linewidths=2, alpha=0.8, zorder=5)
    
    # Add labels
    for i, name in enumerate(names):
        # Shorten name for label
        short_name = name.replace('GNN + ', '').replace(' (Full)', '').replace(' + Attn', '+Attn')
        offset = (20, 10) if 'Mamba' in name else (20, -15)
        ax3.annotate(short_name, xy=(params[i], avg_errors[i]), 
                    xytext=offset, textcoords='offset points',
                    fontsize=9, fontweight='bold' if 'Mamba' in name else 'normal',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    ax3.set_xlabel('Model Parameters', fontweight='bold')
    ax3.set_ylabel('Average Error (R + X) / 2 (%)', fontweight='bold')
    ax3.set_title('(c) Model Complexity vs Performance\n(Bubble size ∝ 1/error)', fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_axisbelow(True)
    
    # (d) Component contribution analysis
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate improvements over baseline
    baseline_error = 3.23  # MLP Baseline R error
    mamba_error = 3.18   # GNN + Mamba R error
    
    improvements = {
        'GNN Only\n(vs MLP)': baseline_error - 3.24,
        'LSTM Only\n(vs MLP)': baseline_error - 3.23,
        'GNN + LSTM\n(vs GNN)': 3.24 - 3.29,
        'GNN + Mamba\n(vs GNN)': 3.24 - 3.18,
        'GNN + Mamba\n(vs LSTM)': 3.23 - 3.18,
    }
    
    comp_names = list(improvements.keys())
    comp_values = list(improvements.values())
    comp_colors = [COLORS['primary'], COLORS['accent'], COLORS['accent'], 
                   COLORS['mamba_green'], COLORS['mamba_green']]
    
    bars4 = ax4.barh(comp_names, comp_values, color=comp_colors, 
                     edgecolor='white', linewidth=1.5, height=0.6)
    ax4.set_xlabel('R Error Improvement (%)', fontweight='bold')
    ax4.set_title('(d) Component Contribution Analysis\n(Positive = Better)', fontweight='bold', pad=15)
    ax4.axvline(x=0, color='black', linewidth=0.8)
    ax4.grid(axis='x', alpha=0.3, linestyle='--')
    ax4.set_axisbelow(True)
    
    for bar, val in zip(bars4, comp_values):
        width = bar.get_width()
        if abs(width) > 0.01:
            sign = '+' if width > 0 else ''
            ax4.text(width + (0.02 if width > 0 else -0.02), 
                    bar.get_y() + bar.get_height()/2, 
                    f'{sign}{width:.2f}%', 
                    ha='left' if width > 0 else 'right', 
                    va='center', fontsize=10, fontweight='bold')
    
    # Main title
    fig.suptitle('Graph-Mamba Ablation Study: Component Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    save_figure(fig, 'fig_ablation_study')
    plt.close()
    
    # Also create a summary table figure
    generate_ablation_table(data_sorted)


def generate_ablation_table(data_sorted):
    """Generate a table figure with ablation results."""
    print("[Ablation] Generating summary table...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    headers = ['Model Variant', 'R Error (%)', 'X Error (%)', 'Avg Error (%)', 'Parameters', 'vs Best']
    
    table_data = []
    best_r = min([d['r_error'] for d in data_sorted])
    
    for d in data_sorted:
        avg = (d['r_error'] + d['x_error']) / 2
        vs_best = f'+{d["r_error"] - best_r:.2f}%' if d['r_error'] > best_r else 'Best'
        table_data.append([
            d['name'],
            f"{d['r_error']:.2f}",
            f"{d['x_error']:.2f}",
            f"{avg:.2f}",
            f"{d['params']:,}",
            vs_best
        ])
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.25, 0.12, 0.12, 0.12, 0.15, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor(COLORS['primary'])
        cell.set_text_props(weight='bold', color='white')
    
    # Highlight best model row
    best_idx = 0  # Already sorted by R error
    for i in range(len(headers)):
        cell = table[(1, i)]  # Row 1 (after header)
        cell.set_facecolor(COLORS['light_blue'])
        cell.set_text_props(weight='bold')
    
    # Alternate row colors
    for i in range(2, len(table_data) + 1):
        if i % 2 == 0:
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#F5F5F5')
    
    plt.title('Ablation Study: Detailed Results Summary', 
              fontsize=14, fontweight='bold', pad=20)
    
    save_figure(fig, 'fig_ablation_table')
    plt.close()


def main():
    print("=" * 70)
    print("Generating Ablation Study Figures")
    print("=" * 70)
    print()
    
    generate_ablation_figure()
    
    print()
    print("=" * 70)
    print("Ablation figures generated successfully!")
    print("=" * 70)
    print()
    print("Output files:")
    print("  - fig_ablation_study.png/pdf")
    print("  - fig_ablation_table.png/pdf")


if __name__ == '__main__':
    main()
