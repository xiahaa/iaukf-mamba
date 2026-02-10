# Generated Figures for Paper

All figures are saved in `../tmp/` with both PNG (for preview) and PDF (for publication) formats.

## Experiments 1-6 Figures

| Figure | Filename | Description |
|--------|----------|-------------|
| 1 | `fig_exp1_basic_performance` | Bar chart comparing all methods (Single, Multi, GM-50, GM-300) |
| 2 | `fig_exp2_sequence_length` | Line plot showing accuracy improvement with sequence length |
| 3 | `fig_exp3_computational_efficiency` | Dual-panel: runtime (log scale) + speedup factors |
| 4 | `fig_exp4_accuracy_speed_tradeoff` | Scatter plot with Pareto frontier |
| 5 | `fig_exp5_consistency` | Box plots showing variance across 3 runs |
| 6 | `fig_exp6_summary_dashboard` | 6-panel comprehensive analysis dashboard |

## Ablation Study Figures

| Figure | Filename | Description |
|--------|----------|-------------|
| 7 | `fig_ablation_study` | 4-panel ablation analysis (R error, X error, complexity, contributions) |
| 8 | `fig_ablation_table` | Detailed results table with all model variants |

## Design Specifications

- **Font**: Arial, 12-14pt labels, 14-16pt titles
- **Colors**: Light professional palette
  - Ocean blue (`#2E86AB`)
  - Magenta (`#A23B72`)
  - Orange (`#F18F01`)
  - Teal (`#2A9D8F`)
  - Mamba green (`#06D6A0`) - highlighted
- **Features**:
  - White borders on bars/markers
  - Value annotations on all bars
  - Grid lines (light, dashed)
  - Shadow effects on legends
  - Comprehensive labels and titles

## Usage in LaTeX

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.48\textwidth]{../tmp/fig_exp1_basic_performance.pdf}
\caption{Experiment 1: Basic Performance Comparison}
\label{fig:exp1}
\end{figure}
```

## Key Results Summary

### Multi-Shot Comparison (300 steps)
| Method | R Error | Time |
|--------|---------|------|
| Multi-snapshot IAUKF | 0.12% | 92.5s |
| Graph-Mamba | 0.30% | 99ms |
| **Gap** | 2.5× | **934× faster** |

### Ablation Study
- **Best model**: GNN + Mamba (Full) - 3.18% R error, 62K params
- **Attention not needed**: +42% params, minimal improvement
- **GNN alone insufficient**: Needs temporal component
- **Mamba > LSTM**: 0.11% better accuracy
