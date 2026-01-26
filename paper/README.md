# Paper: Graph Mamba for Power Grid Parameter Estimation

## Files in This Directory

### Main Paper Files
- **`main.tex`** - Main LaTeX manuscript with Introduction, Related Work, Problem Formulation, Methodology, and Experimental Setup sections complete
- **`results_section.tex`** - Complete Results section (copy into main.tex)
- **`discussion_conclusion.tex`** - Complete Discussion and Conclusion sections (copy into main.tex)
- **`references.bib`** - Bibliography file with all citations

### Figures (from `../tmp/`)
All figures are already generated at 300 DPI and ready for inclusion:
- `fig1_architecture.png` - System architecture diagram
- `fig2_training_curves.png` - Training curves (all phases)
- `fig3_tracking_performance.png` - IAUKF vs Mamba tracking
- `fig4_error_distribution.png` - Statistical analysis
- `fig5_computational_efficiency.png` - Speed comparison

### Tables (from `../tmp/`)
All LaTeX tables are ready:
- `table1_main_comparison.tex` - Main performance results
- `table2_ablation.tex` - Component analysis
- `table3_architecture.tex` - Model specifications
- `table4_efficiency.tex` - Computational comparison
- `table5_statistics.tex` - Statistical summary
- `table6_phases.tex` - Experimental validation
- `table7_related_work.tex` - SOTA comparison

## Paper Structure

### Current Status

✅ **Complete Sections**:
1. Abstract (186 words)
2. Introduction (complete with 4 subsections)
3. Related Work (complete with subsections on traditional methods, DL, and gaps)
4. Problem Formulation (complete)
5. Methodology (complete with architecture details)
6. Experimental Setup (complete)

📝 **To Be Integrated**:
- Results section (in `results_section.tex`)
- Discussion section (in `discussion_conclusion.tex`)
- Conclusion section (in `discussion_conclusion.tex`)

## How to Complete the Paper

### Step 1: Integrate Results Section

Open `main.tex` and find the Results section (around line 500):

```latex
% RESULTS
\section{Results}
\label{sec:results}
```

Replace the placeholder content with the content from `results_section.tex`.

### Step 2: Integrate Discussion and Conclusion

Replace the Discussion and Conclusion placeholder sections with content from `discussion_conclusion.tex`.

### Step 3: Update Citations

The `references.bib` file contains placeholder citations. You need to:

1. **Update the baseline paper citation** - Replace the `baseline_paper` entry with the actual IAUKF paper from `../docs/ref.pdf`

2. **Add missing references** - Some citations are placeholders and need real references:
   - Recent GNN papers for power systems
   - Latest Mamba paper (2023)
   - Your institution/funding sources

### Step 4: Customize Author Information

In `main.tex`, update:

```latex
\author{%
\IEEEauthorblockN{Your Name\IEEEauthorrefmark{1}}
\IEEEauthorblockA{\IEEEauthorrefmark{1}Department of Electrical Engineering,\\
Your University,\\
Email: your.email@university.edu}
}
```

### Step 5: Add Acknowledgments

At the end of the Conclusion section, update:

```latex
\noindent\textbf{Acknowledgments}: This work was supported by [funding sources].
The authors thank [collaborators] for helpful discussions.
```

## Compiling the Paper

### Using Command Line

```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Using Overleaf

1. Create a new project on Overleaf
2. Upload `main.tex`, `references.bib`
3. Copy all figures from `../tmp/` to the project
4. Compile (Overleaf handles the build process automatically)

### Using TeXShop/TeXworks/Other Editor

1. Open `main.tex`
2. Set compiler to pdfLaTeX
3. Compile with bibliography support
4. View the PDF output

## Key Numbers to Remember

When writing/revising, keep these key results in mind:

- **65% improvement** over IAUKF
- **3.18% R error, 3.06% X error** (vs 9.13%, 8.61%)
- **5× faster inference** (10ms vs 50ms)
- **20× faster adaptation** (1-2 steps vs 40+)
- **71% variance reduction** (±2.7% vs ±9.2%)
- **2.3× more reliable** (78.6% vs 34.2% with <5% error)
- **62,346 parameters** (efficient model)

## Current Word/Page Count

**Estimated**: 12-14 pages (IEEE two-column format)

Breakdown:
- Abstract: 186 words
- Introduction: ~2 pages
- Related Work: ~1.5 pages
- Problem Formulation: ~1 page
- Methodology: ~3 pages
- Experimental Setup: ~2 pages
- Results: ~4 pages
- Discussion: ~1.5 pages
- Conclusion: ~0.5 pages

**Target for IEEE Transactions**: 16 pages typical

## Submission Checklist

Before submitting to IEEE Transactions on Power Systems:

### Content
- [ ] All sections complete and integrated
- [ ] All figures included and referenced
- [ ] All tables included and referenced
- [ ] All citations complete and properly formatted
- [ ] Abstract within 250 word limit
- [ ] Keywords appropriate and complete

### Formatting
- [ ] IEEE Transactions style (IEEEtran.cls)
- [ ] Two-column format
- [ ] Figures at 300 DPI or higher
- [ ] Equations numbered
- [ ] Consistent notation throughout

### Technical
- [ ] All claims supported by results
- [ ] No typos or grammatical errors
- [ ] Consistent terminology
- [ ] All acronyms defined on first use
- [ ] Limitations acknowledged

### Supplementary
- [ ] Create supplementary materials file
- [ ] Prepare code repository for release
- [ ] Write cover letter
- [ ] Complete author information forms

## Tips for Writing

### Do's
✅ Lead with strongest results (65% improvement)
✅ Use active voice ("We propose" not "is proposed")
✅ Refer to figures frequently
✅ Quantify everything
✅ Be generous to prior work
✅ Acknowledge limitations honestly

### Don'ts
❌ Overclaim ("revolutionize" → "improve")
❌ Hide limitations
❌ Use vague terms ("better" → "65% better")
❌ Forget to cite related work
❌ Rush the abstract (write it last)

## Getting Help

- **For technical details**: See `../docs/PAPER_READY_SUMMARY.md`
- **For writing guidance**: See `../docs/PUBLICATION_GUIDE.md`
- **For quick start**: See `../QUICK_START_PAPER.md`
- **For all results**: See `../docs/FINAL_RESULTS.md`

## Version Control

**Current Version**: Draft v1.0
**Date**: January 26, 2026
**Status**: Ready for integration and final polishing

## Next Steps

1. **This week**: Integrate Results, Discussion, Conclusion sections
2. **Next week**: Proofread and polish
3. **Week 3**: Internal review and revision
4. **Week 4**: Submit to IEEE Transactions on Power Systems

Good luck with your paper! 🎓📝✨
