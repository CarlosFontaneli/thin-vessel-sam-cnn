# The paper: source, figures, tables and statistics

```
paper/
├── manuscript/          LaTeX source (revtex4-1): main.tex, sections/, references.bib
│   └── figures/         generated figure PDFs and table .tex files that the source includes (committed)
├── figures_tables/      one script per figure or table, common.py, run_all.sh, check_plotted.py
│   └── expected/        the values each float of the paper plots or prints
└── statistics/          paper_wilcoxon.py: every pairwise test of the paper
```

## Compile

```bash
make paper                                                   # = cd paper/manuscript && latexmk -pdf -outdir=../../outputs/paper main.tex
```

The PDF is written to `outputs/paper/main.pdf` (not committed; the PDF of each release is attached to it on GitHub).
It needs TeX Live with `latexmk`, BibTeX, the `revtex4-1` class and the packages loaded by `main.tex`. The source is
that of the submitted paper with its comments removed.

## Regenerate the figures and tables

```bash
make figures                                                 # = paper/figures_tables/run_all.sh
```

From the tables in `results/` it computes the statistical tests (`outputs/statistics/all_tests.csv`), then every
figure and table: the PDFs and `.tex` tables that the paper includes go to `manuscript/figures/`, the PNG previews and
the exact plotted values (`plotted/<stem>.csv`) to `outputs/figures/`. Finally `check_plotted.py` compares the plotted
values byte for byte with `expected/`. Fig. 3 also reads the saved test masks (`$OUTPUTS_DIR/predictions/`, written by
`scripts/predict.py`); without them it is skipped and its committed PDF is kept. Point `RESULTS_DIR` at another folder
to plot tables you produced yourself.

| Paper float | Script | Inputs |
|---|---|---|
| Table II (`tab:global_baseline`) | `figures_tables/tab01_global_baseline.py` | per-image metrics, tests |
| Table III (`tab:medsam_vs_cnn`) | `figures_tables/tab03_medsam_vs_cnn.py` | per-image metrics, convergence, tests (after `fig05`) |
| Table IV (`tab:padding`) | `figures_tables/tab02_padding_protocol.py` | per-image metrics (both protocols), cost, configs, tests |
| Table V (`tab:search_space`) | `figures_tables/tabA1_search_space.py` | search trials, configs; prints the total GPU hours |
| Table VI (`tab:compact_models`) | `figures_tables/tabA2_compact_models.py` | configs, cost, per-image metrics |
| Table VII (`tab:patch_size`) | `figures_tables/tabA3_patch_size.py` | search trials |
| Table VIII (`tab:pretraining_gain`) | `figures_tables/tab04_pretraining_gain.py` | per-image metrics, tests (after `fig05`) |
| Tables IX-X (`tab:pvalues`, `tab:pvalues_other`) | `figures_tables/tabA4_pvalues.py` | tests |
| Fig. 1 (`fig:recall_caliber`) | `figures_tables/fig01_recall_by_caliber.py` | per-image metrics |
| Fig. 2 (`fig:thin_pr`) | `figures_tables/fig02_thin_precision_recall.py` | per-image metrics |
| Fig. 3 (`fig:thin_errors`) | `figures_tables/fig07_thin_errors.py` | saved test masks, per-image metrics |
| Fig. 4 (`fig:cost`) | `figures_tables/fig03_cost.py` | cost |
| Fig. 5 (`fig:compact_tradeoff`) | `figures_tables/fig04_compact_tradeoff.py` | per-image metrics, cost |
| Fig. 6 (`fig:pretraining`) | `figures_tables/fig05_pretraining.py` | per-image metrics |
| Fig. 7 (`fig:convergence`) | `figures_tables/fig06_convergence.py` | convergence |
| p-values of every comparison (Appendix C) | `statistics/paper_wilcoxon.py` | per-image metrics |

The script names follow the order in which the floats were written, not their final numbers; the LaTeX label in each
row identifies the float.

Conventions: MedSAM fine-tuned and from scratch on DRIVE and VessMap use the upscaled protocol everywhere except
Table IV. Colour = architecture; fill = regime (fine-tuned filled, from scratch hollow, compact half filled; CNNs
filled). Tables give mean ± standard deviation (ddof = 1) over the test images in %; figures give means.

## Licence of the paper

The MIT licence of the repository covers the code, not the paper. The text and figures of the paper (`manuscript/`)
are © 2026 the authors; all rights reserved. The paper is under review; after publication, cite the published
version.
