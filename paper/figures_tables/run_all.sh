#!/usr/bin/env bash
# Regenerates the statistics and every results figure and table of the paper from results/, in dependency order, into
# paper/manuscript/figures/ (PDF, LaTeX) and outputs/ (PNG previews, plotted values, tests), then checks that the plotted
# values equal those of the paper (expected/).
# Usage: paper/figures_tables/run_all.sh   (from anywhere; the thinvessel package must be installed)
# Fig. 3 (fig07_thin_errors.py) also needs the saved test predictions under $OUTPUTS_DIR/predictions; without them it is
# skipped with a message and the committed PDF of the figure is kept.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

"$PYTHON" "$HERE/../statistics/paper_wilcoxon.py"
for script in tab01_global_baseline fig01_recall_by_caliber fig02_thin_precision_recall fig03_cost \
              fig04_compact_tradeoff fig05_pretraining fig06_convergence tab03_medsam_vs_cnn tab04_pretraining_gain \
              tab02_padding_protocol tabA1_search_space tabA2_compact_models tabA3_patch_size tabA4_pvalues; do
    echo "== $script"
    "$PYTHON" "$HERE/$script.py"
done
OUTPUTS="${OUTPUTS_DIR:-$HERE/../../outputs}"
if [ -d "$OUTPUTS/predictions" ]; then
    echo "== fig07_thin_errors"
    "$PYTHON" "$HERE/fig07_thin_errors.py"
else
    echo "== fig07_thin_errors skipped: no predictions under $OUTPUTS/predictions"
fi
"$PYTHON" "$HERE/check_plotted.py"
