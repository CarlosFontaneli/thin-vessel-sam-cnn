# Scripts

Command-line entry points, in the order a reproduction runs them. Each takes `--data-root`, `--weights-dir` and
`--outputs-dir` (defaults: `$DATA_ROOT` or `./data`, `$WEIGHTS_DIR` or `./weights`, `$OUTPUTS_DIR` or `./outputs`)
and `--help`. Full commands, costs and the loop over the 36 configurations: [`../docs/reproduction.md`](../docs/reproduction.md).

| Script | Does | Reads | Writes | Cost |
|---|---|---|---|---|
| `check_data.py` | checks the datasets against the paper's split sizes | `data/` | report on screen | seconds |
| `search.py -c <config>` | runs (or resumes) the 100-trial Optuna search of one configuration | `data/`, `weights/` | `outputs/search/<study>/` | 1–345 GPU h |
| `train.py -c <config>` | trains one configuration as it is: the selected run of its search | `data/`, `weights/` | `outputs/train/<study>/` | 0.05–60 GPU h |
| `predict.py -c <config> [--protocol upscaled]` | writes the test predictions of a trained model | checkpoint, `data/` | `outputs/predictions/<study>/<protocol>/` | minutes |
| `evaluate.py` | scores all saved predictions into the per-image table | `outputs/predictions/` | `per_image_metrics.csv` | ~10 CPU min |
| `analyse.py` | cost, convergence of the selected runs, search records | `configs/`, `outputs/search/` | `cost.csv`, `convergence.csv`, `search_trials.csv` | CPU minutes |

`<config>` is `configs/<regime>/<dataset>/<model>.yaml` and `<study>` is `<regime>/<dataset>/<model>`. MedSAM on
DRIVE and VessMap is predicted with `--protocol upscaled`, the protocol of every MedSAM result on these datasets in
the paper. Compare the tables of your own runs with those of the paper's runs in `results/`.
