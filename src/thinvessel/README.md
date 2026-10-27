# `thinvessel` package

The library behind every stage of the paper. The command-line entry points are in [`../../scripts/`](../../scripts/).

| Module | Paper | Content |
|---|---|---|
| `config.py` | §3.3, App. A | configuration loading, path resolution (`DATA_ROOT`, `WEIGHTS_DIR`, `OUTPUTS_DIR`), the 36 studies, search-space sampling and `apply_params` (sampled values → training configuration) |
| `data/` | §3.1, App. A.1 | `dataset.py`: paired image/mask datasets, training augmentation, FIVES mask correction; `splits.py`: DRIVE, FIVES, HRF and VessMap splits |
| `models/` | §3.2 | LW-Net, ConvUNeXt, ConvNeXt, MedSAM, LiteMedSAM; `build_model` maps a configuration to a model; `third_party/`: vendored SAM and TinyViT |
| `training/` | §3.3, App. A | `module.py`: Lightning module (soft Dice loss, AdamW + cosine schedule, sliding-window validation); `search.py`: one trial, the Optuna search, training of a configuration as it is |
| `evaluation/` | §3.4 | `inference.py`: native and upscaled protocols, test predictions; `metrics.py`: the single metric implementation; `scoring.py`: per-image table of all saved predictions |
| `analysis/` | §3.4, App. A | `cost.py`: parameters and fvcore FLOPs; `convergence.py`: epochs to 95% of the best validation Dice; `search_records.py`: trial records of the searches |

From a configuration to the paper's numbers:

```
configs/<regime>/<dataset>/<model>.yaml
  → config.load_config / resolve_paths
  → training.search.run_search (search) or train_configuration (selected run)   → checkpoint
  → evaluation.inference.predict_dataset                                       → test masks
  → evaluation.scoring.score_predictions (metrics.score_pair per image)         → results/per_image_metrics.csv
```

Every module docstring names the paper section it implements and the units of its values (px; fractions, not %).
Behaviour kept from the experiments, quirks included, is described in
[`../../docs/known_deviations.md`](../../docs/known_deviations.md).
