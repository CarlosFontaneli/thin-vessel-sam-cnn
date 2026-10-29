# Tests

Fast unit tests on CPU, with synthetic images and no datasets, weights or GPU: `make test` (or `pytest`), about ten
seconds.

| File | Checks |
|---|---|
| `test_metrics.py` | Dice, skeleton, radius, thin-vessel recall and precision on synthetic masks, the empty-prediction conventions, mask loading |
| `test_data.py` | crop or upscale of training samples, DRIVE mask names, the FIVES border correction, the HRF/VessMap split sizes |
| `test_models.py` | parameter counts of Table I and of compact models, forward shapes, sliding windows per image (also against MONAI) |
| `test_config.py` | all 36 configurations load and are consistent (AdamW, 1000 epochs, 50-epoch patience), path resolution, the search mapping (`optimizer` and `patience` not applied) |
