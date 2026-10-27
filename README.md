# Thin retinal vessel segmentation: SAM-based vs convolutional architectures

Code, results and LaTeX source of the paper

> **Evaluating SAM-based and Convolutional Architectures in Thin Retinal Vessel Segmentation: Model Capacity,
> Computational Cost and Foundation Pre-training.** Carlos Eduardo Fontaneli and Cesar H. Comin, Federal University of
> São Carlos (UFSCar). Submitted to *Computerized Medical Imaging and Graphics*, 2026.

We compare a plain ViT (MedSAM), a hybrid ViT (LiteMedSAM) and three CNNs (LW-Net, ConvUNeXt, ConvNeXt) on the thin
vessels of three retinal datasets (DRIVE, HRF, FIVES) and a microscopy dataset (VessMap). The transformer-based models
are fine-tuned from their published weights, trained from scratch, and searched as compact models; every
configuration has its own Optuna search of 100 trials.

<p align="center"><img src="assets/thin_errors.png" width="720" alt="Thin centerline pixels found by all, some or none of the five models"></p>

This repository contains exactly what the paper's experiments ran: data handling, the five architectures, the
hyperparameter search, training, inference, metrics, cost, statistics, every figure and table, and the paper's
source. From a clone, without data or a GPU, every number, figure, table and p-value of the paper can be regenerated
bit for bit, and the paper compiled (see [Verification](#verification)).

## Main results

Dice / thin-vessel centerline recall (%, mean over the test images; CNNs trained from scratch, transformer-based
models fine-tuned; MedSAM on DRIVE and VessMap evaluated at its training scale). Full results, standard deviations
and significance tests: Tables II-X of the paper.

| Model | DRIVE | HRF | FIVES | VessMap |
|---|---|---|---|---|
| LW-Net | 81.1 / 57.5 | 76.9 / 24.7 | 81.8 / 25.9 | 92.8 / 80.1 |
| ConvUNeXt | 77.6 / 52.2 | 79.7 / 23.3 | 89.3 / 33.8 | 92.9 / 78.7 |
| ConvNeXt | 77.5 / 44.2 | 76.9 / 25.4 | 90.2 / 36.5 | 91.0 / 69.9 |
| LiteMedSAM | **82.3** / **64.3** | 80.6 / **40.1** | 88.1 / **37.3** | **93.0** / **83.8** |
| MedSAM | 81.3 / 56.9 | **81.7** / 26.9 | **92.2** / 36.2 | 92.8 / 82.6 |

Bold: highest mean of a column (significance in the paper). Thin vessels: centerline radius below 1.5 px.

## Quickstart

```bash
git clone https://github.com/CarlosFontaneli/thin-vessel-sam-cnn.git && cd thin-vessel-sam-cnn
python -m venv .venv && source .venv/bin/activate
make install          # pip install -e ".[dev]" (Python >= 3.11; the paper's environment used 3.12)
make test             # unit tests, about ten seconds on CPU
make figures          # statistics, every figure and table from results/, and a byte-by-byte check against the paper
make paper            # compiles paper/manuscript into outputs/paper/main.pdf (needs a TeX Live installation)
```

To retrain, predict and score, see [`docs/reproduction.md`](docs/reproduction.md); the datasets and the published
MedSAM/LiteMedSAM weights are described in [`docs/data.md`](docs/data.md).

## Repository layout

| Path | Content |
|---|---|
| `src/thinvessel/` | the package: configuration, data, models (`third_party/` = vendored SAM and TinyViT), training and search, evaluation, analysis |
| `configs/` | the 36 configurations (selected trial of each search and its search space) and the DRIVE splits |
| `scripts/` | command-line entry points: `check_data`, `search`, `train`, `predict`, `evaluate`, `analyse` |
| `results/` | the committed tables behind every number of the paper ([`results/README.md`](results/README.md)) |
| `paper/manuscript/` | LaTeX source of the paper; `figures/` holds the generated figures and tables it includes |
| `paper/figures_tables/`, `paper/statistics/` | the scripts that produce every figure, table and p-value ([`paper/README.md`](paper/README.md)) |
| `docs/` | data, reproduction, selected hyperparameters, known deviations |
| `tests/` | unit tests |
| `data/`, `weights/`, `outputs/` | not committed: datasets, published weights, everything the code writes |

## Known deviations

The code keeps the quirks of the experiments, because removing them would change the results. The most important:
the searches sampled `optimizer` and `patience` but never applied them (every run used AdamW and the configured
patience); on HRF, FIVES and VessMap the test images were also used for validation, as the paper states; MedSAM on
DRIVE and VessMap was validated with the padded protocol and tested with the upscaled one. All of them, with the
evidence, are in [`docs/known_deviations.md`](docs/known_deviations.md); the selected hyperparameters of the 36
searches are in [`docs/selected_hyperparameters.md`](docs/selected_hyperparameters.md).

## Verification

Before release, the code was checked against the records of the paper's runs: the per-image metrics recomputed from
the saved test predictions equal the paper's for every metric, image and configuration; the 36 configurations equal
the selected trials' recorded configurations, and the search mapping reproduces 3,611 of the 3,612 recorded trial
configurations (the other is a documented failed trial); the parameter and FLOP counts equal those measured on the
trained checkpoints; the statistical tests and the plotted values of every figure and table are byte-identical to
the paper's; and the paper compiled from `paper/manuscript/` is pixel-identical to the build of the authors' manuscript, except the code-availability URL.

## Licence

The code is under the MIT licence ([`LICENSE`](LICENSE)); vendored and adapted code keeps its own licence
([`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)). The paper's text and figures are not covered by the MIT licence;
see [`paper/README.md`](paper/README.md).

## Citation

```bibtex
@article{fontaneli2026thinvessel,
  title   = {Evaluating {SAM}-based and Convolutional Architectures in Thin Retinal Vessel Segmentation: Model
             Capacity, Computational Cost and Foundation Pre-training},
  author  = {Fontaneli, Carlos Eduardo and Comin, Cesar H.},
  journal = {Computerized Medical Imaging and Graphics},
  year    = {2026},
  note    = {Submitted; volume, pages and DOI pending}
}
```

See also [`CITATION.cff`](CITATION.cff). Contact: Cesar H. Comin (corresponding author), <comin@ufscar.br>.
