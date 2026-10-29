# How well do SAM-based models segment thin retinal vessels?

Thin vessels, one or two pixels wide, are the easiest part of a retinal vessel map to miss. It is often assumed that
vision transformers, which look at an image in patches of many pixels, miss them more than convolutional networks do.
Our paper tests how far that holds: it compares MedSAM and LiteMedSAM, two Segment Anything models for medical
images, with three convolutional networks on four datasets (DRIVE, HRF, FIVES and VessMap), looking at thin vessels
in particular.

This repository has the code to reproduce the paper's experiments, from training the models to every figure and
table.

> **Evaluating SAM-based and Convolutional Architectures in Thin Retinal Vessel Segmentation: Model Capacity,
> Computational Cost and Foundation Pre-training.** Carlos E. Fontaneli and Cesar H. Comin, Federal University of
> São Carlos (UFSCar).

<p align="center"><img src="assets/thin_errors.png" width="760" alt="Thin centerline pixels found by all, some or none of the five models"></p>
<p align="center"><em>Thin vessels found by the five models compared in the paper: (a–d) the window of each test set with the most thin centerline pixels, (e) the share of thin centerline pixels found by all five models, by some of them, or by none, over all test images.</em></p>

## Main findings

For pre-trained plain ViTs, our results do not support the common expectation that they miss thin vessels, although
similar thin-vessel performance can be obtained with much smaller models.

Fine-tuned from its published weights, MedSAM, a plain ViT that splits the image into 16-pixel patches, found thin
vessels as well as the best CNN on all four datasets: its thin-vessel recall was within −0.6 to +2.5 points of the
best CNN's, a difference that was not significant, although its thin-vessel precision was lower on DRIVE and VessMap.
On HRF and FIVES it also had a significantly higher Dice than the best CNN. The hybrid LiteMedSAM had the highest
mean thin-vessel recall on every dataset.

<p align="center"><img src="assets/recall_by_caliber.png" width="760" alt="Centerline recall by vessel radius for the five models on the four datasets"></p>
<p align="center"><em>Centerline recall by vessel radius. The first bin, radius below 1.5 px, is the thin vessels.</em></p>

Much smaller models can do as well. A compact plain ViT trained from scratch lost MedSAM's performance, but LW-Net, a
CNN with 0.07 M parameters against MedSAM's 93.7 M, matched fine-tuned MedSAM in Dice and thin-vessel recall on DRIVE
and VessMap, at a small fraction of its cost per image.

Foundation pre-training made the difference: it raised MedSAM's Dice significantly on every dataset, and without it
MedSAM fell significantly below the best CNN in thin-vessel recall on three datasets.

<p align="center"><img src="assets/pretraining.png" width="760" alt="Dice, thin-vessel recall and precision of MedSAM and LiteMedSAM fine-tuned and from scratch, next to the CNNs"></p>
<p align="center"><em>MedSAM and LiteMedSAM fine-tuned (filled) and trained from scratch (hollow), next to the three CNNs.</em></p>

## Quickstart

```bash
git clone https://github.com/CarlosFontaneli/thin-vessel-sam-cnn.git && cd thin-vessel-sam-cnn
python -m venv .venv && source .venv/bin/activate
make install          # Python 3.11 or newer
make test             # unit tests, about ten seconds on CPU
make figures          # every figure and table of the paper, from the tables in results/
make paper            # compiles the paper into outputs/paper/main.pdf (needs TeX Live)
```

The figures and tables need neither the datasets nor a GPU. To train the models and evaluate them yourself, see
[`docs/data.md`](docs/data.md) for the datasets and weights and [`docs/reproduction.md`](docs/reproduction.md) for
the commands.

## What is here

- `src/thinvessel/`: the Python package (data, models, training, evaluation)
- `configs/`: one configuration per model and dataset, with the hyperparameters used in the paper
- `scripts/`: the command-line tools, from training to scoring
- `results/`: the tables behind the paper's numbers
- `paper/`: the LaTeX source of the paper and the scripts of its figures and tables
- `docs/`: data, reproduction and hyperparameter details

## Licence

The code is under the MIT licence ([`LICENSE`](LICENSE)); vendored code keeps its own licence
([`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)). The text and figures of the paper are not covered by the MIT
licence (see [`paper/README.md`](paper/README.md)).

## Citation

```bibtex
@unpublished{fontaneli2026thinvessel,
  title  = {Evaluating {SAM}-based and Convolutional Architectures in Thin Retinal Vessel Segmentation: Model
            Capacity, Computational Cost and Foundation Pre-training},
  author = {Fontaneli, Carlos E. and Comin, Cesar H.},
  year   = {2026},
  note   = {Manuscript under review}
}
```

See also [`CITATION.cff`](CITATION.cff). Contacts: Carlos E. Fontaneli, <carlos.fontaneli@estudante.ufscar.br>; Cesar H. Comin, <comin@ufscar.br>.
