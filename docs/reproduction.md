# Reproduction

## Environment

Python ≥ 3.11 (the pinned numpy, SciPy, scikit-image and pandas need it). The experiments ran with Python 3.12.13,
PyTorch 2.10.0 (CUDA 12.8 wheels), PyTorch Lightning 2.6.1, MONAI 1.5.2, Optuna 4.7.0, scikit-image 0.26.0, SciPy
1.17.1 and fvcore 0.1.5 (`0.1.5.post20221221`); `pyproject.toml` pins every direct dependency to the version of
that environment.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # = make install; or: conda env create -f environment.yml && conda activate thinvessel
```

For a GPU build other than the default PyTorch wheel, install `torch==2.10.0` and `torchvision==0.25.0` from the
PyTorch index for your CUDA version first.

| Variable | Option | Default | Holds |
|---|---|---|---|
| `DATA_ROOT` | `--data-root` | `./data` | the four datasets ([`data.md`](data.md)) |
| `WEIGHTS_DIR` | `--weights-dir` | `./weights` | `medsam_vit_b.pth`, `lite_medsam.pth` |
| `OUTPUTS_DIR` | `--outputs-dir` | `./outputs` | searches, trained models, predictions, per-image metrics, analysis tables |

## Stages

A configuration is `configs/<regime>/<dataset>/<model>.yaml` with regime `cnn`, `finetuned`, `scratch` or `compact`
([`../configs/README.md`](../configs/README.md)). Outputs use the same `<regime>/<dataset>/<model>` layout under
`$OUTPUTS_DIR`.

| Stage | Command (one configuration) | Output | Cost |
|---|---|---|---|
| Search (100 trials) | `python scripts/search.py -c configs/cnn/drive/lwnet.yaml` | `search/<study>/` (study db, `trial_<n>/`, best checkpoint) | 2,093 GPU h for the 36 searches (1–345 GPU h each) |
| Train the selected run only | `python scripts/train.py -c configs/finetuned/drive/medsam.yaml` | `train/<study>/best_model_val_dice_<d>.ckpt` | 243 GPU h for the 36 runs (0.05–60 GPU h each) |
| Predict the test set | `python scripts/predict.py -c configs/finetuned/drive/medsam.yaml --protocol upscaled` | `predictions/<study>/<protocol>/` | minutes per run |
| Score all predictions | `python scripts/evaluate.py` | `metrics/per_image_metrics.csv` (the 36 runs) | about 10 min on 14 CPU cores |
| Cost, convergence, search records | `python scripts/analyse.py` | `analysis/{cost,convergence,search_trials}.csv` | a few CPU minutes |

The GPU hours are those of the paper's runs on RTX 3090 (CNNs) and RTX PRO 5000/4000 (transformer-based models)
GPUs. MedSAM trains at 1024 px and needs a GPU with about 24 GB or more.

All 36 runs, from the trained models to the result tables:

```bash
for cfg in configs/{cnn,finetuned,scratch,compact}/*/*.yaml; do
  python scripts/train.py -c "$cfg"                        # or scripts/search.py to repeat the search
  python scripts/predict.py -c "$cfg" --protocol native
done
for cfg in configs/{finetuned,scratch}/{drive,vessmap}/medsam.yaml; do
  python scripts/predict.py -c "$cfg" --protocol upscaled   # MedSAM on DRIVE and VessMap: the protocol of the paper
done
python scripts/evaluate.py
python scripts/analyse.py               # convergence and search records need the search outputs
```

Retrained models will not match the paper exactly (GPU nondeterminism, one seed per configuration), so their tables
will differ slightly from those in `results/`, which hold the tables of the paper's own runs.

## Protocols and metrics

- **Training** (Section 3.3, Appendix A): random crops of the model's input size (256 px; 1024 px for MedSAM; images
  smaller than the input are upscaled), flips and rotations; soft Dice loss; AdamW with cosine annealing over 1000
  epochs; early stopping after 50 epochs without improvement of the validation Dice.
- **Inference** (Section 3.4): MONAI sliding windows of the model's input size (25% overlap, Gaussian weighting,
  threshold 0.5) at native resolution (`--protocol native`; smaller images are padded). Every MedSAM result on DRIVE
  and VessMap uses `--protocol upscaled` (image upscaled to 1024 px, one window, logits resized back), MedSAM's
  training scale there; the padded protocol appears only in Table IV.
- **Metrics** (`thinvessel.evaluation.metrics`): Dice over the full image; centerline recall and precision from the
  scikit-image skeleton (Zhang-Suen), no tolerance; vessel radius from SciPy's Euclidean distance transform; thin
  vessels have radius < 1.5 px; recall in the bins [0.5, 1.5), [1.5, 2.5), [2.5, 3.5), [3.5, 4.5), [4.5, ∞) px.

## Determinism

Every trial seeds all generators with 42 (`pytorch_lightning.seed_everything`); each training sample draws its own
seed for its augmentation; data-loader workers are spawned. cuDNN and GPU arithmetic are not deterministic and the
paper's runs used several GPU types, so a retrained model differs from the paper's. Each configuration was trained
once. With the original checkpoints, this code and the experiments' code give identical predictions on the same GPU;
on another GPU they differed by at most 22 pixels per test image.
