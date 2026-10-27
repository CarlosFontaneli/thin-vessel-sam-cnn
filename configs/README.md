# Configurations

One file per configuration and dataset, 36 in all: `configs/<regime>/<dataset>/<model>.yaml`.

| Regime | Models | Paper |
|---|---|---|
| `cnn` | `lwnet`, `convunext`, `convnext` | CNNs trained from random initialization |
| `finetuned` | `litemedsam`, `medsam` | fine-tuned from the published weights |
| `scratch` | `litemedsam`, `medsam` | full-size models trained from random initialization |
| `compact` | `litemedsam`, `medsam` | compact models (architecture searched), from random initialization |

Datasets: `drive`, `hrf`, `fives`, `vessmap`. Splits: `splits/drive_{train,val,test}.csv` (DRIVE image base names);
the other datasets are split as described in `src/thinvessel/data/splits.py`.

Each file holds **the settings of the selected trial of its hyperparameter search**, as recorded in that trial's
`trial_<n>/config.yaml`, and **the search space** as recorded in the Optuna study. Training a file as it is
(`scripts/train.py`) repeats the run reported in the paper; searching with it (`scripts/search.py`) repeats the
search. The differences from the records (paths, three keys that no code read, one added flag) are listed in
[`../docs/known_deviations.md`](../docs/known_deviations.md); the selected values are tabulated in
[`../docs/selected_hyperparameters.md`](../docs/selected_hyperparameters.md).

## Schema

```yaml
seed: 42                    # pytorch_lightning.seed_everything at the start of every trial; also the TPE seed
model_name: medsam          # lwnet | convunext | convnext | litemedsam | medsam
dataset_name: drive         # drive | hrf | fives | vessmap
selected_trial: 85          # number of the selected trial of the search (metadata; not used in training)
model_params:               # keyword arguments of the architecture (thinvessel.models.build_model)
  pretrained_weights_path: medsam_vit_b.pth   # fine-tuned models only; relative to WEIGHTS_DIR
  medsam_params: {...}      # MedSAM: vit_b_params, prompt_encoder_params, mask_decoder_params, transformer_params
dataset_params:
  train_image_dir: DRIVE/training/images     # *_dir: relative to DATA_ROOT
  train_csv: ../../splits/drive_train.csv    # *_csv: relative to this file
  # explicit splits: train/val/test _image_dir, _mask_dir (and _csv); FIVES: allow_test_as_val: true
  # random splits (HRF, VessMap): image_dir, mask_dir, train_size (0.7), seed (default 42)
  image_size: 1024          # training input (px): crop size, or the size small images are resized to
  augment: true
  augmentations: {hflip_prob: 0.5, vflip_prob: 0.5, rotation_degrees: 30.0}
  batch_size: 2
  num_workers: 4            # data-loader workers (spawned); affects the random augmentation stream
  persistent_workers: false # (optional, default false)
training_params:
  optimizer: AdamW          # the only optimizer used
  lr: 6.4e-04
  weight_decay: 7.2e-06
  max_epochs: 1000          # also T_max of the cosine schedule
  patience: 50              # early stopping, in validations
  check_val_every_n_epoch: 1 # (optional, default 1)
  roi_size: 1024            # sliding-window size of validation and inference (px)
  deep_sup_weight: 1.0      # LW-Net only: weight of the coarse output's loss
search_space:               # order = order of the Optuna suggest calls (it matters for the TPE sampler)
  optimizer: {type: categorical, choices: [torch.optim.AdamW, torch.optim.SGD]}  # sampled, never applied
  lr: {type: float, low: 1.0e-07, high: 1.0e-03, log: true}
  weight_decay: {type: float, low: 1.0e-06, high: 0.1, log: true}
  patience: {type: int, low: 20, high: 100}                                       # sampled, never applied
  batch_size: {type: categorical, choices: [2]}
```

`type` is `float` (`low`, `high`, `log`), `int` (`low`, `high`) or `categorical` (`choices`; list values are stored
as strings such as `"[8, 16, 32]"`). Where a sampled value goes is defined by `thinvessel.config.apply_params`:
`lr` and `weight_decay` to `training_params`, `batch_size` to `dataset_params`, the compact-model keys (`patch_size`,
`embed_dim`, `depth`, `num_heads`, `mlp_ratio`, `window_size`, `embed_dims`, `depths`, `window_sizes`) into the
transformer's parameters, and `layers`, `base_c`, `drop_path_rate` to `model_params`.
