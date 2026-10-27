# Known deviations, quirks and protocol notes

The code in this repository computes what the experiments of the paper computed. Where the experiments had a
bug or an unusual choice, it is **kept** and described here, because changing it would make the code disagree
with the paper. Each item says where it lives in the code.

## Hyperparameter search

1. **`optimizer` and `patience` are sampled but never applied.** Every search space starts with
   `optimizer` (AdamW/SGD/RMSprop) and most contain `patience`. The mapping from sampled values to the training
   configuration (`thinvessel.config.apply_params`) has no rule for them, so every trial trained with AdamW and
   the patience of its configuration. The suggestions are kept because they consume random numbers of the TPE
   sampler: removing them would change the sequence of sampled trials. The private repository wrote a
   `best_run_summary.txt` with the *sampled* values (e.g. "SGD"), which is misleading; this release writes
   `best_trial.json` (trial number and validation Dice) instead. The configuration actually used by a trial is
   its `trial_<n>/config.yaml`.
2. **The Optuna objective is the last validation Dice of a trial, not the best.** The checkpoint kept for a
   trial is the one with the best `val/dice`, and the trial selected for the paper is the one with the highest
   best `val/dice` (`thinvessel.analysis.convergence`), but `study.best_trial` in the Optuna database refers to the
   last values. The median pruner compares the validation Dice reported at each validation epoch.
3. **`val/dice` is pooled over the validation images** (intersection and union summed over all images, then
   one Dice), not the mean of per-image Dice that the paper reports on the test images
   (`thinvessel.training.module.BinaryDiceMetric`).
4. **Changes during a search** (recorded in the header of each affected configuration):
   - FIVES ConvNeXt: trials 0-1 validated every epoch with patience 50, trials 2-101 every 10 epochs with
     patience 5. The selected trial is 0.
   - FIVES fine-tuned LiteMedSAM: trials 0-14 validated every epoch (patience 50), trials 15-101 every 10 epochs
     (patience 5); from trial 16 on, `patience` was no longer in the search space. The selected trial is 2, and its
     configuration holds the first search space.
   - VessMap compact LiteMedSAM: trials 86-101 used `persistent_workers: true`; the selected trial 71 did not.
   - DRIVE compact MedSAM: trial 0 failed after 8 s. The code then did not set the prompt encoder's embedding grid
     to `img_size // patch_size` (its record has patch size 4 with a 16x16 grid). The fix was made and the search
     restarted; trial 1 drew exactly the same values (see item 5) and is the selected trial.
5. **Interrupted searches restart the sampler.** A search that was stopped and resumed (studies with `RUNNING`
   trials in the database: on FIVES ConvNeXt, ConvUNeXt, fine-tuned LiteMedSAM and MedSAM and from-scratch
   MedSAM; VessMap compact LiteMedSAM) creates a new `TPESampler(seed=42)`, whose random sequence starts again. Running a
   search from scratch without interruptions therefore does not reproduce the same sequence of trials. To
   reproduce a reported run, train its selected configuration (`scripts/train.py`), not the search.
6. **LW-Net widths are stored as strings.** The CNN searches recorded `layers: '[8, 16, 32]'` (a string; LW-Net
   parses it) in the trial configurations; `apply_params` keeps that, so a new search writes the same records.
7. **Searches without a database.** VessMap LW-Net and ConvUNeXt have an empty `optuna_study.db`; their trials are
   read from `train.log` (sampled values and start times; no validation Dice), and their search spaces from the
   configuration files of the experiments, which agree with every logged value.
8. **Totals.** The 36 searches ran 3,612 trials: 3,602 finished (599 complete, 2,801 pruned, 2 failed and the 200
   trials of the two searches without a database) and 10 interrupted; 2,093 GPU hours (sum of the trial durations).

## Configuration files of the experiments

The released configurations were rebuilt from the trial records (`trial_<n>/config.yaml` of the selected trial
and the search space of the Optuna study), not from the experiments' `configs/*.yaml`, which differ from what ran:

- HRF LW-Net, ConvUNeXt and ConvNeXt: `image_size: [2336, 3504]` instead of 256, `num_workers: 2` instead of 16, no
  `check_val_every_n_epoch` (the runs validated every 10 epochs) and `patience: 50` instead of 5; their search
  spaces list `layers`/`base_c` (not searched on HRF) and batch sizes up to 32 or 128 (the searches used 2-16).
- FIVES fine-tuned LiteMedSAM: `patience: 5` and `check_val_every_n_epoch: 10` (the selected trial used 50 and every
  epoch), `num_workers: 8` (2 was used), no `patience` in the search space (it was sampled in trials 0-15).
- DRIVE compact MedSAM: patch sizes `[1, 2, 4, 8, 16]`; the search used `[1, 4, 8, 16]` (2 px ran out of memory in a
  test run before the search).
- VessMap compact LiteMedSAM: `persistent_workers: true` (the selected trial did not use it).

Differences between the released configurations and the trial records, all without effect on training:

- paths are relative to `DATA_ROOT`, the split lists and `WEIGHTS_DIR` instead of the experiments' folders;
- removed keys that no code read: `augmentations.affine_prob` (no affine augmentation was ever applied),
  `skeleton_dir` (VessMap) and `train_size` (DRIVE and FIVES, which have explicit splits);
- added `allow_test_as_val: true` (FIVES): the experiments' code used the test folder for validation silently
  when no validation folder was given; the code now requires this flag;
- added `selected_trial` and `search_space`.

Replaying the sampled values of all 3,612 recorded trials through `apply_params` on the released configurations
reproduces every recorded `trial_<n>/config.yaml` except the failed trial 0 of DRIVE compact MedSAM (item 4 above).

## Data and splits

9. **Validation on the test images (HRF, FIVES, VessMap).** No separate validation set was held out: checkpoint
   selection, early stopping and the search used the test images, as the paper states. DRIVE validates on
   training images 37-40 (`configs/splits/drive_val.csv`).
10. **FIVES mask correction.** A mask is corrupted when the sum of its outer one-pixel border (grey levels 0-255)
    exceeds 500; its 2 px border is then set to 0 when the mask is read (`thinvessel.data.FIVESDataset`). This
    flags 106 of the 600 training masks and 5 of the 200 test masks, and it also applies to the test ground truth
    that the metrics use. The experiments' repository has a cache file for the list (`fives_corrupted_masks.json`)
    which was empty, so every run detected the masks this way; the release keeps only this mechanism.
11. **HRF and VessMap splits** shuffle the sorted file names with `random.Random(42)`. The experiments' training
    code took the seed from `dataset_params.seed` and its prediction script from the top-level `seed`; both were
    42. The release uses `dataset_params.seed` (default 42) for both.
12. **Augmentation.** The image is rotated with nearest-neighbour interpolation (torchvision's default), like the
    mask. Images smaller than the input size are resized (bilinear; mask nearest-exact) instead of cropped.

## Models

13. **Box prompt of fine-tuned LiteMedSAM.** The fixed box embedding is computed when the model is built, before
    the published weights are loaded, and `lite_medsam.pth` does not contain it. The fine-tuned LiteMedSAM
    therefore used a box embedding from the randomly initialised (frozen) prompt encoder, which its fine-tuned
    decoder learned to read. Fine-tuned MedSAM is built from its checkpoint first, so its box embedding is the
    published one.
14. **LayerNorm epsilon of fine-tuned MedSAM at inference.** Training built fine-tuned MedSAM through SAM's model
    registry (LayerNorm `eps=1e-6`). The experiments' prediction script later rebuilt every checkpoint from its
    stored `medsam_params` without loading the published weights, which gives PyTorch's default `eps=1e-5`, and
    this release does the same (`thinvessel.evaluation.inference.load_trained_module`). The upscaled-protocol
    predictions of MedSAM on DRIVE and VessMap (every MedSAM result on these datasets in the paper) were computed
    this way, and the release reproduces them pixel for pixel. The earlier predictions (padded protocol of Table
    V; HRF and FIVES) were computed by the previous version of the script, which used the registry (`1e-6`). On
    DRIVE the two builds differ in 15 pixels over the 20 test images, less than the difference between GPUs.
15. **ConvNeXt parameters** include the unused classification head of torchvision's `convnext_large`
    (211,401,257 parameters in total, as reported).
16. **MedSAM parameters** include the 6,220 parameters of the frozen prompt encoder (93,735,472 in total).

## Determinism

17. Every trial calls `pytorch_lightning.seed_everything(42)`; each training sample draws its own seed for the
    flips, rotation and crop; data-loader workers are started with `spawn`. cuDNN and GPU kernels are not
    deterministic and the runs used several GPU types (RTX 3090 for the CNNs, RTX PRO 5000 and RTX PRO 4000 for
    the transformer-based models), so a retrained model differs slightly from the paper's. One run per
    configuration was trained. Inference with the original checkpoints on another GPU (RTX 3060) differed from the
    saved masks of the paper by at most 22 pixels per image for LW-Net, ConvUNeXt and LiteMedSAM, and by 0 pixels
    for the predictions made on that GPU; the release and the experiments' code give identical masks on the same GPU.
18. **Float parsing of the tables.** The paper's figure and table scripts read the per-image metrics, the tests
    and the convergence summary from CSV with pandas' default float parser, and the per-image values of Table IV,
    the cost and the search records from parquet, JSON and SQLite. The released scripts parse each table the same
    way (`paper/figures_tables/common.read_table`), so that every printed and plotted value is bit-identical to the
    paper; parsing the CSVs exactly changes some means in the last digit.

## Metrics

19. An empty prediction scores 0 for the centerline recall and precision and for the recall of every radius bin
    with ground-truth pixels, but NaN (undefined, left out of the means) for the thin-vessel precision; one image
    has an undefined thin-vessel precision (VessMap, MedSAM from scratch, padded protocol).
20. The first radius bin also takes radii below 0.5 px and the last bin radii above its upper edge (100 px), so the
    bins cover every centerline pixel.
