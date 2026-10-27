# Results tables

The inputs of every number, figure, table and p-value of the paper, from the paper's own runs. `make figures` reads
them; the scripts that write them are named below. Values are fractions (not %) unless stated.

Configuration ids: `lwnet_scratch`, `convunext_scratch`, `convnext_scratch` (CNNs, from scratch);
`litemedsam_pretrained`, `medsam_pretrained` (fine-tuned); `litemedsam_scratch`, `medsam_scratch` (from scratch);
`litemedsam_compact_scratch`, `medsam_compact_scratch` (compact, from scratch). Datasets: `drive`, `hrf`, `fives`,
`vessmap`.

## `per_image_metrics.csv` — `scripts/evaluate.py`

One row per dataset, configuration, protocol and test image (2,476 rows).

| Column | Meaning |
|---|---|
| `protocol` | `native` (sliding window at native resolution; padded for MedSAM on DRIVE and VessMap) or `upscaled` (MedSAM fine-tuned and from scratch on DRIVE and VessMap; the protocol of the paper's results) |
| `sample_id` | `sample_<i>`, the index of the image in the test set (same images across configurations) |
| `height`, `width` | native image size (px) |
| `dice`, `iou`, `pixel_precision`, `pixel_recall` | pixel overlap of the binarised prediction (threshold 0.5) over the full image |
| `thin_vessel_recall`, `n_gt_thin_centerline_pixels` | share of ground-truth centerline pixels with radius < 1.5 px inside the prediction, and its denominator |
| `thin_vessel_precision`, `n_pred_thin_centerline_pixels` | share of predicted centerline pixels with radius < 1.5 px (in the prediction) inside the ground truth, and its denominator; empty when undefined (one image) |
| `centerline_recall`, `centerline_precision` | the same over all centerline pixels |
| `centerline_recall_radius_<a>_<b>`, `n_gt_centerline_pixels_radius_<a>_<b>` | recall of the ground-truth centerline with radius in [a, b) px (`inf`: no upper edge), and its denominator |

## `cost.csv` — `scripts/analyse.py --steps cost`

One row per configuration and dataset: `params_total`, `params_trainable`, `params_m` (millions), `flops` and `gflops`
(fvcore, one forward pass over one window; one multiply-add = one FLOP), `roi_size` (window, px),
`windows_per_image` (sliding windows over a test image) and `gflops_per_image`.

## `convergence.csv` — `scripts/analyse.py --steps convergence`

One row per search: `selected_trial` (highest best validation Dice), `selected_best_val_dice`, `selected_best_epoch`,
`selected_epochs_to_target` (first epoch at 95% of the best validation Dice), `selected_val_interval` (epochs),
`selected_target_censored` (target already reached at the first validation), `trials_with_curves`.

## `search_trials.csv` — `scripts/analyse.py --steps search`

One row per trial of the 36 searches (3,612 rows): `number`, `state` (Optuna state; `LOGGED` for the two searches
without a study database), `hours` (duration), `best_val_dice` (maximum over the trial's validations), one column per
sampled parameter (empty where a search did not sample it), the settings the trial trained with (`applied_optimizer`,
`applied_max_epochs`, `applied_check_val_every_n_epoch`, `applied_patience`) and `out_of_memory`.
