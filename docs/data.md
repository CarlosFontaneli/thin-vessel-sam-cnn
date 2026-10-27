# Data and weights

## Datasets

The datasets are not redistributed. Download them from their sources and arrange them under `$DATA_ROOT`
(default `./data`; scripts also take `--data-root`).

| Dataset | Source | Images | Split used in the paper |
|---|---|---|---|
| DRIVE | <https://drive.grand-challenge.org> (Staal et al., 2004) | 40, 565×584 px | official 20 training / 20 test; training images 37-40 for validation (`configs/splits/drive_*.csv`) |
| HRF | <https://www5.cs.fau.de/research/data/fundus-images/> (Budai et al., 2013) | 45, 3504×2336 px | 31 training / 14 test, random with seed 42 |
| FIVES | <https://doi.org/10.6084/m9.figshare.19688169> (Jin et al., 2022) | 800, 2048×2048 px | official 600 training / 200 test |
| VessMap | doi:10.1371/journal.pone.0322048 (Viana da Silva et al., 2025; data at the link of that article) | 100, 256×256 px | 70 training / 30 test, random with seed 42 |

```
$DATA_ROOT/
├── DRIVE/training/images/21_training.tif ...     DRIVE/training/1st_manual/21_manual1.gif ...
│   DRIVE/test/images/01_test.tif ...             DRIVE/test/1st_manual/01_manual1.gif ...
├── HRF/images/01_dr.JPG ...                      HRF/manual1/01_dr.tif ...
├── FIVES/train/Original/1_A.png ...              FIVES/train/Ground truth/1_A.png ...
│   FIVES/test/Original/1_A.png ...               FIVES/test/Ground truth/1_A.png ...
└── VessMAP/images/<id>.tiff ...                  VessMAP/labels/<id>.png ...   (annotator 1)
```

- **Validation.** DRIVE validates on training images 37-40. HRF, FIVES and VessMap hold out no validation set: as
  the paper states, the test images were used for checkpoint selection, early stopping and the search.
- **Random splits** (HRF, VessMap) shuffle the images, sorted by file name, with `random.Random(42)`; the first 70%
  are the training images (`src/thinvessel/data/splits.py`).
- **FIVES masks.** 106 training and 5 test masks label their border as vessel. When the sum of a mask's outer 1-pixel
  border exceeds 500, its 2 px border is set to background as the mask is read. Nothing has to be prepared by hand.

Check a local copy against the paper's split sizes:

```bash
python scripts/check_data.py      # DRIVE 16/4/20, HRF 31/14/14, FIVES 600/200/200 (106/5 corrected), VessMap 70/30/30
```

## Published weights

The fine-tuned runs start from the published checkpoints of Ma et al. (for research use only, per their authors).
Place them in `$WEIGHTS_DIR` (default `./weights`; scripts also take `--weights-dir`):

| File | Source | Bytes | SHA-256 |
|---|---|---:|---|
| `medsam_vit_b.pth` | MedSAM, <https://github.com/bowang-lab/MedSAM> ("model checkpoint" link of the README, Google Drive folder `1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN`) | 375,049,145 | `34b34b78c1d18cb8c6bf84cf9c00e135d6d6c965699f3c0e31ef1bc9dcb5be74` |
| `lite_medsam.pth` | LiteMedSAM, <https://github.com/bowang-lab/MedSAM/tree/LiteMedSAM> (Google Drive file `18Zed-TUTsmr2zc5CHUWd5Tu13nb6vq6z`) | 39,370,479 | `79d8c9dca6db4d69d3f905579e5250af05e859fff9c1f543e89a513c3028ce76` |

No weights are needed to train from scratch, to run inference with a trained checkpoint or to measure the cost. The
trained checkpoints of the paper are not released.
