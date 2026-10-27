"""Datasets, augmentation, splits and the FIVES correction on synthetic images (Section 3.1, Appendix A.1)."""

import numpy as np
import torch
from PIL import Image

from thinvessel.data import FIVESDataset, PairedVesselDataset, build_test_dataset, random_split_indices


def make_pairs(folder, n: int, size: int, mask_suffix: str = "") -> None:
    (folder / "images").mkdir(parents=True)
    (folder / "masks").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(folder / "images" / f"{i:02d}.png")
        m = np.zeros((size, size), np.uint8)
        m[size // 2, 4:-4] = 255  # off the border, which FIVES checks
        Image.fromarray(m).save(folder / "masks" / f"{i:02d}{mask_suffix}.png")


def test_training_crop_and_upscale(tmp_path) -> None:
    make_pairs(tmp_path / "big", 2, 300)
    make_pairs(tmp_path / "small", 2, 100)
    big = PairedVesselDataset(
        str(tmp_path / "big/images"), str(tmp_path / "big/masks"), 256, mode="train", apply_transform=True
    )
    small = PairedVesselDataset(
        str(tmp_path / "small/images"), str(tmp_path / "small/masks"), 256, mode="train", apply_transform=True
    )
    torch.manual_seed(0)
    for ds in (big, small):
        img, mask = ds[0]
        assert img.shape == (3, 256, 256) and mask.shape == (1, 256, 256)
        assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}  # nearest-exact keeps the mask binary
    img, mask = PairedVesselDataset(str(tmp_path / "small/images"), str(tmp_path / "small/masks"), 256, mode="val")[0]
    assert img.shape == (3, 100, 100)  # validation and test use the full image


def test_drive_mask_names(tmp_path) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "images" / "21_training.png")
    Image.new("L", (8, 8)).save(tmp_path / "masks" / "21_manual1.png")
    assert len(PairedVesselDataset(str(tmp_path / "images"), str(tmp_path / "masks"), 8)) == 1


def test_fives_border_correction(tmp_path) -> None:
    make_pairs(tmp_path, 2, 64)
    bad = np.array(Image.open(tmp_path / "masks" / "01.png"))
    bad[:, 0] = 255  # a vessel-labelled border column: border sum far above the threshold
    Image.fromarray(bad).save(tmp_path / "masks" / "01.png")
    ds = FIVESDataset(str(tmp_path / "images"), str(tmp_path / "masks"), 64, mode="val")
    assert ds.corrupted_bases == {"01"}
    _, mask = ds[1]
    assert mask[0, :, :2].sum() == 0 and mask[0, :2, :].sum() == 0 and mask[0, 32, 10] == 1
    _, mask = ds[0]
    assert mask[0, 32, 4] == 1 and mask[0].sum() == 56  # an uncorrupted mask is unchanged


def test_random_split_sizes() -> None:
    train, held = random_split_indices(45, 0.7, 42)
    assert (len(train), len(held)) == (31, 14) and not set(train) & set(held)
    train, held = random_split_indices(100, 0.7, 42)
    assert (len(train), len(held)) == (70, 30)
    assert random_split_indices(100, 0.7, 42) == [train, held]


def test_test_dataset_of_random_split(tmp_path) -> None:
    make_pairs(tmp_path, 10, 16)
    params = {
        "image_dir": str(tmp_path / "images"),
        "mask_dir": str(tmp_path / "masks"),
        "image_size": 16,
        "train_size": 0.7,
    }
    ds = build_test_dataset("vessmap", params)
    assert len(ds) == 3 and list(ds.indices) == random_split_indices(10, 0.7, 42)[1]
