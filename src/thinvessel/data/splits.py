"""Training, validation and test splits of the four datasets (Section 3.1).

- DRIVE: the official 20 training and 20 test images; training images 37-40 are the validation
  set (lists in ``configs/splits/drive_*.csv``).
- FIVES: the official 600 training and 200 test images; no validation set, the test images are
  used for validation (``allow_test_as_val: true``).
- HRF and VessMap: no official split. The images, sorted by file name, are shuffled with
  ``random.Random(seed)`` (``seed`` = ``dataset_params.seed``, 42 by default) and the first
  ``int(train_size * n)`` form the training set (``train_size: 0.7``: 31 of 45 HRF images, 70 of 100
  VessMap images). The other images are both the validation and the test set.
"""

import logging
import os
import random
from typing import Any, Dict, List

from torch.utils.data import Dataset, Subset

from thinvessel.data.dataset import FIVESDataset, PairedVesselDataset

logger = logging.getLogger(__name__)

DEFAULT_SPLIT_SEED = 42

# Native size (height, width) in px of the test images of each dataset.
NATIVE_IMAGE_SIZE: Dict[str, tuple] = {
    "drive": (584, 565),
    "hrf": (2336, 3504),
    "fives": (2048, 2048),
    "vessmap": (256, 256),
}


def dataset_class(dataset_name: str) -> type:
    """Dataset class of a dataset name (FIVES has its own, for the mask correction)."""
    name = dataset_name.lower()
    if name not in NATIVE_IMAGE_SIZE:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return FIVESDataset if name == "fives" else PairedVesselDataset


def random_split_indices(n_images: int, train_size: float, seed: int) -> List[List[int]]:
    """Training and held-out indices of a dataset without an official split.

    Args:
        n_images: Number of images (in sorted file-name order).
        train_size: Fraction of training images.
        seed: Seed of ``random.Random``.

    Returns:
        ``[train_indices, held_out_indices]``.
    """
    indices = list(range(n_images))
    random.Random(seed).shuffle(indices)
    n_train = int(train_size * n_images)
    return [indices[:n_train], indices[n_train:]]


def build_training_splits(dataset_name: str, dataset_params: Dict[str, Any]) -> Dict[str, Dataset]:
    """Training and validation datasets of a configuration.

    Args:
        dataset_name: ``drive``, ``hrf``, ``fives`` or ``vessmap``.
        dataset_params: ``dataset_params`` of the configuration, paths resolved.

    Returns:
        ``{"train_ds": ..., "val_ds": ...}``.

    Raises:
        ValueError: If an explicit split defines no validation folder and ``allow_test_as_val``
            is not set.
    """
    cls = dataset_class(dataset_name)
    image_size = dataset_params["image_size"]
    augment = dataset_params.get("augment", True)
    augmentations = dataset_params.get("augmentations")

    if "train_image_dir" in dataset_params and "test_image_dir" in dataset_params:
        train_ds = cls(
            image_dir=dataset_params["train_image_dir"],
            mask_dir=dataset_params["train_mask_dir"],
            image_size=image_size,
            mode="train",
            apply_transform=augment,
            augmentations=augmentations,
            split_csv=dataset_params.get("train_csv"),
        )
        if "val_image_dir" in dataset_params:
            val_dirs = (dataset_params["val_image_dir"], dataset_params["val_mask_dir"], dataset_params.get("val_csv"))
        elif dataset_params.get("allow_test_as_val", False):
            logger.warning("No validation split configured: the test images are used for validation.")
            val_dirs = (
                dataset_params["test_image_dir"],
                dataset_params["test_mask_dir"],
                dataset_params.get("test_csv"),
            )
        else:
            raise ValueError(
                "The configuration defines 'test_image_dir' but no 'val_image_dir'; define a validation split or "
                "set 'allow_test_as_val: true'."
            )
        val_ds = cls(
            image_dir=val_dirs[0],
            mask_dir=val_dirs[1],
            image_size=image_size,
            mode="val",
            apply_transform=False,
            split_csv=val_dirs[2],
        )
        return {"train_ds": train_ds, "val_ds": val_ds}

    image_dir, mask_dir = dataset_params["image_dir"], dataset_params["mask_dir"]
    probe = cls(image_dir=image_dir, mask_dir=mask_dir, image_size=image_size, apply_transform=False)
    train_idx, held_out_idx = random_split_indices(
        len(probe), dataset_params.get("train_size", 0.8), dataset_params.get("seed", DEFAULT_SPLIT_SEED)
    )
    train_ds = cls(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_size=image_size,
        mode="train",
        apply_transform=augment,
        augmentations=augmentations,
    )
    val_ds = cls(image_dir=image_dir, mask_dir=mask_dir, image_size=image_size, mode="val", apply_transform=False)
    return {"train_ds": Subset(train_ds, train_idx), "val_ds": Subset(val_ds, held_out_idx)}


def build_test_dataset(dataset_name: str, dataset_params: Dict[str, Any]) -> Dataset:
    """Test dataset of a configuration: the official test images, or the held-out images of HRF and VessMap.

    Args:
        dataset_name: ``drive``, ``hrf``, ``fives`` or ``vessmap``.
        dataset_params: ``dataset_params`` of the configuration, paths resolved.

    Returns:
        A dataset of full test images, in the order of their sample ids (``sample_00000``, ...).
    """
    cls = dataset_class(dataset_name)
    image_size = dataset_params["image_size"]
    if "test_image_dir" in dataset_params:
        return cls(
            image_dir=dataset_params["test_image_dir"],
            mask_dir=dataset_params["test_mask_dir"],
            image_size=image_size,
            mode="val",
            apply_transform=False,
            split_csv=dataset_params.get("test_csv"),
        )
    full = cls(
        image_dir=dataset_params["image_dir"],
        mask_dir=dataset_params["mask_dir"],
        image_size=image_size,
        mode="val",
        apply_transform=False,
    )
    _, held_out_idx = random_split_indices(
        len(full), dataset_params.get("train_size", 0.8), dataset_params.get("seed", DEFAULT_SPLIT_SEED)
    )
    return Subset(full, held_out_idx)


def held_out_names(dataset: Dataset) -> List[str]:
    """File names of the images of a test dataset, in sample order (for reports and checks)."""
    if isinstance(dataset, Subset):
        return [os.path.basename(dataset.dataset.images[i]) for i in dataset.indices]
    return [os.path.basename(p) for p in dataset.images]
