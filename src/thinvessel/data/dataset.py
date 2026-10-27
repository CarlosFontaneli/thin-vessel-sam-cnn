"""Paired image/mask datasets, training augmentation and the FIVES mask correction (Section 3.1, Appendix A.1).

A training sample is one image with random horizontal and vertical flips (probability 0.5 each)
and a random rotation in [-30, 30] degrees, then a random crop of the model's input size. An
image smaller than the input size (DRIVE and VessMap with MedSAM's 1024 px input) is resized to
it instead of cropped: bilinear for the image, nearest-exact for the mask. Validation and test
samples are the full images, unchanged.
"""

import logging
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
MASK_EXTENSIONS = (".png", ".PNG", ".tif", ".TIF", ".gif", ".GIF", ".jpg", ".jpeg")

DEFAULT_AUGMENTATIONS: Dict[str, float] = {
    "hflip_prob": 0.5,
    "vflip_prob": 0.5,
    "rotation_degrees": 30.0,
}

# A FIVES mask is corrupted when the sum of its outer one-pixel border (grey levels 0-255) exceeds
# this value; its 2 px border is then set to background (106 training and 5 test masks).
FIVES_BORDER_THRESHOLD = 500
FIVES_BORDER_PX = 2


class PairedVesselDataset(Dataset):
    """Images and vessel masks of one folder pair, matched by file name.

    Images are listed from ``image_dir`` (sorted by name, optionally restricted to the base names
    of a split list); the mask of an image has the same base name in ``mask_dir``, or, for
    DRIVE, the base name with ``_training``/``_test`` replaced by ``_manual1``. Files are read on
    access.

    Args:
        image_dir: Folder of the images.
        mask_dir: Folder of the masks.
        image_size: Training input size (px), an int or ``(height, width)``.
        mode: ``"train"``, ``"val"`` or ``"test"``. Augmentation is applied in ``"train"`` only.
        apply_transform: Whether to augment training samples.
        augmentations: Overrides of :data:`DEFAULT_AUGMENTATIONS`.
        split_csv: Text file with one image base name per line; ``None`` uses every image.

    Raises:
        FileNotFoundError: If a folder does not exist.
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_size: Union[int, Sequence[int]],
        mode: str = "train",
        apply_transform: bool = False,
        augmentations: Optional[Dict[str, float]] = None,
        split_csv: Optional[str] = None,
    ) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        self.mode = mode
        self.apply_transform_flag = apply_transform
        self.aug_params: Dict[str, float] = dict(DEFAULT_AUGMENTATIONS)
        if augmentations is not None:
            self.aug_params.update(augmentations)

        if not os.path.exists(self.image_dir) or not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"Missing dataset folder: {self.image_dir} or {self.mask_dir}.")

        all_images = [f for f in os.listdir(self.image_dir) if f.lower().endswith(IMAGE_EXTENSIONS)]
        if split_csv and os.path.exists(split_csv):
            with open(split_csv, encoding="utf-8") as f:
                bases = [line.strip() for line in f if line.strip()]
            self.image_files = []
            for base in bases:
                for name in all_images:
                    if os.path.splitext(name)[0] == base:
                        self.image_files.append(name)
                        break
        else:
            self.image_files = all_images
        self.image_files.sort()

        self.images: List[str] = []
        self.labels: List[str] = []
        for name in self.image_files:
            mask_path = _find_mask(self.mask_dir, os.path.splitext(name)[0])
            if mask_path is None:
                logger.warning("Missing mask for image %s", name)
                continue
            self.images.append(os.path.join(self.image_dir, name))
            self.labels.append(mask_path)
        logger.info("%s: mode=%s, %d image/mask pairs", type(self).__name__, self.mode, len(self.images))

    def __len__(self) -> int:
        return len(self.images)

    def _get_original_pair(self, idx: int) -> Tuple[Image.Image, Image.Image]:
        """Reads one image (RGB) and its mask (grey levels) before any augmentation."""
        return Image.open(self.images[idx]).convert("RGB"), Image.open(self.labels[idx]).convert("L")

    def apply_transform(self, image: Image.Image, mask: Image.Image, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flips, rotates and crops (or upscales) one training sample.

        Args:
            image: RGB image.
            mask: Grey-level mask.
            seed: Seeds Python's and PyTorch's generators for this sample.

        Returns:
            Image ``(3, H, W)`` and mask ``(1, H, W)`` tensors in [0, 1], ``(H, W) = image_size``.
        """
        random.seed(seed)
        torch.manual_seed(seed)

        if random.random() < self.aug_params["hflip_prob"]:
            image, mask = TF.hflip(image), TF.hflip(mask)
        if random.random() < self.aug_params["vflip_prob"]:
            image, mask = TF.vflip(image), TF.vflip(mask)
        degrees = self.aug_params["rotation_degrees"]
        angle = random.uniform(-degrees, degrees)
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

        img_w, img_h = image.size
        tgt_h, tgt_w = self.image_size
        if img_h < tgt_h or img_w < tgt_w:
            image = TF.resize(image, self.image_size)
            mask = TF.resize(mask, self.image_size, interpolation=InterpolationMode.NEAREST_EXACT)
        else:
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.image_size)
            image, mask = TF.crop(image, i, j, h, w), TF.crop(mask, i, j, h, w)
        return TF.to_tensor(image), TF.to_tensor(mask)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(image, mask)`` tensors: augmented in training mode, the full image otherwise.

        A per-sample seed is drawn from PyTorch's generator in every mode, as in the experiments.

        Raises:
            ValueError: If the image and mask sizes differ.
        """
        image, mask = self._get_original_pair(idx)
        if image.size != mask.size:
            raise ValueError(f"Image and mask size mismatch: {image.size} vs {mask.size}")
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
        if self.mode == "train" and self.apply_transform_flag:
            return self.apply_transform(image, mask, seed=seed)
        return TF.to_tensor(image), TF.to_tensor(mask)


class FIVESDataset(PairedVesselDataset):
    """FIVES, with the border of corrupted masks set to background.

    Some FIVES masks label the pixels along the image border as vessel. When the sum of a mask's
    outer one-pixel border exceeds :data:`FIVES_BORDER_THRESHOLD`, its :data:`FIVES_BORDER_PX` px
    border is set to 0 when the mask is read (106 of the 600 training masks and 5 of the 200 test
    masks). The check runs over every mask of the folder when the dataset is created.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.corrupted_bases: Set[str] = set()
        for name in self.image_files:
            base = os.path.splitext(os.path.basename(name))[0]
            mask_path = next(
                (
                    os.path.join(self.mask_dir, base + ext)
                    for ext in MASK_EXTENSIONS
                    if os.path.exists(os.path.join(self.mask_dir, base + ext))
                ),
                None,
            )
            if mask_path is None:
                continue
            label = np.array(Image.open(mask_path).convert("L"))
            border = int(label[0].sum() + label[-1].sum() + label[:, 0].sum() + label[:, -1].sum())
            if border > FIVES_BORDER_THRESHOLD:
                self.corrupted_bases.add(base)
        logger.info("FIVES: %d masks with a corrupted border in %s", len(self.corrupted_bases), self.mask_dir)

    def _get_original_pair(self, idx: int) -> Tuple[Image.Image, Image.Image]:
        image, mask = super()._get_original_pair(idx)
        if os.path.splitext(os.path.basename(self.image_files[idx]))[0] in self.corrupted_bases:
            label = np.array(mask)
            b = FIVES_BORDER_PX
            label[:b] = 0
            label[-b:] = 0
            label[:, :b] = 0
            label[:, -b:] = 0
            mask = Image.fromarray(label)
        return image, mask


def _find_mask(mask_dir: str, base: str) -> Optional[str]:
    """Mask path of an image base name: same base name, or the DRIVE ``_manual1`` name."""
    drive_base = base.replace("_training", "_manual1").replace("_test", "_manual1")
    for ext in MASK_EXTENSIONS:
        candidate = os.path.join(mask_dir, base + ext)
        if os.path.exists(candidate):
            return candidate
        candidate = os.path.join(mask_dir, drive_base + ext)
        if os.path.exists(candidate):
            return candidate
    return None
