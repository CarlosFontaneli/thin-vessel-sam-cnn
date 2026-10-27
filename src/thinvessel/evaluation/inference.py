"""Test-set inference: sliding window at native resolution, or the upscaled protocol (Section 3.4).

- ``native``: MONAI sliding-window inference over the full test image at its native resolution
  (windows of the model's input size, 25% overlap, Gaussian weighting); an image smaller than a
  window is padded to it (the *padded* protocol for MedSAM on DRIVE and VessMap).
- ``upscaled``: an image smaller than the training input size (``dataset_params.image_size``) is
  first resized to it (bilinear), as in training, inferred in one window, and its logits are
  resized back to the native resolution (bilinear). The paper uses it for MedSAM on DRIVE and
  VessMap; for every other model and dataset it is identical to ``native``.

The sigmoid output is binarised at 0.5. Predicted and ground-truth masks are written as
``predicted_masks/sample_<i>.png`` and ``ground_truth/sample_<i>.png`` (0/255), where ``i`` is the
index of the image in the test dataset (sorted file names; the held-out order for HRF and VessMap).
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from thinvessel.training.module import VesselSegmentationModule

logger = logging.getLogger(__name__)

PROTOCOLS = ("native", "upscaled")
THRESHOLD = 0.5


def load_trained_module(checkpoint_path: Path, device: torch.device) -> Tuple[VesselSegmentationModule, Dict[str, Any]]:
    """Rebuilds a trained model from its Lightning checkpoint.

    The architecture comes from the checkpoint's ``hyper_parameters``. Every weight is restored by
    a strict load, so the published MedSAM/LiteMedSAM initialisation is not needed: the model is
    built with ``pretrained_weights_path = None``. For fine-tuned MedSAM this builds the encoder
    from ``medsam_params`` (LayerNorm eps 1e-5) instead of SAM's registry (eps 1e-6, used in
    training), as in the experiments (see README, "Known deviations").

    Args:
        checkpoint_path: ``.ckpt`` written by the search or by ``scripts/train.py``.
        device: Device of the model.

    Returns:
        The module in eval mode and the checkpoint's hyperparameters.

    Security:
        Lightning checkpoints store their hyperparameters with pickle, so they are loaded with
        ``weights_only=False``: load only checkpoints you trust.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    model_params = dict(hp["model_params"])
    if model_params.get("pretrained_weights_path") is not None:
        model_params["pretrained_weights_path"] = None
    module = VesselSegmentationModule(hp["model_name"], model_params, hp["training_params"])
    for name, buf in module.named_buffers():
        if "dense_prompt_embedding" in name:
            buf.data = buf.data.clone()  # an expanded view; give it its own memory before copying into it
    module.load_state_dict(ckpt["state_dict"])
    return module.to(device).eval(), hp


def infer_logits(
    module: VesselSegmentationModule, image: torch.Tensor, train_hw: Optional[Tuple[int, int]] = None
) -> torch.Tensor:
    """Logits ``(1, 1, H, W)`` of one image ``(1, 3, H, W)`` at native resolution.

    Args:
        module: Trained module (its ``inferer`` is the sliding-window inferer of validation).
        image: Input image.
        train_hw: Training input size for the upscaled protocol, or ``None`` for ``native``.
    """
    native_hw = (int(image.shape[-2]), int(image.shape[-1]))
    rescale = train_hw is not None and (native_hw[0] < train_hw[0] or native_hw[1] < train_hw[1])
    model_input = F.interpolate(image, size=train_hw, mode="bilinear", align_corners=False) if rescale else image
    logits = module.inferer(model_input, module.forward)
    if isinstance(logits, tuple):
        logits = logits[-1]
    if rescale:
        logits = F.interpolate(logits, size=native_hw, mode="bilinear", align_corners=False)
    return logits


def predict_dataset(
    module: VesselSegmentationModule,
    dataset: Dataset,
    output_dir: Path,
    device: torch.device,
    train_hw: Optional[Tuple[int, int]] = None,
) -> int:
    """Writes the binarised prediction and the ground truth of every image of a dataset.

    Args:
        module: Trained module in eval mode.
        dataset: Test dataset (full images).
        output_dir: Folder that receives ``predicted_masks/`` and ``ground_truth/``.
        device: Device of the module.
        train_hw: Training input size for the upscaled protocol, or ``None``.

    Returns:
        Number of images.
    """
    pred_dir, gt_dir = output_dir / "predicted_masks", output_dir / "ground_truth"
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    with torch.no_grad():
        for idx, (image, gt) in enumerate(loader):
            logits = infer_logits(module, image.to(device), train_hw)
            probs = torch.sigmoid(logits).cpu().numpy().squeeze()
            name = f"sample_{idx:05d}.png"
            Image.fromarray((probs > THRESHOLD).astype(np.uint8) * 255, mode="L").save(pred_dir / name)
            Image.fromarray((gt.numpy().squeeze() > 0.5).astype(np.uint8) * 255, mode="L").save(gt_dir / name)
            if (idx + 1) % 5 == 0 or idx + 1 == len(loader):
                logger.info("Predicted %d / %d images", idx + 1, len(loader))
    return len(loader)
