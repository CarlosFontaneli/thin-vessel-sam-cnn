"""Lightning module: soft Dice loss, AdamW with cosine annealing, sliding-window validation (Section 3.3, Appendix A).

Loss: MONAI's ``DiceLoss(sigmoid=True, squared_pred=True)``, the soft Dice loss with squared terms
in the denominator and smoothing 1e-5, computed per image and averaged over the batch. For LW-Net
the loss is ``loss(refined) + deep_sup_weight * loss(coarse)`` with ``deep_sup_weight: 1.0``.

Optimiser: AdamW (``lr``, ``weight_decay`` from the configuration) and
``CosineAnnealingLR(T_max=max_epochs, eta_min=1e-6)`` stepped once per epoch.

Validation: MONAI sliding-window inference over each full validation image (windows of
``roi_size`` px: 1024 for MedSAM, 256 otherwise; 25% overlap; Gaussian weighting; 4 windows per
batch). ``val/dice`` pools intersection and union over all validation images (a micro-average,
not the mean of per-image Dice), with the sigmoid output binarised at 0.5.
"""

import logging
from typing import Any, Dict, Tuple, Union

import monai
import pytorch_lightning as pl
import torch
from monai.inferers import SlidingWindowInferer
from torchmetrics import Metric

from thinvessel.models import build_model

logger = logging.getLogger(__name__)

SLIDING_WINDOW_OVERLAP = 0.25
SLIDING_WINDOW_BATCH = 4


def default_roi_size(model_name: str) -> int:
    """Input window (px) of an architecture: 1024 for MedSAM, 256 for the others."""
    return 1024 if model_name.lower() == "medsam" else 256


class BinaryDiceMetric(Metric):
    """Dice of binarised predictions, pooled over all images of an epoch: 2|A∩B| / (|A| + |B| + 1e-6)."""

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, threshold: float = 0.5, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.threshold = threshold
        self.add_state("intersection", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("union", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulates one batch of logits and binary targets."""
        preds_bin = (torch.sigmoid(preds) > self.threshold).float()
        if preds_bin.shape != targets.shape:
            targets = targets.view_as(preds_bin)
        self.intersection += (preds_bin * targets).sum()
        self.union += preds_bin.sum() + targets.sum()

    def compute(self) -> torch.Tensor:
        return (2.0 * self.intersection) / (self.union + 1e-6)


class BinaryIoUMetric(Metric):
    """IoU of binarised predictions, pooled over all images of an epoch: |A∩B| / (|A∪B| + 1e-6)."""

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, threshold: float = 0.5, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.threshold = threshold
        self.add_state("intersection", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("union", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulates one batch of logits and binary targets."""
        preds_bin = (torch.sigmoid(preds) > self.threshold).float()
        if preds_bin.shape != targets.shape:
            targets = targets.view_as(preds_bin)
        inter = (preds_bin * targets).sum()
        self.intersection += inter
        self.union += preds_bin.sum() + targets.sum() - inter

    def compute(self) -> torch.Tensor:
        return self.intersection / (self.union + 1e-6)


class VesselSegmentationModule(pl.LightningModule):
    """Training, validation and optimisation of one architecture.

    The constructor arguments are saved as the checkpoint's ``hyper_parameters``; inference
    rebuilds the model from them.

    Args:
        model_name: Architecture (see :func:`thinvessel.models.build_model`).
        model_params: Architecture arguments.
        training_params: ``lr``, ``weight_decay``, ``max_epochs``, ``roi_size``, ``deep_sup_weight``,
            ``optimizer`` (must be ``AdamW``).
    """

    def __init__(self, model_name: str, model_params: Dict[str, Any], training_params: Dict[str, Any]) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.model_params = model_params
        self.training_params = training_params

        logger.info("Instantiating model: %s", model_name)
        self.model = build_model(model_name, model_params)
        self.loss_fn = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
        self.deep_sup_weight = float(training_params.get("deep_sup_weight", 0.4))

        roi = training_params.get("roi_size", default_roi_size(model_name))
        self.roi_size = (roi, roi) if isinstance(roi, int) else tuple(roi)
        self.inferer = SlidingWindowInferer(
            roi_size=self.roi_size,
            sw_batch_size=SLIDING_WINDOW_BATCH,
            overlap=SLIDING_WINDOW_OVERLAP,
            mode="gaussian",
        )
        self.val_dice = BinaryDiceMetric()
        self.val_iou = BinaryIoUMetric()

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.model(x)

    def _loss(self, preds: Any, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loss and the logits used for the metrics (the refined output for LW-Net)."""
        if isinstance(preds, tuple):
            coarse, refined = preds
            return self.loss_fn(refined, mask) + self.deep_sup_weight * self.loss_fn(coarse, mask), refined
        return self.loss_fn(preds, mask), preds

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        image, mask = batch[0], batch[1]
        loss, _ = self._loss(self(image), mask)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=image.size(0))
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        image, mask = batch[0], batch[1]
        loss, final = self._loss(self.inferer(image, self.forward), mask)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=image.size(0))
        self.val_dice.update(final, mask)
        self.val_iou.update(final, mask)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("val/dice", self.val_dice.compute(), prog_bar=True)
        self.log("val/iou", self.val_iou.compute(), prog_bar=True)
        self.val_dice.reset()
        self.val_iou.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        """AdamW and an epoch-wise cosine schedule from ``lr`` to 1e-6 over ``max_epochs``.

        Raises:
            ValueError: If ``training_params.optimizer`` is not ``AdamW`` (every run used AdamW).
        """
        name = self.training_params.get("optimizer", "AdamW")
        if name != "AdamW":
            raise ValueError(f"Unsupported optimizer '{name}': every run of the paper used AdamW.")
        lr = float(self.training_params.get("lr", 1e-4))
        weight_decay = float(self.training_params.get("weight_decay", 1e-4))
        logger.info("Optimizer AdamW: lr=%.2e, weight_decay=%.2e", lr, weight_decay)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.training_params.get("max_epochs", 1000), eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1}}
