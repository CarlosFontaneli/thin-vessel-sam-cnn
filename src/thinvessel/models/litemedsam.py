"""LiteMedSAM (Ma et al., 2024): a TinyViT encoder with SAM's prompt encoder and mask decoder (Section 3.2).

The prompt is a fixed box covering the 256 px window, embedded once when the model is built, and
the prompt encoder is never trained. For fine-tuning, the published checkpoint (``lite_medsam.pth``)
is loaded after the model is built. The box embedding is therefore computed with the randomly
initialised prompt encoder, because the checkpoint does not contain it (see README, "Known
deviations"). LiteMedSAM has 9,791,300 parameters.
"""

import logging
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from thinvessel.models.third_party.litemedsam import LiteMedSAM
from thinvessel.models.third_party.segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from thinvessel.models.third_party.tiny_vit_sam import TinyViT

logger = logging.getLogger(__name__)


class LiteMedSAMSeg(nn.Module):
    """LiteMedSAM with a fixed whole-window box prompt (the trained model lives under ``model``).

    Args:
        litemedsam_params: ``tiny_vit_params``, ``prompt_encoder_params``, ``mask_decoder_params``
            and ``transformer_params``.
        freeze_image_encoder: Freeze the TinyViT encoder (``false`` in every run).
        pretrained_weights_path: Published LiteMedSAM checkpoint, or ``None`` for random weights.

    Raises:
        FileNotFoundError: If ``pretrained_weights_path`` does not exist.
    """

    def __init__(
        self,
        litemedsam_params: Dict[str, Any],
        freeze_image_encoder: bool = False,
        pretrained_weights_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        tiny_vit_params = litemedsam_params.get("tiny_vit_params", {})
        img_size = tiny_vit_params.get("img_size", 256)
        image_encoder = TinyViT(**tiny_vit_params)
        prompt_encoder = PromptEncoder(**litemedsam_params.get("prompt_encoder_params", {}))
        transformer = TwoWayTransformer(**litemedsam_params.get("transformer_params", {}))
        mask_decoder = MaskDecoder(transformer=transformer, **litemedsam_params.get("mask_decoder_params", {}))
        self.model = LiteMedSAM(
            image_encoder=image_encoder,
            mask_decoder=mask_decoder,
            prompt_encoder=prompt_encoder,
            img_size=(img_size, img_size),
            freeze_image_encoder=freeze_image_encoder,
        )
        if pretrained_weights_path is not None:
            if not os.path.exists(pretrained_weights_path):
                raise FileNotFoundError(f"Pretrained weights not found at {pretrained_weights_path}.")
            self._load_weights(pretrained_weights_path)

    def _load_weights(self, path: str) -> None:
        """Loads the published weights (non-strict: the checkpoint has no box-embedding buffers)."""
        logger.info("LiteMedSAM: loading %s", path)
        with torch.no_grad():
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            state_dict = ckpt["model"] if "model" in ckpt else ckpt
            # The dense prompt embedding is an expanded view; clone it so that it owns its memory.
            for name, buf in self.model.named_buffers():
                if "dense_prompt_embedding" in name:
                    buf.data = buf.data.clone()
            msg = self.model.load_state_dict(state_dict, strict=False)
        if msg.missing_keys:
            logger.warning("LiteMedSAM: keys not in the checkpoint: %s", msg.missing_keys[:5])
        if msg.unexpected_keys:
            logger.warning("LiteMedSAM: unexpected checkpoint keys: %s", msg.unexpected_keys[:5])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits ``(B, 1, H, W)`` for an input ``(B, 3, H, W)``."""
        return self.model(x)
