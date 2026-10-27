"""MedSAM (Ma et al., 2024): SAM's ViT-B image encoder, prompt encoder and mask decoder (Section 3.2).

The prompt is a fixed box covering the whole input window. Its sparse and dense embeddings are
computed once, when the model is built, with the prompt encoder, which is never trained. The
decoder predicts at a quarter of the input resolution; its logits are resized bilinearly to the
input. The fine-tuned model is built from the published checkpoint through SAM's model registry;
the models trained from scratch (full size and compact) are built from ``medsam_params``.

MedSAM fine-tuned and from scratch has 93,735,472 parameters (the 6,220 of the frozen prompt
encoder included).
"""

import contextlib
import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from thinvessel.models.third_party.segment_anything import sam_model_registry
from thinvessel.models.third_party.segment_anything.modeling import (
    ImageEncoderViT,
    MaskDecoder,
    PromptEncoder,
    TwoWayTransformer,
)
from thinvessel.models.third_party.segment_anything.modeling import image_encoder as _vit_module

logger = logging.getLogger(__name__)

_GRID_CACHE: Dict[Tuple[int, int, str], torch.Tensor] = {}


def cached_get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """SAM's ``get_rel_pos`` with the relative-coordinate index grid cached per size and device.

    Numerically identical to the vendored function; it only avoids rebuilding the index grid at
    every attention call.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1), size=max_rel_dist, mode="linear"
        )
        resized = resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        resized = rel_pos
    key = (q_size, k_size, str(rel_pos.device))
    if key not in _GRID_CACHE:
        q_coords = torch.arange(q_size, device=rel_pos.device)[:, None] * max(k_size / q_size, 1.0)
        k_coords = torch.arange(k_size, device=rel_pos.device)[None, :] * max(q_size / k_size, 1.0)
        _GRID_CACHE[key] = ((q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)).long()
    return resized[_GRID_CACHE[key]]


class MedSAMSeg(nn.Module):
    """MedSAM with a fixed whole-window box prompt.

    Args:
        pretrained_weights_path: Published MedSAM checkpoint (``medsam_vit_b.pth``) for fine-tuning,
            or ``None`` to build the model from ``medsam_params`` with random weights.
        img_size: Input window (px) when built from the checkpoint.
        freeze_image_encoder: Freeze the image encoder (``false`` in every run).
        medsam_params: ``vit_b_params``, ``prompt_encoder_params``, ``mask_decoder_params`` and
            ``transformer_params`` of a model trained from scratch.

    Raises:
        FileNotFoundError: If ``pretrained_weights_path`` does not exist.
        ValueError: If neither weights nor ``medsam_params`` are given.
    """

    def __init__(
        self,
        pretrained_weights_path: Optional[str] = None,
        img_size: int = 1024,
        freeze_image_encoder: bool = False,
        medsam_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        _vit_module.get_rel_pos = cached_get_rel_pos
        self.img_size = img_size
        self.frozen_image_encoder = freeze_image_encoder

        if pretrained_weights_path is not None:
            if not os.path.exists(pretrained_weights_path):
                raise FileNotFoundError(f"Pretrained weights not found at {pretrained_weights_path}.")
            logger.info("MedSAM: loading %s", pretrained_weights_path)
            # The registry builds ViT-B with LayerNorm eps=1e-6 (SAM's value); see README, "Known deviations".
            sam = sam_model_registry["vit_b"](checkpoint=pretrained_weights_path)
            self.image_encoder, self.mask_decoder, self.prompt_encoder = (
                sam.image_encoder,
                sam.mask_decoder,
                sam.prompt_encoder,
            )
        elif medsam_params is not None:
            vit_params = medsam_params.get("vit_b_params", {})
            # Built this way, the encoder's LayerNorms keep PyTorch's default eps=1e-5.
            self.image_encoder = ImageEncoderViT(**vit_params)
            self.prompt_encoder = PromptEncoder(**medsam_params.get("prompt_encoder_params", {}))
            self.mask_decoder = MaskDecoder(
                transformer=TwoWayTransformer(**medsam_params.get("transformer_params", {})),
                **medsam_params.get("mask_decoder_params", {}),
            )
            self.img_size = vit_params.get("img_size", self.img_size)
        else:
            raise ValueError("MedSAM needs pretrained_weights_path or medsam_params.")

        if freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
        for p in self.prompt_encoder.parameters():
            p.requires_grad = False

        box = torch.tensor([[0, 0, self.img_size - 1, self.img_size - 1]]).float()
        with torch.no_grad():
            sparse, dense = self.prompt_encoder(points=None, boxes=box, masks=None)
        self.register_buffer("sparse_prompt_embedding", sparse.clone())
        self.register_buffer("dense_prompt_embedding", dense.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits ``(B, 1, H, W)`` for an input ``(B, 3, H, W)`` with ``H = W = img_size``."""
        if x.shape[2:] != (self.img_size, self.img_size):
            logger.warning("MedSAM expects %dx%d inputs, got %s.", self.img_size, self.img_size, tuple(x.shape[2:]))
        ctx = torch.no_grad() if self.frozen_image_encoder else contextlib.nullcontext()
        with ctx:
            embeddings = self.image_encoder(x)
        b = x.shape[0]
        low_res, _ = self.mask_decoder(
            image_embeddings=embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=self.sparse_prompt_embedding.expand(b, -1, -1),
            dense_prompt_embeddings=self.dense_prompt_embedding.expand(b, -1, -1, -1),
            multimask_output=False,
        )
        return F.interpolate(low_res, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
