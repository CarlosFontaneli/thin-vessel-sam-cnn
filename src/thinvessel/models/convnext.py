"""ConvNeXt: a ConvNeXt-Large encoder (torchvision) with a transposed-convolution decoder (Section 3.2).

The encoder is ``torchvision.models.convnext_large`` (``features`` only; its stride is 32), trained
from random initialization. The decoder is three transposed convolutions (strides 4, 4, 2) with
BatchNorm and ReLU between them. The model has 211,401,257 parameters; this count includes the
unused classification head of the torchvision model, which stays registered as ``backbone``.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torchvision.models import ConvNeXt_Large_Weights, convnext_large


class ConvNeXtSeg(nn.Module):
    """ConvNeXt-Large encoder with a transposed-convolution decoder.

    Args:
        in_c: Input channels (must be 3).
        n_classes: Output channels.
        drop_path_rate: Stochastic-depth rate of the encoder (searched in [0, 0.5]).
        pretrained: Initialise the encoder with torchvision's ImageNet weights (``false`` in every run).
        convnext_params: Optional ``encoder_params.dims`` (encoder widths; the last one is the
            decoder input) and ``decoder_params`` (``decoder_channels``, ``kernel_sizes``, ``strides``).
            ``encoder_params.depths`` is recorded in the configurations but not used: the depths
            are those of ``convnext_large``.

    Raises:
        ValueError: If ``in_c`` is not 3.
    """

    def __init__(
        self,
        in_c: int = 3,
        n_classes: int = 1,
        drop_path_rate: float = 0.1,
        pretrained: bool = False,
        convnext_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if in_c != 3:
            raise ValueError(f"ConvNeXt requires exactly 3 input channels, got {in_c}")
        self.in_c = in_c
        self.n_classes = n_classes

        dims = [192, 384, 768, 1536]
        decoder_chans, kernel_sizes, strides = [512, 128], [4, 4, 2], [4, 4, 2]
        if convnext_params:
            dims = convnext_params.get("encoder_params", {}).get("dims", dims)
            dec = convnext_params.get("decoder_params", {})
            decoder_chans = dec.get("decoder_channels", decoder_chans)
            kernel_sizes = dec.get("kernel_sizes", kernel_sizes)
            strides = dec.get("strides", strides)

        weights = ConvNeXt_Large_Weights.DEFAULT if pretrained else None
        self.backbone = convnext_large(weights=weights, stochastic_depth_prob=drop_path_rate)
        self.encoder = self.backbone.features

        c1, c2 = decoder_chans
        k1, k2, k3 = kernel_sizes
        s1, s2, s3 = strides
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dims[-1], c1, kernel_size=k1, stride=s1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c1, c2, kernel_size=k2, stride=s2),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c2, n_classes, kernel_size=k3, stride=s3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits ``(B, n_classes, H, W)``; resized bilinearly to ``(H, W)`` when ``H`` or ``W`` is not a multiple
        of 32."""
        if x.shape[1] != 3:
            raise ValueError(f"Expected 3 channels in input, got {x.shape[1]}")
        h, w = x.shape[2], x.shape[3]
        logits = self.decoder(self.encoder(x))
        if logits.shape[2] != h or logits.shape[3] != w:
            logits = nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return logits
