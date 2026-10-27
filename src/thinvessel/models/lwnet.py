"""LW-Net: our implementation of Little W-Net (Galdran et al., 2022), Section 3.2.

Two U-Nets in sequence: the second receives the image and the sigmoid of the first one's output.
The model returns both outputs (coarse, refined); the training loss is the sum of the Dice loss
of the two (``deep_sup_weight: 1.0``), and validation and inference use the refined output.
The building blocks are adapted from the Little W-Net code of A. Galdran (MIT License, see
``THIRD_PARTY_NOTICES.md``). With ``layers = [8, 16, 32]`` the model has 68,482 parameters.
"""

import json
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution without bias."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ConvBlock(nn.Module):
    """[MaxPool] -> (Conv -> ReLU -> BN) x 2, with an optional 1x1 residual shortcut."""

    def __init__(self, in_c: int, out_c: int, k_sz: int = 3, shortcut: bool = False, pool: bool = True) -> None:
        super().__init__()
        self.shortcut = nn.Sequential(conv1x1(in_c, out_c), nn.BatchNorm2d(out_c)) if shortcut else False
        pad = (k_sz - 1) // 2
        self.pool = nn.MaxPool2d(kernel_size=2) if pool else False
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k_sz, padding=pad),
            nn.ReLU(),
            nn.BatchNorm2d(out_c),
            nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding=pad),
            nn.ReLU(),
            nn.BatchNorm2d(out_c),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool:
            x = self.pool(x)
        out = self.block(x)
        return out + self.shortcut(x) if self.shortcut else out


class UpsampleBlock(nn.Module):
    """x2 upsampling: transposed convolution (``transp_conv``) or bilinear + 1x1 convolution (``up_conv``)."""

    def __init__(self, in_c: int, out_c: int, up_mode: str = "transp_conv") -> None:
        super().__init__()
        if up_mode == "transp_conv":
            layers: List[nn.Module] = [nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)]
        elif up_mode == "up_conv":
            layers = [
                nn.Upsample(mode="bilinear", scale_factor=2, align_corners=False),
                nn.Conv2d(in_c, out_c, kernel_size=1),
            ]
        else:
            raise ValueError(f"Upsampling mode '{up_mode}' not supported. Use 'transp_conv' or 'up_conv'.")
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvBridgeBlock(nn.Module):
    """Conv -> ReLU -> BN on a skip connection (channels preserved)."""

    def __init__(self, channels: int, k_sz: int = 3) -> None:
        super().__init__()
        pad = (k_sz - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=k_sz, padding=pad), nn.ReLU(), nn.BatchNorm2d(channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpConvBlock(nn.Module):
    """Upsample, concatenate the (bridged) skip connection, then a :class:`ConvBlock`.

    When downsampling truncated an odd size (565 -> 282 -> 564), the upsampled tensor is
    zero-padded on the right and bottom to the size of the skip connection.
    """

    def __init__(
        self,
        in_c: int,
        out_c: int,
        k_sz: int = 3,
        up_mode: str = "up_conv",
        conv_bridge: bool = False,
        shortcut: bool = False,
    ) -> None:
        super().__init__()
        self.conv_bridge = conv_bridge
        self.up_layer = UpsampleBlock(in_c, out_c, up_mode=up_mode)
        self.conv_layer = ConvBlock(2 * out_c, out_c, k_sz=k_sz, shortcut=shortcut, pool=False)
        if self.conv_bridge:
            self.conv_bridge_layer = ConvBridgeBlock(out_c, k_sz=k_sz)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        up = self.up_layer(x)
        diff_h, diff_w = skip.size(2) - up.size(2), skip.size(3) - up.size(3)
        if diff_h > 0 or diff_w > 0:
            up = nn.functional.pad(up, (0, diff_w, 0, diff_h))
        skip = self.conv_bridge_layer(skip) if self.conv_bridge else skip
        return self.conv_layer(torch.cat([up, skip], dim=1))


class UNet(nn.Module):
    """U-Net with ``len(layers)`` levels of widths ``layers``; Kaiming-normal convolutions, unit BatchNorm."""

    def __init__(
        self,
        in_c: int,
        n_classes: int,
        layers: Sequence[int],
        k_sz: int = 3,
        up_mode: str = "transp_conv",
        conv_bridge: bool = True,
        shortcut: bool = True,
    ) -> None:
        super().__init__()
        self.first = ConvBlock(in_c=in_c, out_c=layers[0], k_sz=k_sz, shortcut=shortcut, pool=False)
        self.down_path = nn.ModuleList(
            ConvBlock(in_c=layers[i], out_c=layers[i + 1], k_sz=k_sz, shortcut=shortcut, pool=True)
            for i in range(len(layers) - 1)
        )
        rev = list(reversed(layers))
        self.up_path = nn.ModuleList(
            UpConvBlock(
                in_c=rev[i], out_c=rev[i + 1], k_sz=k_sz, up_mode=up_mode, conv_bridge=conv_bridge, shortcut=shortcut
            )
            for i in range(len(layers) - 1)
        )
        self.final = nn.Conv2d(layers[0], n_classes, kernel_size=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first(x)
        skips = []
        for down in self.down_path:
            skips.append(x)
            x = down(x)
        for up, skip in zip(self.up_path, reversed(skips)):
            x = up(x, skip)
        return self.final(x)


class LWNet(nn.Module):
    """Little W-Net: two U-Nets, the second refining the first.

    Args:
        in_c: Input channels.
        n_classes: Output channels.
        layers: Widths of the three levels, a list or its JSON string (``"[8, 16, 32]"``).
        k_sz: Convolution kernel size.
        up_mode: ``transp_conv`` or ``up_conv``.
        conv_bridge: Convolve the skip connections.
        shortcut: Residual shortcuts in the convolution blocks.
    """

    def __init__(
        self,
        in_c: int = 3,
        n_classes: int = 1,
        layers: Optional[Union[Sequence[int], str]] = None,
        k_sz: int = 3,
        up_mode: str = "transp_conv",
        conv_bridge: bool = True,
        shortcut: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(layers, str):
            layers = json.loads(layers)
        layers = list(layers) if layers is not None else [8, 16, 32]
        self.u_net1 = UNet(in_c, n_classes, layers, k_sz, up_mode, conv_bridge, shortcut)
        self.u_net2 = UNet(in_c + n_classes, n_classes, layers, k_sz, up_mode, conv_bridge, shortcut)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the coarse and the refined logits, each ``(B, n_classes, H, W)``."""
        out1 = self.u_net1(x)
        out2 = self.u_net2(torch.cat([x, torch.sigmoid(out1)], dim=1))
        return out1, out2
