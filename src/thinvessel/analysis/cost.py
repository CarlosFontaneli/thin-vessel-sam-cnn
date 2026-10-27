"""Parameters and FLOPs per window and per test image (Section 3.4, Fig. 3).

Parameters are counted with ``Module.parameters()`` (a tensor shared by two submodules counts
once; frozen parameters count). FLOPs are counted by fvcore for one forward pass over one window
of the model's input size (``training_params.roi_size``), one fused multiply-add = one FLOP. The
cost per image is the cost per window times the number of sliding windows MONAI uses over a full
test image (25% overlap; one window when the image is smaller than the window, padded or upscaled).
Both depend only on the architecture, so they are measured on the model built from its
configuration, without trained weights.
"""

import logging
import math
from typing import Any, Dict, Tuple

import torch

from thinvessel.models import build_model
from thinvessel.training.module import SLIDING_WINDOW_OVERLAP, default_roi_size

logger = logging.getLogger(__name__)


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Total and trainable parameters of a model."""
    params = list(model.parameters())
    return sum(p.numel() for p in params), sum(p.numel() for p in params if p.requires_grad)


def measure_flops(model: torch.nn.Module, roi: int, channels: int = 3) -> int:
    """fvcore FLOPs of one forward pass of a ``(1, channels, roi, roi)`` input (model in eval mode)."""
    from fvcore.nn import FlopCountAnalysis

    with torch.no_grad():
        analysis = FlopCountAnalysis(model, torch.zeros(1, channels, roi, roi))
        analysis.unsupported_ops_warnings(False)
        analysis.uncalled_modules_warnings(False)
        return int(analysis.total())


def windows_along(size: int, roi: int, overlap: float = SLIDING_WINDOW_OVERLAP) -> int:
    """Sliding windows along one axis of ``size`` px (MONAI's ``dense_patch_slices`` grid)."""
    if size <= roi:
        return 1
    step = max(int(roi * (1.0 - overlap)), 1)
    return int(math.ceil((size - roi) / step)) + 1


def windows_per_image(native_hw: Tuple[int, int], roi: int) -> int:
    """Sliding windows over a full image of ``native_hw = (H, W)`` px."""
    return windows_along(native_hw[0], roi) * windows_along(native_hw[1], roi)


def measure_configuration(config: Dict[str, Any], native_hw: Tuple[int, int]) -> Dict[str, Any]:
    """Cost of the architecture of a configuration.

    Args:
        config: Configuration (``model_name``, ``model_params``, ``training_params``).
        native_hw: Native size of the dataset's test images (px).

    Returns:
        ``params_total``, ``params_trainable``, ``params_m`` (millions, 6 decimals), ``flops``,
        ``gflops`` (per window, 6 decimals), ``roi_size`` (px), ``windows_per_image`` and
        ``gflops_per_image`` (``gflops * windows``, 12 decimals).
    """
    params = dict(config["model_params"])
    if params.get("pretrained_weights_path") is not None:
        params["pretrained_weights_path"] = None  # the architecture is the same with random weights
    model = build_model(config["model_name"], params).eval()
    total, trainable = count_parameters(model)
    roi = int(config["training_params"].get("roi_size") or default_roi_size(config["model_name"]))
    flops = measure_flops(model, roi)
    gflops = round(flops / 1e9, 6)
    windows = windows_per_image(native_hw, roi)
    logger.info("%s: %d parameters, %.3f GFLOPs per window, %d windows", config["model_name"], total, gflops, windows)
    return {
        "params_total": total,
        "params_trainable": trainable,
        "params_m": round(total / 1e6, 6),
        "flops": flops,
        "gflops": gflops,
        "roi_size": roi,
        "windows_per_image": windows,
        "gflops_per_image": round(gflops * windows, 12),
    }
