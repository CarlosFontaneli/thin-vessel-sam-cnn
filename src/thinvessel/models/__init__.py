"""The five architectures of the paper (Section 3.2, Table I).

:func:`build_model` maps a configuration's ``model_name`` and ``model_params`` to a module.
"""

from typing import Any, Dict

import torch.nn as nn

from thinvessel.models.convnext import ConvNeXtSeg
from thinvessel.models.convunext import ConvUNeXtAdapter
from thinvessel.models.litemedsam import LiteMedSAMSeg
from thinvessel.models.lwnet import LWNet
from thinvessel.models.medsam import MedSAMSeg

MODEL_CLASSES = {
    "lwnet": LWNet,
    "convunext": ConvUNeXtAdapter,
    "convnext": ConvNeXtSeg,
    "litemedsam": LiteMedSAMSeg,
    "medsam": MedSAMSeg,
}


def build_model(model_name: str, model_params: Dict[str, Any]) -> nn.Module:
    """Instantiates an architecture.

    Args:
        model_name: ``lwnet``, ``convunext``, ``convnext``, ``litemedsam`` or ``medsam``.
        model_params: Keyword arguments of the architecture (``model_params`` of a configuration).

    Returns:
        The model; LW-Net returns ``(coarse, refined)`` logits, the others one logit map.

    Raises:
        ValueError: If the model name is unknown.
    """
    name = model_name.lower()
    if name not in MODEL_CLASSES:
        raise ValueError(f"Unsupported model_name '{model_name}'. Must be one of {list(MODEL_CLASSES)}")
    return MODEL_CLASSES[name](**model_params)


__all__ = ["MODEL_CLASSES", "ConvNeXtSeg", "ConvUNeXtAdapter", "LWNet", "LiteMedSAMSeg", "MedSAMSeg", "build_model"]
