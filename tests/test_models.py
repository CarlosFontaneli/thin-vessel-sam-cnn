"""Architectures: parameter counts of Table I and forward shapes on CPU (Section 3.2)."""

from pathlib import Path

import pytest
import torch

from thinvessel.analysis.cost import count_parameters, windows_per_image
from thinvessel.config import load_config
from thinvessel.data import NATIVE_IMAGE_SIZE
from thinvessel.models import build_model

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def from_config(rel: str):
    cfg = load_config(CONFIGS / f"{rel}.yaml")
    params = dict(cfg["model_params"])
    if params.get("pretrained_weights_path"):
        params["pretrained_weights_path"] = None
    return build_model(cfg["model_name"], params).eval()


def test_lwnet() -> None:
    model = from_config("cnn/drive/lwnet")
    assert count_parameters(model)[0] == 68_482
    coarse, refined = model(torch.zeros(1, 3, 61, 67))  # odd sizes are padded back in the decoder
    assert coarse.shape == refined.shape == (1, 1, 61, 67)


@pytest.mark.parametrize("rel,n_params", [("cnn/hrf/convunext", 3_505_697), ("cnn/drive/convunext", 13_825_089)])
def test_convunext(rel: str, n_params: int) -> None:
    model = from_config(rel)
    assert count_parameters(model)[0] == n_params
    assert model(torch.zeros(1, 3, 64, 64)).shape == (1, 1, 64, 64)


def test_convnext() -> None:
    model = from_config("cnn/drive/convnext")
    assert count_parameters(model)[0] == 211_401_257
    assert model(torch.zeros(1, 3, 64, 64)).shape == (1, 1, 64, 64)


def test_litemedsam_and_compact() -> None:
    lite = from_config("finetuned/drive/litemedsam")
    assert count_parameters(lite) == (9_791_300, 9_785_080)  # the prompt encoder is frozen
    compact = from_config("compact/hrf/litemedsam")
    assert count_parameters(compact)[0] == 5_369_082
    with torch.no_grad():
        assert compact(torch.zeros(1, 3, 256, 256)).shape == (1, 1, 256, 256)


def test_medsam_and_compact() -> None:
    medsam = from_config("scratch/drive/medsam")
    assert count_parameters(medsam) == (93_735_472, 93_729_252)
    compact = from_config("compact/hrf/medsam")
    assert count_parameters(compact)[0] == 6_043_184
    with torch.no_grad():
        assert compact(torch.zeros(1, 3, 256, 256)).shape == (1, 1, 256, 256)


def test_windows_per_image() -> None:
    expected = {
        256: {"drive": 9, "hrf": 216, "fives": 121, "vessmap": 1},
        1024: {"drive": 1, "hrf": 15, "fives": 9, "vessmap": 1},
    }
    for roi, per_ds in expected.items():
        for ds, n in per_ds.items():
            assert windows_per_image(NATIVE_IMAGE_SIZE[ds], roi) == n


def test_windows_match_monai() -> None:
    from monai.data.utils import dense_patch_slices
    from monai.inferers.utils import _get_scan_interval

    for hw in NATIVE_IMAGE_SIZE.values():
        for roi in (256, 1024):
            size = tuple(max(s, roi) for s in hw)
            scan = _get_scan_interval(size, (roi, roi), 2, (0.25, 0.25))
            assert len(dense_patch_slices(size, (roi, roi), scan)) == windows_per_image(hw, roi)
