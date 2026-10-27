"""Metrics on synthetic masks (Section 3.4)."""

import math

import numpy as np
import pytest
from PIL import Image

from thinvessel.evaluation.metrics import (
    centerline_metrics,
    distance_map,
    extract_centerline,
    load_binary_mask,
    overlap_metrics,
    radius_binned_precision,
    radius_binned_recall,
    score_pair,
)


def vessels() -> np.ndarray:
    """A 1 px wide (thin) horizontal vessel and a 7 px wide vertical one, apart from each other."""
    m = np.zeros((64, 64), np.uint8)
    m[10, 5:60] = 1  # radius 1 px
    m[20:60, 40:47] = 1  # radius up to 4 px
    return m


def test_overlap_metrics() -> None:
    gt = vessels()
    pred = gt.copy()
    pred[20:60, 40:47] = 0
    r = overlap_metrics(pred, gt)
    inter, p, g = int(pred.sum()), int(pred.sum()), int(gt.sum())
    assert r["dice"] == pytest.approx(2 * inter / (p + g))
    assert r["precision"] == 1.0 and r["recall"] == pytest.approx(inter / g)
    assert overlap_metrics(np.zeros_like(gt), np.zeros_like(gt))["dice"] == 1.0
    assert overlap_metrics(np.zeros_like(gt), gt)["precision"] == 0.0


def test_skeleton_and_radius() -> None:
    gt = vessels()
    skel = extract_centerline(gt)
    assert set(np.unique(skel)) <= {0, 1}
    assert skel[10, 10] == 1  # a 1 px line is its own skeleton
    assert distance_map(gt)[10, 10] == 1.0
    assert distance_map(gt)[40, 43] == 4.0
    assert extract_centerline(np.zeros((8, 8))).sum() == 0


def test_thin_recall_perfect_and_partial() -> None:
    gt = vessels()
    bins = radius_binned_recall(gt, gt)
    assert bins[0]["recall"] == 1.0 and bins[0]["n_pixels"] > 0
    pred = gt.copy()
    pred[10, :] = 0  # miss the thin vessel only
    bins = radius_binned_recall(gt, pred)
    assert bins[0]["recall"] == 0.0
    assert all(b["recall"] == 1.0 for b in bins[1:] if b["n_pixels"] > 0)


def test_empty_prediction_conventions() -> None:
    gt = vessels()
    empty = np.zeros_like(gt)
    bins = radius_binned_recall(gt, empty)
    assert all((b["recall"] == 0.0) if b["n_pixels"] else math.isnan(b["recall"]) for b in bins)
    assert centerline_metrics(gt, empty) == {"centerline_recall": 0.0, "centerline_precision": 0.0}
    assert math.isnan(radius_binned_precision(gt, empty)[0]["precision"])
    with pytest.raises(ValueError):
        radius_binned_recall(empty, gt)


def test_thin_precision_uses_prediction_radius() -> None:
    gt = vessels()
    pred = np.zeros_like(gt)
    pred[30, 5:30] = 1  # a thin false vessel outside the ground truth
    first = radius_binned_precision(gt, pred)[0]
    assert first["n_pixels"] > 0 and first["precision"] == 0.0


def test_score_pair_columns() -> None:
    gt = vessels()
    row = score_pair(gt, gt)
    for col in (
        "dice",
        "thin_vessel_recall",
        "thin_vessel_precision",
        "centerline_recall",
        "centerline_precision",
        "centerline_recall_radius_4.5_inf",
        "n_gt_centerline_pixels_radius_1.5_2.5",
        "height",
        "width",
    ):
        assert col in row
    assert row["dice"] == 1.0 and row["centerline_recall"] == 1.0 and row["centerline_precision"] == 1.0


def test_load_binary_mask(tmp_path) -> None:
    a = np.array([[0, 127, 128, 255]], np.uint8)
    Image.fromarray(a).save(tmp_path / "m.png")
    assert load_binary_mask(tmp_path / "m.png").tolist() == [[0, 0, 1, 1]]
