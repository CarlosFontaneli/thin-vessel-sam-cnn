"""Per-image segmentation metrics (Section 3.4).

All metrics take a binarised prediction ``P`` and ground truth ``G`` of the same shape (non-zero =
vessel) and return fractions in [0, 1].

- Dice ``2|G∩P| / (|G| + |P|)`` over the full image (and IoU, pixel precision and recall).
- Centerline recall ``|S(G)∩P| / |S(G)|`` and centerline precision ``|S(P)∩G| / |S(P)|``, where
  ``S`` is the skeleton from ``skimage.morphology.skeletonize`` (Zhang-Suen thinning in 2D),
  with no distance tolerance.
- Recall by radius: the ground-truth skeleton pixels are grouped by their radius, the value of
  the Euclidean distance transform of ``G`` (SciPy) at the pixel, in px, into the bins
  ``[0.5, 1.5), [1.5, 2.5), [2.5, 3.5), [3.5, 4.5), [4.5, inf)``. The first bin also takes radii
  below 0.5 and the last one radii above its upper edge. A vessel is thin when its radius is
  below 1.5 px: thin-vessel recall is the recall of the first bin.
- Thin-vessel precision: the share of the predicted skeleton pixels whose radius *in the
  prediction* is below 1.5 px that lie inside ``G``.

Undefined values (empty denominator) are NaN, except that an empty prediction scores 0 for the
centerline recall and precision and for the recall of every non-empty bin.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import scipy.ndimage as ndi
from PIL import Image
from skimage.morphology import skeletonize

logger = logging.getLogger(__name__)

RADIUS_BINS: List[float] = [0.5, 1.5, 2.5, 3.5, 4.5, 100.0]
THIN_RADIUS_PX = RADIUS_BINS[1]


def load_binary_mask(path: Union[str, Path]) -> np.ndarray:
    """Reads a mask image as ``uint8`` in {0, 1}: grey levels >= 128 of 255 are vessel."""
    img = np.array(Image.open(path).convert("L"))
    return (
        ((img.astype(np.float32) / 255.0) >= 0.5).astype(np.uint8) if img.max() > 1 else (img >= 0.5).astype(np.uint8)
    )


def extract_centerline(mask: np.ndarray) -> np.ndarray:
    """Skeleton (``uint8`` in {0, 1}) of a binary mask; all zeros for an empty mask."""
    binary = mask > 0
    if not binary.any():
        return np.zeros(mask.shape, dtype=np.uint8)
    return skeletonize(binary).astype(np.uint8)


def distance_map(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance (px, ``float32``) of every vessel pixel to the nearest background pixel."""
    binary = (mask > 0).astype(np.uint8)
    if not binary.any():
        return np.zeros(mask.shape, dtype=np.float32)
    return ndi.distance_transform_edt(binary).astype(np.float32)


def _bin_masks(skeleton: np.ndarray, radius: np.ndarray, bins: Sequence[float]) -> List[np.ndarray]:
    """Skeleton pixels of each radius bin; the first and last bins are open below and above."""
    out = []
    last = len(bins) - 2
    for i in range(len(bins) - 1):
        lo, hi = float(bins[i]), float(bins[i + 1])
        if i == 0:
            out.append((skeleton == 1) & (radius < hi))
        elif i == last:
            out.append((skeleton == 1) & (radius >= lo))
        else:
            out.append((skeleton == 1) & (radius >= lo) & (radius < hi))
    return out


def overlap_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Pixel Dice, IoU, precision and recall (both masks empty: 1; empty prediction: precision 0)."""
    p, g = pred > 0, gt > 0
    p_sum, g_sum, inter = int(p.sum()), int(g.sum()), int((p & g).sum())
    if p_sum + g_sum == 0:
        return {"dice": 1.0, "iou": 1.0, "precision": 1.0, "recall": 1.0}
    return {
        "dice": float(2.0 * inter / (p_sum + g_sum)),
        "iou": float(inter / (p_sum + g_sum - inter)),
        "precision": float(inter / p_sum) if p_sum > 0 else 0.0,
        "recall": float(inter / g_sum) if g_sum > 0 else 1.0,
    }


def radius_binned_recall(gt: np.ndarray, pred: np.ndarray, bins: Sequence[float] = RADIUS_BINS) -> List[Dict[str, Any]]:
    """Centerline recall per ground-truth radius bin.

    Returns:
        One dict per bin: ``range`` (px), ``n_pixels`` (ground-truth skeleton pixels), ``n_hit``
        (of those, inside the prediction) and ``recall`` (NaN for an empty bin; 0 for every
        non-empty bin when the prediction is empty).

    Raises:
        ValueError: If the shapes differ or the ground truth is empty.
    """
    g, p = (gt > 0).astype(np.uint8), (pred > 0).astype(np.uint8)
    if g.shape != p.shape:
        raise ValueError(f"Shape mismatch: gt {g.shape} vs pred {p.shape}")
    if not g.any():
        raise ValueError("Ground truth mask contains 0 foreground pixels.")
    empty_pred = not p.any()
    if empty_pred:
        logger.warning("Empty prediction: recall 0 in every non-empty bin.")
    out = []
    for (lo, hi), sel in zip(zip(bins[:-1], bins[1:]), _bin_masks(extract_centerline(g), distance_map(g), bins)):
        n = int(sel.sum())
        hit = 0 if empty_pred or n == 0 else int((sel & (p == 1)).sum())
        recall = (0.0 if n > 0 else float("nan")) if empty_pred else (hit / n if n > 0 else float("nan"))
        out.append({"range": [float(lo), float(hi)], "n_pixels": n, "n_hit": hit, "recall": float(recall)})
    return out


def radius_binned_precision(
    gt: np.ndarray, pred: np.ndarray, bins: Sequence[float] = RADIUS_BINS
) -> List[Dict[str, Any]]:
    """Centerline precision per radius bin of the *predicted* vessels (radius from the prediction's EDT).

    Returns:
        One dict per bin: ``range`` (px), ``n_pixels`` (predicted skeleton pixels), ``n_hit`` (of
        those, inside the ground truth) and ``precision`` (NaN for an empty bin).
    """
    g, p = (gt > 0).astype(np.uint8), (pred > 0).astype(np.uint8)
    if g.shape != p.shape:
        raise ValueError(f"Shape mismatch: gt {g.shape} vs pred {p.shape}")
    out = []
    for (lo, hi), sel in zip(zip(bins[:-1], bins[1:]), _bin_masks(extract_centerline(p), distance_map(p), bins)):
        n, hit = int(sel.sum()), int((sel & (g == 1)).sum())
        out.append(
            {
                "range": [float(lo), float(hi)],
                "n_pixels": n,
                "n_hit": hit,
                "precision": float(hit / n) if n > 0 else float("nan"),
            }
        )
    return out


def centerline_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Centerline recall and precision over all calibers (0 and 0 for an empty prediction).

    Raises:
        ValueError: If the shapes differ or the ground truth is empty.
    """
    g, p = (gt > 0).astype(np.uint8), (pred > 0).astype(np.uint8)
    if g.shape != p.shape:
        raise ValueError(f"Shape mismatch: gt {g.shape} vs pred {p.shape}")
    if not g.any():
        raise ValueError("Ground truth mask contains 0 foreground pixels.")
    if not p.any():
        logger.warning("Empty prediction: centerline recall and precision 0.")
        return {"centerline_recall": 0.0, "centerline_precision": 0.0}
    s_g, s_p = extract_centerline(g), extract_centerline(p)
    return {
        "centerline_recall": float(np.sum((s_g == 1) & (p == 1)) / max(int(s_g.sum()), 1)),
        "centerline_precision": float(np.sum((s_p == 1) & (g == 1)) / max(int(s_p.sum()), 1)),
    }


def bin_column(lo: float, hi: float, prefix: str) -> str:
    """Column name of a radius bin, e.g. ``centerline_recall_radius_1.5_2.5`` (``inf`` for the last bin)."""
    return f"{prefix}_radius_{lo:g}_{'inf' if hi >= 100 else f'{hi:g}'}"


def score_pair(pred: np.ndarray, gt: np.ndarray, bins: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """Every per-image value of the paper for one prediction.

    Returns:
        ``height``, ``width``, ``dice``, ``iou``, ``pixel_precision``, ``pixel_recall``,
        ``thin_vessel_recall`` and its denominator ``n_gt_thin_centerline_pixels``,
        ``thin_vessel_precision`` and ``n_pred_thin_centerline_pixels``, ``centerline_recall``,
        ``centerline_precision``, and for the other radius bins ``centerline_recall_radius_<lo>_<hi>``
        with ``n_gt_centerline_pixels_radius_<lo>_<hi>``.
    """
    bins = list(RADIUS_BINS if bins is None else bins)
    ov = overlap_metrics(pred, gt)
    rec = radius_binned_recall(gt, pred, bins)
    thin_prec = radius_binned_precision(gt, pred, bins)[0]
    cl = centerline_metrics(gt, pred)
    row: Dict[str, Any] = {
        "height": int(gt.shape[0]),
        "width": int(gt.shape[1]),
        "dice": ov["dice"],
        "iou": ov["iou"],
        "pixel_precision": ov["precision"],
        "pixel_recall": ov["recall"],
        "thin_vessel_recall": rec[0]["recall"],
        "n_gt_thin_centerline_pixels": rec[0]["n_pixels"],
        "thin_vessel_precision": thin_prec["precision"],
        "n_pred_thin_centerline_pixels": thin_prec["n_pixels"],
        **cl,
    }
    for b in rec[1:]:
        lo, hi = b["range"]
        row[bin_column(lo, hi, "centerline_recall")] = b["recall"]
        row[bin_column(lo, hi, "n_gt_centerline_pixels")] = b["n_pixels"]
    return row
