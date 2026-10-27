"""Scoring of saved test predictions into the per-image table of the paper (Section 3.4).

Predictions live under ``<outputs>/predictions/<regime>/<dataset>/<model>/<protocol>/`` (see
:mod:`thinvessel.evaluation.inference`). :func:`score_predictions` scores every image of every
study found there and returns one row per (dataset, configuration, protocol, image) with the
columns of :func:`thinvessel.evaluation.metrics.score_pair`.
"""

import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from thinvessel.config import Study, all_studies
from thinvessel.evaluation.metrics import load_binary_mask, score_pair

logger = logging.getLogger(__name__)

KEY_COLUMNS = ["dataset", "configuration", "protocol", "sample_id"]


def paper_protocol(study: Study) -> str:
    """Protocol of the paper's results: ``upscaled`` for full-size MedSAM on DRIVE and VessMap, else ``native``."""
    if study.model == "medsam" and study.regime in ("finetuned", "scratch") and study.dataset in ("drive", "vessmap"):
        return "upscaled"
    return "native"


def prediction_dir(outputs_dir: Path, study: Study, protocol: str) -> Path:
    """Folder of the saved test predictions of a study under one protocol."""
    return outputs_dir / "predictions" / study.relpath / protocol


def _score(task: Tuple[Dict[str, Any], str, str]) -> Dict[str, Any]:
    meta, pred_path, gt_path = task
    return {**meta, **score_pair(load_binary_mask(pred_path), load_binary_mask(gt_path))}


def score_predictions(outputs_dir: Path, workers: int = 8, studies: Optional[List[Study]] = None) -> pd.DataFrame:
    """Scores the saved test predictions of the studies.

    Both protocols are scored where they exist (the padded/native and the upscaled predictions of
    MedSAM on DRIVE and VessMap); for every other study only ``native`` is expected.

    Args:
        outputs_dir: Outputs folder.
        workers: Worker processes.
        studies: Studies to score (default: the 36 of the paper).

    Returns:
        The per-image table, sorted by dataset, configuration, protocol and sample id.

    Raises:
        FileNotFoundError: If a study has no native predictions, or if a predicted mask has no ground truth.
    """
    tasks = []
    for study in studies or all_studies():
        protocols = ["native", "upscaled"] if paper_protocol(study) == "upscaled" else ["native"]
        for protocol in protocols:
            folder = prediction_dir(outputs_dir, study, protocol)
            preds = sorted((folder / "predicted_masks").glob("*.png"))
            if not preds:
                raise FileNotFoundError(f"No predictions in {folder}")
            for pred in preds:
                gt = folder / "ground_truth" / pred.name
                if not gt.exists():
                    raise FileNotFoundError(f"No ground truth for {pred}")
                meta = {
                    "dataset": study.dataset,
                    "configuration": study.configuration,
                    "protocol": protocol,
                    "sample_id": pred.stem,
                }
                tasks.append((meta, str(pred), str(gt)))
    logger.info("Scoring %d predicted masks with %d workers", len(tasks), workers)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_score, tasks, chunksize=4))
    return pd.DataFrame(rows).sort_values(KEY_COLUMNS).reset_index(drop=True)
