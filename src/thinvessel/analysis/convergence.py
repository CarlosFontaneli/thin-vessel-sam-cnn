"""Epochs until the selected run first reached 95% of its best validation Dice (Section 3.4, Fig. 6).

Read from the ``trial_<n>/metrics.csv`` files of a search. The selected run is the trial with the
highest validation Dice among the trials with at least 5 epochs of metrics. The epoch count is
resolved to the validation interval of the run (1 or 10 epochs); it is *censored* when the target
was already reached at the first validation.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

TARGET_FRACTION = 0.95
MIN_EPOCHS = 5


def epochs_to_fraction(metrics: pd.DataFrame, fraction: float = TARGET_FRACTION) -> Optional[int]:
    """First epoch whose ``val/dice`` reaches ``fraction`` of the run's best (None without a positive value)."""
    val = metrics.dropna(subset=["val/dice"])
    if val.empty or float(val["val/dice"].max()) <= 0.0:
        return None
    return int(val.loc[val["val/dice"] >= fraction * float(val["val/dice"].max()), "epoch"].iloc[0])


def trial_curve_summary(metrics_csv: Path, max_epochs: int = 1000) -> Optional[Dict[str, Any]]:
    """Best validation Dice, its epoch, epochs to target and validation schedule of one trial.

    Returns:
        ``None`` for a trial with fewer than :data:`MIN_EPOCHS` epochs or no validation.
    """
    df = pd.read_csv(metrics_csv, usecols=lambda c: c in ("epoch", "val/dice"))
    if df.empty or "epoch" not in df or df["epoch"].dropna().nunique() < MIN_EPOCHS:
        return None
    df = df[df["epoch"] < max_epochs]
    if "val/dice" not in df or not df["val/dice"].notna().any():
        return None
    best = df.loc[df["val/dice"].idxmax()]
    epochs = np.sort(df.dropna(subset=["val/dice"])["epoch"].unique())
    return {
        "best_val_dice": float(best["val/dice"]),
        "best_epoch": int(best["epoch"]),
        "epochs_to_target": epochs_to_fraction(df),
        "first_val_epoch": int(epochs[0]),
        "val_interval": int(np.median(np.diff(epochs))) if len(epochs) > 1 else None,
    }


def summarize_study(study_dir: Path) -> Dict[str, Any]:
    """Convergence of the selected run of one search.

    Args:
        study_dir: Folder of the search (``trial_<n>/metrics.csv``).

    Returns:
        ``selected_trial`` (``trial_<n>``), ``selected_best_val_dice``, ``selected_best_epoch``,
        ``selected_epochs_to_target``, ``selected_val_interval``, ``selected_target_censored`` and
        ``trials_with_curves``.

    Raises:
        ValueError: If no trial has a curve, or two trials tie for the best validation Dice.
    """
    rows = {}
    for path in sorted(study_dir.glob("trial_*/metrics.csv")):
        summary = trial_curve_summary(path)
        if summary is not None:
            rows[path.parent.name] = summary
    if not rows:
        raise ValueError(f"No trial curves in {study_dir}")
    top = max(r["best_val_dice"] for r in rows.values())
    winners = [name for name, r in rows.items() if r["best_val_dice"] == top]
    if len(winners) != 1:
        raise ValueError(f"{study_dir}: trials {winners} tie for the best validation Dice.")
    sel = rows[winners[0]]
    return {
        "selected_trial": winners[0],
        "selected_best_val_dice": sel["best_val_dice"],
        "selected_best_epoch": sel["best_epoch"],
        "selected_epochs_to_target": sel["epochs_to_target"],
        "selected_val_interval": sel["val_interval"],
        "selected_target_censored": bool(
            sel["epochs_to_target"] is not None and sel["epochs_to_target"] == sel["first_val_epoch"]
        ),
        "trials_with_curves": len(rows),
    }
