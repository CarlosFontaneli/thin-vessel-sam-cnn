"""Hyperparameter search with Optuna and training of one configuration (Section 3.3, Appendix A).

One search = one Optuna study of 100 trials with the TPE sampler (``seed`` of the configuration)
and the median pruner on the validation Dice, stored in ``optuna_study.db`` so that an interrupted
search resumes. Every trial reseeds all generators with the configuration's ``seed`` (42), trains
for at most ``max_epochs`` epochs and stops early after ``patience`` validations without
improvement of ``val/dice``; the checkpoint with the best ``val/dice`` is kept.

:func:`train_configuration` runs the same trial code on a configuration as it is, without a
study: on a selected-trial configuration from ``configs/`` it repeats the run that the paper
reports (none of the 36 selected trials was pruned).
"""

import gc
import glob
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import optuna
import pytorch_lightning as pl
import torch
import yaml
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from thinvessel.config import apply_params, resolve_paths, sample_params
from thinvessel.data import build_training_splits
from thinvessel.training.module import VesselSegmentationModule

logger = logging.getLogger(__name__)

MONITOR = "val/dice"
CHECKPOINT_NAME = "best_model"
# Keys of a configuration file that describe the search, not the training run.
METADATA_KEYS = ("search_space", "selected_trial")


def setup_logging(log_dir: Path) -> None:
    """Logs to ``<log_dir>/train.log`` and to standard output.

    Args:
        log_dir: Output folder of a search or a training run.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    file_handler = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
    root.addHandler(file_handler)
    root.addHandler(console)


def best_validation_row(metrics_csv: Path) -> Dict[str, float]:
    """Row of a Lightning ``metrics.csv`` with the highest ``val/dice`` (empty if there is none)."""
    import csv

    if not metrics_csv.exists():
        return {}
    best, best_val = {}, -float("inf")
    with open(metrics_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            value = row.get(MONITOR)
            if value and float(value) > best_val:
                best_val, best = float(value), row
    return {k: float(v) for k, v in best.items() if v}


def _copy_latest_metrics(run_dir: Path) -> Path:
    """Copies the newest ``logs/**/metrics.csv`` of a run to ``<run_dir>/metrics.csv``."""
    dest = run_dir / "metrics.csv"
    candidates = [Path(p) for p in glob.glob(str(run_dir / "**" / "metrics.csv"), recursive=True) if Path(p) != dest]
    if candidates:
        shutil.copy(max(candidates, key=lambda p: p.stat().st_mtime), dest)
    return dest


def _loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Training (shuffled) and validation (one full image per batch) loaders with spawned workers."""
    params = config["dataset_params"]
    splits = build_training_splits(config["dataset_name"], params)
    logger.info("Dataset splits: train=%d, val=%d", len(splits["train_ds"]), len(splits["val_ds"]))
    common = dict(
        num_workers=params.get("num_workers", 4),
        pin_memory=True,
        persistent_workers=params.get("persistent_workers", False),
        multiprocessing_context="spawn",
    )
    train = DataLoader(splits["train_ds"], batch_size=params.get("batch_size", 16), shuffle=True, **common)
    val = DataLoader(splits["val_ds"], batch_size=1, shuffle=False, **common)
    return train, val


def run_trial(
    config: Dict[str, Any], run_dir: Path, max_epochs: int, extra_callbacks: Optional[list] = None
) -> Tuple[float, Dict[str, float]]:
    """Trains one configuration and keeps its best checkpoint as ``<run_dir>/best_model.ckpt``.

    Args:
        config: Configuration to train, paths resolved.
        run_dir: Output folder (checkpoint, ``metrics.csv``, Lightning logs).
        max_epochs: Maximum number of epochs.
        extra_callbacks: Further Lightning callbacks (the Optuna pruning callback in a search).

    Returns:
        The ``val/dice`` of the last validation (the value reported to Optuna) and the
        ``metrics.csv`` row of the best validation.
    """
    pl.seed_everything(config.get("seed", 42))
    train_loader, val_loader = _loaders(config)
    module = VesselSegmentationModule(config["model_name"], config["model_params"], config["training_params"])
    tp = config["training_params"]
    callbacks = list(extra_callbacks or []) + [
        ModelCheckpoint(dirpath=run_dir, filename=CHECKPOINT_NAME, monitor=MONITOR, mode="max", save_top_k=1),
        EarlyStopping(monitor=MONITOR, patience=tp.get("patience", 50), mode="max"),
        RichProgressBar(),
    ]
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        logger=CSVLogger(save_dir=run_dir, name="logs"),
        enable_progress_bar=True,
        check_val_every_n_epoch=tp.get("check_val_every_n_epoch", 1),
        log_every_n_steps=1,
    )
    trainer.fit(module, train_loader, val_loader)
    last = float(trainer.callback_metrics.get(MONITOR, 0.0))
    best = best_validation_row(_copy_latest_metrics(run_dir))
    logger.info("Run finished: last val/dice %.4f, best val/dice %.4f", last, best.get(MONITOR, float("nan")))
    return last, best


def _is_recoverable(exc: BaseException) -> bool:
    """Errors that prune a trial instead of stopping the search (GPU memory, CUDA, data-loader workers)."""
    msg = str(exc).lower()
    return (
        isinstance(exc, torch.cuda.OutOfMemoryError)
        or "out of memory" in msg
        or "cuda" in msg
        or "dataloader worker" in msg
        or "exited unexpectedly" in msg
    )


def _best_existing_trial(study_dir: Path) -> Dict[str, Any]:
    """Best trial whose checkpoint is still on disk (to resume a search)."""
    best: Dict[str, Any] = {"score": -1.0, "checkpoint": None, "trial": None}
    for ckpt in study_dir.glob(f"trial_*/{CHECKPOINT_NAME}.ckpt"):
        try:
            number = int(ckpt.parent.name.split("_", 1)[1])
        except (ValueError, IndexError):
            continue
        score = best_validation_row(ckpt.parent / "metrics.csv").get(MONITOR, -1.0)
        if score > best["score"]:
            best = {"score": score, "checkpoint": ckpt, "trial": number}
    return best


def run_search(
    config: Dict[str, Any],
    config_path: Path,
    study_dir: Path,
    data_root: Path,
    weights_dir: Path,
    n_trials: int = 100,
    max_epochs: Optional[int] = None,
) -> optuna.Study:
    """Runs (or resumes) the hyperparameter search of one configuration.

    Each trial writes the configuration it trains with to ``trial_<n>/config.yaml`` (relative
    paths), the record of what the trial used. Only the best trial keeps its checkpoint, copied to
    ``best_model_val_dice_<dice>.ckpt`` in ``study_dir``. A trial that fails with a recoverable
    error (out of GPU memory, CUDA or data-loader worker errors) is pruned.

    Args:
        config: Configuration with a ``search_space`` (paths unresolved).
        config_path: Path of the configuration file.
        study_dir: Output folder of the study.
        data_root: Root of the datasets.
        weights_dir: Directory of the published checkpoints.
        n_trials: Total number of finished (complete or pruned) trials to reach.
        max_epochs: Maximum epochs per trial (default ``training_params.max_epochs``).

    Returns:
        The Optuna study.
    """
    search_space = config.get("search_space")
    if not search_space:
        raise ValueError("The configuration has no 'search_space'.")
    epochs = max_epochs or int(config["training_params"]["max_epochs"])
    study_dir.mkdir(parents=True, exist_ok=True)
    best = _best_existing_trial(study_dir)

    def objective(trial: optuna.Trial) -> float:
        nonlocal best
        sampled = sample_params(trial, search_space)
        trial_config = apply_params({k: v for k, v in config.items() if k not in METADATA_KEYS}, sampled)
        trial_dir = study_dir / f"trial_{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        with open(trial_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(trial_config, f)
        logger.info("Starting Trial %d with params: %s", trial.number, sampled)
        try:
            last, best_row = run_trial(
                resolve_paths(trial_config, config_path, data_root, weights_dir),
                trial_dir,
                epochs,
                extra_callbacks=[PyTorchLightningPruningCallback(trial, monitor=MONITOR)],
            )
            ckpt = trial_dir / f"{CHECKPOINT_NAME}.ckpt"
            score = best_row.get(MONITOR, last)
            if ckpt.exists():
                if score > best["score"]:
                    if best["checkpoint"] is not None and Path(best["checkpoint"]).exists():
                        os.remove(best["checkpoint"])
                    for old in study_dir.glob("best_model_val_dice_*.ckpt"):
                        old.unlink()
                    shutil.copy(ckpt, study_dir / f"best_model_val_dice_{score:.4f}.ckpt")
                    best = {"score": score, "checkpoint": ckpt, "trial": trial.number}
                    (study_dir / "best_trial.json").write_text(
                        json.dumps({"trial": trial.number, "best_val_dice": score}, indent=1), encoding="utf-8"
                    )
                else:
                    os.remove(ckpt)
            return last
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if not _is_recoverable(exc):
                raise
            logger.warning(
                "Trial %d failed due to recoverable error: %s. Pruning trial...", trial.number, type(exc).__name__
            )
            raise optuna.TrialPruned() from exc
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    study = optuna.create_study(
        study_name=f"{config['dataset_name']}_{config['model_name']}",
        storage=f"sqlite:///{(study_dir / 'optuna_study.db').resolve()}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.get("seed", 42)),
        pruner=optuna.pruners.MedianPruner(),  # Optuna's default pruner, made explicit
    )
    finished = sum(t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED) for t in study.trials)
    if finished >= n_trials:
        logger.info("Study already has %d finished trials (target %d).", finished, n_trials)
        return study
    logger.info("Running %d more trials (target %d, finished %d).", n_trials - finished, n_trials, finished)
    study.optimize(objective, n_trials=n_trials - finished)
    return study


def train_configuration(
    config: Dict[str, Any],
    config_path: Path,
    run_dir: Path,
    data_root: Path,
    weights_dir: Path,
    max_epochs: Optional[int] = None,
) -> Path:
    """Trains a configuration as it is (e.g. the selected trial of a search) with the trial code.

    Args:
        config: Configuration (paths unresolved); ``search_space`` and ``selected_trial`` are ignored.
        config_path: Path of the configuration file.
        run_dir: Output folder.
        data_root: Root of the datasets.
        weights_dir: Directory of the published checkpoints.
        max_epochs: Maximum epochs (default ``training_params.max_epochs``).

    Returns:
        Path of the best checkpoint, ``best_model_val_dice_<dice>.ckpt``.
    """
    run_config = {k: v for k, v in config.items() if k not in METADATA_KEYS}
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(run_config, f)
    epochs = max_epochs or int(run_config["training_params"]["max_epochs"])
    _, best_row = run_trial(resolve_paths(run_config, config_path, data_root, weights_dir), run_dir, epochs)
    ckpt = run_dir / f"{CHECKPOINT_NAME}.ckpt"
    final = run_dir / f"best_model_val_dice_{best_row.get(MONITOR, 0.0):.4f}.ckpt"
    ckpt.rename(final)
    return final
