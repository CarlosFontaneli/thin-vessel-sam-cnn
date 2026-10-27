"""Trial records of the hyperparameter searches (Appendix A, Tables V-VII).

For each trial of a search: its number and Optuna state, the best validation Dice it reached (the
maximum over its validations, pruned trials included; the value Optuna stores is the *last*
validation Dice), its duration, the sampled parameters, the settings it trained with
(``trial_<n>/config.yaml``) and whether it ran out of GPU memory (``train.log``).

A search without a usable ``optuna_study.db`` (missing or empty; VessMap LW-Net and ConvUNeXt) is
read from ``train.log``: sampled parameters and start times, no validation Dice, state ``LOGGED``.
"""

import ast
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

from thinvessel.config import ARCHITECTURE_KEYS

_STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_START = re.compile(r"Starting Trial (\d+) with params: (\{.*\})")
_OOM = re.compile(r"Trial (\d+) failed due to recoverable error: OutOfMemoryError")


def has_study_db(study_dir: Path) -> bool:
    """True if the search has a non-empty ``optuna_study.db``."""
    db = study_dir / "optuna_study.db"
    return db.exists() and db.stat().st_size > 0


def load_trials(study_dir: Path) -> pd.DataFrame:
    """Trials of one search: ``number``, ``state``, ``best_val_dice``, ``hours`` and one column per parameter.

    Categorical parameters are decoded to their choice; numeric ones are Optuna's stored floats.
    """
    if has_study_db(study_dir):
        con = sqlite3.connect(f"file:{study_dir / 'optuna_study.db'}?mode=ro", uri=True)
        trials = pd.read_sql(
            "select trial_id, number, state, (julianday(datetime_complete) - "
            "julianday(datetime_start)) * 24 as hours from trials",
            con,
        )
        best = pd.read_sql(
            "select trial_id, max(intermediate_value) as best_val_dice "
            "from trial_intermediate_values group by trial_id",
            con,
        )
        params = pd.read_sql("select trial_id, param_name, param_value, distribution_json from trial_params", con)
        con.close()

        def decode(row: pd.Series) -> Any:
            dist = json.loads(row["distribution_json"])
            if dist["name"] == "CategoricalDistribution":
                return dist["attributes"]["choices"][int(row["param_value"])]
            return row["param_value"]

        params["value"] = params.apply(decode, axis=1)
        wide = params.pivot(index="trial_id", columns="param_name", values="value").reset_index()
        out = trials.merge(best, on="trial_id", how="left").merge(wide, on="trial_id", how="left")
        return out.drop(columns="trial_id").sort_values("number").reset_index(drop=True)

    rows, last_time = [], None
    for line in (study_dir / "train.log").read_text(errors="ignore").splitlines():
        m = _STAMP.match(line)
        if not m:
            continue
        t = pd.Timestamp(m.group(1))
        start = _START.search(line)
        if start:
            if rows:
                rows[-1]["hours"] = (t - rows[-1].pop("_start")).total_seconds() / 3600
            rows.append(
                {
                    "number": int(start.group(1)),
                    "state": "LOGGED",
                    "best_val_dice": np.nan,
                    "_start": t,
                    **ast.literal_eval(start.group(2)),
                }
            )
        last_time = t
    if rows:
        rows[-1]["hours"] = (last_time - rows[-1].pop("_start")).total_seconds() / 3600
    return pd.DataFrame(rows)


def out_of_memory_trials(study_dir: Path) -> set:
    """Numbers of the trials that ran out of GPU memory (pruned), from ``train.log``."""
    log = study_dir / "train.log"
    return {int(n) for n in _OOM.findall(log.read_text(errors="ignore"))} if log.exists() else set()


def applied_value(config: Dict[str, Any], key: str) -> Any:
    """Value of a sampled parameter in the configuration a trial trained with."""
    if key in ("lr", "weight_decay"):
        return config["training_params"][key]
    if key == "batch_size":
        return config["dataset_params"][key]
    mp = config["model_params"]
    if key in ARCHITECTURE_KEYS:
        inner = (
            mp["medsam_params"]["vit_b_params"] if "medsam_params" in mp else mp["litemedsam_params"]["tiny_vit_params"]
        )
        return inner[key]
    return mp[key]


def trial_settings(study_dir: Path, number: int) -> Dict[str, Any]:
    """Training settings of one trial (``trial_<n>/config.yaml``): optimizer, epochs, validation interval, patience."""
    path = study_dir / f"trial_{number}" / "config.yaml"
    if not path.exists():
        return {}
    tp = yaml.safe_load(path.read_text())["training_params"]
    return {
        "optimizer": tp["optimizer"],
        "max_epochs": tp["max_epochs"],
        "check_val_every_n_epoch": tp.get("check_val_every_n_epoch", 1),
        "patience": tp["patience"],
    }


def study_records(study_dir: Path) -> pd.DataFrame:
    """Trial table of one search with the applied settings and the out-of-memory flag."""
    trials = load_trials(study_dir)
    oom = out_of_memory_trials(study_dir)
    settings = pd.DataFrame(
        [
            {"number": int(n), **{f"applied_{k}": v for k, v in trial_settings(study_dir, int(n)).items()}}
            for n in trials["number"]
        ]
    )
    out = trials.merge(settings, on="number", how="left")
    out["out_of_memory"] = out["number"].astype(int).isin(oom)
    return out
