"""Produces the cost, convergence and search-record tables that the figures and tables read.

    python scripts/analyse.py [--steps cost convergence search] [--output-dir results]

- ``cost.csv``: parameters, GFLOPs per window and per test image of the 36 configurations, measured
  on the architectures of ``configs/`` (no data or weights needed).
- ``convergence.csv``: selected trial and epochs to 95% of its best validation Dice, per search
  (reads ``<outputs>/search/<study>/trial_<n>/metrics.csv``).
- ``search_trials.csv``: one row per trial of every search (reads ``optuna_study.db``,
  ``train.log`` and ``trial_<n>/config.yaml`` under ``<outputs>/search/``).
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from thinvessel.analysis.convergence import summarize_study
from thinvessel.analysis.cost import measure_configuration
from thinvessel.analysis.search_records import study_records
from thinvessel.config import add_path_arguments, all_studies, load_config, resolve_path_arguments
from thinvessel.data import NATIVE_IMAGE_SIZE

logger = logging.getLogger("analyse")
STEPS = ("cost", "convergence", "search")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS))
    parser.add_argument("--configs-dir", type=Path, default=Path(__file__).resolve().parents[1] / "configs")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default <outputs>/analysis.")
    add_path_arguments(parser)
    args = parser.parse_args()
    _, _, outputs_dir = resolve_path_arguments(args)
    out_dir = args.output_dir or outputs_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    if "cost" in args.steps:
        rows = []
        for s in all_studies():
            config = load_config(args.configs_dir / s.regime / s.dataset / f"{s.model}.yaml")
            rows.append(
                {
                    "configuration": s.configuration,
                    "dataset": s.dataset,
                    **measure_configuration(config, NATIVE_IMAGE_SIZE[s.dataset]),
                }
            )
        pd.DataFrame(rows).to_csv(out_dir / "cost.csv", index=False)
        logger.info("cost.csv: %d configurations", len(rows))

    if "convergence" in args.steps:
        rows = [
            {
                "regime": s.regime,
                "dataset": s.dataset,
                "model": s.model,
                "configuration": s.configuration,
                **summarize_study(outputs_dir / "search" / s.relpath),
            }
            for s in all_studies()
        ]
        pd.DataFrame(rows).to_csv(out_dir / "convergence.csv", index=False)
        logger.info("convergence.csv: %d searches", len(rows))

    if "search" in args.steps:
        frames = []
        for s in all_studies():
            records = study_records(outputs_dir / "search" / s.relpath)
            records.insert(0, "configuration", s.configuration)
            records.insert(1, "dataset", s.dataset)
            frames.append(records)
        table = pd.concat(frames, ignore_index=True)
        table.to_csv(out_dir / "search_trials.csv", index=False)
        logger.info("search_trials.csv: %d trials", len(table))


if __name__ == "__main__":
    main()
