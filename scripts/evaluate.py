"""Scores the saved test predictions of the 36 studies into one per-image table (Section 3.4).

    python scripts/evaluate.py [--output results/per_image_metrics.csv]

Reads ``<outputs>/predictions/<regime>/<dataset>/<model>/{native,upscaled}/``. The table has one row
per dataset, configuration, protocol and test image, with every metric of the paper (columns:
``thinvessel.evaluation.metrics.score_pair``). It is the input of the statistics and of the figures.
"""

import argparse
import logging
from pathlib import Path

from thinvessel.config import add_path_arguments, resolve_path_arguments
from thinvessel.evaluation.scoring import score_predictions


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output", type=Path, default=None, help="CSV to write (default <outputs>/metrics/per_image_metrics.csv)."
    )
    parser.add_argument("--workers", type=int, default=8, help="Worker processes (default 8).")
    add_path_arguments(parser)
    args = parser.parse_args()
    _, _, outputs_dir = resolve_path_arguments(args)
    table = score_predictions(outputs_dir, workers=args.workers)
    out = args.output or outputs_dir / "metrics" / "per_image_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    logging.getLogger("evaluate").info("%d rows written to %s", len(table), out)


if __name__ == "__main__":
    main()
