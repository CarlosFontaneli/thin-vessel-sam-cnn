"""Runs (or resumes) the hyperparameter search of one configuration (Section 3.3, Appendix A).

    python scripts/search.py --config configs/cnn/drive/lwnet.yaml [--trials 100]

Output: ``<outputs>/search/<regime>/<dataset>/<model>/`` with ``optuna_study.db``, ``train.log``,
``trial_<n>/{config.yaml, metrics.csv}``, the best checkpoint ``best_model_val_dice_<dice>.ckpt``
and ``best_trial.json``.
"""

import argparse
import warnings
from pathlib import Path

from thinvessel.config import Study, add_path_arguments, load_config, resolve_path_arguments
from thinvessel.training.search import run_search, setup_logging

warnings.filterwarnings("ignore", category=UserWarning, message=".*non-tuple sequence for multidimensional indexing.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*The cuda.cudart module is deprecated.*")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", required=True, type=Path, help="Configuration file under configs/.")
    parser.add_argument("--trials", "-t", type=int, default=100, help="Finished trials to reach (default 100).")
    parser.add_argument("--epochs", "-e", type=int, default=None, help="Maximum epochs per trial (default: config).")
    add_path_arguments(parser)
    args = parser.parse_args()
    data_root, weights_dir, outputs_dir = resolve_path_arguments(args)
    out = outputs_dir / "search" / Study.from_config_path(args.config).relpath
    setup_logging(out)
    run_search(load_config(args.config), args.config, out, data_root, weights_dir, args.trials, args.epochs)


if __name__ == "__main__":
    main()
