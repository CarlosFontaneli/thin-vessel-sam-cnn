"""Trains one configuration as it is: the selected trial of its search (Section 3.3, Appendix A).

    python scripts/train.py --config configs/finetuned/drive/medsam.yaml

This repeats the run the paper reports for that configuration and dataset, without the search
(same seed and trial code). Output: ``<outputs>/train/<regime>/<dataset>/<model>/`` with
``config.yaml``, ``train.log``, ``metrics.csv`` and ``best_model_val_dice_<dice>.ckpt``.
Use ``--epochs 2`` for a quick check of the setup.
"""

import argparse
import warnings
from pathlib import Path

from thinvessel.config import Study, add_path_arguments, load_config, resolve_path_arguments
from thinvessel.training.search import setup_logging, train_configuration

warnings.filterwarnings("ignore", category=UserWarning, message=".*non-tuple sequence for multidimensional indexing.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*The cuda.cudart module is deprecated.*")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", required=True, type=Path, help="Configuration file under configs/.")
    parser.add_argument("--epochs", "-e", type=int, default=None, help="Maximum epochs (default: config, 1000).")
    add_path_arguments(parser)
    args = parser.parse_args()
    data_root, weights_dir, outputs_dir = resolve_path_arguments(args)
    out = outputs_dir / "train" / Study.from_config_path(args.config).relpath
    setup_logging(out)
    train_configuration(load_config(args.config), args.config, out, data_root, weights_dir, args.epochs)


if __name__ == "__main__":
    main()
