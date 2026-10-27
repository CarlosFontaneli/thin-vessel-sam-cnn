"""Checks the dataset folders against the splits of the paper (Section 3.1).

    python scripts/check_data.py [--data-root data] [--datasets drive hrf fives vessmap]

For each dataset it builds the training, validation and test sets exactly as training and inference do, from
the released configurations, and checks their sizes: DRIVE 16/4/20, HRF 31/14/14, FIVES 600/200/200,
VessMap 70/30/30 (training/validation/test; the test images of HRF, FIVES and VessMap are also the validation
set). For FIVES it also counts the masks with a corrupted border (106 training, 5 test). The expected layout is
in the README.
"""

import argparse
import logging
import sys
from pathlib import Path

from thinvessel.config import DATASETS, add_path_arguments, load_config, resolve_path_arguments, resolve_paths
from thinvessel.data import build_test_dataset, build_training_splits

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
EXPECTED = {"drive": (16, 4, 20), "hrf": (31, 14, 14), "fives": (600, 200, 200), "vessmap": (70, 30, 30)}
FIVES_CORRUPTED = (106, 5)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    add_path_arguments(parser)
    args = parser.parse_args()
    data_root, weights_dir, _ = resolve_path_arguments(args)
    ok = True
    for ds in args.datasets:
        path = CONFIGS / "cnn" / ds / "lwnet.yaml"
        params = resolve_paths(load_config(path), path, data_root, weights_dir)["dataset_params"]
        try:
            splits = build_training_splits(ds, params)
            test = build_test_dataset(ds, params)
        except (FileNotFoundError, ValueError) as exc:
            print(f"{ds}: MISSING ({exc})")
            ok = False
            continue
        sizes = (len(splits["train_ds"]), len(splits["val_ds"]), len(test))
        good = sizes == EXPECTED[ds]
        line = f"{ds}: train/val/test = {sizes[0]}/{sizes[1]}/{sizes[2]} (expected {'/'.join(map(str, EXPECTED[ds]))})"
        if ds == "fives":
            corrupted = (len(splits["train_ds"].corrupted_bases), len(test.corrupted_bases))
            good &= corrupted == FIVES_CORRUPTED
            line += f", corrupted masks train/test = {corrupted[0]}/{corrupted[1]} (expected 106/5)"
        print(("OK   " if good else "FAIL ") + line)
        ok &= good
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
