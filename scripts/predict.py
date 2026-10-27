"""Writes the test-set predictions of a trained model (Section 3.4).

    python scripts/predict.py --config configs/finetuned/drive/medsam.yaml --protocol upscaled

``--protocol native`` (default): sliding-window inference at native resolution (images smaller
than the window are padded). ``--protocol upscaled``: images smaller than the training input size
are upscaled to it and the logits resized back (the protocol of every MedSAM result on DRIVE and
VessMap). Without ``--checkpoint``, the best checkpoint of ``<outputs>/train/<study>/`` is used,
else that of ``<outputs>/search/<study>/``.

Output: ``<outputs>/predictions/<regime>/<dataset>/<model>/<protocol>/{predicted_masks,ground_truth}/``.
"""

import argparse
import logging
from pathlib import Path

import torch

from thinvessel.config import Study, add_path_arguments, load_config, resolve_path_arguments, resolve_paths
from thinvessel.data import build_test_dataset
from thinvessel.evaluation.inference import PROTOCOLS, load_trained_module, predict_dataset

logger = logging.getLogger("predict")


def find_checkpoint(outputs_dir: Path, study: Study) -> Path:
    """Best checkpoint of a study: from ``train/``, else from ``search/`` (highest Dice in the file name)."""
    for stage in ("train", "search"):
        found = sorted((outputs_dir / stage / study.relpath).glob("best_model_val_dice_*.ckpt"))
        if found:
            return max(found, key=lambda p: float(p.stem.rsplit("_", 1)[1]))
    raise FileNotFoundError(f"No checkpoint for {study.relpath} under {outputs_dir}/{{train,search}}.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", required=True, type=Path, help="Configuration file under configs/.")
    parser.add_argument("--checkpoint", "-k", type=Path, default=None, help="Checkpoint (.ckpt) to evaluate.")
    parser.add_argument("--protocol", choices=PROTOCOLS, default="native", help="Inference protocol.")
    add_path_arguments(parser)
    args = parser.parse_args()
    data_root, weights_dir, outputs_dir = resolve_path_arguments(args)
    study = Study.from_config_path(args.config)
    config = resolve_paths(load_config(args.config), args.config, data_root, weights_dir)
    checkpoint = args.checkpoint or find_checkpoint(outputs_dir, study)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Checkpoint %s on %s, protocol %s", checkpoint, device, args.protocol)

    module, _ = load_trained_module(checkpoint, device)
    dataset = build_test_dataset(config["dataset_name"], config["dataset_params"])
    size = config["dataset_params"]["image_size"]
    train_hw = ((size, size) if isinstance(size, int) else tuple(size)) if args.protocol == "upscaled" else None
    out = outputs_dir / "predictions" / study.relpath / args.protocol
    n = predict_dataset(module, dataset, out, device, train_hw)
    logger.info("%d test images written to %s", n, out)


if __name__ == "__main__":
    main()
