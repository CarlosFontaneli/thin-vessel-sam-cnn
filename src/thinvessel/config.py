"""Configuration files, studies and path resolution (Section 3.3, Appendix A).

A configuration file (``configs/<regime>/<dataset>/<model>.yaml``) holds the settings of the
selected trial of one hyperparameter search, its ``search_space`` and the number of the selected
trial. ``configs/README.md`` describes the schema.

Paths in a configuration are relative. They are resolved in one place, :func:`resolve_paths`:

- ``dataset_params.*_dir`` against the data root (``--data-root``, else ``$DATA_ROOT``, else ``./data``);
- ``dataset_params.*_csv`` against the directory of the configuration file;
- ``model_params.pretrained_weights_path`` against the weights directory (``--weights-dir``, else
  ``$WEIGHTS_DIR``, else ``./weights``).

Runtime outputs (searches, trained models, predictions, metrics) go under the outputs directory
(``--outputs-dir``, else ``$OUTPUTS_DIR``, else ``./outputs``).
"""

import ast
import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

PathLike = Union[str, Path]

DATASETS = ("drive", "hrf", "fives", "vessmap")
MODELS = ("lwnet", "convunext", "convnext", "litemedsam", "medsam")

# Regime of a configuration = name of its folder under configs/.
REGIMES = ("cnn", "finetuned", "scratch", "compact")

# Configuration id used in the per-image tables, the figures and the statistics, per (regime, model).
_CONFIGURATION_IDS: Dict[tuple, str] = {
    ("cnn", "lwnet"): "lwnet_scratch",
    ("cnn", "convunext"): "convunext_scratch",
    ("cnn", "convnext"): "convnext_scratch",
    ("finetuned", "litemedsam"): "litemedsam_pretrained",
    ("finetuned", "medsam"): "medsam_pretrained",
    ("scratch", "litemedsam"): "litemedsam_scratch",
    ("scratch", "medsam"): "medsam_scratch",
    ("compact", "litemedsam"): "litemedsam_compact_scratch",
    ("compact", "medsam"): "medsam_compact_scratch",
}

# Keys of a search space that train.py routed into the architecture of the transformer-based models.
ARCHITECTURE_KEYS = (
    "embed_dim",
    "depth",
    "num_heads",
    "window_size",
    "patch_size",
    "embed_dims",
    "depths",
    "mlp_ratio",
    "window_sizes",
)

REQUIRED_KEYS = ("seed", "model_name", "dataset_name", "model_params", "dataset_params", "training_params")


@dataclass(frozen=True)
class Study:
    """One configuration on one dataset: one hyperparameter search and its selected run.

    Attributes:
        regime: ``cnn``, ``finetuned``, ``scratch`` or ``compact``.
        dataset: ``drive``, ``hrf``, ``fives`` or ``vessmap``.
        model: ``lwnet``, ``convunext``, ``convnext``, ``litemedsam`` or ``medsam``.
    """

    regime: str
    dataset: str
    model: str

    def __post_init__(self) -> None:
        if (self.regime, self.model) not in _CONFIGURATION_IDS or self.dataset not in DATASETS:
            raise ValueError(f"Unknown study {self.regime}/{self.dataset}/{self.model}.")

    @classmethod
    def from_config_path(cls, path: PathLike) -> "Study":
        """Reads the study from a configuration path ``.../<regime>/<dataset>/<model>.yaml``."""
        p = Path(path)
        return cls(regime=p.parent.parent.name, dataset=p.parent.name, model=p.stem)

    @property
    def configuration(self) -> str:
        """Configuration id of the per-image tables (e.g. ``medsam_pretrained``)."""
        return _CONFIGURATION_IDS[(self.regime, self.model)]

    @property
    def relpath(self) -> str:
        """``<regime>/<dataset>/<model>``, the layout of the configuration and results folders."""
        return f"{self.regime}/{self.dataset}/{self.model}"


def all_studies() -> list:
    """The 36 studies of the paper, ordered by regime, dataset and model."""
    return [Study(regime, dataset, model) for (regime, model) in _CONFIGURATION_IDS for dataset in DATASETS]


def resolve_dir(cli_value: Optional[PathLike], env_var: str, default: str) -> Path:
    """Directory from a command-line value, else an environment variable, else a default.

    Args:
        cli_value: Value given on the command line, or ``None``.
        env_var: Name of the environment variable (``DATA_ROOT``, ``WEIGHTS_DIR`` or ``OUTPUTS_DIR``).
        default: Default, relative to the current directory.

    Returns:
        The absolute directory path.
    """
    raw = cli_value if cli_value is not None else os.environ.get(env_var, default)
    return Path(raw).expanduser().resolve()


def add_path_arguments(parser: Any) -> None:
    """Adds ``--data-root``, ``--weights-dir`` and ``--outputs-dir`` to an ``argparse`` parser."""
    parser.add_argument("--data-root", default=None, help="Datasets (default $DATA_ROOT, else ./data).")
    parser.add_argument(
        "--weights-dir", default=None, help="MedSAM/LiteMedSAM checkpoints (default $WEIGHTS_DIR, else ./weights)."
    )
    parser.add_argument("--outputs-dir", default=None, help="Runtime outputs (default $OUTPUTS_DIR, else ./outputs).")


def resolve_path_arguments(args: Any) -> tuple:
    """``(data_root, weights_dir, outputs_dir)`` from the arguments of :func:`add_path_arguments`."""
    return (
        resolve_dir(args.data_root, "DATA_ROOT", "data"),
        resolve_dir(args.weights_dir, "WEIGHTS_DIR", "weights"),
        resolve_dir(args.outputs_dir, "OUTPUTS_DIR", "outputs"),
    )


def load_config(path: PathLike) -> Dict[str, Any]:
    """Reads a configuration file.

    Args:
        path: Path to a ``.yaml`` configuration.

    Returns:
        The configuration dictionary, paths unresolved.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If a required top-level key is missing.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    missing = [k for k in REQUIRED_KEYS if k not in config]
    if missing:
        raise KeyError(f"{path} lacks the keys {missing}.")
    return config


def resolve_paths(config: Dict[str, Any], config_path: PathLike, data_root: Path, weights_dir: Path) -> Dict[str, Any]:
    """Returns a copy of ``config`` whose relative data, split and weight paths are absolute.

    Args:
        config: Configuration from :func:`load_config` (or a checkpoint's hyperparameters).
        config_path: Path of the configuration file (split lists are relative to its folder).
        data_root: Root of the datasets.
        weights_dir: Directory of the published MedSAM and LiteMedSAM checkpoints.

    Returns:
        The configuration with absolute paths.
    """
    out = copy.deepcopy(config)
    base = Path(config_path).resolve().parent
    params = out["dataset_params"]
    for key, value in params.items():
        if isinstance(value, str) and key.endswith("_dir"):
            params[key] = str(data_root / value)
        elif isinstance(value, str) and key.endswith("_csv"):
            params[key] = str((base / value).resolve())
    weights = out["model_params"].get("pretrained_weights_path")
    if weights is not None:
        out["model_params"]["pretrained_weights_path"] = str(weights_dir / weights)
    return out


def sample_params(trial: Any, search_space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Samples one value per entry of a search space with an Optuna trial, in the file's order.

    The order of the ``suggest_*`` calls consumes the TPE sampler's random numbers, so it must not
    change: it follows the order of ``search_space`` in the configuration file.

    Args:
        trial: ``optuna.Trial``.
        search_space: ``{name: {type: float|int|categorical, low, high, log, choices}}``.

    Returns:
        The sampled values by name.
    """
    sampled: Dict[str, Any] = {}
    for name, spec in search_space.items():
        kind = spec["type"]
        if kind == "float":
            sampled[name] = trial.suggest_float(
                name, float(spec["low"]), float(spec["high"]), log=spec.get("log", False)
            )
        elif kind == "int":
            sampled[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
        elif kind == "categorical":
            sampled[name] = trial.suggest_categorical(name, spec["choices"])
    return sampled


def apply_params(config: Dict[str, Any], sampled: Dict[str, Any]) -> Dict[str, Any]:
    """Builds the configuration a trial trains with from the sampled hyperparameters.

    This is the mapping of the experiments' ``train.py``: learning rate and weight decay go to
    ``training_params``, the batch size to ``dataset_params``, the architecture keys of the compact
    models into ``medsam_params.vit_b_params`` or ``litemedsam_params.tiny_vit_params`` (a MedSAM
    patch size also sets the prompt encoder's embedding grid to ``img_size // patch_size``), and any
    other key that exists in ``model_params`` (or ``layers``) to ``model_params``. List-valued
    categorical choices are stored as strings such as ``"[8, 16, 32]"``; the architecture keys are
    parsed here, while ``layers`` is kept as a string, as the CNN searches recorded it (LW-Net
    parses it).

    ``optimizer`` and ``patience`` are sampled by every search but match none of these rules, so
    they never reach the training run: every trial used AdamW and the patience of the
    configuration. The searches kept sampling them because removing them would change the sequence
    of the TPE sampler (see README, "Known deviations").

    Args:
        config: Configuration with ``model_params``, ``dataset_params`` and ``training_params``.
        sampled: Values from :func:`sample_params`.

    Returns:
        A new configuration (the input is not modified) without ``search_space``.
    """
    out = {k: copy.deepcopy(v) for k, v in config.items() if k != "search_space"}
    model_params, dataset_params, training_params = (out["model_params"], out["dataset_params"], out["training_params"])
    for key, value in sampled.items():
        if key in ("lr", "weight_decay"):
            training_params[key] = value
        elif key == "batch_size":
            dataset_params[key] = value
        elif key in ARCHITECTURE_KEYS:
            parsed = _parse_list(value)
            if "medsam_params" in model_params and "vit_b_params" in model_params["medsam_params"]:
                vit = model_params["medsam_params"]["vit_b_params"]
                vit[key] = parsed
                if key == "patch_size":
                    grid = vit.get("img_size", 256) // parsed
                    if "prompt_encoder_params" in model_params["medsam_params"]:
                        model_params["medsam_params"]["prompt_encoder_params"]["image_embedding_size"] = [grid, grid]
            elif "litemedsam_params" in model_params and "tiny_vit_params" in model_params["litemedsam_params"]:
                model_params["litemedsam_params"]["tiny_vit_params"][key] = parsed
        elif key in model_params or key == "layers":
            # Stored as sampled, as in the CNN searches: LW-Net parses a "[8, 16, 32]" string itself.
            model_params[key] = value
    return out


def _parse_list(value: Any) -> Any:
    """Parses a categorical choice stored as a list literal (``"[8, 16, 32]"``); other values unchanged."""
    if isinstance(value, str) and value.startswith("["):
        return ast.literal_eval(value)
    return value
