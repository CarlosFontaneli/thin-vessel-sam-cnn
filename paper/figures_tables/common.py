"""Shared paths, model order, styles and data loading of the paper's figures and tables.

Every figure and table script imports this module.

- Inputs: the committed tables in ``results/`` (override with ``RESULTS_DIR``): ``per_image_metrics.csv``
  (``scripts/evaluate.py``), ``cost.csv``, ``convergence.csv``, ``search_trials.csv`` (``scripts/analyse.py``); the
  tests ``outputs/statistics/all_tests.csv`` (``paper/statistics/paper_wilcoxon.py``; override with
  ``SIGNIFICANCE_CSV``). Fig. 3 (``fig07_thin_errors.py``) also reads the saved test masks under
  ``$OUTPUTS_DIR/predictions``.
- Outputs: the PDFs and LaTeX tables that the paper includes, in ``paper/manuscript/figures/`` (override with
  ``FIGURES_DIR``); PNG previews and the exact plotted values ``plotted/<stem>.csv`` in ``outputs/figures/`` (the
  outputs folder is ``$OUTPUTS_DIR``, default ``outputs/``).
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from thinvessel.config import Study, all_studies  # noqa: E402
from thinvessel.evaluation.scoring import paper_protocol  # noqa: E402

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", REPO_ROOT / "results"))
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", REPO_ROOT / "outputs"))
FIGURES_DIR = Path(os.environ.get("FIGURES_DIR", REPO_ROOT / "paper" / "manuscript" / "figures"))
PREVIEW_DIR = OUTPUTS_DIR / "figures"
PLOTTED_DIR = PREVIEW_DIR / "plotted"

PER_IMAGE_CSV = RESULTS_DIR / "per_image_metrics.csv"
COST_CSV = RESULTS_DIR / "cost.csv"
CONVERGENCE_CSV = RESULTS_DIR / "convergence.csv"
SEARCH_CSV = RESULTS_DIR / "search_trials.csv"
SIGNIFICANCE_CSV = Path(os.environ.get("SIGNIFICANCE_CSV", OUTPUTS_DIR / "statistics" / "all_tests.csv"))

COLUMN_WIDTH_IN = 3.4
PAGE_WIDTH_IN = 7.0

DATASETS: List[str] = ["drive", "hrf", "fives", "vessmap"]
DATASET_LABELS: Dict[str, str] = {"drive": "DRIVE", "hrf": "HRF", "fives": "FIVES", "vessmap": "VessMap"}
N_TEST_IMAGES: Dict[str, int] = {"drive": 20, "hrf": 14, "fives": 200, "vessmap": 30}

# One model order everywhere: CNNs from smallest to largest, the hybrid, the plain ViT.
ARCH_ORDER: List[str] = ["lwnet", "convunext", "convnext", "litemedsam", "medsam"]
REGIME_LABELS: Dict[str, str] = {"finetuned": "fine-tuned", "scratch": "from scratch", "compact": "compact"}

# Colour, marker and line of each architecture (validated categorical palette).
CANONICAL_MODEL_STYLES: Dict[str, Dict[str, str]] = {
    "convnext": {"name": "ConvNeXt", "color": "#1d5b92", "marker": "o", "linestyle": "-"},
    "convunext": {"name": "ConvUNeXt", "color": "#45a075", "marker": "s", "linestyle": "-"},
    "lwnet": {"name": "LW-Net", "color": "#a06aec", "marker": "D", "linestyle": "-"},
    "litemedsam": {"name": "LiteMedSAM", "color": "#e65719", "marker": "^", "linestyle": "--"},
    "medsam": {"name": "MedSAM", "color": "#9d0367", "marker": "v", "linestyle": "--"},
}

# Configurations: (architecture, regime). CNNs are trained from scratch only and carry the regime "cnn"
# (always drawn filled, like the fine-tuned ViTs of the comparison basis).
CONFIGS: Dict[str, Dict[str, str]] = {
    "lwnet_scratch": {"arch": "lwnet", "regime": "cnn"},
    "convunext_scratch": {"arch": "convunext", "regime": "cnn"},
    "convnext_scratch": {"arch": "convnext", "regime": "cnn"},
    "litemedsam_pretrained": {"arch": "litemedsam", "regime": "finetuned"},
    "litemedsam_scratch": {"arch": "litemedsam", "regime": "scratch"},
    "litemedsam_compact_scratch": {"arch": "litemedsam", "regime": "compact"},
    "medsam_pretrained": {"arch": "medsam", "regime": "finetuned"},
    "medsam_scratch": {"arch": "medsam", "regime": "scratch"},
    "medsam_compact_scratch": {"arch": "medsam", "regime": "compact"},
}
COMPARISON_CONFIGS: List[str] = [
    "lwnet_scratch",
    "convunext_scratch",
    "convnext_scratch",
    "litemedsam_pretrained",
    "medsam_pretrained",
]
RADIUS_BIN_COLUMNS: List[str] = [
    "thin_vessel_recall",
    "centerline_recall_radius_1.5_2.5",
    "centerline_recall_radius_2.5_3.5",
    "centerline_recall_radius_3.5_4.5",
    "centerline_recall_radius_4.5_inf",
]
RADIUS_BIN_LABELS: List[str] = ["[0.5, 1.5)", "[1.5, 2.5)", "[2.5, 3.5)", "[3.5, 4.5)", "[4.5, ∞)"]


def study_of(config: str, ds: str) -> Study:
    """The study (regime, dataset, model) of a configuration id on a dataset."""
    spec = CONFIGS[config]
    return Study(spec["regime"], ds, spec["arch"])


def arch_color(arch: str) -> str:
    return CANONICAL_MODEL_STYLES[arch]["color"]


def arch_marker(arch: str) -> str:
    return CANONICAL_MODEL_STYLES[arch]["marker"]


def arch_name(arch: str) -> str:
    return CANONICAL_MODEL_STYLES[arch]["name"]


def config_label(config: str) -> str:
    """Display name: the architecture, plus the regime for ViTs (e.g. 'MedSAM, compact')."""
    spec = CONFIGS[config]
    name = arch_name(spec["arch"])
    return name if spec["regime"] == "cnn" else f"{name}, {REGIME_LABELS[spec['regime']]}"


def marker_kwargs(config: str, size: float = 5.0) -> Dict[str, Any]:
    """Marker style of a configuration: colour = architecture, fill = regime."""
    spec = CONFIGS[config]
    color = arch_color(spec["arch"])
    kw: Dict[str, Any] = {
        "marker": arch_marker(spec["arch"]),
        "markersize": size,
        "color": color,
        "markeredgecolor": color,
        "markeredgewidth": 1.0,
    }
    if spec["regime"] in ("cnn", "finetuned"):
        kw.update(markerfacecolor=color, markeredgecolor="#222222", markeredgewidth=0.5)
    elif spec["regime"] == "scratch":
        kw.update(markerfacecolor="white")
    else:  # compact: left half filled; thicker edge with round joins so it reads apart from the hollow marker
        from matplotlib.markers import MarkerStyle

        kw.update(
            marker=MarkerStyle(arch_marker(spec["arch"]), fillstyle="left", joinstyle="round"),
            markerfacecoloralt="white",
            markerfacecolor=color,
            markeredgewidth=1.3,
        )
    return kw


def bar_kwargs(config: str) -> Dict[str, Any]:
    """Bar style of a configuration: solid for CNNs/fine-tuned, hatched white for scratch/compact."""
    spec = CONFIGS[config]
    color = arch_color(spec["arch"])
    if spec["regime"] in ("cnn", "finetuned"):
        return {"facecolor": color, "edgecolor": "#222222", "linewidth": 0.4}
    hatch = "////" if spec["regime"] == "scratch" else "...."
    return {"facecolor": "white", "edgecolor": color, "hatch": hatch, "linewidth": 0.8}


def paper_style() -> None:
    """Publication style at the paper's font size (8 pt base, ticks 7 pt), vector fonts, 300 dpi."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.titleweight": "bold",
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.title_fontsize": 7,
            "figure.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#888888",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#E5E5E5",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "standard",
            "savefig.pad_inches": 0.05,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "hatch.linewidth": 0.6,
            "lines.linewidth": 1.2,
        }
    )


def percent_axis(axis: Any) -> None:
    """Tick labels in % for a metric plotted as a fraction."""
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.0f}"))


def save(fig: plt.Figure, stem: str) -> List[Path]:
    """Writes ``<FIGURES_DIR>/<stem>.pdf`` (vector, for the paper) and ``<PREVIEW_DIR>/<stem>.png`` (300 dpi)."""
    paths = []
    for folder, ext in ((FIGURES_DIR, "pdf"), (PREVIEW_DIR, "png")):
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{stem}.{ext}"
        # no creation date in the PDF, so that a rerun reproduces the committed file byte for byte
        fig.savefig(path, dpi=300, metadata={"CreationDate": None} if ext == "pdf" else None)
        paths.append(path)
    plt.close(fig)
    return paths


def save_values(df: pd.DataFrame, stem: str) -> Path:
    """Writes the exact values a figure plots or a table prints to ``<PLOTTED_DIR>/<stem>.csv``."""
    path = PLOTTED_DIR / f"{stem}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_tex(stem: str, tex: str) -> Path:
    """Writes a LaTeX table to ``<FIGURES_DIR>/<stem>.tex`` (included by the paper)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{stem}.tex"
    path.write_text(tex, encoding="utf-8")
    return path


def read_table(path: Path, exact: bool = False) -> pd.DataFrame:
    """Reads a CSV of ``results/``.

    pandas' default float parser can be one ulp off the written value; ``exact=True`` parses exactly. Each loader
    parses its table the way the paper's scripts read the same values: the per-image metrics, the tests and the
    convergence summary from CSV (default parser), the cost (JSON), the search records (SQLite) and the per-image
    values of Table IV (parquet) exactly. This keeps every printed and plotted value bit-identical to the paper.
    """
    return pd.read_csv(path, float_precision="round_trip" if exact else None, low_memory=False)


def load_per_image(include_padded: bool = False, exact: bool = False) -> pd.DataFrame:
    """Per-image metrics of the nine configurations under the paper's protocol.

    MedSAM (fine-tuned and from scratch) on DRIVE and VessMap uses the upscaled protocol; every other
    configuration the native sliding window. ``include_padded`` also keeps the native (padded) rows
    of those four MedSAM runs (Table IV only); ``exact`` parses the floats exactly (see :func:`read_table`).

    Returns:
        One row per (dataset, configuration, protocol, sample_id) with the metric columns.
    """
    df = read_table(PER_IMAGE_CSV, exact)
    paper = {(s.dataset, s.configuration): paper_protocol(s) for s in all_studies()}
    keep = [
        paper[(d, c)] == p or (include_padded and p == "native")
        for d, c, p in zip(df["dataset"], df["configuration"], df["protocol"])
    ]
    out = df[keep].reset_index(drop=True)
    counts = (
        out[out["protocol"] == out.apply(lambda r: paper[(r["dataset"], r["configuration"])], axis=1)]
        .groupby(["dataset", "configuration"])
        .size()
    )
    for ds in DATASETS:
        for cfg in CONFIGS:
            if counts.get((ds, cfg), 0) != N_TEST_IMAGES[ds]:
                raise ValueError(f"{cfg} on {ds}: {counts.get((ds, cfg), 0)} images, expected {N_TEST_IMAGES[ds]}.")
    return out


def means(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean over test images per configuration (rows) and dataset (columns); NaN cells skipped."""
    return df.groupby(["configuration", "dataset"])[metric].mean().unstack("dataset")


def sds(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Standard deviation over test images (ddof=1) per configuration (rows) and dataset (columns)."""
    return df.groupby(["configuration", "dataset"])[metric].std().unstack("dataset")


def pm(mean_pct: float, sd_pct: float, bold: Any = False, tight: bool = False) -> str:
    """LaTeX cell 'mean ± SD' (both in %, one decimal); ``bold`` "tie" underlines, any other true value sets bold."""
    cell = rf"${mean_pct:.1f}{{\pm}}{sd_pct:.1f}$" if tight else rf"${mean_pct:.1f} \pm {sd_pct:.1f}$"
    if bold == "tie":
        return rf"\underline{{{cell}}}"
    return rf"{{\boldmath{cell}}}" if bold else cell


# Statistical tests: two-sided Wilcoxon signed-rank on per-image differences, Holm correction inside each family,
# alpha = 0.05 (paper/statistics/paper_wilcoxon.py). The scripts read its output and check that its means are theirs.
ALPHA = 0.05
TEST_NAMES: Dict[str, str] = {  # configuration -> name in all_tests.csv
    "lwnet_scratch": "LW-Net",
    "convunext_scratch": "ConvUNeXt",
    "convnext_scratch": "ConvNeXt",
    "litemedsam_pretrained": "LiteMedSAM FT",
    "litemedsam_scratch": "LiteMedSAM scratch",
    "litemedsam_compact_scratch": "compact LiteMedSAM",
    "medsam_pretrained": "MedSAM FT",
    "medsam_scratch": "MedSAM scratch",
    "medsam_compact_scratch": "compact MedSAM",
}
MARKS = r"Bold: best; underlined: not significantly different from the best"
STAR = r"$^{*}$: significant"
MARKS_PAIR = r"Bold: best"  # two values and a starred Delta: no underline


def ds_label(ds: str, rows: int = 0) -> str:
    """Dataset cell with the number of test images below the name; wrapped in a multirow when rows > 0."""
    lab = rf"\shortstack{{{DATASET_LABELS[ds]}\\($n={N_TEST_IMAGES[ds]}$)}}"
    return rf"\multirow{{{rows}}}{{*}}{{{lab}}}" if rows else lab


def bold_group(tests: pd.DataFrame, dataset: str, metric: str, values: Dict[str, Any]) -> Dict[str, Any]:
    """Marks for values compared on one dataset and metric: "best" (bold) for the highest mean, "tie" (underlined)
    for every value whose test against it is not significant, False otherwise. ``values`` maps a key to
    (configuration or test name, mean as a fraction)."""
    top = max(values, key=lambda k: values[k][1])
    out = {}
    for k, (name, _) in values.items():
        if k == top:
            out[k] = "best"
        else:
            out[k] = False if test_result(tests, dataset, metric, name, values[top][0])["significant"] else "tie"
    return out


def load_tests() -> pd.DataFrame:
    return read_table(SIGNIFICANCE_CSV)


def test_result(
    tests: pd.DataFrame, dataset: str, metric: str, a: str, b: str, mean_a: float = None, mean_b: float = None
) -> Dict[str, Any]:
    """The test of configuration ``a`` against ``b`` (configuration ids or names of all_tests.csv).

    Returns ``significant``, ``p_holm``, ``a_higher`` (sign of the Hodges-Lehmann estimate of a - b) and ``n``. Given
    means (fractions) must equal those of the test file, unless the test paired fewer images than the dataset has.
    """
    na, nb = TEST_NAMES.get(a, a), TEST_NAMES.get(b, b)
    sel = tests[(tests["dataset"] == dataset) & (tests["metric"] == metric)]
    row = sel[(sel["A"] == na) & (sel["B"] == nb)]
    flip = row.empty
    if flip:
        row = sel[(sel["A"] == nb) & (sel["B"] == na)]
    if len(row) != 1:
        raise ValueError(f"No unique test for {na} vs {nb} on {dataset}, {metric}.")
    r = row.iloc[0]
    ma, mb = (r["mean_b"], r["mean_a"]) if flip else (r["mean_a"], r["mean_b"])
    complete = int(r["n"]) == N_TEST_IMAGES[dataset]
    for given, stored in ((mean_a, ma), (mean_b, mb)) if complete else ():
        if given is not None and not np.isclose(given, stored, atol=1e-9):
            raise ValueError(
                f"{na} vs {nb} on {dataset}, {metric}: mean {given} differs from the test file ({stored})."
            )
    hl = -r["hodges_lehmann"] if flip else r["hodges_lehmann"]
    return {
        "significant": bool(r["p_holm"] < ALPHA),
        "p_holm": float(r["p_holm"]),
        "a_higher": bool(hl > 0),
        "n": int(r["n"]),
    }


def fmt_p(p: float) -> str:
    """p-value as printed: '$<$0.001', three decimals below 0.1, two otherwise."""
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def signed_delta(delta: float, significant: bool) -> str:
    """Difference in points with a sign and, when significant, an asterisk."""
    star = "^{*}" if significant else ""
    return f"${delta:+.1f}{star}$"


def load_complexity() -> pd.DataFrame:
    """Parameters and GFLOPs of every configuration and dataset (``cost.csv``)."""
    df = read_table(COST_CSV, exact=True)
    return df[["configuration", "dataset", "params_m", "gflops", "windows_per_image", "gflops_per_image", "roi_size"]]


def load_config_file(config: str, ds: str) -> Dict[str, Any]:
    """Released configuration of a study: the settings of its selected trial and its search space."""
    s = study_of(config, ds)
    with open(CONFIGS_DIR / s.regime / s.dataset / f"{s.model}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_trials(config: str, ds: str) -> pd.DataFrame:
    """Trials of one search (``search_trials.csv``): number, state, best validation Dice, hours, sampled parameters,
    the applied settings (``applied_*``) and ``out_of_memory``. Columns of parameters the search did not sample are
    dropped. A search without a study database has state ``LOGGED`` and no validation Dice."""
    df = read_table(SEARCH_CSV, exact=True)
    sub = df[(df["configuration"] == config) & (df["dataset"] == ds)].drop(columns=["configuration", "dataset"])
    keep = [c for c in sub.columns if sub[c].notna().any() or c in ("best_val_dice", "hours")]
    sub = sub[keep].sort_values("number").reset_index(drop=True)
    for c in sub.columns:  # a column shared by several searches can mix numbers and list strings across them
        if not pd.api.types.is_numeric_dtype(sub[c]) and not pd.api.types.is_bool_dtype(sub[c]):
            try:
                sub[c] = pd.to_numeric(sub[c])
            except (TypeError, ValueError):
                pass
    return sub


def search_space(config: str, ds: str) -> Dict[str, Dict[str, Any]]:
    """Distribution of every sampled parameter (from the released configuration), in Optuna's json layout."""
    space = {}
    for name, s in load_config_file(config, ds)["search_space"].items():
        if s["type"] == "categorical":
            space[name] = {"name": "CategoricalDistribution", "attributes": {"choices": s["choices"]}}
        elif s["type"] == "int":
            space[name] = {"name": "IntDistribution", "attributes": {"low": s["low"], "high": s["high"], "log": False}}
        else:
            space[name] = {
                "name": "FloatDistribution",
                "attributes": {"low": float(s["low"]), "high": float(s["high"]), "log": s.get("log", False)},
            }
    return space


def selected_trial(config: str, ds: str) -> int:
    """Number of the selected trial (highest validation Dice) of a search; checked against the configuration."""
    s = study_of(config, ds)
    conv = read_table(CONVERGENCE_CSV)
    row = conv[(conv["regime"] == s.regime) & (conv["dataset"] == ds) & (conv["model"] == s.model)]
    if len(row) != 1:
        raise ValueError(f"No unique selected trial for {config} on {ds}.")
    number = int(str(row["selected_trial"].iloc[0]).replace("trial_", ""))
    if number != int(load_config_file(config, ds)["selected_trial"]):
        raise ValueError(f"{config} on {ds}: convergence.csv selects trial {number}, the configuration another.")
    return number


def convergence_row(config: str, ds: str) -> pd.Series:
    """Convergence of the selected run of a search (``convergence.csv``)."""
    s = study_of(config, ds)
    conv = read_table(CONVERGENCE_CSV)
    return conv[(conv["regime"] == s.regime) & (conv["dataset"] == ds) & (conv["model"] == s.model)].iloc[0]


def legend_handles(configs: List[str], size: float = 5.0, line: bool = False) -> List[Any]:
    """Legend entries for configurations, with the same marker style as the plotted points."""
    from matplotlib.lines import Line2D

    handles = []
    for cfg in configs:
        kw = marker_kwargs(cfg, size)
        ls = CANONICAL_MODEL_STYLES[CONFIGS[cfg]["arch"]]["linestyle"] if line else "none"
        handles.append(Line2D([0], [0], linestyle=ls, label=config_label(cfg), **kw))
    return handles


__all__ = [name for name in dir() if not name.startswith("_")] + ["json", "np", "pd", "plt"]
