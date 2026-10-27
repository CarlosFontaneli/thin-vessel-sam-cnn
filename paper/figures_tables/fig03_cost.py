"""Fig. 4 (fig:cost): parameters and GFLOPs per test image of every architecture and dataset (log scale).

Fine-tuned and from-scratch runs of a transformer-based model share the architecture and the inference window, so they
have one bar (the script checks that their measured cost is identical), labelled with the architecture name only
; the caption says that the bar stands for both runs.
Output: ``paper/manuscript/figures/fig03_cost.{pdf,png}``.
"""

from matplotlib.patches import Patch

from common import (
    DATASET_LABELS,
    DATASETS,
    PAGE_WIDTH_IN,
    bar_kwargs,
    config_label,
    load_complexity,
    np,
    paper_style,
    plt,
    save,
    save_values,
)

BARS = [
    "lwnet_scratch",
    "convunext_scratch",
    "convnext_scratch",
    "litemedsam_pretrained",
    "litemedsam_compact_scratch",
    "medsam_pretrained",
    "medsam_compact_scratch",
]
SHARED = {"litemedsam_pretrained": "litemedsam_scratch", "medsam_pretrained": "medsam_scratch"}


def label(cfg: str) -> str:
    if cfg in SHARED:
        return config_label(cfg).replace(", fine-tuned", "")
    return config_label(cfg)


def fmt(v: float) -> str:
    return f"{v:.2f}" if v < 0.1 else f"{v:.1f}"


def main() -> None:
    paper_style()
    cx = load_complexity().set_index(["configuration", "dataset"])
    for ft, sc in SHARED.items():
        for ds in DATASETS:
            a, b = cx.loc[(ft, ds)], cx.loc[(sc, ds)]
            if not np.isclose(a["params_m"], b["params_m"]) or not np.isclose(
                a["gflops_per_image"], b["gflops_per_image"]
            ):
                raise ValueError(f"{ft} and {sc} differ in cost on {ds}")
    plotted = cx.loc[BARS].reset_index()
    save_values(
        plotted[["configuration", "dataset", "params_m", "gflops", "windows_per_image", "gflops_per_image"]],
        "fig03_cost",
    )

    fig, axes = plt.subplots(2, 1, figsize=(PAGE_WIDTH_IN, 4.6), sharex=True)
    width = 0.84 / len(BARS)
    x0 = np.arange(len(DATASETS))
    panels = [("params_m", "Parameters (M)", (0.03, 3e3)), ("gflops_per_image", "GFLOPs per image", (0.5, 2e5))]
    for i, (ax, (col, ylabel, ylim)) in enumerate(zip(axes, panels)):
        for j, cfg in enumerate(BARS):
            xs = x0 - 0.42 + width * (j + 0.5)
            vals = [cx.loc[(cfg, ds), col] for ds in DATASETS]
            ax.bar(xs, vals, width=width * 0.92, **bar_kwargs(cfg))
            for xi, v in zip(xs, vals):
                ax.text(xi, v * 1.25, fmt(v), ha="center", va="bottom", rotation=90, fontsize=7)
        ax.set_yscale("log")
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
        ax.set_title(f"({'ab'[i]})", loc="left")
    axes[1].set_xticks(x0)
    axes[1].set_xticklabels([DATASET_LABELS[d] for d in DATASETS], fontsize=8)
    handles = [Patch(label=label(cfg), **bar_kwargs(cfg)) for cfg in BARS]
    # Column-major legend: CNNs | LiteMedSAM | MedSAM (an empty slot keeps the columns aligned).
    handles.insert(5, Patch(facecolor="none", edgecolor="none", label=""))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.89), h_pad=0.6)
    save(fig, "fig03_cost")


if __name__ == "__main__":
    main()
