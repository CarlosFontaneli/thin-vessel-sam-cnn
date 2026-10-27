"""Fig. 7 (fig:convergence): epochs to 95% of the best validation Dice of each selected run, nine configurations.

"≤" marks runs that had already reached the target at their first validation (censored value).
Output: ``paper/manuscript/figures/fig06_convergence.{pdf,png}``.
"""

from matplotlib.patches import Patch

from common import (
    CONVERGENCE_CSV,
    DATASET_LABELS,
    DATASETS,
    PAGE_WIDTH_IN,
    bar_kwargs,
    config_label,
    np,
    paper_style,
    plt,
    read_table,
    save,
    save_values,
)

BARS = [
    "lwnet_scratch",
    "convunext_scratch",
    "convnext_scratch",
    "litemedsam_pretrained",
    "litemedsam_scratch",
    "litemedsam_compact_scratch",
    "medsam_pretrained",
    "medsam_scratch",
    "medsam_compact_scratch",
]


def main() -> None:
    paper_style()
    conv = read_table(CONVERGENCE_CSV).set_index(["configuration", "dataset"])
    if conv.index.duplicated().any():
        raise ValueError("Duplicate (configuration, dataset) rows in the convergence summary.")
    plotted = conv.loc[
        [(c, d) for c in BARS for d in DATASETS],
        ["selected_epochs_to_target", "selected_target_censored", "selected_val_interval", "selected_best_epoch"],
    ].reset_index()
    save_values(plotted, "fig06_convergence")

    fig, ax = plt.subplots(figsize=(PAGE_WIDTH_IN, 2.9))
    width = 0.86 / len(BARS)
    x0 = np.arange(len(DATASETS))
    for j, cfg in enumerate(BARS):
        xs = x0 - 0.43 + width * (j + 0.5)
        vals = [float(conv.loc[(cfg, ds), "selected_epochs_to_target"]) for ds in DATASETS]
        cens = [bool(conv.loc[(cfg, ds), "selected_target_censored"]) for ds in DATASETS]
        ax.bar(xs, vals, width=width * 0.9, **bar_kwargs(cfg))
        for xi, v, c in zip(xs, vals, cens):
            ax.text(xi, v + 4, f"{'≤' if c else ''}{v:.0f}", ha="center", va="bottom", rotation=90, fontsize=7)
    ax.set_xticks(x0)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS], fontsize=8)
    ax.set_ylabel("Epochs to 95% of best\nvalidation Dice")
    ax.set_ylim(0, plotted["selected_epochs_to_target"].max() * 1.22)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    handles = [Patch(label=config_label(c), **bar_kwargs(c)) for c in BARS]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.8))
    save(fig, "fig06_convergence")


if __name__ == "__main__":
    main()
