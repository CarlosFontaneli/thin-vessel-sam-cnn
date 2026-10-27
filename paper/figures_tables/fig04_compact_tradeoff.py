"""Fig. 5 (fig:compact_tradeoff): thin-vessel recall and Dice against GFLOPs per image, with the compact models.

Rows: thin-vessel recall, Dice. Columns: datasets. Within a row every panel spans the same y-width
(same scale, offset per dataset). x: GFLOPs per test image, log scale; every panel spans the same number of
decades, centred on its data. Markers are drawn semi-opaque, compact models on top, so overlapping points
stay visible (values are not moved).
Output: ``paper/manuscript/figures/fig04_compact_tradeoff.{pdf,png}``.
"""

from common import (
    CONFIGS,
    DATASET_LABELS,
    DATASETS,
    PAGE_WIDTH_IN,
    legend_handles,
    load_complexity,
    load_per_image,
    marker_kwargs,
    means,
    np,
    paper_style,
    pd,
    percent_axis,
    plt,
    save,
    save_values,
)

CONFIGS_SHOWN = [
    "lwnet_scratch",
    "convunext_scratch",
    "convnext_scratch",
    "litemedsam_pretrained",
    "litemedsam_compact_scratch",
    "medsam_pretrained",
    "medsam_scratch",
    "medsam_compact_scratch",
]
ROWS = [("thin_vessel_recall", r"Centerline recall $R_\mathrm{thin}$ (%)", 0.32), ("dice", "Dice (%)", 0.14)]
X_MARGIN = 2.5  # factor of free space on each side of the widest panel's data
Z_ORDER = {"cnn": 3, "finetuned": 4, "scratch": 5, "compact": 6}
ALPHA = 0.85


def main() -> None:
    paper_style()
    df = load_per_image()
    cost = load_complexity().set_index(["configuration", "dataset"])["gflops_per_image"]
    values = {metric: means(df, metric) for metric, _, _ in ROWS}
    decades = max(
        np.log10(max(cost.loc[(c, d)] for c in CONFIGS_SHOWN) / min(cost.loc[(c, d)] for c in CONFIGS_SHOWN))
        for d in DATASETS
    ) + 2 * np.log10(X_MARGIN)
    rows = []
    fig, axes = plt.subplots(2, 4, figsize=(PAGE_WIDTH_IN, 4.1))
    for r, (metric, ylabel, span) in enumerate(ROWS):
        for c, ds in enumerate(DATASETS):
            ax = axes[r, c]
            ys = values[metric].loc[CONFIGS_SHOWN, ds]
            if ys.max() - ys.min() > span - 0.02:
                raise ValueError(f"{metric} spread on {ds} exceeds the common span")
            for cfg in CONFIGS_SHOWN:
                ax.plot(
                    cost.loc[(cfg, ds)],
                    ys.loc[cfg],
                    linestyle="none",
                    alpha=ALPHA,
                    zorder=Z_ORDER[CONFIGS[cfg]["regime"]],
                    **marker_kwargs(cfg, 6.2),
                )
                rows.append(
                    {
                        "dataset": ds,
                        "configuration": cfg,
                        "metric": metric,
                        "mean": ys.loc[cfg],
                        "gflops_per_image": cost.loc[(cfg, ds)],
                    }
                )
            xs = np.array([cost.loc[(cfg, ds)] for cfg in CONFIGS_SHOWN])
            ax.set_xscale("log")
            centre = (np.log10(xs.min()) + np.log10(xs.max())) / 2
            ax.set_xlim(10 ** (centre - decades / 2), 10 ** (centre + decades / 2))
            mid = (ys.max() + ys.min()) / 2
            ax.set_ylim(mid - span / 2, mid + span / 2)
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.1 if span > 0.2 else 0.05))
            percent_axis(ax.yaxis)
            ax.set_title(f"({'abcdefgh'[r * 4 + c]}) {DATASET_LABELS[ds]}", loc="left")
            if r == 1:
                ax.set_xlabel("GFLOPs per image")
            if c == 0:
                ax.set_ylabel(ylabel)
    fig.legend(
        handles=legend_handles(CONFIGS_SHOWN, 5.5), loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0)
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9), h_pad=0.8, w_pad=0.8)
    save_values(pd.DataFrame(rows), "fig04_compact_tradeoff")
    save(fig, "fig04_compact_tradeoff")


if __name__ == "__main__":
    main()
