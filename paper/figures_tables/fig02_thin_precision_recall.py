"""Fig. 2 (fig:thin_pr): thin-vessel precision vs thin-vessel recall, five models, one panel per dataset.

Every panel spans the same width on both axes (same scale, offset per dataset), so distances between
points are comparable across datasets while each dataset's points stay readable.
Output: ``paper/manuscript/figures/fig02_thin_precision_recall.{pdf,png}``.
"""

from common import (
    COMPARISON_CONFIGS,
    DATASET_LABELS,
    DATASETS,
    PAGE_WIDTH_IN,
    legend_handles,
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

SPAN = 0.30  # axis width (both axes, every panel)


def centred_limits(values: np.ndarray) -> tuple:
    mid = (np.nanmin(values) + np.nanmax(values)) / 2
    lo = max(0.0, mid - SPAN / 2)
    return lo, lo + SPAN


def main() -> None:
    paper_style()
    df = load_per_image()
    rec = means(df, "thin_vessel_recall")
    prec = means(df, "thin_vessel_precision")
    if any(rec.loc[COMPARISON_CONFIGS].max(axis=0) - rec.loc[COMPARISON_CONFIGS].min(axis=0) > SPAN - 0.04) or any(
        prec.loc[COMPARISON_CONFIGS].max(axis=0) - prec.loc[COMPARISON_CONFIGS].min(axis=0) > SPAN - 0.04
    ):
        raise ValueError("Data spread exceeds the common axis span; increase SPAN.")
    rows = []
    fig, axes = plt.subplots(1, 4, figsize=(PAGE_WIDTH_IN, 2.35))
    for i, (ax, ds) in enumerate(zip(axes, DATASETS)):
        for cfg in COMPARISON_CONFIGS:
            ax.plot(rec.loc[cfg, ds], prec.loc[cfg, ds], linestyle="none", **marker_kwargs(cfg, 6))
            rows.append(
                {
                    "dataset": ds,
                    "configuration": cfg,
                    "thin_vessel_recall": rec.loc[cfg, ds],
                    "thin_vessel_precision": prec.loc[cfg, ds],
                }
            )
        ax.set_xlim(*centred_limits(rec.loc[COMPARISON_CONFIGS, ds].to_numpy()))
        ax.set_ylim(*centred_limits(prec.loc[COMPARISON_CONFIGS, ds].to_numpy()))
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        percent_axis(ax.xaxis)
        percent_axis(ax.yaxis)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"({'abcd'[i]}) {DATASET_LABELS[ds]}", loc="left")
        ax.set_xlabel(r"Centerline recall $R_\mathrm{thin}$ (%)")
    axes[0].set_ylabel(r"Centerline precision $P_\mathrm{thin}$ (%)")
    fig.legend(
        handles=legend_handles(COMPARISON_CONFIGS, 5.5),
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=1.2)
    save_values(pd.DataFrame(rows), "fig02_thin_precision_recall")
    save(fig, "fig02_thin_precision_recall")


if __name__ == "__main__":
    main()
