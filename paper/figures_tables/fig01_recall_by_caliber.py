"""Fig. 1 (fig:recall_caliber): centerline recall by vessel caliber (radius bin), five models, four datasets.

Lines (left axis): mean centerline recall per radius bin. Grey bars (right axis, own scale per panel):
number of ground-truth centerline pixels in each bin, summed over the test images of the dataset.
MedSAM on DRIVE and VessMap with the upscaled protocol.
Output: ``paper/manuscript/figures/fig01_recall_by_caliber.{pdf,png}``.
"""

from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

from common import (
    CANONICAL_MODEL_STYLES,
    COMPARISON_CONFIGS,
    CONFIGS,
    DATASET_LABELS,
    DATASETS,
    N_TEST_IMAGES,
    PAGE_WIDTH_IN,
    RADIUS_BIN_COLUMNS,
    RADIUS_BIN_LABELS,
    legend_handles,
    load_per_image,
    marker_kwargs,
    np,
    paper_style,
    pd,
    percent_axis,
    plt,
    save,
    save_values,
)

BAR_COLOR = "#d9d9d9"
BIN_LOWER_EDGES = [0.5, 1.5, 2.5, 3.5, 4.5]


def bin_means(df):
    """Mean recall per (dataset, configuration, bin), images with an empty bin skipped."""
    sub = df[df["configuration"].isin(COMPARISON_CONFIGS)]
    return sub.groupby(["dataset", "configuration"])[RADIUS_BIN_COLUMNS].mean()


def gt_pixel_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Ground-truth centerline pixels per (dataset, bin), summed over test images.

    The counts depend only on the ground truth; the script checks that every configuration has the same count per
    image and bin and that every test image is present.
    """
    cols = ["n_gt_thin_centerline_pixels"] + [
        c.replace("centerline_recall", "n_gt_centerline_pixels") for c in RADIUS_BIN_COLUMNS[1:]
    ]
    if df.groupby(["dataset", "sample_id"])[cols].nunique().max().max() != 1:
        raise ValueError("Ground-truth centerline counts differ between configurations.")
    one = df.drop_duplicates(["dataset", "sample_id"])
    for ds in DATASETS:
        if one[one["dataset"] == ds]["sample_id"].nunique() != N_TEST_IMAGES[ds]:
            raise ValueError(f"{ds}: unexpected number of test images.")
    counts = one.groupby("dataset")[cols].sum()
    counts.columns = pd.Index([float(b) for b in BIN_LOWER_EDGES], name="radius_bin_min")
    return counts


def human(v: float, _pos=None) -> str:
    if v >= 1e6:
        return f"{v / 1e6:g}M"
    if v >= 1e3:
        return f"{v / 1e3:g}k"
    return f"{v:g}"


def main() -> None:
    paper_style()
    df = load_per_image()
    m = bin_means(df)
    counts = gt_pixel_counts(df)
    long = m.reset_index().melt(id_vars=["dataset", "configuration"], var_name="bin_column", value_name="mean_recall")
    cnt = counts.reset_index().melt(id_vars="dataset", var_name="radius_bin_min", value_name="gt_centerline_pixels")
    save_values(
        long.merge(
            cnt.assign(bin_column=cnt["radius_bin_min"].map(dict(zip(BIN_LOWER_EDGES, RADIUS_BIN_COLUMNS)))),
            on=["dataset", "bin_column"],
            how="left",
        ),
        "fig01_recall_by_caliber",
    )

    fig, axes = plt.subplots(2, 2, figsize=(PAGE_WIDTH_IN, 5.0), sharex=True, sharey=True)
    x = np.arange(len(RADIUS_BIN_COLUMNS))
    for i, (ax, ds) in enumerate(zip(axes.flat, DATASETS)):
        twin = ax.twinx()
        twin.bar(x, counts.loc[ds].to_numpy(), width=0.55, color=BAR_COLOR, edgecolor="none", zorder=0)
        twin.set_ylim(0, counts.loc[ds].max() * 1.12)
        twin.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        twin.yaxis.set_major_formatter(FuncFormatter(human))
        twin.tick_params(axis="y", labelsize=7, colors="#666666")
        twin.spines["right"].set_visible(True)
        twin.spines["right"].set_color("#aaaaaa")
        twin.grid(False)
        if i % 2 == 1:
            twin.set_ylabel("GT centerline pixels", color="#666666")
        ax.set_zorder(twin.get_zorder() + 1)  # lines above the bars
        ax.patch.set_visible(False)
        for cfg in COMPARISON_CONFIGS:
            arch = CONFIGS[cfg]["arch"]
            ax.plot(
                x,
                m.loc[(ds, cfg)].to_numpy(),
                linestyle=CANONICAL_MODEL_STYLES[arch]["linestyle"],
                **marker_kwargs(cfg, 4.5),
            )
        ax.set_title(f"({'abcd'[i]}) {DATASET_LABELS[ds]}", loc="left")
        ax.set_ylim(0, 1.02)
        ax.set_yticks(np.linspace(0, 1, 6))
        percent_axis(ax.yaxis)
        ax.set_xticks(x)
        ax.set_xticklabels(RADIUS_BIN_LABELS)
    for ax in axes[1]:
        ax.set_xlabel("Ground-truth vessel radius (px)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Centerline recall (%)")
    handles = legend_handles(COMPARISON_CONFIGS, 4.5, line=True)
    handles.append(Patch(facecolor=BAR_COLOR, edgecolor="none", label="GT centerline pixels (right axis)"))
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
        columnspacing=1.2,
        handletextpad=0.4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=0.8, w_pad=1.6)
    save(fig, "fig01_recall_by_caliber")


if __name__ == "__main__":
    main()
