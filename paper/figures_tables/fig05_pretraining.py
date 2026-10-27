"""Fig. 6 (fig:pretraining): fine-tuned vs from-scratch MedSAM and LiteMedSAM against the CNNs, per dataset.

Panels, side by side (page width): Dice, thin-vessel recall, thin-vessel precision (the two thin-vessel panels
share one y-range).
In each dataset group: the three CNNs (left), then LiteMedSAM and
MedSAM, each with its fine-tuned (filled) and from-scratch (hollow) mean joined by a segment.
Output: ``paper/manuscript/figures/fig05_pretraining.{pdf,png}``.
"""

from common import (
    DATASET_LABELS,
    DATASETS,
    PAGE_WIDTH_IN,
    arch_color,
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

CNNS = ["lwnet_scratch", "convunext_scratch", "convnext_scratch"]
PAIRS = [
    ("litemedsam_pretrained", "litemedsam_scratch", "litemedsam"),
    ("medsam_pretrained", "medsam_scratch", "medsam"),
]
PANELS = [
    ("dice", "Dice (%)", (0.75, 0.95)),
    ("thin_vessel_recall", r"Centerline recall $R_\mathrm{thin}$ (%)", (0.2, 0.9)),
    ("thin_vessel_precision", r"Centerline precision $P_\mathrm{thin}$ (%)", (0.2, 0.9)),
]
# Positions inside a dataset group (width 1): spread over most of the group so that the markers do not overlap
# .
CNN_OFFSETS = [-0.39, -0.25, -0.11]
PAIR_OFFSETS = [0.11, 0.33]
# Distance between the centres of two dataset groups, and a light band behind each group, so that every
# group reads as one dataset (the offsets above keep the spacing inside a group).
GROUP_PITCH = 1.3
BAND = (-0.5, 0.44)  # band edges around the group centre
BAND_COLOR = "#efefef"


def main() -> None:
    paper_style()
    df = load_per_image()
    rows = []
    fig, axes = plt.subplots(1, len(PANELS), figsize=(PAGE_WIDTH_IN, 2.6))
    for p, (ax, (metric, ylabel, ylim)) in enumerate(zip(axes, PANELS)):
        m = means(df, metric)
        for k, ds in enumerate(DATASETS):
            c = k * GROUP_PITCH
            ax.axvspan(c + BAND[0], c + BAND[1], color=BAND_COLOR, lw=0, zorder=0)
            for cfg, off in zip(CNNS, CNN_OFFSETS):
                ax.plot(c + off, m.loc[cfg, ds], linestyle="none", **marker_kwargs(cfg, 4.5))
                rows.append({"metric": metric, "dataset": ds, "configuration": cfg, "mean": m.loc[cfg, ds]})
            for (ft, sc, arch), off in zip(PAIRS, PAIR_OFFSETS):
                ax.plot([c + off, c + off], [m.loc[ft, ds], m.loc[sc, ds]], color=arch_color(arch), lw=1.0, zorder=1)
                ax.plot(c + off, m.loc[ft, ds], linestyle="none", zorder=2, **marker_kwargs(ft, 5.0))
                ax.plot(c + off, m.loc[sc, ds], linestyle="none", zorder=2, **marker_kwargs(sc, 5.0))
                for cfg in (ft, sc):
                    rows.append({"metric": metric, "dataset": ds, "configuration": cfg, "mean": m.loc[cfg, ds]})
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        percent_axis(ax.yaxis)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"({'abc'[p]})", loc="left")
        ax.set_xticks(np.arange(len(DATASETS)) * GROUP_PITCH)
        ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS], fontsize=7)
        ax.set_xlim(BAND[0] - 0.08, (len(DATASETS) - 1) * GROUP_PITCH + BAND[1] + 0.08)
    handles = legend_handles(CNNS + [c for pair in PAIRS for c in pair[:2]], 4.5)
    # Column-major, 2 rows: LW-Net/ConvUNeXt | ConvNeXt/(blank) | LiteMedSAM ft/scratch | MedSAM ft/scratch.
    handles.insert(3, plt.Line2D([], [], linestyle="none", label=""))
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
        columnspacing=1.2,
        handletextpad=0.3,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.82), w_pad=1.2)
    save_values(pd.DataFrame(rows), "fig05_pretraining")
    save(fig, "fig05_pretraining")


if __name__ == "__main__":
    main()
