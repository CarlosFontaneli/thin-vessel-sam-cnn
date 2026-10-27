"""Fig. 3 (fig:thin_errors): where the five models of the comparison find and miss thin vessels.

For every ground-truth thin centerline pixel (skeleton pixel with EDT radius below 1.5 px, as in the metrics), counts
how many of the five models (LW-Net, ConvUNeXt, ConvNeXt, fine-tuned LiteMedSAM, fine-tuned MedSAM; MedSAM with the
upscaled protocol on DRIVE and VessMap) have it inside their binarized prediction.
(a-d) One crop per dataset, the window of the test set with the most thin centerline pixels: thin centerline found
by all five models, by some, or by none, over the ground-truth vessels (dark grey on black).
(e) Share of the thin centerline pixels of all test images found by all, some or none of the models.
Also reports, per model, the share of its missed thin pixels that lie within 2 px of its prediction (``near_miss``).
Reads the saved test predictions (``$OUTPUTS_DIR/predictions``; skeleton, radius and masks with the functions of
``thinvessel.evaluation.metrics``); checks that the per-model mean R_thin equals the per-image table before plotting.
Output: ``paper/manuscript/figures/fig07_thin_errors.{pdf,png}``.
"""

from scipy.ndimage import binary_dilation, distance_transform_edt

from common import (
    COMPARISON_CONFIGS,
    DATASET_LABELS,
    DATASETS,
    OUTPUTS_DIR,
    PAGE_WIDTH_IN,
    load_per_image,
    means,
    np,
    paper_style,
    pd,
    plt,
    save,
    save_values,
    study_of,
)
from thinvessel.evaluation.metrics import THIN_RADIUS_PX, distance_map, extract_centerline, load_binary_mask
from thinvessel.evaluation.scoring import paper_protocol, prediction_dir

CROP_PX = {"drive": 160, "hrf": 320, "fives": 256, "vessmap": 128}  # about a quarter of the shorter image side
# Diverging encoding (two poles, neutral midpoint) with the paper's orange and green (validated: CVD dE 10.5 deutan,
# normal-vision dE 19.4); crops on black like a ground-truth mask, vessels in dark grey.
COLORS = {"none": "#e65719", "some": "#bdbdbd", "all": "#45a075"}
TEXT_ON = {"none": "white", "some": "#222222", "all": "white"}
LABELS = {"none": "missed by all five", "some": "found by some", "all": "found by all five"}
VESSEL_GREY = "#333333"
BACKGROUND = "#000000"


def run_dir(config: str, ds: str):
    """Saved test predictions of a comparison model under the paper's protocol (upscaled MedSAM on DRIVE, VessMap)."""
    study = study_of(config, ds)
    return prediction_dir(OUTPUTS_DIR, study, paper_protocol(study))


def load_mask(path) -> np.ndarray:
    return load_binary_mask(path).astype(bool)


def analyse(ds: str):
    names = sorted(p.name for p in (run_dir(COMPARISON_CONFIGS[0], ds) / "ground_truth").glob("*.png"))
    counts = {"none": 0, "some": 0, "all": 0}
    near = {c: [0, 0] for c in COMPARISON_CONFIGS}  # missed thin pixels within 2 px of the prediction, all missed
    rthin = {c: [] for c in COMPARISON_CONFIGS}
    best = (-1, None)
    cs = CROP_PX[ds]
    for name in names:
        gt = load_mask(run_dir(COMPARISON_CONFIGS[0], ds) / "ground_truth" / name)
        thin = extract_centerline(gt).astype(bool) & (distance_map(gt) < THIN_RADIUS_PX)
        if not thin.any():
            continue
        found = np.zeros(gt.shape, int)
        for cfg in COMPARISON_CONFIGS:
            if not (load_mask(run_dir(cfg, ds) / "ground_truth" / name) == gt).all():
                raise ValueError(f"{ds} {name}: ground truths differ between runs.")
            pred = load_mask(run_dir(cfg, ds) / "predicted_masks" / name)
            hit = thin & pred
            found += hit
            rthin[cfg].append(hit.sum() / thin.sum())
            miss = thin & ~pred
            dist = distance_transform_edt(~pred) if pred.any() else np.full(gt.shape, np.inf)
            near[cfg][0] += int((miss & (dist <= 2)).sum())
            near[cfg][1] += int(miss.sum())
        n = len(COMPARISON_CONFIGS)
        counts["all"] += int((thin & (found == n)).sum())
        counts["none"] += int((thin & (found == 0)).sum())
        counts["some"] += int((thin & (found > 0) & (found < n)).sum())
        step = cs // 2
        for y in range(0, max(1, gt.shape[0] - cs + 1), step):
            for x in range(0, max(1, gt.shape[1] - cs + 1), step):
                k = int(thin[y : y + cs, x : x + cs].sum())
                if k > best[0]:
                    best = (
                        k,
                        (
                            name,
                            y,
                            x,
                            gt[y : y + cs, x : x + cs],
                            thin[y : y + cs, x : x + cs],
                            found[y : y + cs, x : x + cs],
                        ),
                    )
    return counts, near, rthin, best[1]


def crop_rgb(gt: np.ndarray, thin: np.ndarray, found: np.ndarray) -> np.ndarray:
    n = len(COMPARISON_CONFIGS)
    rgb = np.zeros(gt.shape + (3,))
    rgb[:] = matplotlib_rgb(BACKGROUND)
    rgb[gt] = matplotlib_rgb(VESSEL_GREY)
    for key, mask in (
        ("some", thin & (found > 0) & (found < n)),
        ("none", thin & (found == 0)),
        ("all", thin & (found == n)),
    ):
        rgb[binary_dilation(mask)] = matplotlib_rgb(COLORS[key])  # one-pixel lines widened to stay visible in print
    return rgb


def matplotlib_rgb(hex_color: str):
    from matplotlib.colors import to_rgb

    return to_rgb(hex_color)


def main() -> None:
    paper_style()
    paper_rthin = means(load_per_image(), "thin_vessel_recall")
    rows, crops = [], {}
    for ds in DATASETS:
        counts, near, rthin, crop = analyse(ds)
        for cfg in COMPARISON_CONFIGS:
            if abs(np.mean(rthin[cfg]) - paper_rthin.loc[cfg, ds]) > 1e-3:
                raise ValueError(
                    f"{ds} {cfg}: R_thin {np.mean(rthin[cfg]):.4f} != paper {paper_rthin.loc[cfg, ds]:.4f}"
                )
        total = sum(counts.values())
        name, y, x = crop[:3]
        row = {
            "dataset": ds,
            "thin_pixels": total,
            "crop_image": name,
            "crop_y": y,
            "crop_x": x,
            "crop_px": CROP_PX[ds],
        }
        row.update({f"share_{k}": counts[k] / total for k in counts})
        row.update({f"near_miss_{cfg}": near[cfg][0] / max(1, near[cfg][1]) for cfg in COMPARISON_CONFIGS})
        rows.append(row)
        crops[ds] = crop[3:]

    fig = plt.figure(figsize=(PAGE_WIDTH_IN, 3.35))
    grid = fig.add_gridspec(2, 4, height_ratios=[1.75, 0.85], hspace=0.35, wspace=0.08)
    for i, ds in enumerate(DATASETS):
        ax = fig.add_subplot(grid[0, i])
        ax.imshow(crop_rgb(*crops[ds]), interpolation="nearest")
        ax.set_title(f"({'abcd'[i]}) {DATASET_LABELS[ds]}", loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#000000")
            spine.set_linewidth(0.5)
    ax = fig.add_subplot(grid[1, :])
    labels = [DATASET_LABELS[ds] for ds in DATASETS][::-1]
    left = np.zeros(len(DATASETS))
    for key in ("none", "some", "all"):
        vals = np.array([r[f"share_{key}"] for r in rows][::-1])
        ax.barh(
            labels, vals, left=left, color=COLORS[key], edgecolor="white", linewidth=1.0, height=0.7, label=LABELS[key]
        )
        for j, (v, l0) in enumerate(zip(vals, left)):
            if v >= 0.08:
                ax.text(l0 + v / 2, j, f"{100 * v:.0f}", ha="center", va="center", fontsize=6.5, color=TEXT_ON[key])
        left += vals
    ax.set_xlim(0, 1)
    ax.set_axisbelow(True)
    ax.grid(False, axis="y")
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}"))
    ax.set_xlabel("Share of the thin centerline pixels of the test images (%)")
    ax.set_title("(e)", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    handles, names = ax.get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.09, right=0.975, top=0.88, bottom=0.12)
    save_values(pd.DataFrame(rows), "fig07_thin_errors")
    save(fig, "fig07_thin_errors")


if __name__ == "__main__":
    main()
