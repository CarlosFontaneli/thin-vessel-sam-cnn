"""Table IV (tab:padding): MedSAM on DRIVE and VessMap, padded vs upscaled inference at the same cost.

Metric means come from the per-image table (the native, i.e. padded, and the upscaled predictions of the same
checkpoints). The cost is the per-window GFLOPs (``cost.csv``) times the number of sliding windows per image,
computed here with MONAI's own window grid (25% overlap) for the padded native image and for the image resized to
the training size; the script fails if the two counts differ or disagree with ``cost.csv``. Values are mean ± standard
deviation over test images; the difference carries an asterisk when the test of family F3 (``all_tests.csv``: the 12
padded vs upscaled comparisons) is significant. Rows: run and metric; columns: dataset.
Output: ``paper/manuscript/figures/tab02_padding_protocol.tex``.
"""

from monai.data.utils import dense_patch_slices
from monai.inferers.utils import _get_scan_interval

from common import (
    DATASET_LABELS,
    MARKS_PAIR,
    N_TEST_IMAGES,
    STAR,
    load_complexity,
    load_config_file,
    load_per_image,
    load_tests,
    np,
    pd,
    pm,
    save_values,
    signed_delta,
    test_result,
    write_tex,
)
from thinvessel.training.module import SLIDING_WINDOW_OVERLAP as OVERLAP

RUNS = [("medsam_pretrained", "fine-tuned"), ("medsam_scratch", "from scratch")]
DATASETS = ["drive", "vessmap"]
METRICS = [
    ("dice", "Dice"),
    ("thin_vessel_recall", r"$\thinrecall$"),
    ("thin_vessel_precision", r"$P_{\mathrm{thin}}$"),
]
PROTOCOLS = [("native", "padded"), ("upscaled", "training scale")]


def n_windows(image_hw, roi):
    """Number of sliding windows MONAI uses for an image (padded up to the roi when smaller)."""
    size = tuple(max(s, r) for s, r in zip(image_hw, roi))
    scan = _get_scan_interval(size, roi, 2, (OVERLAP, OVERLAP))
    return len(dense_patch_slices(size, roi, scan))


def main() -> None:
    per_image = load_per_image(include_padded=True, exact=True)  # as the step-12 parquet the paper read
    cost = load_complexity().set_index(["configuration", "dataset"])
    rows, checks = [], []
    for cfg, regime in RUNS:
        for ds in DATASETS:
            yml = load_config_file(cfg, ds)
            train_size = int(yml["dataset_params"]["image_size"])
            roi = int(yml["training_params"]["roi_size"])
            grp = per_image[(per_image["configuration"] == cfg) & (per_image["dataset"] == ds)]
            sizes = grp[["height", "width"]].drop_duplicates()
            if len(sizes) != 1:
                raise ValueError(f"{ds}: test images differ in size")
            hw = (int(sizes["height"].iloc[0]), int(sizes["width"].iloc[0]))
            win = {"native": n_windows(hw, (roi, roi)), "upscaled": n_windows((train_size, train_size), (roi, roi))}
            if win["native"] != win["upscaled"]:
                raise ValueError(f"{cfg} {ds}: window counts differ {win}")
            measured = cost.loc[(cfg, ds)]
            if int(measured["windows_per_image"]) != win["native"] or int(measured["roi_size"]) != roi:
                raise ValueError(f"{cfg} {ds}: measured windows {measured['windows_per_image']} vs computed {win}")
            checks.append(
                f"{cfg} {ds}: native {hw[0]}x{hw[1]}, training size {train_size}, roi {roi}, windows "
                f"padded={win['native']} training-scale={win['upscaled']}"
            )
            for proto, proto_label in PROTOCOLS:
                sub = grp[grp["protocol"] == proto]
                if len(sub) != N_TEST_IMAGES[ds]:
                    raise ValueError(f"{cfg} {ds} {proto}: {len(sub)} images")
                row = {
                    "configuration": cfg,
                    "regime": regime,
                    "dataset": ds,
                    "protocol": proto_label,
                    "n_images": len(sub),
                    "windows_per_image": win[proto],
                    "gflops_per_window": measured["gflops"],
                    "gflops_per_image": measured["gflops"] * win[proto],
                }
                for metric, _ in METRICS:
                    row[metric] = sub[metric].mean()
                    row[f"{metric}_sd"] = sub[metric].std()
                    row[f"{metric}_n"] = int(sub[metric].notna().sum())
                rows.append(row)
    table = pd.DataFrame(rows)
    print("\n".join(checks))

    # Layout: one row per run and metric; for each dataset the padded value, the upscaled value and their difference
    # (upscaled minus padded, percentage points, from the printed one-decimal values so that every row adds up;
    # checked against the unrounded difference). The cost is the same in every row and under both protocols
    # (asserted) and is stated in the caption.
    tests = load_tests()
    test_names = {"medsam_pretrained": "MedSAM FT", "medsam_scratch": "MedSAM scratch"}
    body = []
    for i, (cfg, regime) in enumerate(RUNS):
        if i:
            body.append(r"\midrule")
        for j, (m, lab) in enumerate(METRICS):
            cells = []
            for ds in DATASETS:
                pair = table[(table["configuration"] == cfg) & (table["dataset"] == ds)].set_index("protocol")
                pad, ts = pair.loc["padded"], pair.loc["training scale"]
                if not np.isclose(pad["gflops_per_image"], ts["gflops_per_image"]):
                    raise ValueError(f"{cfg} {ds}: GFLOPs per image differ between protocols")
                p_val, t_val = round(100 * pad[m], 1), round(100 * ts[m], 1)
                delta = round(t_val - p_val, 1)
                if abs(delta - 100 * (ts[m] - pad[m])) > 0.1 + 1e-9:
                    raise ValueError(f"{cfg} {ds} {m}: printed difference {delta} far from unrounded")
                t = test_result(tests, ds, m, f"{test_names[cfg]} upscaled", f"{test_names[cfg]} padded", ts[m], pad[m])
                b_up, b_pad = ("best", False) if ts[m] >= pad[m] else (False, "best")
                cells += [
                    pm(100 * pad[m], 100 * pad[f"{m}_sd"], b_pad),
                    pm(100 * ts[m], 100 * ts[f"{m}_sd"], b_up),
                    signed_delta(delta, t["significant"]),
                ]
                table.loc[(table["configuration"] == cfg) & (table["dataset"] == ds), f"{m}_p_holm"] = t["p_holm"]
            run = rf"\multirow{{{len(METRICS)}}}{{*}}{{{regime.capitalize()}}}" if j == 0 else ""
            body.append(f"{run} & {lab} & {' & '.join(cells)} \\\\")
    save_values(table, "tab02_padding_protocol")
    header1 = " & ".join(
        rf"\multicolumn{{3}}{{c}}{{{DATASET_LABELS[ds]} ($n={N_TEST_IMAGES[ds]}$)}}" for ds in DATASETS
    )
    cmid = " ".join(rf"\cmidrule(lr){{{3 + 3 * k}-{5 + 3 * k}}}" for k in range(len(DATASETS)))
    header2 = " & ".join(["Padded", "Upscaled", r"$\Delta$"] * len(DATASETS))
    if len(set(table["gflops_per_image"].round(6))) != 1:
        raise ValueError("GFLOPs per image differ between rows")
    notes = []
    for _, r in table.iterrows():
        for m, lab in METRICS:
            if r[f"{m}_n"] < r["n_images"]:
                undefined = int(r["n_images"] - r[f"{m}_n"])
                count = "one image" if undefined == 1 else f"{undefined} images"
                notes.append(
                    f" {lab}, {DATASET_LABELS[r['dataset']]}, {r['regime']}, {r['protocol']}: mean over "
                    f"{r[f'{m}_n']} images ({count} without predicted thin vessels)."
                )
    missing_note = "".join(notes)
    if len(set(table["windows_per_image"].astype(int))) != 1:
        raise ValueError("Window counts differ between runs")
    tex = rf"""% Generated by paper/figures_tables/tab02_padding_protocol.py -- do not edit by hand.
% Source: results/per_image_metrics.csv (native = padded and upscaled protocols); GFLOPs per window from
% results/cost.csv; windows per image from MONAI's grid.
\begin{{table*}}[tbp]
\caption{{The same \medsam{{}} checkpoints evaluated with the padded and the upscaled protocol; $\Delta$: upscaled minus
padded. {MARKS_PAIR}; {STAR}.{missing_note}}}
\label{{tab:padding}}
\begin{{ruledtabular}}
\begin{{tabular}}{{ll{"ccc" * len(DATASETS)}}}
 & & {header1} \\
{cmid}
\medsam{{}} & Metric & {header2} \\
\midrule
{chr(10).join(body)}
\end{{tabular}}
\end{{ruledtabular}}
\end{{table*}}
"""
    write_tex("tab02_padding_protocol", tex)


if __name__ == "__main__":
    main()
