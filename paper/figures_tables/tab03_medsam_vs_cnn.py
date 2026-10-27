"""Table III (tab:medsam_vs_cnn): fine-tuned MedSAM against the best CNN of each dataset.

For each dataset and metric (Dice, thin-vessel
recall): the mean of fine-tuned MedSAM (upscaled protocol on DRIVE and VessMap), the highest mean among the three
CNNs (named by a superscript) and the difference in percentage points, computed from the printed one-decimal
values, with an asterisk when the Wilcoxon test (``all_tests.csv``, family F1) finds the two significantly different.
Means are given with the standard deviation over test images. Every mean is checked against
``outputs/figures/plotted/fig05_pretraining.csv`` (run ``fig05_pretraining.py``
first). The full data of Fig. 6 (pre-training effect) is in ``tab04_pretraining_gain.py``. Training difficulty:
epochs until the
selected run first reached 95% of its best validation Dice, for fine-tuned MedSAM and for the CNN with the highest Dice.
Sources: ``per_image_metrics.csv`` (via ``common.load_per_image``), ``convergence.csv``.
Output: ``paper/manuscript/figures/tab03_medsam_vs_cnn.tex``.
"""

from common import (
    DATASETS,
    MARKS_PAIR,
    PLOTTED_DIR,
    STAR,
    bold_group,
    convergence_row,
    ds_label,
    load_per_image,
    load_tests,
    means,
    np,
    pd,
    pm,
    read_table,
    save_values,
    sds,
    signed_delta,
    test_result,
    write_tex,
)

CNNS = ["lwnet_scratch", "convunext_scratch", "convnext_scratch"]
CNN_SHORT = {"lwnet_scratch": "L", "convunext_scratch": "U", "convnext_scratch": "N"}
MEDSAM = "medsam_pretrained"
METRICS = [("dice", "Dice"), ("thin_vessel_recall", r"$\thinrecall$")]


def epochs(config: str, ds: str) -> str:
    row = convergence_row(config, ds)
    value = f"{int(row['selected_epochs_to_target'])}"
    return rf"$\le${value}" if bool(row["selected_target_censored"]) else value


def _epochs_value(e: str) -> int:
    """Epoch count of a printed cell ("$\\le$N" or "N")."""
    return int(e.replace(r"$\le$", ""))


def main() -> None:
    df = load_per_image()
    tests = load_tests()
    fig5 = read_table(PLOTTED_DIR / "fig05_pretraining.csv").set_index(["metric", "dataset", "configuration"])
    rows, body = [], []
    best_dice_cnn = {}
    cells = {ds: [] for ds in DATASETS}
    for metric, _ in METRICS:
        m = means(df, metric) * 100
        sd = sds(df, metric) * 100
        for ds in DATASETS:
            for cfg in CNNS + [MEDSAM]:
                if not np.isclose(m.loc[cfg, ds], 100 * fig5.loc[(metric, ds, cfg), "mean"]):
                    raise ValueError(f"{cfg} on {ds}, {metric}: differs from the Fig. 6 data.")
        for ds in DATASETS:
            printed = m[ds].round(1)
            best = printed.loc[CNNS].idxmax()
            if (printed.loc[CNNS] == printed[best]).sum() > 1:
                raise ValueError(f"Tie between CNNs on {ds}, {metric}.")
            delta = printed[MEDSAM] - printed[best]
            if abs(delta - (m.loc[MEDSAM, ds] - m.loc[best, ds])) > 0.1:
                raise ValueError(f"Rounded difference far from the unrounded one on {ds}, {metric}.")
            if metric == "dice":
                best_dice_cnn[ds] = best
            t = test_result(tests, ds, metric, MEDSAM, best, m.loc[MEDSAM, ds] / 100, m.loc[best, ds] / 100)
            bold = bold_group(
                tests, ds, metric, {"m": (MEDSAM, m.loc[MEDSAM, ds] / 100), "c": (best, m.loc[best, ds] / 100)}
            )
            bold = {k: (v if v == "best" else False) for k, v in bold.items()}  # the starred Delta gives the test
            cells[ds] += [
                pm(m.loc[MEDSAM, ds], sd.loc[MEDSAM, ds], bold["m"]),
                pm(m.loc[best, ds], sd.loc[best, ds], bold["c"]) + rf"\textsuperscript{{{CNN_SHORT[best]}}}",
                signed_delta(delta, t["significant"]),
            ]
            rows.append(
                {
                    "dataset": ds,
                    "metric": metric,
                    "medsam_ft": m.loc[MEDSAM, ds],
                    "medsam_ft_sd": sd.loc[MEDSAM, ds],
                    "best_cnn": best,
                    "best_cnn_value": m.loc[best, ds],
                    "best_cnn_sd": sd.loc[best, ds],
                    "delta_printed_pp": delta,
                    "p_holm": t["p_holm"],
                    "significant": t["significant"],
                }
            )
    for ds in DATASETS:
        e_med, e_cnn = epochs(MEDSAM, ds), epochs(best_dice_cnn[ds], ds)
        rows.append(
            {
                "dataset": ds,
                "metric": "epochs_to_95pct",
                "medsam_ft": e_med,
                "best_cnn": best_dice_cnn[ds],
                "best_cnn_value": e_cnn,
                "delta_printed_pp": None,
            }
        )
        fewer = min(_epochs_value(e_med), _epochs_value(e_cnn))
        bm = [rf"\textbf{{{e}}}" if _epochs_value(e) == fewer else e for e in (e_med, e_cnn)]
        body.append(f"{ds_label(ds)} & " + " & ".join(cells[ds]) + f" & {bm[0]} & {bm[1]} \\\\")
    save_values(pd.DataFrame(rows), "tab03_medsam_vs_cnn")

    k = len(METRICS)
    header1 = " & ".join(rf"\multicolumn{{3}}{{c}}{{{label}}}" for _, label in METRICS)
    cmid = (
        " ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(k))
        + rf" \cmidrule(lr){{{2 + 3 * k}-{3 + 3 * k}}}"
    )
    header2 = " & ".join(r"\medsam{} & Best CNN & $\Delta$" for _ in METRICS)
    tex = rf"""% Generated by paper/figures_tables/tab03_medsam_vs_cnn.py -- do not edit by hand.
% Source: results/per_image_metrics.csv (MedSAM DRIVE/VessMap upscaled protocol); results/convergence.csv.
\begin{{table*}}[tbp]
\caption{{Fine-tuned \medsam{{}} against the best CNN of each metric and dataset (L: \lwnet{{}}, U: \convunext{{}}, N:
\convnext{{}}); $\Delta$: \medsam{{}} minus the best CNN. {MARKS_PAIR}; {STAR}. Epochs: of \medsam{{}} and of the
CNN with the highest Dice
(bold: fewer; ``$\le$'': reached at the first validation).}}
\label{{tab:medsam_vs_cnn}}
\begin{{ruledtabular}}
\begin{{tabular}}{{l{"ccc" * k}cc}}
 & {header1} & \multicolumn{{2}}{{c}}{{Epochs}} \\
{cmid}
Dataset & {header2} & \medsam{{}} & CNN \\
\midrule
{chr(10).join(body)}
\end{{tabular}}
\end{{ruledtabular}}
\end{{table*}}
"""
    write_tex("tab03_medsam_vs_cnn", tex)


if __name__ == "__main__":
    main()
