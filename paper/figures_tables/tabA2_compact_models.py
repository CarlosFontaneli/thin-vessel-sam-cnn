"""Table VI (tab:compact_models): architecture of the compact models selected on each dataset, next to MedSAM and
LiteMedSAM.

For each dataset, the architecture of the selected trial of the compact-model search (the configuration it
trained with), its parameter count, its test-set Dice, thin-vessel recall and thin-vessel precision, and the
values the search could choose from. Reference rows: the architecture of MedSAM and LiteMedSAM (the same for the
fine-tuned and from-scratch runs on every dataset); their per-dataset metrics are in Table VIII (tab:pretraining_gain).
On DRIVE the search selected a compact LiteMedSAM larger than LiteMedSAM; the caption states it (checked here).
Sources: the released configurations (architecture of the selected trials, searched values), ``cost.csv``
(parameters, GFLOPs per image), ``per_image_metrics.csv`` (metrics).
Output: ``paper/manuscript/figures/tabA2_compact_models.tex``.
"""

import json

from common import (
    DATASET_LABELS,
    DATASETS,
    load_complexity,
    load_config_file,
    load_per_image,
    means,
    pd,
    pm,
    save_values,
    sds,
    search_space,
    write_tex,
)

METRICS = ["dice", "thin_vessel_recall", "thin_vessel_precision"]

MEDSAM_KEYS = ["img_size", "patch_size", "embed_dim", "depth", "num_heads", "mlp_ratio", "window_size"]
LITE_KEYS = ["img_size", "embed_dims", "depths", "num_heads", "mlp_ratio", "window_sizes"]


def arch(config: str, ds: str) -> dict:
    mp = load_config_file(config, ds)["model_params"]
    if "medsam_params" in mp:
        return {k: mp["medsam_params"]["vit_b_params"][k] for k in MEDSAM_KEYS}
    return {k: mp["litemedsam_params"]["tiny_vit_params"][k] for k in LITE_KEYS}


def searched(config: str, ds: str) -> dict:
    return {
        n: d["attributes"]["choices"]
        for n, d in search_space(config, ds).items()
        if n not in ("optimizer", "batch_size", "lr", "weight_decay", "patience")
    }


def fmt(v) -> str:
    if isinstance(v, str):
        v = json.loads(v)
    if isinstance(v, list):
        return ",".join(fmt(x) for x in v)
    return f"{v:g}" if isinstance(v, float) else str(v)


def main() -> None:
    cx = load_complexity().set_index(["configuration", "dataset"])
    per_image = load_per_image()
    metric_means = {metric: means(per_image, metric) * 100 for metric in METRICS}
    metric_sds = {metric: sds(per_image, metric) * 100 for metric in METRICS}
    rows, body = [], []
    blocks = [
        ("medsam", "medsam_pretrained", "medsam_compact_scratch", r"\medsam{}"),
        ("litemedsam", "litemedsam_pretrained", "litemedsam_compact_scratch", r"\litemedsam{}"),
    ]
    spaces = {}
    for model, ref_cfg, compact_cfg, macro in blocks:
        ref = [arch(ref_cfg, ds) for ds in DATASETS]
        if any(r != ref[0] for r in ref):
            raise ValueError(f"{ref_cfg}: architecture differs between datasets.")
        scratch_ref = [arch(ref_cfg.replace("pretrained", "scratch"), ds) for ds in DATASETS]
        if any(r != ref[0] for r in scratch_ref):
            raise ValueError(f"{model}: from-scratch architecture differs from the fine-tuned one.")
        ref_params = {
            round(cx.loc[(c, ds), "params_m"], 1)
            for ds in DATASETS
            for c in (ref_cfg, ref_cfg.replace("pretrained", "scratch"))
        }
        if len(ref_params) != 1:
            raise ValueError(f"{model}: parameter counts {ref_params}.")
        entries = [("", "all", ref[0], ref_params.pop())]
        for ds in DATASETS:
            entries.append(
                ("compact", DATASET_LABELS[ds], arch(compact_cfg, ds), cx.loc[(compact_cfg, ds), "params_m"])
            )
            spaces[(model, ds)] = searched(compact_cfg, ds)
        for i, (regime, ds_label, a, params) in enumerate(entries):
            if model == "medsam":
                grid = a["img_size"] // a["patch_size"]
                cells = [
                    a["img_size"],
                    a["patch_size"],
                    f"${grid}\\times{grid}$",
                    a["embed_dim"],
                    a["depth"],
                    a["num_heads"],
                    fmt(a["mlp_ratio"]),
                    a["window_size"],
                ]
            else:
                cells = [
                    a["img_size"],
                    "--",
                    "--",
                    fmt(a["embed_dims"]),
                    fmt(a["depths"]),
                    fmt(a["num_heads"]),
                    fmt(a["mlp_ratio"]),
                    fmt(a["window_sizes"]),
                ]
            first = rf"\multirow{{5}}{{*}}{{{macro}}}" if i == 0 else ""
            label = "original" if regime == "" else f"compact, {ds_label}"
            if regime == "":  # metrics differ per dataset and regime: given in the pre-training table
                scores = [r"\multicolumn{3}{c}{Table~\ref{tab:pretraining_gain}}"]
                values = {}
            else:
                ds = DATASETS[i - 1]
                values = {metric: metric_means[metric].loc[compact_cfg, ds] for metric in METRICS}
                values.update({f"{metric}_sd": metric_sds[metric].loc[compact_cfg, ds] for metric in METRICS})
                scores = [pm(values[metric], values[f"{metric}_sd"], tight=True) for metric in METRICS]
            body.append(
                f"{first} & {label} & "
                + " & ".join(str(c) for c in cells)
                + f" & {params:.1f} & "
                + " & ".join(scores)
                + " \\\\"
            )
            rows.append(
                {
                    "model": model,
                    "regime": regime,
                    "dataset": ds_label,
                    **{k: str(v) for k, v in a.items()},
                    "params_m": params,
                    **values,
                }
            )
        body.append(r"\midrule")
    body.pop()
    save_values(pd.DataFrame(rows), "tabA2_compact_models")

    # Searched values, stated in the caption (checked to be the same on every dataset unless named).
    def choices(model: str, key: str) -> dict:
        return {ds: spaces[(model, ds)].get(key) for ds in DATASETS}

    def text(vals) -> str:
        return ", ".join(fmt(v) for v in vals)

    patch = choices("medsam", "patch_size")
    if patch["drive"] != [1, 4, 8, 16] or any(patch[d] != [1, 2, 4, 8, 16] for d in ("hrf", "fives", "vessmap")):
        raise ValueError(f"Unexpected patch-size choices: {patch}")
    med_fixed = {k: choices("medsam", k) for k in ("embed_dim", "depth", "num_heads", "mlp_ratio", "window_size")}
    for k, v in med_fixed.items():
        if len({json.dumps(x) for x in v.values()}) != 1:
            raise ValueError(f"MedSAM {k} choices differ between datasets: {v}")
    lite = {k: choices("litemedsam", k) for k in ("embed_dims", "depths", "num_heads", "window_sizes", "mlp_ratio")}
    for k in ("embed_dims", "depths"):
        if len({json.dumps(x) for x in lite[k].values()}) != 1:
            raise ValueError(f"LiteMedSAM {k} choices differ between datasets.")
    for k in ("num_heads", "window_sizes", "mlp_ratio"):
        if lite[k]["drive"] is None or any(lite[k][d] is not None for d in ("hrf", "fives", "vessmap")):
            raise ValueError(f"LiteMedSAM {k} expected to be searched on DRIVE only: {lite[k]}")
    m = {k: text(v["drive"]) for k, v in med_fixed.items()}

    # DRIVE: the compact LiteMedSAM search returned LiteMedSAM's own stage widths and depths with other settings,
    # a model larger than LiteMedSAM (stated in the caption; checked here).
    lite_ref = arch("litemedsam_pretrained", "drive")
    lite_drive = arch("litemedsam_compact_scratch", "drive")
    p_ref, p_cmp = (cx.loc[(c, "drive"), "params_m"] for c in ("litemedsam_pretrained", "litemedsam_compact_scratch"))
    g_ref, g_cmp = (
        cx.loc[(c, "drive"), "gflops_per_image"] for c in ("litemedsam_pretrained", "litemedsam_compact_scratch")
    )
    if not (
        p_cmp > p_ref
        and g_cmp > g_ref
        and lite_drive["embed_dims"] == lite_ref["embed_dims"]
        and lite_drive["depths"] == lite_ref["depths"]
    ):
        raise ValueError("Compact LiteMedSAM on DRIVE is no longer larger than LiteMedSAM with its stage sizes.")
    lt = {k: " or ".join(f"({fmt(x)})" for x in v["drive"]) for k, v in lite.items() if k != "mlp_ratio"}

    tex = rf"""% Generated by paper/figures_tables/tabA2_compact_models.py -- do not edit by hand.
% Source: configs/ (selected trials and search spaces); results/cost.csv (parameters).
\begin{{table*}}[tbp]
\caption{{Selected compact models and the original \medsam{{}} and \litemedsam{{}} (same architecture fine-tuned and
from scratch). Input, Patch: px; Tokens: grid of one window; Emb.: embedding dimension; MLP: ratio of the hidden
width of the feed-forward (MLP) block to the embedding dimension; Window:
attention window; Params: M; \litemedsam{{}}: per stage. Searched for compact \medsam{{}}: patch size
{text(patch["hrf"])}
({text(patch["drive"])} on DRIVE), embedding dimension {m["embed_dim"]}, depth {m["depth"]}, heads {m["num_heads"]}, MLP
ratio {m["mlp_ratio"]}, attention window {m["window_size"]}; for compact \litemedsam{{}}: embedding dimensions
{lt["embed_dims"]}, depths {lt["depths"]}, and on DRIVE only heads {lt["num_heads"]}, attention windows
{lt["window_sizes"]} and MLP ratio {text(lite["mlp_ratio"]["drive"])}. Dice, $\thinrecall$,
$P_{{\mathrm{{thin}}}}$: compact
models (originals in Table~\ref{{tab:pretraining_gain}}).}}
\label{{tab:compact_models}}
\scriptsize
\begin{{ruledtabular}}
\begin{{tabular}}{{llccccccccrccc}}
Model & Version & Input & Patch & Tokens & Emb. & Depth & Heads & MLP & Window & Params & Dice & $\thinrecall$ &
$P_{{\mathrm{{thin}}}}$ \\
\midrule
{chr(10).join(body)}
\end{{tabular}}
\end{{ruledtabular}}
\end{{table*}}
"""
    write_tex("tabA2_compact_models", tex)


if __name__ == "__main__":
    main()
