"""Table V (tab:search_space): the hyperparameter search of every configuration and dataset, and its compute time.

For each of the 36 Optuna studies: the ranges of the parameters that reached the training run (learning rate,
weight decay, batch size, the architecture parameter of each CNN), the validation interval and patience of
the selected run, the number of trials and the GPU hours (sum of trial durations).

Sources: ``search_trials.csv`` (trial states, durations and sampled values from each study's ``optuna_study.db``,
or from ``train.log`` for the two studies without a database, VessMap LW-Net and ConvUNeXt; the settings each trial
trained with, from its ``trial_<n>/config.yaml``) and the released configurations (search spaces and the settings
of the selected trials).

``train.py`` also samples ``optimizer`` and ``patience`` but does not pass them to the training run: every
trial trained with AdamW and the configuration's patience. The script checks this on every trial and
reports the optimizer and patience as fixed.
Output: ``paper/manuscript/figures/tabA1_search_space.tex``.
"""

import yaml

from common import (
    CONFIGS,
    DATASET_LABELS,
    DATASETS,
    load_config_file,
    load_trials,
    np,
    pd,
    save_values,
    search_space,
    selected_trial,
    write_tex,
)
from thinvessel.analysis.search_records import applied_value as applied
from thinvessel.config import ARCHITECTURE_KEYS

MACROS = {
    "lwnet": r"\lwnet{}",
    "convunext": r"\convunext{}",
    "convnext": r"\convnext{}",
    "litemedsam": r"\litemedsam{}",
    "medsam": r"\medsam{}",
}
REGIME_TEXT = {"cnn": "", "finetuned": "fine-tuned", "scratch": "from scratch", "compact": "compact"}
ARCH_KEYS = set(ARCHITECTURE_KEYS)
FINISHED = {"COMPLETE", "PRUNED", "FAIL", "LOGGED"}


def power(x: float) -> str:
    e = int(round(np.log10(x)))
    if not np.isclose(x, 10.0**e):
        raise ValueError(f"{x} is not a power of ten.")
    return f"10^{{{e}}}"


def log_range(d: dict) -> str:
    a = d["attributes"]
    if not a["log"]:
        raise ValueError("Expected a log-uniform range.")
    return f"${power(a['low'])}$--${power(a['high'])}$"


def arch_text(arch: str, regime: str, space: dict) -> str:
    if regime == "compact":
        return r"Table~\ref{tab:compact_models}"
    if "layers" in space:
        ch = [yaml.safe_load(c) for c in space["layers"]["attributes"]["choices"]]
        if any(w != [w[0], 2 * w[0], 4 * w[0]] for w in ch):
            raise ValueError(f"LW-Net widths do not double across levels: {ch}")
        return "first width " + ", ".join(str(w[0]) for w in ch)
    if "base_c" in space:
        return "base width " + ", ".join(str(c) for c in space["base_c"]["attributes"]["choices"])
    if "drop_path_rate" in space:
        a = space["drop_path_rate"]["attributes"]
        return f"drop-path rate {a['low']:g}--{a['high']:g}"
    return "--"


def main() -> None:
    rows, body = [], []
    total_hours, total_trials = 0.0, 0
    for cfg, spec in CONFIGS.items():
        for k, ds in enumerate(DATASETS):
            trials = load_trials(cfg, ds)
            space = search_space(cfg, ds)
            sel = selected_trial(cfg, ds)

            # The selected trial is the finished trial with the highest validation Dice (where it is recorded).
            done = trials[trials["state"].isin(FINISHED)]
            if done["best_val_dice"].notna().any():
                best = int(done.loc[done["best_val_dice"].idxmax(), "number"])
                if best != sel:
                    raise ValueError(f"{cfg} {ds}: selected trial {sel}, best finished trial {best}.")

            # Fixed settings and applied parameters, checked on every trial that wrote a configuration.
            intervals, patience_epochs = set(), set()
            for _, t in trials[trials["applied_optimizer"].notna()].iterrows():
                if t["applied_optimizer"] != "AdamW" or int(t["applied_max_epochs"]) != 1000:
                    raise ValueError(
                        f"{cfg} {ds} trial {t['number']}: {t['applied_optimizer']}, {t['applied_max_epochs']} epochs."
                    )
                interval = int(t["applied_check_val_every_n_epoch"])
                intervals.add(interval)
                patience_epochs.add(int(t["applied_patience"]) * interval)
            if patience_epochs != {50}:
                raise ValueError(f"{cfg} {ds}: patience in epochs {patience_epochs}.")
            sel_cfg = load_config_file(cfg, ds)
            sel_row = trials[trials["number"] == sel].iloc[0]
            for key in space:
                if key in ("optimizer", "patience"):
                    continue
                sampled, used = (
                    yaml.safe_load(v) if isinstance(v, str) and v.startswith("[") else v
                    for v in (sel_row[key], applied(sel_cfg, key))
                )
                if not (np.isclose(sampled, used) if isinstance(used, float) else sampled == used):
                    raise ValueError(f"{cfg} {ds}: {key} sampled {sampled}, trained with {used}.")
            for key in space:
                if (
                    key not in ("lr", "weight_decay", "batch_size")
                    and key not in ARCH_KEYS
                    and key not in ("layers", "base_c", "drop_path_rate", "optimizer", "patience")
                ):
                    raise ValueError(f"{cfg} {ds}: unexpected parameter {key}.")
            if "LOGGED" in set(trials["state"]):  # no database: logged values must lie in the configured space
                for key, d in space.items():
                    vals = trials[key].dropna()
                    if d["name"] == "CategoricalDistribution":
                        ok = vals.isin(d["attributes"]["choices"]).all()
                    else:
                        ok = vals.between(d["attributes"]["low"], d["attributes"]["high"]).all()
                    if not ok:
                        raise ValueError(f"{cfg} {ds}: logged {key} outside the configured space.")

            n_trials = int(trials["state"].isin(FINISHED).sum())
            hours = float(trials["hours"].sum())
            total_hours += hours
            total_trials += n_trials
            sel_interval = sel_cfg["training_params"].get("check_val_every_n_epoch", 1)
            batch = ", ".join(str(b) for b in space["batch_size"]["attributes"]["choices"])
            row = {
                "configuration": cfg,
                "dataset": ds,
                "lr": log_range(space["lr"]),
                "weight_decay": log_range(space["weight_decay"]),
                "batch_size": batch,
                "architecture": arch_text(spec["arch"], spec["regime"], space),
                "val_interval_selected": sel_interval,
                "val_intervals_all": sorted(intervals),
                "trials": n_trials,
                "gpu_hours": hours,
                "selected_trial": sel,
            }
            rows.append(row)
            name = MACROS[spec["arch"]] + (f", {REGIME_TEXT[spec['regime']]}" if REGIME_TEXT[spec["regime"]] else "")
            first = rf"\multirow{{4}}{{*}}{{{name}}}" if k == 0 else ""
            body.append(
                f"{first} & {DATASET_LABELS[ds]} & {row['lr']} & {row['weight_decay']} & {batch} & "
                f"{row['architecture']} & {sel_interval} & {n_trials} & {hours:.1f} \\\\"
            )
        body.append(r"\midrule")
    body.pop()
    save_values(pd.DataFrame(rows), "tabA1_search_space")

    total_days = total_hours / 24
    print(
        f"Total: {len(rows)} studies, {total_trials} finished trials, {total_hours:.1f} GPU hours "
        f"({total_days:.1f} GPU days)."
    )
    mixed = [r for r in rows if len(r["val_intervals_all"]) > 1]
    print("Studies with mixed validation intervals:", [(r["configuration"], r["dataset"]) for r in mixed])

    tex = rf"""% Generated by paper/figures_tables/tabA1_search_space.py -- do not edit by hand.
% Source: results/search_trials.csv (exported from the Optuna studies, trial configurations and train.log by
% scripts/analyse.py) and the released configurations under configs/.
\begin{{table*}}[tbp]
\caption{{Hyperparameter search (100 trials; 101 where one failed). Learning rate and weight decay: log-uniform in the
ranges shown. CNN architecture parameter: \lwnet{{}}, width of the first of its three levels (doubled at each level);
\convunext{{}}, base width; \convnext{{}}, stochastic-depth rate. Val.: validation interval of the selected run
(epochs); Trials: finished trials; GPU~h: sum of the trial durations.}}
\label{{tab:search_space}}
\footnotesize
\begin{{ruledtabular}}
\begin{{tabular}}{{llcccccrr}}
Configuration & Dataset & Learning rate & Weight decay & Batch size & Architecture & Val. & Trials & GPU~h \\
\midrule
{chr(10).join(body)}
\end{{tabular}}
\end{{ruledtabular}}
\end{{table*}}
"""
    write_tex("tabA1_search_space", tex)


if __name__ == "__main__":
    main()
