"""Wilcoxon signed-rank tests of the paper (Section 3.4, Appendix C).

Two-sided Wilcoxon signed-rank test on the per-image paired differences (SciPy, ``method="auto"``: exact null
distribution up to 50 pairs without zero differences, normal approximation otherwise; zero differences discarded),
with the Holm correction inside each family, alpha = 0.05. The direction of a test is the sign of the Hodges-Lehmann
estimate (median of the Walsh averages of the differences). An image with an undefined value is left out of the pairs
that involve it.

Families:
  F1  per dataset x metric (Dice, thin-vessel recall, thin-vessel precision): the 36 pairs of the 9 configurations.
  F2  per dataset x metric (centerline recall, centerline precision; Table II only): the 10 pairs of the 5 models.
  F3  padded vs upscaled MedSAM (Table IV): 2 models x 2 datasets x 3 metrics = 12 tests, one family.

MedSAM fine-tuned and from scratch on DRIVE and VessMap use the upscaled protocol; the padded (native) protocol
enters only F3.

Input: ``results/per_image_metrics.csv`` (``--per-image``). Output: ``all_tests.csv`` and ``summary.csv`` in
``$OUTPUTS_DIR/statistics/``, default ``outputs/statistics/`` (``--output-dir``).

    python paper/statistics/paper_wilcoxon.py
"""

import argparse
import os
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from thinvessel.config import all_studies
from thinvessel.evaluation.scoring import paper_protocol

HERE = Path(__file__).resolve().parent
ALPHA = 0.05
DATASETS = ["drive", "hrf", "fives", "vessmap"]
MAIN_METRICS = ["dice", "thin_vessel_recall", "thin_vessel_precision"]
CL_METRICS = ["centerline_recall", "centerline_precision"]

NAMES = {  # configuration -> name in the tables
    "lwnet_scratch": "LW-Net",
    "convunext_scratch": "ConvUNeXt",
    "convnext_scratch": "ConvNeXt",
    "litemedsam_pretrained": "LiteMedSAM FT",
    "litemedsam_scratch": "LiteMedSAM scratch",
    "litemedsam_compact_scratch": "compact LiteMedSAM",
    "medsam_pretrained": "MedSAM FT",
    "medsam_scratch": "MedSAM scratch",
    "medsam_compact_scratch": "compact MedSAM",
}
ORDER = list(NAMES.values())
FIVE = ["LW-Net", "ConvUNeXt", "ConvNeXt", "LiteMedSAM FT", "MedSAM FT"]


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-adjusted p-values (step-down, monotone, capped at 1)."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.maximum.accumulate((m - np.arange(m)) * p[order]).clip(max=1.0)
    out = np.empty(m)
    out[order] = adj
    return out


def hodges_lehmann(d: np.ndarray) -> float:
    """Median of the Walsh averages (d_i + d_j) / 2, i <= j."""
    i, j = np.triu_indices(len(d))
    return float(np.median((d[i] + d[j]) / 2))


def rank_biserial(d: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation of the non-zero differences."""
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def paper_data(per_image: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-image values under the paper's protocol, and the padded rows of MedSAM on DRIVE and VessMap (F3)."""
    df = pd.read_csv(per_image)
    paper = {(s.dataset, s.configuration): paper_protocol(s) for s in all_studies()}
    proto = pd.Series([paper[(d, c)] for d, c in zip(df["dataset"], df["configuration"])], index=df.index)
    data = df[df["protocol"] == proto].copy()
    padded = df[(proto == "upscaled") & (df["protocol"] == "native")].copy()
    data["model"] = data["configuration"].map(NAMES)
    padded["model"] = padded["configuration"].map(NAMES)
    return data, padded


def test_pair(x: pd.Series, y: pd.Series) -> Dict[str, float]:
    pair = pd.concat([x, y], axis=1, keys=["a", "b"]).dropna()
    d = (pair["a"] - pair["b"]).to_numpy()
    w = stats.wilcoxon(d, alternative="two-sided", method="auto")
    return dict(
        n=len(d),
        mean_a=pair["a"].mean(),
        mean_b=pair["b"].mean(),
        mean_diff=d.mean(),
        median_diff=float(np.median(d)),
        hodges_lehmann=hodges_lehmann(d),
        rank_biserial=rank_biserial(d),
        a_better=int((d > 0).sum()),
        b_better=int((d < 0).sum()),
        ties=int((d == 0).sum()),
        W=w.statistic,
        p_raw=w.pvalue,
    )


def finish(df: pd.DataFrame, by: Optional[List[str]]) -> pd.DataFrame:
    df["p_holm"] = df.groupby(by)["p_raw"].transform(holm) if by else holm(df["p_raw"].to_numpy())
    df["significant"] = df["p_holm"] < ALPHA
    df["significant_uncorrected"] = df["p_raw"] < ALPHA
    df["verdict"] = np.where(
        ~df["significant"], "not significantly different", np.where(df["hodges_lehmann"] > 0, "A higher", "B higher")
    )
    # a significant verdict whose direction disagrees with the difference of means (heavy outlier images)
    df["direction_vs_means"] = np.where(
        df["significant"] & (np.sign(df["hodges_lehmann"]) != np.sign(df["mean_diff"])), "DISAGREES", ""
    )
    return df


def pairwise(
    data: pd.DataFrame, metrics: List[str], models: List[str], family: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows, summ = [], []
    for ds in DATASETS:
        for metric in metrics:
            piv = data[data["dataset"] == ds].pivot(index="sample_id", columns="model", values=metric)[models]
            order = piv.mean().sort_values(ascending=False).index.tolist()  # best mean first, as the tables
            complete = piv.dropna()
            fr = stats.friedmanchisquare(*[complete[c] for c in models])
            for a, b in combinations(models, 2):
                rows.append(dict(family=family, dataset=ds, metric=metric, A=a, B=b, **test_pair(piv[a], piv[b])))
            for c in order:
                summ.append(
                    dict(
                        family=family,
                        dataset=ds,
                        metric=metric,
                        model=c,
                        n=int(piv[c].notna().sum()),
                        mean=piv[c].mean(),
                        sd=piv[c].std(),
                        median=piv[c].median(),
                        min=piv[c].min(),
                        friedman_p=fr.pvalue,
                    )
                )
    return finish(pd.DataFrame(rows), ["dataset", "metric"]), pd.DataFrame(summ)


def padding(data: pd.DataFrame, padded: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in ["MedSAM FT", "MedSAM scratch"]:
        for ds in ["drive", "vessmap"]:
            up = data[(data.model == model) & (data.dataset == ds)].set_index("sample_id")
            pa = padded[(padded.model == model) & (padded.dataset == ds)].set_index("sample_id")
            for metric in MAIN_METRICS:
                rows.append(
                    dict(
                        family="F3 padding",
                        dataset=ds,
                        metric=metric,
                        A=f"{model} upscaled",
                        B=f"{model} padded",
                        **test_pair(up[metric], pa[metric]),
                    )
                )
    return finish(pd.DataFrame(rows), None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-image", type=Path, default=HERE.parents[1] / "results" / "per_image_metrics.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("OUTPUTS_DIR", HERE.parents[1] / "outputs")) / "statistics",
    )
    args = parser.parse_args()
    data, padded = paper_data(args.per_image)
    f1, s1 = pairwise(data, MAIN_METRICS, ORDER, "F1 nine configurations")
    f2, s2 = pairwise(data, CL_METRICS, FIVE, "F2 five models, overall centerline")
    f3 = padding(data, padded)
    tests = pd.concat([f1, f2, f3], ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tests.to_csv(args.output_dir / "all_tests.csv", index=False)
    pd.concat([s1, s2], ignore_index=True).to_csv(args.output_dir / "summary.csv", index=False)
    print(
        f"{len(tests)} tests; significant after Holm: {int(tests.significant.sum())}; "
        f"verdict changes without Holm: {int((tests.significant != tests.significant_uncorrected).sum())}; "
        f"direction disagreeing with means: {int((tests.direction_vs_means == 'DISAGREES').sum())}"
    )


if __name__ == "__main__":
    main()
