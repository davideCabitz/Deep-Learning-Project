"""
stats_eval — bootstrap statistics suite over per-source dumps (per_source.py).

Single owner of every statistical claim in the report. All inference is bootstrap-based,
by deliberate methodological choice:

  * Per-source hit@K is BINARY. Rank tests (Wilcoxon signed-rank) are ill-posed on
    Bernoulli paired differences — ties and zero-differences are discarded, the tested
    median is degenerate, and asymptotic assumptions fail. The paired bootstrap uses every
    observation, assumes nothing about the distribution, and returns effect size and
    significance from one coherent framework.
  * The 13 benchmark queries are a FIXED benchmark (spec §3.1), not a random sample of a
    query population — so queries are strata and are never resampled; only sources
    (the true random unit, spec §3.1.1 "≥ 5 valid ground-truth targets") are.

Metrics are spec-locked: Recall@K (hit rate) and Precision@K at K ∈ {1, 5, 10} only
(spec §3.1.3). Everything here is a *statistic about* those metrics, never a new metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eval import parse_query
from per_source import KS, TOP_KEEP

_ALPHA = 0.05          # reported CIs are 95% (two-sided) throughout
DEFAULT_B = 1000


def _metric_fields(ks=KS) -> list[str]:
    return [f"hit{k}" for k in ks] + [f"prec{k}" for k in ks]


def metric_label(field: str) -> str:
    # "hit5" -> "R@5", "prec10" -> "P@10" — the spec §3.1.3 names used in every table.
    return ("R@" + field[3:]) if field.startswith("hit") else ("P@" + field[4:])


def _check_paired(a: dict, b: dict) -> None:
    # Paired inference is only valid on the identical source population (same GT JSON).
    if list(a) != list(b):
        raise ValueError("dumps cover different query sets — pairing invalid")
    for q in a:
        if not np.array_equal(a[q]["sources"], b[q]["sources"]):
            raise ValueError(f"source sets differ for query {q!r} — pairing invalid")


_CHUNK = 1000    # bootstrap replicates per resampling block — bounds peak memory at
                 # chunk × max n_q indices (~44 MB) regardless of how large B grows


def _bootstrap_means(rng: np.random.Generator, arrays: list[np.ndarray],
                     B: int) -> list[np.ndarray]:
    # Resampled means for several metric arrays of the SAME source population, sharing
    # one index draw per replicate (the sharing is what makes paired tests paired).
    # Chunked so B = 10⁴⁺ stays memory-safe on the largest queries (n = 5469).
    n = len(arrays[0])
    out = [np.empty(B) for _ in arrays]
    for lo in range(0, B, _CHUNK):
        hi = min(lo + _CHUNK, B)
        idx = rng.integers(0, n, size=(hi - lo, n))
        for a, dest in zip(arrays, out):
            dest[lo:hi] = a[idx].mean(axis=1)
    return out


@dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap distribution of the macro-mean of one metric: point estimate,
    bootstrap mean/std and the percentile 95% CI, plus the raw B replicates."""
    point: float
    mean: float
    std: float
    ci_lo: float
    ci_hi: float
    samples: np.ndarray


# Stratified bootstrap CI, B=1000 — the uncertainty statement for every reported mean.
# Per replicate: resample sources WITH replacement within each query (strata fixed),
# average per query, macro-average across queries — the exact statistic of the CSV MEAN
# row, so the CI brackets the published number. Percentile CI (adequate at n=27…5469
# sources per query; BCa refinement changes nothing at this sample size).
def stratified_bootstrap(dump: dict, B: int = DEFAULT_B, seed: int = 0,
                         ks=KS) -> dict[str, BootstrapResult]:
    rng = np.random.default_rng(seed)
    fields = _metric_fields(ks)
    per_query = {f: [] for f in fields}          # each entry: [B] bootstrap means
    point = {f: [] for f in fields}
    for q, arrays in dump.items():
        vals = [arrays[f].astype(np.float64) for f in fields]
        means = _bootstrap_means(rng, vals, B)
        for f, v, m in zip(fields, vals, means):
            per_query[f].append(m)
            point[f].append(v.mean())
    out = {}
    for f in fields:
        macro = np.mean(per_query[f], axis=0)                    # [B]
        lo, hi = np.percentile(macro, [100 * _ALPHA / 2, 100 * (1 - _ALPHA / 2)])
        out[f] = BootstrapResult(float(np.mean(point[f])), float(macro.mean()),
                                 float(macro.std(ddof=1)), float(lo), float(hi), macro)
    return out


@dataclass(frozen=True)
class DiffResult:
    """Paired-difference bootstrap for Δ = macro(A) − macro(B): effect size (absolute Δ
    with CI, relative gain vs B), and a two-sided bootstrap p-value."""
    delta: float
    rel_gain: float
    ci_lo: float
    ci_hi: float
    p_value: float
    samples: np.ndarray


# Paired significance test — the SAME resampled source indices are applied to both
# models (pairing removes between-source variance from Δ, which is what makes the test
# powerful on binary outcomes). p = 2·min(P(Δ*≤0), P(Δ*≥0)) with the (c+1)/(B+1)
# small-sample correction, so p is never optimistically zero. Significant iff the 95% CI
# of Δ excludes 0 — the CI and the p-value can never contradict each other by design.
def paired_bootstrap_diff(a: dict, b: dict, B: int = DEFAULT_B, seed: int = 0,
                          ks=KS) -> dict[str, DiffResult]:
    _check_paired(a, b)
    rng = np.random.default_rng(seed)
    fields = _metric_fields(ks)
    deltas = {f: [] for f in fields}
    point_a, point_b = {f: [] for f in fields}, {f: [] for f in fields}
    for q in a:
        vals = [a[q][f].astype(np.float64) for f in fields] + \
               [b[q][f].astype(np.float64) for f in fields]
        means = _bootstrap_means(rng, vals, B)                   # shared idx — the pairing
        for j, f in enumerate(fields):
            deltas[f].append(means[j] - means[j + len(fields)])
            point_a[f].append(vals[j].mean())
            point_b[f].append(vals[j + len(fields)].mean())
    out = {}
    for f in fields:
        d = np.mean(deltas[f], axis=0)                           # [B] macro Δ replicates
        lo, hi = np.percentile(d, [100 * _ALPHA / 2, 100 * (1 - _ALPHA / 2)])
        c_le, c_ge = int((d <= 0).sum()), int((d >= 0).sum())
        p = 2 * min((c_le + 1) / (B + 1), (c_ge + 1) / (B + 1))
        pa, pb = float(np.mean(point_a[f])), float(np.mean(point_b[f]))
        rel = (pa - pb) / pb if pb > 0 else float("inf")
        out[f] = DiffResult(pa - pb, rel, float(lo), float(hi), min(p, 1.0), d)
    return out


# Holm–Bonferroni step-down — controls family-wise error across the {metrics × model
# pairs} family without Bonferroni's full conservativeness. Input/output keyed dicts.
def holm_bonferroni(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running_max = {}, 0.0
    for rank, (key, p) in enumerate(items):
        running_max = max(running_max, min(1.0, (m - rank) * p))
        adjusted[key] = running_max
    return {k: adjusted[k] for k in pvals}


# ---------------------------------------------------------------------------
# Supplementary statistics on R/P (same dumps, no new metrics)
# ---------------------------------------------------------------------------

# Query taxonomy — the axis of the compositional-robustness analysis. Categories follow
# the spec's own complexity levels (§3.1) with pure negation split out because it is the
# hard case the project targets (fusion of subtractive constraints).
def query_category(query_str: str) -> str:
    pos, neg = parse_query(query_str)
    if len(pos) == 1 and not neg:
        return "simple positive"
    if len(neg) == 1 and not pos:
        return "simple negative"
    if neg and not pos:
        return "pure negation"
    return "composed"


def category_breakdown(dump: dict, ks=KS) -> dict[str, dict[str, float]]:
    # Macro R@K/P@K per query category (each query weighs 1/|category|, spec convention).
    cats: dict[str, list[dict]] = {}
    for q, arrays in dump.items():
        cats.setdefault(query_category(q), []).append(arrays)
    return {
        cat: {metric_label(f): float(np.mean([a[f].mean() for a in arrs]))
              for f in _metric_fields(ks)}
        for cat, arrs in cats.items()
    }


def win_tie_loss(a: dict, b: dict, k: int = 10) -> dict:
    # Disaggregate WHERE the gain comes from. Per query: does A's mean beat B's?
    # Per source (paired, K fixed): 2×2 contingency both/only-A/only-B/neither — shows
    # whether A fixes B's failures or merely trades sources.
    _check_paired(a, b)
    wins = ties = losses = 0
    cont = {"both": 0, "only_a": 0, "only_b": 0, "neither": 0}
    for q in a:
        ma, mb = a[q][f"hit{k}"].mean(), b[q][f"hit{k}"].mean()
        wins, ties, losses = wins + (ma > mb), ties + (ma == mb), losses + (ma < mb)
        ha, hb = a[q][f"hit{k}"].astype(bool), b[q][f"hit{k}"].astype(bool)
        cont["both"] += int((ha & hb).sum())
        cont["only_a"] += int((ha & ~hb).sum())
        cont["only_b"] += int((~ha & hb).sum())
        cont["neither"] += int((~ha & ~hb).sum())
    return {"per_query": {"win": wins, "tie": ties, "loss": losses},
            "per_source@k": cont}


def macro_vs_micro(dump: dict, ks=KS) -> dict[str, dict[str, float]]:
    # Macro (each query 1/13 — the spec/CSV convention) vs micro (each source equal).
    # num_sources spans 27…5469, so agreement between the two proves the headline number
    # is not an artifact of the averaging convention.
    fields = _metric_fields(ks)
    macro = {f: np.mean([a[f].mean() for a in dump.values()]) for f in fields}
    micro = {f: np.concatenate([a[f] for a in dump.values()]).mean() for f in fields}
    return {"macro": {metric_label(f): float(macro[f]) for f in fields},
            "micro": {metric_label(f): float(micro[f]) for f in fields}}


def recall_curve(dump: dict, gt_list: list[dict], ks=range(1, TOP_KEEP + 1)) -> np.ndarray:
    # Macro Recall@K for EVERY K = 1…10 from the stored top-10 prefixes — locates where
    # in the ranking the gain materializes, at zero extra scoring cost.
    gt_by_query = {e["query"]: e["ground_truth"] for e in gt_list}
    curves = []
    for q, arrays in dump.items():
        gt = {int(s): {int(t) for t in ts} for s, ts in gt_by_query[q].items()}
        top = arrays[f"top{TOP_KEEP}"]
        hits = np.array([[len(set(top[i, :k]) & gt[int(s)]) > 0 for k in ks]
                         for i, s in enumerate(arrays["sources"])], dtype=np.float64)
        curves.append(hits.mean(axis=0))
    return np.mean(curves, axis=0)                               # [len(ks)] macro R@K


# ---------------------------------------------------------------------------
# Plots — matplotlib only, headless-safe (visualize.py conventions)
# ---------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_ci_bars(results_by_model: dict[str, dict[str, BootstrapResult]],
                 fields=("hit1", "hit5", "hit10"), path=None, title="Recall@K with 95% CI"):
    # Grouped bars + 95% CI error bars — the headline comparison figure. Error bars are
    # the bootstrap percentile CI, not ±std, so overlap reads directly as significance.
    plt = _plt()
    models = list(results_by_model)
    x = np.arange(len(fields))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(models):
        res = [results_by_model[m][f] for f in fields]
        pts = [r.point for r in res]
        err = np.array([[r.point - r.ci_lo for r in res], [r.ci_hi - r.point for r in res]])
        ax.bar(x + i * width, pts, width, yerr=err, capsize=3, label=m)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([metric_label(f) for f in fields])
    ax.set_ylabel("macro mean"), ax.set_title(title), ax.legend(fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig


def plot_bootstrap_violin(results_by_model: dict[str, dict[str, BootstrapResult]],
                          field="hit1", path=None, title=None):
    # Violin of the B bootstrap replicates per model — the margin of improvement shown
    # as full distributions: separated violins are a visual significance statement.
    plt = _plt()
    models = list(results_by_model)
    data = [results_by_model[m][field].samples for m in models]
    fig, ax = plt.subplots(figsize=(9, 5))
    parts = ax.violinplot(data, showmedians=True)
    ax.boxplot(data, widths=0.08, showfliers=False)
    ax.set_xticks(range(1, len(models) + 1))
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(f"bootstrap macro {metric_label(field)}")
    ax.set_title(title or f"Bootstrap distribution of {metric_label(field)} (B={len(data[0])})")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig
