"""
stats_eval verification — synthetic Bernoulli dumps, no artifacts.

Load-bearing checks: #1 (CI covers the true mean and shrinks ~1/√n), #2 (paired test:
identical models → p ≈ 1; a forced improvement → p → small and CI excludes 0), #3
(Holm ordering/monotonicity), #4 (taxonomy + supplementary statistics on known inputs).

Run:  python test/test_stats_eval.py   (or: pytest test/test_stats_eval.py)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "Baselines"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "Training-Free"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "Trained"))

from stats_eval import (category_breakdown, holm_bonferroni, macro_vs_micro,
                        paired_bootstrap_diff, query_category, recall_curve,
                        stratified_bootstrap, win_tie_loss)


def _fake_dump(p_hit: float, n: int, seed: int, queries=("+A", "-B")) -> dict:
    # One synthetic dump: Bernoulli(p_hit) hits, precision = hit * 0.1, sources 0..n-1.
    rng = np.random.default_rng(seed)
    dump = {}
    for qi, q in enumerate(queries):
        hits = (rng.random(n) < p_hit).astype(np.uint8)
        dump[q] = {"sources": np.arange(n, dtype=np.int64) + qi * n,
                   "top10": np.zeros((n, 10), dtype=np.int64)}
        for k in (1, 5, 10):
            dump[q][f"hit{k}"] = hits
            dump[q][f"prec{k}"] = hits.astype(np.float32) * 0.1
    return dump


def test_ci_covers_and_shrinks():
    big = stratified_bootstrap(_fake_dump(0.3, 4000, 1), B=500, seed=0)["hit5"]
    small = stratified_bootstrap(_fake_dump(0.3, 250, 1), B=500, seed=0)["hit5"]
    assert big.ci_lo < 0.3 < big.ci_hi
    width_big, width_small = big.ci_hi - big.ci_lo, small.ci_hi - small.ci_lo
    ratio = width_small / width_big                 # expect ≈ sqrt(4000/250) = 4
    assert 2.5 < ratio < 6.0, ratio


def test_paired_diff_null_and_signal():
    base = _fake_dump(0.3, 2000, 2)
    null = paired_bootstrap_diff(base, base, B=500, seed=0)["hit10"]
    assert null.p_value > 0.9 and null.delta == 0.0

    # A = B plus 10% extra guaranteed hits — an unambiguous improvement signal.
    better = {}
    for q, arrs in base.items():
        extra = (np.arange(len(arrs["sources"])) % 10 == 0)
        better[q] = dict(arrs)
        for k in (1, 5, 10):
            better[q][f"hit{k}"] = np.maximum(arrs[f"hit{k}"], extra).astype(np.uint8)
    sig = paired_bootstrap_diff(better, base, B=500, seed=0)["hit10"]
    assert sig.p_value < 0.05 and sig.ci_lo > 0 and sig.delta > 0.05


def test_holm():
    adj = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == 0.03                          # 3 * 0.01
    assert adj["b"] >= adj["c"] and all(0 < v <= 1 for v in adj.values())


def test_taxonomy_and_supplementary():
    assert query_category("+Smiling") == "simple positive"
    assert query_category("-Young") == "simple negative"
    assert query_category("-Male, -Mustache") == "pure negation"
    assert query_category("+Chubby, -Young") == "composed"

    dump = _fake_dump(0.5, 100, 3)
    cats = category_breakdown(dump)
    assert set(cats) == {"simple positive", "simple negative"}

    mm = macro_vs_micro(dump)
    assert abs(mm["macro"]["R@5"] - mm["micro"]["R@5"]) < 1e-12   # equal n per query

    wtl = win_tie_loss(dump, dump)
    assert wtl["per_query"]["tie"] == 2
    assert wtl["per_source@k"]["only_a"] == 0 == wtl["per_source@k"]["only_b"]

    # recall_curve: top10 row [s+1, ...zeros]; GT = {s+1} → hit at every K ≥ 1.
    n = 20
    dump_rc = {"+A": {"sources": np.arange(n, dtype=np.int64),
                      "top10": np.tile(np.arange(10, dtype=np.int64), (n, 1))}}
    dump_rc["+A"]["top10"][:, 0] = np.arange(n) + 100
    gt_list = [{"query": "+A", "ground_truth": {str(s): [int(s) + 100] for s in range(n)}}]
    curve = recall_curve(dump_rc, gt_list)
    np.testing.assert_allclose(curve, np.ones(10))


if __name__ == "__main__":
    for fn in (test_ci_covers_and_shrinks, test_paired_diff_null_and_signal, test_holm,
               test_taxonomy_and_supplementary):
        fn()
        print(f"  PASS {fn.__name__}")
    print("all stats_eval tests passed")
