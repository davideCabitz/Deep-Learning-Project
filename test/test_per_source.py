"""
per_source verification — synthetic ground truth, no artifacts.

Load-bearing checks: #1 (per-source atoms equal known hand-computed values), #2 (means of
the dump reproduce eval.evaluate_query bit-for-bit — the parity the whole statistics suite
rests on), #3 (npz round-trip is lossless), #4 (parity gate actually fails on a mismatch).

Run:  python test/test_per_source.py   (or: pytest test/test_per_source.py)
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval import evaluate_query
from per_source import (check_parity_with_csv, dump_all, load_per_source,
                        per_source_outcomes, save_per_source_npz)

# Two sources with fixed rankings: src 0 hits at rank 1, src 5 hits first at rank 7.
GT = {"0": [10, 11], "5": [20]}
RANKINGS = {
    0: [10, 1, 2, 3, 4, 6, 7, 8, 9, 11],       # hit@1; 2 GT in top-10
    5: [1, 2, 3, 4, 6, 7, 20, 8, 9, 10],       # miss@1/@5, hit@10
}
get_ranking = RANKINGS.__getitem__


def test_known_values():
    out = per_source_outcomes(GT, get_ranking)
    assert out["sources"].tolist() == [0, 5]
    assert out["hit1"].tolist() == [1, 0]
    assert out["hit5"].tolist() == [1, 0]
    assert out["hit10"].tolist() == [1, 1]
    np.testing.assert_allclose(out["prec1"], [1.0, 0.0])
    np.testing.assert_allclose(out["prec5"], [0.2, 0.0])
    np.testing.assert_allclose(out["prec10"], [0.2, 0.1])
    assert out["top10"].shape == (2, 10)


def test_means_match_evaluate_query():
    out = per_source_outcomes(GT, get_ranking)
    ref = evaluate_query(GT, get_ranking)
    for k in (1, 5, 10):
        assert abs(out[f"hit{k}"].mean() - ref[f"recall@{k}"]) < 1e-12
        assert abs(out[f"prec{k}"].mean() - ref[f"precision@{k}"]) < 1e-7


def test_npz_roundtrip():
    gt_list = [{"query": "+Fake, -Attr", "ground_truth": GT}]
    dump = dump_all(gt_list, lambda q: get_ranking)
    with tempfile.TemporaryDirectory() as td:
        path = save_per_source_npz("unit", dump, Path(td) / "unit.npz")
        loaded = load_per_source(path)
    assert list(loaded) == ["+Fake, -Attr"]
    for field, arr in dump["+Fake, -Attr"].items():
        np.testing.assert_array_equal(loaded["+Fake, -Attr"][field], arr)


def test_parity_gate_fails_on_mismatch():
    dump = dump_all([{"query": "q", "ground_truth": GT}], lambda q: get_ranking)
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "fake.csv"
        csv_path.write_text(
            "query,R@1,R@5,R@10,P@1,P@5,P@10,num_sources\n"
            "q,0.5000,0.5000,1.0000,0.5000,0.1000,0.9999,2\n")  # P@10 wrong on purpose
        try:
            check_parity_with_csv(dump, csv_path)
        except AssertionError as e:
            assert "P@10" in str(e)
        else:
            raise AssertionError("parity gate did not fire on a corrupted CSV")


def test_parity_gate_passes_on_truth():
    dump = dump_all([{"query": "q", "ground_truth": GT}], lambda q: get_ranking)
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "true.csv"
        csv_path.write_text(
            "query,R@1,R@5,R@10,P@1,P@5,P@10,num_sources\n"
            "q,0.5000,0.5000,1.0000,0.5000,0.1000,0.1500,2\n")
        check_parity_with_csv(dump, csv_path)


if __name__ == "__main__":
    for fn in (test_known_values, test_means_match_evaluate_query, test_npz_roundtrip,
               test_parity_gate_fails_on_mismatch, test_parity_gate_passes_on_truth):
        fn()
        print(f"  PASS {fn.__name__}")
    print("all per_source tests passed")
