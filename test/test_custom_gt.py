"""
Custom CelebA ground-truth generator verification.

Unit-checks the per-image protocol on a synthetic toy corpus with hand-computed valid
sets: strict ± target satisfaction, the Hamming ≤ 2 identity radius, the violate-every-
constraint SOURCE policy, the ≥5-target viability floor, and self-exclusion. The frozen
JSON is checked for eval-engine compatibility, and the exact-reconstruction gate against
the instructor's JSON runs as an integration test when local artifacts exist.

Run:  pytest test/test_custom_gt.py
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import generate_custom_gt as gg


def _toy_corpus():
    # 8 images × 5 attrs. attr0 is the query axis; attrs 1..4 are the identity axes.
    # With query "+attr0" (free = 1..4), from source row 5 (attr0 = 0, policy-valid):
    #   rows 0,1 in (Hamming 0), 2 in (1), 3 in (2, boundary), 4 out (Hamming 3),
    #   rows 6,7 out (strict: attr0 = 0).
    return torch.tensor([
        [1, 0, 0, 0, 0],   # 0 — target, Hamming 0 from row 5
        [1, 0, 0, 0, 0],   # 1 — target, Hamming 0
        [1, 1, 0, 0, 0],   # 2 — target, Hamming 1
        [1, 1, 1, 0, 0],   # 3 — target, Hamming 2 (boundary, inside)
        [1, 1, 1, 1, 0],   # 4 — outside radius (Hamming 3)
        [0, 0, 0, 0, 0],   # 5 — the source (violates +attr0, as policy requires)
        [0, 1, 0, 0, 1],   # 6 — fails strict +attr0
        [0, 0, 0, 0, 1],   # 7 — fails strict +attr0
    ], dtype=torch.float32)


def test_strict_and_hamming_boundary():
    A = _toy_corpus()
    pi, ni, free = [0], [], [1, 2, 3, 4]
    gt = gg.targets_for_sources(A, pi, ni, free, [5])
    assert gt["5"] == [0, 1, 2, 3]      # Hamming ≤ 2 in; Hamming 3 and non-strict out


def test_negative_constraint_excludes():
    A = _toy_corpus()
    # "-attr4": rows 6,7 strictly excluded as targets; free = 0..3.
    pi, ni, free = [], [4], [0, 1, 2, 3]
    gt = gg.targets_for_sources(A, pi, ni, free, [6])   # row 6 has attr4=1: policy-valid
    assert 7 not in gt["6"]
    assert 2 in gt["6"]                                  # ham 0 on free columns


def test_source_policy_excludes_satisfying_sources(monkeypatch):
    # Sources already satisfying any constraint are twin-retrieval, not compositional
    # edits — the policy must exclude them even when they have plenty of targets.
    A = _toy_corpus()
    monkeypatch.setattr(gg, "query_indices", lambda q: ([0], [], [1, 2, 3, 4]))
    monkeypatch.setattr(gg, "MIN_TARGETS", 2)
    gt = gg.build_ground_truth("+attr0", A)
    assert all(A[int(s), 0] == 0 for s in gt)            # every source violates +attr0
    assert "0" not in gt and "1" not in gt               # attr0=1 rows excluded as sources


def test_source_never_its_own_target():
    A = _toy_corpus()
    pi, ni, free = [0], [], [1, 2, 3, 4]
    gt = gg.targets_for_sources(A, pi, ni, free, [0, 5])
    for src, targets in gt.items():
        assert int(src) not in targets


def test_full_ground_truth_is_uncapped_and_floored(monkeypatch):
    # The REAL ground truth: every policy-valid source with ≥ MIN_TARGETS appears, with
    # its complete target list — no subsampling, no caps.
    A = torch.cat([torch.ones(7, 5), torch.zeros(9, 5)])     # 7 strict targets, 9 sources
    monkeypatch.setattr(gg, "query_indices", lambda q: ([0], [], [1, 2, 3, 4]))
    gt = gg.build_ground_truth("+attr0", A)
    assert len(gt) == 0                                       # Hamming 4 > 2: nothing valid
    A2 = torch.cat([torch.ones(7, 1), torch.zeros(7, 4)], dim=1)  # targets at Hamming 0
    A2 = torch.cat([A2, torch.zeros(9, 5)])
    gt = gg.build_ground_truth("+attr0", A2)
    assert len(gt) == 9                                       # ALL valid sources kept
    assert all(len(t) == 7 for t in gt.values())              # FULL target lists kept


def test_min_targets_floor_drops_sources(monkeypatch):
    # 3 valid targets < MIN_TARGETS=5 ⇒ the source is dropped (spec §3.1.2 floor).
    A = torch.cat([torch.ones(3, 1), torch.zeros(3, 4)], dim=1)   # 3 strict targets
    A = torch.cat([A, torch.zeros(4, 5)])                          # 4 candidate sources
    monkeypatch.setattr(gg, "query_indices", lambda q: ([0], [], [1, 2, 3, 4]))
    assert gg.build_ground_truth("+attr0", A) == {}


def test_generated_json_is_eval_compatible():
    # The frozen custom JSON must load and parse through the project's own eval engine.
    from eval import find_eval_json, load_eval_json, parse_query
    json_path = find_eval_json().parent / "celeba_custom_evaluation.json"
    if not json_path.exists():
        pytest.skip("celeba_custom_evaluation.json not generated yet "
                    "(run src/generate_custom_gt.py)")
    gt = load_eval_json(json_path)
    assert len(gt) >= 1
    for entry in gt:
        assert set(entry.keys()) == {"query", "ground_truth"}
        pos, neg = parse_query(entry["query"])          # must parse without error
        assert pos or neg
        for src, targets in entry["ground_truth"].items():
            assert int(src) >= 0
            assert len(targets) >= gg.MIN_TARGETS
            assert int(src) not in targets              # source excluded from its targets


def test_official_json_exact_reconstruction():
    # Integration credibility gate (first 3 queries for test speed; the CLI runs all 14):
    # the generator must reproduce the instructor's JSON IDENTICALLY from the DB.
    try:
        from data_loader import load_attributes
        A = load_attributes()
    except (FileNotFoundError, RuntimeError):
        pytest.skip("cached attribute tensor not available")
    assert gg.verify_official(A, limit=3) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
