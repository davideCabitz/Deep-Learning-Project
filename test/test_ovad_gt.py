"""
OVAD ground-truth generator verification.

Unit checks the masked relative-Hamming rule on synthetic tri-state toy matrices
(unlabeled -1 entries excluded from both numerator and denominator, strict ± satisfaction
and source-violation both require LABELED values), the ≥5-targets viability floor, and
that the produced JSON is consumable by eval.load_eval_json / parse_query with exact
(uncapped) target sets.

Run:  pytest test/test_ovad_gt.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ovad import generate_gt as gg


def test_masked_hamming_rate_excludes_unlabeled():
    # Pair-wise mask: an attribute counts only when BOTH sides are labeled.
    # src vs cand: attrs (1,1) match, (0,1) mismatch, (-1,1) and (0,-1) masked out
    # -> 1 mismatch / 2 valid = 0.5.
    src = torch.tensor([[1, 0, -1, 0]], dtype=torch.int8)
    cand = torch.tensor([[1, 1, 1, -1]], dtype=torch.int8)
    rate = gg._masked_hamming_rate_batch(src, cand)
    assert rate.shape == (1, 1)
    assert rate[0, 0].item() == 0.5


def test_masked_hamming_rate_inf_when_no_valid_comparisons():
    # Zero jointly-labeled attributes -> +inf (never a valid target), not 0 (which would
    # silently make fully-unlabeled pairs perfect matches).
    src = torch.tensor([[-1, -1, 0]], dtype=torch.int8)
    cand = torch.tensor([[1, 0, -1]], dtype=torch.int8)
    rate = gg._masked_hamming_rate_batch(src, cand)
    assert rate[0, 0].item() == float("inf")


def test_masked_hamming_rate_batch_shape_and_values():
    # [S=2, M=3] all-pairs form; identical labeled rows -> 0.0.
    src = torch.tensor([[1, 0], [0, 1]], dtype=torch.int8)
    cand = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.int8)
    rate = gg._masked_hamming_rate_batch(src, cand)
    assert rate.shape == (2, 3)
    assert rate[0].tolist() == [0.0, 1.0, 0.5]
    assert rate[1].tolist() == [1.0, 0.0, 0.5]


def test_strictly_satisfies_requires_labeled_values():
    # Clause 1 is tri-state aware: -1 on a queried attribute proves nothing -> excluded.
    attrs = torch.tensor([
        [1, 0],    # satisfies +a0, -a1
        [1, -1],   # a1 unlabeled -> cannot verify -a1
        [-1, 0],   # a0 unlabeled -> cannot verify +a0
        [0, 0],    # violates +a0
    ], dtype=torch.int8)
    mask = gg._strictly_satisfies(attrs, pi=[0], ni=[1])
    assert mask.tolist() == [True, False, False, False]


def test_source_violates_requires_labeled_values():
    # Clause 0 is tri-state aware: "violates +a" needs labeled-negative 0, not merely !=1.
    attrs = torch.tensor([
        [0, 1],    # violates +a0 AND -a1 -> valid source
        [0, -1],   # a1 unlabeled -> cannot prove violation of -a1
        [-1, 1],   # a0 unlabeled -> cannot prove violation of +a0
        [1, 1],    # satisfies +a0 -> not a source (twin retrieval)
    ], dtype=torch.int8)
    mask = gg._source_violates(attrs, pi=[0], ni=[1])
    assert mask.tolist() == [True, False, False, False]


def test_source_and_target_pools_are_disjoint():
    # An instance labeled on the queried attributes cannot both violate and satisfy.
    attrs = torch.randint(-1, 2, (64, 6), dtype=torch.int8)
    sat = gg._strictly_satisfies(attrs, pi=[0], ni=[3])
    viol = gg._source_violates(attrs, pi=[0], ni=[3])
    assert not (sat & viol).any()


def test_generated_json_is_eval_compatible_and_exact():
    # The frozen benchmark JSON must load through the project's own eval engine, and each
    # source's target list must be the COMPLETE regenerated set (no caps) — this is what
    # makes Precision@K / Recall@K real.
    import pytest
    from eval import find_eval_json, load_eval_json, parse_query
    json_path = find_eval_json().parent / "ovad_evaluation.json"
    if not json_path.exists():
        pytest.skip("ovad_evaluation.json not generated yet (run src/ovad/generate_gt.py)")
    gt = load_eval_json(json_path)
    assert len(gt) >= 1
    from ovad.data_loader import load_attributes
    try:
        attrs = load_attributes("test")                 # benchmark corpus = TEST split
    except FileNotFoundError:
        attrs = None                                    # index/split not built here
    for entry in gt:
        assert set(entry.keys()) == {"query", "ground_truth"}
        pos, neg = parse_query(entry["query"])          # must parse without error
        assert pos or neg
        for src, targets in entry["ground_truth"].items():
            assert int(src) >= 0
            assert len(targets) >= gg.MIN_TARGETS
            assert int(src) not in targets              # disjoint pools -> never a target
            if attrs is not None:                       # split-local index space
                assert int(src) < attrs.shape[0]
                assert all(t < attrs.shape[0] for t in targets)
        if attrs is not None:
            # Exactness: the frozen entry IS the full uncapped ground truth.
            assert entry["ground_truth"] == gg.build_ground_truth(entry["query"], attrs)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
