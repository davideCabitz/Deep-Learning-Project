"""
AwA2 ground-truth generator verification.

Unit checks the adaptive relaxed-Hamming rule on a synthetic 4-class toy matrix (strict ±
satisfaction + freeHam ≤ min over satisfying classes + HAMMING_SLACK), the ≥5-targets
viability floor, and that the produced JSON is consumable by eval.load_eval_json /
parse_query with exact (uncapped) target sets.

Run:  pytest test/test_awa2_gt.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from awa2 import generate_gt as gg


def _toy_matrix():
    # 4 classes × 5 attrs. attr0 is the query axis; attrs 1..4 are the identity axes.
    #   c0 = 1 0 0 0 0
    #   c1 = 1 0 0 0 1   (Hamming 1 from c0 on free attrs)
    #   c2 = 1 1 1 0 0   (Hamming 2 from c0)
    #   c3 = 1 1 1 1 1   (Hamming 4 from c0 — outside radius)
    return torch.tensor([
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
    ], dtype=torch.float32)


def test_valid_target_classes_strict_and_hamming(monkeypatch):
    cm = _toy_matrix()
    # Query "+attr0": free = attrs 1..4. From source c0, valid = strict(attr0=1) AND
    # freeHam ≤ min(0, over satisfying) + slack 2.
    pi, ni, free = [0], [], [1, 2, 3, 4]
    mask = gg._valid_target_classes(cm, pi, ni, free, src_class=0)
    assert mask.tolist() == [True, True, True, False]  # c3 excluded: Ham 4 > 0 + slack 2


def test_adaptive_radius_reaches_distant_satisfying_classes():
    # The fixed-2 rule is empty on AwA2 (min inter-class free-Ham 3–21); the adaptive rule
    # must still find the NEAREST satisfying classes. Source c0 violates "+attr0"; the
    # satisfying classes sit at free-Ham 3 and 4 (≤ 3+2) and 8 (> 3+2, excluded).
    cm = torch.tensor([
        [0, 1, 1, 1, 0, 0, 0, 0, 0],   # c0 — source, attr0=0
        [1, 0, 0, 0, 1, 0, 0, 0, 0],   # c1 — satisfies, free-Ham 4 from c0
        [1, 1, 1, 0, 0, 1, 1, 0, 0],   # c2 — satisfies, free-Ham 3 (the minimum)
        [1, 0, 0, 0, 1, 1, 1, 1, 1],   # c3 — satisfies, free-Ham 8 > min+slack
    ], dtype=torch.float32)
    pi, ni, free = [0], [], list(range(1, 9))
    mask = gg._valid_target_classes(cm, pi, ni, free, src_class=0)
    assert mask.tolist() == [False, True, True, False]


def test_valid_target_classes_empty_when_nothing_satisfies():
    cm = _toy_matrix()
    # "-attr0": no toy class has attr0=0 -> all-False, no min() crash.
    mask = gg._valid_target_classes(cm, [], [0], [1, 2, 3, 4], src_class=0)
    assert not mask.any()


def test_negative_constraint_excludes_classes():
    cm = _toy_matrix()
    # Query "-attr4": classes with attr4=1 (c1, c3) are strictly excluded.
    pi, ni, free = [], [4], [0, 1, 2, 3]
    mask = gg._valid_target_classes(cm, pi, ni, free, src_class=0)
    assert mask[1].item() is False
    assert mask[3].item() is False
    assert mask[0].item() is True


def test_source_policy_excludes_satisfying_classes():
    # Clause 0 (violate-every-constraint, the verified CelebA official policy): classes
    # already satisfying the constraints are twin-retrieval sources and must be excluded.
    cm = _toy_matrix()
    # "+attr0": every toy class satisfies it -> NO class is a valid source.
    assert gg._source_class_mask(cm, [0], []).tolist() == [False, False, False, False]
    # "-attr4": c1, c3 have attr4=1 (violate the constraint) -> only they are sources.
    assert gg._source_class_mask(cm, [], [4]).tolist() == [False, True, False, True]
    # Mixed "+attr1, -attr4": source needs attr1=0 AND attr4=1 -> only c1.
    assert gg._source_class_mask(cm, [1], [4]).tolist() == [False, True, False, False]


def test_min_targets_floor():
    # A valid target class with < MIN_TARGETS images ⇒ source dropped (viability floor).
    # Exercised via the class-mask core (matrix-agnostic; build_ground_truth's query parse
    # binds the global 85-attr ATTR_TO_IDX, which a toy matrix does not share).
    cm = torch.tensor([[1, 0], [1, 0]], dtype=torch.float32)  # 2 classes, both attr0=1
    labels = torch.tensor([0, 0, 0, 1])                        # 3 + 1 images — under 5 each
    pi, ni, free = [0], [], [1]
    n_classes = cm.shape[0]
    class_to_imgs = {c: (labels == c).nonzero(as_tuple=True)[0] for c in range(n_classes)}
    kept = {}
    for src_class in range(n_classes):
        mask = gg._valid_target_classes(cm, pi, ni, free, src_class)
        tgt = torch.cat([class_to_imgs[c] for c in range(n_classes) if mask[c]]) \
            if mask.any() else torch.empty(0, dtype=torch.long)
        if len(tgt) < gg.MIN_TARGETS:
            continue
        for s in class_to_imgs[src_class].tolist():
            if len(tgt[tgt != s]) >= gg.MIN_TARGETS:
                kept[str(s)] = True
    assert kept == {}  # nothing meets the ≥5 floor


def test_generated_json_is_eval_compatible():
    # The frozen benchmark JSON must load and parse through the project's own eval engine.
    from eval import find_eval_json, load_eval_json, parse_query
    json_path = find_eval_json().parent / "awa2_evaluation.json"
    if not json_path.exists():
        import pytest
        pytest.skip("awa2_evaluation.json not generated yet (run src/awa2/generate_gt.py)")
    gt = load_eval_json(json_path)
    assert len(gt) >= 1
    from awa2.data_loader import load_class_attribute_matrix, load_class_labels
    try:
        cm, labels = load_class_attribute_matrix(), load_class_labels("test")
    except FileNotFoundError:
        cm = labels = None                              # split not built on this machine
    for entry in gt:
        assert set(entry.keys()) == {"query", "ground_truth"}
        pos, neg = parse_query(entry["query"])          # must parse without error
        assert pos or neg
        pi, ni, free = gg.query_indices(entry["query"])
        for src, targets in entry["ground_truth"].items():
            assert int(src) >= 0
            assert len(targets) >= gg.MIN_TARGETS
            assert int(src) not in targets              # source excluded from its targets
            if cm is not None:
                c = labels[int(src)]
                # Clause 0: source violates every ± constraint.
                assert all(cm[c, a] == 0 for a in pi) and all(cm[c, b] == 1 for b in ni)
                # Targets: strict satisfaction + adaptive radius, and the set is EXACT —
                # every test image of a valid target class is present (no cap).
                mask = gg._valid_target_classes(cm, pi, ni, free, int(c))
                expected = (mask[labels]).nonzero(as_tuple=True)[0].tolist()
                assert targets == [t for t in expected if t != int(src)]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
