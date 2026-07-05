"""
TriStateQueryGenerator verification (the OVAD adapter's synthetic-query sampler seam).

On a toy tri-state matrix, every emitted query must obey OVAD semantics: query axes only
where the reference is LABELED; positives strictly satisfy the constraints AND the
masked relative-Hamming rate <= 0.05; hard negatives are labeled-1 on a violated
negative attribute and labeled-1 on all positives.

Run:  pytest test/test_ovad_query_generator.py
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "Baselines"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "Training-Free"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "Trained"))

from ovad.adapter import OVAD_ATTR_TO_IDX, _make_tristate_query_generator
from ovad.generate_gt import HAMMING_RATE, _masked_hamming_rate_batch


@pytest.fixture(scope="module")
def sampler():
    import phi_gate
    TSQG = _make_tristate_query_generator()
    torch.manual_seed(0)
    # 400 instances × 6 attrs, tri-state, positively-biased so queries find targets.
    A = torch.randint(0, 3, (400, 6), dtype=torch.int8) - 1
    cfg = phi_gate.PhiConfig(seed=0)
    return TSQG(A, cfg), A


def test_sampled_queries_obey_tristate_semantics(sampler):
    gen, A = sampler
    n_checked = 0
    for _ in range(300):
        out = gen.sample_query()
        if out is None:
            continue
        r, pos_names, neg_names, pos_sel, hard_sel = out
        # The sampler labels toy columns with real OVAD names (OVAD_NAMES[col]); the
        # index map inverts them back to the toy column ids.
        pi = [OVAD_ATTR_TO_IDX[n] for n in pos_names]
        ni = [OVAD_ATTR_TO_IDX[n] for n in neg_names]
        assert all(c < A.shape[1] for c in pi + ni)

        # Reference is LABELED on every query axis: ==1 for +, ==0 for -.
        assert all(A[r, a] == 1 for a in pi)
        assert all(A[r, b] == 0 for b in ni)
        assert r not in pos_sel.tolist() and r not in hard_sel.tolist()

        free = torch.tensor([c for c in range(A.shape[1]) if c not in set(pi) | set(ni)],
                            dtype=torch.long)
        for t in pos_sel.tolist():
            # Positives: strict labeled satisfaction + masked rate <= 0.05.
            assert all(A[t, a] == 1 for a in pi) and all(A[t, b] == 0 for b in ni)
            rate = _masked_hamming_rate_batch(A[r].unsqueeze(0)[:, free],
                                              A[t].unsqueeze(0)[:, free])[0, 0]
            assert rate <= HAMMING_RATE
        for h in hard_sel.tolist():
            # Hard negatives: labeled-1 on all positives, labeled-1 on >=1 negative attr.
            assert all(A[h, a] == 1 for a in pi)
            if ni:
                assert any(A[h, b] == 1 for b in ni)
        n_checked += 1
    assert n_checked > 20, "sampler produced too few valid queries to be a useful test"


def test_never_queries_unlabeled_reference_attribute(sampler):
    gen, A = sampler
    for _ in range(300):
        out = gen.sample_query()
        if out is None:
            continue
        r = out[0]
        cols = [OVAD_ATTR_TO_IDX[n] for n in out[1] + out[2]]
        assert all(A[r, c] != -1 for c in cols)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
