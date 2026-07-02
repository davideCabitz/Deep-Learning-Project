"""
Tier-4 Bayes-Φ verification.

Cheap, self-contained checks of tier4_bayes.py on synthetic tensors — no CLIP, no
real feature DB (one optional artifact-backed check is skipped when artifacts are
missing). The two load-bearing checks are #1 (the Poisson-binomial DP is exact) and
#3 (Φ at init reproduces the closed-form Stage-A score — the "can only improve"
guarantee the whole method rests on).

Run:  python test/test_tier4_bayes.py   (or: pytest test/test_tier4_bayes.py)
"""

import itertools
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_loader import _get_artifacts_dir
from tier4_bayes import (
    EPS,
    N_ATTR,
    BayesPhi,
    Tier4Config,
    hamming_tail_log,
    listnet_scores,
    make_stage_a_chunk,
    make_stage_b_chunk,
    query_indices,
)


def _fake_db(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Synthetic unit features + valid posteriors in (EPS, 1−EPS).
    g = torch.Generator().manual_seed(seed)
    feats = F.normalize(torch.randn(n, 512, generator=g), dim=1)
    P = torch.rand(n, N_ATTR, generator=g).clamp(EPS, 1 - EPS)
    return feats, P


# ---------------------------------------------------------------------------
# 1. Poisson-binomial DP is EXACT (vs brute-force enumeration over subsets)
# ---------------------------------------------------------------------------

def test_tail_dp_exact():
    g = torch.Generator().manual_seed(0)
    F_ = 8
    p_ref = torch.rand(2, F_, generator=g).clamp(0.05, 0.95)
    p_db = torch.rand(3, F_, generator=g).clamp(0.05, 0.95)
    got = hamming_tail_log(p_ref, p_db, hamming_max=2).exp()

    d = p_db.unsqueeze(0) * (1 - p_ref).unsqueeze(1) + (1 - p_db).unsqueeze(0) * p_ref.unsqueeze(1)
    want = torch.zeros(2, 3)
    for bits in itertools.product([0, 1], repeat=F_):        # enumerate all disagreement patterns
        if sum(bits) > 2:
            continue
        mask = torch.tensor(bits, dtype=torch.float32)
        want += (d * mask + (1 - d) * (1 - mask)).prod(dim=2)
    assert torch.allclose(got, want, atol=1e-6), f"DP tail mismatch: {(got - want).abs().max()}"
    print("[OK] 1. Poisson-binomial DP matches brute-force enumeration")


# ---------------------------------------------------------------------------
# 2. Tail sanity — identical profiles maximize the tail; hamming_max=F → prob 1
# ---------------------------------------------------------------------------

def test_tail_sanity():
    p_ref = torch.tensor([[0.9, 0.1, 0.8, 0.2]])
    same = p_ref.clone()
    diff = 1 - p_ref
    t_same = hamming_tail_log(p_ref, same, 2)
    t_diff = hamming_tail_log(p_ref, diff, 2)
    assert t_same.item() > t_diff.item(), "identical profile must out-score the flipped one"
    full = hamming_tail_log(p_ref, diff, hamming_max=4)
    assert torch.allclose(full.exp(), torch.ones(1, 1), atol=1e-6), "K≤F tail must be prob 1"
    print("[OK] 2. tail sanity (identity ordering, total-probability edge)")


# ---------------------------------------------------------------------------
# 3. Φ at init ≡ Stage A closed form (the load-bearing invariant)
# ---------------------------------------------------------------------------

def test_init_equals_stage_a():
    torch.manual_seed(0)
    cfg = Tier4Config()
    feats, P = _fake_db(200, seed=1)
    model = BayesPhi(cfg)
    pi, ni, free = query_indices("+Smiling, -Male")
    a = make_stage_a_chunk(feats, P, cfg)([3, 7], pi, ni, free)
    b = make_stage_b_chunk(model, feats, P, cfg)([3, 7], pi, ni, free)
    assert (a - b).abs().max() < 1e-4, f"init deviates from Stage A: {(a - b).abs().max()}"
    print("[OK] 3. BayesPhi at init reproduces the closed-form Stage-A score")


# ---------------------------------------------------------------------------
# 4. Query parsing → index sets partition the 40 attributes
# ---------------------------------------------------------------------------

def test_query_indices():
    pi, ni, free = query_indices("-Smiling, +Eyeglasses, +Wearing_Hat")
    assert len(pi) == 2 and len(ni) == 1 and len(free) == 37
    assert set(pi) | set(ni) | set(free) == set(range(N_ATTR))
    assert not (set(pi) & set(ni))
    print("[OK] 4. query parsing partitions the attribute set")


# ---------------------------------------------------------------------------
# 5. ListNet gradient direction — raising a positive's score lowers the loss
# ---------------------------------------------------------------------------

def test_listnet_direction():
    base = torch.tensor([1.0, 0.5, 0.4, 0.3])
    better = torch.tensor([2.0, 0.5, 0.4, 0.3])
    l0 = listnet_scores(base, n_pos=1, tau=2.0)
    l1 = listnet_scores(better, n_pos=1, tau=2.0)
    assert l1 < l0, "raising the positive's score must reduce ListNet loss"
    print("[OK] 5. ListNet loss direction")


# ---------------------------------------------------------------------------
# 6. End-to-end seam on real artifacts (skipped when artifacts are absent)
# ---------------------------------------------------------------------------

def test_end_to_end_seam():
    art = _get_artifacts_dir()
    needed = ["clip_image_features_test.pt", "clip_image_features_train.pt",
              "celeba_attributes_train.pt"]
    if not all((art / f).exists() for f in needed):
        print("[SKIP] 6. artifacts missing — end-to-end seam not checked")
        return
    from clip_features import load_image_features
    from tier3_train_utils import assert_db_frozen
    from tier4_bayes import load_or_train_probes, make_get_ranking, posteriors

    cfg = Tier4Config(probe_epochs=1)
    dev = torch.device("cpu")
    probe = load_or_train_probes(cfg, dev)
    feats = load_image_features()
    P = posteriors(probe, feats)
    get_ranking = make_get_ranking("+Smiling, -Male", feats, P, cfg)
    assert_db_frozen(feats, get_ranking, probe_src=0)         # §5 self-exclusion + frozen DB
    print("[OK] 6. end-to-end Stage-A seam on real artifacts (frozen-DB tripwire passed)")


if __name__ == "__main__":
    test_tail_dp_exact()
    test_tail_sanity()
    test_init_equals_stage_a()
    test_query_indices()
    test_listnet_direction()
    test_end_to_end_seam()
    print("\nAll Tier-4 Bayes-Phi checks passed.")
