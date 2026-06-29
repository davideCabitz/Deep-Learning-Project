"""
Tier-2a Track S verification — kept out of src/ per project convention (tests live in test/).

Cheap, self-contained checks of the manifold machinery and the asymmetric-subspace retrieval seam.
No CLIP, no real feature DB: the geometry is exercised on random unit vectors and a tiny synthetic
corpus, so this runs in well under a second. Covers the 8 checks in S_plan.md §7.4.
Run:  python test/test_tier2a_S.py   (or: pytest test/test_tier2a_S.py)
"""

import sys
from pathlib import Path

import torch

# src/ uses bare imports (e.g. `from manifold import ...`); put it on the path so we can import Track S.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from manifold import log_map, exp_map, build_subspace
from tier2a_S import TrackSConfig, _project_db_into_subspace, make_get_ranking
from clip_prompts import build_prompts_for_attribute
from data_loader import ATTRIBUTE_NAMES


def _rand_unit(n, d, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(n, d, generator=g), dim=1)


def _synthetic_bank(seed):
    """A [40, width, 512] random unit bank wide enough that every attribute's true stack slices in."""
    width = max(len(build_prompts_for_attribute(n)) for n in ATTRIBUTE_NAMES) + 2
    return _rand_unit(40 * width, 512, seed).reshape(40, width, 512)


# 1 — Log/exp round-trip: exp_mu(log_mu(x)) ≈ x in mu's hemisphere (where the maps invert).
def test_log_exp_roundtrip():
    mu = _rand_unit(1, 64, 0)[0]
    X = _rand_unit(50, 64, 1)
    X = torch.where((X @ mu).unsqueeze(1) < 0, -X, X)             # keep inside the injectivity radius
    recon = exp_map(mu, log_map(mu, X))
    assert torch.allclose(recon, X, atol=1e-5), (recon - X).abs().max().item()


# 2 — V_k⁺ orthonormality: SVD right singular vectors form an orthonormal basis of S⁺.
def test_positive_subspace_orthonormal():
    _, V_k = build_subspace(_rand_unit(30, 64, 2), k=10)
    assert V_k.shape == (64, 10)
    assert torch.allclose(V_k.T @ V_k, torch.eye(10), atol=1e-5)


# 3 — P⁺ idempotence: the projector P⁺ = V_k⁺ V_k⁺ᵀ satisfies P⁺ P⁺ = P⁺ and fixes its own span.
def test_positive_projector_idempotent():
    _, V_k = build_subspace(_rand_unit(30, 64, 3), k=8)
    P = V_k @ V_k.T
    assert torch.allclose(P @ P, P, atol=1e-5)
    assert torch.allclose(P @ V_k[:, 0], V_k[:, 0], atol=1e-5)    # a basis vector is unmoved


# 4 — Complement orthogonality: (I − P⁻) P⁻ = 0, so the negation complement removes exactly S⁻.
def test_negative_complement_orthogonal():
    _, V_k = build_subspace(_rand_unit(30, 64, 4), k=8)
    P = V_k @ V_k.T
    assert torch.allclose((torch.eye(64) - P) @ P, torch.zeros(64, 64), atol=1e-5)


# 5 — neg_norms non-negative: projection energy ‖proj_{S⁻}(v_d)‖ is a norm, so it is ≥ 0 everywhere.
def test_neg_norms_non_negative():
    bank = _synthetic_bank(5)
    image_features = _rand_unit(200, 512, 6)
    stack = bank[ATTRIBUTE_NAMES.index("Male"), : len(build_prompts_for_attribute("Male"))]
    neg_norms = _project_db_into_subspace(stack, image_features, k=10, use_rotation=True).norm(dim=1)
    assert (neg_norms >= 0).all()


# 6 — End-to-end ranking: a mixed +/− query yields a full permutation with the source ranked last.
def test_get_ranking_end_to_end():
    bank = _synthetic_bank(7)
    N = 200
    image_features = _rand_unit(N, 512, 8)
    get_ranking = make_get_ranking("+Smiling, -Male", image_features, bank, TrackSConfig(k_pos=10, k_neg=10))
    src = 42
    ranking = get_ranking(src)
    assert sorted(ranking) == list(range(N))                     # a permutation of all indices
    assert ranking[-1] == src                                    # source excluded → pushed to the end


# 7 — Empty T⁺ fallback: a negation-only query (no positive subspace) ranks by penalty, source last.
def test_empty_positive_fallback():
    bank = _synthetic_bank(9)
    N = 150
    image_features = _rand_unit(N, 512, 10)
    get_ranking = make_get_ranking("-Male, -Mustache", image_features, bank, TrackSConfig(k_neg=10))
    ranking = get_ranking(17)
    assert sorted(ranking) == list(range(N))
    assert ranking[-1] == 17


# 8 — Empty T⁻ fallback: a positive-only query reduces to pure CLAY on S⁺, source last.
def test_empty_negative_fallback():
    bank = _synthetic_bank(11)
    N = 150
    image_features = _rand_unit(N, 512, 12)
    get_ranking = make_get_ranking("+Eyeglasses", image_features, bank, TrackSConfig(k_pos=10))
    ranking = get_ranking(99)
    assert sorted(ranking) == list(range(N))
    assert ranking[-1] == 99


# Extra — stacked variant must also run end-to-end (the §6.3 ablation toggle), not just per-condition.
def test_stacked_variant_runs():
    bank = _synthetic_bank(13)
    N = 120
    image_features = _rand_unit(N, 512, 14)
    cfg = TrackSConfig(k_pos=8, k_neg=8, per_condition=False)
    ranking = make_get_ranking("+Black_Hair, +Smiling, -Wavy_Hair", image_features, bank, cfg)(5)
    assert sorted(ranking) == list(range(N))
    assert ranking[-1] == 5


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  [OK] {t.__name__}")
    print(f"[OK] all {len(tests)} Track S tests passed")


if __name__ == "__main__":
    _run_all()
