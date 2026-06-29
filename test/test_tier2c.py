"""
Tier-2a SV-union verification (planSV.md §6).

Cheap, self-contained checks of tier2a_SVunion.py on synthetic unit vectors — no
CLIP, no real feature DB, no disk I/O. The two load-bearing checks are #2 (the
negative subspace lives in the GLOBAL tangent plane T_μ) and #5 (at k_neg=1 the
method nests Track V's single-direction negation).

Run:  python test/test_tier2a_SVunion.py   (or: pytest test/test_tier2a_SVunion.py)
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from manifold import log_map, exp_map, tangent_mean
from tier2c import (
    SVunionConfig,
    K_CACHE,
    _build_visual_neg_subspace,
    _build_union_basis,
    _compose_query_svunion,
    mine_neg_subspaces,
    make_get_ranking,
)
from tier2b import _compose_query_ext
from data_loader import ATTRIBUTE_NAMES, ATTR_TO_IDX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_unit(n: int, d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, d, generator=g), dim=1)


def _same_hemisphere(X: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    # Flip rows with xᵀμ < 0 so all points share μ's open hemisphere (log/exp well-defined).
    signs = (X @ mu).sign().unsqueeze(1)
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return X * signs


def _cluster_has_attr(mu: torch.Tensor, d: int, m: int, seed: int, jitter: float) -> torch.Tensor:
    # A tight cluster of unit "has-attribute" images around a shared point off μ — so the
    # log-maps carry a dominant common direction (the structure the nesting property needs).
    g = torch.Generator().manual_seed(seed)
    center = F.normalize(mu + 0.6 * torch.randn(d, generator=g), dim=0)
    pts = center.unsqueeze(0) + jitter * torch.randn(m, d, generator=g)
    return _same_hemisphere(F.normalize(pts, dim=1), mu)


# ---------------------------------------------------------------------------
# #1 — subspace orthonormality
# ---------------------------------------------------------------------------

def test_subspace_orthonormal():
    # Q_bᵀ Q_b ≈ I_k — the mined basis columns are orthonormal.
    d, k = 64, 8
    mu = _rand_unit(1, d, 0)[0].double()
    X_b = _cluster_has_attr(mu, d, 200, 1, jitter=0.15).double()
    Q = _build_visual_neg_subspace(X_b, mu, k)
    gram = Q.T @ Q
    assert torch.allclose(gram, torch.eye(k, dtype=Q.dtype), atol=1e-5), (gram - torch.eye(k)).abs().max().item()


# ---------------------------------------------------------------------------
# #2 — subspace lives in the GLOBAL tangent plane T_μ (guards the §3.2 subtlety)
# ---------------------------------------------------------------------------

def test_subspace_in_global_tangent_plane():
    # ‖Q_bᵀ μ‖ ≈ 0 — every column ⊥ μ, i.e. the subspace lives in the same T_μ as the query.
    d, k = 64, 10
    mu = _rand_unit(1, d, 2)[0].double()
    X_b = _cluster_has_attr(mu, d, 300, 3, jitter=0.2).double()
    Q = _build_visual_neg_subspace(X_b, mu, k)
    assert (Q.T @ mu).norm().item() < 1e-5, (Q.T @ mu).norm().item()


# ---------------------------------------------------------------------------
# #3 — rejection idempotence
# ---------------------------------------------------------------------------

def test_rejection_idempotent():
    # Rejecting onto the complement twice == once (projection is idempotent).
    d, k = 64, 6
    mu = _rand_unit(1, d, 4)[0].double()
    X_b = _cluster_has_attr(mu, d, 200, 5, jitter=0.2).double()
    Q = _build_visual_neg_subspace(X_b, mu, k)
    v = log_map(mu, _same_hemisphere(_rand_unit(1, d, 6).double(), mu)).squeeze(0)
    once  = v - Q @ (Q.T @ v)
    twice = once - Q @ (Q.T @ once)
    assert torch.allclose(once, twice, atol=1e-6), (once - twice).abs().max().item()


# ---------------------------------------------------------------------------
# #4 — complement output is orthogonal to span(Q_all)
# ---------------------------------------------------------------------------

def test_complement_orthogonal_to_span():
    # (I − Q_allQ_allᵀ)v ⊥ span(Q_all) — the rejected vector has no energy left in the subspace.
    d = 64
    subspaces = torch.zeros(len(ATTRIBUTE_NAMES), d, K_CACHE, dtype=torch.float64)
    mu = _rand_unit(1, d, 7)[0].double()
    for name, seed in (("Male", 8), ("Mustache", 9)):
        X_b = _cluster_has_attr(mu, d, 200, seed, jitter=0.2).double()
        subspaces[ATTR_TO_IDX[name]] = _build_visual_neg_subspace(X_b, mu, K_CACHE)
    Q_all = _build_union_basis(subspaces, ["Male", "Mustache"], k_neg=5)
    v = torch.randn(d, dtype=torch.float64)
    residual = v - Q_all @ (Q_all.T @ v)
    assert (Q_all.T @ residual).norm().item() < 1e-6, (Q_all.T @ residual).norm().item()


# ---------------------------------------------------------------------------
# #5 — nesting: SV-union(k_neg=1) ≈ Track V single-direction negation
# ---------------------------------------------------------------------------

def test_nesting_kneg1_matches_track_v():
    # On a negation query, the SV-union(k_neg=1) query and the Track V query agree (cos > 0.99),
    # because the top singular direction of the log-mapped has-X cluster ≈ V's tangent-mean axis.
    d = 64
    mu = _rand_unit(1, d, 10)[0].double()
    X_b = _cluster_has_attr(mu, d, 400, 11, jitter=0.04)   # tight cluster → dominant shared axis
    idx = ATTR_TO_IDX["Male"]

    directions = torch.zeros(len(ATTRIBUTE_NAMES), d, dtype=torch.float64)
    directions[idx] = tangent_mean(mu, X_b.double())       # Track V's mined direction
    subspaces = torch.zeros(len(ATTRIBUTE_NAMES), d, K_CACHE, dtype=torch.float64)
    subspaces[idx] = _build_visual_neg_subspace(X_b.double(), mu, K_CACHE)

    v_ref = _same_hemisphere(_rand_unit(1, d, 12).double(), mu)[0]
    cfg = SVunionConfig(k_neg=1)
    q_sv = _compose_query_svunion(v_ref, [], ["Male"], mu, directions, subspaces, cfg)
    q_v  = _compose_query_ext(v_ref, [], ["Male"], mu, directions, alpha=1.0)
    cos = float(q_sv @ q_v)
    assert cos > 0.99, cos


# ---------------------------------------------------------------------------
# #6 — empty T_neg degenerates to Track V positive composition
# ---------------------------------------------------------------------------

def test_empty_neg_equals_track_v_positive():
    # With no negatives, rejection is identity → query == Track V positive composition exactly.
    d = 64
    mu = _rand_unit(1, d, 13)[0]
    v_ref = _same_hemisphere(_rand_unit(1, d, 14), mu)[0]
    directions = torch.randn(len(ATTRIBUTE_NAMES), d) * 0.1
    directions = directions - (directions @ mu).unsqueeze(1) * mu   # project onto T_μ
    subspaces = torch.zeros(len(ATTRIBUTE_NAMES), d, K_CACHE)
    cfg = SVunionConfig(k_neg=10)
    q_sv = _compose_query_svunion(v_ref, ["Smiling"], [], mu, directions, subspaces, cfg)
    q_v  = _compose_query_ext(v_ref, ["Smiling"], [], mu, directions, alpha=1.0)
    assert torch.allclose(q_sv, q_v, atol=1e-6), (q_sv - q_v).abs().max().item()


# ---------------------------------------------------------------------------
# #7 — end-to-end seam (CONTRACT §5): source excluded, full permutation
# ---------------------------------------------------------------------------

def test_seam_source_excluded_and_full_permutation():
    # get_ranking returns every index once and never ranks the source itself.
    N, d = 120, 64
    mu = _rand_unit(1, d, 15)[0]
    feats = _same_hemisphere(_rand_unit(N, d, 16), mu)
    directions = torch.zeros(len(ATTRIBUTE_NAMES), d)
    subspaces = torch.zeros(len(ATTRIBUTE_NAMES), d, K_CACHE)
    subspaces[ATTR_TO_IDX["Male"]] = _build_visual_neg_subspace(
        _cluster_has_attr(mu, d, 150, 17, jitter=0.2), mu, K_CACHE,
    )
    for side in ("query", "db"):
        get_ranking = make_get_ranking(
            "+Smiling, -Male", feats, mu, directions, subspaces, SVunionConfig(k_neg=5, reject_on=side),
        )
        ranking = get_ranking(42)
        assert sorted(ranking) == list(range(N)), side
        assert ranking[-1] == 42, side


# ---------------------------------------------------------------------------
# #8 — empty T_pos + negation runs and yields a valid ranking
# ---------------------------------------------------------------------------

def test_empty_pos_with_negation_valid():
    # The headline stress case (-Male, -Mustache): no positives, pure visual-region deletion.
    N, d = 120, 64
    mu = _rand_unit(1, d, 18)[0]
    feats = _same_hemisphere(_rand_unit(N, d, 19), mu)
    directions = torch.zeros(len(ATTRIBUTE_NAMES), d)
    subspaces = torch.zeros(len(ATTRIBUTE_NAMES), d, K_CACHE)
    for name, seed in (("Male", 20), ("Mustache", 21)):
        subspaces[ATTR_TO_IDX[name]] = _build_visual_neg_subspace(
            _cluster_has_attr(mu, d, 150, seed, jitter=0.2), mu, K_CACHE,
        )
    get_ranking = make_get_ranking(
        "-Male, -Mustache", feats, mu, directions, subspaces, SVunionConfig(k_neg=10),
    )
    ranking = get_ranking(7)
    assert sorted(ranking) == list(range(N))
    assert ranking[-1] == 7


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------

def test_config_rejects_illegal_values():
    # Illegal configs fail loudly at construction, not silently in the scoring loop.
    for bad in (lambda: SVunionConfig(reject_on="both"), lambda: SVunionConfig(k_neg=0)):
        try:
            bad()
            assert False, "expected ValueError"
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# mining smoke test
# ---------------------------------------------------------------------------

def test_mine_neg_subspaces_shapes_and_zero_guard():
    # mine_neg_subspaces returns [40, d, k]; an attribute with no examples gets a zero block.
    N, d = 300, 48
    mu = _rand_unit(1, d, 22)[0]
    feats = _same_hemisphere(_rand_unit(N, d, 23), mu)
    attrs = torch.zeros(N, 40)
    g = torch.Generator().manual_seed(24)
    for j in range(39):                                    # leave attr 39 with zero examples
        idx = torch.randperm(N, generator=g)[: N // 3]
        attrs[idx, j] = 1.0
    subspaces = mine_neg_subspaces(feats, attrs, mu, k=12)
    assert subspaces.shape == (40, d, 12)
    assert subspaces[39].abs().max().item() == 0.0        # zero-guarded empty attribute


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  [OK] {t.__name__}")
    print(f"[OK] all {len(tests)} Tier-2a SV-union tests passed")


if __name__ == "__main__":
    _run_all()
