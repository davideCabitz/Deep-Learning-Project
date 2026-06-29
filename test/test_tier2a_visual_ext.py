"""
Tier-2a Visual Extension verification.

Tests _compute_clip_weights, mine_weighted_directions, _compose_query_ext,
and make_get_ranking_ext on synthetic random unit vectors — no CLIP, no real
feature DB, no disk I/O. Runs in < 1 s.

Run:  python test/test_tier2a_visual_ext.py   (or: pytest test/test_tier2a_visual_ext.py)
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from manifold import log_map
from tier2a_visual_extension import (
    _compute_clip_weights,
    _compose_query_ext,
    make_get_ranking_ext,
)
from data_loader import ATTRIBUTE_NAMES, ATTR_TO_IDX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_unit(n: int, d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, d, generator=g), dim=1)


def _same_hemisphere(X: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    signs = (X @ mu).sign().unsqueeze(1)
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return X * signs


def _make_directions(mu: torch.Tensor, d: int, seed: int) -> torch.Tensor:
    # [40, d] tangent vectors at μ — small random perturbations in the tangent plane.
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(40, d, generator=g) * 0.1
    raw = raw - (raw @ mu).unsqueeze(1) * mu
    return raw


# ---------------------------------------------------------------------------
# _compute_clip_weights
# ---------------------------------------------------------------------------

def test_clip_weights_shape():
    # Output must be [N_train, n_text] matching input dimensions.
    N, A, d = 50, 8, 32
    train = _rand_unit(N, d, 0)
    text  = _rand_unit(A, d, 1)
    w = _compute_clip_weights(train, text)
    assert w.shape == (N, A)


def test_clip_weights_sum_to_one():
    # Softmax rows must sum to 1.0.
    train = _rand_unit(60, 32, 2)
    text  = _rand_unit(10, 32, 3)
    w = _compute_clip_weights(train, text)
    row_sums = w.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(60), atol=1e-5)


def test_clip_weights_non_negative():
    # Softmax output is always non-negative.
    train = _rand_unit(30, 32, 4)
    text  = _rand_unit(5, 32, 5)
    w = _compute_clip_weights(train, text)
    assert (w >= 0).all()


# ---------------------------------------------------------------------------
# _compose_query_ext — output contract
# ---------------------------------------------------------------------------

def test_ext_compose_returns_unit():
    # _compose_query_ext must return a unit vector regardless of α.
    d   = 64
    mu  = _rand_unit(1, d, 10)[0]
    v_ref = _same_hemisphere(_rand_unit(1, d, 11), mu)[0]
    dirs  = _make_directions(mu, d, 12)
    q = _compose_query_ext(v_ref, ["Smiling"], ["Bald"], mu, dirs, alpha=1.5)
    assert abs(q.norm().item() - 1.0) < 1e-5


def test_ext_compose_no_conditions_close_to_ref():
    # With empty T_pos and T_neg and zero directions the result should equal v_ref.
    d   = 64
    mu  = _rand_unit(1, d, 13)[0]
    v_ref = _same_hemisphere(_rand_unit(1, d, 14), mu)[0]
    dirs  = torch.zeros(40, d)
    q = _compose_query_ext(v_ref, [], [], mu, dirs)
    assert torch.allclose(q, v_ref, atol=1e-5), (q - v_ref).abs().max().item()


def test_ext_alpha_scales_positive_push():
    # Higher α should push the query further from v_ref along the positive direction.
    d   = 64
    mu  = _rand_unit(1, d, 15)[0]
    v_ref = _same_hemisphere(_rand_unit(1, d, 16), mu)[0]
    dirs  = _make_directions(mu, d, 17)

    q1 = _compose_query_ext(v_ref, ["Smiling"], [], mu, dirs, alpha=0.5)
    q2 = _compose_query_ext(v_ref, ["Smiling"], [], mu, dirs, alpha=2.0)

    dist1 = (q1 - v_ref).norm().item()
    dist2 = (q2 - v_ref).norm().item()
    assert dist2 > dist1, f"α=2.0 should push further than α=0.5: {dist2:.4f} vs {dist1:.4f}"


def test_ext_negation_removes_attribute_axis():
    # After negation the query tangent vector must be orthogonal to the negated direction.
    d   = 64
    mu  = _rand_unit(1, d, 18)[0]
    v_ref = _same_hemisphere(_rand_unit(1, d, 19), mu)[0]
    dirs  = _make_directions(mu, d, 20)
    attr  = "Smiling"

    q = _compose_query_ext(v_ref, [], [attr], mu, dirs)
    q_tan = log_map(mu, q.unsqueeze(0)).squeeze(0)
    v_hat = F.normalize(dirs[ATTR_TO_IDX[attr]], dim=0)
    component = abs((q_tan @ v_hat).item())
    assert component < 1e-4, f"Negated axis component too large: {component:.2e}"


def test_ext_joint_negation_removes_both_axes():
    # Joint QR negation must zero out BOTH negated attribute axes simultaneously.
    d   = 64
    mu  = _rand_unit(1, d, 21)[0]
    v_ref = _same_hemisphere(_rand_unit(1, d, 22), mu)[0]
    dirs  = _make_directions(mu, d, 23)

    q = _compose_query_ext(v_ref, [], ["Smiling", "Bald"], mu, dirs)
    q_tan = log_map(mu, q.unsqueeze(0)).squeeze(0)

    for attr in ["Smiling", "Bald"]:
        v_hat = F.normalize(dirs[ATTR_TO_IDX[attr]], dim=0)
        component = abs((q_tan @ v_hat).item())
        assert component < 1e-4, f"Attr '{attr}' axis not removed: {component:.2e}"


def test_ext_negation_order_independent():
    # Joint QR negation must be order-independent (sequential rejection is not).
    d   = 64
    mu  = _rand_unit(1, d, 24)[0]
    v_ref = _same_hemisphere(_rand_unit(1, d, 25), mu)[0]
    dirs  = _make_directions(mu, d, 26)

    q_ab = _compose_query_ext(v_ref, [], ["Smiling", "Bald"], mu, dirs)
    q_ba = _compose_query_ext(v_ref, [], ["Bald", "Smiling"], mu, dirs)
    assert torch.allclose(q_ab, q_ba, atol=1e-5), (q_ab - q_ba).abs().max().item()


# ---------------------------------------------------------------------------
# make_get_ranking_ext — retrieval contract (CONTRACT §5/§7)
# ---------------------------------------------------------------------------

def test_ext_ranking_is_full_permutation_gde():
    # get_ranking must return all N indices exactly once.
    N, d = 150, 64
    mu   = _rand_unit(1, d, 30)[0]
    feats = _same_hemisphere(_rand_unit(N, d, 31), mu)
    dirs  = _make_directions(mu, d, 32)
    get_ranking = make_get_ranking_ext("+Smiling", feats, mu, dirs, use_gde=True)
    ranking = get_ranking(0)
    assert sorted(ranking) == list(range(N))


def test_ext_source_excluded_gde():
    # Source index must be ranked last (pushed to -inf).
    N, d = 100, 64
    mu   = _rand_unit(1, d, 33)[0]
    feats = _same_hemisphere(_rand_unit(N, d, 34), mu)
    dirs  = _make_directions(mu, d, 35)
    src  = 42
    get_ranking = make_get_ranking_ext("+Smiling", feats, mu, dirs, use_gde=True)
    assert get_ranking(src)[-1] == src


def test_ext_ranking_is_full_permutation_lde():
    # Same contract for the LDE path through make_get_ranking_ext.
    N, d = 150, 64
    mu   = _rand_unit(1, d, 36)[0]
    feats = _rand_unit(N, d, 37)
    dirs  = _make_directions(mu, d, 38)
    get_ranking = make_get_ranking_ext("+Smiling, -Bald", feats, mu, dirs, use_gde=False)
    ranking = get_ranking(7)
    assert sorted(ranking) == list(range(N))


def test_ext_source_excluded_lde():
    # Source excluded on LDE path too.
    N, d = 100, 64
    mu   = _rand_unit(1, d, 39)[0]
    feats = _rand_unit(N, d, 40)
    dirs  = _make_directions(mu, d, 41)
    src  = 13
    get_ranking = make_get_ranking_ext("+Smiling", feats, mu, dirs, use_gde=False)
    assert get_ranking(src)[-1] == src


def test_ext_alpha_one_matches_base_gde():
    # With α=1.0 and uniform directions, ext must produce the same ranking as base GDE.
    from tier1_GDE import make_get_ranking as base_make_get_ranking

    N, d = 100, 64
    mu   = _rand_unit(1, d, 42)[0]
    feats = _same_hemisphere(_rand_unit(N, d, 43), mu)
    dirs  = _make_directions(mu, d, 44)
    src   = 5

    ranking_base = base_make_get_ranking("+Smiling, -Bald", feats, mu, dirs, use_gde=True)(src)
    ranking_ext  = make_get_ranking_ext("+Smiling, -Bald", feats, mu, dirs, alpha=1.0, use_gde=True)(src)
    assert ranking_base == ranking_ext, "α=1.0 ext must match base GDE ranking exactly"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  [OK] {t.__name__}")
    print(f"[OK] all {len(tests)} Tier-2a Visual Ext tests passed")


if __name__ == "__main__":
    _run_all()
