"""
Tier-1 GDE (GDE + LDE ablation) verification.

Cheap, self-contained checks of manifold.py and tier1_GDE.py on synthetic
random unit vectors — no CLIP, no real feature DB, no disk I/O. Runs in < 1 s.

Run:  python test/test_tier1_GDE.py   (or: pytest test/test_tier1_GDE.py)
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from manifold import log_map, exp_map, intrinsic_mean, tangent_mean
from tier1_GDE import (
    _compose_query_gde,
    _compose_query_lde,
    make_get_ranking,
    mine_directions,
)
from data_loader import ATTRIBUTE_NAMES, ATTR_TO_IDX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_unit(n: int, d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, d, generator=g), dim=1)


def _same_hemisphere(X: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    # Flip any row where xᵀμ < 0 so all points are on the same hemisphere as μ.
    signs = (X @ mu).sign().unsqueeze(1)
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return X * signs


# ---------------------------------------------------------------------------
# manifold.py — log / exp maps
# ---------------------------------------------------------------------------

def test_log_exp_roundtrip():
    # exp_μ(log_μ(x)) ≈ x for points in the same open hemisphere as μ.
    mu = _rand_unit(1, 64, 0)[0]
    X  = _same_hemisphere(_rand_unit(50, 64, 1), mu)
    recon = exp_map(mu, log_map(mu, X))
    assert torch.allclose(recon, X, atol=1e-5), (recon - X).abs().max().item()


def test_log_at_tangency_is_zero():
    # log_μ(μ) = 0 — the eps guard must not produce NaN / inf.
    mu  = _rand_unit(1, 32, 2)[0]
    out = log_map(mu, mu.unsqueeze(0))
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_exp_zero_is_mu():
    # exp_μ(0) = μ — lifting the zero tangent vector returns the base point.
    mu  = _rand_unit(1, 32, 3)[0]
    out = exp_map(mu, torch.zeros(1, 32))
    assert torch.allclose(out.squeeze(0), mu, atol=1e-6)


def test_exp_map_output_is_unit():
    # exp_map outputs must be unit vectors (they live on S^{d-1}).
    mu = _rand_unit(1, 64, 4)[0]
    V  = torch.randn(20, 64) * 0.5          # arbitrary tangent vectors
    out = exp_map(mu, V)
    norms = out.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_log_output_orthogonal_to_mu():
    # Every tangent vector log_μ(x) must be orthogonal to μ.
    mu  = _rand_unit(1, 64, 5)[0]
    X   = _same_hemisphere(_rand_unit(30, 64, 6), mu)
    T   = log_map(mu, X)
    dots = (T @ mu).abs()
    assert dots.max() < 1e-5, dots.max().item()


# ---------------------------------------------------------------------------
# manifold.py — intrinsic_mean
# ---------------------------------------------------------------------------

def test_intrinsic_mean_is_unit():
    # The returned mean must be a unit vector.
    X  = _rand_unit(40, 64, 7)
    mu = intrinsic_mean(X, n_iter=30)
    assert abs(mu.norm().item() - 1.0) < 1e-5


def test_intrinsic_mean_fixed_point():
    # The Karcher mean satisfies Σ Log_μ(x_i) ≈ 0 (first-order condition).
    X  = _rand_unit(50, 32, 8)
    X  = _same_hemisphere(X, X[0])
    mu = intrinsic_mean(X, n_iter=100)
    residual = log_map(mu, X).mean(dim=0).norm().item()
    assert residual < 1e-4, residual


# ---------------------------------------------------------------------------
# manifold.py — tangent_mean
# ---------------------------------------------------------------------------

def test_tangent_mean_shape_and_type():
    # tangent_mean returns a [d] tensor of the same dtype as the input.
    mu = _rand_unit(1, 64, 9)[0]
    X  = _same_hemisphere(_rand_unit(20, 64, 10), mu)
    v  = tangent_mean(mu, X)
    assert v.shape == (64,)
    assert v.dtype == X.dtype


def test_tangent_mean_orthogonal_to_mu():
    # The tangent mean lives in T_μS — it must be orthogonal to μ.
    mu = _rand_unit(1, 64, 11)[0]
    X  = _same_hemisphere(_rand_unit(20, 64, 12), mu)
    v  = tangent_mean(mu, X)
    assert abs((v @ mu).item()) < 1e-5


# ---------------------------------------------------------------------------
# tier1_GDE.py — composition
# ---------------------------------------------------------------------------

def _make_synthetic_directions(mu: torch.Tensor, d: int, seed: int) -> torch.Tensor:
    # Build a plausible [40, d] direction table: each row is a small tangent vector at μ.
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(40, d, generator=g) * 0.1
    # Project each direction onto the tangent plane at μ.
    raw = raw - (raw @ mu).unsqueeze(1) * mu
    return raw


def test_gde_compose_returns_unit():
    # _compose_query_gde must return a unit vector (it lives on S^{d-1}).
    d   = 64
    mu  = _rand_unit(1, d, 13)[0]
    v_ref = _same_hemisphere(_rand_unit(1, d, 14), mu)[0]
    dirs  = _make_synthetic_directions(mu, d, 15)
    q = _compose_query_gde(v_ref, ["Smiling"], ["Bald"], mu, dirs)
    assert abs(q.norm().item() - 1.0) < 1e-5


def test_lde_compose_returns_unit():
    # _compose_query_lde must return a unit vector (F.normalize guarantees it).
    d   = 64
    mu  = _rand_unit(1, d, 16)[0]
    v_ref = _rand_unit(1, d, 17)[0]
    dirs  = _make_synthetic_directions(mu, d, 18)
    q = _compose_query_lde(v_ref, ["Smiling"], ["Bald"], dirs)
    assert abs(q.norm().item() - 1.0) < 1e-5


def test_gde_no_conditions_close_to_ref():
    # With empty T_pos and T_neg the GDE query should be close to v_ref (only log→exp).
    d   = 64
    mu  = _rand_unit(1, d, 19)[0]
    v_ref = _same_hemisphere(_rand_unit(1, d, 20), mu)[0]
    dirs  = torch.zeros(40, d)
    q = _compose_query_gde(v_ref, [], [], mu, dirs)
    # exp_μ(log_μ(v_ref)) reconstructs v_ref up to float32 precision.
    assert torch.allclose(q, v_ref, atol=1e-5), (q - v_ref).abs().max().item()


def test_negation_reduces_attribute_component():
    # After negation, the attribute axis component of the query should be near zero.
    d   = 64
    mu  = _rand_unit(1, d, 21)[0]
    v_ref = _same_hemisphere(_rand_unit(1, d, 22), mu)[0]
    dirs  = _make_synthetic_directions(mu, d, 23)
    attr  = "Smiling"
    q = _compose_query_gde(v_ref, [], [attr], mu, dirs)
    q_tan = log_map(mu, q.unsqueeze(0)).squeeze(0)
    v_a   = dirs[ATTR_TO_IDX[attr]]
    v_hat = F.normalize(v_a, dim=0)
    component = abs((q_tan @ v_hat).item())
    assert component < 1e-4, component


# ---------------------------------------------------------------------------
# tier1_GDE.py — retrieval seam (CONTRACT §5/§7)
# ---------------------------------------------------------------------------

def test_ranking_is_full_permutation_gde():
    # get_ranking must return all N indices exactly once (a full permutation).
    N, d = 150, 64
    mu   = _rand_unit(1, d, 24)[0]
    feats = _same_hemisphere(_rand_unit(N, d, 25), mu)
    dirs  = _make_synthetic_directions(mu, d, 26)
    get_ranking = make_get_ranking("+Smiling", feats, mu, dirs, use_gde=True)
    ranking = get_ranking(0)
    assert sorted(ranking) == list(range(N))


def test_source_excluded_gde():
    # The source index must never appear in the top positions — it's pushed to last.
    N, d = 100, 64
    mu   = _rand_unit(1, d, 27)[0]
    feats = _same_hemisphere(_rand_unit(N, d, 28), mu)
    dirs  = _make_synthetic_directions(mu, d, 29)
    src   = 42
    get_ranking = make_get_ranking("+Smiling", feats, mu, dirs, use_gde=True)
    ranking = get_ranking(src)
    assert ranking[-1] == src


def test_ranking_is_full_permutation_lde():
    # Same contract check for the LDE ablation path.
    N, d = 150, 64
    mu   = _rand_unit(1, d, 30)[0]
    feats = _rand_unit(N, d, 31)
    dirs  = _make_synthetic_directions(mu, d, 32)
    get_ranking = make_get_ranking("+Smiling, -Bald", feats, mu, dirs, use_gde=False)
    ranking = get_ranking(7)
    assert sorted(ranking) == list(range(N))


def test_source_excluded_lde():
    N, d = 100, 64
    mu   = _rand_unit(1, d, 33)[0]
    feats = _rand_unit(N, d, 34)
    dirs  = _make_synthetic_directions(mu, d, 35)
    src   = 13
    get_ranking = make_get_ranking("+Smiling", feats, mu, dirs, use_gde=False)
    ranking = get_ranking(src)
    assert ranking[-1] == src


# ---------------------------------------------------------------------------
# mine_directions smoke test (synthetic attributes)
# ---------------------------------------------------------------------------

def test_mine_directions_shapes():
    # mine_directions must return (mu [d], directions [40, d]) with correct shapes.
    N, d = 300, 64
    feats = _rand_unit(N, d, 36)
    # Give every attribute at least 5 positive examples.
    attrs = torch.zeros(N, 40)
    g = torch.Generator().manual_seed(37)
    for j in range(40):
        idx = torch.randperm(N, generator=g)[:max(5, N // 4)]
        attrs[idx, j] = 1.0
    mu, directions = mine_directions(feats, attrs, n_iter=10)
    assert mu.shape == (d,)
    assert abs(mu.norm().item() - 1.0) < 1e-5
    assert directions.shape == (40, d)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  [OK] {t.__name__}")
    print(f"[OK] all {len(tests)} Tier-1 GDE tests passed")


if __name__ == "__main__":
    _run_all()
