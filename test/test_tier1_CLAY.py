"""
Tier-1 (CLAY) verification — kept out of src/ per project convention (tests live in test/).

Cheap, self-contained checks of the manifold machinery and the retrieval seam. No CLIP, no real
feature DB: the geometry is exercised on random unit vectors and a tiny synthetic corpus, so this
runs in well under a second. Run:  python test/test_tier1_CLAY.py   (or: pytest test/test_tier1_CLAY.py)
"""

import sys
from pathlib import Path

import torch

# src/ uses bare imports (e.g. `from data_loader import ...`); put it on the path so we can import tier1_CLAY.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tier1_CLAY import _log_map, _align_rotation, _build_subspace, _stack_condition_prompts, make_get_ranking
from clip_prompts import build_prompts_for_attribute
from data_loader import ATTRIBUTE_NAMES


def _exp_map(mu, V):
    """Exponential map T_mu → sphere (GDE.md eq. 13) — the inverse of _log_map, only needed to test it."""
    norms = V.norm(dim=1, keepdim=True)
    out = torch.cos(norms) * mu + torch.sin(norms) * torch.where(norms > 0, V / norms, V)
    return out


def _rand_unit(n, d, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(n, d, generator=g), dim=1)


def test_log_exp_roundtrip():
    """exp_mu(log_mu(x)) ≈ x for points in the same hemisphere as mu (where the maps invert)."""
    mu = _rand_unit(1, 64, 0)[0]
    # Keep points reasonably near mu (xᵀmu > 0) so we stay inside the injectivity radius.
    X = _rand_unit(50, 64, 1)
    X = torch.where((X @ mu).unsqueeze(1) < 0, -X, X)
    recon = _exp_map(mu, _log_map(mu, X))
    assert torch.allclose(recon, X, atol=1e-5), (recon - X).abs().max().item()


def test_log_map_at_tangency_is_zero():
    """A point exactly at mu maps to the tangent-space origin (the eps guard must not blow up)."""
    mu = _rand_unit(1, 32, 2)[0]
    out = _log_map(mu, mu.unsqueeze(0))
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-7)


def test_align_rotation():
    """H sends a→b, is orthogonal (HᵀH=I, norm-preserving), and leaves span{a,b}^⊥ untouched."""
    a = _rand_unit(1, 48, 3)[0]
    b = _rand_unit(1, 48, 4)[0]
    H = _align_rotation(a, b)
    assert torch.allclose(H @ a, b, atol=1e-5), (H @ a - b).abs().max().item()
    assert torch.allclose(H.T @ H, torch.eye(48), atol=1e-5)          # orthogonal
    x = _rand_unit(1, 48, 5)[0]
    assert abs((H @ x).norm() - x.norm()) < 1e-5                       # preserves length
    # A vector orthogonal to the plane span{a,b} is fixed by H. Build the plane's orthonormal
    # basis {a, u2} (a single Gram-Schmidt pass against a,b would NOT orthogonalize, since a⊥̸b).
    u2 = torch.nn.functional.normalize(b - (a @ b) * a, dim=0)
    w = _rand_unit(1, 48, 6)[0]
    w = w - (w @ a) * a - (w @ u2) * u2
    w = torch.nn.functional.normalize(w, dim=0)
    assert torch.allclose(H @ w, w, atol=1e-5)


def test_align_rotation_parallel_is_identity():
    """Already-aligned means → no rotation (avoids a 0/0 in the plane construction)."""
    a = _rand_unit(1, 16, 7)[0]
    assert torch.allclose(_align_rotation(a, a), torch.eye(16), atol=1e-7)


def test_subspace_orthonormal_and_idempotent():
    """V_k has orthonormal columns, so P_c = V_k V_kᵀ is an idempotent projector and fixes its own span."""
    T_c = _rand_unit(20, 64, 8)
    _, V_k = _build_subspace(T_c, k=5)
    assert V_k.shape == (64, 5)
    assert torch.allclose(V_k.T @ V_k, torch.eye(5), atol=1e-5)        # orthonormal columns
    P = V_k @ V_k.T
    assert torch.allclose(P @ P, P, atol=1e-5)                        # idempotent
    x = V_k[:, 0]                                                      # already in the span
    assert torch.allclose(P @ x, x, atol=1e-5)


def test_k_clamped_to_stack_height():
    """With few prompts the paper's large k is unreachable; k_eff must clamp to the stack height."""
    T_c = _rand_unit(4, 64, 9)
    _, V_k = _build_subspace(T_c, k=50)
    assert V_k.shape[1] == 4


def test_stack_strips_padding():
    """The stacked condition matrix uses the TRUE prompt count per attribute, not the padded width."""
    name = "Smiling"
    n_true = len(build_prompts_for_attribute(name))
    width = n_true + 7                                                 # padded bank is wider than reality
    bank = _rand_unit(40 * width, 512, 10).reshape(40, width, 512)
    T_c = _stack_condition_prompts([name], [], bank)
    assert T_c.shape == (n_true, 512)


def test_get_ranking_end_to_end():
    """A 1-attribute query runs end-to-end: ranking is a full permutation with the source ranked last."""
    name = "Smiling"
    width = len(build_prompts_for_attribute(name)) + 5
    bank = _rand_unit(40 * width, 512, 11).reshape(40, width, 512)
    N = 200
    image_features = _rand_unit(N, 512, 12)
    get_ranking = make_get_ranking(f"+{name}", image_features, bank, k=10)
    src = 42
    ranking = get_ranking(src)
    assert sorted(ranking) == list(range(N))                          # a permutation of all indices
    assert ranking[-1] == src                                         # source excluded → pushed to the end


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  [OK] {t.__name__}")
    print(f"[OK] all {len(tests)} Tier-1 tests passed")


if __name__ == "__main__":
    _run_all()
