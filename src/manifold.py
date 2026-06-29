"""
Riemannian primitives on the unit hypersphere S^{d-1}.

All inputs are assumed L2-normalized (unit rows / unit vectors). No normalisation
is enforced here — callers that pass non-unit vectors get geometrically wrong results.
Used exclusively by tier1_GDE.py; tier1_CLAY.py keeps its own _log_map to avoid a
sideways peer import (deliberate DRY trade-off, CLAUDE.md §imports).
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Log / Exp maps  (GDE.md App. A, Eq. 13–14)
# ---------------------------------------------------------------------------

def log_map(mu: torch.Tensor, X: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Log map — projects sphere points X onto the tangent plane at mu.
    # log_μ(x) = θ · (x − cosθ·μ) / sinθ,  θ = arccos(xᵀμ).
    # Points at μ (θ≈0) map to 0; the eps guard prevents 0/0.
    # Args: mu [d] unit vector; X [m, d] unit rows.  Returns [m, d] tangent vectors.
    dots = (X @ mu).clamp(-1.0, 1.0)          # cosθ per row  [m]
    theta = torch.arccos(dots)                 # geodesic distances  [m]
    tangent = X - dots.unsqueeze(1) * mu       # component ⊥ to μ, ‖·‖ = sinθ
    norms = tangent.norm(dim=1, keepdim=True)
    scale = torch.where(norms > eps, theta.unsqueeze(1) / norms, torch.zeros_like(norms))
    return tangent * scale                     # [m, d]


def exp_map(mu: torch.Tensor, V: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Exp map — lifts tangent vectors V at μ back onto the sphere.
    # exp_μ(v) = cos(‖v‖)·μ + sin(‖v‖)·(v/‖v‖).
    # Projects V onto T_μS first (removes any μ-component) so the formula holds
    # even when callers pass slightly non-tangent vectors (numerical drift).
    # Args: mu [d] unit vector; V [m, d] tangent vectors.  Returns [m, d] unit rows.
    V = V - (V @ mu).unsqueeze(1) * mu         # project onto tangent plane at μ
    norms = V.norm(dim=1, keepdim=True)        # ‖v‖ per row  [m, 1]
    v_hat = torch.where(norms > eps, V / norms, torch.zeros_like(V))
    return torch.cos(norms) * mu + torch.sin(norms) * v_hat   # [m, d]


# ---------------------------------------------------------------------------
# Intrinsic (Karcher) mean  (GDE.md App. A, Alg. 1)
# ---------------------------------------------------------------------------

def intrinsic_mean(
    X: torch.Tensor,
    weights: torch.Tensor | None = None,
    n_iter: int = 50,
    lr: float = 1.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    # Intrinsic (Karcher) mean — point on S^{d-1} minimising Σ w_i · d²(μ, x_i).
    # Iterates: μ ← Exp_μ( lr · Σ w_i Log_μ(x_i) ) until convergence (GDE Alg. 1).
    # Init: Euclidean mean normalised (warm start — converges in <20 iters on CelebA).
    # Args: X [m, d] unit rows; weights [m] (uniform if None).  Returns [d] unit vector.
    m = X.shape[0]
    if weights is None:
        weights = X.new_ones(m) / m
    else:
        weights = weights / weights.sum()      # normalise just in case

    mu = F.normalize(X.T @ weights, dim=0)    # warm-start: normalised weighted mean

    for _ in range(n_iter):
        logs = log_map(mu, X)                  # [m, d] tangent vectors
        grad = (weights.unsqueeze(1) * logs).sum(dim=0)   # [d] weighted tangent mean
        if grad.norm() < eps:
            break
        mu = F.normalize(exp_map(mu, (lr * grad).unsqueeze(0)).squeeze(0), dim=0)

    return mu                                  # [d] unit vector


# ---------------------------------------------------------------------------
# Tangent mean = primitive direction  (GDE.md §3.2, Prop. 1 / Eq. 7)
# ---------------------------------------------------------------------------

def tangent_mean(
    mu: torch.Tensor,
    X: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    # Tangent mean — primitive direction for an attribute on the sphere.
    # v_a = (1/|Z_a|) Σ_{x ∈ Z_a} Log_μ(x)  (GDE Prop. 1, Eq. 7).
    # Lives in T_μS^{d-1}; length encodes average angular displacement from μ.
    # (Note: NOT normalised — the magnitude carries signal for GDE composition.)
    # Args: mu [d] global mean; X [m, d] unit rows of images with attribute a.
    # Returns: [d] tangent vector (not necessarily unit).
    m = X.shape[0]
    if weights is None:
        weights = X.new_ones(m) / m
    else:
        weights = weights / weights.sum()
    logs = log_map(mu, X)                      # [m, d]
    return (weights.unsqueeze(1) * logs).sum(dim=0)   # [d]


# ---------------------------------------------------------------------------
# Rotation + subspace construction  (shared by tier1_CLAY, tier2a, tier2b, tier2c)
# ---------------------------------------------------------------------------

def align_rotation(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Minimal rotation H: a → b, identity on span{a,b}^⊥ (CLAY.md §3.2).
    # Closes the modality gap by rotating the visual mean onto the text mean without
    # disturbing intra-DB relationships. Returns I when a and b already (anti)coincide.
    d = a.shape[0]
    eye = torch.eye(d, dtype=a.dtype)
    c = float(a @ b)
    u2 = b - c * a
    s = u2.norm()
    if s < eps:
        return eye
    u2 = u2 / s
    P = torch.outer(a, a) + torch.outer(u2, u2)
    R = c * torch.outer(a, a) + s * torch.outer(u2, a) - s * torch.outer(a, u2) + c * torch.outer(u2, u2)
    return eye - P + R


def build_subspace(T_c: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Manifold-aware subspace — mu_c = normalize(mean(T_c)); SVD on log_{mu_c}(T_c) → V_k.
    # Returns (mu_c [d], V_k [d, k_eff]); k clamped to the stack height.
    # span(V_k) is the conditional similarity subspace (CLAY.md §3.2).
    mu_c = F.normalize(T_c.mean(dim=0), dim=0)
    L = log_map(mu_c, T_c)
    _, _, Vh = torch.linalg.svd(L, full_matrices=False)
    k_eff = min(k, Vh.shape[0])
    V_k = Vh[:k_eff].T                            # [d, k_eff]
    return mu_c, V_k
