"""
Riemannian primitives on the unit hypersphere S^{d-1}.

All inputs are assumed L2-normalized (unit rows / unit vectors). No normalisation
is enforced here — callers that pass non-unit vectors get geometrically wrong results.
Used exclusively by tier2a_visual.py; tier1.py keeps its own _log_map to avoid a
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
    # Zero tangent vectors map back to μ (the eps guard keeps v/‖v‖ safe).
    # Args: mu [d] unit vector; V [m, d] tangent vectors.  Returns [m, d] unit rows.
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
        mu = F.normalize(exp_map(mu.unsqueeze(0), (lr * grad).unsqueeze(0)).squeeze(0), dim=0)

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
