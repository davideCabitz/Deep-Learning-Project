"""
Tier-1 — CLAY reproduction (the method-to-beat).

CLAY (Lim et al., 2026) reframes CLIP's space into a *text-conditional similarity space*: it
builds a manifold-aware textual subspace from the condition prompts and measures reference↔DB
similarity INSIDE that subspace, leaving the visual DB frozen (CLAY.md §3.2).

    T_c = stack of condition prompts                         # one SVD over ALL conditions
    mu_c = normalize(mean(T_c))                              # tangent point
    L = log_{mu_c}(T_c) = U S V^T  ;  V_k = top-k cols       # manifold-aware subspace
    m_CLAY(v) = V_k^T · log_{mu_c}( H(v) )                   # rotate (modality gap) → log → project
    csim(v_ref, v_d | c) = cos( m_CLAY(v_ref), m_CLAY(v_d) )

This is FAITHFUL pure CLAY: conditions are stacked naïvely pre-SVD (the exact bottleneck the
project attacks, project_specification.md §1) and there is NO +/− arithmetic — CLAY has no native
notion of negation, so '+' and '−' attributes are treated as undifferentiated condition axes. It is
a focus/preserve operation, not a query modification; the asymmetric +/− handling is Tier-2a's job.

Plugs into the eval engine through the shared get_ranking callback (CONTRACT §5/§7).
Run:  python src/tier1.py
"""

import torch

from data_loader import ATTRIBUTE_NAMES
from clip_features import load_image_features
from clip_prompts import load_prompt_bank, build_prompts_for_attribute
from eval import parse_query, evaluate_all, format_results_table, load_eval_json, find_eval_json
from results_io import save_results_csv, output_dir


# ---------------------------------------------------------------------------
# Manifold math — closed forms on the unit hypersphere (GDE.md App. A.1)
# ---------------------------------------------------------------------------
def _log_map(mu, X, eps=1e-6):
    """Logarithmic map of sphere points X onto the tangent space at mu (GDE.md eq. 14 / CLAY.md eq. 3).

    log_mu(x) = theta · (x − mu·(xᵀmu)) / sin(theta), theta = arccos(xᵀmu): sends each unit row of
    X to the tangent vector at mu whose length equals the geodesic (angular) distance. Points at the
    tangency (theta≈0) map to 0; the eps guard keeps the 0/0 there numerically safe.

    Args:
        mu : [d] unit tangent point.
        X  : [m, d] unit rows on the sphere.
    Returns:
        [m, d] tangent vectors (each orthogonal to mu).
    """
    dots = (X @ mu).clamp(-1.0, 1.0)                  # cos(theta) per row
    theta = torch.arccos(dots)
    tangent = X - dots.unsqueeze(1) * mu              # component of x orthogonal to mu, ‖·‖ = sin(theta)
    norms = tangent.norm(dim=1, keepdim=True)
    scale = torch.where(norms > eps, theta.unsqueeze(1) / norms, torch.zeros_like(norms))
    return tangent * scale


def _align_rotation(a, b, eps=1e-6):
    """Minimal rotation matrix H sending unit a → unit b, identity on span{a,b}^⊥ (CLAY.md §3.2).

    Rotates only within the 2-D plane spanned by the two means, so it aligns the visual mean to the
    text mean (closing the modality gap / conic effect) WITHOUT disturbing the intra-DB relationships
    the similarity depends on. Returns I when a and b already (anti)coincide.

    Args:
        a, b : [d] unit vectors.
    Returns:
        [d, d] rotation matrix H with H @ a ≈ b.
    """
    d = a.shape[0]
    I = torch.eye(d, dtype=a.dtype)
    c = float(a @ b)
    u2 = b - c * a                                    # part of b orthogonal to a
    s = u2.norm()
    if s < eps:                                       # parallel/antiparallel → nothing to rotate
        return I
    u2 = u2 / s
    # In the orthonormal plane basis (a, u2): a=(1,0)→(c,s), u2=(0,1)→(−s,c). Off-plane left untouched.
    P = torch.outer(a, a) + torch.outer(u2, u2)       # projector onto the plane
    R = c * torch.outer(a, a) + s * torch.outer(u2, a) - s * torch.outer(a, u2) + c * torch.outer(u2, u2)
    return I - P + R


# ---------------------------------------------------------------------------
# Subspace construction + DB projection
# ---------------------------------------------------------------------------
def _stack_condition_prompts(T_pos, T_neg, prompt_bank):
    """Stack every condition's (unpadded) prompt vectors into one matrix T_c (CLAY's naïve pre-SVD stack).

    The cached bank is padded to a common width with duplicate rows; the true per-attribute count
    comes from build_prompts_for_attribute() (deterministic, no CLIP), so we slice off the padding —
    duplicate rows would skew mu_c and the singular values. Positives and negatives are merged here:
    pure CLAY draws no +/− distinction.

    Returns:
        [m, d] unit rows, m = Σ_attr (#prompts for that attr).
    """
    rows = []
    for name in T_pos + T_neg:
        j = ATTRIBUTE_NAMES.index(name)
        n_j = len(build_prompts_for_attribute(name))
        rows.append(prompt_bank[j, :n_j])
    return torch.cat(rows, dim=0)


def _build_subspace(T_c, k):
    """Manifold-aware textual subspace: tangent point mu_c + top-k right singular vectors V_k.

    mu_c = normalized mean of the condition texts; SVD on their log-mapped (tangent) images yields the
    directions of greatest spread → span(V_k) is the conditional similarity subspace (CLAY.md §3.2).
    k is clamped to the stack height m (with m few prompts the paper's k=50 is unreachable).

    Returns:
        (mu_c [d], V_k [d, k_eff]).
    """
    mu_c = torch.nn.functional.normalize(T_c.mean(dim=0), dim=0)
    L = _log_map(mu_c, T_c)
    # full_matrices=False → Vh is [min(m,d), d]; rows of Vh are the right singular vectors.
    _, _, Vh = torch.linalg.svd(L, full_matrices=False)
    k_eff = min(k, Vh.shape[0])
    V_k = Vh[:k_eff].T                                # [d, k_eff]
    return mu_c, V_k


def _project_db(image_features, mu_c, V_k, use_rotation=True):
    """Project the whole frozen DB into the conditional subspace: D = normalize_rows( log_{mu_c}(H·v) · V_k ).

    Computed ONCE per query (it never depends on the source). Each row is L2-normalized so a later dot
    product is the cosine CLAY scores with; the reference is simply row src of D (it lives in the same
    corpus). H aligns the visual mean to the text mean unless the rotation ablation is off.

    Returns:
        [N, k_eff] unit-normalized subspace coordinates.
    """
    V = image_features
    if use_rotation:
        mu_vd = torch.nn.functional.normalize(image_features.mean(dim=0), dim=0)
        H = _align_rotation(mu_vd, mu_c)
        V = V @ H.T                                   # rotate every DB vector (still unit norm)
    coords = _log_map(mu_c, V) @ V_k                  # [N, k_eff]
    return torch.nn.functional.normalize(coords, dim=1, eps=1e-12)


# ---------------------------------------------------------------------------
# Retrieval — the shared score / get_ranking seam (CONTRACT §7)
# ---------------------------------------------------------------------------
def make_get_ranking(query_str, image_features, prompt_bank, k=50, use_rotation=True):
    """Build the get_ranking(src_idx) → ranking callback, precomputing the per-query subspace once.

    mu_c, V_k, H and the projected DB depend only on the query, so they are built here and reused for
    every source of that query (mirrors tier0.make_get_ranking); per-source cost is one [N,k]@[k]
    cosine. This is the efficient path the eval engine calls.
    """
    T_pos, T_neg = parse_query(query_str)
    T_c = _stack_condition_prompts(T_pos, T_neg, prompt_bank)
    mu_c, V_k = _build_subspace(T_c, k)
    D = _project_db(image_features, mu_c, V_k, use_rotation=use_rotation)

    def get_ranking(src_idx):
        scores = D @ D[src_idx]                       # cosine in the subspace (rows are unit)
        scores[src_idx] = float("-inf")               # source never ranks itself (CONTRACT §5)
        return torch.argsort(scores, descending=True).tolist()

    return get_ranking


def score(T_pos, T_neg, v_ref_idx, image_features, prompt_bank, k=50, use_rotation=True):
    """Rank the corpus for one (query, source) pair via CLAY — CONTRACT §7-parity single-shot wrapper.

    Convenience entry matching the shared score(...) shape; builds the subspace then ranks once. For
    full-benchmark scoring prefer make_get_ranking (it reuses the subspace across a query's sources).

    Returns:
        ranking : list[int], best-first, source excluded (CONTRACT §5).
    """
    T_c = _stack_condition_prompts(T_pos, T_neg, prompt_bank)
    mu_c, V_k = _build_subspace(T_c, k)
    D = _project_db(image_features, mu_c, V_k, use_rotation=use_rotation)
    scores = D @ D[v_ref_idx]
    scores[v_ref_idx] = float("-inf")
    return torch.argsort(scores, descending=True).tolist()


def evaluate_tier1(k=50, use_rotation=True, ks=(1, 5, 10), save=True):
    """Score Tier-1 (CLAY) on the full evaluation benchmark, print the table, and save CSV."""
    image_features = load_image_features()
    prompt_bank = load_prompt_bank()
    gt_list = load_eval_json(find_eval_json())

    def make(query_str):
        return make_get_ranking(query_str, image_features, prompt_bank, k=k, use_rotation=use_rotation)

    results = evaluate_all(gt_list, make, ks=ks)
    rot = "rotH" if use_rotation else "norot"
    print(f"\nTier-1 (CLAY, k={k}, {rot}) - {len(gt_list)} queries\n")
    print(format_results_table(results, ks=ks))

    if save:
        save_results_csv(results, output_dir() / f"tier1_k{k}_{rot}.csv", ks=ks)
    return results


if __name__ == "__main__":
    evaluate_tier1()
