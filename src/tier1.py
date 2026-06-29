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
<<<<<<< HEAD
from results_saver import save_results_csv, output_dir
=======
from results_saver import save_results_csv, output_subdir
>>>>>>> 76a6805d5fb14858c904c3a94d367b1b53664043


# ---------------------------------------------------------------------------
# Manifold math — closed forms on the unit hypersphere (GDE.md App. A.1)
# ---------------------------------------------------------------------------
def _log_map(mu, X, eps=1e-6):
    # Log map — log_μ(x) = θ·(x − cosθ·μ)/sinθ, θ = arccos(xᵀμ). Sends unit rows of X
    # to tangent vectors at μ whose length equals the geodesic distance (GDE eq. 14 / CLAY eq. 3).
    # Points at tangency (θ≈0) map to 0; eps guard prevents 0/0. Args: mu [d], X [m,d] → [m,d].
    dots = (X @ mu).clamp(-1.0, 1.0)                  # cos(theta) per row
    theta = torch.arccos(dots)
    tangent = X - dots.unsqueeze(1) * mu              # component of x orthogonal to mu, ‖·‖ = sin(theta)
    norms = tangent.norm(dim=1, keepdim=True)
    scale = torch.where(norms > eps, theta.unsqueeze(1) / norms, torch.zeros_like(norms))
    return tangent * scale


def _align_rotation(a, b, eps=1e-6):
    # Minimal rotation H: a → b, identity on span{a,b}^⊥ (CLAY.md §3.2).
    # Closes the modality gap by rotating the visual mean onto the text mean without
    # disturbing intra-DB relationships. Returns I when a and b already (anti)coincide.
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
    # Stack condition prompts into T_c [m, d] — CLAY's naïve pre-SVD merge (no +/− split).
    # Slices the true per-attribute count to strip padding (duplicate rows skew mu_c and SVD).
    # m = Σ_attr (#true prompts for that attr); positives and negatives merged — that's Tier-1's limit.
    rows = []
    for name in T_pos + T_neg:
        j = ATTRIBUTE_NAMES.index(name)
        n_j = len(build_prompts_for_attribute(name))
        rows.append(prompt_bank[j, :n_j])
    return torch.cat(rows, dim=0)


def _build_subspace(T_c, k):
    # Manifold-aware textual subspace — mu_c = normalize(mean(T_c)); SVD on log_{mu_c}(T_c) → V_k.
    # span(V_k) is the conditional similarity subspace (CLAY.md §3.2). k clamped to stack height.
    mu_c = torch.nn.functional.normalize(T_c.mean(dim=0), dim=0)
    L = _log_map(mu_c, T_c)
    # full_matrices=False → Vh is [min(m,d), d]; rows of Vh are the right singular vectors.
    _, _, Vh = torch.linalg.svd(L, full_matrices=False)
    k_eff = min(k, Vh.shape[0])
    V_k = Vh[:k_eff].T                                # [d, k_eff]
    return mu_c, V_k


def _project_db(image_features, mu_c, V_k, use_rotation=True):
    # Project frozen DB into the conditional subspace: D = normalize( log_{mu_c}(H·v) · V_k ).
    # Computed once per query; rows are L2-normalized so dot product == cosine. H closes the
    # modality gap (rotation ablation toggle); reference is just row src of D.
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
    # Build get_ranking(src_idx) → ranking, precomputing the subspace once per query.
    # Per-source cost is one [N,k]@[k] dot product — subspace (mu_c, V_k, D) is shared.
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
    # Single-shot scorer — CONTRACT §7 parity. Prefer make_get_ranking for full-benchmark
    # runs (it reuses the subspace across all sources of a query).
    T_c = _stack_condition_prompts(T_pos, T_neg, prompt_bank)
    mu_c, V_k = _build_subspace(T_c, k)
    D = _project_db(image_features, mu_c, V_k, use_rotation=use_rotation)
    scores = D @ D[v_ref_idx]
    scores[v_ref_idx] = float("-inf")
    return torch.argsort(scores, descending=True).tolist()


def evaluate_tier1(k=50, use_rotation=True, ks=(1, 5, 10), save=True):
    # Score Tier-1 (CLAY) on the full benchmark, print table, save CSV.
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
        save_results_csv(results, output_subdir("tier1") / f"tier1_k{k}_{rot}.csv", ks=ks)
    return results


if __name__ == "__main__":
    evaluate_tier1()
