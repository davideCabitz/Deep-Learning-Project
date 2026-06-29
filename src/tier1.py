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
from results_saver import save_results_csv, output_dir
# Closed-form hypersphere geometry lives in the shared manifold module (both Tier-1 and Track S
# depend on it — no peer-to-peer reuse). Aliased to the legacy private names this file already uses.
from manifold import log_map as _log_map, align_rotation as _align_rotation, build_subspace as _build_subspace


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
