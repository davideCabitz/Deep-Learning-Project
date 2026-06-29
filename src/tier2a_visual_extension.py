"""
Tier-2a Track V — Extended Visual-Prototype Compositional Retrieval.

Extends tier1_GDE.py (GDE/LDE) with three training-free improvements,
each independently togglable as an ablation axis:

  1. CLIP-weighted direction mining (GDE §3.3.1, Prop. 2) — replaces uniform
     tangent means with per-image weights w_i = softmax(f_i @ T.T)[a], so
     images strongly matching attribute a's text prompt contribute more.
     Requires no new Colab extraction: clip_attr_text_features.pt already exists.

  2. Configurable push strength α (GDE §4.5) — scales every positive direction
     before geodesic addition: q_tan += α·v_a. α=1.0 reproduces the base method.

  3. Joint subspace negation (PoS-Subspaces §2.2, Eq. 5) — for multi-attribute
     negation, builds W = stack(v̂_a for a∈T_neg) and projects onto its orthogonal
     complement in one step via W⁺ (pseudoinverse), instead of sequential scalar
     rejections which are order-dependent and don't orthogonalise the axes.

Run:  python src/tier2a_visual_extension.py
"""

import torch
import torch.nn.functional as F
from pathlib import Path

from data_loader import ATTRIBUTE_NAMES, ATTR_TO_IDX, _get_artifacts_dir
from clip_features import load_image_features, load_attribute_text_features
from eval import parse_query, evaluate_all, format_results_table, load_eval_json, find_eval_json
from results_saver import save_results_csv, output_subdir
from manifold import log_map, exp_map, tangent_mean
from tier1_GDE import (
    _load_train_features,
    _load_train_attributes,
    load_or_mine_directions,
    _compose_query_lde,
)


# ---------------------------------------------------------------------------
# CLIP-weighted direction mining  (GDE §3.3.1, Prop. 2)
# ---------------------------------------------------------------------------

def _compute_clip_weights(
    train_features: torch.Tensor,
    text_features: torch.Tensor,
) -> torch.Tensor:
    # Per-image, per-attribute CLIP softmax weights.
    # w[i, a] = softmax(train_features[i] @ text_features.T)[a].
    # Measures how strongly image i matches attribute a's text prompt relative
    # to all other attributes — the denoising weight from GDE §3.3.1, Eq. 12.
    # Returns [N_train, 40] float32.
    logits = train_features @ text_features.T     # [N_train, 40]
    return F.softmax(logits, dim=1)               # [N_train, 40]


def mine_weighted_directions(
    train_features: torch.Tensor,
    train_attributes: torch.Tensor,
    mu: torch.Tensor,
    n_iter: int = 50,
) -> torch.Tensor:
    # CLIP-weighted tangent means — denoised primitive directions (GDE Prop. 2).
    # v_a = Σ_{i: has_a} w[i,a] · Log_μ(x_i) / Σ w[i,a], where w is the CLIP
    # image-text softmax. Down-weights images where attribute a is incidental noise.
    # (Note: mu is passed in rather than recomputed — caller owns the mean so both
    # uniform and weighted directions share the same tangent point, keeping them
    # comparable as a clean ablation.)
    text_features = load_attribute_text_features()   # [40, 512]
    clip_weights = _compute_clip_weights(train_features, text_features)  # [N_train, 40]

    d = train_features.shape[1]
    directions = torch.zeros(len(ATTRIBUTE_NAMES), d, dtype=train_features.dtype)

    for j, name in enumerate(ATTRIBUTE_NAMES):
        mask = train_attributes[:, j] > 0.5              # [N_train] bool
        has_a = train_features[mask]
        if has_a.shape[0] == 0:
            print(f"  [!] No train images with attribute '{name}' — direction set to zero.")
            continue
        w = clip_weights[mask, j]                         # [k] weights for has-a images
        directions[j] = tangent_mean(mu, has_a, weights=w)
        print(f"  [{j+1:02d}/40] {name}: {mask.sum().item()} images", end="\r")

    print("\n[OK] Weighted direction mining complete.")
    return directions                                     # [40, 512]


def load_or_mine_weighted_directions(
    force: bool = False,
    n_iter: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Load cached weighted directions if available, else mine and cache.
    # Uses a separate cache key from the uniform variant so both can coexist.
    # mu is always loaded from the uniform cache — it's the shared tangent point.
    cache_path = _get_artifacts_dir() / "visual_directions_weighted.pt"
    mu, _ = load_or_mine_directions(n_iter=n_iter)        # reuse uniform mu

    if cache_path.exists() and not force:
        print(f"[OK] Loading cached weighted directions: {cache_path}")
        directions = torch.load(cache_path, weights_only=True)
        return mu, directions

    train_features   = _load_train_features()
    train_attributes = _load_train_attributes()
    print("Computing CLIP-weighted directions…")
    directions = mine_weighted_directions(train_features, train_attributes, mu, n_iter=n_iter)

    torch.save(directions, cache_path)
    print(f"  Saved: {cache_path}")
    return mu, directions


# ---------------------------------------------------------------------------
# Extended query composition
# ---------------------------------------------------------------------------

def _compose_query_ext(
    v_ref: torch.Tensor,
    T_pos: list[str],
    T_neg: list[str],
    mu: torch.Tensor,
    directions: torch.Tensor,
    alpha: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    # Extended GDE composition — geodesic addition with α scaling + joint subspace negation.
    # Affirmation: q_tan = Log_μ(v_ref) + Σ_{a∈T_pos} α·v_a  → Exp_μ(q_tan).
    # Negation: build W = orth(stack(v̂_a for a∈T_neg)), project q_tan onto W⊥ in one step.
    # Joint projection via W⁺ avoids order-dependence of sequential scalar rejections and
    # correctly handles non-orthogonal attribute axes (PoS-Subspaces §2.2, Eq. 5).
    q_tan = log_map(mu, v_ref.unsqueeze(0)).squeeze(0)   # [d]

    for name in T_pos:
        q_tan = q_tan + alpha * directions[ATTR_TO_IDX[name]]

    if T_neg:
        # Stack unit directions for all negated attributes → [k, d].
        neg_vecs = []
        for name in T_neg:
            v_a = directions[ATTR_TO_IDX[name]]
            norm = v_a.norm()
            if norm > eps:
                neg_vecs.append(v_a / norm)

        if neg_vecs:
            W = torch.stack(neg_vecs, dim=0)              # [k, d]
            # Orthogonalise W via thin QR so columns of Q span the same subspace
            # but are orthonormal — makes the projection numerically stable.
            Q, _ = torch.linalg.qr(W.T)                  # Q: [d, k]
            # Π⊥(q_tan) = q_tan − Q(Qᵀq_tan)  (project onto complement of span(W))
            q_tan = q_tan - Q @ (Q.T @ q_tan)

    return F.normalize(
        exp_map(mu, q_tan.unsqueeze(0)).squeeze(0), dim=0
    )


# ---------------------------------------------------------------------------
# Retrieval seam
# ---------------------------------------------------------------------------

def make_get_ranking_ext(
    query_str: str,
    image_features: torch.Tensor,
    mu: torch.Tensor,
    directions: torch.Tensor,
    alpha: float = 1.0,
    use_gde: bool = True,
) -> callable:
    # Build the get_ranking(src_idx) → list[int] callback for the extended method.
    T_pos, T_neg = parse_query(query_str)

    def get_ranking(src_idx: int) -> list[int]:
        v_ref = image_features[src_idx]

        if use_gde:
            q = _compose_query_ext(v_ref, T_pos, T_neg, mu, directions, alpha=alpha)
        else:
            q = _compose_query_lde(v_ref, T_pos, T_neg, directions)

        scores = image_features @ q
        scores[src_idx] = float("-inf")
        return torch.argsort(scores, descending=True).tolist()

    return get_ranking


# ---------------------------------------------------------------------------
# Evaluation entry points
# ---------------------------------------------------------------------------

def _run_evaluate_ext(
    tag: str,
    use_gde: bool,
    weighted: bool,
    alpha: float,
    ks: tuple = (1, 5, 10),
    save: bool = True,
) -> dict:
    # Shared evaluation loop for all extension variants.
    # tag encodes the full variant name written to the CSV filename.
    image_features = load_image_features()

    if weighted:
        mu, directions = load_or_mine_weighted_directions()
    else:
        mu, directions = load_or_mine_directions()

    gt_list = load_eval_json(find_eval_json())

    def make(query_str: str):
        return make_get_ranking_ext(
            query_str, image_features, mu, directions,
            alpha=alpha, use_gde=use_gde,
        )

    results = evaluate_all(gt_list, make, ks=ks)
    print(f"\nTier-2a Visual Ext ({tag}) — {len(gt_list)} queries\n")
    print(format_results_table(results, ks=ks))

    if save:
        save_results_csv(results, output_subdir("tier2a_visual_ext") / f"tier2a_visual_ext_{tag}.csv", ks=ks)
    return results


def evaluate_gde_weighted(alpha: float = 1.0, ks=(1, 5, 10), save=True) -> dict:
    # Tier-2a Ext — GDE + CLIP-weighted directions + configurable α (full method).
    return _run_evaluate_ext("gde_weighted", use_gde=True, weighted=True, alpha=alpha, ks=ks, save=save)


def evaluate_gde_uniform_alpha(alpha: float = 1.0, ks=(1, 5, 10), save=True) -> dict:
    # Tier-2a Ext — GDE + uniform directions + α sweep (isolates α contribution).
    return _run_evaluate_ext(f"gde_alpha{alpha}", use_gde=True, weighted=False, alpha=alpha, ks=ks, save=save)


def evaluate_lde_weighted(alpha: float = 1.0, ks=(1, 5, 10), save=True) -> dict:
    # Tier-2a Ext — LDE ablation + CLIP-weighted directions (flat geometry, denoised).
    return _run_evaluate_ext("lde_weighted", use_gde=False, weighted=True, alpha=alpha, ks=ks, save=save)


if __name__ == "__main__":
    # Run the full ablation grid: weighted vs uniform × α ∈ {0.5, 1.0, 1.5}.
    for a in (0.5, 1.0, 1.5):
        evaluate_gde_uniform_alpha(alpha=a)
    evaluate_gde_weighted(alpha=1.0)
    evaluate_lde_weighted(alpha=1.0)
