"""
Tier-3F — CLAY symmetric subspace negation (training-free, DGP backbone).

Negation as orthogonal-complement projection of a per-attribute CLAY subspace, applied
SYMMETRICALLY to both the query and the frozen DB (CLAY §3.1 Eq.1):
    per b∈T−:  V_b = CLAY subspace of attribute b's prompt stack (log_map@μ_b → SVD → top-k)
               strip the V_b component from every vector:  v ↦ v − V_b(V_bᵀ·log_{μ_b}(H·v))
    csim = cos( m(v_ref), m(v_d) )   measured in the complement space

Prior negation removes a single VISUAL attribute axis from the query only (asymmetric). CLAY's
symmetric formulation instead reshapes the similarity SPACE: it removes the whole negated-attribute
subspace from BOTH sides, so two images are "similar under −b" only on dimensions orthogonal to b.
This is the geometry-only counterpart to tier3_negsteer's text-space steering — same goal, no
classifier, pure manifold algebra (reusing tier1_CLAY wholesale).

Affirmation + identity stay the DGP composition (tier2d_dgp); negation is done at scoring time by
the complement projection, not by tangent-space rejection. Training-free, frozen DB never
re-encoded (only projected via matmul), spec-compliant (§3.2 step 2) and CLAY-faithful (§3.2).

Run:  python src/tier3_symneg.py
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from data_loader import ATTRIBUTE_NAMES, ATTR_TO_IDX
from clip_features import load_image_features
from clip_prompts import load_prompt_bank, build_prompts_for_attribute
from eval import parse_query, evaluate_query, format_results_table, load_eval_json, find_eval_json
from results_saver import save_results_csv, output_subdir
from manifold import log_map, exp_map
from tier1_GDE import load_or_mine_directions
from tier1_CLAY import _log_map, _align_rotation, _build_subspace
from tier2d_dgp import DGPConfig, compute_mu_txt, _centered_stack, _gated_directions_batch


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SymNegConfig:
    # Tier-3F — DGP composition + CLAY symmetric subspace negation.
    tau: float = 0.07              # DGP gate temperature (inherited)
    alpha: float = 1.0             # DGP positive push strength (inherited)
    center: bool = True            # DGP modality-gap centering (inherited)
    k: int = 50                    # CLAY subspace rank (paper default)
    seed: int = 0

    def tag(self) -> str:
        return f"symneg_k{self.k}_tau{self.tau:g}_a{self.alpha:g}"


# ---------------------------------------------------------------------------
# Per-attribute negation subspace + orthogonal-complement projection
# ---------------------------------------------------------------------------

def _neg_subspace(neg_names: list[str], prompt_bank: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    # ONE CLAY subspace (μ_b, V_b) over ALL negated attributes' UNPADDED prompt stacks stacked
    # pre-SVD (CLAY-faithful, §3.2). Building a single subspace for the whole T− set avoids the
    # geometry error of chaining log_maps on already-projected non-unit residuals — the complement
    # is taken once, at a single tangent point μ_b, exactly as CLAY defines its condition subspace.
    rows = []
    for name in neg_names:
        j = ATTR_TO_IDX[name]
        n_j = len(build_prompts_for_attribute(name))
        rows.append(prompt_bank[j, :n_j])
    T_b = torch.cat(rows, dim=0)                             # [Σ n_b, d]
    return _build_subspace(T_b, k)                           # (μ_b [d], V_b [d, k_eff])


def _complement_project(
    features: torch.Tensor, mu_b: torch.Tensor, V_b: torch.Tensor, mu_vis: torch.Tensor,
) -> torch.Tensor:
    # Project features onto the ORTHOGONAL COMPLEMENT of V_b in the tangent space at μ_b:
    #   rotate (modality gap) → log_map@μ_b → t − V_b(V_bᵀt) → normalize.
    # Rows expressing attribute b collapse in the removed dims and can no longer rank on that axis.
    H = _align_rotation(mu_vis, mu_b)
    rotated = features @ H.T                                 # [N, d] (unit after rotation)
    t = _log_map(mu_b, rotated)                              # [N, d] tangent at μ_b
    residual = t - (t @ V_b) @ V_b.T                        # [N, d] orthogonal complement
    return F.normalize(residual, dim=1, eps=1e-12)


# ---------------------------------------------------------------------------
# Composition — DGP affirmation only (negation handled at scoring time)
# ---------------------------------------------------------------------------

def _compose_pos_only(
    V_ref: torch.Tensor,
    T_pos: list[str],
    mu: torch.Tensor,
    stacks: dict[str, torch.Tensor],
    cfg: SymNegConfig,
) -> torch.Tensor:
    # DGP identity + positive-attribute composition (no negation here — that is the complement's job).
    q_tan = log_map(mu, V_ref)
    for name in T_pos:
        d_c = _gated_directions_batch(V_ref, stacks[name], cfg.tau)
        q_tan = q_tan + cfg.alpha * d_c
    return F.normalize(exp_map(mu, q_tan), dim=1)


# ---------------------------------------------------------------------------
# Retrieval seam — CONTRACT §5/§7
# ---------------------------------------------------------------------------

def make_get_ranking(
    query_str: str,
    image_features: torch.Tensor,
    mu: torch.Tensor,
    prompt_bank: torch.Tensor,
    mu_txt: torch.Tensor,
    mu_vis: torch.Tensor,
    cfg: SymNegConfig = SymNegConfig(),
) -> callable:
    # Compose q (identity + T+), then score in the complement of each T− attribute's CLAY subspace.
    # For multiple T− attributes the complements are applied sequentially (DB projected once/query).
    # With no T−, this reduces to plain DGP cosine (positive-only path).
    T_pos, T_neg = parse_query(query_str)
    stacks = {name: _centered_stack(name, prompt_bank, mu_txt, cfg.center) for name in T_pos}

    # Build the projected DB once per query from a SINGLE subspace over all T− attributes.
    if T_neg:
        mu_b, V_b = _neg_subspace(T_neg, prompt_bank, cfg.k)
        DB_proj = _complement_project(image_features, mu_b, V_b, mu_vis)   # [N, d] unit in complement
    else:
        DB_proj = image_features

    def get_ranking(src_idx: int) -> list[int]:
        q = _compose_pos_only(image_features[src_idx].unsqueeze(0), T_pos, mu, stacks, cfg).squeeze(0)
        if T_neg:
            # Project the query into the same complement, then cosine-score.
            q_proj = _complement_project(q.unsqueeze(0), mu_b, V_b, mu_vis).squeeze(0)
            scores = DB_proj @ q_proj
        else:
            scores = DB_proj @ q
        scores[src_idx] = float("-inf")
        return torch.argsort(scores, descending=True).tolist()

    return get_ranking


# ---------------------------------------------------------------------------
# Evaluation entry point
# ---------------------------------------------------------------------------

def evaluate_symneg(cfg: SymNegConfig = SymNegConfig(), ks=(1, 5, 10), save: bool = True) -> dict:
    # Score CLAY symmetric subspace negation on the 14-query benchmark, print table, save CSV.
    image_features = load_image_features()
    prompt_bank = load_prompt_bank()
    mu, _ = load_or_mine_directions()
    mu_txt = compute_mu_txt(prompt_bank)
    mu_vis = F.normalize(image_features.mean(dim=0), dim=0)
    gt_list = load_eval_json(find_eval_json())

    results = {}
    for entry in gt_list:
        get_ranking = make_get_ranking(
            entry["query"], image_features, mu, prompt_bank, mu_txt, mu_vis, cfg,
        )
        results[entry["query"]] = evaluate_query(entry["ground_truth"], get_ranking, ks)

    print(f"\nTier-3F SymNeg ({cfg.tag()}) — {len(gt_list)} queries\n")
    print(format_results_table(results, ks=ks))

    if save:
        save_results_csv(results, output_subdir("tier3_symneg") / f"tier3_{cfg.tag()}.csv", ks=ks)
    return results


if __name__ == "__main__":
    # Default k, plus a small global sweep of the subspace rank.
    for k in (20, 50, 100):
        evaluate_symneg(SymNegConfig(k=k))
