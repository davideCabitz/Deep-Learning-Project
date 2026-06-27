"""
Tier-0 — vanilla latent-arithmetic baseline (the method to beat).

The simplest possible compositional retrieval: take the reference image's CLIP
vector, push it toward the positive attributes and away from the negatives by plain
vector addition in CLIP space, then rank the corpus by cosine similarity.

    query = v_ref + alpha * ( Σ t(+attr) − Σ t(−attr) )

No manifold awareness, no subspaces — that's Tier-1 (CLAY) and Tier-2 (ours). This
file exists to produce the first real Recall@K/Precision@K numbers (Milestone M1) and
the floor every fancier method must clear.

Plugs into the eval engine through the shared get_ranking callback (CONTRACT §5/§7).
Run:  python src/tier0.py
"""

import torch

from data_loader import ATTR_TO_IDX
from clip_features import load_image_features, load_attribute_text_features
from eval import parse_query, evaluate_all, format_results_table, load_eval_json, find_eval_json
from results_saver import save_results_csv, output_dir


def score(T_pos, T_neg, v_ref_idx, image_features, attr_text_features, alpha=1.0):
    # Tier-0 scorer (baseline, the floor every later tier must beat).
    # q = normalize( v_ref + α·(Σ t⁺ − Σ t⁻) ), then rank corpus by cosine.
    # Raw latent arithmetic — no subspaces, no learning. Returns dataset indices
    # best-first with the source excluded (CONTRACT §5).
    # (Note: adding raw text vectors to an image vector is geometrically crude
    # because of CLIP's modality gap — Tier-0 ENHANCED corrects exactly this.)
    query = image_features[v_ref_idx].clone()

    delta = torch.zeros_like(query)
    for name in T_pos:
        delta += attr_text_features[ATTR_TO_IDX[name]]
    for name in T_neg:
        delta -= attr_text_features[ATTR_TO_IDX[name]]
    query = query + alpha * delta

    # Unit-normalize so the corpus dot product IS cosine similarity.
    query = torch.nn.functional.normalize(query, p=2, dim=0)
    scores = image_features @ query

    scores[v_ref_idx] = float("-inf")        # never rank a source against itself (§5)

    return torch.argsort(scores, descending=True).tolist()


def make_get_ranking(query_str, image_features, attr_text_features, alpha=1.0):
    # Curry one parsed query into the get_ranking(src_idx) callback eval expects.
    T_pos, T_neg = parse_query(query_str)
    return lambda src_idx: score(T_pos, T_neg, src_idx, image_features, attr_text_features, alpha=alpha)


def evaluate_tier0(alpha=1.0, ks=(1, 5, 10), save=True):
    # Run Tier-0 over the full benchmark: load tables, score, print table, save CSV.
    image_features = load_image_features()
    attr_text_features = load_attribute_text_features()
    gt_list = load_eval_json(find_eval_json())

    def make(query_str):
        return make_get_ranking(query_str, image_features, attr_text_features, alpha=alpha)

    results = evaluate_all(gt_list, make, ks=ks)
    print(f"\nTier-0 (latent arithmetic, alpha={alpha}) - {len(gt_list)} queries\n")
    print(format_results_table(results, ks=ks))

    if save:
        save_results_csv(results, output_dir() / f"tier0_alpha{alpha}.csv", ks=ks)
    return results


if __name__ == "__main__":
    evaluate_tier0()
