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

import csv
from pathlib import Path

import torch

from data_loader import ATTR_TO_IDX
from clip_features import load_image_features, load_attribute_text_features
from eval import parse_query, evaluate_all, format_results_table, load_eval_json, find_eval_json


def _output_dir():
    """Project-root output/ folder for saved results (created on demand)."""
    out = Path(__file__).parent.parent / "output"
    out.mkdir(exist_ok=True)
    return out


def score(T_pos, T_neg, v_ref_idx, image_features, attr_text_features, alpha=1.0):
    """
    Rank the corpus for one (query, source image) pair via latent arithmetic.

    Args:
        T_pos, T_neg     : lists of attribute names (from parse_query).
        v_ref_idx        : int, the source image's dataset index.
        image_features   : [N, 512] L2-normalized image table (CONTRACT §6).
        attr_text_features : [40, 512] L2-normalized attribute text table (row j == attr j).
        alpha            : weight on the text delta (1.0 = vanilla; tunable for ablation).

    Returns:
        ranking : list[int] of dataset indices, best-first, source excluded (CONTRACT §5).
    """
    query = image_features[v_ref_idx].clone()

    delta = torch.zeros_like(query)
    for name in T_pos:
        delta += attr_text_features[ATTR_TO_IDX[name]]
    for name in T_neg:
        delta -= attr_text_features[ATTR_TO_IDX[name]]
    query = query + alpha * delta

    # Cosine similarity == dot product since both sides are unit-normalized.
    query = torch.nn.functional.normalize(query, p=2, dim=0)
    scores = image_features @ query

    # The source image must never appear in its own ranking (CONTRACT §5).
    scores[v_ref_idx] = float("-inf")

    ranking = torch.argsort(scores, descending=True).tolist()
    return ranking


def make_get_ranking(query_str, image_features, attr_text_features, alpha=1.0):
    """Build the get_ranking(src_idx) -> ranking callback the eval engine expects."""
    T_pos, T_neg = parse_query(query_str)
    return lambda src_idx: score(T_pos, T_neg, src_idx, image_features, attr_text_features, alpha=alpha)


def save_results_csv(results, path, ks=(1, 5, 10)):
    """Write evaluate_all() output to CSV: one row per query + a final mean row.

    Columns: query, R@k..., P@k..., num_sources. Opens cleanly in Excel/pandas.
    """
    cols = ["query"] + [f"R@{k}" for k in ks] + [f"P@{k}" for k in ks] + ["num_sources"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for query, res in results.items():
            writer.writerow(
                [query]
                + [f"{res[f'recall@{k}']:.4f}" for k in ks]
                + [f"{res[f'precision@{k}']:.4f}" for k in ks]
                + [res["num_sources"]]
            )
        # Macro-average over queries (each query weighted equally).
        n = len(results)
        mean_row = ["MEAN"]
        for metric in [f"recall@{k}" for k in ks] + [f"precision@{k}" for k in ks]:
            mean_row.append(f"{sum(r[metric] for r in results.values()) / n:.4f}")
        mean_row.append("")
        writer.writerow(mean_row)
    print(f"  Saved: {path}")


def evaluate_tier0(alpha=1.0, ks=(1, 5, 10), save=True):
    """Score Tier-0 on the full evaluation benchmark, print the table, and save CSV."""
    image_features = load_image_features()
    attr_text_features = load_attribute_text_features()
    gt_list = load_eval_json(find_eval_json())

    def make(query_str):
        return make_get_ranking(query_str, image_features, attr_text_features, alpha=alpha)

    results = evaluate_all(gt_list, make, ks=ks)
    print(f"\nTier-0 (latent arithmetic, alpha={alpha}) - {len(gt_list)} queries\n")
    print(format_results_table(results, ks=ks))

    if save:
        save_results_csv(results, _output_dir() / f"tier0_alpha{alpha}.csv", ks=ks)
    return results


if __name__ == "__main__":
    evaluate_tier0()
