"""
AwA2 Tier-0 — CLIP zero-shot baseline (transfer study lower bound).

The CelebA champion Tier-0 config (src/clip_enhanced.py "all_fixes"): prompt-bank
ensembling (FIX 3), modality-gap centering (FIX 1), unit-delta normalization (FIX 2).
  t̂(a) = normalize( normalize(meanₖ bank[a,k]) − μ_txt )
  d     = normalize( Σ t̂⁺ − Σ t̂⁻ )
  q     = normalize( (v_ref − μ_img) + α·d + μ_img ),  cosine-scored vs the corpus.

Raw latent arithmetic in corrected CLIP geometry — no SVD, no learning. The floor CLAY
and the fusion model must beat on AwA2. The scoring core mirrors clip_enhanced.score();
it is re-expressed here (≈15 lines) against AwA2 loaders/indices rather than importing
the CelebA module, whose scorer binds CelebA's ATTR_TO_IDX and fixed artifact paths.

Requires artifacts/awa2/clip_image_features_test.pt + clip_attr_prompt_bank.pt (VM job).

Run:  python src/awa2/tier0_baseline.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / "Baselines"))
sys.path.insert(0, str(Path(__file__).parents[1] / "Training-Free"))
sys.path.insert(0, str(Path(__file__).parents[1] / "Trained"))

from eval import (evaluate_all, find_eval_json, format_results_table,
                  load_eval_json, parse_query)
from results_saver import output_subdir, save_results_csv
from awa2.data_loader import (ATTR_TO_IDX, _get_artifacts_dir, _get_project_root,
                              load_image_features)

AWA2_EVAL_JSON = "awa2_evaluation.json"


def _load_prompt_bank():
    path = _get_artifacts_dir() / "clip_attr_prompt_bank.pt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt bank not found (VM job). Expected: {path}")
    return torch.load(path)


def build_attr_directions():
    # FIX 3 then FIX 1: t̂(a) = normalize( normalize(meanₖ bank[a,k]) − μ_txt ).
    # Returns ([85,512] unit dirs, μ_img) so the image side centers with the same μ_img.
    image_features = load_image_features("test")
    mu_img = image_features.mean(dim=0)
    bank = _load_prompt_bank()  # [85, n, 512] unit rows
    attr = torch.nn.functional.normalize(bank.mean(dim=1), p=2, dim=1)
    mu_txt = attr.mean(dim=0)
    attr = torch.nn.functional.normalize(attr - mu_txt, p=2, dim=1)
    return attr, mu_img


def score(T_pos, T_neg, v_ref_idx, image_features, attr_directions, mu_img, alpha=1.0):
    # Tier-0 enhanced scorer in AwA2 index space (mirrors clip_enhanced.score, all fixes on).
    v_ref = image_features[v_ref_idx].clone()
    delta = torch.zeros_like(v_ref)
    for name in T_pos:
        delta += attr_directions[ATTR_TO_IDX[name]]
    for name in T_neg:
        delta -= attr_directions[ATTR_TO_IDX[name]]
    if delta.norm() > 1e-8:                                   # FIX 2: unit delta
        delta = torch.nn.functional.normalize(delta, p=2, dim=0)
    query = (v_ref - mu_img) + alpha * delta + mu_img         # FIX 1: centered frame
    query = torch.nn.functional.normalize(query, p=2, dim=0)
    scores = image_features @ query
    scores[v_ref_idx] = float("-inf")                         # §5 self-exclusion
    return torch.argsort(scores, descending=True).tolist()


def make_get_ranking(query_str, image_features, attr_directions, mu_img, alpha=1.0):
    T_pos, T_neg = parse_query(query_str)
    return lambda src_idx: score(T_pos, T_neg, src_idx, image_features,
                                 attr_directions, mu_img, alpha=alpha)


def _load_awa2_gt():
    return load_eval_json(find_eval_json().parent / AWA2_EVAL_JSON)


def evaluate_baseline(alpha=1.0, ks=(1, 5, 10), save=True):
    # Full AwA2 benchmark: print table, save CSV + per-source NPZ (parity-gated).
    image_features = load_image_features("test")
    attr_directions, mu_img = build_attr_directions()
    gt_list = _load_awa2_gt()

    def make(query_str):
        return make_get_ranking(query_str, image_features, attr_directions, mu_img, alpha=alpha)

    results = evaluate_all(gt_list, make, ks=ks)
    print(f"\nAwA2 Tier-0 baseline (alpha={alpha}) - {len(gt_list)} queries\n")
    print(format_results_table(results, ks=ks))

    if save:
        csv_path = output_subdir("awa2") / "tier0_baseline.csv"
        save_results_csv(results, csv_path, ks=ks)
        _dump_per_source(make, gt_list, csv_path)
    return results


def _dump_per_source(make, gt_list, csv_path):
    from per_source import check_parity_with_csv, dump_all, save_per_source_npz
    dump = dump_all(gt_list, make)
    check_parity_with_csv(dump, csv_path)
    ps_dir = output_subdir("awa2") / "per_source"
    ps_dir.mkdir(exist_ok=True)
    save_per_source_npz("awa2_tier0_baseline", dump, ps_dir / "awa2_tier0_baseline.npz")


if __name__ == "__main__":
    # Run (needs VM-built features): python src/awa2/tier0_baseline.py
    evaluate_baseline()
