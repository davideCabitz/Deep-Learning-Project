"""
OVAD Tier-0 — CLIP zero-shot baseline (object-instance retrieval lower bound).

Same champion CelebA/AwA2 config (prompt-bank ensembling, modality-gap centering, unit-
delta normalization), scored over OVAD's instance-crop corpus instead of whole images:
  t̂(a) = normalize( normalize(meanₖ bank[a,k]) − μ_txt )
  d     = normalize( Σ t̂⁺ − Σ t̂⁻ )
  q     = normalize( (v_ref − μ_img) + α·d + μ_img ),  cosine-scored vs the corpus.

Requires artifacts/ovad/clip_image_features.pt + clip_attr_prompt_bank.pt.

Run:  python src/ovad/tier0_baseline.py
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
from ovad.adapter import OVAD_ATTR_TO_IDX_DUAL as ATTR_TO_IDX
from ovad.data_loader import _get_artifacts_dir, load_image_features

OVAD_EVAL_JSON = "ovad_evaluation.json"


def _load_prompt_bank():
    path = _get_artifacts_dir() / "clip_attr_prompt_bank.pt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt bank not found. Run src/ovad/clip_prompts.py first.\n"
                                f"Expected: {path}")
    return torch.load(path)


def build_attr_directions():
    # FIX 3 then FIX 1: t̂(a) = normalize( normalize(meanₖ bank[a,k]) − μ_txt ).
    # μ_img = mean of the retrieval DB (the test split) — a label-free corpus statistic.
    image_features = load_image_features("test")
    mu_img = image_features.mean(dim=0)
    bank = _load_prompt_bank()  # [117, n, 512] unit rows
    attr = torch.nn.functional.normalize(bank.mean(dim=1), p=2, dim=1)
    mu_txt = attr.mean(dim=0)
    attr = torch.nn.functional.normalize(attr - mu_txt, p=2, dim=1)
    return attr, mu_img


def score(T_pos, T_neg, v_ref_idx, image_features, attr_directions, mu_img, alpha=1.0):
    v_ref = image_features[v_ref_idx].clone()
    delta = torch.zeros_like(v_ref)
    for name in T_pos:
        delta += attr_directions[ATTR_TO_IDX[name]]
    for name in T_neg:
        delta -= attr_directions[ATTR_TO_IDX[name]]
    if delta.norm() > 1e-8:
        delta = torch.nn.functional.normalize(delta, p=2, dim=0)
    query = (v_ref - mu_img) + alpha * delta + mu_img
    query = torch.nn.functional.normalize(query, p=2, dim=0)
    scores = image_features @ query
    scores[v_ref_idx] = float("-inf")                         # §5 self-exclusion
    return torch.argsort(scores, descending=True).tolist()


def make_get_ranking(query_str, image_features, attr_directions, mu_img, alpha=1.0):
    T_pos, T_neg = parse_query(query_str)
    return lambda src_idx: score(T_pos, T_neg, src_idx, image_features,
                                 attr_directions, mu_img, alpha=alpha)


def _load_ovad_gt():
    return load_eval_json(find_eval_json().parent / OVAD_EVAL_JSON)


def evaluate_baseline(alpha=1.0, ks=(1, 5, 10), save=True):
    # Retrieval DB = the test split (the single split benchmark all OVAD methods share).
    image_features = load_image_features("test")
    attr_directions, mu_img = build_attr_directions()
    gt_list = _load_ovad_gt()

    def make(query_str):
        return make_get_ranking(query_str, image_features, attr_directions, mu_img, alpha=alpha)

    results = evaluate_all(gt_list, make, ks=ks)
    print(f"\nOVAD Tier-0 baseline (alpha={alpha}) - {len(gt_list)} queries\n")
    print(format_results_table(results, ks=ks))

    if save:
        csv_path = output_subdir("ovad") / "tier0_baseline.csv"
        save_results_csv(results, csv_path, ks=ks)
        _dump_per_source(make, gt_list, csv_path)
    return results


def _dump_per_source(make, gt_list, csv_path):
    from per_source import check_parity_with_csv, dump_all, save_per_source_npz
    dump = dump_all(gt_list, make)
    check_parity_with_csv(dump, csv_path)
    ps_dir = output_subdir("ovad") / "per_source"
    ps_dir.mkdir(exist_ok=True)
    save_per_source_npz("ovad_tier0_baseline", dump, ps_dir / "ovad_tier0_baseline.npz")


if __name__ == "__main__":
    # Run (needs artifacts/ovad/*.pt): python src/ovad/tier0_baseline.py
    evaluate_baseline()
