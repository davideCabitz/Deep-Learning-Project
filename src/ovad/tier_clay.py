"""
OVAD CLAY — the method-to-beat on object-instance retrieval.

Faithful CLAY (Lim et al., 2026): stack condition prompts, build a manifold-aware textual
subspace by SVD on the log-mapped stack, project the frozen instance-crop DB into it,
score by cosine inside that subspace. The manifold math is identical across datasets, so
this module REUSES tier1_CLAY's pure functions (_build_subspace, _project_db) and only
re-expresses the prompt-stacking step against OVAD's 117 attributes and prompt bank.

Requires artifacts/ovad/clip_image_features.pt + clip_attr_prompt_bank.pt.

Run:  python src/ovad/tier_clay.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from eval import (evaluate_all, find_eval_json, format_results_table,
                  load_eval_json, parse_query)
from results_saver import output_subdir, save_results_csv
from tier1_CLAY import _build_subspace, _project_db  # shared manifold math (DRY)
from ovad.adapter import IS_HAS_BY_NAME
from ovad.clip_prompts import build_prompts_for_attribute
from ovad.data_loader import ATTRIBUTE_NAMES, _get_artifacts_dir, load_image_features

OVAD_EVAL_JSON = "ovad_evaluation.json"


def _load_prompt_bank():
    path = _get_artifacts_dir() / "clip_attr_prompt_bank.pt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt bank not found. Run src/ovad/clip_prompts.py first.\n"
                                f"Expected: {path}")
    return torch.load(path)


def _true_prompt_count(name: str) -> int:
    return len(build_prompts_for_attribute(name, IS_HAS_BY_NAME[name]))


def _stack_condition_prompts(T_pos, T_neg, prompt_bank):
    # OVAD variant of tier1_CLAY._stack_condition_prompts: merge ± condition prompts into
    # T_c [m, d], stripping per-attribute padding via OVAD's true prompt counts.
    rows = []
    for name in T_pos + T_neg:
        j = ATTRIBUTE_NAMES.index(name)
        n_j = _true_prompt_count(name)
        rows.append(prompt_bank[j, :n_j])
    return torch.cat(rows, dim=0)


def make_get_ranking(query_str, image_features, prompt_bank, k=50, use_rotation=True):
    T_pos, T_neg = parse_query(query_str)
    T_c = _stack_condition_prompts(T_pos, T_neg, prompt_bank)
    mu_c, V_k = _build_subspace(T_c, k)
    D = _project_db(image_features, mu_c, V_k, use_rotation=use_rotation)

    def get_ranking(src_idx):
        scores = D @ D[src_idx]
        scores[src_idx] = float("-inf")               # §5 self-exclusion
        return torch.argsort(scores, descending=True).tolist()

    return get_ranking


def _load_ovad_gt():
    return load_eval_json(find_eval_json().parent / OVAD_EVAL_JSON)


def evaluate_clay(k=50, use_rotation=True, ks=(1, 5, 10), save=True):
    # Retrieval DB = the test split (the single split benchmark all OVAD methods share).
    image_features = load_image_features("test")
    prompt_bank = _load_prompt_bank()
    gt_list = _load_ovad_gt()

    def make(query_str):
        return make_get_ranking(query_str, image_features, prompt_bank, k=k,
                                use_rotation=use_rotation)

    results = evaluate_all(gt_list, make, ks=ks)
    rot = "rotH" if use_rotation else "norot"
    print(f"\nOVAD CLAY (k={k}, {rot}) - {len(gt_list)} queries\n")
    print(format_results_table(results, ks=ks))

    if save:
        csv_path = output_subdir("ovad") / "tier_clay.csv"
        save_results_csv(results, csv_path, ks=ks)
        _dump_per_source(make, gt_list, csv_path)
    return results


def _dump_per_source(make, gt_list, csv_path):
    from per_source import check_parity_with_csv, dump_all, save_per_source_npz
    dump = dump_all(gt_list, make)
    check_parity_with_csv(dump, csv_path)
    ps_dir = output_subdir("ovad") / "per_source"
    ps_dir.mkdir(exist_ok=True)
    save_per_source_npz("ovad_tier_clay", dump, ps_dir / "ovad_tier_clay.npz")


if __name__ == "__main__":
    # Run (needs artifacts/ovad/*.pt): python src/ovad/tier_clay.py
    evaluate_clay()
