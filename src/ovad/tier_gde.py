"""
OVAD GDE — visual-prototype compositional retrieval (Berasi et al.), train-mined arm.

Mines per-attribute directions as tangent means on S^{d-1} from OVAD TRAIN-split
instance features (the synthetic image-level split, data_loader.build_split), then
composes queries geodesically:
  affirmation: q = Exp_μ( Log_μ(v_ref) + Σ_{a∈T⁺} v_a )
  negation   : orthogonal rejection of v̂_a from the tangent query (not subtraction).
Scores the frozen TEST-split DB by cosine. Tri-state mining rule: a direction's support
is the train instances LABELED POSITIVE (att == 1) — never -1, which proves nothing.
Attributes with no positive support get a zero direction (skipped in composition).

Includes the LDE ablation (flat Euclidean arithmetic, Trager et al.) — the AwA2 champion.
Mining is a cached mean computation (one pass, no gradients) — local-safe.

Run:  python src/ovad/tier_gde.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / "Baselines"))
sys.path.insert(0, str(Path(__file__).parents[1] / "Training-Free"))
sys.path.insert(0, str(Path(__file__).parents[1] / "Trained"))

from eval import (evaluate_all, find_eval_json, format_results_table,
                  load_eval_json, parse_query)
from manifold import exp_map, intrinsic_mean, log_map, tangent_mean
from results_saver import output_subdir, save_results_csv
from ovad.adapter import OVAD_ATTR_TO_IDX_DUAL as ATTR_TO_IDX
from ovad.data_loader import (ATTRIBUTE_NAMES, _get_artifacts_dir,
                              load_attributes, load_image_features)

OVAD_EVAL_JSON = "ovad_evaluation.json"


def mine_directions(n_iter: int = 50) -> tuple[torch.Tensor, torch.Tensor]:
    # μ = intrinsic_mean(train split); v_a = tangent_mean(μ, {x: att_vec[x,a] == 1}).
    # Per-instance tri-state labels ⇒ support = labeled positives only; support counts
    # are logged because some of the 117 attributes have tiny train-positive pools.
    train_features = load_image_features("train")
    train_attributes = load_attributes("train")

    print("Computing intrinsic mean of train split…")
    mu = intrinsic_mean(train_features, n_iter=n_iter)

    directions = torch.zeros(len(ATTRIBUTE_NAMES), train_features.shape[1],
                             dtype=train_features.dtype)
    for j, name in enumerate(ATTRIBUTE_NAMES):
        mask = train_attributes[:, j] == 1
        if not mask.any():
            print(f"  [!] No labeled-positive train instances for '{name}' — zero direction.")
            continue
        directions[j] = tangent_mean(mu, train_features[mask])
        print(f"  [{j + 1:03d}/{len(ATTRIBUTE_NAMES)}] {name}: {int(mask.sum())} instances",
              end="\r")
    print("\n[OK] Direction mining complete.")
    return mu, directions


def load_or_mine_directions(force: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    # Cache contract mirrors tier1_GDE: artifacts/ovad/visual_directions.pt
    # {'mu', 'directions'}. The DualScore trainer reuses this same cache for its μ.
    cache_path = _get_artifacts_dir() / "visual_directions.pt"
    if cache_path.exists() and not force:
        print(f"[OK] Loading cached directions: {cache_path}")
        ckpt = torch.load(cache_path, weights_only=True)
        return ckpt["mu"], ckpt["directions"]
    mu, directions = mine_directions()
    torch.save({"mu": mu, "directions": directions}, cache_path)
    print(f"  Saved: {cache_path}")
    return mu, directions


def _compose_gde(v_ref, T_pos, T_neg, mu, directions, eps=1e-8):
    # Geodesic addition in tangent space; negation = orthogonal rejection (mirrors
    # tier1_GDE._compose_query_gde with OVAD index tables).
    q_tan = log_map(mu, v_ref.unsqueeze(0)).squeeze(0)
    for name in T_pos:
        q_tan = q_tan + directions[ATTR_TO_IDX[name]]
    for name in T_neg:
        v_a = directions[ATTR_TO_IDX[name]]
        if v_a.norm() < eps:
            continue
        v_hat = v_a / v_a.norm()
        q_tan = q_tan - (q_tan @ v_hat) * v_hat
    return F.normalize(exp_map(mu, q_tan.unsqueeze(0)).squeeze(0), dim=0)


def _compose_lde(v_ref, T_pos, T_neg, directions):
    # LDE ablation — flat arithmetic, subtraction negation (Trager et al.).
    q = v_ref.clone()
    for name in T_pos:
        q = q + directions[ATTR_TO_IDX[name]]
    for name in T_neg:
        q = q - directions[ATTR_TO_IDX[name]]
    return F.normalize(q, dim=0)


def make_get_ranking(query_str, image_features, mu, directions, use_gde=True):
    T_pos, T_neg = parse_query(query_str)

    def get_ranking(src_idx):
        v_ref = image_features[src_idx]
        q = (_compose_gde(v_ref, T_pos, T_neg, mu, directions) if use_gde
             else _compose_lde(v_ref, T_pos, T_neg, directions))
        scores = image_features @ q
        scores[src_idx] = float("-inf")               # §5 self-exclusion
        return torch.argsort(scores, descending=True).tolist()

    return get_ranking


def _load_ovad_gt():
    return load_eval_json(find_eval_json().parent / OVAD_EVAL_JSON)


def evaluate_gde(use_gde=True, ks=(1, 5, 10), save=True):
    # Full OVAD benchmark (test split); CSV + parity-gated per-source NPZ for BOTH arms.
    tag = "gde" if use_gde else "lde"
    image_features = load_image_features("test")
    mu, directions = load_or_mine_directions()
    gt_list = _load_ovad_gt()

    def make(query_str):
        return make_get_ranking(query_str, image_features, mu, directions, use_gde=use_gde)

    results = evaluate_all(gt_list, make, ks=ks)
    print(f"\nOVAD GDE ({tag}) - {len(gt_list)} queries\n")
    print(format_results_table(results, ks=ks))

    if save:
        csv_path = output_subdir("ovad") / f"tier_gde_{tag}.csv"
        save_results_csv(results, csv_path, ks=ks)
        _dump_per_source(make, gt_list, csv_path, tag)
    return results


def _dump_per_source(make, gt_list, csv_path, tag):
    from per_source import check_parity_with_csv, dump_all, save_per_source_npz
    dump = dump_all(gt_list, make)
    check_parity_with_csv(dump, csv_path)
    ps_dir = output_subdir("ovad") / "per_source"
    ps_dir.mkdir(exist_ok=True)
    save_per_source_npz(f"ovad_tier_{tag}", dump, ps_dir / f"ovad_tier_{tag}.npz")


if __name__ == "__main__":
    # Local-safe (mining is a cached mean computation): python src/ovad/tier_gde.py
    evaluate_gde(use_gde=True)
    evaluate_gde(use_gde=False)
