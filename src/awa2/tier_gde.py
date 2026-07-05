"""
AwA2 GDE — visual-prototype compositional retrieval (Berasi et al.), the transfer
study's best-bet training-free challenger.

Mines per-attribute directions as tangent means on S^{d-1} from AwA2 TRAIN features
(no text encoding, no modality gap), then composes queries geodesically:
  affirmation: q = Exp_μ( Log_μ(v_ref) + Σ_{a∈T⁺} v_a )
  negation   : orthogonal rejection of v̂_a from the tangent query (not subtraction).
Scores the frozen TEST DB by cosine.

Why this is the AwA2 candidate: the λ-sweep diagnostic showed AwA2 performance is
monotone in CLIP-identity weight and the baseline's weak queries (+horns, +longneck) are
those where the TEXT delta steers poorly. Class-level attributes make train-mined VISUAL
directions unusually clean (class-centroid contrasts) — attacking exactly that weakness
while keeping the identity-preserving query form. Mining is a cached mean computation
(one pass over train features, no gradients) — safe to run locally.

Includes the LDE ablation (flat Euclidean arithmetic, Trager et al.) as on CelebA.

Run:  python src/awa2/tier_gde.py
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
from awa2.data_loader import (ATTR_TO_IDX, ATTRIBUTE_NAMES, _get_artifacts_dir,
                              load_attributes, load_image_features)

AWA2_EVAL_JSON = "awa2_evaluation.json"


def mine_directions(n_iter: int = 50) -> tuple[torch.Tensor, torch.Tensor]:
    # μ = intrinsic_mean(train corpus); v_a = tangent_mean(μ, {x: has attribute a}).
    # Class-level labels ⇒ each has-a set is a union of whole classes — the mined
    # direction is a clean class-centroid contrast (the AwA2 advantage over text deltas).
    train_features = load_image_features("train")
    train_attributes = load_attributes("train")

    print("Computing intrinsic mean of train corpus…")
    mu = intrinsic_mean(train_features, n_iter=n_iter)

    directions = torch.zeros(len(ATTRIBUTE_NAMES), train_features.shape[1],
                             dtype=train_features.dtype)
    for j, name in enumerate(ATTRIBUTE_NAMES):
        mask = train_attributes[:, j] > 0.5
        if not mask.any():
            print(f"  [!] No train images with attribute '{name}' — zero direction.")
            continue
        directions[j] = tangent_mean(mu, train_features[mask])
        print(f"  [{j + 1:02d}/{len(ATTRIBUTE_NAMES)}] {name}: {int(mask.sum())} images", end="\r")
    print("\n[OK] Direction mining complete.")
    return mu, directions


def load_or_mine_directions(force: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    # Cache contract mirrors tier1_GDE: artifacts/awa2/visual_directions.pt
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
    # tier1_GDE._compose_query_gde with AwA2 index tables).
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


def _load_awa2_gt():
    return load_eval_json(find_eval_json().parent / AWA2_EVAL_JSON)


def evaluate_gde(use_gde=True, ks=(1, 5, 10), save=True):
    # Full AwA2 benchmark; CSV + parity-gated per-source NPZ for BOTH arms (LDE turned out
    # to be AwA2's best method, so it enters the statistical suite as a model, not an ablation).
    tag = "gde" if use_gde else "lde"
    image_features = load_image_features("test")
    mu, directions = load_or_mine_directions()
    gt_list = _load_awa2_gt()

    def make(query_str):
        return make_get_ranking(query_str, image_features, mu, directions, use_gde=use_gde)

    results = evaluate_all(gt_list, make, ks=ks)
    print(f"\nAwA2 GDE ({tag}) - {len(gt_list)} queries\n")
    print(format_results_table(results, ks=ks))

    if save:
        csv_path = output_subdir("awa2") / f"tier_gde_{tag}.csv"
        save_results_csv(results, csv_path, ks=ks)
        _dump_per_source(make, gt_list, csv_path, tag)
    return results


def _dump_per_source(make, gt_list, csv_path, tag):
    from per_source import check_parity_with_csv, dump_all, save_per_source_npz
    dump = dump_all(gt_list, make)
    check_parity_with_csv(dump, csv_path)
    ps_dir = output_subdir("awa2") / "per_source"
    ps_dir.mkdir(exist_ok=True)
    save_per_source_npz(f"awa2_tier_{tag}", dump, ps_dir / f"awa2_tier_{tag}.npz")


if __name__ == "__main__":
    # Local-safe (mining is a cached mean computation): python src/awa2/tier_gde.py
    evaluate_gde(use_gde=True)
    evaluate_gde(use_gde=False)
