"""
Tier-1 GDE — Visual-Prototype Compositional Retrieval (Berasi et al., GDE on CelebA).

Mines per-attribute primitive directions as tangent means on S^{d-1} from the
TRAIN split (no label leakage), then composes a query vector geodesically and
ranks the TEST corpus by cosine similarity.

Pipeline:
  TRAIN features + labels
      → intrinsic mean μ of the full corpus
      → v_a = tangent_mean(μ, {x : has attribute a})   for each a   [GDE §3.2, Prop. 1]
  At query time (TEST split):
      affirmation : q ← Exp_μ( Log_μ(v_ref) + Σ_{a∈T_pos} v_a )
      negation    : q ← Exp_μ( Log_μ(q) − (Log_μ(q)·v̂_a)·v̂_a )  (orthogonal rejection)
      score  : cosine(q, image_features[i])  for all i in TEST DB

Negation is orthogonal rejection (not subtraction): removes the attribute axis
entirely from the query tangent vector, avoiding the overshoot pathology described
in Alhamoud et al. 2025 (affirmation bias) and Oldfield et al. 2023 (PoS-Grounded
Subspaces, Eq. 5).

Ablation baseline: LDE (Linear Decomposable Embeddings, Trager et al. ICCV 2023)
is the same pipeline with log/exp maps replaced by flat Euclidean arithmetic — an
exact ablation of the spherical geometry.

Run:  python src/tier1_GDE.py
"""

import torch
import torch.nn.functional as F
from pathlib import Path

from data_loader import ATTRIBUTE_NAMES, ATTR_TO_IDX, _get_artifacts_dir
from clip_features import load_image_features
from eval import parse_query, evaluate_all, format_results_table, load_eval_json, find_eval_json
from results_saver import save_results_csv, output_subdir
from manifold import log_map, exp_map, intrinsic_mean, tangent_mean


# ---------------------------------------------------------------------------
# Train-split loaders
# ---------------------------------------------------------------------------

def _load_train_features() -> torch.Tensor:
    # Load [N_train, 512] L2-normalized CLIP vectors for the CelebA train split.
    path = _get_artifacts_dir() / "clip_image_features_train.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Train features not found at {path}.\n"
            "Run notebooks/colab_extract_train_features.ipynb in Colab and place the "
            "output in artifacts/."
        )
    return torch.load(path, weights_only=True)


def _load_train_attributes() -> torch.Tensor:
    # Load [N_train, 40] float32 attribute mask (0.0 / 1.0) for the train split.
    path = _get_artifacts_dir() / "celeba_attributes_train.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Train attributes not found at {path}.\n"
            "Run notebooks/colab_extract_train_features.ipynb in Colab and place the "
            "output in artifacts/."
        )
    return torch.load(path, weights_only=True)


# ---------------------------------------------------------------------------
# Direction mining  (GDE §3.2)
# ---------------------------------------------------------------------------

def mine_directions(
    train_features: torch.Tensor,
    train_attributes: torch.Tensor,
    n_iter: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Mine per-attribute primitive directions from the train split.
    # μ = intrinsic_mean(all train images);  v_a = tangent_mean(μ, has-a images).
    # Returns (mu [512], directions [40, 512]) — directions[j] is the tangent vector
    # for ATTRIBUTE_NAMES[j]; images without any has-a example get a zero vector.
    print("Computing intrinsic mean of train corpus…")
    mu = intrinsic_mean(train_features, n_iter=n_iter)   # [512]

    d = train_features.shape[1]
    directions = torch.zeros(len(ATTRIBUTE_NAMES), d, dtype=train_features.dtype)

    for j, name in enumerate(ATTRIBUTE_NAMES):
        mask = train_attributes[:, j] > 0.5            # boolean [N_train]
        has_a = train_features[mask]
        if has_a.shape[0] == 0:
            print(f"  [!] No train images with attribute '{name}' — direction set to zero.")
            continue
        directions[j] = tangent_mean(mu, has_a)        # [512] tangent vector
        print(f"  [{j+1:02d}/40] {name}: {mask.sum().item()} images", end="\r")

    print("\n[OK] Direction mining complete.")
    return mu, directions


def load_or_mine_directions(
    force: bool = False,
    n_iter: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Load cached directions if available, else mine and cache them.
    # Cached at artifacts/visual_directions.pt as {'mu': ..., 'directions': ...}.
    cache_path = _get_artifacts_dir() / "visual_directions.pt"

    if cache_path.exists() and not force:
        print(f"[OK] Loading cached directions: {cache_path}")
        ckpt = torch.load(cache_path, weights_only=True)
        return ckpt["mu"], ckpt["directions"]

    train_features   = _load_train_features()
    train_attributes = _load_train_attributes()
    mu, directions   = mine_directions(train_features, train_attributes, n_iter=n_iter)

    torch.save({"mu": mu, "directions": directions}, cache_path)
    print(f"  Saved: {cache_path}")
    return mu, directions


# ---------------------------------------------------------------------------
# Query composition  (GDE §3.2, Prop. 1 + Alhamoud / Oldfield negation)
# ---------------------------------------------------------------------------

def _compose_query_gde(
    v_ref: torch.Tensor,
    T_pos: list[str],
    T_neg: list[str],
    mu: torch.Tensor,
    directions: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    # GDE query composition — geodesic addition in tangent space + rejection negation.
    # Affirmation: q_tan = Log_μ(v_ref) + Σ_{a∈T_pos} v_a  → Exp_μ(q_tan).
    # Negation   : remove axis v̂_a from q_tan via orthogonal rejection for each a∈T_neg.
    # (Note: rejection in tangent space before exp_map, not in ambient space — stays
    # on the sphere and avoids the double-normalization artefact.)
    q_tan = log_map(mu, v_ref.unsqueeze(0)).squeeze(0)   # [d] tangent at μ

    for name in T_pos:
        q_tan = q_tan + directions[ATTR_TO_IDX[name]]    # add primitive direction

    for name in T_neg:
        v_a = directions[ATTR_TO_IDX[name]]
        v_a_norm = v_a.norm()
        if v_a_norm < eps:
            continue
        v_hat = v_a / v_a_norm
        q_tan = q_tan - (q_tan @ v_hat) * v_hat          # orthogonal rejection

    return F.normalize(exp_map(mu, q_tan.unsqueeze(0)).squeeze(0), dim=0)


def _compose_query_lde(
    v_ref: torch.Tensor,
    T_pos: list[str],
    T_neg: list[str],
    directions: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    # LDE ablation — flat Euclidean arithmetic (no log/exp maps).
    # q = normalize( v_ref + Σ_{a∈T_pos} v_a − Σ_{a∈T_neg} v_a ).
    # (Trager et al. ICCV 2023, §3 / Lemma 3: composition in ambient space.)
    # Negation uses subtraction here (LDE has no manifold-aware rejection operator).
    q = v_ref.clone()
    for name in T_pos:
        q = q + directions[ATTR_TO_IDX[name]]
    for name in T_neg:
        q = q - directions[ATTR_TO_IDX[name]]
    return F.normalize(q, dim=0)


# ---------------------------------------------------------------------------
# Retrieval seam — CONTRACT §5/§7
# ---------------------------------------------------------------------------

def make_get_ranking(
    query_str: str,
    image_features: torch.Tensor,
    mu: torch.Tensor,
    directions: torch.Tensor,
    use_gde: bool = True,
) -> callable:
    # Build the get_ranking(src_idx) → list[int] callback for one query.
    # The query vector is constructed once per query; per-source cost is one dot product.
    T_pos, T_neg = parse_query(query_str)

    def get_ranking(src_idx: int) -> list[int]:
        v_ref = image_features[src_idx]   # [512] unit vector

        if use_gde:
            q = _compose_query_gde(v_ref, T_pos, T_neg, mu, directions)
        else:
            q = _compose_query_lde(v_ref, T_pos, T_neg, directions)

        scores = image_features @ q       # cosine similarities [N]  (rows are unit)
        scores[src_idx] = float("-inf")   # source never ranks itself (CONTRACT §5)
        return torch.argsort(scores, descending=True).tolist()

    return get_ranking


# ---------------------------------------------------------------------------
# Evaluation entry points
# ---------------------------------------------------------------------------

def _run_evaluate(tag: str, use_gde: bool, ks=(1, 5, 10), save: bool = True) -> dict:
    # Shared evaluation loop for GDE and LDE variants.
    image_features = load_image_features()
    mu, directions = load_or_mine_directions()
    gt_list        = load_eval_json(find_eval_json())

    def make(query_str: str):
        return make_get_ranking(query_str, image_features, mu, directions, use_gde=use_gde)

    results = evaluate_all(gt_list, make, ks=ks)
    print(f"\nTier-1 GDE ({tag}) — {len(gt_list)} queries\n")
    print(format_results_table(results, ks=ks))

    if save:
        save_results_csv(results, output_subdir("tier1_GDE") / f"tier1_GDE_{tag}.csv", ks=ks)
    return results


def evaluate_tier1_GDE(ks=(1, 5, 10), save=True) -> dict:
    # Tier-1 GDE — geodesic composition + rejection negation (the main method).
    return _run_evaluate("gde", use_gde=True, ks=ks, save=save)


def evaluate_tier1_GDE_lde(ks=(1, 5, 10), save=True) -> dict:
    # Tier-1 GDE LDE ablation — flat arithmetic, no manifold geometry (Trager et al.).
    # Establishes what the spherical geometry (GDE) buys over the Euclidean baseline.
    return _run_evaluate("lde", use_gde=False, ks=ks, save=save)


if __name__ == "__main__":
    evaluate_tier1_GDE()
    evaluate_tier1_GDE_lde()
