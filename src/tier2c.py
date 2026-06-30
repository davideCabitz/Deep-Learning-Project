"""
Tier-2c — Visual-Subspace Negation.

Takes the better half of Tier-2b (visual-prototype composition) and Tier-2a (asymmetric conditional
subspaces) and drops the weaker half of each:

  - keep 2b's negation MECHANISM — orthogonal rejection on the QUERY's tangent vector,
    entirely in image space (no modality gap, no rotation H);
  - drop 2b's negation GEOMETRY — a single mined direction v_X_hat — and replace it with 2a's
    richer k-dimensional SUBSPACE;
  - drop 2a's negation SOURCE — a TEXT prompt stack that must cross the modality gap — and
    mine the subspace from the TRAIN images that actually HAVE attribute X.

"Not Male" stops meaning "delete one average-Male axis" (2b) or "penalise text-Male energy"
(2a) and becomes "delete the whole visual *Male region* from the query." Everything on the
positive side (geodesic addition) and the scoring (cosine on the frozen test DB) is inherited
verbatim from Tier-2b.

The single load-bearing invariant: every negative subspace is log-mapped at the GLOBAL mean mu
— the same tangent point as the query vector — so the rejection is geometrically meaningful.
This is why this file mines its own subspaces instead of reusing manifold.build_subspace, which
log-maps at each stack's LOCAL mean.

No label leakage: all geometry is mined from the TRAIN split; the test DB and the GT JSON are
never inspected to build it (identical discipline to Tier-2b).

Run:  python src/tier2c.py
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from data_loader import ATTRIBUTE_NAMES, ATTR_TO_IDX, _get_artifacts_dir
from clip_features import load_image_features
from eval import parse_query, evaluate_query, format_results_table, load_eval_json, find_eval_json
from results_saver import save_results_csv, output_subdir
from manifold import log_map, exp_map
from tier1_GDE import (
    _load_train_features,
    _load_train_attributes,
    load_or_mine_directions,
)

# Max width every negative subspace is mined to once; per-config k_neg just slices [:k] from
# this. Comfortably below both d=512 and m_b (thousands per CelebA attr), so k_eff == K_CACHE.
K_CACHE = 50


# ---------------------------------------------------------------------------
# Configuration — the ablation surface, bundled so call sites stay short.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SVunionConfig:
    """SV-union hyperparameters (planSV.md §4.1). Defaults are the headline starting point."""
    k_neg: int = 10            # dim of each negative VISUAL subspace; sweep 1,5,10,20 (k=1 ≈ Track V)
    alpha: float = 1.0         # positive push strength (Track V's α)
    reject_on: str = "query"   # "query" (headline) | "db" (ablation: project the DB into the complement)

    def __post_init__(self):
        # Fail loudly at construction — an illegal config must never reach the scoring loop.
        if self.reject_on not in ("query", "db"):
            raise ValueError(f"reject_on must be 'query' or 'db', got {self.reject_on!r}")
        if self.k_neg < 1:
            raise ValueError(f"k_neg must be >= 1, got {self.k_neg}")

    def tag(self) -> str:
        # Filename-safe descriptor, e.g. "query_kneg10_a1.0".
        return f"{self.reject_on}_kneg{self.k_neg}_a{self.alpha}"


# ---------------------------------------------------------------------------
# Negative visual subspace mining  (planSV.md §3.2–3.3)
# ---------------------------------------------------------------------------

def _build_visual_neg_subspace(
    X_b: torch.Tensor, X_not_b: torch.Tensor, mu: torch.Tensor, k: int
) -> torch.Tensor:
    # Visual negative subspace for one attribute — CONTRAST directions of has-X vs not-X at μ.
    # L = Log_μ(X_b) − mean(Log_μ(X_not_b));  Q_b = top-k right singular vectors of L, descending.
    # Recentring the has-X cloud on the not-X tangent centroid cancels the shared face-manifold
    # variance (pose, identity, lighting, co-occurring attrs) BEFORE the SVD, so the principal
    # directions span "X-ness" instead of generic face structure — the visual analogue of 2a's
    # single-concept text stack. This is what makes k_neg>1 add attribute-removal power rather
    # than punch a hole in the structure retrieval needs (see plan §3, §5).
    # SVD's right singular vectors are the eigenvectors of the Gram LᵀL, so we eigendecompose the
    # [d,d] Gram instead of SVD-ing the [m,d] L: same V, no [m,d] U materialised, scales to m≫d.
    # (Critical: log-map at the GLOBAL μ — the query's tangent point — NOT each stack's local mean,
    # or the rejection projects onto a subspace living in a different tangent plane. §3.2.)
    c_neg = log_map(mu, X_not_b).mean(dim=0)              # [d] tangent centroid of the not-X cloud
    L = log_map(mu, X_b) - c_neg                          # [m, d] has-X residuals vs the not-X centre
    gram = L.T @ L                                        # [d, d] symmetric PSD second moment
    evals, evecs = torch.linalg.eigh(gram)               # ascending eigenpairs
    k_eff = min(k, evecs.shape[1])
    return evecs[:, -k_eff:].flip(1).contiguous()        # [d, k_eff] top-k, descending importance


def mine_neg_subspaces(
    train_features: torch.Tensor,
    train_attributes: torch.Tensor,
    mu: torch.Tensor,
    k: int = K_CACHE,
) -> torch.Tensor:
    # Mine all 40 negative visual subspaces from the train split → [40, d, k].
    # subspaces[j] = Q_j for ATTRIBUTE_NAMES[j]; attrs with no train example get a zero block
    # (guarded — never happens for the 40 CelebA attrs, but an illegal state must be loud).
    d = train_features.shape[1]
    subspaces = torch.zeros(len(ATTRIBUTE_NAMES), d, k, dtype=train_features.dtype)

    for j, name in enumerate(ATTRIBUTE_NAMES):
        mask = train_attributes[:, j] > 0.5             # train images that HAVE attribute j
        X_b = train_features[mask]
        X_not_b = train_features[~mask]                 # complement: the not-X contrast cloud
        if X_b.shape[0] == 0 or X_not_b.shape[0] == 0:
            # Need BOTH clouds for a contrast subspace — a one-sided attribute is an illegal state.
            print(f"  [!] Attribute '{name}' missing has-X or not-X examples — subspace set to zero.")
            continue
        Q = _build_visual_neg_subspace(X_b, X_not_b, mu, k)   # [d, k_eff]
        subspaces[j, :, : Q.shape[1]] = Q                # left-pad short bases with zeros (k_eff < k)
        print(f"  [{j+1:02d}/40] {name}: {X_b.shape[0]} images", end="\r")

    print("\n[OK] Negative visual subspace mining complete.")
    return subspaces                                     # [40, d, k]


def load_or_mine_neg_subspaces(force: bool = False) -> torch.Tensor:
    # Load cached negative subspaces if present, else mine and cache them.
    # Cached at artifacts/visual_neg_subspaces_contrast.pt as a single [40, d, K_CACHE] tensor; μ
    # comes from the directions cache so subspaces and the query share one global tangent point.
    # (Distinct "_contrast" key from the original raw-cloud subspaces so both can coexist as an
    # ablation — the cloud-variance subspaces are the falsified baseline, see plan §3/§5.)
    cache_path = _get_artifacts_dir() / "visual_neg_subspaces_contrast.pt"

    if cache_path.exists() and not force:
        print(f"[OK] Loading cached negative subspaces: {cache_path}")
        return torch.load(cache_path, weights_only=True)

    mu, _ = load_or_mine_directions()                    # shared global mean μ (mined/cached by Track V)
    train_features   = _load_train_features()
    train_attributes = _load_train_attributes()
    print("Mining negative visual subspaces (40 attrs)…")
    subspaces = mine_neg_subspaces(train_features, train_attributes, mu, k=K_CACHE)

    torch.save(subspaces, cache_path)
    print(f"  Saved: {cache_path}")
    return subspaces


# ---------------------------------------------------------------------------
# Query composition  (planSV.md §3.1 + §3.4)
# ---------------------------------------------------------------------------

def _positive_tangent_batch(
    V_ref: torch.Tensor,
    T_pos: list[str],
    mu: torch.Tensor,
    directions: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    # Positive tangent vectors for a batch of references — Track V affirmation, verbatim.
    # q_tan = Log_μ(V_ref) + Σ α·v_a, broadcast over rows. Single owner of the affirmation step
    # (the single-vector seam and the batched benchmark driver both build their query from this).
    q_tan = log_map(mu, V_ref)                           # [m, d]
    for name in T_pos:
        q_tan = q_tan + alpha * directions[ATTR_TO_IDX[name]]
    return q_tan


def _compose_positive_tangent(
    v_ref: torch.Tensor,
    T_pos: list[str],
    mu: torch.Tensor,
    directions: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    # Single-vector adapter over _positive_tangent_batch (CONTRACT §7 seam path).
    return _positive_tangent_batch(v_ref.unsqueeze(0), T_pos, mu, directions, alpha).squeeze(0)


def _build_union_basis(
    subspaces: torch.Tensor,
    T_neg: list[str],
    k_neg: int,
) -> torch.Tensor | None:
    # Orthonormal basis of the UNION of the query's negative visual subspaces (planSV.md §3.4).
    # W = [ Q_b1[:,:k] | Q_b2[:,:k] | … ];  Q_all = qr(W).Q  → [d, k_total].
    # Thin QR re-orthonormalises the concatenation so overlapping attribute subspaces
    # (e.g. Male ∩ Mustache) span the union without double-counting. None when T_neg is empty —
    # the caller then treats rejection as the identity, degenerating cleanly to Track V positive.
    if not T_neg:
        return None
    k = min(k_neg, subspaces.shape[-1])
    cols = [subspaces[ATTR_TO_IDX[b]][:, :k] for b in T_neg]
    W = torch.cat(cols, dim=1)                           # [d, k_total]
    Q_all, _ = torch.linalg.qr(W)                        # [d, k_total] orthonormal columns
    return Q_all


def _compose_query_svunion(
    v_ref: torch.Tensor,
    T_pos: list[str],
    T_neg: list[str],
    mu: torch.Tensor,
    directions: torch.Tensor,
    subspaces: torch.Tensor,
    cfg: SVunionConfig,
) -> torch.Tensor:
    # SV-union query (reject_on="query", headline) — positive geodesic addition + subspace rejection.
    # q = normalize( Exp_μ( Π⊥(Log_μ(v_ref) + Σα·v_a) ) ),  Π⊥ = complement of the union span Q_all.
    # The whole visual region of every negated attribute is deleted from the query in T_μ, then the
    # query is lifted back to the sphere and scored exactly as Track V. (k_neg=1 ≈ Track V negation.)
    q_tan = _compose_positive_tangent(v_ref, T_pos, mu, directions, cfg.alpha)
    Q_all = _build_union_basis(subspaces, T_neg, cfg.k_neg)
    if Q_all is not None:
        q_tan = q_tan - Q_all @ (Q_all.T @ q_tan)        # reject the union span
    return F.normalize(exp_map(mu, q_tan.unsqueeze(0)).squeeze(0), dim=0)


# ---------------------------------------------------------------------------
# Retrieval seam — CONTRACT §5/§7
# ---------------------------------------------------------------------------

def _make_get_ranking_query(
    T_pos, T_neg, image_features, mu, directions, subspaces, cfg,
) -> callable:
    # Headline seam — negation lives in the QUERY. Per-source: compose q, score DB by cosine.
    def get_ranking(src_idx: int) -> list[int]:
        q = _compose_query_svunion(
            image_features[src_idx], T_pos, T_neg, mu, directions, subspaces, cfg,
        )
        scores = image_features @ q                      # cosine [N] (rows unit, q unit)
        scores[src_idx] = float("-inf")                  # source never ranks itself (CONTRACT §5)
        return torch.argsort(scores, descending=True).tolist()

    return get_ranking


def _make_get_ranking_db(
    T_pos, T_neg, image_features, mu, directions, subspaces, cfg,
) -> callable:
    # Ablation seam — negation lives in the SPACE (visual analogue of NCS, approaches doc §2).
    # Project the whole DB onto the complement once per query, then score positives-only queries
    # against it by RAW inner product. DB_perp rows are NOT renormalised — their shrunken norm is
    # the suppression fraction and the cosine uses it correctly; renormalising would amplify exactly
    # the negated-attribute images we removed (approaches doc §"Do Not Normalize After Projecting").
    Q_all = _build_union_basis(subspaces, T_neg, cfg.k_neg)
    db_perp = image_features
    if Q_all is not None:
        db_perp = image_features - (image_features @ Q_all) @ Q_all.T   # [N, d], complement

    def get_ranking(src_idx: int) -> list[int]:
        q_tan = _compose_positive_tangent(image_features[src_idx], T_pos, mu, directions, cfg.alpha)
        q = F.normalize(exp_map(mu, q_tan.unsqueeze(0)).squeeze(0), dim=0)
        scores = db_perp @ q                             # raw inner product in the complement [N]
        scores[src_idx] = float("-inf")
        return torch.argsort(scores, descending=True).tolist()

    return get_ranking


def make_get_ranking(
    query_str: str,
    image_features: torch.Tensor,
    mu: torch.Tensor,
    directions: torch.Tensor,
    subspaces: torch.Tensor,
    cfg: SVunionConfig = SVunionConfig(),
) -> callable:
    # CONTRACT §7 seam — curry one query string into get_ranking(src_idx). Dispatches on reject_on:
    # "query" rejects from the per-source query, "db" rejects from the shared DB (precomputed once).
    T_pos, T_neg = parse_query(query_str)
    seam = _make_get_ranking_db if cfg.reject_on == "db" else _make_get_ranking_query
    return seam(T_pos, T_neg, image_features, mu, directions, subspaces, cfg)


# ---------------------------------------------------------------------------
# Evaluation entry points
# ---------------------------------------------------------------------------

def _load_artifacts():
    # Load the four read-only inputs every config shares (DB, μ, directions, subspaces, GT).
    image_features = load_image_features()
    mu, directions = load_or_mine_directions()
    subspaces      = load_or_mine_neg_subspaces()
    gt_list        = load_eval_json(find_eval_json())
    return image_features, mu, directions, subspaces, gt_list


def _batched_rankings(
    T_pos, T_neg, image_features, mu, directions, subspaces, cfg, sources, top_k, chunk=512,
) -> dict:
    # All of a query's sources ranked at once → {src_idx: top-`top_k` ranking}. Memory-bounded by `chunk`.
    # Batches the per-source query vectors into one Exp_μ + GEMM (image_features @ Q.T), turning thousands
    # of matrix-vector products into a few BLAS GEMMs — the speedup that makes the full grid tractable on
    # CPU. The union basis (and, for reject_on="db", the DB complement) is source-independent, so it is
    # built ONCE per query; only the cheap per-source query composition runs inside the chunk loop.
    Q_all = _build_union_basis(subspaces, T_neg, cfg.k_neg)
    base = image_features
    if cfg.reject_on == "db" and Q_all is not None:                    # negation lives in the DB space
        base = image_features - (image_features @ Q_all) @ Q_all.T     # complement, not renormalized

    rankings = {}
    for start in range(0, len(sources), chunk):
        cols = torch.as_tensor(sources[start:start + chunk], dtype=torch.long)
        q_tan = _positive_tangent_batch(image_features[cols], T_pos, mu, directions, cfg.alpha)
        if cfg.reject_on == "query" and Q_all is not None:            # negation lives in the query
            q_tan = q_tan - (q_tan @ Q_all) @ Q_all.T                 # reject each row's union span
        Q = F.normalize(exp_map(mu, q_tan), dim=1)                    # [m, d] unit queries
        scores = base @ Q.T                                           # [N, m]
        scores[cols, torch.arange(len(cols))] = float("-inf")        # source never ranks itself (§5)
        top = torch.topk(scores, top_k, dim=0).indices               # [top_k, m] — metrics read only the prefix
        for j, src in enumerate(cols.tolist()):
            rankings[src] = top[:, j].tolist()
    return rankings


def _evaluate_one(cfg, image_features, mu, directions, subspaces, gt_list, ks, save) -> dict:
    # Score one config over the full benchmark, print the table, optionally save the CSV.
    # Single owner of the eval→print→save block so evaluate_svunion and run_ablation can't diverge.
    # Ranks each query's sources via the batched driver, then defers metric averaging to eval.evaluate_query.
    top_k = max(ks)
    results = {}
    for entry in gt_list:
        T_pos, T_neg = parse_query(entry["query"])
        sources = [int(s) for s in entry["ground_truth"]]
        rankings = _batched_rankings(
            T_pos, T_neg, image_features, mu, directions, subspaces, cfg, sources, top_k,
        )
        results[entry["query"]] = evaluate_query(entry["ground_truth"], rankings.__getitem__, ks)

    print(f"\nTier-2a SV-union ({cfg.tag()}) — {len(gt_list)} queries\n")
    print(format_results_table(results, ks=ks))

    if save:
        save_results_csv(
            results, output_subdir("tier2c") / f"tier2c_{cfg.tag()}.csv", ks=ks,
        )
    return results


def evaluate_svunion(cfg: SVunionConfig = SVunionConfig(), ks=(1, 5, 10), save=True) -> dict:
    # SV-union — visual-subspace query rejection on top of Track V composition (the main method).
    image_features, mu, directions, subspaces, gt_list = _load_artifacts()
    return _evaluate_one(cfg, image_features, mu, directions, subspaces, gt_list, ks, save)


def _ablation_configs() -> list[SVunionConfig]:
    # planSV.md §7 grid: reject_on × k_neg at α=1, plus an α sweep at the default k_neg=10.
    configs = [
        SVunionConfig(k_neg=k, alpha=1.0, reject_on=side)
        for side in ("query", "db")
        for k in (1, 5, 10, 20)
    ]
    configs += [SVunionConfig(k_neg=10, alpha=a, reject_on="query") for a in (0.5, 1.5)]
    return configs


def run_ablation(ks=(1, 5, 10)):
    # Run the full ablation grid once, reusing the loaded DB / μ / directions / subspaces.
    image_features, mu, directions, subspaces, gt_list = _load_artifacts()
    for cfg in _ablation_configs():
        _evaluate_one(cfg, image_features, mu, directions, subspaces, gt_list, ks, save=True)


if __name__ == "__main__":
    run_ablation()
