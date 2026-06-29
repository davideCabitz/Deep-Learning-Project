"""
Tier-2a — Asymmetric Conditional Subspaces.

Fixes the failure every earlier tier shares: negation queries (-Male, -Mustache) score 0.000,
because CLIP cannot represent "not X" as a point on the sphere (SpaceVLM's impossibility theorem).
Tier-2a represents each polarity as a *subspace* instead, scored asymmetrically:

    For each positive attribute a:  S⁺_a = PGA subspace of a's prompt stack   (the "near A" region)
    For each negative attribute b:  S⁻_b = PGA subspace of b's prompt stack   (the "far from N" region)
    score(src, d) = cos(v_src, v_d)  +  mean_a cos_{S⁺_a}(v_d, v_src)  −  λ · Σ_b ‖proj_{S⁻_b}(v_d)‖

The identity anchor (leading cosine) preserves reference identity while the positive and negative
subspaces impose constraints. Polarity-split stacks (one SVD per attribute) ensure no attribute with
more prompts dominates. Per-condition subspaces follow PoS-grounded philosophy; per-polarity stacking
is an ablation toggle. The λ·‖proj_{S⁻}‖ negation penalty is the asymmetric negation: energy inside
a negative subspace means "looks like the thing we want absent", so it is pushed down.

Plugs into the eval engine through the shared get_ranking callback (CONTRACT §5/§7); reuses
frozen hypersphere primitives from manifold.py. CSVs land in output/tier2a/ — one per config.
Run:  python src/tier2a.py
"""

from dataclasses import dataclass

import torch

from data_loader import ATTRIBUTE_NAMES
from clip_features import load_image_features
from clip_prompts import load_prompt_bank, build_prompts_for_attribute
from eval import parse_query, evaluate_query, format_results_table, load_eval_json, find_eval_json
from manifold import log_map, align_rotation, build_subspace
from results_saver import save_results_csv, output_subdir

_EPS_NORM = 1e-12  # row-normalize guard: images with near-zero projection in S⁺ must not divide by 0


# ---------------------------------------------------------------------------
# Configuration — the four ablation knobs of S_plan.md §6, bundled so the public
# surface stays a single object instead of a long parameter list down every call.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrackSConfig:
    """Track S hyperparameters. Defaults are S_plan.md §6's recommended starting point."""
    k_pos: int = 20             # S⁺ subspace dimensionality (§6.1)
    k_neg: int = 20             # S⁻ subspace dimensionality (§6.1)
    lam: float = 0.1            # negation penalty weight; literature points to the mild 0.05–0.3 band (§6.2)
    use_rotation: bool = True   # modality-gap rotation H on/off (§6.4)
    per_condition: bool = True  # headline: one subspace per attribute; False = stacked-per-polarity (§6.3)
    identity_anchor: bool = True  # add the reference-identity cosine; False = plan-literal pure-subspace

    def tag(self) -> str:
        # Filename-safe descriptor of this config, e.g. "percond_anchor_k20_20_lam0.1_rotH".
        variant = "percond" if self.per_condition else "stacked"
        anchor = "anchor" if self.identity_anchor else "noanchor"
        rot = "rotH" if self.use_rotation else "norot"
        return f"{variant}_{anchor}_k{self.k_pos}_{self.k_neg}_lam{self.lam}_{rot}"


# ---------------------------------------------------------------------------
# Stage 1 — polarity-split prompt stacks (S_plan.md §3 Stage 1)
# ---------------------------------------------------------------------------
def _attr_prompt_stack(name, prompt_bank):
    """One attribute's UNPADDED prompt stack [n_j, d] (strip the bank's duplicate padding).

    The cached bank pads every stack to a common width with repeated rows; the true count comes
    from build_prompts_for_attribute() (deterministic, no CLIP). Padding rows share a direction —
    keeping them would fold their variance into the first singular value and skew mu_c / S.
    """
    j = ATTRIBUTE_NAMES.index(name)
    n_j = len(build_prompts_for_attribute(name))
    return prompt_bank[j, :n_j]


def _polarity_groups(names, prompt_bank, per_condition):
    """Group one polarity's prompts into the stacks each gets its own subspace from.

    per_condition=True → one group per attribute (equal geometric weight to every condition,
    PoS-Subspaces' per-role philosophy). per_condition=False → one merged group for the whole
    polarity (CLAY-style stacking; biased toward the attribute with more prompts). Empty `names`
    yields no groups, which the score handles as a degenerate (pure-positive / pure-negative) case.
    """
    stacks = [_attr_prompt_stack(name, prompt_bank) for name in names]
    if per_condition or not stacks:
        return stacks
    return [torch.cat(stacks, dim=0)]


# ---------------------------------------------------------------------------
# Stages 2–5 — build a subspace from a stack and project the frozen DB into it
# ---------------------------------------------------------------------------
def _project_db_into_subspace(prompt_stack, image_features, k, use_rotation):
    """Tangent coords of every DB image in the stack's subspace: log_{mu_c}(H·v) @ V_k → [N, k_eff].

    Builds the manifold-aware subspace (mu_c, V_k) once, optionally rotates the visual cone onto
    THIS subspace's text mean mu_c to close the modality gap (per-subspace H, exactly as Tier-1's
    _project_db does), then projects. Returns RAW coords; the caller decides between row-normalizing
    (positive: cosine in S⁺) and taking the row norm (negative: energy in S⁻). Computed once per
    query and reused across all of that query's sources — the efficiency gain inherited from CLAY.
    """
    mu_c, V_k = build_subspace(prompt_stack, k)
    V = image_features
    if use_rotation:
        mu_img = torch.nn.functional.normalize(image_features.mean(dim=0), dim=0)
        V = V @ align_rotation(mu_img, mu_c).T        # align visual mean → this subspace's text mean
    return log_map(mu_c, V) @ V_k                     # [N, k_eff]


# ---------------------------------------------------------------------------
# Stage 6 — precompute the per-query subspaces, then score (CONTRACT §7 seam)
# ---------------------------------------------------------------------------
def _precompute(T_pos, T_neg, image_features, prompt_bank, config):
    """Build the per-query, source-independent subspaces ONCE (the expensive part).

    Positive groups → row-normalized DB coords D⁺ (cosine-ready), one per group. Negative groups →
    per-image energy ‖proj_{S⁻}‖ summed into a single penalty vector (None when there are no
    negatives). Reused across every source of the query — CLAY's precompute-once design.
    """
    pos_db = [
        torch.nn.functional.normalize(
            _project_db_into_subspace(group, image_features, config.k_pos, config.use_rotation),
            dim=1, eps=_EPS_NORM,
        )
        for group in _polarity_groups(T_pos, prompt_bank, config.per_condition)
    ]

    neg_groups = _polarity_groups(T_neg, prompt_bank, config.per_condition)
    neg_penalty = None
    if neg_groups:
        neg_penalty = torch.zeros(image_features.shape[0])
        for group in neg_groups:
            coords = _project_db_into_subspace(group, image_features, config.k_neg, config.use_rotation)
            neg_penalty = neg_penalty + coords.norm(dim=1)

    return pos_db, neg_penalty


def _combine_scores(image_features, cols, pos_db, neg_penalty, config):
    """Composite Track S score for a BATCH of reference columns: [N, m] (single owner of the formula).

    score(:, j) = cos(v_d, v_cols[j])  +  mean_a cos_{S⁺_a}(v_d, v_cols[j])  −  λ·Σ_b ‖proj_{S⁻_b}(v_d)‖.
    Batching the source columns turns thousands of per-source matrix-vector products into a few BLAS
    GEMMs — the speedup that makes the full benchmark/ablation tractable on CPU.
    """
    scores = torch.zeros(image_features.shape[0], len(cols))
    if config.identity_anchor:                         # identity anchor — base cosine to each reference
        scores = scores + image_features @ image_features[cols].T
    if pos_db:                                         # mean cosine across the positive subspaces
        scores = scores + sum(D @ D[cols].T for D in pos_db) / len(pos_db)
    if neg_penalty is not None:                        # asymmetric negation penalty (broadcast over columns)
        scores = scores - config.lam * neg_penalty.unsqueeze(1)
    return scores


def _batched_rankings(T_pos, T_neg, image_features, prompt_bank, config, sources, top_k, chunk=512):
    """All of a query's sources ranked at once → {src_idx: top-`top_k` ranking}. Memory-bounded by `chunk`.

    Precompute the subspaces once, then score sources in chunks so the score matrix stays [N, chunk].
    Each source is excluded from its own column (CONTRACT §5), then torch.topk takes only the prefix the
    metrics consume — Recall@K/Precision@K read just ranking[:K], so materialising the full 19962-length
    order per source (hundreds of millions of Python ints across the benchmark) would be pure waste.
    """
    pos_db, neg_penalty = _precompute(T_pos, T_neg, image_features, prompt_bank, config)
    rankings = {}
    for start in range(0, len(sources), chunk):
        cols = torch.as_tensor(sources[start:start + chunk], dtype=torch.long)
        scores = _combine_scores(image_features, cols, pos_db, neg_penalty, config)
        scores[cols, torch.arange(len(cols))] = float("-inf")   # each source never ranks itself
        top = torch.topk(scores, top_k, dim=0).indices            # [top_k, len(cols)] — column j ranks source cols[j]
        for j, src in enumerate(cols.tolist()):
            rankings[src] = top[:, j].tolist()
    return rankings


def make_get_ranking(query_str, image_features, prompt_bank, config=TrackSConfig()):
    """Curry one query string into the get_ranking(src_idx) callback the eval engine expects (CONTRACT §7).

    Single-source path: builds the subspaces, then scores one reference column on demand. For the full
    benchmark prefer the batched driver (evaluate_tier2a_s), which ranks all of a query's sources together.
    """
    T_pos, T_neg = parse_query(query_str)
    pos_db, neg_penalty = _precompute(T_pos, T_neg, image_features, prompt_bank, config)

    def get_ranking(src_idx):
        cols = torch.as_tensor([src_idx], dtype=torch.long)
        scores = _combine_scores(image_features, cols, pos_db, neg_penalty, config)[:, 0]
        scores[src_idx] = float("-inf")
        return torch.argsort(scores, descending=True).tolist()

    return get_ranking


def score(T_pos, T_neg, v_ref_idx, image_features, prompt_bank, config=TrackSConfig()):
    """CONTRACT §7-parity single-shot wrapper: build the subspaces, rank the corpus for one source."""
    pos_db, neg_penalty = _precompute(T_pos, T_neg, image_features, prompt_bank, config)
    cols = torch.as_tensor([v_ref_idx], dtype=torch.long)
    scores = _combine_scores(image_features, cols, pos_db, neg_penalty, config)[:, 0]
    scores[v_ref_idx] = float("-inf")
    return torch.argsort(scores, descending=True).tolist()


# ---------------------------------------------------------------------------
# Benchmark driver + ablation sweep
# ---------------------------------------------------------------------------
def _evaluate(config, image_features, prompt_bank, gt_list, ks):
    """Score one config over every query via the batched ranker, aggregating with eval.evaluate_query.

    Per query, rank all of its sources at once, hand eval an O(1) {src: ranking} lookup, and let the
    shared evaluate_query own the metric averaging (no duplicate metric logic — DRY, CONTRACT-faithful).
    """
    top_k = max(ks)                                    # the metrics only ever read ranking[:max(ks)]
    results = {}
    for entry in gt_list:
        T_pos, T_neg = parse_query(entry["query"])
        sources = [int(s) for s in entry["ground_truth"]]
        rankings = _batched_rankings(T_pos, T_neg, image_features, prompt_bank, config, sources, top_k)
        results[entry["query"]] = evaluate_query(entry["ground_truth"], rankings.__getitem__, ks)
    return results


def evaluate_tier2a_s(config=TrackSConfig(), ks=(1, 5, 10), save=True):
    """Score one Track S config on the full benchmark, print the table, save output/tier2a_S/*.csv."""
    image_features = load_image_features()
    prompt_bank = load_prompt_bank()
    gt_list = load_eval_json(find_eval_json())

    results = _evaluate(config, image_features, prompt_bank, gt_list, ks)
    print(f"\nTier-2a Track S ({config.tag()}) - {len(gt_list)} queries\n")
    print(format_results_table(results, ks=ks))

    if save:
        save_results_csv(results, output_subdir("tier2a") / f"tier2a_{config.tag()}.csv", ks=ks)
    return results


def _ablation_configs():
    """The S_plan.md §6 sweep: default + one axis varied at a time, others held at default."""
    default = TrackSConfig()
    configs = [default]
    configs += [TrackSConfig(lam=lam) for lam in (0.05, 0.2, 0.5)]                       # §6.2
    configs += [TrackSConfig(k_pos=k, k_neg=k) for k in (5, 10, 50)]                     # §6.1
    configs += [TrackSConfig(per_condition=False)]                                       # §6.3 stacked
    configs += [TrackSConfig(use_rotation=False)]                                        # §6.4 no rotation
    configs += [TrackSConfig(identity_anchor=False)]                                     # plan-literal pure subspace
    return configs


def run_ablation(ks=(1, 5, 10)):
    """Run the full ablation sweep once, reusing the loaded DB/bank across every config."""
    image_features = load_image_features()
    prompt_bank = load_prompt_bank()
    gt_list = load_eval_json(find_eval_json())

    for config in _ablation_configs():
        results = _evaluate(config, image_features, prompt_bank, gt_list, ks)
        print(f"\nTier-2a Track S ({config.tag()}) - {len(gt_list)} queries\n")
        print(format_results_table(results, ks=ks))
        save_results_csv(results, output_subdir("tier2a") / f"tier2a_{config.tag()}.csv", ks=ks)


if __name__ == "__main__":
    run_ablation()
