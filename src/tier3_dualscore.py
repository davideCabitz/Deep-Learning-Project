"""
Tier-3 DualScore Φ (P2) — presence/absence factorization, query-conditioned fusion.

Geometry cannot enforce "absence": on an identity-anchored query, orthogonal rejection
of a text/visual axis barely reorders the frozen DB, so `-Male,-Mustache` flat-lines at
0.000 everywhere. The one signal that ever moved it (Tier-4 hybrid) is a discriminative
attribute probe used as a scored penalty — but hybrid caps it with a SINGLE global β,
scores negation only, and trains probes separately from the ranking objective.

DualScore keeps the strong positive fusion Φ and adds a calibrated presence channel used
on BOTH polarities, with QUERY-CONDITIONED strengths trained jointly by ListNet:

    q⁺        = FusionPhiPos(v_ref, T⁺)                 # geodesic identity+positive fusion (frozen recipe)
    P[:,j]    = σ( probe_j(x) )   ∈ [0,1]               # calibrated "image x has attribute j" (cached, frozen)
    s_cos     = X · q⁺
    s_neg     = Σ_{b∈T⁻} ( 1 − P[:,b] )                 # reward ABSENCE of each negated attribute
    s_pos     = Σ_{a∈T⁺}       P[:,a]                   # reward PRESENCE of each positive attribute
    (β, γ)    = softplus( FusionHead([ĥ_ref, mean T̂⁺, mean T̂⁻]) )   # query-conditioned weights
    score     = s_cos + β·s_neg + γ·s_pos               # composite, ranked

Query-conditioning is the upgrade over hybrid's global β: the head sees the actual query
and sets how hard the discriminative channels bite per query. For a pure-positive query
with γ→0 this reduces to the Tier-3 positive baseline; the presence channel only ever adds
a calibrated axis geometry cannot supply. DB CLIP features stay frozen; the probe matrix P
is built once offline and frozen (spec §3.2), never re-encoding the DB.

Two-stage training (VM-ready):
    Stage 1 — BCE-fit attribute probes on frozen TRAIN features (reused from tier3_hybrid).
    Stage 2 — freeze probes, train Φ⁺ + FusionHead with composite-score ListNet.

Run:  python src/tier3_dualscore.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import ATTR_TO_IDX
from clip_features import load_image_features, FEATURE_DIM
from clip_prompts import load_prompt_bank
from eval import parse_query, evaluate_query, format_results_table, load_eval_json, find_eval_json
from results_saver import save_results_csv, output_subdir
from tier1_GDE import load_or_mine_directions, _load_train_features, _load_train_attributes
from tier2d_dgp import compute_mu_txt
from tier3_dgp import QueryGenerator, build_raw_stacks
# Reuse the discriminative + positive machinery and the composite-score ListNet (single owner).
from tier3_hybrid import (
    Tier4Config, AttributeProbes, FusionPhiPos, train_probes, log,
    listnet_loss as listnet_loss_composite,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Configuration — extends Tier4Config with the presence-channel knobs.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DualScoreConfig(Tier4Config):
    # Query-conditioned fusion head width (maps query summary → (β, γ)).
    fusion_hidden: int = 128
    n_rand_neg: int = 16          # random negatives so pure-positive queries still get contrast


# ---------------------------------------------------------------------------
# Query-conditioned fusion head — sets (β, γ) from the query summary.
# ---------------------------------------------------------------------------
class FusionHead(nn.Module):
    # Reads a query summary [ĥ_ref, mean(T̂⁺), mean(T̂⁻)] → two non-negative scalars (β, γ)
    # weighting the absence and presence channels. Zero-init tail ⇒ β=γ=softplus(0)≈0.693 at
    # start (a mild, safe default); the head then learns per-query how hard each channel bites.
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, summary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # summary [3d] → (β, γ) each a positive scalar.
        raw = F.softplus(self.net(summary))               # [2] ≥ 0
        return raw[0], raw[1]


# ---------------------------------------------------------------------------
# DualScore model — positive fusion Φ + query-conditioned fusion head.
# (Probes are held separately and frozen in stage 2, exactly like tier3_hybrid.)
# ---------------------------------------------------------------------------
class DualScoreRetriever(nn.Module):
    def __init__(self, cfg: DualScoreConfig, mu: torch.Tensor, mu_txt: torch.Tensor):
        super().__init__()
        self.phi = FusionPhiPos(cfg, mu, mu_txt)
        self.head = FusionHead(cfg.d_model, cfg.fusion_hidden)
        self.register_buffer("mu_txt", mu_txt)

    def _center(self, T: torch.Tensor) -> torch.Tensor:
        return F.normalize(T - self.mu_txt, dim=-1)

    def query_weights(
        self, v_ref: torch.Tensor, T_pos_stacks: list[torch.Tensor], T_neg_stacks: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Build the query summary and map it to (β, γ). Empty-polarity means → zero vector, so a
        # pure-positive query's summary carries no spurious negative signal.
        d = v_ref.shape[1]
        h_ref = (v_ref + self.phi.mlp_ref(v_ref)).squeeze(0)          # [d]
        pos_mean = (torch.stack([self._center(T).mean(0) for T in T_pos_stacks]).mean(0)
                    if T_pos_stacks else v_ref.new_zeros(d))
        neg_mean = (torch.stack([self._center(T).mean(0) for T in T_neg_stacks]).mean(0)
                    if T_neg_stacks else v_ref.new_zeros(d))
        return self.head(torch.cat([h_ref, pos_mean, neg_mean]))


# ---------------------------------------------------------------------------
# Composite score — presence/absence channels fused with cosine.
# ---------------------------------------------------------------------------
def composite_score(
    q_pos: torch.Tensor,          # [d] unit positive query
    feats: torch.Tensor,          # [C, d] candidate CLIP features (frozen)
    probs: torch.Tensor,          # [C, 40] cached σ(probe) presence probabilities
    pos_cols: list[int],
    neg_cols: list[int],
    beta: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    # score = cos + β·Σ_{b∈T⁻}(1 − P_b) + γ·Σ_{a∈T⁺}P_a.  Absence reward for negatives, presence
    # reward for positives; both from the calibrated probe channel, weighted per-query by (β, γ).
    cos = feats @ q_pos                                              # [C]
    s_neg = (1.0 - probs[:, neg_cols]).sum(dim=1) if neg_cols else feats.new_zeros(feats.shape[0])
    s_pos = probs[:, pos_cols].sum(dim=1) if pos_cols else feats.new_zeros(feats.shape[0])
    return cos + beta * s_neg + gamma * s_pos


# ---------------------------------------------------------------------------
# Stage 2 training — freeze probes, train Φ⁺ + FusionHead with composite ListNet.
# ---------------------------------------------------------------------------
def train_dualscore(
    model: DualScoreRetriever,
    probes: AttributeProbes,
    train_feats: torch.Tensor,
    train_attrs: torch.Tensor,
    raw_stacks: dict[str, torch.Tensor],
    cfg: DualScoreConfig,
) -> list[float]:
    # ListNet over [positives ∪ hard-negatives ∪ random-negatives] scored by the composite. The
    # T⁻-violating hard negatives are exactly what the absence channel must push down — the only
    # place (β, γ) and Φ⁺ get a meaningful joint gradient. Probes are frozen (calibrated axes).
    for p in probes.parameters():
        p.requires_grad_(False)
    probes.eval()

    # Cache presence probs for ALL train features once — frozen, read-only during training.
    with torch.no_grad():
        train_probs = torch.sigmoid(probes(train_feats))            # [N, 40]

    gen = QueryGenerator(train_attrs, cfg.phi_gen_config())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    rng = torch.Generator().manual_seed(cfg.seed + 7)
    N = train_feats.shape[0]

    steps_per_epoch = max(1, gen.N // (cfg.batch_queries * 50))
    total_steps = cfg.epochs * steps_per_epoch
    history: list[float] = []
    step, t_start = 0, time.time()

    for epoch in range(cfg.epochs):
        model.train()
        running, seen = 0.0, 0
        for _ in range(steps_per_epoch):
            opt.zero_grad()
            batch_loss, n = torch.zeros((), device=DEVICE), 0
            for _ in range(cfg.batch_queries):
                sample = gen.sample_query()
                if sample is None:
                    continue
                r, pos_names, neg_names, pos_ids, hard_ids = sample
                pos_cols = [ATTR_TO_IDX[a] for a in pos_names]
                neg_cols = [ATTR_TO_IDX[b] for b in neg_names]

                v_ref = train_feats[r].unsqueeze(0)
                q = model.phi(v_ref, [raw_stacks[nm] for nm in pos_names]).squeeze(0)   # [d]
                beta, gamma = model.query_weights(
                    v_ref, [raw_stacks[nm] for nm in pos_names], [raw_stacks[nm] for nm in neg_names],
                )

                rand_ids = torch.randint(N, (cfg.n_rand_neg,), generator=rng)
                cand = torch.cat([pos_ids, hard_ids, rand_ids]).to(DEVICE)
                cand_feats = train_feats[cand]
                cand_probs = train_probs[cand]

                scores = composite_score(q, cand_feats, cand_probs, pos_cols, neg_cols, beta, gamma)
                batch_loss = batch_loss + listnet_loss_composite(scores, pos_ids.numel(), cfg.tau)
                n += 1

            if n == 0:
                continue
            loss = batch_loss / n
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += loss.item() * n
            seen += n
            step += 1

        sched.step()
        history.append(running / max(seen, 1))
        elapsed = time.time() - t_start
        eta = elapsed / max(step, 1) * (total_steps - step)
        log(f"  [dualscore] epoch {epoch + 1:02d}/{cfg.epochs}  ListNet={history[-1]:.4f}  eta={eta/60:.1f}min")

    return history


# ---------------------------------------------------------------------------
# Retrieval seam — CONTRACT §5/§7 (composite score; probe matrix cached per query).
# ---------------------------------------------------------------------------
def make_get_ranking(
    query_str: str,
    image_features: torch.Tensor,
    db_probs: torch.Tensor,
    model: DualScoreRetriever,
    raw_stacks: dict[str, torch.Tensor],
) -> callable:
    # Curry one query. (β, γ) and the presence sums are source-independent → computed once per query;
    # only the per-source q⁺ + cosine runs inside get_ranking.
    T_pos, T_neg = parse_query(query_str)
    pos_cols = [ATTR_TO_IDX[a] for a in T_pos]
    neg_cols = [ATTR_TO_IDX[b] for b in T_neg]
    model.eval()

    @torch.no_grad()
    def get_ranking(src_idx: int) -> list[int]:
        v_ref = image_features[src_idx].unsqueeze(0)
        q = model.phi(v_ref, [raw_stacks[nm] for nm in T_pos]).squeeze(0)
        beta, gamma = model.query_weights(
            v_ref, [raw_stacks[nm] for nm in T_pos], [raw_stacks[nm] for nm in T_neg],
        )
        scores = composite_score(q, image_features, db_probs, pos_cols, neg_cols, beta, gamma)
        scores[src_idx] = float("-inf")
        return torch.argsort(scores, descending=True).tolist()

    return get_ranking


def evaluate_dualscore(
    model: DualScoreRetriever,
    probes: AttributeProbes,
    raw_stacks: dict[str, torch.Tensor],
    ks=(1, 5, 10),
    save: bool = True,
    tag: str = "dualscore",
) -> dict:
    # Score DualScore on the 14-query benchmark. Caches the DB presence matrix once (frozen DB).
    image_features = load_image_features().to(DEVICE)
    probes.eval()
    with torch.no_grad():
        db_probs = torch.sigmoid(probes(image_features))            # [N, 40] built once, read-only
    gt_list = load_eval_json(find_eval_json())

    results = {}
    for entry in gt_list:
        get_ranking = make_get_ranking(entry["query"], image_features, db_probs, model, raw_stacks)
        results[entry["query"]] = evaluate_query(entry["ground_truth"], get_ranking, ks)

    print(f"\nTier-3 DualScore ({tag}) — {len(gt_list)} queries\n")
    print(format_results_table(results, ks=ks))
    if save:
        save_results_csv(results, output_subdir("tier3_dualscore") / f"tier3_{tag}.csv", ks=ks)
    return results


# ---------------------------------------------------------------------------
# Main — two-stage pipeline.
# ---------------------------------------------------------------------------
def main(cfg: DualScoreConfig = DualScoreConfig()) -> None:
    torch.manual_seed(cfg.seed)
    log(f"device={DEVICE}  probe_epochs={cfg.probe_epochs}  phi_epochs={cfg.epochs}")
    if DEVICE == "cpu":
        log("WARNING: CUDA not available — training on CPU will be slow.")

    train_feats = _load_train_features().to(DEVICE)
    train_attrs = _load_train_attributes()
    prompt_bank = load_prompt_bank()
    mu, _ = load_or_mine_directions()
    mu_txt = compute_mu_txt(prompt_bank)
    raw_stacks = {k: v.to(DEVICE) for k, v in build_raw_stacks(prompt_bank).items()}

    assert train_feats.shape[1] == FEATURE_DIM, train_feats.shape
    assert train_attrs.shape[1] == 40, train_attrs.shape
    assert len(raw_stacks) == 40, len(raw_stacks)

    log("STAGE 1 — training attribute probes (BCE on frozen train features) …")
    probes, _ = train_probes(train_feats, train_attrs, cfg)

    model = DualScoreRetriever(cfg, mu.to(DEVICE), mu_txt.to(DEVICE)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in probes.parameters())
    log(f"params total (Φ⁺ + head + probes) = {n_params/1e6:.3f}M")

    log("STAGE 2 — training Φ⁺ + FusionHead (composite ListNet, probes frozen) …")
    train_dualscore(model, probes, train_feats, train_attrs, raw_stacks, cfg)

    out_dir = output_subdir("tier3_dualscore")
    torch.save(model.state_dict(), out_dir / "tier3_dualscore_phi.pt")
    torch.save(probes.state_dict(), out_dir / "tier3_dualscore_probes.pt")
    log(f"[OK] saved weights → {out_dir}")

    evaluate_dualscore(model, probes, raw_stacks)


if __name__ == "__main__":
    main()
