"""
Tier-3 Contrastive — Fusion-DGP Φ with a listwise ranking loss.

Identical architecture to tier3_dgp.py (FusionPhi cross-attention gate, geodesic
composition on S^{d-1}) but the training objective is replaced:

  InfoNCE (tier3_dgp) → ListNet ranking loss over the full retrieval list.

ListNet (Cao et al. 2007) directly optimises the permutation probability of the
ranked list, making the loss metric-consistent with P@K / R@K.  Given query q and
K database vectors, it minimises the cross-entropy between the ideal score
distribution (positives get score 1, negatives 0) and the softmax of the dot
products — a differentiable ranking surrogate that InfoNCE does not provide.

  L = − Σ_i  y_i · log softmax_i(s / τ)   where s_i = q · x_i / τ

Everything else (CLIP frozen, same QueryGenerator, same manifold geometry) is
unchanged so the ONLY ablation variable is the loss function.

Train:  python src/tier3_contrastive.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import ATTRIBUTE_NAMES
from clip_features import load_image_features
from clip_prompts import load_prompt_bank
from eval import parse_query, evaluate_query, format_results_table, load_eval_json, find_eval_json
from results_saver import save_results_csv, output_subdir
from manifold import log_map, exp_map
from tier1_GDE import load_or_mine_directions, _load_train_features, _load_train_attributes
from tier2d_dgp import compute_mu_txt, _attr_stack
from tier3_dgp import FusionPhi, PhiConfig, QueryGenerator, build_raw_stacks


# ---------------------------------------------------------------------------
# Configuration — only the loss-specific knobs differ from PhiConfig.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ContrastiveConfig:
    # Tier-3 Contrastive — FusionPhi trained with ListNet ranking loss.
    d_model: int = 512
    n_heads: int = 4
    alpha: float = 1.0
    share_pos_neg: bool = True
    center: bool = True

    # Training
    lr: float = 1e-4
    weight_decay: float = 1e-2
    tau: float = 0.07              # softmax temperature for the ranking loss
    epochs: int = 30
    batch_queries: int = 64
    k_pos_max: int = 3
    k_neg_max: int = 2
    n_pos_targets: int = 8
    n_hard_neg: int = 32           # more negatives for a richer ranking list
    hamming_max: int = 2
    seed: int = 0

    def as_phi_config(self) -> PhiConfig:
        return PhiConfig(
            d_model=self.d_model, n_heads=self.n_heads, alpha=self.alpha,
            share_pos_neg=self.share_pos_neg, center=self.center,
            lr=self.lr, weight_decay=self.weight_decay, tau=self.tau,
            epochs=self.epochs, batch_queries=self.batch_queries,
            k_pos_max=self.k_pos_max, k_neg_max=self.k_neg_max,
            n_pos_targets=self.n_pos_targets, n_hard_neg=self.n_hard_neg,
            hamming_max=self.hamming_max, seed=self.seed,
        )


# ---------------------------------------------------------------------------
# ListNet ranking loss
# ---------------------------------------------------------------------------

def listnet_loss(q: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor, tau: float) -> torch.Tensor:
    # ListNet — cross-entropy between ideal P(permutation) and model-scored P(permutation).
    # y_i = 1/|pos| for positives, 0 for negatives; model scores are q·x_i/τ.
    # L = − Σ_i y_i · log softmax_i(s).  Differentiates through the full ranked list,
    # unlike InfoNCE which only contrasts pos vs. hard neg pairs.
    # q [d]; pos [P, d]; neg [M, d] — all unit.
    all_vecs = torch.cat([pos, neg], dim=0)           # [P+M, d]
    scores = (all_vecs @ q) / tau                     # [P+M]
    n_pos = pos.shape[0]
    labels = torch.zeros(scores.shape[0], device=q.device)
    labels[:n_pos] = 1.0 / n_pos                      # uniform weight over positives
    log_probs = F.log_softmax(scores, dim=0)
    return -(labels * log_probs).sum()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_contrastive(cfg: ContrastiveConfig = ContrastiveConfig(), device: str | None = None) -> FusionPhi:
    # Train FusionPhi with ListNet loss. Returns the trained model (CPU).
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg.seed)
    print(f"Device: {dev}" + (f"  GPU: {torch.cuda.get_device_name(0)}" if dev.type == "cuda" else ""))

    train_feats = _load_train_features().to(dev)
    train_attrs = _load_train_attributes()
    prompt_bank = load_prompt_bank()
    mu, _ = load_or_mine_directions()
    mu_txt = compute_mu_txt(prompt_bank)

    raw_stacks = {k: v.to(dev) for k, v in build_raw_stacks(prompt_bank).items()}
    model = FusionPhi(cfg.as_phi_config(), mu.to(dev), mu_txt.to(dev)).to(dev)
    gen = QueryGenerator(train_attrs, cfg.as_phi_config())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    steps_per_epoch = max(1, gen.N // (cfg.batch_queries * 50))
    total_steps = cfg.epochs * steps_per_epoch
    step = 0
    t_start = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        running, seen = 0.0, 0
        t_epoch = time.time()

        for _ in range(steps_per_epoch):
            opt.zero_grad()
            batch_loss, n = torch.zeros((), device=dev), 0
            for _ in range(cfg.batch_queries):
                sample = gen.sample_query()
                if sample is None:
                    continue
                r, pos_names, neg_names, pos_ids, hard_ids = sample
                v_ref = train_feats[r].unsqueeze(0)
                q = model(
                    v_ref,
                    [raw_stacks[nm] for nm in pos_names],
                    [raw_stacks[nm] for nm in neg_names],
                ).squeeze(0)
                pos = train_feats[pos_ids.to(dev)]
                neg = train_feats[hard_ids.to(dev)] if hard_ids.numel() else train_feats[:0]
                batch_loss = batch_loss + listnet_loss(q, pos, neg, cfg.tau)
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
        elapsed = time.time() - t_start
        epoch_t = time.time() - t_epoch
        remaining = elapsed / max(step, 1) * (total_steps - step)
        print(
            f"  epoch {epoch+1:02d}/{cfg.epochs}  "
            f"ListNet={running/max(seen,1):.4f}  "
            f"epoch_time={epoch_t:.0f}s  "
            f"eta={remaining/60:.1f}min"
        )

    return model.cpu()


# ---------------------------------------------------------------------------
# Retrieval seam — CONTRACT §5/§7
# ---------------------------------------------------------------------------

def make_get_ranking(
    query_str: str,
    image_features: torch.Tensor,
    model: FusionPhi,
    raw_stacks: dict[str, torch.Tensor],
) -> callable:
    # Curry one query string into get_ranking(src_idx). Identical seam to tier3_dgp.
    T_pos, T_neg = parse_query(query_str)
    model.eval()

    @torch.no_grad()
    def get_ranking(src_idx: int) -> list[int]:
        q = model(
            image_features[src_idx].unsqueeze(0),
            [raw_stacks[nm] for nm in T_pos],
            [raw_stacks[nm] for nm in T_neg],
        ).squeeze(0)
        scores = image_features @ q
        scores[src_idx] = float("-inf")
        return torch.argsort(scores, descending=True).tolist()

    return get_ranking


# ---------------------------------------------------------------------------
# Evaluation entry point
# ---------------------------------------------------------------------------

def evaluate_contrastive(model: FusionPhi, ks=(1, 5, 10), save: bool = True, tag: str = "contrastive") -> dict:
    # Score a trained model on the 14-query benchmark, print the table, save the CSV.
    image_features = load_image_features()
    prompt_bank = load_prompt_bank()
    raw_stacks = build_raw_stacks(prompt_bank)
    gt_list = load_eval_json(find_eval_json())

    results = {}
    for entry in gt_list:
        get_ranking = make_get_ranking(entry["query"], image_features, model, raw_stacks)
        results[entry["query"]] = evaluate_query(entry["ground_truth"], get_ranking, ks)

    print(f"\nTier-3 Contrastive ({tag}) — {len(gt_list)} queries\n")
    print(format_results_table(results, ks=ks))

    if save:
        save_results_csv(results, output_subdir("tier3_contrastive") / f"tier3_{tag}.csv", ks=ks)
    return results


if __name__ == "__main__":
    cfg = ContrastiveConfig()
    model = train_contrastive(cfg)
    weights_path = output_subdir("tier3_contrastive") / "tier3_contrastive_phi.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"Weights saved: {weights_path}")
    evaluate_contrastive(model)
