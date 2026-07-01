"""
Tier-3 Contrastive + Negation — FusionPhiNeg trained with ListNet ranking loss.

Combines the two orthogonal improvements from tier3_contrastive and tier3_negation:

  tier3_contrastive : FusionPhi    + ListNet   (better loss, same negation)
  tier3_negation    : FusionPhiNeg + InfoNCE   (better negation, same loss)
  THIS FILE         : FusionPhiNeg + ListNet   (both improvements together)

Hypothesis: InfoNCE did not give the NegationHead enough gradient signal to learn
meaningful suppression coefficients. ListNet flows gradients through the full ranked
list and should provide stronger, more metric-consistent supervision for λ — the
learned rejection scale that generalises orthogonal rejection.

Architecture: identical to tier3_negation (FusionPhi cross-attention gate for
affirmation, NegationHead MLP for learned λ-scaled rejection). Loss: ListNet
(Cao et al. 2007), identical to tier3_contrastive. n_hard_neg=32 so the ranking
list is wide enough to give ListNet a meaningful distribution to optimise.

Train:  python src/tier3_contrastive_negation.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip_features import load_image_features
from clip_prompts import load_prompt_bank
from eval import parse_query, evaluate_query, format_results_table, load_eval_json, find_eval_json
from results_saver import save_results_csv, output_subdir
from tier1_GDE import load_or_mine_directions, _load_train_features, _load_train_attributes
from tier2d_dgp import compute_mu_txt
from tier3_dgp import PhiConfig, QueryGenerator, build_raw_stacks
from tier3_contrastive import listnet_loss, ContrastiveConfig
from tier3_negation import FusionPhiNeg, NegationHead


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ContrastiveNegationConfig:
    # FusionPhiNeg + ListNet — union of ContrastiveConfig and PhiConfig knobs.
    d_model: int = 512
    n_heads: int = 4
    alpha: float = 1.0
    share_pos_neg: bool = True
    center: bool = True

    lr: float = 1e-4
    weight_decay: float = 1e-2
    tau: float = 0.07
    epochs: int = 30
    batch_queries: int = 64
    k_pos_max: int = 3
    k_neg_max: int = 2
    n_pos_targets: int = 8
    n_hard_neg: int = 32           # wider list for ListNet
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
# Training
# ---------------------------------------------------------------------------

def train_contrastive_negation(
    cfg: ContrastiveNegationConfig = ContrastiveNegationConfig(),
    device: str | None = None,
) -> FusionPhiNeg:
    # Train FusionPhiNeg with ListNet loss. Returns the trained model (CPU).
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg.seed)
    print(f"Device: {dev}" + (f"  GPU: {torch.cuda.get_device_name(0)}" if dev.type == "cuda" else ""))

    train_feats = _load_train_features().to(dev)
    train_attrs = _load_train_attributes()
    prompt_bank = load_prompt_bank()
    mu, _ = load_or_mine_directions()
    mu_txt = compute_mu_txt(prompt_bank)

    raw_stacks = {k: v.to(dev) for k, v in build_raw_stacks(prompt_bank).items()}
    model = FusionPhiNeg(cfg.as_phi_config(), mu.to(dev), mu_txt.to(dev)).to(dev)
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
    model: FusionPhiNeg,
    raw_stacks: dict[str, torch.Tensor],
) -> callable:
    # Curry one query string into get_ranking(src_idx).
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

def evaluate_contrastive_negation(
    model: FusionPhiNeg,
    ks=(1, 5, 10),
    save: bool = True,
    tag: str = "contrastive_negation",
) -> dict:
    # Score on the 14-query benchmark, print table, save CSV.
    image_features = load_image_features()
    prompt_bank = load_prompt_bank()
    raw_stacks = build_raw_stacks(prompt_bank)
    gt_list = load_eval_json(find_eval_json())

    results = {}
    for entry in gt_list:
        get_ranking = make_get_ranking(entry["query"], image_features, model, raw_stacks)
        results[entry["query"]] = evaluate_query(entry["ground_truth"], get_ranking, ks)

    print(f"\nTier-3 Contrastive+Negation ({tag}) — {len(gt_list)} queries\n")
    print(format_results_table(results, ks=ks))

    if save:
        save_results_csv(
            results,
            output_subdir("tier3_contrastive_negation") / f"tier3_{tag}.csv",
            ks=ks,
        )
    return results


if __name__ == "__main__":
    cfg = ContrastiveNegationConfig()
    model = train_contrastive_negation(cfg)
    weights_path = output_subdir("tier3_contrastive_negation") / "tier3_contrastive_negation_phi.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"Weights saved: {weights_path}")
    evaluate_contrastive_negation(model)
