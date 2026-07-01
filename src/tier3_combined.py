"""
Tier-3 Combined — FusionPhiNeg + ListNet + distance-based hard negative mining
                  + attribute disentanglement auxiliary loss.

Three orthogonal improvements over tier3_contrastive (the current best, P@1=0.0470):

  1. ListNet ranking loss (from tier3_contrastive) — metric-consistent objective.

  2. Learned NegationHead (from tier3_negation) — λ-scaled rejection trained by
     ListNet's stronger gradient signal (InfoNCE was too weak to train λ; ListNet
     flows gradients through the full ranked list).

  3. Distance-based hard negative mining — the current QueryGenerator picks hard
     negatives by attribute logic alone (satisfy T+ but violate T-). This tier adds
     a second stage: among those candidates, keep the top-K closest to the current
     query q in CLIP space (online mining). These are the images the model is most
     confused about right now — the hardest possible training signal. This is
     standard curriculum learning (Schroff et al. 2015, FaceNet) and spec-compliant
     (train split only, no GT leakage).

  4. Attribute disentanglement auxiliary loss — regularises the model to produce
     orthogonal directions for different attributes. For each training step, we
     compute the composed query q for k_dis randomly sampled attribute pairs and
     penalise the absolute cosine similarity between their directions:

       L_dis = (1/|pairs|) Σ_{(a,b)} |cos(d_a, d_b)|

     This directly attacks CLIP's attribute entanglement: attributes like Male and
     Facial_Hair are correlated in CLIP space; the disentanglement loss pushes the
     model to produce directions that are as orthogonal as possible, making negation
     geometrically cleaner.

     Total loss: L = L_ListNet + λ_dis · L_dis   (λ_dis=0.1 default)

Train:  python src/tier3_combined.py
"""

from __future__ import annotations

import time
import random
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
from tier2d_dgp import compute_mu_txt
from tier3_dgp import PhiConfig, FusionPhi, QueryGenerator, build_raw_stacks
from tier3_contrastive import listnet_loss
from tier3_negation import FusionPhiNeg, NegationHead


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CombinedConfig:
    # Tier-3 Combined — all three improvements over tier3_contrastive.
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
    n_hard_neg: int = 32           # attribute-based pool; distance mining subsets this
    n_hard_neg_final: int = 16     # keep this many after distance-based reranking
    hamming_max: int = 2
    seed: int = 0

    # Disentanglement loss
    lambda_dis: float = 0.1        # weight of L_dis relative to L_ListNet
    n_dis_pairs: int = 8           # attribute pairs sampled per step for L_dis

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
# Distance-based hard negative reranking
# ---------------------------------------------------------------------------

def _distance_mine(
    q: torch.Tensor,
    hard_ids: torch.Tensor,
    train_feats: torch.Tensor,
    k: int,
) -> torch.Tensor:
    # Among the attribute-selected hard negatives, keep the k closest to q in CLIP
    # space (highest cosine similarity = hardest for the model right now).
    # q [d] unit; hard_ids [M]; returns [min(k, M)] indices into train_feats.
    if hard_ids.numel() <= k:
        return hard_ids
    cands = train_feats[hard_ids]          # [M, d]
    sims  = cands @ q                      # [M] cosine similarities
    top   = torch.topk(sims, k).indices   # [k] indices into hard_ids
    return hard_ids[top]


# ---------------------------------------------------------------------------
# Attribute disentanglement loss
# ---------------------------------------------------------------------------

def _disentanglement_loss(
    model: FusionPhiNeg,
    train_feats: torch.Tensor,
    raw_stacks: dict[str, torch.Tensor],
    n_pairs: int,
    rng: random.Random,
    dev: torch.device,
) -> torch.Tensor:
    # Sample n_pairs distinct attribute pairs, compute a composed direction for each
    # using a random reference, then penalise |cos(d_a, d_b)| for each pair.
    # Uses a single shared reference to keep cost low (one log_map per call).
    attr_names = list(raw_stacks.keys())
    r_idx = rng.randint(0, train_feats.shape[0] - 1)
    v_ref = train_feats[r_idx].unsqueeze(0)           # [1, d]

    # Collect directions: for each attribute, run the affirmation path of FusionPhi.
    # We detach v_ref from the compute graph for the direction computation to keep
    # the disentanglement loss focused on the gate output, not the reference encoder.
    h_ref = v_ref + model.phi.mlp_ref(v_ref)          # [1, d]
    directions: dict[str, torch.Tensor] = {}
    sampled_attrs = rng.sample(attr_names, min(n_pairs * 2, len(attr_names)))
    for name in sampled_attrs:
        T_hat = model.phi._center(raw_stacks[name])
        d = model.phi._direction(h_ref, T_hat, model.phi.attn_pos)  # [1, d]
        directions[name] = F.normalize(d.squeeze(0), dim=0)         # [d]

    names = list(directions.keys())
    if len(names) < 2:
        return torch.zeros((), device=dev)

    # Sample n_pairs distinct pairs and compute |cos(d_a, d_b)|
    pairs_tried, loss = 0, torch.zeros((), device=dev)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            cos_sim = (directions[names[i]] * directions[names[j]]).sum()
            loss = loss + cos_sim.abs()
            pairs_tried += 1
            if pairs_tried >= n_pairs:
                break
        if pairs_tried >= n_pairs:
            break

    return loss / max(pairs_tried, 1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_combined(
    cfg: CombinedConfig = CombinedConfig(),
    device: str | None = None,
) -> FusionPhiNeg:
    # Train the full combined model. Returns trained model (CPU).
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    print(f"Device: {dev}" + (f"  GPU: {torch.cuda.get_device_name(0)}" if dev.type == "cuda" else ""))
    print(f"Combined: ListNet + NegationHead + distance mining + disentanglement (λ={cfg.lambda_dis})")

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
        running_rank, running_dis, seen = 0.0, 0.0, 0
        t_epoch = time.time()

        for _ in range(steps_per_epoch):
            opt.zero_grad()
            batch_rank, batch_dis, n = torch.zeros((), device=dev), torch.zeros((), device=dev), 0

            for _ in range(cfg.batch_queries):
                sample = gen.sample_query()
                if sample is None:
                    continue
                r, pos_names, neg_names, pos_ids, hard_ids = sample
                v_ref = train_feats[r].unsqueeze(0)

                # Forward pass — get query
                q = model(
                    v_ref,
                    [raw_stacks[nm] for nm in pos_names],
                    [raw_stacks[nm] for nm in neg_names],
                ).squeeze(0)                                    # [d]

                # Distance-based hard negative reranking
                if hard_ids.numel() > 0:
                    with torch.no_grad():
                        hard_ids_mined = _distance_mine(
                            q.detach(), hard_ids.to(dev), train_feats, cfg.n_hard_neg_final,
                        )
                else:
                    hard_ids_mined = hard_ids.to(dev)

                pos = train_feats[pos_ids.to(dev)]
                neg = train_feats[hard_ids_mined] if hard_ids_mined.numel() else train_feats[:0]

                batch_rank = batch_rank + listnet_loss(q, pos, neg, cfg.tau)
                n += 1

            if n == 0:
                continue

            # Disentanglement loss — one call per step (shared reference, cheap)
            batch_dis = _disentanglement_loss(model, train_feats, raw_stacks, cfg.n_dis_pairs, rng, dev)

            loss = batch_rank / n + cfg.lambda_dis * batch_dis
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            running_rank += (batch_rank / n).item() * n
            running_dis  += batch_dis.item()
            seen += n
            step += 1

        sched.step()
        elapsed  = time.time() - t_start
        epoch_t  = time.time() - t_epoch
        remaining = elapsed / max(step, 1) * (total_steps - step)
        print(
            f"  epoch {epoch+1:02d}/{cfg.epochs}  "
            f"ListNet={running_rank/max(seen,1):.4f}  "
            f"Dis={running_dis/max(steps_per_epoch,1):.4f}  "
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

def evaluate_combined(
    model: FusionPhiNeg,
    ks=(1, 5, 10),
    save: bool = True,
    tag: str = "combined",
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

    print(f"\nTier-3 Combined ({tag}) — {len(gt_list)} queries\n")
    print(format_results_table(results, ks=ks))

    if save:
        save_results_csv(results, output_subdir("tier3_combined") / f"tier3_{tag}.csv", ks=ks)
    return results


if __name__ == "__main__":
    cfg = CombinedConfig()
    model = train_combined(cfg)
    weights_path = output_subdir("tier3_combined") / "tier3_combined_phi.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"Weights saved: {weights_path}")
    evaluate_combined(model)
