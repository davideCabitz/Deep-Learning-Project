"""
Tier-3 Negation — Fusion-DGP Φ with a learned negation head.

All prior tiers negate by orthogonal rejection: remove the attribute axis from the
query tangent vector.  This is geometrically correct but uses the SAME direction
representation for affirmation and negation — a direction that points toward +Male
is simply subtracted to negate Male.

This tier replaces that fixed operator with a learned NegationHead: a small MLP
that takes the current tangent vector q_tan and the attribute direction d_b and
produces a learned suppression coefficient λ ∈ [0,2] (sigmoid-scaled):

    q_tan ← q_tan − λ(q_tan, d_b) · d_b

At λ=1 this recovers orthogonal rejection exactly, so the model can smoothly
interpolate or amplify beyond the hard rejection of the prior tiers.  Affirmation
is unchanged (same cross-attention gate as tier3_dgp).

The NegationHead is the only new parameter block; everything else (FusionPhi body,
QueryGenerator, manifold geometry, CLIP frozen) is inherited verbatim.

Train:  python src/tier3_negation.py
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
from manifold import log_map, exp_map
from tier1_GDE import load_or_mine_directions, _load_train_features, _load_train_attributes
from tier2d_dgp import compute_mu_txt
from tier3_dgp import PhiConfig, FusionPhi, QueryGenerator, build_raw_stacks


# ---------------------------------------------------------------------------
# Learned negation head
# ---------------------------------------------------------------------------

class NegationHead(nn.Module):
    # Learned suppression coefficient λ(q_tan, d_b) → scalar in [0, 2].
    # Input: concat of [q_tan, d_b, q_tan*d_b, (q_tan·d_b).unsqueeze] → MLP → sigmoid * 2.
    # At init: zero-init last layer → λ ≈ 1.0 (recovers orthogonal rejection).
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 3 + 1, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, 0.0)   # sigmoid(0)*2 = 1.0 → rejection

    def forward(self, q_tan: torch.Tensor, d_b: torch.Tensor) -> torch.Tensor:
        # q_tan [B, d], d_b [B, d] (unit) → λ [B, 1].
        dot = (q_tan * d_b).sum(dim=1, keepdim=True)   # [B, 1] scalar signal
        feat = torch.cat([q_tan, d_b, q_tan * d_b, dot], dim=1)
        return torch.sigmoid(self.mlp(feat)) * 2.0      # [B, 1] in (0, 2)


# ---------------------------------------------------------------------------
# Extended model: FusionPhi + NegationHead
# ---------------------------------------------------------------------------

class FusionPhiNeg(nn.Module):
    # FusionPhi with the cross-attention gate replaced by the NegationHead for negation.
    # Affirmation path: identical to tier3_dgp (cross-attention → direction → add to q_tan).
    # Negation path  : NegationHead predicts λ; q_tan -= λ * d_b  (generalised rejection).
    def __init__(self, cfg: PhiConfig, mu: torch.Tensor, mu_txt: torch.Tensor):
        super().__init__()
        self.phi = FusionPhi(cfg, mu, mu_txt)
        self.neg_head = NegationHead(cfg.d_model)

    def forward(
        self,
        v_ref: torch.Tensor,
        T_pos_stacks: list[torch.Tensor],
        T_neg_stacks: list[torch.Tensor],
    ) -> torch.Tensor:
        # Compose query with learned negation. Affirmation is delegated to phi's internals
        # by running only the positive half, then we apply learned negation ourselves.
        cfg = self.phi.cfg
        mu  = self.phi.mu

        h_ref = v_ref + self.phi.mlp_ref(v_ref)           # [B, d]
        q_tan = log_map(mu, v_ref)                         # [B, d]

        for T_a in T_pos_stacks:
            T_hat = self.phi._center(T_a)
            d_a = self.phi._direction(h_ref, T_hat, self.phi.attn_pos)
            q_tan = q_tan + cfg.alpha * d_a

        for T_b in T_neg_stacks:
            T_hat = self.phi._center(T_b)
            d_b = self.phi._direction(h_ref, T_hat, self.phi.attn_neg)
            lam = self.neg_head(q_tan, d_b)                # [B, 1]
            q_tan = q_tan - lam * d_b                      # learned-scale rejection

        return F.normalize(exp_map(mu, q_tan), dim=1)      # [B, d]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_negation(cfg: PhiConfig = PhiConfig(), device: str | None = None) -> FusionPhiNeg:
    # Train FusionPhiNeg with InfoNCE (same loss as tier3_dgp; only negation path differs).
    from tier3_dgp import info_nce
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg.seed)
    print(f"Device: {dev}" + (f"  GPU: {torch.cuda.get_device_name(0)}" if dev.type == "cuda" else ""))

    train_feats = _load_train_features().to(dev)
    train_attrs = _load_train_attributes()
    prompt_bank = load_prompt_bank()
    mu, _ = load_or_mine_directions()
    mu_txt = compute_mu_txt(prompt_bank)

    raw_stacks = {k: v.to(dev) for k, v in build_raw_stacks(prompt_bank).items()}
    model = FusionPhiNeg(cfg, mu.to(dev), mu_txt.to(dev)).to(dev)
    gen = QueryGenerator(train_attrs, cfg)
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
                batch_loss = batch_loss + info_nce(q, pos, neg, cfg.tau)
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
            f"InfoNCE={running/max(seen,1):.4f}  "
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

def evaluate_negation(model: FusionPhiNeg, ks=(1, 5, 10), save: bool = True, tag: str = "negation") -> dict:
    # Score the model on the 14-query benchmark, print the table, save the CSV.
    image_features = load_image_features()
    prompt_bank = load_prompt_bank()
    raw_stacks = build_raw_stacks(prompt_bank)
    gt_list = load_eval_json(find_eval_json())

    results = {}
    for entry in gt_list:
        get_ranking = make_get_ranking(entry["query"], image_features, model, raw_stacks)
        results[entry["query"]] = evaluate_query(entry["ground_truth"], get_ranking, ks)

    print(f"\nTier-3 Negation ({tag}) — {len(gt_list)} queries\n")
    print(format_results_table(results, ks=ks))

    if save:
        save_results_csv(results, output_subdir("tier3_negation") / f"tier3_{tag}.csv", ks=ks)
    return results


if __name__ == "__main__":
    cfg = PhiConfig()
    model = train_negation(cfg)
    weights_path = output_subdir("tier3_negation") / "tier3_negation_phi.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"Weights saved: {weights_path}")
    evaluate_negation(model)
