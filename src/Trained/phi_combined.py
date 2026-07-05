"""
Phi-Combined — FusionPhiNeg + ListNet + distance-based hard negative mining
               + attribute disentanglement auxiliary loss.

Three orthogonal improvements over the plain InfoNCE-trained Phi-Gate (the prior best):

  1. ListNet ranking loss — metric-consistent objective. (NegationHead and this loss
     were each first isolated as standalone ablations before being combined here; the
     standalone variants are retired now that this combined model supersedes them.)

  2. Learned NegationHead — λ-scaled rejection trained by ListNet's stronger gradient
     signal (InfoNCE was too weak to train λ; ListNet flows gradients through the
     full ranked list).

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

Train:  python src/phi_combined.py
"""

from __future__ import annotations

import time
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip_features import load_image_features
from clip_prompts import load_prompt_bank
from eval import parse_query, evaluate_query, format_results_table, load_eval_json, find_eval_json
from results_saver import save_results_csv, output_subdir
from gde import load_or_mine_directions, _load_train_features, _load_train_attributes
from dgp import compute_mu_txt
from manifold import log_map, exp_map
from phi_gate import FusionPhi, PhiConfig, QueryGenerator, build_raw_stacks
from fusion_train_utils import distance_mine, disentanglement_loss


# ---------------------------------------------------------------------------
# ListNet loss (previously a standalone ablation; the sole surviving consumer
# after that ablation was retired, so it now lives here directly)
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
# NegationHead + FusionPhiNeg (previously a standalone ablation; the sole surviving
# consumer after that ablation was retired, so they now live here directly)
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


class FusionPhiNeg(nn.Module):
    # FusionPhi with the cross-attention gate replaced by the NegationHead for negation.
    # Affirmation path: identical to phi_gate (cross-attention → direction → add to q_tan).
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
        mu = self.phi.mu

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
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CombinedConfig:
    # Phi-Combined — all three improvements over the plain InfoNCE-trained Phi-Gate.
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
                        hard_ids_mined = distance_mine(
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
            batch_dis = disentanglement_loss(model, train_feats, raw_stacks, cfg.n_dis_pairs, rng, dev)

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
        save_results_csv(results, output_subdir("phi_combined") / f"tier3_{tag}.csv", ks=ks)
    return results


if __name__ == "__main__":
    cfg = CombinedConfig()
    model = train_combined(cfg)
    weights_path = output_subdir("phi_combined") / "tier3_combined_phi.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"Weights saved: {weights_path}")
    evaluate_combined(model)
