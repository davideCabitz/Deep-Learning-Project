"""
Tier-3 NegGate Φ (P1) — learned gated subspace rejection, ranking-supervised.

Every prior tier negates by a FIXED rank-1 orthogonal rejection of one averaged
attribute axis, applied after the query is pinned to Log_μ(v_ref):
    q_tan ← q_tan − (q_tan·d_b) · d_b
On an identity-anchored query this barely reorders the frozen DB, so the "absence"
constraint is never really enforced — every method scores 0.000 on `-Male,-Mustache`.

NegGate replaces that with a reference/query-conditioned rejection of a small
per-attribute CLAY subspace:
    B_b   = centered paraphrase directions of attribute b               [d, k_neg]
    a_b   = softmax( (W_q h_ref)·(W_k B_b)ᵀ / √d_gate )                 [k_neg]  which dims matter
    C_b   = orthonormal( B_b · diag-weighted by a_b )   (QR)            [d, r]
    λ_b   = 2·sigmoid( g([q_tan, ĉ_b, q_tan·ĉ_b]) )                    ∈ (0,2)  strength
    q_tan ← q_tan − λ_b · C_b (C_bᵀ q_tan)                             gated subspace rejection

k_neg=1 + zero-init g (λ=1) recovers the exact Tier-3 rejection, so the model can
only improve on it. The decisive change is TRAINING SIGNAL: the ListNet list already
contains "satisfies T+, violates T−" hard negatives (label 0), and the rejection is on
the gradient path to (W_q, W_k, g), so the model is explicitly trained to demote them —
which the post-gate fixed rejection never was.

Positive path, disentanglement loss, and distance-based hard-neg mining are inherited
verbatim from tier3_combined (CLIP frozen, DB never re-encoded).

Train:  python src/tier3_neggate.py
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
from manifold import log_map, exp_map
from tier1_GDE import load_or_mine_directions, _load_train_features, _load_train_attributes
from tier2d_dgp import compute_mu_txt
from tier3_dgp import PhiConfig, FusionPhi, QueryGenerator, build_raw_stacks
from tier3_contrastive import listnet_loss
from tier3_train_utils import distance_mine, disentanglement_loss


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NegGateConfig:
    # Tier-3 NegGate — FusionPhi positive path + learned gated subspace rejection.
    d_model: int = 512
    n_heads: int = 4
    alpha: float = 1.0
    share_pos_neg: bool = True
    center: bool = True

    # Negation gate
    k_neg: int = 8                 # candidate rejection dims per negated attribute (CLAY subspace width)
    d_gate: int = 64               # low-rank gate dimension (which dims matter)

    # Training (mirrors CombinedConfig so results are directly comparable)
    lr: float = 1e-4
    weight_decay: float = 1e-2
    tau: float = 0.07
    epochs: int = 30
    batch_queries: int = 64
    k_pos_max: int = 3
    k_neg_max: int = 2
    n_pos_targets: int = 8
    n_hard_neg: int = 32
    n_hard_neg_final: int = 16
    hamming_max: int = 2
    seed: int = 0

    # Auxiliary disentanglement loss
    lambda_dis: float = 0.1
    n_dis_pairs: int = 8

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
# Learned gated subspace negation
# ---------------------------------------------------------------------------

class GatedSubspaceNegation(nn.Module):
    # Reference-conditioned rejection of a per-attribute CLAY subspace.
    # Low-rank gate picks WHICH of the k_neg candidate dims matter for this reference;
    # a q_tan-conditioned MLP sets HOW HARD to reject (λ). QR makes the gated columns an
    # orthonormal basis so the removal is a proper subspace projection, not a skewed one.
    # Zero-init tail ⇒ λ=1 at start; with k_neg=1 this is exactly Tier-3's rank-1 rejection.
    def __init__(self, cfg: NegGateConfig):
        super().__init__()
        d = cfg.d_model
        self.w_q = nn.Linear(d, cfg.d_gate, bias=False)
        self.w_k = nn.Linear(d, cfg.d_gate, bias=False)
        self.scale = 1.0 / (cfg.d_gate ** 0.5)
        self.g = nn.Sequential(
            nn.Linear(2 * d + 1, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        nn.init.zeros_(self.g[-1].weight)          # λ = 2·sigmoid(0) = 1.0 at init → orthogonal rejection
        nn.init.zeros_(self.g[-1].bias)

    def forward(self, h_ref: torch.Tensor, q_tan: torch.Tensor, B_b: torch.Tensor) -> torch.Tensor:
        # h_ref, q_tan [B, d]; B_b [k, d] centered paraphrase directions of ONE negated attr.
        # Returns q_tan with the gated subspace removed. Per-reference gate → per-reference basis,
        # so we build each row independently and re-stack (no in-place write on a grad tensor).
        qk = self.w_q(h_ref)                                   # [B, d_gate]
        kk = self.w_k(B_b)                                     # [k, d_gate]
        a = F.softmax((qk @ kk.T) * self.scale, dim=1)         # [B, k] per-reference dim weights
        rows = []
        for i in range(h_ref.shape[0]):
            q_i = q_tan[i]                                     # [d]
            cols = a[i].unsqueeze(1) * B_b                     # [k, d] reweighted candidate axes
            Q, _ = torch.linalg.qr(cols.T)                     # [d, r] orthonormal basis of the span
            proj = Q @ (Q.T @ q_i)                             # component of q_tan inside the subspace
            c_hat = F.normalize(proj, dim=0, eps=1e-8)         # summary direction for the strength MLP
            dot = (q_i * c_hat).sum().unsqueeze(0)             # [1]
            lam = 2.0 * torch.sigmoid(self.g(torch.cat([q_i, c_hat, dot]))).squeeze(0)  # scalar in (0,2)
            rows.append(q_i - lam * proj)
        return torch.stack(rows, dim=0)                        # [B, d]


# ---------------------------------------------------------------------------
# NegGate model — FusionPhi positive path + GatedSubspaceNegation
# ---------------------------------------------------------------------------

class FusionPhiNegGate(nn.Module):
    # Positive/identity composition delegated to FusionPhi; negation done by the learned
    # gated subspace rejection. q = normalize(Exp_μ( Log_μ(v_ref) + Σα·d_a − Σ λ_b·Proj_b )).
    def __init__(self, cfg: NegGateConfig, mu: torch.Tensor, mu_txt: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.phi = FusionPhi(cfg.as_phi_config(), mu, mu_txt)
        self.neg = GatedSubspaceNegation(cfg)

    def forward(
        self,
        v_ref: torch.Tensor,
        T_pos_stacks: list[torch.Tensor],
        T_neg_stacks: list[torch.Tensor],
    ) -> torch.Tensor:
        mu = self.phi.mu
        h_ref = v_ref + self.phi.mlp_ref(v_ref)               # [B, d]
        q_tan = log_map(mu, v_ref)                            # [B, d]

        for T_a in T_pos_stacks:
            d_a = self.phi._direction(h_ref, self.phi._center(T_a), self.phi.attn_pos)
            q_tan = q_tan + self.cfg.alpha * d_a

        for T_b in T_neg_stacks:
            B_b = self.phi._center(T_b)                       # [k_b, d] centered paraphrase dirs
            q_tan = self.neg(h_ref, q_tan, B_b)

        return F.normalize(exp_map(mu, q_tan), dim=1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_neggate(cfg: NegGateConfig = NegGateConfig(), device: str | None = None) -> FusionPhiNegGate:
    # Train NegGate with ListNet + λ_dis·disentanglement, distance-mined hard negatives.
    # The ranking list carries the T−-violating hard negatives (label 0), so the gate learns
    # to demote them — the training signal the fixed rejection lacked. Returns model (CPU).
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    print(f"Device: {dev}" + (f"  GPU: {torch.cuda.get_device_name(0)}" if dev.type == "cuda" else ""))
    print(f"NegGate: ListNet + gated subspace rejection (k_neg={cfg.k_neg}) + disentanglement (λ={cfg.lambda_dis})")

    train_feats = _load_train_features().to(dev)
    train_attrs = _load_train_attributes()
    prompt_bank = load_prompt_bank()
    mu, _ = load_or_mine_directions()
    mu_txt = compute_mu_txt(prompt_bank)

    raw_stacks = {k: v.to(dev) for k, v in build_raw_stacks(prompt_bank).items()}
    model = FusionPhiNegGate(cfg, mu.to(dev), mu_txt.to(dev)).to(dev)
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
            batch_rank, n = torch.zeros((), device=dev), 0

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
                ).squeeze(0)                                    # [d]

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

            batch_dis = disentanglement_loss(model, train_feats, raw_stacks, cfg.n_dis_pairs, rng, dev)
            loss = batch_rank / n + cfg.lambda_dis * batch_dis
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            running_rank += (batch_rank / n).item() * n
            running_dis += batch_dis.item()
            seen += n
            step += 1

        sched.step()
        elapsed = time.time() - t_start
        epoch_t = time.time() - t_epoch
        remaining = elapsed / max(step, 1) * (total_steps - step)
        print(
            f"  epoch {epoch+1:02d}/{cfg.epochs}  "
            f"ListNet={running_rank/max(seen,1):.4f}  "
            f"Dis={running_dis/max(steps_per_epoch,1):.4f}  "
            f"epoch_time={epoch_t:.0f}s  eta={remaining/60:.1f}min"
        )

    return model.cpu()


# ---------------------------------------------------------------------------
# Retrieval seam — CONTRACT §5/§7
# ---------------------------------------------------------------------------

def make_get_ranking(
    query_str: str,
    image_features: torch.Tensor,
    model: FusionPhiNegGate,
    raw_stacks: dict[str, torch.Tensor],
) -> callable:
    # Curry one query string into get_ranking(src_idx): compose q, cosine-score frozen DB.
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

def evaluate_neggate(model: FusionPhiNegGate, ks=(1, 5, 10), save: bool = True, tag: str = "neggate") -> dict:
    # Score on the 14-query benchmark, print table, save CSV.
    image_features = load_image_features()
    prompt_bank = load_prompt_bank()
    raw_stacks = build_raw_stacks(prompt_bank)
    gt_list = load_eval_json(find_eval_json())

    results = {}
    for entry in gt_list:
        get_ranking = make_get_ranking(entry["query"], image_features, model, raw_stacks)
        results[entry["query"]] = evaluate_query(entry["ground_truth"], get_ranking, ks)

    print(f"\nTier-3 NegGate ({tag}) — {len(gt_list)} queries\n")
    print(format_results_table(results, ks=ks))

    if save:
        save_results_csv(results, output_subdir("tier3_neggate") / f"tier3_{tag}.csv", ks=ks)
    return results


if __name__ == "__main__":
    cfg = NegGateConfig()
    model = train_neggate(cfg)
    weights_path = output_subdir("tier3_neggate") / "tier3_neggate_phi.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"Weights saved: {weights_path}")
    evaluate_neggate(model)
