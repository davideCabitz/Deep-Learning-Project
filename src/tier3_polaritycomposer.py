"""
Tier-3 PolarityComposer Φ (P3) — set-transformer over CLAY subspace tokens, geodesic warm-start.

The only method that ever cracked double-negation (`-Male,-Mustache`) was the fully learned
composer — a learned operator can express negation the fixed arithmetic-with-rejection cannot.
But it collapsed overall (MEAN R@1=0.0052) for two fixable reasons: (1) it learned composition
from scratch off a near-identity residual, and (2) it consumed only attribute CENTROIDS, throwing
away CLAY subspace structure.

PolarityComposer fixes both:
  • Warm start from the GEODESIC query. The anchor token is q_tan⁰ = Log_μ(v_ref); the transformer
    outputs a residual Δq and the query is q = normalize(Exp_μ(q_tan⁰ + Δq)). Zero-init output ⇒
    Δq=0 at init ⇒ the model starts exactly at GDE-style identity composition, inheriting the
    positive-query quality the from-scratch composer lacked.
  • CLAY subspace tokens. Each attribute contributes its top-r CLAY tangent-SVD directions
    (build_subspace on the modality-centered paraphrase stack), not a single centroid — the
    concrete CLAY/GDE structure the composer omitted. Polarity embeddings (ref/+/−) let self-
    attention learn asymmetric, cross-attribute affirmation/negation interactions.

Bounded to 2 layers and warm-started, so it trains in the same regime as the other tiers.
Trained with ListNet + distance-based hard-negative mining (the T⁻-violating hard negatives are
the signal for negation). CLIP frozen, DB never re-encoded.

Train:  python src/tier3_polaritycomposer.py
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
from manifold import log_map, exp_map, build_subspace
from tier1_GDE import load_or_mine_directions, _load_train_features, _load_train_attributes
from tier2d_dgp import compute_mu_txt
from tier3_dgp import PhiConfig, QueryGenerator, build_raw_stacks
from tier3_contrastive import listnet_loss
from tier3_train_utils import distance_mine


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PolarityComposerConfig:
    # Tier-3 PolarityComposer — set-transformer refining a geodesic query.
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 2              # shallow: 2 layers over a short token set (warm-started)
    ffn_mult: int = 2
    dropout: float = 0.1
    r_sub: int = 4                 # CLAY subspace directions (tokens) per attribute
    center: bool = True            # modality-gap centering of prompt stacks before SVD

    # Training
    lr: float = 1e-4
    weight_decay: float = 1e-2
    tau: float = 0.07
    epochs: int = 30               # extend to 60 if the ListNet curve is still descending
    batch_queries: int = 64
    k_pos_max: int = 3
    k_neg_max: int = 2
    n_pos_targets: int = 8
    n_hard_neg: int = 32
    n_hard_neg_final: int = 16
    hamming_max: int = 2
    seed: int = 0

    def phi_gen_config(self) -> PhiConfig:
        # Borrow PhiConfig only for QueryGenerator (identical sampling protocol).
        return PhiConfig(
            d_model=self.d_model, tau=self.tau, epochs=self.epochs,
            batch_queries=self.batch_queries, k_pos_max=self.k_pos_max,
            k_neg_max=self.k_neg_max, n_pos_targets=self.n_pos_targets,
            n_hard_neg=self.n_hard_neg, hamming_max=self.hamming_max, seed=self.seed,
        )


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class PolarityComposer(nn.Module):
    # Set-transformer over [anchor q_tan⁰, {+ CLAY dirs}, {− CLAY dirs}] → residual Δq on the
    # geodesic query. Polarity embeddings (0=ref, 1=+, 2=−) are the only structural signal; the
    # transformer learns cross-attribute and asymmetric +/− interaction. Zero-init head ⇒ warm
    # start at GDE composition (Δq=0), so training refines a strong prior rather than learning
    # composition from zero (fixing the from-scratch composer's collapse).
    def __init__(self, cfg: PolarityComposerConfig, mu: torch.Tensor, mu_txt: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.polarity_emb = nn.Embedding(3, d)
        nn.init.normal_(self.polarity_emb.weight, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.n_heads, dim_feedforward=d * cfg.ffn_mult,
            dropout=cfg.dropout, batch_first=True, norm_first=True,
        )
        # enable_nested_tensor=False: incompatible with norm_first and irrelevant for our short,
        # unpadded token sets — set explicitly to silence PyTorch's fallback warning.
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers, enable_nested_tensor=False)

        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d),
        )
        nn.init.zeros_(self.head[-1].weight)          # Δq = 0 at init → warm start at Exp_μ(Log_μ(v_ref))
        nn.init.zeros_(self.head[-1].bias)

        self.register_buffer("mu", mu)
        self.register_buffer("mu_txt", mu_txt)

    def _subspace_dirs(self, T: torch.Tensor) -> torch.Tensor:
        # Top-r CLAY tangent-SVD directions of one attribute's (optionally centered) prompt stack.
        # Returns [r_eff, d] ambient-space direction tokens (r_eff ≤ r_sub, ≤ stack height).
        T_hat = F.normalize(T - self.mu_txt, dim=-1) if self.cfg.center else T
        _, V_k = build_subspace(T_hat, self.cfg.r_sub)     # V_k [d, k_eff]
        return V_k.T                                        # [k_eff, d]

    def forward(
        self,
        v_ref: torch.Tensor,                                # [1, d] unit (single query per call)
        T_pos_stacks: list[torch.Tensor],
        T_neg_stacks: list[torch.Tensor],
    ) -> torch.Tensor:
        # Compose one query: build the token set, refine the geodesic anchor by Δq, lift to sphere.
        mu = self.mu
        q_tan0 = log_map(mu, v_ref)                          # [1, d] geodesic anchor

        toks = [q_tan0 + self.polarity_emb(v_ref.new_zeros(1, dtype=torch.long))]  # ref token
        for T_a in T_pos_stacks:
            dirs = self._subspace_dirs(T_a)                 # [r, d]
            toks.append(dirs + self.polarity_emb(dirs.new_ones(dirs.shape[0], dtype=torch.long)))
        for T_b in T_neg_stacks:
            dirs = self._subspace_dirs(T_b)                 # [r, d]
            toks.append(dirs + self.polarity_emb(dirs.new_full((dirs.shape[0],), 2, dtype=torch.long)))

        seq = torch.cat(toks, dim=0).unsqueeze(0)           # [1, L, d]
        z = self.encoder(seq)                               # [1, L, d]
        dq = self.head(z[:, 0])                             # [1, d] residual from the anchor slot

        q_tan = q_tan0 + dq
        return F.normalize(exp_map(mu, q_tan), dim=1)       # [1, d] unit


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_polaritycomposer(
    cfg: PolarityComposerConfig = PolarityComposerConfig(), device: str | None = None,
) -> PolarityComposer:
    # Train with ListNet + distance-mined hard negatives on synthetic train-split queries.
    # Returns the trained model (CPU).
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg.seed)
    print(f"Device: {dev}" + (f"  GPU: {torch.cuda.get_device_name(0)}" if dev.type == "cuda" else ""))
    print(f"PolarityComposer: {cfg.n_layers}-layer set-transformer, r_sub={cfg.r_sub}, ListNet + distance mining")

    train_feats = _load_train_features().to(dev)
    train_attrs = _load_train_attributes()
    prompt_bank = load_prompt_bank()
    mu, _ = load_or_mine_directions()
    mu_txt = compute_mu_txt(prompt_bank)

    raw_stacks = {k: v.to(dev) for k, v in build_raw_stacks(prompt_bank).items()}
    model = PolarityComposer(cfg, mu.to(dev), mu_txt.to(dev)).to(dev)
    gen = QueryGenerator(train_attrs, cfg.phi_gen_config())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    steps_per_epoch = max(1, gen.N // (cfg.batch_queries * 50))
    total_steps = cfg.epochs * steps_per_epoch
    step, t_start = 0, time.time()

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

                if hard_ids.numel() > 0:
                    with torch.no_grad():
                        hard_ids_mined = distance_mine(
                            q.detach(), hard_ids.to(dev), train_feats, cfg.n_hard_neg_final,
                        )
                else:
                    hard_ids_mined = hard_ids.to(dev)

                pos = train_feats[pos_ids.to(dev)]
                neg = train_feats[hard_ids_mined] if hard_ids_mined.numel() else train_feats[:0]
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
            f"  epoch {epoch+1:02d}/{cfg.epochs}  ListNet={running/max(seen,1):.4f}  "
            f"epoch_time={epoch_t:.0f}s  eta={remaining/60:.1f}min"
        )

    return model.cpu()


# ---------------------------------------------------------------------------
# Retrieval seam — CONTRACT §5/§7
# ---------------------------------------------------------------------------

def make_get_ranking(
    query_str: str,
    image_features: torch.Tensor,
    model: PolarityComposer,
    raw_stacks: dict[str, torch.Tensor],
) -> callable:
    # Curry one query string into get_ranking(src_idx): compose q, cosine-score the frozen DB.
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

def evaluate_polaritycomposer(
    model: PolarityComposer, ks=(1, 5, 10), save: bool = True, tag: str = "polaritycomposer",
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

    print(f"\nTier-3 PolarityComposer ({tag}) — {len(gt_list)} queries\n")
    print(format_results_table(results, ks=ks))

    if save:
        save_results_csv(results, output_subdir("tier3_polaritycomposer") / f"tier3_{tag}.csv", ks=ks)
    return results


if __name__ == "__main__":
    cfg = PolarityComposerConfig()
    model = train_polaritycomposer(cfg)
    weights_path = output_subdir("tier3_polaritycomposer") / "tier3_polaritycomposer.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"Weights saved: {weights_path}")
    evaluate_polaritycomposer(model)
