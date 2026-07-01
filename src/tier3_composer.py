"""
Tier-3 Composer — Fully learned compositional query builder (CLIP frozen).

All prior tiers compose the query as explicit latent arithmetic:
    q_tan = Log_μ(v_ref) + Σ α·d_a  −  Σ reject(d_b)

The Composer replaces this formula entirely.  A small transformer (the Composer)
takes the full sequence [v_ref, t⁺₁, …, t⁺ₙ, t⁻₁, …, t⁻ₘ] as input tokens and
produces a single query vector q by pooling the CLS output:

    q = normalize( MLP( TransformerEncoder([v_ref; T_pos; T_neg]) [CLS] ) )

Positive and negative tokens are distinguished by learned polarity embeddings
(+1 / −1), not by separate processing paths.  The model learns composition,
negation, and their interaction jointly from ranking supervision.

No log/exp maps, no hand-coded rejection — the Composer is architecture-level
evidence that the manifold arithmetic can be replaced by a data-driven operator.

CLIP stays frozen.  The Composer is trained end-to-end with InfoNCE on synthetic
train-split queries (identical QueryGenerator as prior tiers).

Train:  python src/tier3_composer.py
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
from tier1_GDE import _load_train_features, _load_train_attributes
from tier2d_dgp import compute_mu_txt, _attr_stack
from tier3_dgp import PhiConfig, QueryGenerator, build_raw_stacks, info_nce


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ComposerConfig:
    # Tier-3 Composer hyperparameters.
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 2              # transformer depth; 2 layers suffices for short sequences
    ffn_mult: int = 2              # FFN hidden = d_model * ffn_mult
    dropout: float = 0.1

    # Training
    lr: float = 1e-4
    weight_decay: float = 1e-2
    tau: float = 0.07
    epochs: int = 30
    batch_queries: int = 64
    k_pos_max: int = 3
    k_neg_max: int = 2
    n_pos_targets: int = 8
    n_hard_neg: int = 16
    hamming_max: int = 2
    seed: int = 0

    def as_phi_config(self) -> PhiConfig:
        # Borrow PhiConfig only for QueryGenerator (same sampling protocol).
        return PhiConfig(
            d_model=self.d_model, n_heads=self.n_heads,
            lr=self.lr, weight_decay=self.weight_decay, tau=self.tau,
            epochs=self.epochs, batch_queries=self.batch_queries,
            k_pos_max=self.k_pos_max, k_neg_max=self.k_neg_max,
            n_pos_targets=self.n_pos_targets, n_hard_neg=self.n_hard_neg,
            hamming_max=self.hamming_max, seed=self.seed,
        )


# ---------------------------------------------------------------------------
# The Composer model
# ---------------------------------------------------------------------------

class QueryComposer(nn.Module):
    # Fully learned compositional retrieval: transformer encoder over attribute token sequence.
    # CLS token → MLP → normalize → query on S^{d-1}.
    # Polarity embeddings distinguish +/- tokens; no explicit arithmetic.
    def __init__(self, cfg: ComposerConfig, mu_txt: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # Learnable CLS token prepended to every sequence.
        self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        # Polarity embeddings: 0=reference, 1=positive, 2=negative.
        self.polarity_emb = nn.Embedding(3, d)
        nn.init.normal_(self.polarity_emb.weight, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.n_heads,
            dim_feedforward=d * cfg.ffn_mult,
            dropout=cfg.dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

        self.register_buffer("mu_txt", mu_txt)

    def _center(self, T: torch.Tensor) -> torch.Tensor:
        # Center and normalize a prompt stack (same FIX-1 as all Tier-3 models).
        return F.normalize(T - self.mu_txt, dim=-1)

    def forward(
        self,
        v_ref: torch.Tensor,             # [B, d] unit
        T_pos_stacks: list[torch.Tensor],  # list of [n_a, d]
        T_neg_stacks: list[torch.Tensor],  # list of [n_b, d]
    ) -> torch.Tensor:
        # Build the token sequence: [CLS, v_ref, mean(T_pos_a)..., mean(T_neg_b)...]
        # Each attribute stack is collapsed to its centroid before entering the sequence;
        # the Composer learns to integrate multiple attributes via self-attention.
        B, d = v_ref.shape

        # ref token
        ref_tok  = v_ref.unsqueeze(1) + self.polarity_emb(torch.zeros(B, 1, dtype=torch.long, device=v_ref.device))

        # positive attribute tokens
        pos_toks = []
        for T_a in T_pos_stacks:
            centroid = self._center(T_a).mean(dim=0, keepdim=True)   # [1, d]
            tok = centroid.expand(B, -1, -1) + \
                  self.polarity_emb(torch.ones(B, 1, dtype=torch.long, device=v_ref.device))
            pos_toks.append(tok)

        # negative attribute tokens
        neg_toks = []
        for T_b in T_neg_stacks:
            centroid = self._center(T_b).mean(dim=0, keepdim=True)
            tok = centroid.expand(B, -1, -1) + \
                  self.polarity_emb(2 * torch.ones(B, 1, dtype=torch.long, device=v_ref.device))
            neg_toks.append(tok)

        cls = self.cls_token.expand(B, -1, -1)                       # [B, 1, d]
        seq = torch.cat([cls, ref_tok] + pos_toks + neg_toks, dim=1) # [B, 2+n_attrs, d]

        out = self.encoder(seq)          # [B, seq_len, d]
        cls_out = out[:, 0]              # [B, d]  CLS output

        q = v_ref + self.head(cls_out)   # residual: start from v_ref, adjust
        return F.normalize(q, dim=1)     # [B, d] unit


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_composer(cfg: ComposerConfig = ComposerConfig(), device: str | None = None) -> QueryComposer:
    # Train the Composer with InfoNCE on synthetic train-split queries. Returns model (CPU).
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg.seed)
    print(f"Device: {dev}" + (f"  GPU: {torch.cuda.get_device_name(0)}" if dev.type == "cuda" else ""))

    train_feats = _load_train_features().to(dev)
    train_attrs = _load_train_attributes()
    prompt_bank = load_prompt_bank()
    mu_txt = compute_mu_txt(prompt_bank)

    raw_stacks = {k: v.to(dev) for k, v in build_raw_stacks(prompt_bank).items()}
    model = QueryComposer(cfg, mu_txt.to(dev)).to(dev)
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
    model: QueryComposer,
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

def evaluate_composer(model: QueryComposer, ks=(1, 5, 10), save: bool = True, tag: str = "composer") -> dict:
    # Score the Composer on the 14-query benchmark, print the table, save the CSV.
    image_features = load_image_features()
    prompt_bank = load_prompt_bank()
    raw_stacks = build_raw_stacks(prompt_bank)
    gt_list = load_eval_json(find_eval_json())

    results = {}
    for entry in gt_list:
        get_ranking = make_get_ranking(entry["query"], image_features, model, raw_stacks)
        results[entry["query"]] = evaluate_query(entry["ground_truth"], get_ranking, ks)

    print(f"\nTier-3 Composer ({tag}) — {len(gt_list)} queries\n")
    print(format_results_table(results, ks=ks))

    if save:
        save_results_csv(results, output_subdir("tier3_composer") / f"tier3_{tag}.csv", ks=ks)
    return results


if __name__ == "__main__":
    cfg = ComposerConfig()
    model = train_composer(cfg)
    weights_path = output_subdir("tier3_composer") / "tier3_composer.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"Weights saved: {weights_path}")
    evaluate_composer(model)
