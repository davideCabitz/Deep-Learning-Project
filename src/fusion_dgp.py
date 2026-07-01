"""
Fusion-DGP (Φ) — Learned Dynamic Gated Projection.

The trained successor to Tier-2d DGP (`src/tier2d_dgp.py`). DGP replaced CLAY's equal-weight
pre-SVD step with a *closed-form* reference-conditioned softmax gate over each attribute's prompt
stack; Φ replaces that hand-coded gate with a **learned cross-attention** over the same stacks,
trained with InfoNCE. Everything else — Step-0 modality-gap centering and the Step-3 geodesic
addition / orthogonal rejection on the sphere — is inherited verbatim, so Φ at init (near-identity
MLP + near-uniform attention) ≈ DGP's uniform-gate config.

Spec compliance (project_specification.md §3.2): the cross-attention IS the learned replacement
for the naïve pre-SVD stack, it dynamically re-weights the text conditions per reference, it
processes multiple textual conditions per polarity, and it defines the +/− interaction (add vs.
reject). CLIP stays frozen — Φ trains only its ~1M params and scores the frozen image DB. Built
from scratch, reusing only our own modules (§6).

Originality (honest framing, FusionModelGuide.md §5): the primitives are published (PGA tangent
maps, InfoNCE, Oldfield rejection, cross-attention); the *composition* is the contribution — a
reference-conditioned attention that structurally substitutes for SVD subspace construction,
feeding a manifold-correct rejection at the shared global μ — and Φ is the learned generalization
of our own closed-form DGP gate.

This module is the tracked, importable mirror of `notebooks/fusion_dgp.ipynb` (the graded
deliverable). It plugs into the shared eval engine through `make_get_ranking` (CONTRACT §5/§7).

Train (needs train artifacts):  python src/fusion_dgp.py
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import ATTRIBUTE_NAMES
from clip_features import load_image_features
from clip_prompts import load_prompt_bank, build_prompts_for_attribute
from eval import parse_query, evaluate_query, format_results_table, load_eval_json, find_eval_json
from results_saver import save_results_csv, output_subdir
from manifold import log_map, exp_map
from tier1_GDE import load_or_mine_directions, _load_train_features, _load_train_attributes
from tier2d_dgp import compute_mu_txt, _attr_stack


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PhiConfig:
    """Φ architecture + training hyperparameters (FusionModelGuide.md §2–4)."""
    d_model: int = 512
    n_heads: int = 4
    alpha: float = 1.0            # positive push strength in tangent space (inherited from DGP/Track-V)
    share_pos_neg: bool = True    # one cross-attn for both polarities (default) | separate (ablation §2 Step 3)
    center: bool = True           # Step-0 modality-gap centering (fixed, not learned)

    # Training
    lr: float = 1e-4
    weight_decay: float = 1e-2
    tau: float = 0.07             # InfoNCE temperature
    epochs: int = 30
    batch_queries: int = 64       # queries per step
    k_pos_max: int = 3            # up to this many positive attrs per synthetic query
    k_neg_max: int = 2            # up to this many negative attrs per synthetic query
    n_pos_targets: int = 8        # positives sampled per query
    n_hard_neg: int = 16          # hard negatives sampled per query
    hamming_max: int = 2          # GT relaxation, identical to eval protocol (spec §3.1.1)
    seed: int = 0


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class FusionPhi(nn.Module):
    # Learned gated projection — cross-attention replaces DGP's closed-form softmax gate.
    # forward(v_ref, T_pos_stacks, T_neg_stacks) → unit query q on S^{d-1}, composed at μ.
    # Geometry (Step 0 centering, Step 3 geodesic add + orthogonal rejection + Exp_μ) is fixed and
    # identical to tier2d_dgp; ONLY the per-attribute direction (Step 1–2) is learned. Near-identity
    # init (zero-init MLP tail) makes the untrained model behave like DGP — a clean ablation anchor.
    def __init__(self, cfg: PhiConfig, mu: torch.Tensor, mu_txt: torch.Tensor):
        super().__init__()
        self.cfg = cfg

        self.mlp_ref = nn.Sequential(
            nn.Linear(cfg.d_model, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, cfg.d_model),
        )
        nn.init.zeros_(self.mlp_ref[-1].weight)              # start at identity: h_ref = v_ref
        nn.init.zeros_(self.mlp_ref[-1].bias)

        self.attn_pos = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True)
        # Shared operator by default (FusionModelGuide §2 Step 3 hypothesis); separate is the ablation.
        self.attn_neg = self.attn_pos if cfg.share_pos_neg else \
            nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True)

        self.register_buffer("mu", mu)
        self.register_buffer("mu_txt", mu_txt)

    # Step 0 — fixed modality-gap centering of a prompt stack (Liang et al.; DGP Step 0).
    def _center(self, T: torch.Tensor) -> torch.Tensor:
        T = T - self.mu_txt if self.cfg.center else T
        return F.normalize(T, dim=-1)

    # Step 1–2 — learned per-attribute direction: attend from the reference onto the paraphrase
    # stack, normalize. The single-query attention is the learned analogue of DGP's softmax gate.
    def _direction(self, h_ref: torch.Tensor, T_hat: torch.Tensor, attn: nn.Module) -> torch.Tensor:
        # h_ref [B, d]; T_hat [n, d] (one attribute's centered stack, shared across the batch).
        q = h_ref.unsqueeze(1)                               # [B, 1, d]
        kv = T_hat.unsqueeze(0).expand(h_ref.shape[0], -1, -1)   # [B, n, d]
        out, _ = attn(q, kv, kv)                             # [B, 1, d]
        return F.normalize(out.squeeze(1), dim=-1)           # [B, d]

    def forward(
        self,
        v_ref: torch.Tensor,                                 # [B, d] unit reference rows
        T_pos_stacks: list[torch.Tensor],                    # list of [n_a, d] RAW prompt stacks (uncentered)
        T_neg_stacks: list[torch.Tensor],                    # list of [n_b, d] RAW prompt stacks
    ) -> torch.Tensor:
        # Compose the query for a batch of references. q = normalize(Exp_μ(q_tan)).
        h_ref = v_ref + self.mlp_ref(v_ref)                  # [B, d]
        q_tan = log_map(self.mu, v_ref)                      # [B, d] tangent at μ

        for T_a in T_pos_stacks:
            d_a = self._direction(h_ref, self._center(T_a), self.attn_pos)
            q_tan = q_tan + self.cfg.alpha * d_a

        for T_b in T_neg_stacks:
            d_b = self._direction(h_ref, self._center(T_b), self.attn_neg)
            coeff = (q_tan * d_b).sum(dim=1, keepdim=True)   # row-wise orthogonal rejection (DGP Step 3)
            q_tan = q_tan - coeff * d_b

        return F.normalize(exp_map(self.mu, q_tan), dim=1)   # [B, d] unit queries


# ---------------------------------------------------------------------------
# Prompt-stack provider — RAW (uncentered) stacks; Φ centers internally so μ_txt is a buffer.
# ---------------------------------------------------------------------------

def build_raw_stacks(prompt_bank: torch.Tensor) -> dict[str, torch.Tensor]:
    # All 40 attributes' UNPADDED raw prompt stacks, keyed by name (centering happens in Φ._center).
    return {name: _attr_stack(name, prompt_bank) for name in ATTRIBUTE_NAMES}


# ---------------------------------------------------------------------------
# Synthetic query generator (train split) — mirrors the eval GT protocol (spec §3.1.1)
# ---------------------------------------------------------------------------

class QueryGenerator:
    # Samples (v_ref, T+, T−, positives, hard negatives) from the TRAIN split.
    # Positives: train images strictly satisfying T+/T− AND Hamming ≤ hamming_max from the reference
    # on the remaining attributes (identical rule to the eval GT). Hard negatives: satisfy T+ but
    # violate ≥1 T− attribute — the images Tier-0 mistakenly retrieves, the only negatives with a
    # meaningful InfoNCE gradient. All masks are vectorized over the [N,40] attribute matrix.
    def __init__(self, train_attrs: torch.Tensor, cfg: PhiConfig):
        self.A = train_attrs                                 # [N, 40] in {0,1}
        self.cfg = cfg
        self.N, self.n_attr = train_attrs.shape
        self.g = torch.Generator().manual_seed(cfg.seed)

    def _rand(self, hi: int) -> int:
        return int(torch.randint(hi, (1,), generator=self.g).item())

    def sample_query(self) -> tuple[int, list[str], list[str], torch.Tensor, torch.Tensor] | None:
        # One synthetic query; returns None if it has too few positives to be a useful training signal.
        r = self._rand(self.N)
        has = (self.A[r] > 0.5)                              # [40] bool
        has_idx = has.nonzero(as_tuple=True)[0].tolist()
        lacks_idx = (~has).nonzero(as_tuple=True)[0].tolist()
        if not has_idx or not lacks_idx:
            return None

        k_pos = 1 + self._rand(self.cfg.k_pos_max)
        k_neg = self._rand(self.cfg.k_neg_max + 1)           # may be 0 (pure-positive query)
        pos_attrs = self._choice(has_idx, min(k_pos, len(has_idx)))
        neg_attrs = self._choice(lacks_idx, min(k_neg, len(lacks_idx)))

        # Constraint masks over the whole train split.
        sat = torch.ones(self.N, dtype=torch.bool)
        for a in pos_attrs:
            sat &= (self.A[:, a] > 0.5)
        for b in neg_attrs:
            sat &= (self.A[:, b] < 0.5)

        # Hamming distance on the REMAINING attributes (exclude the constrained ones), vs the reference.
        constrained = set(pos_attrs) | set(neg_attrs)
        free_cols = [c for c in range(self.n_attr) if c not in constrained]
        free = torch.tensor(free_cols, dtype=torch.long)
        ham = (self.A[:, free] != self.A[r, free]).sum(dim=1)    # [N]

        pos_mask = sat & (ham <= self.cfg.hamming_max)
        pos_mask[r] = False                                  # the reference is never its own target
        pos_ids = pos_mask.nonzero(as_tuple=True)[0]
        if pos_ids.numel() < 1:
            return None

        # Hard negatives: satisfy all positives but violate ≥1 negative.
        sat_pos = torch.ones(self.N, dtype=torch.bool)
        for a in pos_attrs:
            sat_pos &= (self.A[:, a] > 0.5)
        viol_neg = torch.zeros(self.N, dtype=torch.bool)
        for b in neg_attrs:
            viol_neg |= (self.A[:, b] > 0.5)
        hard_mask = sat_pos & viol_neg
        hard_mask[r] = False
        hard_ids = hard_mask.nonzero(as_tuple=True)[0]
        if neg_attrs and hard_ids.numel() < 1:
            return None                                      # a negation query with no hard negs is useless

        pos_sel = self._take(pos_ids, self.cfg.n_pos_targets)
        hard_sel = self._take(hard_ids, self.cfg.n_hard_neg) if hard_ids.numel() else hard_ids
        pos_names = [ATTRIBUTE_NAMES[a] for a in pos_attrs]
        neg_names = [ATTRIBUTE_NAMES[b] for b in neg_attrs]
        return r, pos_names, neg_names, pos_sel, hard_sel

    def _choice(self, pool: list[int], k: int) -> list[int]:
        perm = torch.randperm(len(pool), generator=self.g)[:k]
        return [pool[i] for i in perm.tolist()]

    def _take(self, ids: torch.Tensor, k: int) -> torch.Tensor:
        if ids.numel() <= k:
            return ids
        perm = torch.randperm(ids.numel(), generator=self.g)[:k]
        return ids[perm]


# ---------------------------------------------------------------------------
# InfoNCE loss
# ---------------------------------------------------------------------------

def info_nce(q: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor, tau: float) -> torch.Tensor:
    # L = −log[ Σ_p exp(q·p/τ) / ( Σ_p exp(q·p/τ) + Σ_n exp(q·n/τ) ) ]  (van den Oord; CLIP lineage).
    # Multi-positive form: the numerator sums over all valid targets, the denominator adds the hard
    # negatives. q [d]; pos [P, d]; neg [M, d] — all unit, so dot product is cosine.
    pos_logits = (pos @ q) / tau                             # [P]
    parts = [pos_logits]
    if neg.numel():
        parts.append((neg @ q) / tau)                        # [M]
    all_logits = torch.cat(parts)                            # [P(+M)]
    log_denom = torch.logsumexp(all_logits, dim=0)
    log_num = torch.logsumexp(pos_logits, dim=0)
    return log_denom - log_num                               # = −log(num/denom)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_phi(cfg: PhiConfig = PhiConfig(), device: str | None = None, verbose: bool = True) -> FusionPhi:
    # Train Φ on synthetic train-split queries with InfoNCE. Returns the trained model (CPU).
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg.seed)

    train_feats = _load_train_features().to(dev)             # [N, d] unit
    train_attrs = _load_train_attributes()                   # [N, 40] (kept on CPU for masking)
    prompt_bank = load_prompt_bank()
    mu, _ = load_or_mine_directions()
    mu_txt = compute_mu_txt(prompt_bank)

    raw_stacks = {k: v.to(dev) for k, v in build_raw_stacks(prompt_bank).items()}
    model = FusionPhi(cfg, mu.to(dev), mu_txt.to(dev)).to(dev)
    gen = QueryGenerator(train_attrs, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    steps_per_epoch = max(1, gen.N // (cfg.batch_queries * 50))   # a light, bounded epoch
    for epoch in range(cfg.epochs):
        model.train()
        running, seen = 0.0, 0
        for _ in range(steps_per_epoch):
            opt.zero_grad()
            batch_loss, n = torch.zeros((), device=dev), 0
            for _ in range(cfg.batch_queries):
                sample = gen.sample_query()
                if sample is None:
                    continue
                r, pos_names, neg_names, pos_ids, hard_ids = sample
                v_ref = train_feats[r].unsqueeze(0)          # [1, d]
                q = model(
                    v_ref,
                    [raw_stacks[n] for n in pos_names],
                    [raw_stacks[n] for n in neg_names],
                ).squeeze(0)                                 # [d]
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
        sched.step()
        if verbose:
            print(f"  epoch {epoch+1:02d}/{cfg.epochs}  InfoNCE={running/max(seen,1):.4f}")

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
    # Curry one query string into get_ranking(src_idx). Per-source: compose q via Φ, score the
    # frozen DB by cosine, exclude the source. Mirrors tier2d_dgp.make_get_ranking exactly.
    T_pos, T_neg = parse_query(query_str)
    model.eval()

    @torch.no_grad()
    def get_ranking(src_idx: int) -> list[int]:
        q = model(
            image_features[src_idx].unsqueeze(0),
            [raw_stacks[n] for n in T_pos],
            [raw_stacks[n] for n in T_neg],
        ).squeeze(0)
        scores = image_features @ q
        scores[src_idx] = float("-inf")
        return torch.argsort(scores, descending=True).tolist()

    return get_ranking


# ---------------------------------------------------------------------------
# Evaluation entry point
# ---------------------------------------------------------------------------

def evaluate_phi(model: FusionPhi, ks=(1, 5, 10), save: bool = True, tag: str = "phi") -> dict:
    # Score a trained Φ on the 14-query benchmark (test split), print the table, save the CSV.
    image_features = load_image_features()
    prompt_bank = load_prompt_bank()
    raw_stacks = build_raw_stacks(prompt_bank)
    gt_list = load_eval_json(find_eval_json())

    results = {}
    for entry in gt_list:
        get_ranking = make_get_ranking(entry["query"], image_features, model, raw_stacks)
        results[entry["query"]] = evaluate_query(entry["ground_truth"], get_ranking, ks)

    print(f"\nFusion-DGP Phi ({tag}) - {len(gt_list)} queries\n")
    print(format_results_table(results, ks=ks))

    if save:
        save_results_csv(results, output_subdir("fusion_dgp") / f"fusion_dgp_{tag}.csv", ks=ks)
    return results


if __name__ == "__main__":
    phi = train_phi()
    evaluate_phi(phi)
