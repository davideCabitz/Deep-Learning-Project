#!/usr/bin/env python
"""
Tier-4 Hybrid — Discriminative-negation compositional retrieval (CLIP frozen).

Every prior tier composes ONE fused query vector q and ranks by cos(q, x). For a
negation query q_tan = Log_μ(v_ref) − rejection, the Log_μ(v_ref) term dominates the
norm and pins q at the reference, so orthogonal rejection only carves out a thin
sliver — the query keeps retrieving identity-preserving look-alikes that still HAVE
the negated attribute. This is the identity-anchor-vs-negation tension that flat-lines
`-Male,-Mustache` to 0.000 across every geometry-only tier (2a/2b/2c/3-dgp/3c/…).

Tier-4 stops treating negation as geometry. It splits scoring into two pathways and
fuses them at score time (the spec's literal "dynamic AND hybrid conditioning"):

    score(x | v_ref) = cos( Φ(v_ref, T⁺), x )  −  β · Σ_{b∈T⁻} σ( probe_b(x) )
                       └──────── positive / identity ────────┘   └── discriminative negation ──┘

  • Positive/identity pathway — a lightweight learned cross-attention fusion Φ composes
    v_ref with the positive text conditions on the sphere (the proven tier3 recipe:
    reference-conditioned gate over prompt paraphrases, geodesic add, ListNet loss).
  • Negative pathway — 40 learned attribute probes (one shared MLP head over the FROZEN
    CLIP features, trained on CelebA TRAIN labels). "−Male" becomes "down-rank every image
    the Male probe fires on" — a high-precision discriminative operation the geometry could
    never do, because it acts on the attribute axis directly instead of nudging an anchored
    query. β (learned) sets how hard the penalty bites.

Why hybrid beats every predecessor's trade-off: it KEEPS ListNet (tier3_contrastive's win)
and the cross-attention gate (tier3_dgp's win) and identity anchoring for + queries (avoids
the tier3_composer collapse), while SOLVING double-negation (the composer's only win) WITHOUT
sacrificing identity — because negatives never perturb q⁺; they act as a separate learned
margin. For a pure-positive query the penalty is 0 and Tier-4 reduces EXACTLY to
tier3_contrastive, so it strictly generalises the current best.

Spec compliance (project_specification.md): CLIP frozen (§3.3); frozen offline DB (§3.2);
probes train on TRAIN-split attribute labels only — the eval GT JSON is never inspected, no
leakage (§3.1.1); learned fusion + gating replaces the naïve pre-SVD stack (§3.2). Built from
our own modules; primitives (linear probes/CAVs, cross-attention, ListNet) are cited, the
two-pathway composition is the contribution.

Two-stage training, headless / VM-ready (auto paths, timestamped progress, ETA, headless plot):
    Stage 1 — BCE-train the attribute probes on frozen train features.
    Stage 2 — freeze probes, train Φ + β for 30 epochs with a composite-score ListNet loss.

Run (from repo root, needs train artifacts):  python src/tier4_hybrid.py
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass

try:                                        # optional: only used for the training-curve PNG
    import matplotlib
    matplotlib.use("Agg")                   # headless: render to file, never open a window
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ModuleNotFoundError:
    HAVE_MPL = False
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import ATTRIBUTE_NAMES, ATTR_TO_IDX
from clip_features import load_image_features, FEATURE_DIM
from clip_prompts import load_prompt_bank
from eval import parse_query, evaluate_query, format_results_table, load_eval_json, find_eval_json
from results_saver import save_results_csv, output_subdir
from manifold import log_map, exp_map
from tier1_GDE import load_or_mine_directions, _load_train_features, _load_train_attributes
from tier2d_dgp import compute_mu_txt
from tier3_dgp import PhiConfig, QueryGenerator, build_raw_stacks


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log(msg: str) -> None:
    # Timestamped, always-flushed print so progress shows up live over SSH/tee.
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Configuration — the full ablation surface, bundled so call sites stay short.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Tier4Config:
    """Tier-4 hybrid hyperparameters. Defaults are the headline VM run."""
    # Architecture (kept small: total trainable params < 1M — see main()'s assert)
    d_model: int = 512
    d_gate: int = 128            # low-rank cross-attention key/query dim (the SVD replacement)
    ref_hidden: int = 256        # reference-encoder MLP width
    probe_hidden: int = 256      # attribute-probe MLP width
    alpha: float = 1.0           # positive push strength in tangent space (Track-V's α)

    # Stage 1 — attribute probes (BCE on frozen train features)
    probe_epochs: int = 20
    probe_lr: float = 1e-3
    probe_batch: int = 1024
    val_frac: float = 0.1        # held-out train slice for probe accuracy readout only

    # Stage 2 — fusion Φ + β (ListNet on composite score)
    epochs: int = 30             # main training length (per request)
    lr: float = 1e-4
    weight_decay: float = 1e-2
    tau: float = 0.07            # ListNet softmax temperature
    batch_queries: int = 64
    k_pos_max: int = 3
    k_neg_max: int = 2
    n_pos_targets: int = 8       # positives per synthetic query
    n_hard_neg: int = 32         # attribute-based hard negatives (satisfy T⁺, violate a T⁻)
    n_rand_neg: int = 16         # random train negatives — gives pure-positive queries contrast
    hamming_max: int = 2         # GT relaxation, identical to eval protocol (spec §3.1.1)
    seed: int = 0

    def phi_gen_config(self) -> PhiConfig:
        # Borrow PhiConfig ONLY to drive QueryGenerator (identical sampling protocol as tier3).
        return PhiConfig(
            d_model=self.d_model, alpha=self.alpha,
            lr=self.lr, weight_decay=self.weight_decay, tau=self.tau,
            epochs=self.epochs, batch_queries=self.batch_queries,
            k_pos_max=self.k_pos_max, k_neg_max=self.k_neg_max,
            n_pos_targets=self.n_pos_targets, n_hard_neg=self.n_hard_neg,
            hamming_max=self.hamming_max, seed=self.seed,
        )


# ---------------------------------------------------------------------------
# Negative pathway — learned attribute probes over the FROZEN CLIP features.
# ---------------------------------------------------------------------------
class AttributeProbes(nn.Module):
    # Discriminative attribute detectors (concept probes; CAV lineage, Kim et al. 2018).
    # p(x) = MLP(x) → 40 logits, one per CelebA attribute; σ(p_b(x)) ≈ P(x has attribute b).
    # Trained by BCE on TRAIN labels; at retrieval, σ(p_b) is the negation penalty for "−b".
    # (This is the piece geometry lacked: negation as a calibrated classifier margin, not a
    # geodesic rejection that barely reranks an identity-anchored query.)
    def __init__(self, d_model: int, hidden: int, n_attr: int = 40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_attr),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B, d] unit CLIP features → [B, 40] raw logits.
        return self.net(x)


def train_probes(
    train_feats: torch.Tensor, train_attrs: torch.Tensor, cfg: Tier4Config,
) -> tuple[AttributeProbes, list[float]]:
    # Stage 1 — BCE-fit the probes on frozen features; hold out a train slice for an honest
    # accuracy readout (validates "CLIP attributes are near-linearly decodable"). No GT touched.
    torch.manual_seed(cfg.seed)
    N = train_feats.shape[0]
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(cfg.seed))
    n_val = int(N * cfg.val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    probes = AttributeProbes(cfg.d_model, cfg.probe_hidden).to(DEVICE)
    opt = torch.optim.Adam(probes.parameters(), lr=cfg.probe_lr)
    bce = nn.BCEWithLogitsLoss()

    Xtr, Ytr = train_feats[tr_idx], train_attrs[tr_idx].float()
    Xva, Yva = train_feats[val_idx].to(DEVICE), train_attrs[val_idx].float().to(DEVICE)

    history: list[float] = []
    for epoch in range(cfg.probe_epochs):
        probes.train()
        order = torch.randperm(Xtr.shape[0])
        running, seen = 0.0, 0
        for start in range(0, Xtr.shape[0], cfg.probe_batch):
            sel = order[start:start + cfg.probe_batch]
            xb, yb = Xtr[sel].to(DEVICE), Ytr[sel].to(DEVICE)
            opt.zero_grad()
            loss = bce(probes(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.shape[0]
            seen += xb.shape[0]
        history.append(running / max(seen, 1))

        probes.eval()
        with torch.no_grad():
            acc = ((probes(Xva) > 0).float() == Yva).float().mean().item()
        log(f"  [probes] epoch {epoch + 1:02d}/{cfg.probe_epochs}  BCE={history[-1]:.4f}  val_acc={acc:.4f}")

    return probes, history


# ---------------------------------------------------------------------------
# Positive pathway — low-rank cross-attention gate + geodesic fusion Φ.
# ---------------------------------------------------------------------------
class LowRankGate(nn.Module):
    # Reference-conditioned direction over a prompt stack (the learned SVD replacement, kept
    # low-rank so the whole model fits < 1M params). a = softmax(W_q h_ref · (W_k T̂)ᵀ / √d_k);
    # d = normalize(a @ T̂). Learned Q/K choose WHICH paraphrases each reference attends to;
    # value is the centered prompt itself (no value matrix — saves params, keeps direction honest).
    def __init__(self, d_model: int, d_gate: int):
        super().__init__()
        self.w_q = nn.Linear(d_model, d_gate, bias=False)
        self.w_k = nn.Linear(d_model, d_gate, bias=False)
        self.scale = 1.0 / math.sqrt(d_gate)

    def forward(self, h_ref: torch.Tensor, T_hat: torch.Tensor) -> torch.Tensor:
        # h_ref [B, d]; T_hat [n, d] (one attribute's centered stack) → direction [B, d].
        q = self.w_q(h_ref)                                  # [B, d_gate]
        k = self.w_k(T_hat)                                  # [n, d_gate]
        a = F.softmax((q @ k.T) * self.scale, dim=1)         # [B, n] per-reference weights
        return F.normalize(a @ T_hat, dim=1)                 # [B, d] gated condition direction


class FusionPhiPos(nn.Module):
    # Positive/identity fusion — composes v_ref with the POSITIVE conditions only (negatives are
    # handled discriminatively by the probes). q⁺ = normalize(Exp_μ( Log_μ(v_ref) + Σ_a α·d_a )).
    # Near-identity init (zero-init MLP tail) ⇒ at step 0 q⁺ ≈ centered-gate DGP — a clean anchor.
    def __init__(self, cfg: Tier4Config, mu: torch.Tensor, mu_txt: torch.Tensor):
        super().__init__()
        self.alpha = cfg.alpha
        self.mlp_ref = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ref_hidden),
            nn.LayerNorm(cfg.ref_hidden),
            nn.GELU(),
            nn.Linear(cfg.ref_hidden, cfg.d_model),
        )
        nn.init.zeros_(self.mlp_ref[-1].weight)              # start at identity: h_ref = v_ref
        nn.init.zeros_(self.mlp_ref[-1].bias)
        self.gate = LowRankGate(cfg.d_model, cfg.d_gate)
        self.register_buffer("mu", mu)
        self.register_buffer("mu_txt", mu_txt)

    def _center(self, T: torch.Tensor) -> torch.Tensor:
        # Fixed modality-gap centering of a raw prompt stack (Liang et al.; DGP Step 0).
        return F.normalize(T - self.mu_txt, dim=-1)

    def forward(self, v_ref: torch.Tensor, T_pos_stacks: list[torch.Tensor]) -> torch.Tensor:
        # v_ref [B, d] unit; T_pos_stacks: list of [n_a, d] RAW stacks → q⁺ [B, d] unit.
        h_ref = v_ref + self.mlp_ref(v_ref)
        q_tan = log_map(self.mu, v_ref)
        for T_a in T_pos_stacks:
            q_tan = q_tan + self.alpha * self.gate(h_ref, self._center(T_a))
        return F.normalize(exp_map(self.mu, q_tan), dim=1)


class HybridRetriever(nn.Module):
    # Wraps the positive fusion Φ and the learned negation strength β (β = softplus(β_raw) > 0,
    # init ≈ 1.0). Probes are held SEPARATELY and frozen in stage 2 — β still gets gradient through
    # the penalty term, so Φ and β co-adapt to the fixed discriminative axes.
    def __init__(self, cfg: Tier4Config, mu: torch.Tensor, mu_txt: torch.Tensor):
        super().__init__()
        self.phi = FusionPhiPos(cfg, mu, mu_txt)
        self.beta_raw = nn.Parameter(torch.tensor(0.5413))   # softplus(0.5413) ≈ 1.0

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self.beta_raw)


# ---------------------------------------------------------------------------
# Composite-score ListNet loss  (Cao et al. 2007, made metric-consistent with R@K/P@K)
# ---------------------------------------------------------------------------
def listnet_loss(scores: torch.Tensor, n_pos: int, tau: float) -> torch.Tensor:
    # L = − Σ_i y_i · log softmax_i(scores/τ); y = 1/|pos| on positives, 0 on negatives.
    # `scores` are the COMPOSITE cos − β·penalty, so gradient flows to both Φ (via cos) and β
    # (via penalty) — the ranking objective directly supervises how hard negation should bite.
    labels = torch.zeros(scores.shape[0], device=scores.device)
    labels[:n_pos] = 1.0 / max(n_pos, 1)
    return -(labels * F.log_softmax(scores / tau, dim=0)).sum()


def _neg_penalty(
    probes: AttributeProbes, feats: torch.Tensor, neg_cols: list[int],
) -> torch.Tensor:
    # Σ_{b∈T⁻} σ(probe_b(x)) for each row of `feats` [C, d] → [C]. Probes are frozen here, so this
    # is constant w.r.t. their params; β (applied by the caller) is the only trainable multiplier.
    if not neg_cols:
        return torch.zeros(feats.shape[0], device=feats.device)
    with torch.no_grad():
        p = torch.sigmoid(probes(feats))[:, neg_cols].sum(dim=1)
    return p


# ---------------------------------------------------------------------------
# Stage 2 training — freeze probes, train Φ + β with composite ListNet.
# ---------------------------------------------------------------------------
def train_hybrid(
    model: HybridRetriever,
    probes: AttributeProbes,
    train_feats: torch.Tensor,
    train_attrs: torch.Tensor,
    raw_stacks: dict[str, torch.Tensor],
    cfg: Tier4Config,
) -> list[float]:
    # ListNet over candidates [positives ∪ hard-negatives ∪ random-negatives], scored by the
    # composite cos − β·penalty. Hard negatives (satisfy T⁺, violate a T⁻) are exactly the images
    # the penalty must push down — the only place β gets a meaningful gradient.
    for p in probes.parameters():                            # freeze the discriminative axes
        p.requires_grad_(False)
    probes.eval()

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
                neg_cols = [ATTR_TO_IDX[b] for b in neg_names]

                v_ref = train_feats[r].unsqueeze(0)
                q = model.phi(v_ref, [raw_stacks[nm] for nm in pos_names]).squeeze(0)   # [d]

                rand_ids = torch.randint(N, (cfg.n_rand_neg,), generator=rng)
                cand = torch.cat([pos_ids, hard_ids, rand_ids]).to(DEVICE)
                cand_feats = train_feats[cand]                                          # [C, d]

                cos = cand_feats @ q                                                    # [C]
                penalty = model.beta * _neg_penalty(probes, cand_feats, neg_cols)       # [C]
                composite = cos - penalty
                batch_loss = batch_loss + listnet_loss(composite, pos_ids.numel(), cfg.tau)
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
        log(f"  [phi] epoch {epoch + 1:02d}/{cfg.epochs}  ListNet={history[-1]:.4f}  "
            f"β={model.beta.item():.3f}  eta={eta / 60:.1f}min")

    return history


# ---------------------------------------------------------------------------
# Retrieval seam — CONTRACT §5/§7 (composite score, penalty precomputed per query)
# ---------------------------------------------------------------------------
def make_get_ranking(
    query_str: str,
    image_features: torch.Tensor,
    model: HybridRetriever,
    probes: AttributeProbes,
    raw_stacks: dict[str, torch.Tensor],
) -> callable:
    # Curry one query into get_ranking(src_idx). The negation penalty is source-INDEPENDENT, so it
    # is computed ONCE over the whole DB per query; only the cheap per-source q⁺ + cosine runs inside.
    T_pos, T_neg = parse_query(query_str)
    neg_cols = [ATTR_TO_IDX[b] for b in T_neg]
    model.eval(); probes.eval()

    with torch.no_grad():
        penalty = model.beta * _neg_penalty(probes, image_features, neg_cols)      # [N]

    @torch.no_grad()
    def get_ranking(src_idx: int) -> list[int]:
        q = model.phi(image_features[src_idx].unsqueeze(0), [raw_stacks[nm] for nm in T_pos]).squeeze(0)
        scores = image_features @ q - penalty
        scores[src_idx] = float("-inf")                      # source never ranks itself (§5)
        return torch.argsort(scores, descending=True).tolist()

    return get_ranking


def evaluate_hybrid(
    model: HybridRetriever,
    probes: AttributeProbes,
    raw_stacks: dict[str, torch.Tensor],
    ks=(1, 5, 10),
    save: bool = True,
    tag: str = "hybrid",
) -> dict:
    # Score Tier-4 on the 14-query benchmark (test split), print the table, save the CSV.
    image_features = load_image_features().to(DEVICE)
    gt_list = load_eval_json(find_eval_json())

    results = {}
    for entry in gt_list:
        get_ranking = make_get_ranking(entry["query"], image_features, model, probes, raw_stacks)
        results[entry["query"]] = evaluate_query(entry["ground_truth"], get_ranking, ks)

    print(f"\nTier-4 Hybrid ({tag}) — {len(gt_list)} queries\n")
    print(format_results_table(results, ks=ks))
    if save:
        save_results_csv(results, output_subdir("tier4_hybrid") / f"tier4_{tag}.csv", ks=ks)
    return results


# ---------------------------------------------------------------------------
# Main — two-stage pipeline, VM-ready.
# ---------------------------------------------------------------------------
def main(cfg: Tier4Config = Tier4Config()) -> None:
    torch.manual_seed(cfg.seed)
    log(f"device={DEVICE}  probe_epochs={cfg.probe_epochs}  phi_epochs={cfg.epochs}  batch_q={cfg.batch_queries}")
    if DEVICE == "cpu":
        log("WARNING: CUDA not available — training on CPU will be slow.")

    # ---- Load frozen artifacts (train split for learning; test DB only for eval) ----
    log("loading frozen artifacts (train features/labels, μ, μ_txt, prompt bank) …")
    train_feats = _load_train_features().to(DEVICE)
    train_attrs = _load_train_attributes()                   # [N, 40] {0,1}, kept on CPU for masking
    prompt_bank = load_prompt_bank()
    mu, _ = load_or_mine_directions()
    mu_txt = compute_mu_txt(prompt_bank)
    raw_stacks = {k: v.to(DEVICE) for k, v in build_raw_stacks(prompt_bank).items()}

    # Tripwires — fail loud on any shape / dim drift before a long run starts.
    assert train_feats.shape[1] == FEATURE_DIM, train_feats.shape
    assert train_attrs.shape[0] == train_feats.shape[0] and train_attrs.shape[1] == 40, train_attrs.shape
    assert len(raw_stacks) == 40, len(raw_stacks)
    log(f"artifacts OK: train_feats={tuple(train_feats.shape)}  train_attrs={tuple(train_attrs.shape)}")

    # ---- Stage 1 — attribute probes (discriminative negation axes) ----
    log("STAGE 1 — training attribute probes (BCE on frozen train features) …")
    probes, probe_hist = train_probes(train_feats, train_attrs, cfg)

    # ---- Stage 2 — fusion Φ + β (composite ListNet) ----
    model = HybridRetriever(cfg, mu.to(DEVICE), mu_txt.to(DEVICE)).to(DEVICE)

    n_phi = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_probe = sum(p.numel() for p in probes.parameters())
    n_total = n_phi + n_probe
    log(f"params: Φ+β={n_phi/1e6:.3f}M  probes={n_probe/1e6:.3f}M  total={n_total/1e6:.3f}M")
    assert n_total < 1_000_000, f"parameter budget exceeded: {n_total} ≥ 1,000,000"

    log("STAGE 2 — training fusion Φ + β (ListNet on composite score, probes frozen) …")
    phi_hist = train_hybrid(model, probes, train_feats, train_attrs, raw_stacks, cfg)

    # ---- Persist weights ----
    out_dir = output_subdir("tier4_hybrid")
    torch.save(model.state_dict(), out_dir / "tier4_hybrid_phi.pt")
    torch.save(probes.state_dict(), out_dir / "tier4_hybrid_probes.pt")
    log(f"[OK] saved weights → {out_dir}")

    # ---- Training curves (headless; skipped if matplotlib is unavailable) ----
    if HAVE_MPL:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(probe_hist); ax[0].set_title("Stage 1 — probe BCE")
        ax[0].set_xlabel("epoch"); ax[0].set_ylabel("BCE")
        ax[1].plot(phi_hist); ax[1].set_title("Stage 2 — Φ ListNet")
        ax[1].set_xlabel("epoch"); ax[1].set_ylabel("ListNet")
        fig.tight_layout()
        fig.savefig(out_dir / "tier4_training_curves.png", dpi=120)
        log(f"[OK] saved training curves → {out_dir / 'tier4_training_curves.png'}")
    else:
        log("matplotlib not installed — skipping training-curve PNG (install it to enable plots)")

    # ---- Evaluate on the 14-query benchmark ----
    log("evaluating on the eval JSON …")
    results = evaluate_hybrid(model, probes, raw_stacks)

    def _mean(res, k):
        return sum(r[f"recall@{k}"] for r in res.values()) / len(res)
    log("---- Tier-4 mean recall ----")
    log(f"R@1={_mean(results, 1):.4f}  R@5={_mean(results, 5):.4f}  R@10={_mean(results, 10):.4f}  "
        f"(learned β={model.beta.item():.3f})")


if __name__ == "__main__":
    main()
