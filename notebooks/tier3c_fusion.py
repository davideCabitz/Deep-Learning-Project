#!/usr/bin/env python
"""
Tier-3c — Learned cross-attention fusion Φ over image-mined visual prototypes.

Standalone training script (headless / SSH / VM). A faithful port of
notebooks/Tier2d_Fusion_Colab.ipynb with the Colab-only scaffolding removed
(no Drive mount, no pip install, no git clone) and real progress reporting:
per-step throughput, per-epoch train/val, and a wall-clock ETA.

Paths are auto-detected from this file's location, so nothing needs editing:
    PROJECT_DIR = <repo root>                 (parent of notebooks/)
    CKPT_DIR    = <repo root>/../tier3c_ckpts

Run from anywhere:
    python -u notebooks/tier3c_fusion.py

The `-u` is optional (this script flushes every print), but harmless.
"""

import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # headless: render to file, never open a window
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Paths — derived from this file so the script is portable across machines.
# --------------------------------------------------------------------------- #
PROJECT_DIR = Path(__file__).resolve().parent.parent          # repo root (parent of notebooks/)
SRC_DIR     = PROJECT_DIR / "src"
CKPT_DIR    = PROJECT_DIR.parent / "tier3c_ckpts"             # sibling of the repo

if not (SRC_DIR / "data_loader.py").is_file():
    raise FileNotFoundError(f"src/ not found under {PROJECT_DIR}. Is this file inside <repo>/notebooks/?")
sys.path.insert(0, str(SRC_DIR))

# --------------------------------------------------------------------------- #
# Config — single source of truth (kept aligned with Tier-2c defaults).
# --------------------------------------------------------------------------- #
SEED = 0

# Geometry
K_NEG        = 10                    # negative-subspace dim per attribute
ALPHA        = 1.0                   # positive push strength (Tier-2c's α)
MAX_TAN_NORM = math.pi / 2 - 1e-3    # GDE App. C.1 cone bound — safe Exp radius

# Φ architecture
N_HEADS        = 4
REJECTION_MODE = "reweight_kdim"     # "reweight_kdim" | "fused_rank1" (ablation)
SHARE_ATTN     = True                # shared pos/neg cross-attention (False = ablation)

# Training
TAU             = 0.07
LR              = 1e-4
WEIGHT_DECAY    = 1e-2
EPOCHS          = 30
STEPS_PER_EPOCH = 200
BATCH           = 64
N_POS           = 8                  # positives sampled per query
N_NEG           = 16                 # hard negatives sampled per query
K_POS_RANGE     = (1, 3)             # number of + attributes per synthetic query
K_NEG_RANGE     = (0, 2)             # number of - attributes per synthetic query
NORM_PENALTY    = 0.1                # weight on relu(‖q_tan‖ - π/2)^2
VAL_QUERIES     = 256                # held-out synthetic queries for checkpoint selection

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log(msg):
    # Timestamped, always-flushed print so progress shows up live over SSH/tee.
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Frozen-helper imports — all geometry/data/eval reused from src/.
# --------------------------------------------------------------------------- #
from data_loader import ATTRIBUTE_NAMES, ATTR_TO_IDX, _get_artifacts_dir
from clip_features import load_image_features, FEATURE_DIM
from clip_prompts import load_prompt_bank, build_prompts_for_attribute
from eval import parse_query, evaluate_all, format_results_table, load_eval_json, find_eval_json
from results_saver import save_results_csv, output_subdir
from manifold import exp_map                       # single owner of the Exp cos/sin formula
from tier1_GDE import load_or_mine_directions, _load_train_features, _load_train_attributes
from tier2c import (
    load_or_mine_neg_subspaces, _compose_query_svunion, SVunionConfig, evaluate_svunion,
)


# --------------------------------------------------------------------------- #
# Numerical-stability wrappers — gradient-safe Log/Exp at the cut locus.
# --------------------------------------------------------------------------- #
def safe_log_map(mu, x, dot_clamp=1.0 - 1e-4, eps=1e-6):
    # Log_μ(x) with the dot clamped to ±(1-1e-4) BEFORE arccos so the derivative
    # -1/√(1-x²) stays bounded near the cut locus. Mirrors manifold.log_map; the
    # tighter pre-arccos clamp is the only delta.
    dot = torch.clamp((x * mu).sum(), -dot_clamp, dot_clamp)
    theta = torch.arccos(dot)
    tangent = x - dot * mu
    norm = tangent.norm()
    return tangent * (theta / norm) if norm > eps else torch.zeros_like(x)


def safe_exp_map(mu, v, max_norm=MAX_TAN_NORM, eps=1e-6):
    # Exp_μ(v) with ‖v‖ clamped to the GDE cone bound (<π/2) before lifting, so a
    # drifting tangent never wraps around the sphere. Direction preserved; only the
    # magnitude saturates when clamped. Reuses manifold.exp_map for cos/sin.
    norm = v.norm()
    factor = torch.clamp(norm, max=max_norm) / norm.clamp_min(eps)
    return exp_map(mu, (v * factor).unsqueeze(0)).squeeze(0)


# --------------------------------------------------------------------------- #
# Φ — learned cross-attention fusion over image-mined prototypes.
# --------------------------------------------------------------------------- #
class FusionPhi(nn.Module):
    """
    Tier-3c — learned cross-attention fusion over image-mined visual prototypes (text-conditioned).
    q = normalize(Exp_μ( Π⊥( Log_μ(v_ref) + Σ_a α·c_a ) )),  c_a = to_tangent(v_a + g_a(h_ref, text)),
    Π⊥ = orthogonal-complement of the union of learned-reweighted Tier-2c subspaces Q̃_b.
    Cross-attention over per-attribute TEXT stacks replaces CLAY's fixed per-condition SVD; the
    manifold composition (geodesic add + rejection at the shared global μ) is the FIXED inductive bias
    from Tier-2c — every additive/rejection object is anchored on an image-mined prototype, so there is
    no modality gap. All residual heads are zero-initialised ⇒ Φ ≡ Tier-2c at init.
    """

    def __init__(self, mu, mu_txt, directions, neg_subspaces, d_model=512, n_heads=4, k_neg=10,
                 alpha=1.0, rejection_mode="reweight_kdim", share_attn=True, max_tan_norm=MAX_TAN_NORM):
        super().__init__()
        if rejection_mode not in ("reweight_kdim", "fused_rank1"):
            raise ValueError(f"rejection_mode must be 'reweight_kdim'|'fused_rank1', got {rejection_mode!r}")
        self.d, self.k_neg, self.alpha = d_model, k_neg, alpha
        self.rejection_mode, self.max_tan_norm = rejection_mode, max_tan_norm

        # Frozen geometry — buffers move with .to() but never train.
        self.register_buffer("mu", mu)
        self.register_buffer("mu_txt", mu_txt)
        self.register_buffer("directions", directions)          # [40, d]
        self.register_buffer("neg_subspaces", neg_subspaces)    # [40, d, K_CACHE]

        # Reference encoder — residual, near-identity at init.
        self.mlp_ref = nn.Sequential(
            nn.Linear(d_model, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, d_model),
        )
        nn.init.zeros_(self.mlp_ref[-1].weight); nn.init.zeros_(self.mlp_ref[-1].bias)

        # One shared cross-attention by default (a separate neg module is the ablation).
        self.attn_pos = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn_neg = self.attn_pos if share_attn else nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        # Positive head [h_ref, ctx_a, v_a] → tangent residual g_a; zero-init → c_a = v_a.
        self.proj_head_pos = nn.Sequential(
            nn.Linear(3 * d_model, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, d_model),
        )
        nn.init.zeros_(self.proj_head_pos[-1].weight); nn.init.zeros_(self.proj_head_pos[-1].bias)

        # Negative head [h_ref, ctx_b] → k column gates Δ; zero-init → w_b = 1 (Q̃_b = Q_b).
        self.select_head = nn.Linear(2 * d_model, k_neg)
        nn.init.zeros_(self.select_head.weight); nn.init.zeros_(self.select_head.bias)

    def _center(self, T):
        # Step 0 — fixed modality-gap centering: drop the text-cone mean, renormalise.
        return F.normalize(T - self.mu_txt, dim=-1)

    def _attend(self, attn, h_ref, T_hat):
        # Cross-attention: query = reference (image cone), key=value = centered text stack → context [d].
        out, _ = attn(h_ref.view(1, 1, self.d), T_hat.unsqueeze(0), T_hat.unsqueeze(0))
        return out.view(self.d)

    def _to_tangent(self, v):
        # Remove the μ-component so the contribution is a valid tangent vector at the shared global μ.
        return v - (v * self.mu).sum() * self.mu

    def _positive_contribution(self, h_ref, a_idx, T_a):
        # c_a = to_tangent(v_a + g_a([h_ref, ctx_a, v_a])). Anchored on image-mined v_a; text only
        # drives ctx_a and the small learned residual g_a (zero at init).
        v_a = self.directions[a_idx]
        ctx_a = self._attend(self.attn_pos, h_ref, self._center(T_a))
        g_a = self.proj_head_pos(torch.cat([h_ref, ctx_a, v_a]))
        return self._to_tangent(v_a + g_a)

    def _neg_reweighted_subspace(self, h_ref, b_idx, T_b):
        # Learned, text-conditioned reweighting of Tier-2c's image-mined subspace columns (all ⊥ μ).
        # reweight_kdim: Q̃_b = Q_b[:,:k]·(1+Δ).  fused_rank1: collapse k columns into ONE learned
        # direction via softmax(Δ). Both keep columns ⊥ μ.
        Q_b = self.neg_subspaces[b_idx][:, : self.k_neg]
        ctx_b = self._attend(self.attn_neg, h_ref, self._center(T_b))
        delta = self.select_head(torch.cat([h_ref, ctx_b]))
        if self.rejection_mode == "fused_rank1":
            r = Q_b @ torch.softmax(delta, dim=0)
            return F.normalize(r, dim=0).unsqueeze(1)
        return Q_b * (1.0 + delta).unsqueeze(0)

    def forward(self, v_ref, pos_idx, neg_idx, pos_stacks, neg_stacks, return_qtan_norm=False):
        h_ref = v_ref + self.mlp_ref(v_ref)                          # = v_ref at init

        h_pos = v_ref.new_zeros(self.d)
        for a_idx, T_a in zip(pos_idx, pos_stacks):
            h_pos = h_pos + self.alpha * self._positive_contribution(h_ref, a_idx, T_a)

        q_tan = safe_log_map(self.mu, v_ref) + h_pos                 # image-cone tangent at μ

        if neg_idx:
            cols = [self._neg_reweighted_subspace(h_ref, b_idx, T_b)
                    for b_idx, T_b in zip(neg_idx, neg_stacks)]
            W = torch.cat(cols, dim=1)                               # [d, Σk]
            Q_all, _ = torch.linalg.qr(W)                            # mirrors tier2c._build_union_basis
            q_tan = q_tan - Q_all @ (Q_all.T @ q_tan)                # HARD orthogonal-complement rejection

        q = F.normalize(safe_exp_map(self.mu, q_tan, self.max_tan_norm), dim=0)
        return (q, q_tan.norm()) if return_qtan_norm else q


# --------------------------------------------------------------------------- #
# Synthetic query generator (train split only) — mirrors the eval GT protocol.
# --------------------------------------------------------------------------- #
def sample_query(attrs_tr, rng, k_pos_range, k_neg_range):
    # Sample one (ref, T+, T-): + attributes the reference HAS, - attributes it LACKS.
    N = attrs_tr.shape[0]
    ref = int(torch.randint(N, (1,), generator=rng).item())
    row = attrs_tr[ref]
    present = torch.nonzero(row > 0.5, as_tuple=False).flatten()
    absent = torch.nonzero(row <= 0.5, as_tuple=False).flatten()
    k_pos = int(torch.randint(k_pos_range[0], k_pos_range[1] + 1, (1,), generator=rng).item())
    k_neg = int(torch.randint(k_neg_range[0], k_neg_range[1] + 1, (1,), generator=rng).item())
    k_pos, k_neg = min(k_pos, present.numel()), min(k_neg, absent.numel())
    pos_idx = present[torch.randperm(present.numel(), generator=rng)[:k_pos]].tolist()
    neg_idx = absent[torch.randperm(absent.numel(), generator=rng)[:k_neg]].tolist()
    return ref, pos_idx, neg_idx


def mine_positives(attrs_tr, ref, pos_idx, neg_idx, hamming_max=2):
    # Valid targets: satisfy all +/- AND Hamming ≤ hamming_max on the unconstrained attributes.
    N, A = attrs_tr.shape
    sat = torch.ones(N, dtype=torch.bool)
    if pos_idx:
        sat &= (attrs_tr[:, pos_idx] > 0.5).all(dim=1)
    if neg_idx:
        sat &= (attrs_tr[:, neg_idx] <= 0.5).all(dim=1)
    constrained = torch.zeros(A, dtype=torch.bool)
    constrained[pos_idx + neg_idx] = True
    other = ~constrained
    ham = (attrs_tr[:, other] != attrs_tr[ref, other]).sum(dim=1)
    valid = sat & (ham <= hamming_max)
    valid[ref] = False
    return torch.nonzero(valid, as_tuple=False).flatten()


def mine_hard_negatives(attrs_tr, pos_idx, neg_idx, positives):
    # Hard negatives satisfy T+ but violate ≥1 T-. With no - constraint, fall back to images that
    # fail T+ (genuine non-matches) so the contrastive denominator is never empty.
    N = attrs_tr.shape[0]
    sat_pos = torch.ones(N, dtype=torch.bool)
    if pos_idx:
        sat_pos &= (attrs_tr[:, pos_idx] > 0.5).all(dim=1)
    if neg_idx:
        hard = sat_pos & (attrs_tr[:, neg_idx] > 0.5).any(dim=1)
    else:
        hard = ~sat_pos
    hard[positives] = False
    return torch.nonzero(hard, as_tuple=False).flatten()


def build_batch(attrs_tr, rng, bank, n_counts, batch, n_pos, n_neg, k_pos_range, k_neg_range, max_retries=20):
    # Assemble `batch` synthetic queries, each with sampled positives/hard-negatives and text stacks.
    # Resamples (relaxing the - constraints first) until a query has both valid positives and negatives.
    queries = []
    while len(queries) < batch:
        chosen = None
        for _ in range(max_retries):
            ref, pos_idx, neg_idx = sample_query(attrs_tr, rng, k_pos_range, k_neg_range)
            P = mine_positives(attrs_tr, ref, pos_idx, neg_idx)
            if P.numel() == 0 and neg_idx:                       # relax: drop - constraints, retry
                neg_idx = []
                P = mine_positives(attrs_tr, ref, pos_idx, neg_idx)
            if P.numel() == 0:
                continue
            Nn = mine_hard_negatives(attrs_tr, pos_idx, neg_idx, P)
            if Nn.numel() == 0:
                continue
            chosen = (ref, pos_idx, neg_idx, P, Nn)
            break
        if chosen is None:
            continue
        ref, pos_idx, neg_idx, P, Nn = chosen
        P = P[torch.randperm(P.numel(), generator=rng)[:n_pos]]
        Nn = Nn[torch.randperm(Nn.numel(), generator=rng)[:n_neg]]
        queries.append(dict(
            ref=ref, pos_idx=pos_idx, neg_idx=neg_idx,
            pos_targets=P.tolist(), neg_targets=Nn.tolist(),
            pos_stacks=[bank[i, : n_counts[i]] for i in pos_idx],
            neg_stacks=[bank[i, : n_counts[i]] for i in neg_idx],
        ))
    return queries


# --------------------------------------------------------------------------- #
# InfoNCE training.
# --------------------------------------------------------------------------- #
def info_nce(q, P_feats, N_feats, tau):
    # Multi-positive InfoNCE: -mean_p log softmax over {positives ∪ hard-negatives} at the true p.
    pos = (P_feats @ q) / tau                          # [n_p]
    neg = (N_feats @ q) / tau                          # [n_n]
    denom = torch.logsumexp(torch.cat([pos, neg]), dim=0)
    return -(pos - denom).mean()


def run_training(phi, feats_tr, attrs_tr, bank, n_counts, cfg):
    # AdamW + cosine schedule; per-step synthetic batch; soft norm penalty + grad clip.
    # Emits per-step throughput and a wall-clock ETA so a headless run is observable.
    opt = torch.optim.AdamW(phi.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"] * cfg["steps_per_epoch"])
    rng = torch.Generator().manual_seed(cfg["seed"])
    val_rng = torch.Generator().manual_seed(cfg["seed"] + 999)

    log(f"building {cfg['val_queries']} held-out validation queries …")
    val_batch = build_batch(attrs_tr, val_rng, bank, n_counts, cfg["val_queries"],
                            cfg["n_pos"], cfg["n_neg"], cfg["k_pos_range"], cfg["k_neg_range"])

    history = {"loss": [], "val_loss": [], "qtan_max": []}
    best_val = math.inf
    Path(cfg["ckpt_dir"]).mkdir(parents=True, exist_ok=True)

    total_steps = cfg["epochs"] * cfg["steps_per_epoch"]
    log_every = max(1, cfg["steps_per_epoch"] // 8)             # ~8 progress lines per epoch
    t_start = time.time()
    done_steps = 0

    for epoch in range(cfg["epochs"]):
        phi.train()
        ep_loss, ep_qtan = 0.0, 0.0
        for step in range(cfg["steps_per_epoch"]):
            batch = build_batch(attrs_tr, rng, bank, n_counts, cfg["batch"],
                                cfg["n_pos"], cfg["n_neg"], cfg["k_pos_range"], cfg["k_neg_range"])
            opt.zero_grad()
            loss, qtan_max = 0.0, 0.0
            for qd in batch:
                q, qn = phi(feats_tr[qd["ref"]], qd["pos_idx"], qd["neg_idx"],
                            qd["pos_stacks"], qd["neg_stacks"], return_qtan_norm=True)
                loss = loss + info_nce(q, feats_tr[qd["pos_targets"]], feats_tr[qd["neg_targets"]], cfg["tau"])
                loss = loss + cfg["norm_penalty"] * F.relu(qn - cfg["max_tan_norm"]) ** 2
                qtan_max = max(qtan_max, qn.item())
            loss = loss / len(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(phi.parameters(), 5.0)
            opt.step(); sched.step()
            ep_loss += loss.item(); ep_qtan = max(ep_qtan, qtan_max)
            done_steps += 1

            if (step + 1) % log_every == 0:
                elapsed = time.time() - t_start
                rate = done_steps / elapsed                     # steps/sec
                eta = (total_steps - done_steps) / rate
                log(f"  epoch {epoch:02d}  step {step + 1:3d}/{cfg['steps_per_epoch']}  "
                    f"loss={loss.item():.4f}  {rate:.2f} it/s  ETA {eta / 60:.1f} min")

        phi.eval()
        with torch.no_grad():
            vl = sum(info_nce(phi(feats_tr[qd["ref"]], qd["pos_idx"], qd["neg_idx"],
                                  qd["pos_stacks"], qd["neg_stacks"]),
                              feats_tr[qd["pos_targets"]], feats_tr[qd["neg_targets"]], cfg["tau"]).item()
                     for qd in val_batch) / len(val_batch)

        history["loss"].append(ep_loss / cfg["steps_per_epoch"])
        history["val_loss"].append(vl); history["qtan_max"].append(ep_qtan)
        torch.save(phi.state_dict(), Path(cfg["ckpt_dir"]) / f"phi_epoch{epoch:02d}.pt")
        tag = ""
        if vl < best_val:
            best_val = vl
            torch.save(phi.state_dict(), Path(cfg["ckpt_dir"]) / "phi_best.pt")
            tag = "  <- best"
        log(f"epoch {epoch:02d} DONE  train={history['loss'][-1]:.4f}  val={vl:.4f}  "
            f"‖q_tan‖max={ep_qtan:.3f}{tag}")

    log(f"training finished in {(time.time() - t_start) / 60:.1f} min  (best val={best_val:.4f})")
    return history


# --------------------------------------------------------------------------- #
# Eval-seam adapter — plug Φ into the unchanged harness.
# --------------------------------------------------------------------------- #
def make_get_ranking_phi(query_str, phi, image_features, bank, n_counts):
    # CONTRACT §7 seam — curry one query string into get_ranking(src_idx); identical contract to the
    # training-free tiers, so it drops into eval.evaluate_all with zero harness changes.
    T_pos, T_neg = parse_query(query_str)
    pos_idx = [ATTR_TO_IDX[a] for a in T_pos]; neg_idx = [ATTR_TO_IDX[b] for b in T_neg]
    pos_stacks = [bank[i, : n_counts[i]] for i in pos_idx]
    neg_stacks = [bank[i, : n_counts[i]] for i in neg_idx]

    @torch.no_grad()
    def get_ranking(src_idx):
        q = phi(image_features[src_idx], pos_idx, neg_idx, pos_stacks, neg_stacks)
        scores = image_features @ q
        scores[src_idx] = float("-inf")                # source never ranks itself (CONTRACT §5)
        return torch.argsort(scores, descending=True).tolist()

    return get_ranking


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def main():
    torch.manual_seed(SEED)
    log(f"device={DEVICE}  K_NEG={K_NEG}  epochs={EPOCHS}  batch={BATCH}")
    log(f"PROJECT_DIR={PROJECT_DIR}")
    log(f"CKPT_DIR={CKPT_DIR}")
    if DEVICE == "cpu":
        log("WARNING: CUDA not available — training on CPU will be extremely slow.")

    # ---- Load frozen artifacts ----
    log("loading frozen artifacts (μ, directions, subspaces, prompt bank) …")
    art_dir = _get_artifacts_dir()
    mu, directions = load_or_mine_directions()
    neg_subspaces  = load_or_mine_neg_subspaces()
    bank           = load_prompt_bank()
    text_feats     = torch.load(art_dir / "clip_attr_text_features.pt", weights_only=True)
    mu_txt         = F.normalize(text_feats.mean(0), dim=0)
    n_counts       = [len(build_prompts_for_attribute(n)) for n in ATTRIBUTE_NAMES]

    mu, directions = mu.to(DEVICE), directions.to(DEVICE)
    neg_subspaces  = neg_subspaces.to(DEVICE)
    bank, mu_txt   = bank.to(DEVICE), mu_txt.to(DEVICE)

    # Tripwires — fail loud on any shape / index drift.
    assert directions.shape == (40, FEATURE_DIM), directions.shape
    assert neg_subspaces.shape[:2] == (40, FEATURE_DIM), neg_subspaces.shape
    assert neg_subspaces.shape[2] >= K_NEG, "cached subspace width < K_NEG"
    assert bank.shape[0] == 40 and bank.shape[2] == FEATURE_DIM, bank.shape
    assert all(0 < n <= bank.shape[1] for n in n_counts), "prompt count exceeds padded bank width"
    log(f"artifacts OK: directions={tuple(directions.shape)} "
        f"neg_subspaces={tuple(neg_subspaces.shape)} bank={tuple(bank.shape)}")

    # ---- Build Φ, verify it starts exactly at Tier-2c ----
    phi = FusionPhi(
        mu, mu_txt, directions, neg_subspaces,
        d_model=FEATURE_DIM, n_heads=N_HEADS, k_neg=K_NEG, alpha=ALPHA,
        rejection_mode=REJECTION_MODE, share_attn=SHARE_ATTN,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in phi.parameters() if p.requires_grad)
    log(f"Φ trainable params: {n_params / 1e6:.2f}M  (CLIP frozen)")

    image_features = load_image_features().to(DEVICE)
    phi.eval()
    test_query = "+Smiling, -Male"
    T_pos, T_neg = parse_query(test_query)
    pos_idx = [ATTR_TO_IDX[a] for a in T_pos]; neg_idx = [ATTR_TO_IDX[b] for b in T_neg]
    pos_stacks = [bank[i, : n_counts[i]] for i in pos_idx]
    neg_stacks = [bank[i, : n_counts[i]] for i in neg_idx]
    cfg2c = SVunionConfig(k_neg=K_NEG, alpha=ALPHA, reject_on="query")
    with torch.no_grad():
        q_phi = phi(image_features[42], pos_idx, neg_idx, pos_stacks, neg_stacks)
        q_2c = _compose_query_svunion(image_features[42], T_pos, T_neg, mu, directions, neg_subspaces, cfg2c)
    cos = float(F.cosine_similarity(q_phi, q_2c, dim=0))
    assert cos > 0.999, f"Φ does not reduce to Tier-2c at init (cos={cos:.6f})"
    log(f"[OK] Φ initialises exactly at Tier-2c (cos={cos:.6f})")

    # ---- Train ----
    attrs_tr = _load_train_attributes()
    feats_tr = _load_train_features().to(DEVICE)
    cfg = dict(
        lr=LR, wd=WEIGHT_DECAY, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, batch=BATCH,
        n_pos=N_POS, n_neg=N_NEG, tau=TAU, seed=SEED, k_pos_range=K_POS_RANGE, k_neg_range=K_NEG_RANGE,
        norm_penalty=NORM_PENALTY, max_tan_norm=MAX_TAN_NORM, val_queries=VAL_QUERIES, ckpt_dir=str(CKPT_DIR),
    )
    history = run_training(phi, feats_tr, attrs_tr, bank, n_counts, cfg)

    # ---- Save loss curves to file (headless) ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["loss"], label="train"); ax[0].plot(history["val_loss"], label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("InfoNCE"); ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(history["qtan_max"]); ax[1].axhline(math.pi / 2, ls="--", c="r", label="π/2 cone bound")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("max ‖q_tan‖"); ax[1].set_title("Tangent norm"); ax[1].legend()
    fig.tight_layout()
    curve_path = CKPT_DIR / "training_curves.png"
    fig.savefig(curve_path, dpi=120)
    log(f"[OK] saved loss curves to {curve_path}")

    # ---- Evaluate best checkpoint ----
    phi.load_state_dict(torch.load(CKPT_DIR / "phi_best.pt", map_location=DEVICE))
    phi.eval()
    log("evaluating best checkpoint on the eval JSON …")
    gt_list = load_eval_json(find_eval_json())
    results = evaluate_all(gt_list, lambda qs: make_get_ranking_phi(qs, phi, image_features, bank, n_counts))
    print(format_results_table(results), flush=True)
    out_csv = output_subdir("tier3c") / "tier3c_phi.csv"
    save_results_csv(results, out_csv)
    log(f"[OK] wrote {out_csv}")

    # ---- Tier-2c vs Tier-3c on identical geometry (the load-bearing comparison) ----
    def _mean(res, k):
        return sum(r[f"recall@{k}"] for r in res.values()) / len(res)
    res_2c = evaluate_svunion(SVunionConfig(k_neg=K_NEG, alpha=ALPHA, reject_on="query"), save=False)
    log("---- final comparison (mean recall) ----")
    log(f"Tier-2c (fixed)      R@1={_mean(res_2c, 1):.4f}  R@5={_mean(res_2c, 5):.4f}  R@10={_mean(res_2c, 10):.4f}")
    log(f"Tier-3c (Φ, learned) R@1={_mean(results, 1):.4f}  R@5={_mean(results, 5):.4f}  R@10={_mean(results, 10):.4f}")


if __name__ == "__main__":
    main()
