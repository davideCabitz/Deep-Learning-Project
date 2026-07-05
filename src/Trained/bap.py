"""
BAP — Amortized Bayesian Attribute-Posterior Fusion (CLIP frozen).

The benchmark's GT rule (spec §3.1.1) is a known generative process over attribute space:
a target is valid iff it strictly satisfies T⁺/T⁻ AND has Hamming ≤ 2 on the free attributes
vs the reference. Every prior tier scores cosine against ONE 512-d query vector — a linear
functional that cannot represent that set, with CLIP-cosine "identity" actively adversarial
on negation queries (the -Male,-Mustache female-twin problem, 0.000 on every geometry tier).
This tier scores the rule itself, under probe uncertainty:

  Stage 0  40 linear probes on frozen TRAIN features → calibrated posteriors p(x) ∈ [0,1]^40.
  Stage A  closed-form Bayes score (training-free given probes):
             S(x|ref) = Σ_{a∈T⁺} log p_a(x) + Σ_{b∈T⁻} log(1−p_b(x))     constraint term
                      + log P( Σ_{c∈free} D_c ≤ 2 )                       identity term
             D_c ~ Bernoulli( p_c(x)(1−p_c(ref)) + (1−p_c(x))·p_c(ref) )  Poisson-binomial DP
  Stage B  amortized fusion Φ: a small MLP over (v_ref, p_ref, T⁺, T⁻) emits per-channel
           weights over the once-built frozen augmented DB [v_x ; log p ; log(1−p) ; p] plus
           a CLIP-channel query and a tail gain κ. Zero-init heads + κ=1 ⇒ Φ at init IS
           Stage A exactly; ListNet training can only improve on it (per-query reliability
           re-weighting, correlated-attribute channels, learned CLIP-identity gain).

No test-label usage anywhere in scoring: probes train on the TRAIN split; the reference's
attributes are estimated by the probes; the eval JSON is read only to compute metrics.
(Test attribute labels are read once for a printed probe-accuracy diagnostic — never scored.)

VM-ready: timestamped progress + ETA, headless training curve, and a final tar.gz bundle of
all results/weights that `scripts/pull_tier4_bayes_results.ps1` copies back into any clone.

Run (from repo root):  python src/bap.py            # full pipeline
                       python src/bap.py --quick    # CPU smoke run
See FinalDestination.md §4.A for the full method derivation and roadmap.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import ATTRIBUTE_NAMES, ATTR_TO_IDX, _get_artifacts_dir
from clip_features import load_image_features, FEATURE_DIM
from eval import format_results_table, load_eval_json, find_eval_json
from results_saver import save_results_csv, output_subdir
from gde import _load_train_features, _load_train_attributes
from phi_gate import PhiConfig, QueryGenerator
from fusion_train_utils import assert_db_frozen
from vm_pipeline import (bundle_results, evaluate_scorer, log, query_indices,
                         save_training_curve, set_log_file)


N_ATTR = 40
EPS = 1e-4          # posterior clamp — keeps log p / log(1−p) finite and bounded (≈ ±9.2)
TAIL_FLOOR = 1e-30  # P(K≤2) floor before log — pure numerical guard, never rank-relevant


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BAPConfig:
    """Tier-4 Bayes-Φ hyperparameters. Defaults are the headline VM run."""
    # Stage 0 — probes
    probe_epochs: int = 10
    probe_lr: float = 1e-2
    probe_wd: float = 1e-5
    probe_batch: int = 8192

    # Stage B — Φ architecture + training
    hidden: int = 512
    epochs: int = 40
    lr: float = 1e-4
    weight_decay: float = 1e-2
    tau_list: float = 2.0        # ListNet temperature on LOG-LIKELIHOOD-scale scores
                                 # (tier3 used 0.07 on cosine ∈ [-1,1]; our scores span ≈ −30..0,
                                 #  so τ≈2 gives comparable softmax sharpness)
    batch_queries: int = 64
    steps_div: int = 25          # steps_per_epoch = N_train // (batch_queries · steps_div)
    k_pos_max: int = 3
    k_neg_max: int = 2
    n_pos_targets: int = 8
    n_hard_neg: int = 32
    n_rand_neg: int = 16
    hamming_max: int = 2
    seed: int = 0

    # Evaluation
    chunk: int = 256             # sources scored per batch (DP state is 3×[chunk, N] floats)

    def phi_gen_config(self) -> PhiConfig:
        # Borrow PhiConfig ONLY to drive QueryGenerator (identical sampling protocol as tier3).
        return PhiConfig(
            k_pos_max=self.k_pos_max, k_neg_max=self.k_neg_max,
            n_pos_targets=self.n_pos_targets, n_hard_neg=self.n_hard_neg,
            hamming_max=self.hamming_max, seed=self.seed,
        )


# ---------------------------------------------------------------------------
# Stage 0 — attribute probes (frozen features → calibrated posteriors)
# ---------------------------------------------------------------------------

def train_probes(cfg: BAPConfig, dev: torch.device) -> nn.Linear:
    # 40 logistic probes as one Linear(512→40), BCE on frozen TRAIN features.
    # Turns raw geometry into the calibrated currency the Bayes score needs
    # (validated in-session: 88.5% mean test accuracy in ~10 CPU-seconds).
    train_feats = _load_train_features().to(dev)
    train_attrs = _load_train_attributes().float().to(dev)
    probe = nn.Linear(FEATURE_DIM, N_ATTR).to(dev)
    opt = torch.optim.Adam(probe.parameters(), lr=cfg.probe_lr, weight_decay=cfg.probe_wd)
    lossf = nn.BCEWithLogitsLoss()
    n = train_feats.shape[0]
    for epoch in range(cfg.probe_epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, cfg.probe_batch):
            idx = perm[i:i + cfg.probe_batch]
            opt.zero_grad()
            loss = lossf(probe(train_feats[idx]), train_attrs[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * idx.numel()
        log(f"  probe epoch {epoch + 1:02d}/{cfg.probe_epochs}  BCE={tot / n:.4f}")
    return probe


def load_or_train_probes(cfg: BAPConfig, dev: torch.device, force: bool = False) -> nn.Linear:
    # Cache contract mirrors load_or_mine_directions: load if present, else train + save.
    path = _get_artifacts_dir() / "bap_probe.pt"
    probe = nn.Linear(FEATURE_DIM, N_ATTR)
    if path.exists() and not force:
        probe.load_state_dict(torch.load(path, map_location="cpu"))
        log(f"[OK] Loaded cached probes: {path}")
    else:
        log("Training attribute probes (Stage 0)...")
        probe = train_probes(cfg, dev).cpu()
        torch.save(probe.state_dict(), path)
        log(f"  Saved: {path}")
    return probe.to(dev)


@torch.no_grad()
def posteriors(probe: nn.Linear, feats: torch.Tensor) -> torch.Tensor:
    # p(x) = σ(probe(v_x)), clamped so the log-channels stay finite. [N, 40].
    return torch.sigmoid(probe(feats)).clamp(EPS, 1 - EPS)


@torch.no_grad()
def probe_accuracy_diagnostic(P_test: torch.Tensor) -> None:
    # DIAGNOSTIC ONLY — reads test labels to report probe quality; never used in scoring.
    attrs = torch.load(_get_artifacts_dir() / "celeba_attributes_test.pt", map_location="cpu")
    acc = ((P_test.cpu() > 0.5).float() == attrs.float()).float().mean(0)
    worst = int(acc.argmin())
    log(f"  probe test acc: mean {acc.mean():.4f}  min {acc[worst]:.4f} ({ATTRIBUTE_NAMES[worst]})")


# ---------------------------------------------------------------------------
# Bayes core — shared by Stage A (closed form) and Stage B (κ-weighted channel)
# ---------------------------------------------------------------------------

def hamming_tail_log(p_ref: torch.Tensor, p_db: torch.Tensor, hamming_max: int) -> torch.Tensor:
    # log P(Σ_c D_c ≤ hamming_max), D_c ~ Bernoulli(p_db·(1−p_ref) + (1−p_db)·p_ref).
    # Exact Poisson-binomial tail via a (hamming_max+1)-state DP over the free attributes —
    # the identity term no prior tier had, and where most of the negation lift comes from.
    # p_ref [S, F] (probe posterior of each reference), p_db [N, F]; returns [S, N].
    S, F_ = p_ref.shape
    N = p_db.shape[0]
    states = [torch.ones(S, N, device=p_db.device)]          # P(K = 0) running product
    states += [torch.zeros(S, N, device=p_db.device) for _ in range(hamming_max)]
    for c in range(F_):
        d = p_db[:, c].unsqueeze(0) * (1 - p_ref[:, c]).unsqueeze(1) \
            + (1 - p_db[:, c]).unsqueeze(0) * p_ref[:, c].unsqueeze(1)      # [S, N]
        omd = 1 - d
        for k in range(hamming_max, 0, -1):                   # descend so states[k-1] is pre-update
            states[k] = states[k] * omd + states[k - 1] * d
        states[0] = states[0] * omd
    return sum(states).clamp_min(TAIL_FLOOR).log()


# ---------------------------------------------------------------------------
# Stage B — the fusion module Φ
# ---------------------------------------------------------------------------
class BayesPhi(nn.Module):
    # Φ(v_ref, p_ref, T⁺, T⁻) → per-channel weights over the frozen augmented DB
    # [v_x ; log p ; log(1−p) ; p] plus a tail gain κ; score = ⟨q_aug, x_aug⟩ + κ·tail.
    # Zero-init heads + κ=1 ⇒ init ≡ Stage A closed form (provable floor = the validated
    # prototype numbers); training learns per-query probe-reliability re-weighting,
    # correlated-attribute corrections (w on unconstrained channels), and a CLIP-identity
    # channel q_v — exactly the three Stage-A failure modes (+Smiling/+Mustache/lipstick).
    def __init__(self, cfg: BAPConfig):
        super().__init__()
        d_in = FEATURE_DIM + 3 * N_ATTR                      # v_ref ⊕ p_ref ⊕ mh_pos ⊕ mh_neg
        self.trunk = nn.Sequential(
            nn.Linear(d_in, cfg.hidden), nn.LayerNorm(cfg.hidden), nn.GELU(),
            nn.Linear(cfg.hidden, cfg.hidden), nn.LayerNorm(cfg.hidden), nn.GELU(),
        )
        self.head_wpos = nn.Linear(cfg.hidden, N_ATTR)       # Δ-weights on log p channels
        self.head_wneg = nn.Linear(cfg.hidden, N_ATTR)       # Δ-weights on log(1−p) channels
        self.head_wp = nn.Linear(cfg.hidden, N_ATTR)         # weights on raw p channels
        self.head_qv = nn.Linear(cfg.hidden, FEATURE_DIM)    # CLIP-channel query (off at init)
        for head in (self.head_wpos, self.head_wneg, self.head_wp, self.head_qv):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.kappa = nn.Parameter(torch.tensor(1.0))         # identity-tail gain

    def forward(
        self,
        v_ref: torch.Tensor,                                  # [B, 512] unit reference rows
        p_ref: torch.Tensor,                                  # [B, 40] probe posterior of refs
        mh_pos: torch.Tensor,                                 # [B, 40] multi-hot T⁺
        mh_neg: torch.Tensor,                                 # [B, 40] multi-hot T⁻
    ) -> dict[str, torch.Tensor]:
        h = self.trunk(torch.cat([v_ref, p_ref, mh_pos, mh_neg], dim=1))
        return {
            "w_pos": mh_pos + self.head_wpos(h),              # Bayes indicators + learned Δ
            "w_neg": mh_neg + self.head_wneg(h),
            "w_p": self.head_wp(h),
            "q_v": self.head_qv(h),
            "kappa": self.kappa,
        }


def phi_scores(
    w: dict[str, torch.Tensor],
    feats: torch.Tensor,                                      # [N, 512] frozen CLIP rows
    P: torch.Tensor,                                          # [N, 40] frozen posteriors
    tail: torch.Tensor,                                       # [B, N] identity tail (log)
) -> torch.Tensor:
    # score = w_pos·log p + w_neg·log(1−p) + w_p·p + q_v·v + κ·tail   → [B, N].
    # The channel tensors are the once-built frozen augmented DB; only w/κ vary per query.
    return (
        w["w_pos"] @ P.log().T
        + w["w_neg"] @ (1 - P).log().T
        + w["w_p"] @ P.T
        + w["q_v"] @ feats.T
        + w["kappa"] * tail
    )


def listnet_scores(scores: torch.Tensor, n_pos: int, tau: float) -> torch.Tensor:
    # ListNet on raw score vectors (Cao et al. 2007) — generalizes tier3_contrastive's
    # cosine-bound listnet_loss to composite scores. y_i = 1/|pos| for positives, else 0.
    labels = torch.zeros_like(scores)
    labels[:n_pos] = 1.0 / n_pos
    return -(labels * F.log_softmax(scores / tau, dim=0)).sum()


# ---------------------------------------------------------------------------
# Stage A — closed-form Bayes scorer (training-free given probes)
# ---------------------------------------------------------------------------

def make_stage_a_chunk(feats: torch.Tensor, P: torch.Tensor, cfg: BAPConfig):
    # Stage A score = constraint log-likelihood + identity tail. No CLIP channel (γ=0):
    # matches the validated prototype exactly; the CLIP channel is Stage B's to learn.
    logP, log1mP = P.log(), (1 - P).log()

    def score_chunk(srcs: list[int], pi: list[int], ni: list[int], free: list[int]) -> torch.Tensor:
        constraint = logP[:, pi].sum(1) + log1mP[:, ni].sum(1)           # [N]
        tail = hamming_tail_log(P[srcs][:, free], P[:, free], cfg.hamming_max)
        return constraint.unsqueeze(0) + tail

    return score_chunk


def make_get_ranking(query_str: str, feats: torch.Tensor, P: torch.Tensor,
                     cfg: BAPConfig, model: BayesPhi | None = None):
    # CONTRACT §5/§7 seam — curry one query into get_ranking(src_idx) (single-source path;
    # bulk evaluation goes through vm_pipeline.evaluate_scorer's chunked driver instead).
    pi, ni, free = query_indices(query_str)
    chunk_fn = make_stage_a_chunk(feats, P, cfg) if model is None \
        else make_stage_b_chunk(model, feats, P, cfg)

    @torch.no_grad()
    def get_ranking(src_idx: int) -> list[int]:
        scores = chunk_fn([src_idx], pi, ni, free).squeeze(0)
        scores[src_idx] = float("-inf")
        return torch.argsort(scores, descending=True).tolist()

    return get_ranking


# ---------------------------------------------------------------------------
# Stage B — training
# ---------------------------------------------------------------------------

def train_phi(
    cfg: BAPConfig,
    probe: nn.Linear,
    dev: torch.device,
    max_steps: int | None = None,
) -> tuple[BayesPhi, list[float]]:
    # ListNet over synthetic train-split queries (QueryGenerator mirrors the GT rule; no
    # leakage). Candidates per query = valid targets ⊕ constraint-violating hard negatives
    # ⊕ random negatives; scores via the full Stage-B composite so the gradient reaches
    # every head. Returns (trained model on dev, per-epoch loss curve).
    torch.manual_seed(cfg.seed)
    train_feats = _load_train_features().to(dev)
    train_attrs = _load_train_attributes()
    P_tr = posteriors(probe, train_feats)                     # frozen channel values
    logP_tr, log1mP_tr = P_tr.log(), (1 - P_tr).log()

    model = BayesPhi(cfg).to(dev)
    gen = QueryGenerator(train_attrs, cfg.phi_gen_config())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    steps_per_epoch = max(1, gen.N // (cfg.batch_queries * cfg.steps_div))
    if max_steps is not None:
        steps_per_epoch = min(steps_per_epoch, max_steps)
    total_steps = cfg.epochs * steps_per_epoch
    log(f"Stage B: {sum(p.numel() for p in model.parameters()):,} params, "
        f"{cfg.epochs} epochs x {steps_per_epoch} steps x {cfg.batch_queries} queries")

    curve, step, t0 = [], 0, time.time()
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
                pi = [ATTR_TO_IDX[a] for a in pos_names]
                ni = [ATTR_TO_IDX[b] for b in neg_names]
                free = [c for c in range(N_ATTR) if c not in set(pi) | set(ni)]

                rand_ids = torch.randint(gen.N, (cfg.n_rand_neg,), generator=gen.g)
                cand = torch.cat([pos_ids, hard_ids, rand_ids]).to(dev)

                mh_pos = torch.zeros(1, N_ATTR, device=dev)
                mh_neg = torch.zeros(1, N_ATTR, device=dev)
                mh_pos[0, pi], mh_neg[0, ni] = 1.0, 1.0
                w = model(train_feats[r].unsqueeze(0), P_tr[r].unsqueeze(0), mh_pos, mh_neg)

                with torch.no_grad():                          # tail is a fixed channel value
                    tail = hamming_tail_log(
                        P_tr[r].unsqueeze(0)[:, free], P_tr[cand][:, free], cfg.hamming_max,
                    )
                scores = (
                    w["w_pos"] @ logP_tr[cand].T
                    + w["w_neg"] @ log1mP_tr[cand].T
                    + w["w_p"] @ P_tr[cand].T
                    + w["q_v"] @ train_feats[cand].T
                    + w["kappa"] * tail
                ).squeeze(0)                                   # [C]
                batch_loss = batch_loss + listnet_scores(scores, pos_ids.numel(), cfg.tau_list)
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
        curve.append(running / max(seen, 1))
        eta = (time.time() - t0) / max(step, 1) * (total_steps - step) / 60
        log(f"  epoch {epoch + 1:02d}/{cfg.epochs}  ListNet={curve[-1]:.4f}  "
            f"kappa={model.kappa.item():.3f}  eta={eta:.1f}min")
    return model, curve


def make_stage_b_chunk(model: BayesPhi, feats: torch.Tensor, P: torch.Tensor, cfg: BAPConfig):
    # Stage-B chunked scorer for the eval driver — one Φ forward per source chunk, then the
    # composite score against the full frozen augmented DB.
    model.eval()

    @torch.no_grad()
    def score_chunk(srcs: list[int], pi: list[int], ni: list[int], free: list[int]) -> torch.Tensor:
        S = len(srcs)
        mh_pos = torch.zeros(S, N_ATTR, device=feats.device)
        mh_neg = torch.zeros(S, N_ATTR, device=feats.device)
        mh_pos[:, pi], mh_neg[:, ni] = 1.0, 1.0
        w = model(feats[srcs], P[srcs], mh_pos, mh_neg)
        tail = hamming_tail_log(P[srcs][:, free], P[:, free], cfg.hamming_max)
        return phi_scores(w, feats, P, tail)

    return score_chunk


@torch.no_grad()
def init_sanity_check(model: BayesPhi, feats, P, cfg, query_str: str, src: int) -> float:
    # Guard the load-bearing invariant: Φ at init must reproduce Stage A bit-for-bit
    # (up to float32 matmul noise). Returns max |Δscore| over the DB for one source.
    pi, ni, free = query_indices(query_str)
    a = make_stage_a_chunk(feats, P, cfg)([src], pi, ni, free)
    b = make_stage_b_chunk(model, feats, P, cfg)([src], pi, ni, free)
    return (a - b).abs().max().item()


# ---------------------------------------------------------------------------
# Results bundle — carry everything back from the VM clone to any local clone
# ---------------------------------------------------------------------------

def bundle_tier4_results(out_dir: Path) -> Path:
    # Tier-4 bundle (output/bap/* + the probe cache) via the shared bundler.
    # Printed scp hint uses the VM coordinates from documents/Tier3_Variations_Report.md §4.
    root = Path(__file__).parent.parent.parent  # file now lives in src/Trained/
    return bundle_results(
        root, root / "output" / "tier4_bayes_bundle.tar.gz",
        [out_dir, _get_artifacts_dir() / "bap_probe.pt"],
        pull_hints=[
            "Pull it back to your local clone with (from local repo root):",
            "  powershell scripts/pull_tier4_bayes_results.ps1",
            "  # or manually:",
            "  scp -P 5040 disi@lab-d2fdded3-9ba4-4188-80fa-16c27a4cd022.westeurope.cloudapp.azure.com:"
            "~/Deep-Learning-Project/output/tier4_bayes_bundle.tar.gz .",
            "  tar -xzf tier4_bayes_bundle.tar.gz   # extracts output/bap/* in place",
        ])


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Tier-4 Bayes-Phi: probes -> Stage A -> Stage B")
    ap.add_argument("--quick", action="store_true", help="CPU smoke run (tiny epochs, 2 queries)")
    ap.add_argument("--skip-stage-b", action="store_true", help="probes + Stage A only")
    ap.add_argument("--force-probes", action="store_true", help="retrain probes even if cached")
    ap.add_argument("--epochs", type=int, default=None, help="override Stage-B epochs")
    ap.add_argument("--device", type=str, default=None, help="cuda | cpu (default: auto)")
    args = ap.parse_args()

    cfg = BAPConfig()
    if args.epochs is not None:
        cfg = replace(cfg, epochs=args.epochs)
    if args.quick:
        cfg = replace(cfg, probe_epochs=2, epochs=2, chunk=64)

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = output_subdir("bap")
    set_log_file(out_dir / "train_log.txt")
    tag = "quick" if args.quick else "full"
    log(f"Device: {dev}" + (f"  GPU: {torch.cuda.get_device_name(0)}" if dev.type == "cuda" else ""))

    # --- Stage 0: probes -----------------------------------------------------
    probe = load_or_train_probes(cfg, dev, force=args.force_probes)
    feats = load_image_features().to(dev)
    P = posteriors(probe, feats)
    probe_accuracy_diagnostic(P)

    gt_list = load_eval_json(find_eval_json())
    if args.quick:
        gt_list = gt_list[:2]

    # --- Stage A: closed-form Bayes scorer ------------------------------------
    log("Stage A — closed-form Bayes scorer")
    assert_db_frozen(feats, make_get_ranking(gt_list[0]["query"], feats, P, cfg))
    results_a = evaluate_scorer("A", make_stage_a_chunk(feats, P, cfg), gt_list, cfg.chunk)
    print("\n" + format_results_table(results_a))
    save_results_csv(results_a, out_dir / f"tier4_bayes_stageA_{tag}.csv")

    if args.skip_stage_b:
        bundle_tier4_results(out_dir)
        return

    # --- Stage B: amortized fusion Φ -------------------------------------------
    log("Stage B — amortized fusion Phi (Bayes-init, ListNet)")
    model, curve = train_phi(cfg, probe, dev, max_steps=5 if args.quick else None)

    fresh = BayesPhi(cfg).to(dev)                    # init sanity on an UNTRAINED copy
    diff = init_sanity_check(fresh, feats, P, cfg, gt_list[0]["query"], next(
        int(s) for s in gt_list[0]["ground_truth"].keys()))
    log(f"  init sanity: max |Phi_init - StageA| = {diff:.2e} (float32 noise expected)")

    torch.save(model.state_dict(), out_dir / "tier4_bayes_phi.pt")
    save_training_curve(curve, out_dir / "tier4_training_curve.png",
                        "Tier-4 Bayes-Phi training", "ListNet loss")

    assert_db_frozen(feats, make_get_ranking(gt_list[0]["query"], feats, P, cfg, model=model))
    results_b = evaluate_scorer("B", make_stage_b_chunk(model, feats, P, cfg), gt_list, cfg.chunk)
    print("\n" + format_results_table(results_b))
    save_results_csv(results_b, out_dir / f"tier4_bayes_stageB_{tag}.csv")

    mean_a = sum(r["recall@5"] for r in results_a.values()) / len(results_a)
    mean_b = sum(r["recall@5"] for r in results_b.values()) / len(results_b)
    log(f"MEAN R@5 — Stage A {mean_a:.4f}  |  Stage B {mean_b:.4f}  "
        f"|  phi_combined baseline 0.1576")

    bundle_tier4_results(out_dir)


if __name__ == "__main__":
    main()
