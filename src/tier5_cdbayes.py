"""
Tier-5 CD-Bayes — Correlation-aware Dependent Bayes (NextStep.md Part III).

Tier-4 scored the GT rule (spec §3.1.1) in probability space but assumed the 40 CelebA
attributes independent — both in the constraint product and in the Poisson-binomial identity
tail — and consumed raw BCE sigmoids as if calibrated. CD-Bayes removes each diagnosed
failure mode (NextStep.md Part II) with one independently-ablatable component:

  C1  Calibrated probes — per-attribute temperature scaling (Guo et al., ICML 2017):
      τ_a fit by NLL on a held-out TRAIN fold, p = σ(z/τ). Fixes (ii) miscalibration.
  C2  Chow-Liu dependency tree (Chow & Liu, IEEE-IT 1968; TAN lineage Friedman et al. 1997):
      max-spanning tree under pairwise MI counted on TRAIN labels; the constraint term
      becomes log p_tree(T⁺=1, T⁻=0 | probe soft evidence) by exact sum-product.
      Fixes (i) — the lipstick-composite pathology.
  C3  Conditioned Hamming tail — one BP pass conditions the free attributes' posteriors on
      the constrained assignment; the EXISTING tier4 Poisson-binomial DP then runs unchanged
      on the conditioned marginals. Fixes (i), second order.
  C4  λ-cosine identity channel — S += λ·cos(v_x, v_ref), λ a single scalar grid-tuned on
      synthetic TRAIN queries. The non-learned counterpart of Stage B's q_v (targets the
      R@1 regression without the capacity to overfit CLIP's color bias).
  C5  QB-Norm — Dynamic Inverted Softmax (Bogolin et al., CVPR 2022) against a querybank of
      synthetic-TRAIN-query score vectors. Fixes (v) hubness; post-hoc, training-free.
  B′  Retrained Φ — tier4's BayesPhi (zero-init invariant intact) on top of the Stage A′
      core score, with LambdaLoss (Wang et al., CIKM 2018) replacing ListNet, a --no-qv
      gate (the hair-color-bias hypothesis test, failure iv), and an N-seed driver.
      (B′ deliberately trains on A′ core — C1+C2+C3 — WITHOUT C4/C5, so the q_v on/off
      comparison isolates the learned CLIP channel instead of confounding it with λ·cos.)

HARD INVARIANT — verified at startup and in test/test_tier5_cdbayes.py: with every component
off (τ=1, no tree, unconditioned tail, λ=0, no QB-Norm) the scorer reproduces Tier-4 Stage A
exactly, and Φ at init reproduces Stage A′ exactly (tier4's init_sanity_check pattern).

ZERO TEST-LABEL USAGE IN SCORING: probes, temperatures, tree + tables, λ, β and the
querybank are all fit on the TRAIN split only. The eval JSON is read only to compute
metrics; test attribute labels are never touched by any scoring path.

Ablation matrix (one run = all rows, each row → its own per-query CSV):

  row                      components            failure mode it tests
  ------------------------ --------------------- ------------------------------------------
  A_tier4repro             all off               parity anchor (= tier4 Stage A)
  A_C1                     C1                    (ii) miscalibration — rare-attr queries
  A_C1C2                   C1+C2                 (i) independence — lipstick composite
  A_C1C2C3  (= A′ core)    C1+C2+C3              (i) 2nd order — negation identity term
  Aprime_C4                A′ core + λ·cos       R@1 recovery, non-learned identity
  Aprime_C5                A′ core + QB-Norm     (v) hubness
  Aprime_C4C5 (= full A′)  A′ core + C4 + C5     the full training-free operating point
  Bprime_listnet           Φ(ListNet) on A′      loss-change control (1 seed by default)
  Bprime_lambdaloss        Φ(LambdaLoss) on A′   (iii) flat target mass — the R@1→R@5 shift
  Bprime_lambdaloss_noqv   + --no-qv             (iv) learned-CLIP-channel color bias

Expected runtime (full matrix, default seeds): Stage-A′ rows + tuning + querybanks ≈ 1 GPU-h;
B′ = 7 trainings (LambdaLoss ×3 seeds, no-qv ×3 seeds, ListNet ×1 control) ≈ 5–6 GPU-h at
tier4's measured ~45 min/run + 7 evals ≈ 0.5 GPU-h. Total ≈ 6.5–7.5 GPU-h (≤ 8 budget).
If a training run drifts past ~1 h, trim with --seeds 2 (cut order per NextStep.md Part V).

Run (from repo root):  python src/tier5_cdbayes.py              # full ablation matrix
                       python src/tier5_cdbayes.py --quick      # CPU smoke run
                       python src/tier5_cdbayes.py --rows A_C1C2,Bprime_lambdaloss
Results land in output/tier5_cdbayes/ and are bundled for
scripts/pull_tier5_cdbayes_results.ps1.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import ATTRIBUTE_NAMES, ATTR_TO_IDX, _get_artifacts_dir
from clip_features import load_image_features
from eval import format_results_table, load_eval_json, find_eval_json
from results_saver import save_results_csv, output_subdir
from tier1_GDE import _load_train_features, _load_train_attributes
from tier3_dgp import QueryGenerator
from tier3_train_utils import assert_db_frozen
from tier4_bayes import (EPS, N_ATTR, BayesPhi, Tier4Config, hamming_tail_log,
                         listnet_scores, load_or_train_probes, make_stage_a_chunk,
                         posteriors)
from vm_pipeline import (bundle_results, evaluate_scorer, log, query_indices,
                         save_training_curve, set_log_file)


NEG = -1e30         # log-space "impossible" — finite so logsumexp never sees -inf − -inf


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Tier5Config:
    """CD-Bayes hyperparameters. Probe/Φ/eval settings are inherited from Tier4Config
    so the parity anchor and Φ architecture stay bit-identical to tier4."""
    tier4: Tier4Config = field(default_factory=Tier4Config)

    # C1 — temperature scaling
    calib_holdout_frac: float = 0.2
    calib_seed: int = 7
    calib_steps: int = 300
    calib_lr: float = 0.05
    ece_bins: int = 15

    # C2 — Chow-Liu tree
    laplace_alpha: float = 1.0

    # C4/C5 — TRAIN-side scalar tuning (never touches test queries)
    tune_queries: int = 200
    tune_gallery: int = 20000
    tune_seed: int = 123
    lambda_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    beta_grid: tuple[float, ...] = (0.125, 0.25, 0.5, 1.0, 2.0)

    # C5 — querybank
    querybank_size: int = 256
    querybank_seed: int = 456

    # B′ — LambdaLoss + seeds
    sigma_lambda: float = 1.0     # pairwise logistic scale in LambdaLoss
    seeds: tuple[int, ...] = (0, 1, 2)


# ---------------------------------------------------------------------------
# C1 — calibrated probes (temperature scaling, Guo et al. 2017)
# ---------------------------------------------------------------------------

def fit_temperature_scalars(logits: torch.Tensor, labels: torch.Tensor,
                            steps: int, lr: float) -> torch.Tensor:
    # C1 core — τ = argmin_τ NLL(σ(z/τ), y) on held-out logits (Guo et al. 2017).
    # One scalar per attribute, optimized jointly in log-space (attributes are independent
    # terms of the same mean-BCE objective, so joint Adam ≡ 40 scalar fits).
    log_tau = torch.zeros(logits.shape[1], device=logits.device, requires_grad=True)
    opt = torch.optim.Adam([log_tau], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits / log_tau.exp(), labels)
        loss.backward()
        opt.step()
    return log_tau.detach().exp()


def expected_calibration_error(p: torch.Tensor, y: torch.Tensor, n_bins: int) -> torch.Tensor:
    # Per-attribute ECE = Σ_b (n_b/M)·|E[y|b] − E[p|b]|, binned on p(A=1) — the standard
    # reliability-diagram gap, reported before/after C1 as a report diagnostic.
    ece = torch.zeros(p.shape[1], device=p.device)
    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=p.device)
    for b in range(n_bins):
        in_bin = (p > edges[b]) & (p <= edges[b + 1]) if b else (p <= edges[1])
        n_b = in_bin.float().sum(0)                                     # [A]
        gap = ((y * in_bin).sum(0) - (p * in_bin).sum(0)).abs()
        ece += torch.where(n_b > 0, gap / p.shape[0], torch.zeros_like(gap))
    return ece


@torch.no_grad()
def calibrated_posteriors(probe: nn.Linear, feats: torch.Tensor,
                          tau: torch.Tensor | None) -> torch.Tensor:
    # p(x) = σ(probe(v_x)/τ), clamped like tier4. tau=None ⇒ bit-identical to tier4's
    # posteriors() — the C1-off arm of the hard invariant.
    if tau is None:
        return posteriors(probe, feats)
    return torch.sigmoid(probe(feats) / tau).clamp(EPS, 1 - EPS)


def load_or_fit_temperatures(probe: nn.Linear, cfg: Tier5Config, dev: torch.device,
                             force: bool = False) -> torch.Tensor:
    # Once-built offline artifact (spec §3.2, same contract as the probe matrix): fit τ on a
    # held-out TRAIN fold, cache with the ECE before/after diagnostics, freeze.
    path = _get_artifacts_dir() / "tier5_temperatures.pt"
    if path.exists() and not force:
        ckpt = torch.load(path, map_location="cpu")
        log(f"[OK] Loaded cached temperatures: {path}  "
            f"(mean ECE {ckpt['ece_before'].mean():.4f} -> {ckpt['ece_after'].mean():.4f})")
        return ckpt["tau"].to(dev)

    log("Fitting per-attribute temperatures (C1)...")
    feats = _load_train_features().to(dev)
    labels = _load_train_attributes().float().to(dev)
    g = torch.Generator().manual_seed(cfg.calib_seed)
    hold = torch.randperm(feats.shape[0], generator=g)[:int(cfg.calib_holdout_frac * feats.shape[0])]
    with torch.no_grad():
        z = probe(feats[hold])
    y = labels[hold]
    tau = fit_temperature_scalars(z, y, cfg.calib_steps, cfg.calib_lr)

    ece_before = expected_calibration_error(torch.sigmoid(z), y, cfg.ece_bins)
    ece_after = expected_calibration_error(torch.sigmoid(z / tau), y, cfg.ece_bins)
    log("  per-attribute  tau  ECE(before)  ECE(after):")
    for a, name in enumerate(ATTRIBUTE_NAMES):
        log(f"    {name:<22s} {tau[a]:6.3f}  {ece_before[a]:.4f}  {ece_after[a]:.4f}")
    log(f"  mean ECE {ece_before.mean():.4f} -> {ece_after.mean():.4f}")

    torch.save({"tau": tau.cpu(), "ece_before": ece_before.cpu(), "ece_after": ece_after.cpu(),
                "holdout_frac": cfg.calib_holdout_frac, "seed": cfg.calib_seed}, path)
    log(f"  Saved: {path}")
    return tau


# ---------------------------------------------------------------------------
# C2 — Chow-Liu dependency tree + exact sum-product (from scratch)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DependencyTree:
    """Rooted forest over the attributes with CPTs; the once-built offline artifact of C2."""
    parent: tuple[int, ...]       # parent[i] = -1 for roots (a forest is a valid instance)
    order: tuple[int, ...]        # topological, roots first
    log_cpt: torch.Tensor         # [n, 2, 2] log p(child=s_c | parent=s_p); root rows unused
    log_marginal: torch.Tensor    # [n, 2] log π_a — root priors AND evidence-ratio denominators
    mi: torch.Tensor              # [n, n] pairwise MI (report diagnostic)

    def to(self, dev: torch.device) -> "DependencyTree":
        return replace(self, log_cpt=self.log_cpt.to(dev),
                       log_marginal=self.log_marginal.to(dev), mi=self.mi.to(dev))


def _pair_counts(A: torch.Tensor) -> torch.Tensor:
    # Raw pairwise co-occurrence counts[a, b, s_a, s_b] from binary labels A [N, n] — the
    # O(n²) sufficient statistics behind MI, CPTs and marginals alike.
    n1 = A.sum(0)                                                       # [n]
    c11 = A.T @ A                                                       # [n, n]
    counts = torch.empty(A.shape[1], A.shape[1], 2, 2)
    counts[:, :, 1, 1] = c11
    counts[:, :, 1, 0] = n1.unsqueeze(1) - c11
    counts[:, :, 0, 1] = n1.unsqueeze(0) - c11
    counts[:, :, 0, 0] = A.shape[0] - counts[:, :, 1, 1] - counts[:, :, 1, 0] - counts[:, :, 0, 1]
    return counts


def mutual_information_matrix(A: torch.Tensor, alpha: float) -> torch.Tensor:
    # I(a,b) = Σ_{s_a,s_b} p(s_a,s_b)·log[ p(s_a,s_b) / (p(s_a)p(s_b)) ] on Laplace-smoothed
    # counts — the edge weights of Chow & Liu (1968). Diagonal zeroed (self-MI is not an edge).
    joint = (_pair_counts(A) + alpha) / (A.shape[0] + 4 * alpha)        # [n, n, 2, 2]
    ma = joint.sum(dim=3)                                               # [n, n, 2] p(s_a)
    mb = joint.sum(dim=2)                                               # [n, n, 2] p(s_b)
    mi = (joint * (joint.log() - ma.unsqueeze(3).log() - mb.unsqueeze(2).log())).sum(dim=(2, 3))
    mi.fill_diagonal_(0.0)
    return mi


def maximum_spanning_tree(weights: torch.Tensor) -> list[int]:
    # Prim's algorithm on the dense n×n weight matrix (n=40 → trivial). Returns parent[],
    # rooted at node 0; deterministic tie-break via argmax's first-max convention.
    n = weights.shape[0]
    parent = [-1] * n
    in_tree = torch.zeros(n, dtype=torch.bool)
    in_tree[0] = True
    best_w = weights[0].clone()                  # best edge weight into each outside node
    best_w[0] = float("-inf")
    best_from = torch.zeros(n, dtype=torch.long)
    for _ in range(n - 1):
        cand = best_w.masked_fill(in_tree, float("-inf"))
        v = int(cand.argmax())
        parent[v] = int(best_from[v])
        in_tree[v] = True
        better = (weights[v] > best_w) & ~in_tree
        best_from[better] = v
        best_w = torch.where(better, weights[v], best_w)
    return parent


def _topological_order(parent: list[int]) -> list[int]:
    # Roots first, every node after its parent (BFS over the forest).
    n = len(parent)
    children: list[list[int]] = [[] for _ in range(n)]
    order = []
    for v, p in enumerate(parent):
        (order if p < 0 else children[p]).append(v)
    i = 0
    while i < len(order):
        order.extend(children[order[i]])
        i += 1
    if len(order) != n:
        raise ValueError("parent[] contains a cycle — not a forest")
    return order


def build_chow_liu_tree(A: torch.Tensor, alpha: float) -> DependencyTree:
    # C2 offline build — MI counting → max-spanning tree → smoothed CPTs (Chow & Liu 1968).
    # Everything derives from TRAIN-label counts; generic in n so tests run on tiny inputs.
    A = A.float().cpu()
    n_img, n = A.shape
    counts = _pair_counts(A)
    mi = mutual_information_matrix(A, alpha)
    parent = maximum_spanning_tree(mi)
    order = _topological_order(parent)

    n1 = A.sum(0)
    log_marginal = torch.stack(
        [(n_img - n1 + alpha), (n1 + alpha)], dim=1).div(n_img + 2 * alpha).log()   # [n, 2]
    log_cpt = torch.zeros(n, 2, 2)
    for c, p in enumerate(parent):
        if p < 0:
            continue
        pair = counts[p, c] + alpha                                     # [2(s_p), 2(s_c)]
        log_cpt[c] = (pair / pair.sum(dim=1, keepdim=True)).log()
    return DependencyTree(tuple(parent), tuple(order), log_cpt, log_marginal, mi)


def load_or_build_tree(cfg: Tier5Config, force: bool = False) -> DependencyTree:
    # Once-built offline artifact (spec §3.2): counted from TRAIN labels, cached, frozen.
    path = _get_artifacts_dir() / "tier5_chowliu.pt"
    if path.exists() and not force:
        ckpt = torch.load(path, map_location="cpu")
        log(f"[OK] Loaded cached Chow-Liu tree: {path}")
        return DependencyTree(tuple(ckpt["parent"]), tuple(ckpt["order"]),
                              ckpt["log_cpt"], ckpt["log_marginal"], ckpt["mi"])

    log("Building Chow-Liu dependency tree (C2)...")
    tree = build_chow_liu_tree(_load_train_attributes(), cfg.laplace_alpha)
    edges = sorted(((tree.mi[c, p].item(), c, p) for c, p in enumerate(tree.parent) if p >= 0),
                   reverse=True)
    for w, c, p in edges[:5]:
        log(f"  top-MI edge: {ATTRIBUTE_NAMES[p]} -- {ATTRIBUTE_NAMES[c]}  (MI={w:.4f})")
    torch.save({"parent": list(tree.parent), "order": list(tree.order), "log_cpt": tree.log_cpt,
                "log_marginal": tree.log_marginal, "mi": tree.mi, "alpha": cfg.laplace_alpha}, path)
    log(f"  Saved: {path}")
    return tree


def evidence_log_ratio(P: torch.Tensor, log_marginal: torch.Tensor) -> torch.Tensor:
    # Scaled-likelihood soft evidence: u_a(s) = log q_a(s) − log π_a(s). Dividing the probe
    # posterior by the TRAIN prior makes sum-product over an EDGELESS forest reproduce the
    # independent product Π q_a exactly — the algebra behind the tier4-parity invariant.
    u = torch.stack([(1 - P).log(), P.log()], dim=2)                    # [N, n, 2]
    return u - log_marginal.unsqueeze(0)


def clamp_evidence(log_u: torch.Tensor, pi: list[int], ni: list[int]) -> torch.Tensor:
    # Constrain T⁺=1 / T⁻=0 by making the forbidden state log-impossible.
    u = log_u.clone()
    u[:, pi, 0] = NEG
    u[:, ni, 1] = NEG
    return u


def _upward_pass(tree: DependencyTree, log_u: torch.Tensor) -> tuple[torch.Tensor, list]:
    # Leaves→roots sum-product: acc[:, a] = u_a + Σ_{c∈children(a)} m_{c→a};
    # m_{c→p}(s_p) = logsumexp_{s_c}[ log p(s_c|s_p) + acc_c(s_c) ]. Messages kept for the
    # downward pass. Batched over the leading dim (the whole DB at once).
    acc = log_u.clone()
    msgs: list[torch.Tensor | None] = [None] * len(tree.parent)
    for c in reversed(tree.order):
        p = tree.parent[c]
        if p < 0:
            continue
        m = torch.logsumexp(tree.log_cpt[c].unsqueeze(0) + acc[:, c].unsqueeze(1), dim=2)
        msgs[c] = m
        acc[:, p] = acc[:, p] + m
    return acc, msgs


def tree_log_partition(tree: DependencyTree, log_u: torch.Tensor) -> torch.Tensor:
    # log Z = log Σ_A p_T(A)·Π_a exp(u_a(A_a)) — exact on trees/forests (one upward pass).
    acc, _ = _upward_pass(tree, log_u)
    z = torch.zeros(log_u.shape[0], device=log_u.device)
    for r, p in enumerate(tree.parent):
        if p < 0:
            z = z + torch.logsumexp(tree.log_marginal[r].unsqueeze(0) + acc[:, r], dim=1)
    return z


def tree_marginals(tree: DependencyTree, log_u: torch.Tensor) -> torch.Tensor:
    # p(A_a = 1 | evidence) for every node — full two-pass belief propagation.
    # Downward: belief_c = u_c + child-msgs + logsumexp_{s_p}[ log p(s_c|s_p) +
    # belief_p − m_{c→p} ] (the parent's belief with c's own contribution divided out).
    acc, msgs = _upward_pass(tree, log_u)
    belief = torch.empty_like(acc)
    for a in tree.order:
        p = tree.parent[a]
        if p < 0:
            belief[:, a] = tree.log_marginal[a].unsqueeze(0) + acc[:, a]
        else:
            down = torch.logsumexp(
                tree.log_cpt[a].unsqueeze(0) + (belief[:, p] - msgs[a]).unsqueeze(2), dim=1)
            belief[:, a] = down + acc[:, a]
    return (belief[:, :, 1] - torch.logsumexp(belief, dim=2)).exp()


def tree_constraint_log(tree: DependencyTree, log_u: torch.Tensor, log_z_free: torch.Tensor,
                        pi: list[int], ni: list[int]) -> torch.Tensor:
    # C2 scoring term — log p_tree(T⁺=1, T⁻=0 | soft evidence) = logZ(clamped) − logZ(free).
    # (With an edgeless forest this collapses to Σ log q⁺ + Σ log(1−q⁻) — tier4's term.)
    if not pi and not ni:
        return torch.zeros(log_u.shape[0], device=log_u.device)
    return tree_log_partition(tree, clamp_evidence(log_u, pi, ni)) - log_z_free


# ---------------------------------------------------------------------------
# Stage A′ scorer — C1..C4 composed, all defaults ⇒ tier4 Stage A exactly
# ---------------------------------------------------------------------------
class StageAPrime:
    # Stage A′ (closed-form CD-Bayes). S(x|ref) = constraint + tail [+ λ·cos(v_x, v_ref)]:
    #   constraint = tree sum-product (C2) or tier4's independent Σlog terms;
    #   tail = tier4's Poisson-binomial DP on conditioned (C3) or raw marginals.
    # Calibration (C1) lives in the P passed in; QB-Norm (C5) wraps outside (post-hoc).
    # Per-query tree work (BP over the whole DB) is memoized so the chunked eval driver
    # pays it once per query, not once per source chunk.
    def __init__(self, feats: torch.Tensor, P: torch.Tensor, tree: DependencyTree | None,
                 conditioned_tail: bool, lam: float, hamming_max: int):
        if conditioned_tail and tree is None:
            raise ValueError("C3 (conditioned tail) requires C2 (the dependency tree)")
        self.feats, self.P = feats, P
        self.logP, self.log1mP = P.log(), (1 - P).log()
        self.tree, self.conditioned_tail, self.lam = tree, conditioned_tail, lam
        self.hamming_max = hamming_max
        if tree is not None:
            self.log_u = evidence_log_ratio(P, tree.log_marginal)
            self.log_z_free = tree_log_partition(tree, self.log_u)
        self._key: tuple | None = None
        self._terms: tuple[torch.Tensor, torch.Tensor] | None = None

    def query_terms(self, pi: list[int], ni: list[int],
                    free: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        # (constraint [N], DB-side tail marginals [N, F]) — the query-dependent, source-
        # independent precompute.
        key = (tuple(pi), tuple(ni))
        if self._key != key:
            if self.tree is not None:
                constraint = tree_constraint_log(self.tree, self.log_u, self.log_z_free, pi, ni)
                if self.conditioned_tail:
                    marg = tree_marginals(self.tree, clamp_evidence(self.log_u, pi, ni))
                    marg = marg[:, free].clamp(EPS, 1 - EPS)
                else:
                    marg = self.P[:, free]
            else:
                constraint = self.logP[:, pi].sum(1) + self.log1mP[:, ni].sum(1)
                marg = self.P[:, free]
            self._key, self._terms = key, (constraint, marg)
        return self._terms

    @torch.no_grad()
    def score(self, v_ref: torch.Tensor, p_ref: torch.Tensor,
              pi: list[int], ni: list[int], free: list[int]) -> torch.Tensor:
        # [S, N] scores for arbitrary reference rows (eval sources OR train-side bank refs).
        constraint, marg = self.query_terms(pi, ni, free)
        tail = hamming_tail_log(p_ref[:, free], marg, self.hamming_max)
        s = constraint.unsqueeze(0) + tail
        if self.lam:
            s = s + self.lam * (v_ref @ self.feats.T)
        return s

    def chunk(self, srcs: list[int], pi: list[int], ni: list[int],
              free: list[int]) -> torch.Tensor:
        # evaluate_scorer seam — sources are rows of the frozen test DB itself.
        return self.score(self.feats[srcs], self.P[srcs], pi, ni, free)


def verify_tier4_invariant(feats: torch.Tensor, P_raw: torch.Tensor, cfg: Tier5Config,
                           gt_list: list[dict]) -> None:
    # HARD INVARIANT — all-off StageAPrime must reproduce tier4's Stage A on real data.
    # Fails the run loudly before any GPU-hour is spent on a scorer that drifted.
    pi, ni, free = query_indices(gt_list[0]["query"])
    srcs = [int(s) for s in gt_list[0]["ground_truth"].keys()][:8]
    want = make_stage_a_chunk(feats, P_raw, cfg.tier4)(srcs, pi, ni, free)
    got = StageAPrime(feats, P_raw, tree=None, conditioned_tail=False, lam=0.0,
                      hamming_max=cfg.tier4.hamming_max).chunk(srcs, pi, ni, free)
    diff = (want - got).abs().max().item()
    assert diff < 1e-4, f"all-off scorer deviates from tier4 Stage A: {diff:.2e}"
    log(f"  hard invariant OK: max |all-off - tier4 Stage A| = {diff:.2e}")


# ---------------------------------------------------------------------------
# C5 — QB-Norm (Dynamic Inverted Softmax, Bogolin et al. CVPR 2022)
# ---------------------------------------------------------------------------

def qbnorm_rescore(scores: torch.Tensor, bank: torch.Tensor, beta: float) -> torch.Tensor:
    # DIS in log space: ŝ(x) = β·s(x) − logΣ_b exp(β·B[b,x]), applied ONLY to queries whose
    # raw top-1 lies in the bank's activation set 𝒜 = {argmax_x B[b,·]} — other queries pass
    # through untouched (the "dynamic" part that makes IS safe when no hub is involved).
    denom = torch.logsumexp(beta * bank, dim=0)                         # [N]
    activated = torch.isin(scores.argmax(dim=1), bank.argmax(dim=1))    # [S]
    return torch.where(activated.unsqueeze(1), beta * scores - denom, scores)


def sample_constraint_queries(gen: QueryGenerator, n: int) -> list[tuple]:
    # n synthetic TRAIN queries as (ref_idx, pi, ni, free) — the shared raw material of
    # λ/β tuning and the querybank. Constraint sets + refs only; positives are recomputed
    # per use against whatever gallery applies.
    out = []
    attempts = 0
    while len(out) < n and attempts < 50 * n:
        attempts += 1
        sample = gen.sample_query()
        if sample is None:
            continue
        r, pos_names, neg_names, _, _ = sample
        pi = [ATTR_TO_IDX[a] for a in pos_names]
        ni = [ATTR_TO_IDX[b] for b in neg_names]
        free = [c for c in range(N_ATTR) if c not in set(pi) | set(ni)]
        out.append((r, pi, ni, free))
    if len(out) < n:
        raise RuntimeError(f"QueryGenerator produced only {len(out)}/{n} usable queries")
    return out


def build_querybank(scorer: StageAPrime, queries: list[tuple], train_feats: torch.Tensor,
                    P_train: torch.Tensor) -> torch.Tensor:
    # Bank of synthetic-TRAIN-query score vectors over the scorer's gallery, built once per
    # scorer config (bank scores must come from the same scoring function they normalize).
    # Refs are TRAIN images, so no test information enters the bank.
    rows = [scorer.score(train_feats[r].unsqueeze(0), P_train[r].unsqueeze(0), pi, ni, free)
            for r, pi, ni, free in queries]
    return torch.cat(rows, dim=0)                                       # [K, N]


def _with_qbnorm(scorer: StageAPrime, bank: torch.Tensor, beta: float):
    # C5 eval wrapper — self-exclusion is applied BEFORE the DIS gate so a source can never
    # count as its own top-1 hub (the driver re-applies it afterwards; idempotent).
    def score_chunk(srcs: list[int], pi: list[int], ni: list[int],
                    free: list[int]) -> torch.Tensor:
        s = scorer.chunk(srcs, pi, ni, free)
        rows = torch.arange(len(srcs), device=s.device)
        s[rows, torch.as_tensor(srcs, device=s.device)] = float("-inf")
        return qbnorm_rescore(s, bank, beta)

    return score_chunk


# ---------------------------------------------------------------------------
# C4/C5 tuning — synthetic TRAIN queries against a TRAIN gallery (zero test usage)
# ---------------------------------------------------------------------------
class TrainTuner:
    # Grid-search harness for the two training-free scalars. Builds one TRAIN gallery and
    # one set of synthetic queries with §3.1.1-rule positives, precomputes each query's
    # A′-core base scores and cosine row ONCE, then sweeps λ / β at zero marginal cost
    # (both transforms are cheap functions of the cached rows). Hit@5 is the criterion —
    # R@5 is the project's primary headline metric.
    def __init__(self, cfg: Tier5Config, tree: DependencyTree, train_feats: torch.Tensor,
                 P_train: torch.Tensor, train_attrs: torch.Tensor, dev: torch.device):
        g = torch.Generator().manual_seed(cfg.tune_seed)
        n_train = train_feats.shape[0]
        self.gal_ids = torch.randperm(n_train, generator=g)[:min(cfg.tune_gallery, n_train)]
        gal_pos = torch.full((n_train,), -1, dtype=torch.long)
        gal_pos[self.gal_ids] = torch.arange(self.gal_ids.numel())
        A_gal = train_attrs[self.gal_ids].float()

        self.scorer = StageAPrime(train_feats[self.gal_ids.to(train_feats.device)],
                                  P_train[self.gal_ids.to(P_train.device)], tree,
                                  conditioned_tail=True, lam=0.0,
                                  hamming_max=cfg.tier4.hamming_max)

        gen = QueryGenerator(train_attrs, replace(cfg.tier4.phi_gen_config(), seed=cfg.tune_seed))
        raw = sample_constraint_queries(gen, cfg.tune_queries)
        self.base, self.cos, self.pos_masks = [], [], []
        for r, pi, ni, free in raw:
            pos = self._gallery_positives(A_gal, train_attrs[r].float(), pi, ni, free,
                                          cfg.tier4.hamming_max)
            if gal_pos[r] >= 0:
                pos[gal_pos[r]] = False                                 # never its own target
            if not pos.any():
                continue
            s = self.scorer.score(train_feats[r].unsqueeze(0), P_train[r].unsqueeze(0),
                                  pi, ni, free).squeeze(0)
            if gal_pos[r] >= 0:
                s[gal_pos[r]] = float("-inf")                           # §5 self-exclusion
            self.base.append(s)
            self.cos.append(train_feats[r] @ self.scorer.feats.T)
            self.pos_masks.append(pos.to(s.device))
        if not self.base:
            raise RuntimeError("no tuning query kept any positive inside the gallery")
        log(f"  tuner ready: {len(self.base)}/{cfg.tune_queries} queries with in-gallery positives")

        bank_gen = QueryGenerator(train_attrs,
                                  replace(cfg.tier4.phi_gen_config(), seed=cfg.querybank_seed))
        bank_q = sample_constraint_queries(bank_gen, cfg.querybank_size)
        self.bank = build_querybank(self.scorer, bank_q, train_feats, P_train)

    @staticmethod
    def _gallery_positives(A_gal: torch.Tensor, a_ref: torch.Tensor, pi: list[int],
                           ni: list[int], free: list[int], hmax: int) -> torch.Tensor:
        # GT rule §3.1.1 on TRAIN labels: strict T⁺/T⁻ satisfaction + Hamming ≤ hmax on the
        # free attributes vs the reference.
        sat = torch.ones(A_gal.shape[0], dtype=torch.bool)
        for a in pi:
            sat &= A_gal[:, a] > 0.5
        for b in ni:
            sat &= A_gal[:, b] < 0.5
        ham = (A_gal[:, free] != a_ref[free]).sum(dim=1)
        return sat & (ham <= hmax)

    def _hit_at_5(self, s: torch.Tensor, pos: torch.Tensor) -> float:
        return float(pos[torch.topk(s, 5).indices].any())

    def sweep_lambda(self, grid: tuple[float, ...]) -> float:
        # C4 tuning — ties break toward the smaller λ (least intervention on a proven floor).
        best_lam, best_hit = grid[0], -1.0
        for lam in grid:
            hit = sum(self._hit_at_5(b + lam * c, p)
                      for b, c, p in zip(self.base, self.cos, self.pos_masks)) / len(self.base)
            log(f"  lambda={lam:<6g} train hit@5={hit:.4f}")
            if hit > best_hit:
                best_lam, best_hit = lam, hit
        return best_lam

    def sweep_beta(self, grid: tuple[float, ...]) -> float:
        # C5 tuning — DIS temperature β against the TRAIN-gallery bank; same tie-break rule.
        best_beta, best_hit = grid[0], -1.0
        for beta in grid:
            hit = sum(self._hit_at_5(qbnorm_rescore(b.unsqueeze(0), self.bank, beta).squeeze(0), p)
                      for b, p in zip(self.base, self.pos_masks)) / len(self.base)
            log(f"  beta={beta:<6g} train hit@5={hit:.4f}")
            if hit > best_hit:
                best_beta, best_hit = beta, hit
        return best_beta


# ---------------------------------------------------------------------------
# Stage B′ — LambdaLoss Φ on the A′-core floor
# ---------------------------------------------------------------------------

def lambda_loss(scores: torch.Tensor, n_pos: int, sigma: float) -> torch.Tensor:
    # Stage B′ loss — LambdaLoss, NDCG-Loss form (Wang et al., CIKM 2018 §5):
    # L = Σ_{i∈pos, j∈neg} |ΔNDCG_ij|·log(1+e^{−σ(s_i−s_j)}), |ΔNDCG_ij| = |G_i−G_j|·|D_i−D_j|
    # with binary gains G = y/maxDCG and discounts D_r = 1/log₂(1+r) at the CURRENT ranks
    # (detached). Top-of-list mistakes carry the largest weight — the designed cure for
    # ListNet's flat target mass (failure iii: the measured R@1→R@5 shift).
    C = scores.numel()
    order = scores.detach().argsort(descending=True)
    ranks = torch.empty(C, device=scores.device)
    ranks[order] = torch.arange(1, C + 1, dtype=ranks.dtype, device=scores.device)
    D = 1.0 / torch.log2(1.0 + ranks)
    max_dcg = (1.0 / torch.log2(torch.arange(2, n_pos + 2, dtype=ranks.dtype,
                                             device=scores.device))).sum()
    g = 1.0 / max_dcg                                                   # |G_pos − G_neg|, binary gains
    diff = scores[:n_pos].unsqueeze(1) - scores[n_pos:].unsqueeze(0)    # [P, Q]
    w = g * (D[:n_pos].unsqueeze(1) - D[n_pos:].unsqueeze(0)).abs()
    return (w * F.softplus(-sigma * diff)).sum()


def _phi_composite(w: dict[str, torch.Tensor], feats: torch.Tensor, P: torch.Tensor,
                   logP: torch.Tensor, log1mP: torch.Tensor, constraint: torch.Tensor,
                   tail: torch.Tensor, mh_pos: torch.Tensor, mh_neg: torch.Tensor,
                   use_qv: bool) -> torch.Tensor:
    # B′ score = A′-core base + Φ residual channels:
    #   S = constraint + Δw⁺·log p + Δw⁻·log(1−p) + w_p·p [+ q_v·v] + κ·tail_cond.
    # BayesPhi returns w⁺ = mh⁺ + head(h); subtracting mh keeps only the learned Δ, so the
    # tree constraint is the base instead of the independent product — zero heads + κ=1
    # ⇒ S ≡ Stage A′ core exactly (the init invariant, exact because 0-weight matmuls are 0).
    s = (constraint.unsqueeze(0)
         + (w["w_pos"] - mh_pos) @ logP.T
         + (w["w_neg"] - mh_neg) @ log1mP.T
         + w["w_p"] @ P.T
         + w["kappa"] * tail)
    if use_qv:
        s = s + w["q_v"] @ feats.T
    return s


def train_phi_prime(cfg: Tier5Config, probe: nn.Linear, tau: torch.Tensor | None,
                    tree: DependencyTree, dev: torch.device, loss_name: str, use_qv: bool,
                    seed: int, max_steps: int | None = None) -> tuple[BayesPhi, list[float]]:
    # Stage B′ trainer — tier4's protocol (QueryGenerator candidates, AdamW+cosine, clip 5.0)
    # with the A′-core base score and a pluggable loss. Base terms (tree constraint,
    # conditioned tail) are fixed channel values (no_grad), exactly like tier4's tail.
    t4 = replace(cfg.tier4, seed=seed)
    torch.manual_seed(seed)
    train_feats = _load_train_features().to(dev)
    train_attrs = _load_train_attributes()
    P_tr = calibrated_posteriors(probe, train_feats, tau)
    logP_tr, log1mP_tr = P_tr.log(), (1 - P_tr).log()
    log_u_tr = evidence_log_ratio(P_tr, tree.log_marginal)
    log_z_free_tr = tree_log_partition(tree, log_u_tr)

    loss_fn = {"listnet": lambda s, n: listnet_scores(s, n, t4.tau_list),
               "lambdaloss": lambda s, n: lambda_loss(s, n, cfg.sigma_lambda)}[loss_name]

    model = BayesPhi(t4).to(dev)
    gen = QueryGenerator(train_attrs, t4.phi_gen_config())
    opt = torch.optim.AdamW(model.parameters(), lr=t4.lr, weight_decay=t4.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t4.epochs)

    steps_per_epoch = max(1, gen.N // (t4.batch_queries * t4.steps_div))
    if max_steps is not None:
        steps_per_epoch = min(steps_per_epoch, max_steps)
    total_steps = t4.epochs * steps_per_epoch
    log(f"B' [{loss_name}{'' if use_qv else ' no-qv'} seed={seed}]: "
        f"{sum(p.numel() for p in model.parameters()):,} params, "
        f"{t4.epochs} epochs x {steps_per_epoch} steps x {t4.batch_queries} queries")

    curve, step, t0 = [], 0, time.time()
    for epoch in range(t4.epochs):
        model.train()
        running, seen = 0.0, 0
        for _ in range(steps_per_epoch):
            opt.zero_grad()
            batch_loss, n = torch.zeros((), device=dev), 0
            for _ in range(t4.batch_queries):
                sample = gen.sample_query()
                if sample is None:
                    continue
                r, pos_names, neg_names, pos_ids, hard_ids = sample
                pi = [ATTR_TO_IDX[a] for a in pos_names]
                ni = [ATTR_TO_IDX[b] for b in neg_names]
                free = [c for c in range(N_ATTR) if c not in set(pi) | set(ni)]

                rand_ids = torch.randint(gen.N, (t4.n_rand_neg,), generator=gen.g)
                cand = torch.cat([pos_ids, hard_ids, rand_ids]).to(dev)

                mh_pos = torch.zeros(1, N_ATTR, device=dev)
                mh_neg = torch.zeros(1, N_ATTR, device=dev)
                mh_pos[0, pi], mh_neg[0, ni] = 1.0, 1.0
                w = model(train_feats[r].unsqueeze(0), P_tr[r].unsqueeze(0), mh_pos, mh_neg)

                with torch.no_grad():                          # base terms are fixed channels
                    u_c = log_u_tr[cand]
                    constraint = tree_constraint_log(tree, u_c, log_z_free_tr[cand], pi, ni)
                    marg = tree_marginals(tree, clamp_evidence(u_c, pi, ni))
                    marg = marg[:, free].clamp(EPS, 1 - EPS)
                    tail = hamming_tail_log(P_tr[r].unsqueeze(0)[:, free], marg, t4.hamming_max)

                scores = _phi_composite(w, train_feats[cand], P_tr[cand], logP_tr[cand],
                                        log1mP_tr[cand], constraint, tail,
                                        mh_pos, mh_neg, use_qv).squeeze(0)
                batch_loss = batch_loss + loss_fn(scores, pos_ids.numel())
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
        log(f"  epoch {epoch + 1:02d}/{t4.epochs}  {loss_name}={curve[-1]:.4f}  "
            f"kappa={model.kappa.item():.3f}  eta={eta:.1f}min")
    return model, curve


def make_stage_b_prime_chunk(model: BayesPhi, scorer: StageAPrime, use_qv: bool):
    # B′ chunked eval scorer over the A′-core scorer's frozen channels (scorer must be the
    # tree+conditioned-tail, λ=0 instance — the exact base Φ was trained on).
    model.eval()

    @torch.no_grad()
    def score_chunk(srcs: list[int], pi: list[int], ni: list[int],
                    free: list[int]) -> torch.Tensor:
        constraint, marg = scorer.query_terms(pi, ni, free)
        S = len(srcs)
        mh_pos = torch.zeros(S, N_ATTR, device=scorer.feats.device)
        mh_neg = torch.zeros(S, N_ATTR, device=scorer.feats.device)
        mh_pos[:, pi], mh_neg[:, ni] = 1.0, 1.0
        w = model(scorer.feats[srcs], scorer.P[srcs], mh_pos, mh_neg)
        tail = hamming_tail_log(scorer.P[srcs][:, free], marg, scorer.hamming_max)
        return _phi_composite(w, scorer.feats, scorer.P, scorer.logP, scorer.log1mP,
                              constraint, tail, mh_pos, mh_neg, use_qv)

    return score_chunk


@torch.no_grad()
def init_sanity_prime(model: BayesPhi, scorer: StageAPrime, query_str: str, src: int) -> float:
    # Guard the B′ invariant: Φ at init must reproduce Stage A′ core (tier4 pattern).
    pi, ni, free = query_indices(query_str)
    a = scorer.chunk([src], pi, ni, free)
    b = make_stage_b_prime_chunk(model, scorer, use_qv=True)([src], pi, ni, free)
    return (a - b).abs().max().item()


# ---------------------------------------------------------------------------
# Ablation matrix
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AblationRow:
    """One Stage-A′ ablation arm; use_lambda/use_qbnorm resolve to tuned scalars at runtime."""
    name: str
    calibrated: bool = False
    tree: bool = False
    conditioned_tail: bool = False
    use_lambda: bool = False
    use_qbnorm: bool = False


@dataclass(frozen=True)
class PhiRow:
    """One Stage-B′ arm. multi_seed=False rows run one seed — the budget-driven default for
    the ListNet control (its job is direction, not variance; NextStep.md IV.1 seeds apply
    to the headline LambdaLoss rows)."""
    name: str
    loss: str
    use_qv: bool
    multi_seed: bool


STAGE_A_ROWS = (
    AblationRow("A_tier4repro"),
    AblationRow("A_C1", calibrated=True),
    AblationRow("A_C1C2", calibrated=True, tree=True),
    AblationRow("A_C1C2C3", calibrated=True, tree=True, conditioned_tail=True),
    AblationRow("Aprime_C4", calibrated=True, tree=True, conditioned_tail=True,
                use_lambda=True),
    AblationRow("Aprime_C5", calibrated=True, tree=True, conditioned_tail=True,
                use_qbnorm=True),
    AblationRow("Aprime_C4C5", calibrated=True, tree=True, conditioned_tail=True,
                use_lambda=True, use_qbnorm=True),
)

PHI_ROWS = (
    PhiRow("Bprime_listnet", "listnet", use_qv=True, multi_seed=False),
    PhiRow("Bprime_lambdaloss", "lambdaloss", use_qv=True, multi_seed=True),
    PhiRow("Bprime_lambdaloss_noqv", "lambdaloss", use_qv=False, multi_seed=True),
)


def _summary_line(results: dict) -> dict[str, float]:
    # Macro means over unique queries (dict keying dedups the doubled `-Young` entry).
    n = len(results)
    return {f"R@{k}": sum(r[f"recall@{k}"] for r in results.values()) / n for k in (1, 5, 10)}


def _chunk_get_ranking(chunk_fn, query_str: str):
    # Adapt a chunked scorer to the single-source get_ranking seam (CONTRACT §5) so the
    # frozen-DB tripwire can exercise it.
    pi, ni, free = query_indices(query_str)

    @torch.no_grad()
    def get_ranking(src_idx: int) -> list[int]:
        s = chunk_fn([src_idx], pi, ni, free).squeeze(0)
        s[src_idx] = float("-inf")
        return torch.argsort(s, descending=True).tolist()

    return get_ranking


# ---------------------------------------------------------------------------
# Results bundle
# ---------------------------------------------------------------------------

def bundle_tier5_results(out_dir: Path) -> Path:
    # Tier-5 bundle: all ablation CSVs/weights/logs + the three offline artifacts, with
    # repo-root-relative arcnames so extraction restores any clone in place.
    root = Path(__file__).parent.parent
    art = _get_artifacts_dir()
    return bundle_results(
        root, root / "output" / "tier5_cdbayes_bundle.tar.gz",
        [out_dir, art / "tier4_probe.pt", art / "tier5_temperatures.pt",
         art / "tier5_chowliu.pt"],
        pull_hints=[
            "Pull it back to your local clone with (from local repo root):",
            "  powershell scripts/pull_tier5_cdbayes_results.ps1",
            "  # or manually:",
            "  scp -P 5037 disi@lab-d2fdded3-9ba4-4188-80fa-16c27a4cd022.westeurope.cloudapp.azure.com:"
            "~/DLproject_VM/Deep-Learning-Project/output/tier5_cdbayes_bundle.tar.gz .",
            "  tar -xzf tier5_cdbayes_bundle.tar.gz   # extracts output/tier5_cdbayes/* in place",
        ])


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Tier-5 CD-Bayes: C1-C5 + Stage B' with a built-in ablation matrix")
    all_rows = [r.name for r in STAGE_A_ROWS] + [r.name for r in PHI_ROWS]
    ap.add_argument("--quick", action="store_true", help="CPU smoke run (tiny everything)")
    ap.add_argument("--rows", type=str, default=None,
                    help=f"comma-separated subset of: {','.join(all_rows)}")
    ap.add_argument("--seeds", type=int, default=None, help="seeds per multi-seed B' row (default 3)")
    ap.add_argument("--full-seeds", action="store_true",
                    help="run every B' row (incl. the ListNet control) with all seeds")
    ap.add_argument("--epochs", type=int, default=None, help="override B' epochs")
    ap.add_argument("--device", type=str, default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--dump-per-source", action="store_true",
                    help="per-source outcome dumps for the published champions "
                         "(Aprime_C4C5 + Bprime_lambdaloss seeds) from cached artifacts "
                         "and saved checkpoints — no tuning, no training")
    ap.add_argument("--force-probes", action="store_true", help="retrain probes even if cached")
    ap.add_argument("--force-artifacts", action="store_true",
                    help="refit temperatures + rebuild the Chow-Liu tree even if cached")
    return ap.parse_args()


def _select_rows(rows_arg: str | None) -> tuple[list[AblationRow], list[PhiRow]]:
    if rows_arg is None:
        return list(STAGE_A_ROWS), list(PHI_ROWS)
    wanted = {r.strip() for r in rows_arg.split(",") if r.strip()}
    known = {r.name for r in STAGE_A_ROWS} | {r.name for r in PHI_ROWS}
    unknown = wanted - known
    if unknown:
        raise SystemExit(f"unknown --rows {sorted(unknown)}; valid: {sorted(known)}")
    return ([r for r in STAGE_A_ROWS if r.name in wanted],
            [r for r in PHI_ROWS if r.name in wanted])


def dump_per_source_champions(cfg: Tier5Config, dev: torch.device, out_dir: Path) -> None:
    # Per-source dumps of the PUBLISHED tier-5 champions for the statistical suite:
    # Stage A′ full (Aprime_C4C5, training-free) + Bprime_lambdaloss seeds 0–2 (learned).
    # Everything is loaded, nothing re-fit — tuned λ/β come from the saved JSON, Φ from the
    # saved checkpoints — so each dump is parity-gated against its published CSV.
    from per_source import check_parity_with_csv, save_per_source_npz

    probe = load_or_train_probes(cfg.tier4, dev, force=False)
    tau = load_or_fit_temperatures(probe, cfg, dev, force=False)
    tree = load_or_build_tree(cfg, force=False).to(dev)
    feats = load_image_features().to(dev)
    P_cal = calibrated_posteriors(probe, feats, tau)
    gt_list = load_eval_json(find_eval_json())

    with open(out_dir / "tier5_tuned_scalars_full.json", encoding="utf-8") as f:
        scalars = json.load(f)
    lam, beta = scalars["lambda"], scalars["beta"]
    log(f"Loaded tuned scalars: lambda={lam} beta={beta}")

    train_feats = _load_train_features().to(dev)
    train_attrs = _load_train_attributes()
    P_train = calibrated_posteriors(probe, train_feats, tau)

    def run(name: str, chunk_fn, csv_name: str) -> None:
        dump: dict = {}
        evaluate_scorer(name, chunk_fn, gt_list, cfg.tier4.chunk, dump=dump)
        check_parity_with_csv(dump, out_dir / csv_name)
        save_per_source_npz(f"tier5_{name}", dump)

    # Stage A′ champion — same construction as the ablation loop's Aprime_C4C5 arm.
    scorer = StageAPrime(feats, P_cal, tree, conditioned_tail=True,
                         lam=lam, hamming_max=cfg.tier4.hamming_max)
    gen = QueryGenerator(train_attrs,
                         replace(cfg.tier4.phi_gen_config(), seed=cfg.querybank_seed))
    bank = build_querybank(scorer, sample_constraint_queries(gen, cfg.querybank_size),
                           train_feats, P_train)
    run("Aprime_C4C5", _with_qbnorm(scorer, bank, beta), "tier5_Aprime_C4C5_full.csv")

    # Stage B′ — saved checkpoints over the λ=0 core they were trained on. The ListNet
    # seed-0 run is the overall CHAMPION (summary CSV: R@10 0.4128); the LambdaLoss seeds
    # are its multi-seed control family. Missing checkpoints are skipped with a warning
    # so a partial local pull still dumps everything it can.
    core = StageAPrime(feats, P_cal, tree, conditioned_tail=True, lam=0.0,
                       hamming_max=cfg.tier4.hamming_max)
    stems = [("Bprime_listnet_seed0", "tier5_Bprime_listnet_seed0")] + \
            [(f"Bprime_lambdaloss_seed{s}", f"tier5_Bprime_lambdaloss_seed{s}")
             for s in cfg.seeds]
    for name, stem in stems:
        ckpt = out_dir / f"{stem}.pt"
        if not ckpt.exists():
            log(f"WARNING: {ckpt.name} not found — skipping {name} dump")
            continue
        model = BayesPhi(cfg.tier4).to(dev)
        model.load_state_dict(torch.load(ckpt, map_location=dev))
        run(name, make_stage_b_prime_chunk(model, core, use_qv=True), f"{stem}_full.csv")


def main() -> None:
    args = _parse_args()
    cfg = Tier5Config()
    if args.epochs is not None:
        cfg = replace(cfg, tier4=replace(cfg.tier4, epochs=args.epochs))
    if args.seeds is not None:
        cfg = replace(cfg, seeds=tuple(range(args.seeds)))
    if args.quick:
        cfg = replace(cfg, tier4=replace(cfg.tier4, probe_epochs=2, epochs=2, chunk=64),
                      calib_steps=50, tune_queries=12, tune_gallery=2000,
                      querybank_size=8, seeds=(0,))

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = output_subdir("tier5_cdbayes")
    if args.dump_per_source:
        dump_per_source_champions(cfg, dev, out_dir)
        return
    set_log_file(out_dir / "train_log.txt")
    tag = "quick" if args.quick else "full"
    a_rows, phi_rows = _select_rows(args.rows)
    log(f"Device: {dev}" + (f"  GPU: {torch.cuda.get_device_name(0)}" if dev.type == "cuda" else ""))
    log(f"Rows: {[r.name for r in a_rows] + [r.name for r in phi_rows]}")

    # --- Offline artifacts: probes (tier4), temperatures (C1), tree (C2) ---------------
    probe = load_or_train_probes(cfg.tier4, dev, force=args.force_probes)
    tau = load_or_fit_temperatures(probe, cfg, dev, force=args.force_artifacts)
    tree = load_or_build_tree(cfg, force=args.force_artifacts).to(dev)

    feats = load_image_features().to(dev)
    P_raw = calibrated_posteriors(probe, feats, None)
    P_cal = calibrated_posteriors(probe, feats, tau)

    gt_list = load_eval_json(find_eval_json())
    if args.quick:
        gt_list = gt_list[:2]

    log("Hard invariant — all-off scorer vs tier4 Stage A")
    verify_tier4_invariant(feats, P_raw, cfg, gt_list)

    # --- TRAIN-side tuning of λ / β (only when a selected row needs them) ----------------
    need_lambda = any(r.use_lambda for r in a_rows)
    need_qbnorm = any(r.use_qbnorm for r in a_rows)
    lam, beta = 0.0, 1.0
    train_feats = train_attrs = P_train = None
    if need_lambda or need_qbnorm:
        train_feats = _load_train_features().to(dev)
        train_attrs = _load_train_attributes()
        P_train = calibrated_posteriors(probe, train_feats, tau)
        log("Tuning training-free scalars on synthetic TRAIN queries...")
        tuner = TrainTuner(cfg, tree, train_feats, P_train, train_attrs, dev)
        if need_lambda:
            lam = tuner.sweep_lambda(cfg.lambda_grid)
            log(f"  tuned lambda = {lam}")
        if need_qbnorm:
            beta = tuner.sweep_beta(cfg.beta_grid)
            log(f"  tuned beta = {beta}")
        with open(out_dir / f"tier5_tuned_scalars_{tag}.json", "w", encoding="utf-8") as f:
            json.dump({"lambda": lam, "beta": beta}, f, indent=2)

    # --- Stage A′ ablation rows ----------------------------------------------------------
    summary: dict[str, dict[str, float]] = {}
    bank_queries = None
    for i, row in enumerate(a_rows):
        log(f"Ablation [{row.name}] "
            f"C1={row.calibrated} C2={row.tree} C3={row.conditioned_tail} "
            f"lam={lam if row.use_lambda else 0} qbnorm={row.use_qbnorm}")
        scorer = StageAPrime(feats, P_cal if row.calibrated else P_raw,
                             tree if row.tree else None, row.conditioned_tail,
                             lam if row.use_lambda else 0.0, cfg.tier4.hamming_max)
        chunk_fn = scorer.chunk
        if row.use_qbnorm:
            if bank_queries is None:
                gen = QueryGenerator(train_attrs,
                                     replace(cfg.tier4.phi_gen_config(), seed=cfg.querybank_seed))
                bank_queries = sample_constraint_queries(gen, cfg.querybank_size)
            bank = build_querybank(scorer, bank_queries, train_feats, P_train)
            chunk_fn = _with_qbnorm(scorer, bank, beta)
        if i == 0:
            assert_db_frozen(feats, _chunk_get_ranking(chunk_fn, gt_list[0]["query"]))
        results = evaluate_scorer(row.name, chunk_fn, gt_list, cfg.tier4.chunk)
        print("\n" + format_results_table(results))
        save_results_csv(results, out_dir / f"tier5_{row.name}_{tag}.csv")
        summary[row.name] = _summary_line(results)

    # --- Stage B′ rows -------------------------------------------------------------------
    if phi_rows:
        core = StageAPrime(feats, P_cal, tree, conditioned_tail=True, lam=0.0,
                           hamming_max=cfg.tier4.hamming_max)
        fresh = BayesPhi(cfg.tier4).to(dev)
        diff = init_sanity_prime(fresh, core, gt_list[0]["query"],
                                 next(int(s) for s in gt_list[0]["ground_truth"].keys()))
        log(f"  B' init sanity: max |Phi_init - StageA'| = {diff:.2e}")
        assert diff < 1e-4, "Phi at init must reproduce Stage A' core"

        for row in phi_rows:
            seeds = cfg.seeds if (row.multi_seed or args.full_seeds) else cfg.seeds[:1]
            for seed in seeds:
                model, curve = train_phi_prime(cfg, probe, tau, tree, dev, row.loss,
                                               row.use_qv, seed,
                                               max_steps=5 if args.quick else None)
                stem = f"tier5_{row.name}_seed{seed}"
                torch.save(model.state_dict(), out_dir / f"{stem}.pt")
                save_training_curve(curve, out_dir / f"{stem}_curve.png",
                                    f"{row.name} seed {seed}", f"{row.loss} loss")
                results = evaluate_scorer(f"{row.name}/s{seed}",
                                          make_stage_b_prime_chunk(model, core, row.use_qv),
                                          gt_list, cfg.tier4.chunk)
                print("\n" + format_results_table(results))
                save_results_csv(results, out_dir / f"{stem}_{tag}.csv")
                summary[f"{row.name}_seed{seed}"] = _summary_line(results)

    # --- Summary + bundle ----------------------------------------------------------------
    with open(out_dir / f"tier5_ablation_summary_{tag}.csv", "w", encoding="utf-8") as f:
        f.write("row,R@1,R@5,R@10\n")
        for name, m in summary.items():
            f.write(f"{name},{m['R@1']:.4f},{m['R@5']:.4f},{m['R@10']:.4f}\n")
            log(f"SUMMARY  {name:<28s} R@1={m['R@1']:.4f}  R@5={m['R@5']:.4f}  R@10={m['R@10']:.4f}")
    log("Baselines — tier4 Stage A R@5 0.2596 | Stage B R@5 0.2811 (NextStep.md Part I)")
    bundle_tier5_results(out_dir)


if __name__ == "__main__":
    main()
