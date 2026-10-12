# Negation-Aware Fusion Models — Design Report

**Date:** 2026-07-01
**Author:** Alfonso Antognozzi
**Status:** Implemented, smoke-validated (CPU), pending VM training run
**Companion plan:** `C:\Users\alfon\.claude\plans\analyze-the-documents-project-specificat-moonlit-bengio.md`

---

## 1. Motivation — the negation floor

Our best fusion model, **Tier-3 Combined** (`output/tier3_combined/tier3_combined.csv`), wins on
overall retrieval but **collapses on negation**:

| Query | R@1 | R@5 | R@10 |
|---|---|---|---|
| MEAN (14 queries) | 0.0563 | 0.1576 | 0.2329 |
| `-Male, -Mustache` | **0.0000** | **0.0000** | **0.0000** |
| `-Heavy_Makeup` | 0.0120 | 0.0448 | 0.0705 |
| `-Young` | 0.0067 | 0.0271 | 0.0463 |
| `+Wearing_Lipstick, -Heavy_Makeup, +Smiling` | 0.0294 | 0.0882 | 0.0882 |

`-Male, -Mustache` scores exactly **0.000 on every geometry tier** (Tier-1 CLAY/GDE, Tier-2a/b/c/d,
Tier-3 DGP/3c/Combined, Tier-3 SymNeg). The task specification (§3.1) makes these mandatory queries,
so this is the single biggest gap between our system and a strong submission.

### 1.1 Root cause (verified against code + all tier CSVs)

Every tier from GDE onward negates the **same** way: it builds the positive query in the tangent
space at the global mean μ, anchored to the reference identity, then applies a **fixed rank-1
orthogonal rejection** of one averaged attribute axis (Oldfield et al. "kill" operator; Alhamoud et
al.):

```
q_tan = Log_μ(v_ref) + Σ_a α·d_a          # positive / identity composition
q_tan ← q_tan − (q_tan · d_b) · d_b        # fixed negation, coefficient always 1.0
q     = normalize(Exp_μ(q_tan))
```

Three structural facts make this a dead end under the frozen-DB constraint:

1. **Rejection barely moves an identity-anchored ranking.** `q_tan` starts at `Log_μ(v_ref)`, which
   pins the query to the reference identity and dominates the norm. Removing one 512-D axis leaves
   the cosine order almost unchanged — the DB is scored on *what is left*, not on *not having b*.
2. **CLIP affirmation bias + attribute entanglement.** `t_-Male ≈ t_+Male` up to sign, and
   `Male ⟂ Mustache` is false in CLIP (cos ≈ 0.83). A single subtracted axis cannot express "any
   value except b."
3. **The frozen closed-form negation is exhausted.** `tier3_symneg` (CLAY symmetric complement on
   both query and DB, training-free) and the learned `NegationHead` (λ-scaled rejection, folded into
   Combined) both still give `-Male,-Mustache` = 0.

### 1.2 The two signals that *did* move negation

Scanning every tier's per-query CSV, only two methods ever broke the `-Male,-Mustache` floor — and
each did it at a cost:

| Method | `-Male,-Mustache` R@10 | `-Young` R@5 | MEAN R@1 | Cost |
|---|---|---|---|---|
| **Tier-3 Composer** (fully learned token transformer) | **0.1111** | 0.0904 | 0.0052 | overall collapse (under-trained from scratch) |
| **Tier-4 Hybrid** (discriminative attribute probes as a penalty) | 0.0370 | 0.0428 | 0.0426 | MEAN below Combined; single global β; zeros the lipstick-composite |

**The lesson:** a *learned* negation operator can express what the fixed
arithmetic-with-rejection skeleton cannot. The two failed attempts point straight at the fix — keep
the strong positive fusion, make negation a first-class *trained* operation, and supervise it with
the ranking metric.

---

## 2. Design principles

All three proposals share the same constraints (confirmed with the team and traced to the spec):

- **Frozen DB, spec-faithful (§3.1/§3.2).** CelebA CLIP image features are constructed once and
  never re-encoded or per-query re-projected at scoring. Negation lives in the **query/scoring
  math**. Per-image side-features (attribute-probe logits) may be **precomputed once offline and
  frozen** — this is still "built once and kept frozen" and does not touch the CLIP DB vectors. A
  `assert_db_frozen` tripwire (in `src/tier3_train_utils.py`) checks the DB tensor is byte-identical
  before and after each query.
- **Lightweight, fast iteration.** One lightweight model (<2M params) + two medium (≤10M). No
  from-scratch deep transformer — the composer already proved that under-trains in 30 epochs.
- **Reused foundations (no re-implementation).** Geometry (`src/manifold.py`: log/exp maps,
  intrinsic mean, CLAY `build_subspace`), the proven positive fusion Φ (`src/tier3_dgp.py`), the
  frozen artifacts, `QueryGenerator` (synthetic train-split queries, no GT leakage), `listnet_loss`,
  and the eval seam are all shared. The hard-neg-mining and disentanglement helpers were extracted
  into `src/tier3_train_utils.py` so `tier3_combined` and the new tiers depend on **one** owner.
- **Pareto reporting.** The three models are deliberately placed at different points: P1 maximizes
  overall MEAN, P2 maximizes negation, P3 balances. We report the trade-off explicitly.

> **Repo state fixed en route.** Commit `0d254bf` ("Removed useless tier3 scripts") had deleted
> `tier3_negation.py`, `tier3_contrastive.py`, and `tier3_hybrid.py`, which `tier3_combined.py`
> still imports — so the best model would not run from the working tree. We restored the three
> modules from `git show 2a4c3fc:…` and re-verified `tier3_combined` imports and reuses
> `listnet_loss`.

---

## 3. The three proposals

### 3.1 P1 — NegGate Φ (lightweight, 1.51M params)

**File:** `src/tier3_neggate.py` · **Pareto role:** maximize overall MEAN without regressing negation.

Keep the entire Tier-3 Combined positive pipeline and ListNet objective, but replace the single
fixed rejection with a **reference/query-conditioned gated rejection of a small CLAY negation
subspace**:

```
B_b   = center(T_b)                                        # centered paraphrase directions  [k, d]
a_b   = softmax( (W_q h_ref)·(W_k B_b)ᵀ / √d_gate )       # low-rank gate: which dims matter [k]
C_b   = orthonormal( a_b ⊙ B_b )   via QR                 # gated basis                       [d, r]
λ_b   = 2·sigmoid( g([q_tan, ĉ_b, q_tan·ĉ_b]) ) ∈ (0,2)  # learned suppression strength
q_tan ← q_tan − λ_b · C_b (C_bᵀ q_tan)                    # gated subspace rejection
```

**Why it can beat Combined where the plain `NegationHead` could not.**
(a) It rejects a *subspace* (top-`k_neg` CLAY tangent directions), not a single averaged axis —
richer, closer to CLAY's "kill" operator. (b) The gate is conditioned on `q_tan`, so suppression
strength is context-dependent. (c) The decisive change is **training signal**: the ListNet list
already contains the "satisfies T⁺, violates T⁻" hard negatives (label 0), and the rejection now
sits on the gradient path to `(W_q, W_k, g)`, so the model is explicitly trained to demote them —
which the post-gate fixed rejection never was. With `k_neg=1` and the zero-init strength MLP
(λ = 1 at start), NegGate reproduces Tier-3's exact rank-1 rejection, so it can only improve on it.
Disentanglement loss and distance-based hard-negative mining are inherited unchanged.

**Expected Pareto point:** highest MEAN of the three; modest but nonzero negation lift.

---

### 3.2 P2 — DualScore Φ (medium, 0.73M params)

**File:** `src/tier3_dualscore.py` · **Pareto role:** maximize negation.

Geometry cannot enforce absence, so DualScore scores "absence" with a **calibrated discriminative
channel** instead — the one idea (Tier-4 hybrid) that ever moved `-Male,-Mustache` and `-Young`,
done properly. Offline, once and frozen: fit attribute probes `p(x) ∈ ℝ⁴⁰` on frozen train features
(BCE), then cache `P = σ(p(X_db)) ∈ [0,1]^{N×40}` for the frozen test DB. At query time:

```
q⁺     = FusionPhiPos(v_ref, T⁺)                 # geodesic identity+positive fusion (frozen recipe)
s_cos  = X · q⁺
s_neg  = Σ_{b∈T⁻} ( 1 − P[:,b] )                 # reward ABSENCE of each negated attribute
s_pos  = Σ_{a∈T⁺}       P[:,a]                   # reward PRESENCE of each positive attribute
(β, γ) = softplus( FusionHead([ĥ_ref, mean T̂⁺, mean T̂⁻]) )   # QUERY-CONDITIONED weights
score  = s_cos + β·s_neg + γ·s_pos               # composite, ranked with ListNet
```

**Why it is not just Tier-4 hybrid.** The DB CLIP features stay frozen; `P` is built once and frozen
exactly like CLAY's offline visual DB. Unlike hybrid: (a) probe fusion is trained jointly against
the *ranking* metric; (b) the weights are **query-conditioned** — the `FusionHead` reads the actual
query and sets how hard each channel bites — rather than one global `β`; (c) it adds a **presence
reward** for positives, not only an absence penalty; (d) it keeps the strong positive Φ. For a
pure-positive query the negation channel is 0 and DualScore reduces to the positive baseline. The
probe channel gives the model a clean, monotone "does image d have attribute b?" axis that geometry
cannot supply — turning negation from fragile axis-removal into calibrated demotion of images that
possess b.

**Expected Pareto point:** best on the negation subset; MEAN competitive with or slightly under P1.

---

### 3.3 P3 — PolarityComposer Φ (medium, 4.73M params)

**File:** `src/tier3_polaritycomposer.py` · **Pareto role:** balance / the originality headline.

Take the composer's structural lesson (a learned operator can express negation the arithmetic
cannot) but remove the two reasons it collapsed: it learned composition from scratch off a
near-identity residual, and it consumed only attribute *centroids*, discarding CLAY structure.

```
tokens = [ q_tan⁰ = Log_μ(v_ref) ⊕ pol(ref),          # anchor = the geodesic query
           {V_a^(1..r) ⊕ pol(+)}_{a∈T⁺},              # top-r CLAY subspace dirs per + attr
           {V_b^(1..r) ⊕ pol(−)}_{b∈T⁻} ]             # top-r CLAY subspace dirs per − attr
z   = SetTransformer(tokens)                          # 2 layers, d=512, 8 heads
Δq  = MLP( z[anchor] )                                # zero-init head ⇒ Δq = 0 at start
q   = normalize( Exp_μ( q_tan⁰ + Δq ) )               # residual on the GEODESIC prior
```

**Why it should not collapse like the composer.** Because the head is zero-init, `Δq = 0` at start,
so the untrained model sits **exactly at GDE-style identity composition** — it inherits the
positive-query quality the from-scratch composer lacked and *refines a strong prior* instead of
learning composition from zero (verified: `cos(q, v_ref) = 1.0` at init). Feeding **CLAY tangent-SVD
subspace tokens** (not raw centroids) is the concrete CLAY/GDE link the composer omitted; polarity
embeddings let self-attention learn asymmetric, cross-attribute affirmation/negation interactions.
Bounded to 2 layers and warm-started, it trains in the same regime as the other tiers (extend to 60
epochs if the ListNet curve is still descending at 30).

**Expected Pareto point:** middle — should improve both MEAN and negation; the most expressive model
and the report's originality highlight (spec §4 rubric).

---

## 4. Spec compliance

| Requirement (project_specification.md) | How each proposal satisfies it |
|---|---|
| §1/§3.2 — replace the naïve pre-SVD stack with dynamic fusion Φ | All keep the learned cross-attention / composition; P1 gates a CLAY subspace, P3 attends over CLAY subspace tokens, P2 gates presence/absence channels per query. |
| §3.2 — frozen offline DB, no re-encoding | CLIP frozen; DB CLIP features never mutated (guarded by `assert_db_frozen`); P2's probe matrix is built once and read-only. |
| §3.1 — multiple +/− conditions, defined interaction | Positive = geodesic add; negation = P1 gated-subspace rejection / P2 calibrated absence reward / P3 learned attention. Handles the mandatory single, composed, and negation-heavy queries. |
| §3.1.1 — Hamming≤2 GT, ≥5 sources, no leakage | `QueryGenerator` mirrors the GT rule on the **train** split; the eval JSON is only read to compute metrics. |
| §3.3 — CLIP ViT-B/32 frozen | All operate on the frozen 512-D features. |
| §6 — built from our own modules, primitives cited | Primitives (PGA tangent maps, CLAY SVD subspaces, InfoNCE/ListNet, Oldfield rejection, cross-attention, CAV probes) are cited; the composition is the contribution. |

---

## 5. Engineering / code quality

- **Single owner per responsibility.** `distance_mine`, `disentanglement_loss`, and the frozen-DB
  tripwire live in `src/tier3_train_utils.py`; both `tier3_combined` and the new tiers import from
  it — removing the earlier sideways coupling. Geometry stays in `manifold.py`, prompts in
  `clip_prompts`, eval in `eval.py`, persistence in `results_saver`.
- **Robustness.** Unit-query normalization, cut-locus-safe log/exp maps, empty-T⁻ reduces to plain
  cosine, QR handles the k_neg=1 edge, gradient clipping. Zero-init tails give safe defaults
  (λ=1 for P1, β=γ≈0.69 for P2, Δq=0 for P3).
- **Validated (CPU smoke).** All 10 tier-3 modules import; **pyflakes clean**; each proposal's
  forward produces unit queries with finite, NaN-free gradients through QR / gate / transformer; the
  frozen-DB invariant passes through each tier's real `make_get_ranking` seam.

---

## 6. How to run (VM)

```bash
python src/tier3_neggate.py           # P1 — lightweight, max MEAN
python src/tier3_dualscore.py         # P2 — probe channel, max negation
python src/tier3_polaritycomposer.py  # P3 — set-transformer, balance (extend to 60 epochs if needed)
```

Each writes weights + CSV to its `output/tier3_*/` directory.

### Success criteria (Pareto table)

Collate MEAN R@1/5/10 and the negation subset (`-Heavy_Makeup`, `-Young`, `-Male,-Mustache`,
`+Wearing_Lipstick,-Heavy_Makeup,+Smiling`) for all three vs. `tier3_combined` and `tier0_enhanced`:

- **P1** ≥ Combined at MEAN, with negation off the exact floor.
- **P2** best on the negation subset.
- **P3** improves both.

Then dump top-5 retrievals for `-Male,-Mustache` and the lipstick-composite for the report's
success/failure figures.

### Honesty note

`-Male, -Mustache` is partly a **cross-gender identity-retrieval ceiling** on CLIP ViT-B/32 (only 27
valid sources, Hamming≤2 forces a same-identity cross-gender match). The proposals are more likely
to move `-Young`, `-Heavy_Makeup`, and the lipstick-composite than to fully solve that one query;
the report should frame it that way rather than promising a high number on it.

---

## References (grounding)

- **CLAY** (Lim et al., 2026) — manifold-aware textual subspaces (tangent log-map → SVD → V_k),
  symmetric conditional similarity, frozen-DB retrieval.
- **GDE** (Berasi et al., 2025) — geodesic decomposability, log/exp maps, intrinsic mean, tangent
  primitive directions on the hypersphere.
- **CLIP** (Radford et al., 2021) — the frozen ViT-B/32 backbone and cosine retrieval.
- **Mind the Gap** (Liang et al., 2022) — modality gap / cone effect; motivates μ_txt centering and
  the CLAY rotation H.
- **Oldfield et al. (2023)** — PoS-grounded tangent subspaces; the isolate/kill (project / reject)
  operators our negation generalizes.
- **Alhamoud et al. (2025), NegBench** — CLIP affirmation bias; the discriminative-data solution
  behind P2's probe channel.
- **Cao et al. (2007), ListNet** — listwise ranking loss, metric-consistent with R@K/P@K.
