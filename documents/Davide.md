# Davide — progress log (Member B)

**Role (per [ROADMAP](ROADMAP.md) §"Member B"):** Representation, Baselines & Reference
Methods. I own CLIP, the frozen feature DB, and the methods-to-beat (Tier-0/1/2a), plus
the experimental-setup and CLIP/CLAY background report sections.

_(Note: I also built the evaluation engine below — nominally Member A's lane — since I
had the spec fresh from writing the docs.)_

_Last updated: 2026-06-29._

---

## Done

### Project scaffolding & documentation

Wrote and owns the four project-level documents everything else depends on:

- **[ROADMAP](ROADMAP.md)** — problem framing, the escalating method tiers (Tier-0 baseline
  → Tier-1 CLAY → Tier-2a training-free rejection → Tier-2b trained Φ), the 10-day / 2-person
  schedule, workload split, risks, and the cut list.
- **[CONTRACT](CONTRACT.md)** — the shared data contract: index-not-filename golden rule,
  the 40 attribute names in fixed order, the `attributes` `[N,40]` tensor spec, query-string
  format, GT-JSON structure, `ranking = list[int]` definition, `image_features` `[N,512]` spec,
  and the shared `get_ranking(src) -> ranking` signature (CONTRACT §7).
- Background/reference docs: [GDE](GDE.md), [CLIP](CLIP.md), [CLAY](CLAY.md),
  [CLAY_compositionality](CLAY_compositionality.md), [SOTA](SOTA.md), [links](links.md).
- Project skeleton notebooks under [skeleton/](../skeleton/).

---

### Evaluation engine — [src/eval.py](../src/eval.py)

The "ruler" every method is scored through. Built as a standalone, importable module so every
method can reuse it without code duplication. If each tier computed its own metrics they would
inevitably diverge — one `eval.py` that all tiers import is the only way to guarantee
apples-to-apples comparison. Testable before any real retrieval method exists.

**Metric definitions (verbatim from spec §3.1.3):**
- **Recall@K (hit rate)** — `1.0` if the top-K ranking shares at least one image with the
  ground-truth set G, else `0.0`. This is a hit-rate, NOT textbook recall `|hits|/|relevant|`.
- **Precision@K** — `|top-K ∩ G| / K`. Denominator is always K, not `|G|`.

Both averaged per source image, then across all valid sources of a query.

**Functions:**
- `find_eval_json()` — walks upward from `__file__` to locate `Evaluation/celeba_evaluation.json`,
  so imports work from any working directory (src/, repo root, or Colab).
- `recall_at_k` / `precision_at_k` — atomic per-source metric computations.
- `evaluate_query(ground_truth, get_ranking, ks)` — scores one query by averaging metrics over
  all its valid sources. Methods plug in through a `get_ranking(source_idx) -> ranking` callback
  (CONTRACT §7 seam); the engine owns the averaging, not the method.
- `evaluate_all(gt_list, make_get_ranking, ks)` — full benchmark loop; `make_get_ranking(query_str)`
  yields the per-query callback.
- `load_eval_json`, `parse_query` — parser handles both JSON style (`+Black_Hair, -Wavy_Hair`) and
  spec-table style (`+ Black Hair & - Wavy Hair`); names normalise to underscores.
- `format_results_table` — column-aligned console table for all queries + MEAN row.

**Self-tests (no CLIP / no attributes needed):** unit tests on both core metrics, plus oracle
(Recall must be 1.0 everywhere) and adversary (must score 0.0) bracketing tests against the real
evaluation JSON, and `parse_query` sanity checks. All pass. The harness is trustworthy before any
retrieval method exists.

---

### CLIP wrapper + frozen feature DB — [src/clip_features.py](../src/clip_features.py)

The model half of the frozen DB (CONTRACT §6). Frozen CLIP ViT-B/32, no training ever.

**Functions:**
- `extract_image_features()` — encodes the full CelebA test split through the vision tower →
  `visual_projection` → `[N, 512]`, L2-normalised. Enforces `shuffle=False` so row `i` ==
  `celeba[i]` (CONTRACT §0 indexing invariant). `_verify()` asserts row-count alignment with
  the attribute tensor + unit norms before saving → `artifacts/clip_image_features_test.pt`.
- `attr_to_prompt` + `extract_attribute_text_features()` — text side: templates each of the
  40 attributes (`"a photo of a person with <attr>"`) through the CLIP text tower →
  `artifacts/clip_attr_text_features.pt` `[40, 512]`. Row `j` == attribute `j`. These are the
  +/− direction vectors Tier-0 uses for latent arithmetic.
- `load_image_features()` / `load_attribute_text_features()` — loaders; zero recomputation.

**Technical note:** used the two-step `vision_model → visual_projection` call explicitly rather
than `get_image_features`, because in transformers 5.x the latter returns an output object, not a
tensor. Ran on Colab GPU ([notebooks/colab_extract_features.ipynb](../notebooks/colab_extract_features.ipynb));
both tables verified locally: shape `[19962, 512]`, unit-normed, row-aligned to the split.

**Output artifacts (frozen, do not re-extract unless forced):**
- `artifacts/clip_image_features_test.pt`
- `artifacts/clip_attr_text_features.pt`

---

### Tier-0 vanilla latent-arithmetic baseline — [src/tier0.py](../src/tier0.py) ← Milestone M1

The spec's vanilla "method to beat" (§3.2.3):

```
query = normalize( v_ref + α · (Σ t⁺ − Σ t⁻) )
```

Ranked by cosine over the frozen DB. Plugs into the eval engine through the shared
`get_ranking(src_idx) -> ranking` seam (CONTRACT §7). `evaluate_tier0()` scores all 14 queries
and writes `output/tier0_alpha{alpha}.csv` (per-query rows + a MEAN row). `alpha` is the
identity ↔ modification dial — the ablation knob for the report.

**Numbers (α=1.0, prompt-bank mean text vectors, all 14 queries):**
MEAN R@1 **0.0243**, R@5 **0.0715**, R@10 **0.1050**.
Best: `+Smiling` R@10 0.248, `+Mustache` R@10 0.219.
**Negation collapses** (`-Male, -Mustache` → 0.000 across every config) — exactly the failure
mode Tier-2a is designed to address.

**Why these numbers are correct, not a bug.** At α=0 (pure image→image similarity, no text),
the mean is MEAN R@5 ≈ 0.050 — only marginally below α=1.0. The GT requires a target to both
satisfy the query constraints AND sit within Hamming ≤ 2 of the source (spec §3.1.1).
- α=0 ignores the text and retrieves visual look-alikes that keep the source's original attribute
  values → they fail the query constraint.
- α=1.0 adds the CLIP text vector which, due to the modality gap, yanks the query into the text
  cone and washes out reference identity.

Neither endpoint wins. This is the fundamental identity-vs-modification tension that motivates
everything above Tier-0, and it makes Tier-0 a legitimate, honest lower bound.

---

### Tier-1 CLAY reproduction — [src/tier1.py](../src/tier1.py) ← Milestone M2

**Faithful pure CLAY** — a training-free method, not a model: pure linear algebra on the frozen
DB. No Colab, no GPU, no new artifacts; reads `clip_image_features_test.pt` +
`clip_attr_prompt_bank.pt` directly.

**Pipeline (CLAY.md §3.2):**

1. Stack ALL condition prompts (`+` and `−` alike — faithful CLAY has no native negation) → prompt
   stack `T_c`.
2. Compute the normalised Euclidean mean `μ_c` of `T_c` (centre of the condition's text cloud).
3. Log-map: project all prompts onto the tangent plane at `μ_c` →
   `L = Log_{μ_c}(T_c)` → `[n, d]`.
4. SVD of `L` → top-`k` right singular vectors `V_k` → the conditional similarity subspace `[d, k]`.
5. Optionally apply the minimal rotation `H` that maps the visual mean `μ_img` onto `μ_c`,
   closing the modality gap without disturbing intra-DB geometry (CLAY.md §3.2 rotation trick).
6. Project the full DB once per query: `coords = Log_{μ_c}(H·V) @ V_k → [N, k]`.
7. Per-source score: `coords[d] · coords[src]` (cosine in the subspace). Precomputed per query,
   O(k) per source — the efficiency design Tier-2a Track S inherits.

**Architectural note on primitives.** tier1.py retains its own private `_log_map`, `_align_rotation`,
`_build_subspace` (underscore-prefixed, never exported). At the time of Tier-1's implementation,
importing these from `manifold.py` (Alfonso's GDE module for Track V) would have been a sideways
peer import forbidden by CLAUDE.md. Refactoring tier1.py mid-project would risk regressing an
already-scored method. The DRY violation is localised and the trade-off explicitly accepted.

**Numbers (k=50, rotation H on, all 14 queries):**
MEAN R@1 0.0067, R@5 0.0227, R@10 **0.0351** — **below Tier-0** (R@5 0.0715). Expected:
faithful CLAY is a focus/preserve-similarity reframing with no +/− arithmetic, so it lags on
modification queries — exactly the naïve-stacking bottleneck Tier-2a Track S attacks.
Best: `+Mustache` R@10 0.093, `+Smiling` R@10 0.078. Negation and composed queries collapse to
0.000 — same failure shape as Tier-0, because polarity-blind CLAY has no negation mechanism.

**Ablations (k-sweep + rotation-H toggle, saved as `output/tier1_k{k}_{rot}.csv`):**
- k-sweep: k=5→10→20→50 → R@10 0.016→0.022→0.035→0.035 — monotonic, plateaus at k≈20.
- Rotation H: norot R@10 0.0313 vs rotH R@10 0.0351 — H helps by closing the modality gap.

**Tests ([test/test_tier1.py](../test/test_tier1.py)):** 8 checks, all green — log/exp round-trip,
rotation orthogonality, subspace orthonormality and idempotence, k-clamping, padding strip,
end-to-end ranking.

---

### Tier-2a Track S — Asymmetric Conditional Subspaces — [src/tier2a_S.py](../src/tier2a_S.py)

The training-free **S** approach of Tier-2a, designed from [S_plan.md](S_plan.md). Where
faithful CLAY (Tier-1) is polarity-blind — stacking `+` and `−` prompts into one subspace —
Track S builds **separate** subspaces per polarity and scores asymmetrically.

**Scoring formula:**

```
score(d, src) = cos(v_d, v_src)                              ← identity anchor
              + mean_a cos_{S⁺_a}(v_d, v_src)               ← manifold-aware positive focus
              − λ · Σ_b ‖proj_{S⁻_b}(v_d)‖                  ← asymmetric negation penalty
```

Each term has a distinct role:

1. **Identity anchor** `cos(v_d, v_src)`. Every query is "an image *like the reference*, with
   +X, without −X" — the reference defines the affirmative "near" region. Without this term,
   pure-subspace scoring discards reference identity: a negation-only query produces a
   reference-independent ranking (identical for every source → R@5 0.000). Toggle:
   `identity_anchor=False` recovers the plan-literal pure-subspace formula as an ablation.

2. **Mean cosine across positive subspaces**. CLAY's manifold-aware conditional similarity,
   restricted to positive-only subspaces so negative attribute directions cannot contaminate it.
   Computed as row-normalised projection coords: `D @ D[src]` (cosine in `S⁺_a`). This measures
   similarity-to-reference inside the subspace — a focus/preserve term, NOT a push-toward-attribute
   term. This is the same limitation CLAY carries, which explains why Track S trails Tier-0 on
   modification-heavy queries.

3. **Asymmetric negation penalty** `λ · ‖proj_{S⁻_b}(v_d)‖`. Energy of `v_d` inside the
   negative subspace means the image "looks like" the thing we want absent, and is pushed down.
   "Not Male" becomes "low energy in the Male subspace" — a region of the sphere, not a single
   anti-Male point. This is the genuinely new piece CLAY lacks entirely.

**Architecture and design decisions.**

- **Polarity split.** `_polarity_groups()` groups prompts per polarity. `per_condition=True`
  (default) gives one subspace per attribute, equal geometric weight to every condition
  (PoS-Subspaces' per-role philosophy). `per_condition=False` merges all prompts into one group
  (CLAY-style, biased toward attributes with more prompts).

- **Precompute-once.** `_precompute()` builds all subspaces once per query and reuses them across
  every source of that query — CLAY's efficiency design, inherited here. Positive groups yield
  row-normalised DB coords `D⁺`. Negative groups yield a single per-image energy penalty vector
  summed across all negative subspaces.

- **Batched scoring.** `_batched_rankings()` chunks sources into blocks of 512, scores with a
  BLAS GEMM (`[N, chunk]` score matrix), excludes each source from its own column
  (CONTRACT §5), and takes only the `top_k` prefix via `torch.topk` — materialising the full
  N-length ranking per source would cost millions of Python ints per benchmark with no metric
  benefit (metrics only read `ranking[:max(k)]`).

- **Results persistence.** Extended [src/results_saver.py](../src/results_saver.py) with
  `output_subdir(name)` so the 10 ablation CSVs group under `output/tier2a_S/` instead of
  scattering across the shared `output/` root. `results_saver` is the single owner of CSV
  persistence — no tier duplicates that logic.

- **Shared sphere geometry.** Track S imports `log_map`, `align_rotation`, `build_subspace`
  from [src/manifold.py](../src/manifold.py) (Alfonso's Riemannian primitives module). This is
  a proper shared dependency — both Track S and Track V depend on manifold.py, which owns the
  sphere geometry once. Track S does NOT reach sideways into tier1.py for its primitives.

**Numbers (best config: k=50, per-condition, identity anchor on, rotation H on):**

| Metric | Tier-0 | Tier-1 CLAY | **Track S** | vs Tier-1 | vs Tier-0 |
|--------|:------:|:-----------:|:-----------:|:---------:|:---------:|
| R@1    | 0.0224 | 0.0067      | **0.0137**  | +2.0×     | −39%      |
| R@5    | 0.0715 | 0.0227      | **0.0454**  | +2.0×     | −37%      |
| R@10   | 0.1050 | 0.0351      | **0.0659**  | +1.9×     | −37%      |

Track S is ~2× Tier-1 on every metric. It trails Tier-0 because the positive subspace term is a
focus/preserve-similarity term rather than a push-toward-attribute term — the same limitation
CLAY carries, which the plan attributes to the conditional-similarity formulation.

**Ablations (10 configs, written to `output/tier2a_S/tier2a_S_{tag}.csv`):**

| Variant | MEAN R@5 | Key takeaway |
|---|:---:|---|
| per-cond, anchor, k=20, λ=0.1, rotH | 0.0390 | default |
| k=50 (best) | **0.0454** | k monotonically helps |
| k=5, k=10 | 0.0317 | diminishing below k=20 |
| per-condition=False (stacked) | 0.0345 | per-condition beats stacked |
| identity_anchor=False | 0.0168 | collapses to ≈ Tier-1 |
| use_rotation=False | 0.0394 | rotation slightly hurts with the anchor |
| λ=0.05 / 0.2 / 0.5 | 0.0391–0.0384 | negation penalty has negligible leverage |

Key takeaways:
- **Identity anchor is decisive** — removing it collapses R@5 0.0390 → 0.0168 ≈ Tier-1 CLAY.
  The anchor is what lifts Track S above CLAY.
- **k monotonic** — more subspace dimensions consistently help; k=50 is the best tested.
- **per-condition beats stacked** — equal geometric weight per attribute outperforms
  prompt-count-biased stacking, as the plan predicted.
- **λ negligible** — the negation penalty has almost no leverage. Root cause: the effective
  penalty `λ · ‖proj_{S⁻}(v_d)‖ ≈ 0.1 × 0.2 ≈ 0.02` is an order of magnitude smaller than
  the `cos(v_d, v_src)` range (~0.1–0.3 across the DB). When the reference image itself has the
  negative attribute, the anchor strongly preferring similar (negative-attribute) images cannot
  be overcome by such a small additive penalty. Fixing this requires a much larger λ, or a
  fundamentally different architecture that reshapes the search direction rather than penalising
  after the fact.
- **Rotation H negligible/slightly hurts with the anchor** — opposite to Tier-1, because the
  identity anchor already operates in image space; rotating onto the text mean then partially
  undoes that alignment.

**Negation analysis.** `-Male, -Mustache` stays 0.000 for every config — but so does Tier-0.
This query has only 27 sources and a Hamming-2 GT constraint that is effectively unwinnable
training-free. Other negation/composed queries do lift off zero with the anchor
(e.g. `+Wearing_Lipstick, -Heavy_Makeup, +Smiling` R@5 up to 0.088). The path to beating Tier-0
on negation requires a modification-style positive term (reward attribute presence, not
similarity-to-reference) — flagged as the next iteration.

**Tests ([test/test_tier2a_S.py](../test/test_tier2a_S.py)):** 9 checks, all green — log/exp
round-trip, S⁺ orthonormality, P⁺ idempotence, S⁻ complement orthogonality, neg-norm
non-negativity, end-to-end ranking, empty T⁺ and empty T⁻ fallbacks, stacked variant.

---

### Deliverable notebook skeleton — [notebooks/Compositional_Retrieval_Report.ipynb](../notebooks/Compositional_Retrieval_Report.ipynb)

The scaffold for the final self-contained Colab submission. Sets up the section structure,
sys.path bootstrap, artifact loading, and the import graph so all tiers can be run and compared
in one notebook. No method logic lives here — all heavy lifting is imported from `src/`.

---

## In progress / next up (my lane — Member B)

- [x] **Tier-1** CLAY reproduction — done (see above).
- [x] **Tier-2a Track S** training-free asymmetric-subspace +/− variant — done; beats CLAY ~2×,
      trails Tier-0. Next iteration: modification-style positive term to clear the Tier-0 bar.
- [ ] **Modality-gap analysis** (justifies the design choices).
- [ ] **β knob / α-β sweep** — port a separate negative weight `β` into `tier0.py`, then sweep
      α/β for the ablation.
- [ ] **Report ownership:** Experimental setup, CLIP/CLAY background, and the
      SVD-k / α-β / rotation-H ablations.

---

## Notes / dependencies

- Data loading + the `attributes` tensor (`-1 → 0/1` conversion) built by Alfonso (Member A):
  [data_loader.py](../src/data_loader.py) / [Phase_A_Data_Preparation.ipynb](../notebooks/Phase_A_Data_Preparation.ipynb).
  Artifacts live in `artifacts/` (gitignored, `*.pt`).
- Sphere-geometry primitives in [src/manifold.py](../src/manifold.py) are a shared module owned
  by Alfonso. Track S imports `log_map`, `align_rotation`, `build_subspace` from there; tier1.py
  keeps private copies — deliberate trade-off, see above.
- My tier sequence (0→1→2a-S) is self-contained on the frozen DB — both test tables
  (`clip_image_features_test.pt`, `clip_attr_text_features.pt`) are cached; everything runs
  offline, no Colab. All tiers plug into the eval engine through the shared
  `get_ranking(src) -> ranking` signature ([CONTRACT](CONTRACT.md) §7).
- We score on **all 14** queries exactly as in the JSON (authoritative), noting the
  14-vs-12 spec mismatch in one sentence in the report ([CONTRACT](CONTRACT.md) §4).
