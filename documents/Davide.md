# Davide — progress log (Member B)

**Role (per [ROADMAP](ROADMAP.md) §"Member B"):** Representation, Baselines & Reference
Methods. I own CLIP, the frozen feature DB, and the methods-to-beat (Tier-0/1/2a), plus
the experimental-setup and CLIP/CLAY background report sections.

_(Note: I also built the evaluation engine below — nominally Member A's lane — since I
had the spec fresh from writing the docs.)_

<<<<<<< HEAD
_Last updated: 2026-06-29._
=======
_Last updated: 2026-06-27._
>>>>>>> a96a64625d1211d007e0ff767636f667cbe0cedd

---

## Done

### Project scaffolding & documentation
- Wrote the **[ROADMAP](ROADMAP.md)**: problem framing, the escalating method tiers
  (Tier-0 baseline → Tier-1 CLAY → Tier-2a training-free rejection → Tier-2b trained Φ),
  the 10-day / 2-person schedule, workload split, risks, and the cut list.
- Wrote the **[CONTRACT](CONTRACT.md)** — the data contract between the two halves:
  the index-not-filename golden rule, the 40 attribute names in fixed order, the
  `attributes` `[N,40]` tensor spec, the query-string format, the GT-JSON structure,
  the `ranking = list[int]` definition, the `image_features` `[N,512]` spec, and the
  shared `score(...) -> ranking` signature.
- Background/reference docs: [GDE](GDE.md), [CLIP](CLIP.md), [CLAY](CLAY.md),
  [CLAY_compositionality](CLAY_compositionality.md), [SOTA](SOTA.md), [links](links.md).
- Project skeleton notebooks under [skeleton/](../skeleton/).

### Evaluation engine — [src/eval.py](../src/eval.py) (+ thin [notebooks/eval.ipynb](../notebooks/eval.ipynb))  ← main deliverable so far
Split into a reusable module (`src/eval.py`) so every method notebook can `import` it,
with the notebook reduced to running/plotting/self-tests.
The "ruler" every method is scored through. Implemented and self-tested:
- `find_eval_json()` — locates `Evaluation/celeba_evaluation.json` from any working dir
  (src/, repo root, or Colab).
- `recall_at_k` / `precision_at_k` — Recall@K is the **hit-rate** definition from the
  spec (1.0 if top-K intersects GT), not textbook recall; Precision@K = `|top-K ∩ G| / K`.
- `evaluate_query` / `evaluate_all` — average metrics over all valid sources, then over
  all queries; methods plug in via a `get_ranking(source_idx) -> ranking` callback.
- `load_eval_json`, `parse_query` (handles both JSON style `+Black_Hair, -Wavy_Hair`
  and spec-table style `+ Eyeglasses & - Smiling`), `format_results_table`.
- **Self-tests (no CLIP / no attributes needed):** unit tests on the two core metrics,
  plus oracle (Recall must be 1.0 everywhere) and adversary (must score 0.0) bracketing
  tests against the real evaluation JSON, and `parse_query` sanity checks. All pass.

This means the evaluation harness is trustworthy *before* any real retrieval method exists.

### CLIP wrapper + frozen feature DB — [src/clip_features.py](../src/clip_features.py)
The model half of the frozen DB (CONTRACT §6). Frozen CLIP ViT-B/32, no training.
- `extract_image_features()` — encodes the whole CelebA test split (vision tower →
  `visual_projection` → `[N,512]`, L2-normalized) in strict index order (`shuffle=False`,
  so row `i` == `celeba[i]`, CONTRACT §0). `_verify()` asserts row-count alignment with
  the attribute tensor + unit norms before saving → `artifacts/clip_image_features_test.pt`.
- `attr_to_prompt` + `extract_attribute_text_features()` — the **text** side: templates
  each of the 40 attributes (`"a photo of a person with <attr>"`) through the CLIP text
  tower → `artifacts/clip_attr_text_features.pt` `[40,512]` (row `j` == attribute `j`).
  These are the +/− direction vectors every tier nudges with.
- `load_image_features()` / `load_attribute_text_features()` — loaders mirroring
  `load_attributes()`.
- **Note:** called the two-step `vision_model → visual_projection` (and `text_model →
  text_projection`) explicitly rather than `get_image_features`, because the latter's
  return type varies across transformers 5.x (returns an output object, not a tensor).
- **Ran on Colab GPU** ([notebooks/colab_extract_features.ipynb](../notebooks/colab_extract_features.ipynb),
  a one-and-done extraction notebook); both `.pt` tables verified locally
  (`[19962,512]`, unit-normed, aligned).

### Tier-0 vanilla latent-arithmetic baseline — [src/tier0.py](../src/tier0.py)  ← Milestone M1 done
`query = v_ref + alpha·(Σ t⁺ − Σ t⁻)`, ranked by cosine over the frozen DB. Matches the
shared `score(...) -> ranking` signature (CONTRACT §7) and plugs straight into the eval
engine; `evaluate_tier0()` scores all 14 queries and writes a CSV
(`output/tier0_alpha<alpha>.csv`, per-query rows + a MEAN row).
- **First real numbers (alpha=1.0):** MEAN R@1 0.022, R@5 0.070, **R@10 0.105**;
  P@1 0.022, P@5 0.017, P@10 0.014.
- Best single attrs: `+Smiling` (R@10 0.248), `+Mustache` (0.219). **Negation collapses**
  (`-Male, -Mustache` → 0.000) and composed queries degrade — exactly the failure modes
  that motivate Tier-2a rejection. Good baseline story for the report.
- `alpha` is the identity↔modification dial (α-ablation knob).

### Tier-1 CLAY reproduction — [src/tier1.py](../src/tier1.py)  ← Milestone M2 done
**Faithful pure CLAY** — a *training-free method*, not a model: pure linear algebra on the frozen
DB, so no Colab/GPU and nothing to export (it consumes `clip_image_features_test.pt` +
`clip_attr_prompt_bank.pt` directly). Pipeline (CLAY.md §3.2): stack ALL condition prompts (+ and −
alike — CLAY has no native negation) → `mu_c` = normalized mean → log-map onto the tangent space →
SVD → top-k right singular vectors `V_k` → rotate visual mean to text mean with `H` (modality gap) →
project DB once per query → cosine in the subspace. Plugs into the eval engine through the same
`get_ranking(src)->ranking` seam (CONTRACT §7); per-query precompute, per-source one `[N,k]@[k]`.
- **Numbers (k=50, +rotation):** MEAN R@1 0.0067, R@5 0.0227, **R@10 0.0351** — **below** Tier-0
  (promptbank 0.024/0.072/0.105). This is the *expected, gradeable* result: faithful CLAY is a
  focus/preserve similarity reframing with **no +/− arithmetic**, so it lags on these *modification*
  queries — exactly the naïve-stacking bottleneck Tier-2a attacks. Best on high-frequency / preserve
  attrs (`+Mustache` R@10 0.093, `+Smiling` 0.078); negation/composed collapse (`-Male, -Mustache`
  0.000), same failure shape as Tier-0.
- **Ablations (report rows, saved as `output/tier1_k{k}_{rot}.csv`):** k-sweep is monotonic
  (k=5→10→20→50: R@10 0.016→0.022→0.035→0.035, plateaus by k≈20); rotation `H` helps
  (norot R@10 0.0313 vs rotH 0.0351). These are the SVD-k and rotation-`H` ablations I own.
- **Tests:** [test/test_tier1.py](../test/test_tier1.py) (new project-root `test/` folder — tests
  live there, never inline in `src/`): 8 checks, all green (log/exp round-trip, rotation
  orthogonality, subspace orthonormality/idempotence, k-clamping, padding strip, end-to-end ranking).

<<<<<<< HEAD
### Tier-2a Track S — Asymmetric Conditional Subspaces — [src/tier2a_S.py](../src/tier2a_S.py)  ← my Tier-2a contribution
The training-free **S** approach (named `tier2a_S` to distinguish it from Alfonso's visual-prototype
**V** approach, [tier2a_visual.py](../src/tier2a_visual.py)). Implemented from [S_plan.md](S_plan.md).
Where faithful CLAY (Tier-1) is *polarity-blind* — it stacks `+` and `−` prompts into one subspace —
Track S builds **separate** subspaces per polarity and scores:
`cos(v_ref,v_d) + mean_a cos_{S⁺_a}(v_d,v_ref) − λ·Σ_b ‖proj_{S⁻_b}(v_d)‖`.
The first term is an **identity anchor** (composition = "an image *like the reference*, with +X, without −X");
the second is CLAY's manifold-aware focus restricted to the **positive-only** subspaces; the third is the
genuinely new piece — an **asymmetric negation penalty** (energy in a negative subspace = "looks like the
thing we want absent", so it is pushed down). This is the subspace generalisation of SpaceVLM's `d̂`
(affirmative − negation), with PoS-Subspaces' per-condition philosophy. Same `get_ranking(src)->ranking`
seam (CONTRACT §7); precompute-per-query, batched scoring over all of a query's sources.

- **Architecture (SOLID, per CLAUDE.md):** lifted the frozen manifold primitives into a new shared
  **[src/manifold.py](../src/manifold.py)** (`log_map` / `exp_map` / `align_rotation` / `build_subspace`)
  so Tier-1 and Track S both *depend on it* instead of Track S reaching sideways into `tier1.py`. Tier-1
  refactored to import them (behaviour-preserving — all 8 Tier-1 tests still green). Added
  `results_saver.output_subdir()` so the 10 ablation CSVs group under **`output/tier2a_S/`**.
- **Numbers (best config, k=50):** MEAN R@1 **0.0137**, R@5 **0.0454**, R@10 **0.0659** — **~2× Tier-1/CLAY**
  (k20 R@5 0.0213), but still **below Tier-0** (promptbank R@5 0.0715). Honest, gradeable outcome: the
  subspace term *preserves similarity toward the reference* rather than *pushing toward the target attribute*,
  so on this **modification-heavy, positive-dominated** benchmark it trails Tier-0's raw text-direction push —
  exactly the focus-vs-modify limitation the plan attributes to CLAY. The win is over CLAY, the method-to-beat.
- **Ablations (10 rows, `output/tier2a_S/tier2a_S_{variant}_{anchor}_k{k+}_{k-}_lam{λ}_{rot}.csv`):**
  - **Identity anchor is decisive** — removing it (`noanchor`, the plan-literal pure-subspace formula)
    collapses R@5 0.0390 → **0.0168 ≈ CLAY**, confirming the anchor is what lifts Track S above Tier-1.
  - **k monotonic** (k5→10→20→50: R@5 0.0317→0.0317→0.0390→**0.0454**); **per-condition > stacked**
    (0.0390 vs 0.0345, as the plan predicted); **λ negligible** (0.05→0.5: 0.0391→0.0384 — the negation
    penalty has little leverage here); **rotation H negligible/slightly hurts with the anchor** (norot 0.0394
    vs rotH 0.0390 — opposite to Tier-1, because the anchor already works in image space).
- **Negation reality check:** `-Male, -Mustache` stays **0.000 for every config — but so does Tier-0**
  (27 sources, very restrictive Hamming-2 GT; effectively unwinnable training-free). Other negation/composed
  queries *do* lift off zero with the anchor (e.g. `+Wearing_Lipstick,-Heavy_Makeup,+Smiling` R@5 up to 0.088).
  So DoD #3/#4 ([S_plan §9](S_plan.md)) are **not** fully met by the literal design; the path to beating Tier-0
  is a *modification-style* positive term (reward attribute presence, not similarity-to-reference) — flagged as
  the next iteration, not silently claimed.
- **Tests:** [test/test_tier2a_S.py](../test/test_tier2a_S.py) — 9 checks, all green (log/exp round-trip,
  S⁺ orthonormality, P⁺ idempotence, S⁻ complement orthogonality, neg-norm non-negativity, end-to-end
  ranking, empty-T⁺ and empty-T⁻ fallbacks, stacked variant).

=======
>>>>>>> a96a64625d1211d007e0ff767636f667cbe0cedd
---

## In progress / next up (my lane — Member B)
- [x] **Tier-1** CLAY reproduction — done (see above).
<<<<<<< HEAD
- [x] **Tier-2a Track S** training-free asymmetric-subspace +/− variant — done (see above); beats CLAY
      ~2×, trails Tier-0. Next iteration: modification-style positive term to clear the Tier-0 bar.
=======
- [ ] **Tier-2a** training-free +/− projection-rejection variant (the guaranteed contribution).
>>>>>>> a96a64625d1211d007e0ff767636f667cbe0cedd
- [ ] **Modality-gap analysis** (justifies the design choices).
- [ ] **β knob / α-β sweep** — port a separate negative weight `β` into `tier0.py` (the old
      [methods.py](../src/methods.py) had it; that file is otherwise superseded by `tier0.py`),
      then sweep α/β for the ablation.
- [ ] **Train-subset features** `clip_image_features_train.pt` — only needed for Tier-2b Φ
      training (Alfonso's lane); deferred until then.
- [ ] **Report ownership:** Experimental setup, CLIP/CLAY background, and the
      SVD-k / α-β / rotation-`H` ablations.

## Notes / dependencies
- Data loading + the `attributes` tensor (`-1 → 0/1` conversion) landed via Alfonso
  (Member A): [data_loader.py](../src/data_loader.py) /
  [Phase_A_Data_Preparation.ipynb](../notebooks/Phase_A_Data_Preparation.ipynb).
  Caches now live in `artifacts/` (gitignored, `*.pt`).
- My tier sequence (0→1→2a) is self-contained on the frozen DB — both test tables
  (`clip_image_features_test.pt`, `clip_attr_text_features.pt`) are cached, so Tier-1/2a
  run **offline, no Colab**. Everything plugs into the eval engine through the shared
  `score(...) -> ranking` signature ([CONTRACT](CONTRACT.md) §7).
- We score on **all 14** queries exactly as in the JSON (authoritative), noting the
  14-vs-12 spec mismatch in one sentence in the report (see [CONTRACT](CONTRACT.md) §4).
