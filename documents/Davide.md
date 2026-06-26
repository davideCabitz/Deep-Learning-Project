# Davide — progress log (Member B)

**Role (per [ROADMAP](ROADMAP.md) §"Member B"):** Representation, Baselines & Reference
Methods. I own CLIP, the frozen feature DB, and the methods-to-beat (Tier-0/1/2a), plus
the experimental-setup and CLIP/CLAY background report sections.

_(Note: I also built the evaluation engine below — nominally Member A's lane — since I
had the spec fresh from writing the docs.)_

_Last updated: 2026-06-26._

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

### Evaluation engine — [src/eval.ipynb](../src/eval.ipynb)  ← main deliverable so far
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

---

## In progress / next up (my lane — Member B)
- [ ] **CLIP ViT-B/32 wrapper** (HF `openai/clip-vit-base-patch32`, frozen) + attribute →
      text-prompt templating.
- [ ] **Offline feature extraction & caching** for the full test corpus + the train subset
      needed for Φ → `clip_image_features_test.pt` / `clip_image_features_train.pt`
      (L2-normalized rows, per [CONTRACT](CONTRACT.md) §6); the frozen retrieval DB.
- [ ] **Tier-0** vanilla latent-arithmetic baseline → first real Recall@K/Precision@K
      numbers through the eval engine (Milestone M1).
- [ ] **Tier-1** CLAY reproduction (tangent/log-map, SVD subspace, rotation `H`, naïve
      multi-condition stacking).
- [ ] **Tier-2a** training-free +/− projection-rejection variant (the guaranteed contribution).
- [ ] **Modality-gap analysis** (justifies the design choices).
- [ ] **Report ownership:** Experimental setup, CLIP/CLAY background, and the
      SVD-k / α-β / rotation-`H` ablations.

## Notes / dependencies
- Data loading + the `attributes` tensor (`-1 → 0/1` conversion) landed via Alfonso
  (Member A): [data_loader.py](../project/data_loader.py) /
  [Phase_A_Data_Preparation.ipynb](../project/Phase_A_Data_Preparation.ipynb).
- My tier sequence (0→1→2a) is self-contained on the frozen DB once features are cached;
  it plugs into the eval engine through the shared `score(...) -> ranking` signature
  ([CONTRACT](CONTRACT.md) §7). Until features exist, the eval harness is testable on fakes.
- We score on **all 14** queries exactly as in the JSON (authoritative), noting the
  14-vs-12 spec mismatch in one sentence in the report (see [CONTRACT](CONTRACT.md) §4).
