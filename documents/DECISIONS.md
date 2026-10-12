# How the project evolved: key decisions

Two people, about three weeks of work (June 15 – July 5, 2026). The project ran as a sequence of experiments, each one motivated by the failure of the previous one. This page lists the decisions that shaped the final system, what triggered each one and what it changed. Every entry links to the document where the decision was originally recorded.

All numbers are mean Recall@K over the 13 unique official CelebA queries. Method names follow the final naming; the [glossary](README.md#glossary) maps them to the "Tier" names used in the process documents.

## Timeline

| # | When | Decision | Effect |
|---|---|---|---|
| 1 | Jun 18–26 | Freeze the database and agree on a data contract | Every method plugs into one evaluation harness |
| 2 | Jun 26–28 | Fix the modality gap before anything else | R@5 0.070 → 0.114 (+63%), the largest training-free gain |
| 3 | Jun 27–29 | Reproduce the assigned papers honestly | CLAY and GDE both score *below* the baseline |
| 4 | Jun 27–30 | Two independent training-free tracks, then DGP | Training-free plateau at R@5 ≈ 0.115; negation stuck at 0.000 |
| 5 | Jun 29 | Retire CLAY adaptive-k | Documented negative result (R@10 0.054 → 0.022) |
| 6 | Jun 30 – Jul 1 | Learn the fusion Φ | Best geometric model: Φ-Combined, R@5 0.158 |
| 7 | Jul 1–2 | Negation-aware Φ variants | Dual Score kept (R@5 0.175); NegGate and PolarityComposer dropped |
| 8 | Jul 2 | **Pivot from geometry to probability** | BAP: R@5 0.260 with no training; negation floor broken |
| 9 | Jul 2 | Try LingoCIR, then drop it | Poor results, removed the same day |
| 10 | Jul 2 | Bayes-initialized A-BAP | Training starts exactly at BAP and can only refine it |
| 11 | Jul 2–5 | Ablate every refinement (CD-Bayes) | Kept 3, rejected 2; ListNet beat LambdaLoss on 3 seeds |
| 12 | Jul 3–5 | Statistical evaluation and custom queries | A-BAP significant on 30/30 paired comparisons |
| 13 | Jul 4 | Correct the AwA2 benchmark | Found that the original setup did not test the real task |
| 14 | Jul 5 | Report negative transfer honestly | A-BAP fails on OVAD; Dual Score transfers best |
| 15 | Jul 5 | Refactor the codebase | Clear Baselines / Training-Free / Trained layout |

**Final result:** A-BAP reaches R@1 0.110, R@5 0.291, R@10 0.413, about 4× the vanilla baseline.

---

## 1. A frozen database and a data contract

**Context.** Two people working in parallel for ten days, on a shared benchmark whose ground truth refers to images by dataset index ([01_ROADMAP](process/01_ROADMAP.md)).
**Decision.** Encode the CelebA test set with CLIP once and never modify it. Write down a [data contract](process/02_CONTRACT.md) covering shapes, the index-not-filename rule and a single `get_ranking(src)` signature.
**Result.** Every later method, from vanilla arithmetic to A-BAP, plugged into the same harness without integration work.

## 2. The modality gap first

**Context.** Vanilla latent arithmetic (`v_ref + Σt⁺ − Σt⁻`) scored R@5 0.070. Image and text embeddings occupy separate cones (Liang et al., 2022).
**Decision.** Centre both modalities before composing (CLIP Enhanced), plus delta normalization and prompt ensembling ([Alfonso's log](logs/Alfonso.md)).
**Result.** R@5 0.070 → 0.114 (+63%). Centering alone outperformed all three fixes combined. No later geometric trick came close to this gain.

## 3. Honest reproductions of the assigned papers

**Context.** The assignment builds on CLAY and GDE.
**Decision.** Re-implement both faithfully before modifying anything ([Davide's log](logs/Davide.md)).
**Result.** CLAY was the worst method of the whole project (best R@5 0.037) and GDE also fell below the baseline (0.061). This confirmed the assignment's premise that CLAY's pre-SVD stack is a bottleneck, and showed that these papers optimize a different objective than identity-preserving retrieval.

## 4. Two training-free tracks, then DGP

**Context.** "−X" means *any value but X*, not vector subtraction. Two ways to express that were split between the members so they could work without syncing ([03_TRAINING_FREE_SPLIT](process/03_TRAINING_FREE_SPLIT.md)).
**Decision.** Track V: image-space prototypes with orthogonal rejection ([04](process/04_TrackV.md); its code became the GDE reproduction and, extended, GDE-VP). Track S: asymmetric text subspaces ([06](process/06_S_plan.md)). Then Subspace Union ([09](process/09_planSV.md)) and DGP, a closed-form reference-conditioned gate replacing CLAY's equal-weight subspace.
**Result.** Best training-free geometric result: DGP, R@5 0.115, level with CLIP Enhanced. **`-Male, -Mustache` scored 0.000 under every configuration.** Four families of subspace methods showed that the plateau was structural, not a tuning problem.

## 5. A negative result kept on record

**Context.** One idea was to fix CLAY's negation by choosing the subspace size adaptively.
**Decision.** Run it properly, then retire it ([08_CLAY_TNEG_ADAPTIVEK_INVESTIGATION](process/08_CLAY_TNEG_ADAPTIVEK_INVESTIGATION.md)).
**Result.** R@10 dropped from 0.054 to 0.022. The investigation was documented and its code deleted.

## 6. Learning the fusion module

**Context.** The assignment's main deliverable is a fusion module Φ ([10_FusionModelGuide](process/10_FusionModelGuide.md)).
**Decision.** Make DGP's gate learnable (Φ-Gate, InfoNCE), then try variants: a ListNet ranking loss, a learned negation head and a free-form Composer ([11](process/11_Tier3_Variations_Report.md)). Finally combine the best ideas (Φ-Combined).
**Result.** Φ-Combined reached R@5 0.158, the best geometric model. Only the Composer ever scored above zero on `-Male, -Mustache` (R@10 0.111), even though it collapsed overall. That was the first hint that negation needed a different *scoring space*, not a better loss.

## 7. Negation-aware variants

**Context.** Φ-Combined still scored exactly 0 on the pure-negation query ([12_Negation_Fusion_Report](process/12_Negation_Fusion_Report.md)).
**Decision.** Three designs at different points of the trade-off: NegGate, Dual Score (attribute-probe channels next to the geometric query) and PolarityComposer.
**Result.** Dual Score reached R@5 0.175, the strongest non-probabilistic model, and was kept. NegGate (0.062) and PolarityComposer (0.013) were dropped.

## 8. The pivot: score the rule, don't approximate it

**Context.** A full audit ([13_FinalDestination](process/13_FinalDestination.md)) found three structural bottlenecks:
1. A single vector compared by cosine cannot express "has A, lacks B".
2. The benchmark defines identity as attribute agreement, not CLIP similarity. On `-Male, -Mustache` the valid targets are exactly the images that cosine similarity ranks lowest.
3. Raw cosines are uncalibrated.

**Decision.** The benchmark's ground-truth rule is fully known, so score it directly as Bayesian inference. 40 logistic probes produce calibrated attribute probabilities, and an exact Poisson-binomial tail expresses "same person, up to two different attributes". An earlier proposal, CKS ([07](process/07_OriginalTrainingFreeApproaches.md)), had the right log-linear form but used raw cosines. The parameter cap was lifted for a 40 GPU-hour budget.
**Result.** BAP reached R@5 0.260 (+65% over Φ-Combined) **without training the fusion**, and `-Male, -Mustache` went from 0.000 to 0.222.

## 9. A path tried and dropped

**Context.** The same audit proposed LingoCIR, composing queries through textual inversion.
**Decision.** Implement it and test it.
**Result.** The results were poor, and the code was removed the same day (commit "Lingocir results bad"). The deliverable mentions the attempt.

## 10. A-BAP: amortize the closed form

**Context.** The assignment requires a fusion module Φ that emits a query embedding, not a hand-written score.
**Decision.** Augment the database once, offline, with the probe channels. Train a small MLP Φ whose zero-initialized heads make it reproduce BAP exactly before training (deviation 0.0).
**Result.** Training can only refine a validated starting point. First trained version: R@5 0.281.

## 11. Every refinement earned its place

**Context.** BAP assumed attributes are independent and treated raw sigmoid outputs as probabilities ([14_NextStep](process/14_NextStep.md)).
**Decision.** Ablate five refinements one by one: calibration (C1), a Chow–Liu dependency tree (C2), a conditioned Hamming tail (C3), a cosine tie-breaking channel (C4) and QB-Norm (C5). Then retrain Φ on top.
**Result.** C1–C3 kept (BAP+), C4 and C5 rejected. LambdaLoss underperformed ListNet on all three seeds. Final A-BAP: **R@1 0.110, R@5 0.291, R@10 0.413**.

## 12. Statistics and stress tests

**Context.** Several queries have only 27–34 source images, so single-query differences are fragile ([14_NextStep](process/14_NextStep.md) Part I).
**Decision.** Stratified and paired bootstrap with Holm–Bonferroni correction. Add nine custom queries from a ground-truth generator verified to reproduce the official file with zero mismatches.
**Result.** A-BAP is significantly better than every baseline on every metric (30/30 comparisons). On a four-constraint custom query, CLIP, CLIP Enhanced and CLAY score 0 at every K, while A-BAP reaches R@10 0.600.

## 13. Correcting the AwA2 benchmark

**Context.** The first AwA2 transfer study showed CelebA's ranking inverting ([15](process/15_awa2_negative_transfer_analysis.md)).
**Decision.** Rebuilding the official CelebA ground truth exactly showed that a valid source must *violate* every constraint. The AwA2 benchmark had violated this rule, so it measured class retrieval rather than the editing task.
**Result.** That benchmark was renamed a surrogate, the generator now refuses to build an invalid one, and the earlier analysis is marked superseded instead of deleted.

## 14. Reporting negative transfer honestly

**Context.** On AwA2 (per-class attributes) and OVAD (~26× less training data per attribute), A-BAP fell to mid-pack and last place respectively ([16](process/16_ovad_negative_transfer_analysis.md)).
**Decision.** Investigate why instead of tuning it away. A λ-sweep showed the failure is structural.
**Result.** The conclusion in the README: scoring a known rule wins when labels are dense and per-image, while combining cosine similarity with learned attribute evidence (Dual Score) is the safer default otherwise.

## 15. Refactor for readers

**Context.** By July 5, `src/` held about 20 `tierN_*.py` files, including several abandoned variants.
**Decision.** Reorganize into `Baselines/`, `Training-Free/` and `Trained/` with descriptive names, and remove the experiments that did not ship. They remain in git history.
**Result.** The codebase matches the structure of the report. The [old → new file map](README.md#old-file-names) connects the process documents to the current files.
