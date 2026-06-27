# Training-Free Approach Split — Tier-2a, two parallel pipelines

> **Purpose.** Divide the Tier-2a *training-free* exploration into **two complete, self-contained
> pipelines**, one per member, so each can build, evaluate, and write up an end-to-end method
> **without ever syncing** with the other. They share only the already-frozen DB + eval harness, and
> meet only in the final comparison table.
>
> **State going in.** Tier-0 ([tier0.py](../src/tier0.py)), Tier-0-enhanced
> ([tier0_enhanced.py](../src/tier0_enhanced.py) — modality-gap centering nearly doubles R@5), and
> Tier-1 CLAY ([tier1.py](../src/tier1.py) — currently *below* Tier-0, because faithful CLAY has no
> +/− arithmetic) are all done and scored through [eval.py](../src/eval.py). See
> [ROADMAP](ROADMAP.md) §"Tier 2a" and the progress logs ([Davide](Davide.md), [Alfonso](Alfonso.md)).

Both tracks attack the **same gradeable insight** — "−X" is **negation** ("any value but X"), not vector
subtraction — but through mechanisms that never touch the same code: **image-space orthogonal rejection**
(Track V) vs. **text-subspace complement** (Track S). Either one, alone, is a credible method that beats
Tier-0 and Tier-1.

---

## 0. The training-free design space (why the split is exhaustive, not a grab-bag)

Every training-free method is one choice along **four axes**:

1. **Constraint representation** — single prompt → prompt ensemble → **text subspace (SVD)** →
   **image-derived direction/prototype (CAV / GDE)**.
2. **Combination operator** — additive (T0) → subspace projection (T1) → **orthogonal rejection** →
   **score-side late fusion**.
3. **+/− symmetry** — symmetric (T0 / T1) vs **asymmetric** (the Tier-2a point).
4. **Where fusion happens** — query-side (one `q`, one cosine) vs score-side (combine several
   similarities at ranking time).

Enumerated candidates and where each lands:

| # | Candidate | Axis it pulls | Owner |
|---|---|---|---|
| 1 | α/β asymmetric sweep on enhanced T0 | additive, asymmetric weight | *(enhanced-T0 ablation — not a new tier)* |
| 2 | Negation as orthogonal **rejection** of a direction | rejection | **V** (image) |
| 3 | **Image-derived attribute directions** (GDE tangent-mean / CAV) | image representation | **V** |
| 4 | **Visual prototype** retrieval (centroid of train images w/ attr) | image representation, score-side | **V** |
| 5 | Tangent (GDE) vs linear (LDE) composition | manifold geometry | **V** (ablation) |
| 6 | **Asymmetric +/− text subspaces** (separate SVD per polarity) | subspace, asymmetric | **S** |
| 7 | **Per-condition subspaces** (kill the naïve pre-SVD stack) | subspace representation | **S** |
| 8 | Negation as **subspace intersection / complement** (SpaceVLM) | subspace, asymmetric | **S** |
| 9 | Singular-value weighting / all-but-the-top | subspace refinement | **S** (ablation) |
| 10 | Score-side late fusion (`identity + sat⁺ − viol⁻`) | fusion location | both, internally |

Candidates **2–5** form one complete **image-space** pipeline (Track V); **6–9** form one complete
**text-space** pipeline (Track S). Candidate 1 stays an enhanced-T0 ablation. This is the balanced,
non-overlapping cut.

---

## 1. Track V — Visual-Prototype Compositional Retrieval  *(Davide / Member B)*

**Idea (one line).** Do the *entire* composition in CLIP **image space**, where there is no modality
gap: represent every attribute by a direction/prototype **mined from CelebA-train images**, push
`v_ref` toward the positives and **orthogonally reject** the negatives on the sphere's tangent space,
then score by cosine. This is the assigned GDE paper turned into a retrieval method.

**What.** A self-contained scorer `src/tier2a_visual.py` exposing the same
`make_get_ranking(query_str, …) → get_ranking(src_idx)` seam as every other tier
([CONTRACT](CONTRACT.md) §7), writing `output/tier2a_visual_*.csv` in the shared CSV schema.

**Why.** The diagnosed root weakness of Tier-0 / Tier-1 is the **modality gap** — text vectors are
yanked into a different cone of the sphere ([Alfonso.md](Alfonso.md), the α=0/α=1 analysis; T0-enhanced
FIX 1 confirmed centering is "the entire win"). Track V removes the gap *by construction*: an attribute
direction computed from images-that-have-X minus images-that-don't lives natively in the **image** cone,
so there is nothing to center and no rotation `H` to fudge. GDE shows these visual primitive directions
are *geodesically decomposable* and beat their linear counterpart — strong reason to expect this clears
both baselines.

**How.**
1. **Mine visual attribute directions (GDE Prop. 1/2).** From `clip_image_features_train.pt` + train
   attributes, for each attribute `a`: take the intrinsic (tangent) mean of train embeddings with
   `a=1` and with `a=0`; the primitive direction is
   `v_a = log_μ(mean₊) − log_μ(mean₋)` on the tangent space at the global mean `μ`.
   *(Linear / LDE variant: skip the log-map — kept as an ablation.)*
2. **Positive composition.** In the tangent space at `v_ref`, add `Σ α·v_a⁺`, then `Exp`-map back to the
   sphere (GDE additive composition).
3. **Negation by rejection — the gradeable insight, image-space.** For each `−a`, **remove** the `v_a`
   component instead of subtracting it: `q ← q − (q·v̂_a)·v̂_a`. "−red hair" deletes the red-hair axis;
   any other hair colour survives — exactly the "any value but X" semantics that plain subtraction
   (anti-X overshoot) gets wrong.
4. **Score.** Cosine of the composed query against the frozen DB; exclude the source
   ([CONTRACT](CONTRACT.md) §5).
5. *(Optional score-side variant.)* Rank by `cos(v_d, v_ref) + Σ proj₊ − Σ |proj₋|`, keeping identity
   and constraints as separate, independently-weighted terms.

**New artifact (keeps the track self-contained).** `clip_image_features_train.pt` +
`celeba_attributes_train.pt` — a one-time Colab extraction. Already listed as Member B's deferred task
in the [ROADMAP](ROADMAP.md), so it falls naturally in this lane and needs **no** input from Alfonso.

**Papers to read first.**
- **GDE** — Berasi et al., CVPR 2025 ([CLAY_compositionality.md](CLAY_compositionality.md), assigned) —
  §3.2 / §3.3, Prop. 1–2 are the exact formulas.
- **Trager et al.**, "Linear Spaces of Meanings", ICCV 2023 — the linear (LDE) baseline to ablate against.
- **Oldfield et al.**, "Parts-of-Speech-Grounded Subspaces", NeurIPS 2023 — PGA on the CLIP sphere.
- **Kim et al.**, "TCAV" — the concept-activation-vector framing for image-derived directions.
- **Alhamoud et al.**, "VLMs Do Not Understand Negation", 2025 — motivates the rejection step.

**Ablations (report rows).** #train images per attribute `k`; GDE (tangent) vs LDE (linear); `α` step
size; rejection ON vs naïve subtraction OFF; score-side vs query-side fusion.

**Deliverables.** `src/tier2a_visual.py`; `output/tier2a_visual_*.csv`;
`test/test_tier2a_visual.py` (log/exp round-trip, rejection orthogonality, end-to-end ranking — mirror
[test/test_tier1.py](../test/test_tier1.py)); report sub-section *"Visual-prototype composition"*.

---

## 2. Track S — Asymmetric Conditional Subspaces  *(Alfonso / Member A)*

**Idea (one line).** Stay in CLAY's text-subspace machinery but fix its two flaws — it stacks +/− into
one naïve SVD and has no negation. Build a **separate positive subspace** and **negative subspace**, and
retrieve in the **intersection of "inside S⁺" and "outside S⁻"**. Pure linear algebra on the
already-cached tensors.

**What.** A self-contained scorer `src/tier2a_subspace.py` with the same `make_get_ranking` seam,
writing `output/tier2a_subspace_*.csv`.

**Why.** Tier-1 sits *below* Tier-0 precisely because faithful CLAY merges + and − prompts into one SVD
and only does focus/preserve — no modification, no negation ([Davide.md](Davide.md), Tier-1 results).
Splitting the subspaces by polarity restores the asymmetry the task demands, and modelling "−X" as the
**orthogonal complement** of the X-subspace is the training-free negation operator (SpaceVLM's result).
This converts CLAY's deficit into a method that should clear both baselines, while staying 100 % offline
(no new artifact).

**How.**
1. **Per-polarity subspaces.** From [clip_attr_prompt_bank.pt](../src/clip_prompts.py): log-map the
   **positive** prompts onto their tangent mean, SVD → `V_k⁺` (CLAY subspace, positives only);
   independently do the **negative** prompts → `V_k⁻`. No naïve merge → kills the pre-SVD bottleneck.
2. **Positive focus.** Project the DB into `span(V_k⁺)` and score reference↔DB cosine there (CLAY's
   conditional-similarity reframing, restricted to what must be *satisfied*).
3. **Negation by complement — the gradeable insight, text-space.** Penalise energy *inside* the negative
   subspace: composite score `= cos_{S⁺}(v_ref, v_d) − λ·‖proj_{S⁻}(v_d)‖`. Equivalently, retrieve in
   `S⁺ ∩ (S⁻)^⊥`. "Not X" = the component orthogonal to the X-subspace, so any non-X value passes.
4. **Per-condition option.** For multi-condition queries, build one subspace per condition and combine by
   intersection rather than one stacked SVD (the explicit fix to CLAY's bottleneck) — the headline
   variant vs. a stacked-positives ablation.
5. **Reuse the frozen manifold helpers** `_log_map` / `_align_rotation` from [tier1.py](../src/tier1.py)
   (read-only — they will not change). *(Clean-architecture note, optional: lift them into a shared
   `src/manifold.py` that both `tier1.py` and this module import, per [CLAUDE.md](../CLAUDE.md)'s
   no-sideways-import rule. Flag it; don't block on it.)*

**Artifacts.** None new — consumes the cached `clip_image_features_test.pt` + `clip_attr_prompt_bank.pt`.
Runs fully offline, no Colab.

**Papers to read first.**
- **CLAY** — Lim et al., CVPR 2026 ([CLAY.md](CLAY.md), assigned) §3.2 — the subspace + rotation pipeline.
- **SpaceVLM** — "Sub-Space Modeling of Negation in VLMs", 2025 — negation as subspace intersection,
  training-free; almost exactly step 3.
- **Oldfield et al.**, "Parts-of-Speech-Grounded Subspaces", NeurIPS 2023 — PGA subspaces.
- **"When Negation Is a Geometry Problem in VLMs"**, 2026 — geometric negation framing.
- **Trager et al.**, ICCV 2023 — linear-subspace lineage.

(All five are already catalogued in [SOTA.md](SOTA.md) §2 / §5.)

**Ablations (report rows).** `k⁺` / `k⁻` subspace dims; rejection weight `λ`; per-condition vs
stacked-positives; rotation `H` ON / OFF (reuses Tier-1's existing toggle).

**Deliverables.** `src/tier2a_subspace.py`; `output/tier2a_subspace_*.csv`;
`test/test_tier2a_subspace.py` (subspace orthonormality / idempotence, complement projection, end-to-end
ranking); report sub-section *"Asymmetric conditional subspaces"*.

---

## 3. Why this split is balanced and fully parallel

| | **Track V** (Davide) | **Track S** (Alfonso) |
|---|---|---|
| Space | image (no modality gap) | text (CLAY subspace) |
| Heavier cost | **data** — 1 Colab extraction of train features | **math** — subspace intersection / complement |
| Lighter cost | math reuses GDE closed forms | data — zero new artifacts |
| Shared code | none (own module, tests, CSVs) | reuses *frozen* `tier1` helpers only |
| Negation mechanism | orthogonal rejection of a visual direction | orthogonal complement of a text subspace |
| Beats baselines via | killing the modality gap at the source | restoring the +/− asymmetry CLAY lacks |

Each member builds, runs, scores, and writes up their track end-to-end **without waiting on or talking
to the other**. The only convergence point is Phase-C's master comparison table (all tiers × K∈{1,5,10}
× 14 queries), which the shared [eval.py](../src/eval.py) + uniform CSV schema make a mechanical merge.

---

## 4. Definition of done

- **For this document:** both members can read their section and start coding with **zero further
  questions**; the two tracks share **no source file**.
- **For each built track:** the scorer plugs into `evaluate_all()` and emits `output/tier2a_*_*.csv` in
  the same schema as `tier0_*.csv` / `tier1_*.csv`; success = **strictly beats Tier-0 R@5** (ideally
  Tier-1), with the negation queries (e.g. `-Male, -Mustache`, currently **0.000** under every method)
  no longer collapsing — the concrete signal that the asymmetric-negation insight works.
