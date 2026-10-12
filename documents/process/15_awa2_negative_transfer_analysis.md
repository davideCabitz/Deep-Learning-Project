# AwA2 Transfer Study — Why the CelebA Ranking Inverts, and What It Reveals About CLIP

> **📜 Process document.** Transfer analysis by Alfonso Antognozzi, July 3, 2026, originally kept in `notebooks/`. "Report §11" below refers to section 11 of [`notebooks/deliverable_report.ipynb`](../../notebooks/deliverable_report.ipynb).

> **⚠ SUPERSEDED (2026-07-04) — read report §11 instead.** The benchmark this document
> analyzes predates the source-policy discovery (a valid source must VIOLATE every ±
> constraint, proven by `generate_custom_gt.py --verify-official`'s exact reconstruction of
> the official CelebA JSON). Every source in it already satisfied its query's constraints,
> so all numbers below measure a **class-retrieval surrogate**, not the spec edit task —
> which is structurally EMPTY on AwA2's class-level attributes (nearest legitimate target
> class sits at free-Hamming 3–21 ≫ the spec radius 2; `src/awa2/generate_gt.py` now
> refuses to freeze it). The GT was renamed `Evaluation/awa2_surrogate_class_retrieval.json`.
> The mechanism analysis below remains valid *for the surrogate task*; the report notebook
> §11 carries the corrected framing.

**Status:** superseded as a transfer study; retained as the surrogate-task mechanism analysis
**Artifacts:** `output/awa2/*.csv`, `output/awa2/per_source/*.npz`, `output/awa2/tier5_diag_*.csv`

---

## 1. The result

We replicated the CelebA evaluation protocol on AwA2 (spec §2): frozen CLIP ViT-B/32
visual DB, a 12-query ±attribute benchmark generated with the spec §3.1.1 relaxed-Hamming
rule (strict ± satisfaction, Hamming ≤ 2 on free attributes, ≥ 5 valid targets, 60
sources per query), Recall/Precision@{1,5,10}. Seven methods, spanning every family we
built for CelebA:

| Mean R@1 | CelebA | AwA2 | family |
|---|---:|---:|---|
| **LDE** (mined visual directions, flat arithmetic) | 0.0224 | **0.8805** | training-free, visual |
| **GDE** (mined visual directions, geodesic + rejection) | 0.0221 | 0.8683 | training-free, visual |
| **Tier-0 enhanced** (text deltas, all fixes) | 0.0397 | 0.8083 | training-free, text |
| **Tier-2d DGP** (τ=100 uniform gate — CelebA's best training-free) | 0.0363 | 0.7653 | training-free, text |
| **Tier-3 DualScore** (trained Φ⁺ + probe channels) | 0.0597 | 0.7569 | trained, CLIP-space |
| **CLAY** (pre-SVD prompt stacking, k=50 rotH) | 0.0098 | 0.7306 | training-free, text |
| **Tier-5 Bprime_listnet** (trained BayesPhi — CelebA's champion) | **0.1104** | 0.6667 | trained, probe-space |

The ranking **inverts almost exactly**: CelebA's champion (tier5) is AwA2's worst
method; CelebA's near-bottom pair (the mined-direction GDE/LDE) is AwA2's top pair. Same
architectures, same training recipes, verified from the logs — this is a property of the
methods' inductive biases, not a bug.

> Note on absolute scales: CelebA (~0.01–0.11) and AwA2 (~0.67–0.88) are not comparable
> currencies — AwA2 is near-ceiling for every method because its ground truth reduces to
> class retrieval (§3). The signal is the ordering flip, not the magnitudes.

## 2. Ruling out the boring explanations

- **Wrong models?** No. Each AwA2 run re-uses the CelebA implementation through a single
  adapter seam (`src/awa2/adapter.py`) that rebinds only data providers (attribute
  tables, loaders, cache paths). The math is byte-identical code.
- **Training failure?** No — tier5's loss decreased smoothly (κ ≈ 0.99, init invariant
  intact) and the diagnostic in §5 shows training *gained* +6 pts over its own base.
  DualScore's two-stage run converged normally (probes 92% val acc).
- **Broken ground truth?** No — validated semantically (`+stripes` targets are exactly
  the raccoon/skunk/tiger/zebra classes) and structurally (schema-identical to
  `celeba_evaluation.json`, consumed by the same `eval.py`).

## 3. The structural difference, mathematically

**CelebA attributes are per-image; AwA2 attributes are class-level** (every image of a
class shares the class's 85-bit row of `predicate-matrix-binary.txt`). This changes the
geometry of the ground-truth set, and with it, which score function can represent it.

**On AwA2 the GT rule collapses to class retrieval.** For source image with class c_s,
the valid-target set is `⋃ {images of class c_t}` over the few classes c_t that satisfy
the constraints and have Hamming(c_s, c_t) ≤ 2. CLIP's visual space clusters animal
photos tightly by species (species are distinct, high-frequency lexical concepts in
CLIP's pre-training), so within-class cosine ≫ between-class cosine. A query of the form
`q = v_ref + small·delta` therefore already ranks the valid set near the top: the
indicator of the GT set is **almost a linear functional of the CLIP embedding**. That is
why every method scores 0.67+, and why methods are ordered by *how much of their score
flows through raw CLIP similarity*.

**On CelebA the GT set is not linearly representable.** Every image is a face — one CLIP
concept — so class-identity margin is near zero; the GT set is defined by a 40-bit
per-image pattern (strict ± bits AND a Hamming ball on 37–39 free bits). A single 512-d
query vector scores by one inner product; a Hamming-ball indicator over ~40 weakly
CLIP-encoded bits is far outside the family of halfspace functions. That is why *all*
cosine-based methods sit at 0.01–0.06 there, and why tier5 — which scores the rule
itself, `Σ_a log p_a + Σ_b log(1−p_b) + log P(Hamming ≤ 2)` — wins on CelebA: it is the
only score whose functional form matches the GT's generative process.

**The inversion follows.** Tier5's Poisson-binomial tail over F ≈ 80 free attributes with
per-attribute probe error ε ≈ 0.14 accumulates an expected F·ε ≈ 11 predicted flips —
an order of magnitude above the Hamming radius of 2 — so on AwA2 the tail term is
essentially probe noise re-ranking images *within* the already-correct class
neighbourhood. The λ-sweep in §5 shows this directly. Conversely, on CelebA the noisy
rule-aware score beats the clean-but-wrong cosine because cosine there is *adversarial*
to the rule (the `-Male,-Mustache` "female twin" failure: 0.000 R@K for every geometry
tier).

## 4. The CLIP problems each method inherited (why they "failed")

All the text-direction methods share the same three CLIP pathologies; AwA2 exposes them
cleanly because the visual-mining alternative sidesteps all three at once.

1. **Modality gap** (Liang et al. 2022). Text and image embeddings occupy two disjoint
   cones on the sphere. A text delta `t̂("horns")`, even after μ_txt centering, has only
   a small component along the *visual* direction that separates horned classes from the
   rest. Effect measured: the baseline's worst queries are exactly the steering-dependent
   ones — `+horns` 0.633, `+longneck` 0.633 — while identity-dominated queries sit at
   0.92+. DGP (τ=100 ≈ prompt-mean directions) inherits the same weakness (0.650/0.667
   on those queries): the gate re-weights *paraphrases of the same text direction*, it
   cannot manufacture visual signal that isn't there.

2. **Non-groundable predicates.** Many of AwA2's 85 predicates (`smart`, `timid`,
   `fast`, `agility`, `newworld`, `oldworld`, ...) are not visible in a single photo.
   For text methods, their prompt embeddings are semantic noise; for probe methods,
   their posteriors are class-prior guesses (probe mean acc 86.4%, min 75%). CelebA's 40
   attributes are, by contrast, almost all directly visible facial properties — the
   probes there both score higher (~88.5%) and, more importantly, carry *per-image*
   information that no other channel has.

3. **No negation semantics** (Alhamoud et al. 2025 affirmation bias). CLIP's text encoder
   cannot represent "not X", so every method approximates negation geometrically —
   subtraction (tier0, LDE), orthogonal rejection (GDE, DGP), or a discriminative absence
   reward (DualScore, tier5). Which approximation wins turns out to be dataset-dependent
   (§6).

CLAY deserves its own line: it stacks all condition prompts and takes SVD, then scores in
the *subspace* — discarding most of the raw identity direction. On AwA2 that is the worst
possible trade (identity IS the signal), which is why faithful CLAY lands second-to-last
at 0.7306 despite being a published method: its pre-SVD stack is precisely the
"naïve bottleneck" the spec asks us to attack, and AwA2 shows the cost from the other
side — it throws away identity, not just expressiveness.

DualScore (0.757) is the instructive middle case: its backbone is CLIP cosine on a fused
q⁺ (so it beats tier5 by 9 points), but its learned additive channels — probe
presence/absence rewards β·s_neg + γ·s_pos — re-introduce probe noise on top of an
already-sufficient cosine, and its trained Φ⁺ composes *text* stacks (modality gap
again). It cannot beat the methods that keep the score purely visual.

## 5. The tier5 diagnostic (mechanism evidence)

Training-free λ-sweep of the Stage A′ closed-form score
(`python src/awa2/tier5_fusion.py --diagnose`):

```
S(x | ref) = constraint(x) + tail(x, ref) + λ · cos(v_x, v_ref)
```

| λ | R@1 | R@5 |
|---|---|---|
| 0 (Bayes core alone) | 0.6069 | 0.7903 |
| 0.5 | 0.6306 | 0.8042 |
| 1 | 0.6458 | 0.8111 |
| 2 | 0.6653 | 0.8306 |
| 4 | 0.7097 | 0.8569 |
| 8 | 0.7653 | 0.8917 |
| *trained Φ (reference)* | *0.6667* | *0.9014* |
| *CLIP baseline (reference)* | *0.8083* | *0.9361* |

Two findings: (1) **training is not the problem** — the trained Φ (0.667) beats its own
λ=0 base (0.607), so the ListNet run worked; the ceiling is the probe-mediated core
itself. (2) **The missing ingredient is exactly CLIP identity** — R@1 rises
monotonically with λ on every query, converging toward the baseline as the score trusts
raw cosine over the probe channels.

> **Honesty caveat:** λ was swept directly on the test benchmark — an *oracle diagnostic*
> that explains the mechanism. Cite the monotonic curve as evidence; do not report the
> best-λ point as a tuned result.

## 6. Why LDE beats GDE (and what that says about the manifold machinery)

First, the relationship (GDE paper, see [reading list](../literature/README.md); [logs/Alfonso.md](../logs/Alfonso.md) §tier2a_visual):
**LDE is not an enhancement of GDE — it is the linear special case GDE generalizes.**
Trager et al.'s linear "ideal words" formulation equals GDE when M = ℝⁿ (intrinsic mean →
arithmetic mean, Exp/Log → identity). In our codebase LDE is kept as a zero-diff ablation
of GDE: same mined directions, same parsing, only the geometric operators differ. So
"LDE wins" literally means **the ablation won — the spherical geometry did not pay for
itself on AwA2** (Alfonso.md had already recorded LDE marginally above GDE at the mean
level on CelebA; AwA2 amplifies the gap to 1.2 points).

Both share the ingredient that makes them win overall: **directions mined in the visual
space**. With class-level labels, the has-a set of attribute a is a union of whole
classes, so the mined direction `v_a = mean(images with a) − μ` is a weighted contrast of
class centroids — by construction the axis that separates valid target classes from the
rest, with zero modality gap. This fixes exactly the text methods' weak queries:
`+horns` 0.633 → 0.917, `+longneck` 0.633 → 0.883.

The GDE−LDE gap decomposes into the two operator differences:

1. **Negation: rejection vs subtraction.** GDE removes the *entire* component of the
   query along v̂_a: `q_tan ← q_tan − (q_tan·v̂_a)v̂_a`. For a broad attribute like
   `quadrapedal` (~34 of 50 classes), v_a points near the dominant direction of animal
   identity itself — the projection `(q_tan·v̂_a)` is large for every reference, so
   rejection amputates much of the identity anchor Log_μ(v_ref). LDE's subtraction shifts
   the query by the *fixed* magnitude ‖v_a‖ (small, since tangent means shrink as the
   has-a set becomes broad and heterogeneous) and leaves the rest of the identity intact.
   Measured: `-quadrapedal` GDE 0.633 vs LDE 0.817 — the single query that accounts for
   most of the mean gap. The CelebA intuition "negation = no preference on the X axis"
   is right when X is a narrow per-image attribute; when X is a broad class-level
   attribute entangled with identity, deleting the axis deletes the anchor.

2. **Curvature handling: Log/Exp maps vs flat arithmetic.** GDE linearizes the sphere at
   one global μ; the log-map scale factor θ/sin θ amplifies with geodesic distance from
   μ. AwA2's corpus spans 50 well-separated species clusters (large θ spread), unlike
   CelebA's single tight "face" cluster, so the tangent-plane approximation is stretched
   exactly where AwA2 lives. For the *coarse* class-retrieval structure of AwA2's GT the
   first-order flat composition (add in ambient space, renormalize) is sufficient and
   introduces no distortion. GDE.md's claim that "linear structures are insufficient for
   visual concepts" concerns recovering fine compositional structure (attribute-object
   decompositions); it does not imply the maps help a retrieval task whose GT is class
   identity.

## 7. Takeaway for the report

One sentence: **on AwA2, method quality is monotone in (a) how much of the score flows
through raw CLIP visual similarity and (b) how visually grounded the attribute
directions are** — visual-mined flat arithmetic (LDE 0.881) > visual-mined geodesic
(GDE 0.868) > text-delta identity (tier0 0.808) > text-direction machinery (DGP 0.765,
DualScore 0.757, CLAY 0.731) > probe-mediated Bayes (tier5 0.667). On CelebA the same
ordering nearly reverses because there the GT rule is per-image, combinatorial, and
cosine-adversarial — the regime the learned fusion was designed for.

The transfer study therefore does what a transfer study should: it locates each method's
inductive bias. Learned rule-scoring fusion (tier5) requires per-image, visually
groundable attributes and an adversarial cosine; identity-plus-delta arithmetic requires
class-clustered embeddings; visual direction mining requires labels whose level
(class vs image) matches the retrieval granularity. None of these is "the best method" —
each is the best method for the attribute structure it assumes.

## 8. Reproduction

```bash
# ground truth + benchmark (needs the raw dataset locally)
python src/awa2/generate_gt.py

# training-free methods (local-safe)
python src/awa2/tier0_baseline.py
python src/awa2/tier_clay.py
python src/awa2/tier_gde.py                 # GDE + LDE ablation (mines directions)
python src/awa2/tier_dgp.py                 # tier2d DGP, CelebA champion config

# trained methods (VM)
python src/awa2/tier5_fusion.py             # Bprime_listnet (~40 min GPU)
python src/awa2/tier_dualscore.py           # DualScore (~5 min GPU)

# diagnostic λ-sweep (training-free)
python src/awa2/tier5_fusion.py --diagnose

# statistical comparison (bootstrap CIs + Holm-corrected paired tests)
python src/awa2/compare.py
```
