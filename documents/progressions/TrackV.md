# Track V — Theory You Must Understand Before Coding

> **Scope.** This document explains the *theory* behind the five papers backing Track V
> (Visual-Prototype Compositional Retrieval). No implementation here — only the ideas,
> formulas, and the chain of reasoning that makes the method correct. When every section
> below reads as "obvious", you are ready to write [tier2a_visual.py](../src/tier2a_visual.py).
>
> **The one-sentence thesis of Track V.** Do the whole composition in CLIP **image space**
> (no modality gap), represent each attribute by a **direction mined from CelebA-train images**,
> push the reference toward positives by **geodesic addition**, handle "−X" by **orthogonal
> rejection** (delete the axis) instead of subtraction, then cosine-score the frozen DB.
>
> **How the five papers fit together.** [GDE](papers/GDE.md) is the spine — it gives the exact
> formulas for mining directions and composing them on the sphere. [Trager](papers/VLM.md) is
> the flat (linear) special case you ablate against. [TCAV](papers/TCAV.md) justifies "attribute
> = a direction learned from has-X vs not-X". [PoS-Subspaces](papers/PSGS_VLM.md) supplies the
> *projection / orthogonal-complement* mechanics and the proof that the sphere needs tangent-space
> geometry. [VLMs-Don't-Understand-Negation](papers/VLM_DO_NOT_UNDERSTAND_NEGATIONSpdf.md) is the
> *why* of the gradeable insight — it diagnoses the exact failure that rejection fixes.

---

## 0. The problem, stated geometrically

A composition query is "image like `v_ref`, **with** attributes X⁺, **without** attributes X⁻"
(e.g. CelebA: *"this person but smiling, and not male, and not wearing a mustache"*). Three sub-problems:

1. **Represent each attribute** as something you can add to / remove from a vector.
2. **Add the positives** — move `v_ref` *toward* "has X⁺".
3. **Negate the negatives** — and this is the crux: what does "not X" *mean* as a vector operation?

Tier-0/Tier-1 answered (1) with **text** embeddings and (2)/(3) with **+/− vector arithmetic** in
the **ambient** space. Both choices are geometrically wrong on CLIP, for two independent reasons that
Track V fixes at the source:

- **The modality gap (fixed by choice of space).** Image and text embeddings occupy two
  *separated cones* of the sphere. Adding a raw text vector to an image vector pushes the query
  off the image cone, so cosine to the (image) DB degrades. → Track V mines directions from
  **images**, so everything lives in the image cone. Nothing to center, no rotation `H` to fudge.
- **Negation ≠ subtraction (fixed by choice of operator).** "−X" should mean *"any value but X"*,
  not *"the opposite of X"*. Subtracting `v_X` overshoots into anti-X. → Track V **rejects**
  (deletes the X-axis), which is the correct "any value but X" semantics.

Everything below is the machinery to make those two fixes precise.

---

## 1. CLIP lives on a sphere — so geometry is spherical, not flat

**Source: [GDE](papers/GDE.md) §3.1; [PoS-Subspaces](papers/PSGS_VLM.md) §2.2.**

CLIP normalizes every embedding, so `‖u‖ = 1`: all embeddings live on the unit hypersphere
`S^{d−1} ⊂ ℝ^d`. The norm carries no information; only **direction** does, and similarity is cosine
`u_xᵀ u_y`. The consequence you must internalize: **straight-line (Euclidean) operations are not
faithful on a sphere.** A straight line between two unit vectors cuts *through* the ball and leaves
the surface; the faithful notion of "straight" on a sphere is the **geodesic** (great-circle arc).

To do linear algebra correctly on the sphere you move to the **tangent space** at a reference point
`μ`, do flat operations there, then map back. Two maps, both closed-form on the sphere:

- **Logarithmic map** `Log_μ(u)` — sends a point `u` on the sphere to a vector in the tangent plane
  `T_μ S^{d−1}` at `μ`. Think: "the initial velocity you'd set off with from `μ` to reach `u` along
  a great circle." Closed form ([GDE](papers/GDE.md) Eq. 14):
  ```
  Log_μ(u) = θ · (I − μμᵀ)(u − μ) / ‖(I − μμᵀ)(u − μ)‖ ,   θ = arccos(uᵀμ)
  ```
  `(I − μμᵀ)` is the projector that removes the component along `μ` — i.e. it lands you *in the
  tangent plane*. `θ` is the geodesic distance (the angle). So `Log_μ(u)` is "unit tangent
  direction toward `u`, scaled by the angle to `u`".
- **Exponential map** `Exp_μ(v)` — the inverse: takes a tangent vector and walks along the geodesic
  back onto the sphere. Closed form ([GDE](papers/GDE.md) Eq. 13):
  ```
  Exp_μ(v) = cos(‖v‖)·μ + sin(‖v‖)·(v/‖v‖)
  ```

**Three properties you will rely on (and should test):**

- **Round-trip:** `Exp_μ(Log_μ(u)) = u` (local inverse). Your unit test #1.
- **Distance preservation near `μ`:** `d_S(u,u') ≈ ‖Log_μ(u) − Log_μ(u')‖` ([GDE](papers/GDE.md)
  Eq. 1), *exact* when `u` or `u'` equals `μ`. This is *why* it's legitimate to do flat averaging in
  the tangent space — it approximates the true spherical objective.
- **Validity domain:** `Log_μ` is undefined at the antipode `−μ` (the "cut locus"). On CLIP this
  never bites because of the **cone effect** — all embeddings sit within a small geodesic ball
  (angle `< π/2` from the mean), which [GDE](papers/GDE.md) verifies empirically (Appendix C.1).

> **Intrinsic mean `μ`.** The natural tangent point is the **intrinsic (Karcher) mean** — the point
> minimizing average squared *geodesic* distance ([GDE](papers/GDE.md) Eq. 2), not the arithmetic
> mean. Its defining property is the one that makes all the algebra clean: it **centers** the
> log-mapped points, `Σ_i w_i Log_μ(u_i) = 0`. Computed by a short gradient descent
> ([GDE](papers/GDE.md) Alg. 1), initialized at the normalized arithmetic mean. Understand this
> centering property — it's what guarantees the decomposition below is unique.

---

## 2. An attribute is a *direction* — TCAV's licence, GDE's construction

### 2.1 Why "attribute = direction" is legitimate (TCAV)

**Source: [TCAV](papers/TCAV.md) §3.2.**

TCAV's reusable idea (ignore the directional-derivative / saliency machinery — not needed here):
a human concept corresponds to a **Concept Activation Vector (CAV)** — a single direction in
activation space that separates *examples-with* the concept from *examples-without*. TCAV learns it
as the normal to a linear classifier's boundary between `{activations of concept examples}` and
`{activations of random/negative examples}`.

Track V uses the **simpler, equivalent-in-spirit** estimator: the **difference of class means**
(`mean₊ − mean₋`), which is the closed-form direction separating the two sets. The takeaway you need
from TCAV is conceptual permission: *a concept like "smiling" or "mustache" is faithfully captured by
one direction mined from labeled images*, and that direction is meaningful enough to sort/score by
cosine (TCAV §4.1.1 sorts images by cosine-to-CAV — exactly your scoring primitive). CelebA's binary
attribute labels are precisely the "with / without" supervision a CAV needs.

### 2.2 The faithful construction is geodesic, not linear (GDE)

**Source: [GDE](papers/GDE.md) §3.2–3.3, Prop. 1–2.**

GDE is the rigorous, sphere-aware version of "attribute = direction", built for exactly our setting
(directions mined from *images*, which are noisy and sparse). The core object:

> **Geodesic decomposability** ([GDE](papers/GDE.md) Def. 1). A set of composite embeddings is
> *geodesically decomposable* if each one is `u_z = Exp_μ(v_{z₁} + … + v_{z_s})` — i.e. it's the
> exp-map of a **sum of per-primitive tangent directions**. Addition happens in the tangent space;
> the manifold composition is the exp-map of that sum.

So the primitive direction `v_a` for attribute `a` is a **tangent vector at `μ`**, and composing
attributes = **adding their tangent vectors then exp-mapping**. The optimal directions have a clean
closed form — they are **tangent means** ([GDE](papers/GDE.md) Prop. 1, Eq. 7):

```
v_a = mean over images with attribute a of  Log_μ(u_image)
```

i.e. log-map every embedding, average the ones in class `a`, and that average tangent vector *is* the
direction. (For a two-axis grid like attribute×object, `v_a` is the mean over the slice sharing `a`.)

**The two image-specific complications GDE solves for you — know they exist:**

- **Noise** ([GDE](papers/GDE.md) §3.3.1, Prop. 2). A real image of "smiling" also contains hair,
  background, lighting — information *not* in the concept. GDE down-weights noisier images via a
  per-image weight `p`. Simplest choice: uniform `p = 1/k`. Better: the **CLIP image-to-text
  softmax** `P(image | "a photo of {a}")` ([GDE](papers/GDE.md) Eq. 12) — images that match the
  attribute's text prompt count more. The denoised per-tuple vector is the weighted tangent mean
  (Eq. 10). *This is an ablation axis for you (uniform vs CLIP-weighted).*
- **Sparsity** ([GDE](papers/GDE.md) §3.3.2, Eq. 11). Some attribute combos have no images. Doesn't
  matter: because directions are *per-primitive* averages, you recover a direction for **every**
  primitive (and thus every composition) as long as each primitive appears in *some* image. Relevant
  if you ever build multi-attribute prototypes.

### 2.3 The headline empirical result that justifies the whole track

**Source: [GDE](papers/GDE.md) §4.3 (and the contrast with [Trager](papers/VLM.md)).**

GDE proves image embeddings are **not** well-approximated by *linear* (flat) decomposition — the
geodesic version wins decisively. On UT-Zappos compositional classification, GDE (image) reaches
AUC 13.9 vs 4.4 for zero-shot CLIP (~318% relative); on group robustness (Waterbirds/CelebA) it beats
even task-specific trained methods, **training-free**. The flat counterpart (LDE) "performs much worse
than GDE on both datasets and across backbones." This is your evidence that (a) image-space directions
carry real semantic content, and (b) **respecting the sphere geometry matters** — it's not a cosmetic
detail. CelebA appears in GDE's own experiments, so the setting transfers directly.

---

## 3. Composition step 1 — adding the positives (geodesic addition)

**Source: [GDE](papers/GDE.md) Def. 1 / §4.5.**

Once each positive attribute has a tangent direction `v_a⁺`, "make `v_ref` more X⁺" is:

```
q = Exp_{μ}( Log_{μ}(v_ref) + Σ_i α · v_{a_i}⁺ )
```

Read it as: lift the reference into the tangent plane, **add** the (weighted) positive directions
there — this is the flat operation the sphere *permits* via the log/exp sandwich — then walk back onto
the sphere. `α` is the push strength (an ablation knob). GDE §4.5 demonstrates this additive
composition produces semantically correct composite embeddings (it even blends two objects via
`Exp_μ(v_{o₁} + v_{o₂})`), which is your evidence the positive step alone behaves sensibly.

> **Tangent point detail.** GDE composes at the global intrinsic mean `μ`. Composing at `μ` vs at
> `v_ref` is a modeling choice — `μ` keeps all directions in one consistent tangent frame and is the
> faithful reading of GDE. Mention the choice in your write-up; don't agonize over it.

---

## 4. Composition step 2 — negation as **rejection** (the gradeable insight)

This is the part the project is actually grading. It rests on two papers: one tells you *why naïve
negation fails*, the other gives you *the projection operator that fixes it*.

### 4.1 Why subtraction is the wrong operator (Alhamoud et al.)

**Source: [VLMs-Don't-Understand-Negation](papers/VLM_DO_NOT_UNDERSTAND_NEGATIONSpdf.md) §4.1.**

The paper's diagnosis, which is the motivation for your whole negation step:

- CLIP-like models exhibit an **affirmation bias** — they "frequently collapse affirmative and
  negated statements into similar embeddings, treating 'a dog' and 'no dog' as nearly
  indistinguishable." On negation MCQs they perform **at or below chance** (e.g. 82% on affirmation
  vs **3%** on negation for VOC2007). Scaling up (ViT-B→L→H, SigLIP, AIMV2) does **not** fix it.
- The PCA analysis (their Fig. 6) shows the geometry: a well-behaved space should separate captions
  along *two distinct axes* — an "object type" axis and a "negation" axis. CLIP fails to; the
  models that *work* place "flowers but not cats" **along the line connecting 'flowers' and
  'not cats'** — i.e. negation is a *direction to move along/remove*, not a point to subtract toward.

The lesson for Track V: don't trust the text encoder to understand "not X", and don't model "not X"
as `−v_X` (which lands you at *anti-X*, an overshoot). Model it as **removing the X-axis entirely**,
so every non-X value survives equally. (The paper's own fix is data-centric fine-tuning — *not* your
route; you take their *diagnosis*, not their solution, and answer it training-free with geometry.)

### 4.2 The operator that does it — orthogonal rejection / complement (PoS-Subspaces)

**Source: [PoS-Subspaces](papers/PSGS_VLM.md) §2.2, Eq. 5.**

Oldfield et al. give the exact, geometry-aware projection mechanics you need. For a subspace spanned
by an orthonormal `W` (for a single attribute, `W = v̂_a`, one unit direction), on the **sphere**:

- **Isolate / keep** the attribute (project *onto* the axis):
  `Π(z)  = Exp_μ( W Wᵀ Log_μ(z) )`
- **Remove / negate** the attribute (project onto the **orthogonal complement**):
  `Π⊥(z) = Exp_μ( (I − W Wᵀ) Log_μ(z) )`     ← **this is "−X"**

For Track V's single-direction case, `Π⊥` is exactly the rejection:

```
q  ←  Exp_μ( Log_μ(q) − (Log_μ(q)·v̂_a) · v̂_a )
```

"Delete the component of `q` along the red-hair axis." The result is **orthogonal to `v̂_a`**: it
encodes *no* preference on that attribute, so "any other hair colour survives" — the precise "any
value but X" semantics §4.1 demanded. Two facts make this the right citation, not just a convenient
one:

1. **It's done in the tangent space**, log-map → linear projection → exp-map — the same sphere-faithful
   sandwich as everything else (PoS-Subspaces §2.2: "an orthogonal projection onto the learnt
   Euclidean subspaces is not guaranteed to result in a vector that remains on the sphere"; their fix
   is to project in `T_μ S^{d−1}`). Doing the rejection in ambient space would silently leave the
   sphere — a bug their geometry rules out.
2. **They prove the complement-projection erases a concept while preserving the rest.** Their
   flagship result: projecting onto the *orthogonal complement* of an attribute/style subspace
   removes that attribute from CLIP-based generation while leaving content intact (Fig. 3–5), and
   *improves* zero-shot classification by isolating the noun subspace (Table 1, 14/15 datasets). That
   is empirical proof that "remove a direction" cleanly deletes a semantic factor — exactly the
   guarantee your "−X" needs.

> **Multi-attribute / score-side variants (optional, ablation rows).** PoS-Subspaces is stated for a
> `k`-dimensional `W`, so if you ever want to negate a *subspace* of an attribute rather than a single
> direction, the same `Π⊥` applies with a wider `W`. And the score-side form
> `score = cos(v_d, v_ref) + Σ proj₊ − Σ |proj₋|` keeps identity and each constraint as separate,
> independently-weighted terms — a fusion-location ablation, conceptually the same primitives.

---

## 5. The ablation against the linear world (Trager et al.)

**Source: [Trager, "Linear Spaces of Meanings"](papers/VLM.md) §3, Prop. 4.**

Trager is the **flat predecessor** of GDE and your LDE baseline. Same idea — composite ≈ sum of
per-factor vectors — but in **ambient Euclidean space**, no sphere:

> **Decomposable embeddings** ([Trager](papers/VLM.md) Def. 1): `u_z ≈ u₀ + u_{z₁} + … + u_{z_k}`.
> The **ideal words** `u_{z_i}` are recovered as plain **arithmetic means** of the embeddings sharing
> that factor ([Trager](papers/VLM.md) Prop. 4) — the *centered* version (`Σ u_{z_i} = 0`,
> [Trager](papers/VLM.md) Lemma 3) makes them unique, the direct analogue of GDE's centering.

GDE states explicitly that it **reduces to Trager when `M = ℝⁿ`** — there the intrinsic mean *is* the
arithmetic mean and log/exp *are* the identity. So your **LDE ablation = Track V with the log/exp maps
removed**: mine directions as raw `mean₊ − mean₋` in ambient space, add/subtract linearly, score. The
point of running it is to *reproduce GDE's central finding on your own CelebA setup*: if GDE (tangent)
beats LDE (linear), you've shown the sphere geometry is doing real work, not decoration.

Two further things worth lifting from Trager for the report:

- **Compositional retrieval already works cross-modally with mean differences** (their DeepFashion2
  result, §5): replacing a text concept with `Mean{v(CONCEPT)} − Mean{v(generic)}` — an *image*-mean
  difference added to text — beats a trained method (PALAVRA), **training-free**. Direct precedent
  that image-derived mean-difference directions are retrieval-grade.
- **Ideal words ≠ word embeddings.** The mined direction for "smiling" is *not* the text embedding of
  "smiling"; it's defined purely by how the factor varies across the data. Same for your visual
  directions — keep the distinction clear when writing up.

---

## 6. The reasoning chain, end to end

Read this as the argument your report must make; if any link is fuzzy, re-read the cited section.

1. CLIP embeddings are unit vectors on a sphere; only direction/cosine matters → spherical geometry,
   tangent-space algebra via Log/Exp. **(§1 — GDE §3.1, PoS §2.2)**
2. A concept is faithfully a single direction separating has-X from not-X. **(§2.1 — TCAV §3.2)**
3. Mined from **images**, that direction lives in the image cone → the modality gap that crippled
   Tier-0/1 is gone *by construction*. **(§0 + §2 — GDE motivation)**
4. The *correct* such direction is the **geodesic tangent mean**, and image directions are provably
   **not** linearly decomposable — geometry matters. **(§2.2–2.3 — GDE Prop. 1–2, §4.3)**
5. Positives compose by **tangent addition + exp-map**. **(§3 — GDE Def. 1)**
6. Negation must be **"any value but X"**, because CLIP's affirmation bias makes subtraction collapse
   / overshoot. **(§4.1 — Alhamoud §4.1)**
7. The operator delivering that is **orthogonal-complement projection in the tangent space**, proven
   to erase one factor while preserving the rest. **(§4.2 — PoS §2.2, Eq. 5)**
8. Stripping the Log/Exp maps gives the **linear (LDE) ablation**; beating it confirms geometry is the
   win. **(§5 — Trager §3, Prop. 4)**
9. ⇒ A query composed this way, cosine-scored against the frozen DB, should **beat Tier-0 and Tier-1**,
   and in particular stop the negation queries (e.g. `−Male, −Mustache`) from collapsing to 0.000 —
   the concrete success signal from [TRAINING_FREE_SPLIT](TRAINING_FREE_SPLIT.md) §4.

---

## 7. What each paper gives you — quick reference

| Paper | File | Role in Track V | Read | What you extract |
|---|---|---|---|---|
| **GDE** (Berasi et al., CVPR 2025) | [GDE.md](papers/GDE.md) | **Spine** — the method | §3.1–3.3, Prop. 1–2, App. A (Eq. 13–14) | Log/Exp closed forms, intrinsic mean, tangent-mean direction, geodesic addition, noise/sparsity handling |
| **VLMs Don't Understand Negation** (Alhamoud et al., 2025) | [VLM_DO_NOT…md](papers/VLM_DO_NOT_UNDERSTAND_NEGATIONSpdf.md) | *Why* negation ≠ subtraction | §4.1 + Fig. 6 | Affirmation bias; negation is an axis to remove, not a point to subtract toward |
| **PoS-Grounded Subspaces** (Oldfield et al., NeurIPS 2023) | [PSGS_VLM.md](papers/PSGS_VLM.md) | The projection / rejection operator | §2.2, Eq. 5 | Tangent-space `Π` (keep) and `Π⊥` (negate); proof that complement erases a factor cleanly |
| **Linear Spaces of Meanings** (Trager et al., ICCV 2023) | [VLM.md](papers/VLM.md) | The **LDE** ablation baseline | §3, Prop. 4, Lemma 3; §5 retrieval | Flat mean-difference directions; centered ideal words; GDE-without-the-maps |
| **TCAV** (Kim et al., ICML 2018) | [TCAV.md](papers/TCAV.md) | Conceptual licence for "attribute = direction" | §3.2 (skip §3.3+) | CAV = direction separating with/without; cosine-sorting by a CAV |

**Reading order:** GDE (deep) → Alhamoud (problem framing) → PoS §2.2 (the operator) → Trager (the
ablation) → TCAV (grounding). Minimum to start coding: **GDE + PoS §2.2 + Alhamoud**. Trager is needed
before the LDE ablation row; TCAV makes the write-up rigorous but doesn't block code.

---

## 8. Before you touch code — comprehension checklist

You understand Track V when you can answer all of these without re-opening the papers:

- [ ] Why does cosine, not Euclidean distance, govern CLIP? Why does that force spherical geometry?
- [ ] What does `Log_μ(u)` return, geometrically? What does `(I − μμᵀ)` do inside it?
- [ ] Why the **intrinsic** mean and not the arithmetic mean? What property does it guarantee?
- [ ] How is the direction `v_a` for an attribute computed, in one sentence? (Tangent mean of has-X.)
- [ ] Why does mining from images kill the modality gap "by construction"?
- [ ] State the positive-composition formula and the negation (rejection) formula from memory.
- [ ] Why is `q − (q·v̂)v̂` "any value but X" and `q − v_X` *not*? What does Alhamoud's PCA show?
- [ ] What exactly changes to turn GDE into the LDE ablation, and what would beating LDE prove?
- [ ] Which CelebA queries are the litmus test for the negation insight, and what's the failing number
      they currently produce?

When all nine are reflexive, open [tier2a_visual.py](../src/tier2a_visual.py) and start with the
Log/Exp round-trip test.
