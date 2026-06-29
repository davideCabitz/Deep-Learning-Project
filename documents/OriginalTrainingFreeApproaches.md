# Original Training-Free Approaches for Compositional Image Retrieval

Proposals for novel training-free methods that go beyond our current tiers. Each is designed to be genuinely original in the context of our pipeline and the cited literature (GDE, CLAY, PoS-Subspaces, SpaceVLM).

---

## Approach 1 — Reference-Conditional Subspace Reweighting ("Adaptive CLAY")

### Core Idea

Standard CLAY builds the conditional subspace `V_k` exclusively from the condition prompts, making it **source-agnostic**: every reference image `v_ref` gets projected into the same subspace regardless of what it looks like. This is geometrically wasteful — some singular directions of the text subspace will be highly aligned with `v_ref`, others nearly orthogonal to it. Directions orthogonal to the reference carry no retrieval signal and dilute the scoring.

The fix: **reweight the singular vectors by their alignment with the specific reference image before projecting the database**. The subspace tilts toward the dimensions most salient for this particular reference, while still being anchored to the textual condition.

### Formal Description

Let the standard CLAY subspace construction yield `V_k ∈ ℝ^{d×k}` (columns are right singular vectors of `log_{mu_c}(T_c)`). For a given reference `v_ref`, compute the alignment of each singular direction with the reference's tangent coordinate:

```
r = V_kᵀ · log_{mu_c}(H · v_ref)      # [k] — projection of v_ref onto each basis direction
w = softmax(r² / τ)                     # [k] — temperature-scaled squared alignment weights
V_k_adapted = V_k · diag(w^{1/2})      # [d, k] — reweighted basis (columns rescaled, not rotated)
```

The adapted scoring then proceeds as in standard CLAY:

```
D_adapted = normalize( log_{mu_c}(H · v_d) @ V_k_adapted )    for all v_d in DB
score(v_ref, v_d | c) = cos( D_adapted[ref], D_adapted[d] )
```

Temperature `τ` controls how sharply the subspace concentrates: `τ → ∞` recovers standard CLAY; `τ → 0` collapses to a rank-1 projection onto the most reference-aligned direction.

### Why It Is Original

- CLAY's subspace is query-aware (text conditions) but **source-blind** — this is the first method to make the subspace source-conditional.
- The reweighting is a diagonal rescaling of the basis, not a rotation, so the subspace stays within CLAY's span. No new directions are introduced; only their relative importance shifts. This makes the method a strict generalization of CLAY.
- None of GDE, CLAY, PoS-Subspaces, or SpaceVLM perform source-adaptive subspace reweighting.

### ⚠ Architectural Flaw — Violates CLAY's Symmetric-Cost Guarantee

This approach is **invalid as stated**. CLAY's core efficiency property is that `m_CLAY(v_d | c)` depends on the query condition `c` only through `V_k`, which is built from text alone. The entire DB can therefore be projected once per query and cached: every source image is then a free `O(k)` lookup into that cached `D`. Per-source cost is `O(k)`, not `O(N·d)`.

`V_k_adapted` depends on `v_ref`, so `D_adapted` must be recomputed for every `(query, source)` pair. That reinstates `O(queries × N × d)` projection work — exactly the asymmetric-cost structure CLAY eliminates (see CLAY Table 4, symmetric vs asymmetric retrieval cost).

**Weakened salvage:** reweight `V_k` by the mean reference embedding across all sources of a given query (query-class-conditional, not source-instance-conditional). This preserves the one-projection-per-query structure at the cost of losing per-source adaptation. It is a weaker, less original method and should be treated as an ablation, not the headline approach.

**This approach is retired as a primary proposal.** Approaches 2–4 do not share this flaw.

---

## Approach 2 — Negative Complement Subspace Projection ("NCS")

### Core Idea

Track S (our current best) penalizes images for having energy inside the negative attribute subspace `S⁻`, but still **scores retrieval in the full ambient space**. The penalty is additive and soft. A fundamentally stronger operation is to **remove the negative concept from the retrieval space entirely** before scoring — project every database vector onto the orthogonal complement of `S⁻`, then run positive scoring in this filtered space.

This turns "not Male" from a soft penalty into a hard geometric constraint: the database is viewed through a lens where the Male axis does not exist.

### Formal Description

For each negative attribute `b ∈ T_neg`, build its subspace `S⁻_b` with basis `Q_b ∈ ℝ^{d×k_neg}` (columns orthonormal, from PGA of `b`'s prompt stack). When multiple negative attributes are present, stack and orthogonalise their bases once:

```
Q_all, _ = qr( [Q_{b1} | Q_{b2} | ...] )    # [d, k_total], orthonormal columns
Π⊥(v_d) = v_d − Q_all (Q_allᵀ v_d)         # project onto complement — DO NOT renormalize (see below)
```

Positive scoring then runs on the raw filtered vectors:

```
query_filtered = Π⊥(v_ref) + Σ_{a∈T_pos} Π⊥(t̂_a)
score(v_ref, v_d | c) = (query_filtered · Π⊥(v_d)) / (||query_filtered|| · ||Π⊥(v_d)||)
```

### ⚠ Critical Flaw in the Naive Formulation — Do Not Normalize After Projecting

The obvious implementation `v_d_filtered = normalize(Π⊥(v_d))` is **actively harmful** and must not be used. The argument:

- Before filtering, two vectors `v_a`, `v_b` may have cosine similarities 0.16 and 0.05 to a query. After complementary projection but *before* renormalization, that gap is preserved — because the vector most loaded on the negative axis loses most of its norm, while the lightly-loaded one loses almost none.
- `normalize()` then rescales both back to unit length, amplifying whatever residual content survived in the heavily-loaded vector by a large factor. In a toy numerical check, post-filter pre-normalize similarities of 0.16 vs 0.05 collapse to 0.5012 vs 0.5012 after renormalization — **identical to four decimal places**.
- The mechanism: stripping a large chunk of a vector's norm and forcing it back to unit length amplifies whatever survives by exactly the suppression factor. Images most entangled with the negative attribute get the largest amplification — the opposite of the intended suppression.
- Practical consequence: an image that was 95% male-coded and 5% other content ends up post-normalize looking nearly identical to an image that was 5% male-coded and 95% other content, even though they were clearly distinguishable before filtering and only one has real residual signal.

### Three Defensible Alternatives

**Option A — Raw inner product in the complement (recommended).** Do not normalize `Π⊥(v_d)` at all. Compute the score as:

```
score = (Π⊥(query) · Π⊥(v_d)) / (||Π⊥(query)|| · ||Π⊥(v_d)||)
```

This is the standard cosine similarity computed inside the complement subspace rather than the full ambient space. The length of `Π⊥(v_d)` relative to `||v_d||` = 1 is the suppression fraction — it tells you how much of this image's identity survived the cut. That information stays visible in the denominator and is used correctly by the cosine formula, not discarded.

**Option B — Original-norm renormalization.** Renormalize using the pre-filter unit norm as reference:

```
v_d_filtered = Π⊥(v_d) / ||v_d||    # = Π⊥(v_d) since ||v_d|| = 1 for unit-norm DB
```

For a unit-norm DB this is equivalent to Option A's numerator — the filtered vector's new norm `||Π⊥(v_d)||` directly encodes the suppression fraction. Vectors mostly about the suppressed attribute end up short; vectors barely affected end up near unit length.

**Option C — Explicit suppression score as auxiliary signal.** Treat the suppression fraction `||Π⊥(v_d)||` as a first-class score component:

```
score(v_d) = cos(Π⊥(query), Π⊥(v_d)) · ||Π⊥(v_d)||^λ
```

This makes the suppression interpretable and tunable: `λ=0` ignores it (Option A's cosine only); `λ=1` linearly down-weights images with small residual norm; `λ>1` applies a sharper penalty to images with little surviving content. The `λ` sweep is a clean ablation axis.

### Why It Is Original

- Track S keeps negative concepts in the scoring space and down-weights them. NCS **removes them from the space** — a strictly stronger geometric operation.
- The projection is applied at the **database level** before any similarity is computed, not at the query level as in PoS-Subspaces' orthogonal rejection.
- The suppression-fraction signal `||Π⊥(v_d)||` is a novel interpretable auxiliary score with no counterpart in any cited method.

### Implementation Notes

- The complement projection can be precomputed once per query (same amortization as CLAY's `_project_db`): build `Q_all` from the negative prompt stacks, project the entire `[N, d]` DB in one matrix op `Π⊥(DB) = DB − DB @ Q_all @ Q_all.T`.
- When `T_neg` is empty, `Π⊥` is the identity — the method degenerates safely to the positive-only baseline.
- Ablation axes: Option A vs C (`λ` sweep), `k_neg ∈ {5, 10, 20}`, NCS alone vs NCS + positive subspace scoring.
- Risk: if `k_neg` is too large, the complement removes content signal along with the negative concept. Keep `k_neg` small and verify with the suppression fraction histogram.

---

## Approach 3 — Source-Anchored Geodesic Interpolation ("SAGI")

### Core Idea

All existing methods modify the reference by **adding or subtracting text directions** — a Euclidean or geodesic displacement from `v_ref`. But the target image we want is not "v_ref displaced by +glasses text"; it is "what v_ref would look like if it were wearing glasses." These are geometrically different.

SAGI finds that target by **geodesically interpolating between the reference and the centroid of images that pseudo-satisfy the positive conditions**, as identified by CLIP similarity to the condition prompt (no label leakage — CLIP itself provides the pseudo-labels). The reference steers the interpolation destination; the text selects the neighborhood to interpolate toward.

### Formal Description

**Step 1 — Pseudo-label the test corpus.** For each positive attribute `a ∈ T_pos`, rank DB images by `cos(v_d, t_a)` and take the top-`n` as the pseudo-positive set `P_a`:

```
P_a = top_n { v_d ∈ DB : cos(v_d, t_a) }
```

**Step 2 — Find the manifold centroid of the pseudo-positive set:**

```
mu_a = intrinsic_mean( P_a )     # Fréchet mean on S^{d-1}
```

**Step 3 — Compose the multi-attribute positive target.** If multiple positive attributes, chain the interpolations:

```
q_0 = v_ref
q_i = Exp_{q_{i-1}}( t · Log_{q_{i-1}}(mu_{a_i}) )    for each a_i ∈ T_pos, t ∈ (0,1]
```

**Step 4 — Apply negation** via orthogonal rejection in the tangent space at the final `q` (same as Tier-1 GDE):

```
for b ∈ T_neg: q_tan = q_tan − (q_tan · v̂_b) v̂_b
query = Exp_{mu}( q_tan )
```

**Step 5 — Rank the DB** by `cos(query, v_d)`.

The interpolation parameter `t` controls how far the query moves from the reference toward the pseudo-positive centroid. `t=0` recovers the reference itself; `t=1` reaches the pseudo-positive centroid directly.

### Why It Is Original

- Every prior method moves the query by adding/subtracting a text direction. SAGI moves it toward an **image-derived manifold centroid**, keeping the query entirely inside the image distribution cone — no modality gap.
- The pseudo-labeling step uses CLIP itself as the oracle (no external labels, no training). This is a self-referential use of the model not explored in any cited paper.
- The geodesic interpolation between the reference and the pseudo-positive centroid is geometrically principled: the query follows the shortest path on `S^{d-1}` between "being v_ref" and "being near the positive cluster," rather than overshooting by adding a text vector of unknown magnitude.
- Strongest originality claim of all four proposals.

### Implementation Notes

- `intrinsic_mean` is already implemented in [manifold.py](../src/manifold.py) — reuse directly.
- `n` (pseudo-positive set size) and `t` (interpolation strength) are the two key hyperparameters. Sweep `n ∈ {50, 200, 500}`, `t ∈ {0.3, 0.5, 0.7, 1.0}`.
- Multiple positive attributes can alternatively be handled by taking the joint top-`n` set (images that score high on *all* positive prompts simultaneously) rather than chaining. Both variants are worth comparing.
- The method requires one `intrinsic_mean` call per positive attribute per query — the only meaningful overhead over the baselines.

---

## Approach 4 — Conditional Kernel Score ("CKS")

### Core Idea

All existing scorers use a sum/difference of cosine similarities. CKS replaces this with a **multiplicative kernel** that requires an image to jointly satisfy all conditions simultaneously, not just on average. A single very high positive score cannot compensate for a failed negative constraint.

The negative constraint appears in the **denominator** — images highly similar to negated attributes are down-ranked as a ratio, not subtracted as a penalty. This has a log-linear model interpretation: the log score is a linear combination of log-cosines.

### Formal Description

```
score(v_d | v_ref, T+, T-) =
    cos(v_d, v_ref)^{α}
    · Π_{a ∈ T_pos} cos(v_d, t̂_a)^{β}
    / Π_{b ∈ T_neg} cos(v_d, t̂_b)^{γ}
```

where `t̂_a` is the attribute direction (de-biased and ensembled, from Tier-0 Enhanced FIX 1+3). Exponents `α`, `β`, `γ` control the relative importance of each term.

In log-space (numerically stable, equivalent ranking):

```
log_score = α · log cos(v_d, v_ref)
           + β · Σ_{a∈T_pos} log cos(v_d, t̂_a)
           − γ · Σ_{b∈T_neg} log cos(v_d, t̂_b)
```

Cosine values are clamped to `[ε, 1]` before taking logs to avoid `-inf`.

### Why It Is Original

- The multiplicative (product) formulation enforces **joint satisfaction** of all conditions — a property that additive scores do not have. An image that scores 0.99 on the reference but 0.01 on the positive attribute still gets a near-zero joint score.
- The ratio formulation for negation is not a penalty subtracted from a running score; it is a division that **amplifies** the down-ranking of strongly negated images non-linearly.
- The log-linear interpretation connects the scorer to probabilistic models (Naive Bayes, log-linear CRFs) — a theoretical grounding no existing method in this project has.
- None of GDE, CLAY, PoS-Subspaces, or SpaceVLM use multiplicative scoring.

### Implementation Notes

- Cosines between `v_d` and the `t̂_a` vectors can be precomputed for all DB images in one matrix multiply: `S = image_features @ attr_directions.T` → `[N, 40]`. The relevant columns are sliced per query. Near-zero overhead.
- The three exponents `(α, β, γ)` are the ablation axes. Start with `α=1, β=1, γ=1` and sweep.
- Edge case: if a cosine is negative (image is anti-similar to the condition), the log is undefined. Two options: (a) clamp to `ε`, which treats anti-similar images harshly; (b) shift cosines to `[0,1]` via `(1 + cos)/2` before logging, which is a softer treatment. Both are worth testing.

---

## Summary Table

| Approach | Key Novelty | Complexity | Strongest Contribution |
|----------|-------------|------------|------------------------|
| **Adaptive CLAY** | Source-conditional subspace reweighting | Low (1 dot product per source) | Strict generalization of CLAY |
| **NCS** | Hard geometric removal of negative concepts from DB | Medium (DB reprojection per query) | Strongest geometric negation |
| **SAGI** | Geodesic interpolation toward image-derived centroid | Medium (intrinsic_mean per attr) | Most conceptually distinct |
| **CKS** | Multiplicative joint-satisfaction scoring | Very low (precomputed matrix) | Log-linear probabilistic grounding |
