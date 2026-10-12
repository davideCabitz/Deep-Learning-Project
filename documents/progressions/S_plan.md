# Track S — Asymmetric Conditional Subspaces: Complete Theoretical and Implementation Reference

> **Purpose.** This document is the single authoritative source for Track S
> (`src/tier2a_subspace.py`). It covers: the failure modes Track S is designed to fix,
> the full geometric theory behind the approach, every mathematical operation with
> justification, the complete step-by-step pipeline, how each of the five backing papers
> contributes, all hyperparameters and ablations, and implementation notes tied to the
> existing codebase. A reader who finishes this document should be able to implement
> Track S from scratch without consulting any other source.

---

## 0. The Problem Track S Must Solve

### 0.1 Observed failures in the project

Every method up to and including Tier-1 (CLAY) exhibits the same concrete failure pattern
when evaluated on the 14 benchmark queries:

| Query type | Example | Tier-0 R@5 | Tier-1 R@5 |
|---|---|---|---|
| Single positive attribute | `+Mustache` | 0.217 | 0.093 |
| Single negative attribute | `-Male` | 0.000 | 0.000 |
| Composed negative | `-Male, -Mustache` | 0.000 | 0.000 |
| Mixed composed | `+Eyeglasses, -Smiling` | low | low |

**Negation queries score 0.000 under every current method.** This is not a measurement
artifact or a bug — the Tier-0 sanity checks confirm the harness is correct (self-retrieval
works, unit-normed DB, source exclusion verified). This is a genuine property of how CLIP
represents negation: it essentially cannot. Track S is designed to fix exactly this.

### 0.2 Why Tier-0 fails

Tier-0 computes the query vector as:

```
q_naive = normalize( v_ref + α · ( Σ_{a∈T+} t(a) − Σ_{a∈T-} t(a) ) )
```

where `v_ref` is the L2-normalized CLIP image embedding of the reference face, and `t(a)`
is the L2-normalized CLIP text embedding of attribute `a`.

**Failure mode 1 — Modality gap.** CLIP's contrastive pretraining produces a well-known
geometric artifact: all image embeddings cluster tightly in one narrow cone of the unit
hypersphere, and all text embeddings cluster in a *different* narrow cone, with a wide empty
band between them (Liang et al., 2022, "Mind the Gap"). This means every text vector
`t(a)` carries a large shared component pointing toward the centre of the text cone — a
direction that encodes "I am a CLIP text embedding" rather than "I am the *Smiling*
attribute." When `t(a)` is added to an image vector, most of that addition is wasted on the
modality-gap offset rather than steering the query toward the target attribute. The
Enhanced-Tier-0 modality-gap centering fix (FIX 1 in `tier0_enhanced.py`) confirms this
empirically: removing the shared text mean `μ_txt` lifts R@5 from 0.070 to 0.114 (+63%)
— a pure mean-subtraction with no other change.

**Failure mode 2 — Subtraction ≠ negation.** When a query requests `-Male`, Tier-0
subtracts `t(Male)` from the query vector. On the unit sphere, subtracting a vector from
a query pushes the query *away* from the "Male" direction — but it pushes it toward a
specific point, specifically the "anti-Male" direction. This is not what the query means.
"Not Male" means *any* acceptable alternative: Female, or ambiguous gender, or any image
that simply lacks the Male attribute. The target set is a large open region on the sphere
("far from Male"), not a single point in a fixed anti-Male direction. Subtracting `t(Male)`
generates a query that lands somewhere specific and biased, and images that merely lack the
Male attribute but are not in that specific direction get penalised. SpaceVLM (Ranjbar et
al., 2025) formalises this: no unit vector can separate attribute-present images from
attribute-absent images with a positive margin under dot-product scoring. Negation is not
representable as a point on the sphere. Track S's approach — treating negation as the
**orthogonal complement of a subspace** — is the correct geometric extension.

### 0.3 Why Tier-1 (faithful CLAY) fails

CLAY's approach is conceptually stronger than Tier-0: instead of adding a text vector
arithmetically, it builds a *subspace* from the condition texts (via SVD on the tangent
space) and measures similarity *inside* that subspace. This removes the modality gap for
the structural similarity measurement, which is why CLAY is the method to beat.

However, faithful CLAY has two fatal flaws for this benchmark:

**Flaw 1 — No +/− distinction (naïve stacking).** CLAY merges all condition prompts
(`T_pos + T_neg`) into a single matrix and runs one SVD over the combined stack. Positive
and negative conditions are treated identically — both are just "condition texts." The
resulting subspace captures directions of variation relevant to *all* mentioned attributes,
but it has no mechanism to distinguish "retrieve images with this" from "retrieve images
without this." CLAY's operation is a *focus/preserve* reframing (attend more to
condition-relevant dimensions), not a *modification* operation. On modification queries like
`+Mustache` (find a person with a Mustache who otherwise resembles the reference) it has
to do real work; on negation queries like `-Male` it is blind.

**Flaw 2 — No asymmetric negation.** The same SVD subspace is used to score all queries.
Even if you could tell CLAY which attributes are negative, there is no operator in vanilla
CLAY that says "remove the energy in this direction." The projection matrix
`P_c = V_k V_k^T` enhances similarity to *any* image with energy in the subspace, whether
that energy comes from being Male or from being Female.

**Observed result:** Tier-1 (k=50, with rotation) scores MEAN R@1=0.0067, R@5=0.0227,
R@10=0.0351 — *below* Tier-0 (R@5=0.070). This is the expected, gradeable outcome:
a method that ignores negation on a benchmark with negation queries should do poorly.
Track S converts CLAY's strengths (manifold-aware subspace, modality-gap correction)
into a method that handles both positive focus and genuine negation.

---

## 1. Geometric Setting: The CLIP Hypersphere

### 1.1 Everything lives on S^{d-1}

CLIP maps both images and texts to vectors in R^d (d=512 for ViT-B/32) that are L2-
normalized, placing them on the unit hypersphere S^{d-1}. Similarity between two items
is measured by cosine similarity, which equals their dot product when both are unit
vectors:

```
sim(u, v) = u · v / (‖u‖ · ‖v‖) = u · v     (when ‖u‖ = ‖v‖ = 1)
```

Geometrically, this is the cosine of the angle between the two vectors. Larger values
(closer to 1) mean a smaller angle (more similar); smaller values (closer to -1) mean
a larger angle (more different). The retrieval task amounts to finding all DB images
`v_d` whose angle with the query vector `q` is smallest.

### 1.2 Modality gap: two cones, not one

Despite sharing the same 512-dimensional ambient space, CLIP image vectors and CLIP text
vectors do **not** overlap. Empirically (and confirmed in this project's Tier-0 analysis),
image vectors all lie inside a narrow cone around the image mean `μ_img`, and text vectors
all lie inside a different narrow cone around the text mean `μ_txt`. The angular separation
between `μ_img` and `μ_txt` is substantial. This is the *modality gap* or *conic effect*.

Consequence for Track S: when we build subspaces from text prompts and project image
vectors into them, there is a systematic misalignment. CLAY's rotation `H` is designed to
bridge this gap: it is a minimal rotation (a 2D rotation in the plane spanned by `μ_img`
and `μ_txt`, identity on the orthogonal complement) that maps the visual mean to the text
mean, aligning the two cones without distorting intra-DB relationships. Track S inherits
this rotation as an optional ablation toggle.

### 1.3 Tangent space and the log/exp maps

The unit hypersphere S^{d-1} is a Riemannian manifold. Standard Euclidean operations
(addition, subtraction, PCA) are not defined on it without distortion. The standard fix is
to work in the *tangent space at a base point*, which is a flat Euclidean space that is a
first-order approximation of the sphere at that point.

**Logarithmic map** `Log_μ(x)` (implemented as `_log_map(mu, X)` in `tier1.py`):
Given a base point `μ ∈ S^{d-1}` and a sphere point `x ∈ S^{d-1}`, the log map sends `x`
to the tangent vector at `μ` that points from `μ` toward `x` along the geodesic
(shortest arc), with length equal to the geodesic distance `θ = arccos(μ · x)`:

```
Log_μ(x) = θ · (x − (μ·x)·μ) / sin(θ),    θ = arccos(μ · x)
```

The numerator `x − (μ·x)·μ` is the component of `x` orthogonal to `μ` (i.e., `x`
with the `μ`-component removed), and has Euclidean norm `sin(θ)`. Dividing by `sin(θ)`
and multiplying by `θ` rescales so the tangent vector has length `θ`. When `θ ≈ 0`
(the point is near the base), the tangent vector is near-zero (the eps guard in `_log_map`
handles the 0/0 numerically).

**Why use the log map?** The sphere is curved, so Euclidean SVD on raw sphere-coordinates
is geometrically incorrect — it is biased toward the equatorial belt and penalises points
near the poles relative to their true geodesic distances. Mapping everything to the tangent
space at the mean `μ_c` makes the coordinate system Euclidean *locally*, so SVD correctly
finds the directions of maximum spread in terms of geodesic distances from `μ_c`. CLAY
demonstrates this matters in practice (their paper's ablation); PoS-Subspaces (Oldfield et
al., 2023) show the manifold-aware version (their "PGA on the hypersphere") outperforms
Euclidean PCA on 14 of 15 zero-shot classification benchmarks.

**Exponential map** `Exp_μ(v)`: The inverse of the log map — given a tangent vector `v`
at `μ`, returns the sphere point reached by travelling from `μ` a geodesic distance `‖v‖`
in the direction `v/‖v‖`. Not used directly in Track S's retrieval pipeline (we work
entirely in the tangent-space projection, then take cosines), but needed for the round-trip
test in `test/test_tier2a_subspace.py`.

---

## 2. The Core Insight: Why Negation Requires a Subspace, Not a Vector

### 2.1 SpaceVLM's impossibility theorem

SpaceVLM (Ranjbar et al., 2025, arXiv:2511.13231) formalizes what the Tier-0 failure
demonstrates empirically:

**Theorem (informal).** There is no unit vector `n ∈ S^{d-1}` such that for all images
`e_I` of an attribute `A` and all images `e_J` of "not A":

```
e_I · n > t    (attribute present)
e_J · n < t    (attribute absent)
```

with a positive margin, for any threshold `t`. Negation is geometrically unrepresentable
as a single point on the sphere under dot-product scoring.

**Why this matters for Track S.** The standard Tier-0 approach of subtracting `t(Male)`
produces a single vector — a point on the sphere — and computes dot products with it.
SpaceVLM's theorem says this single point *cannot* correctly separate Male-present images
from Male-absent images. Any vector close enough to "Female" in the image space will score
high, but images that simply lack the Male attribute (androgynous faces, children, faces
with unusual gender presentation) will score low or even negatively, because they are not
in the "Female" direction.

The geometric insight is that "images without attribute A" is not a point but a *region*
of the sphere — specifically, the complement of the region containing attribute-A images.
A subspace (and its orthogonal complement) can represent a region; a single vector cannot.

### 2.2 SpaceVLM's spherical-cap intersection

SpaceVLM's training-free solution for a query "A but not N" is:

1. Encode the affirmative concept: `e_a = text_encoder("a photo of A")`
2. Encode the negated concept: `e_n = text_encoder("a photo of N")`
3. Define a spherical cap: `N(x) = { z : x·z ≥ t }` (images within angular radius
   `arccos(t)` of `x`; the `t=0.9` to `0.95` range works robustly)
4. Target region: `N(e_a) ∩ N^c(e_n)` — close to A, far from N
5. Central direction of that intersection region (along the great-circle arc between `e_n`
   and `e_a`):
   ```
   d̂ = [sin(α + θ/2) / sin(θ)] · e_a  −  [sin(α − θ/2) / sin(θ)] · e_n
   ```
   where `α = arccos(t)`, `θ = arccos(e_a · e_n)` is the angle between the two concept
   vectors.
6. Score: `s(e_I) = e_I · normalize(d̂)`

This is mathematically clean: instead of a naive subtraction `e_a − e_n` (which just
shifts toward a specific anti-N direction), SpaceVLM finds the central direction of the
*feasible set* "near A and far from N." The formula is a weighted combination that
accounts for the relative positions of the two concepts on the sphere.

**How Track S generalises this.** SpaceVLM works with single text vectors. Track S
replaces each single concept vector with a *subspace* — a k-dimensional span of many
prompts about that concept. Instead of `N(e_a)` (a spherical cap around a point), Track
S works with the span of the positive-attribute subspace `S^+`. Instead of `N^c(e_n)` (the
region far from a single negation point), Track S works with `(S^-)^⊥` (the orthogonal
complement of the negative-attribute subspace). This is a multi-dimensional generalisation
that is more expressive: a k-dimensional subspace captures the full spread of how the
positive (or negative) attribute can appear across many different face images and phrasings,
rather than a single noisy text embedding.

Track S's composite score:

```
score(v_ref, v_d | T+, T-) = cos_{S+}(v_ref, v_d) − λ · ‖proj_{S-}(v_d)‖
```

is the direct subspace generalisation of SpaceVLM's `d̂` formula: the first term rewards
similarity inside the positive subspace (the "near A" part), and the second term penalises
energy inside the negative subspace (the "far from N" part).

### 2.3 The linear negation direction exists (geometric confirmation)

The "When Negation Is a Geometry Problem" paper (Sammani et al., 2026, arXiv:2603.20554)
provides an independent confirmation that CLIP's text encoder *already* contains a
structured negation direction, even though it fails to use it at inference:

- A linear binary classifier trained on 4K (affirmative, negated) caption pairs in CLIP's
  text encoder hidden states achieves ≥99% test accuracy at layer 4.
- This means there is a clean linear subspace direction in CLIP's internal representation
  that separates "concept X" from "not X."
- The direction peaks in intermediate layers, not early or late ones.

**Implication for Track S.** If a linear negation direction exists in CLIP's hidden states,
then the SVD of the positive-prompt covariance matrix `T_+^T T_+` (in tangent space) will
produce a direction `V_k^+` that correlates with the affirmative concept, and the SVD of
`T_-^T T_-` will produce a direction `V_k^-` that correlates with the negated concept —
because those are the directions of maximum variance in each prompt set, and the prompts
about an attribute cluster along its own linear direction. This empirically validates that
separate SVDs on `T_+` and `T_-` will produce geometrically meaningful, separable
subspaces `S^+` and `S^-`. Track S's approach is not just a heuristic; it exploits a
structure that actually exists in the model.

---

## 3. The Complete Track S Pipeline

Track S builds `src/tier2a_subspace.py` implementing the same
`make_get_ranking(query_str, …) → get_ranking(src_idx)` interface as every other tier
(CONTRACT §7). The file writes `output/tier2a_subspace_*.csv` in the shared CSV schema.

The pipeline has **five stages** executed per query, plus one DB projection that is
reused across all sources of the same query.

### Stage 1 — Parse the query and build polarity-split prompt stacks

```
parse_query(query_str) → (T_pos: list[str], T_neg: list[str])
```

This is already implemented in `eval.py` and used by Tier-1. For a query like
`"+Black_Hair, -Wavy_Hair"` it returns `T_pos = ["Black_Hair"]`,
`T_neg = ["Wavy_Hair"]`. For `"-Male, -Mustache"` it returns
`T_pos = []`, `T_neg = ["Male", "Mustache"]`.

For each attribute name in `T_pos`, retrieve its full (unpadded) prompt stack from
`clip_attr_prompt_bank.pt` (the `[40, n, 512]` artifact built by `clip_prompts.py`):

```python
j = ATTRIBUTE_NAMES.index(attr_name)
n_j = len(build_prompts_for_attribute(attr_name))   # true count, no padding
T_pos_prompts = prompt_bank[j, :n_j]               # [n_j, 512], unit rows
```

Do the same for each attribute in `T_neg`. The result is two matrices:

```
T+ : [m+, d]   — all positive-polarity prompt embeddings, unit rows
T- : [m-, d]   — all negative-polarity prompt embeddings, unit rows
```

**Why unpadded?** The bank pads shorter stacks with duplicate rows (repeated rows have
the same direction, adding zero new SVD span). Including them in `T+` or `T-` would bias
`μ_c^+` toward that attribute's first prompt and skew the singular values — a duplicate
row folds its variance into the first singular value rather than distributing it correctly.
(This is the same padding-strip logic `_stack_condition_prompts` applies in `tier1.py`.)

**Why separate stacks?** Tier-1 merges `T+` and `T-` into one matrix `T_c` and runs a
single SVD. The resulting subspace spans directions relevant to *both* positive and negative
attributes but has no polarity information. Separate stacks allow two independent SVDs,
each capturing the specific geometric spread of prompts for one polarity. This is the key
architectural departure from CLAY — the "asymmetric" in Track S's name.

**Edge case: T+ or T- empty.** For queries with only positive attributes (e.g., `+Smiling`),
`T-` is empty and there is no negative subspace to build. The score reduces to pure CLAY-
style positive-subspace similarity. For queries with only negative attributes (e.g.,
`-Male, -Mustache`), `T+` is empty and the score is a pure negation penalty: rank by
`−λ · ‖proj_{S-}(v_d)‖` (images with least energy in the negative subspace rank first).
Handle both cases explicitly in the implementation to avoid SVD on a zero-height matrix.

### Stage 2 — Build the positive subspace S^+

**Tangent point `μ_c^+`:**

```
μ_c^+ = normalize( mean(T+, dim=0) )    # [d], unit vector
```

This is the Fréchet mean approximation on the sphere — the normalized Euclidean mean of
the positive-prompt rows. For unit vectors that are all roughly pointing in the same
direction (which they are, since they all describe the same attribute from different
phrasings), this is an accurate approximation of the true intrinsic mean.

**Log-map to tangent space:**

```
L+ = Log_{μ_c^+}(T+)    # [m+, d], tangent vectors at μ_c^+
```

Each row of `L+` is the tangent vector from `μ_c^+` to the corresponding row of `T+`,
with Euclidean length equal to the geodesic distance. The matrix `L+` lives in the flat
tangent space at `μ_c^+` — it is now safe to do Euclidean linear algebra.

**SVD to find directions of maximum spread:**

```
L+ = U+ S+ Vh+    (full_matrices=False SVD)
V_k+ = Vh+[:k+].T    # [d, k+], the top-k+ right singular vectors
```

`torch.linalg.svd(L+, full_matrices=False)` returns `U+` (`[m+, r]`), `S+` (`[r]`),
`Vh+` (`[r, d]`) where `r = min(m+, d)`. The *rows* of `Vh+` are the right singular
vectors, which are the orthonormal directions in the tangent space that explain the most
variance in the positive-prompt distribution. The top-k+ columns of `Vh+.T` form the
positive subspace basis matrix `V_k+` of shape `[d, k+]`.

**Why SVD rather than PCA?** Formally SVD on the zero-mean-adjusted matrix gives PCA.
Here we have *not* subtracted the row mean before SVD — we log-mapped to the tangent
space at `μ_c^+` (which removes the mean *in intrinsic/geodesic terms*) and then ran
SVD directly. This is the manifold-aware analogue of PCA, and it is the same construction
PoS-Subspaces uses for their PGA (Principal Geodesic Analysis) on the sphere. The
tangent-space log-map plays the role of mean-centering. After log-mapping, tangent vectors
at `μ_c^+` are by definition orthogonal to `μ_c^+`, so the first singular direction will
not be dominated by `μ_c^+` itself — it genuinely captures spread among the prompts.

**Projection matrix:**

```
P+ = V_k+ @ V_k+.T    # [d, d], the orthogonal projector onto S+
```

`P+` projects any vector into the `k+`-dimensional subspace spanned by the top-k+ right
singular vectors of the log-mapped positive prompts. It is symmetric and idempotent:
`P+ = P+.T`, `P+ @ P+ = P+`. The subspace `S+ = span(V_k+)` is the set of all vectors
in the tangent space that are linear combinations of the basis directions.

### Stage 3 — Build the negative subspace S^-

Identical construction with `T-`:

```
μ_c^- = normalize( mean(T-, dim=0) )
L-    = Log_{μ_c^-}(T-)
L-    = U- S- Vh-       (SVD)
V_k-  = Vh-[:k-].T      # [d, k-]
P-    = V_k- @ V_k-.T   # [d, d]
```

`P-` projects any vector into the k^- -dimensional subspace capturing the negative-
attribute direction spread.

**Key insight.** `S+` and `S-` are built from different sets of prompts (different
attributes, opposite polarity) and have different tangent points (`μ_c^+` ≠ `μ_c^-`
in general). They are geometrically independent subspaces. The fact that they may share
some dimensions (e.g., if both positive and negative attributes involve gender-related
semantics) is handled naturally by the scoring formula: energy in `S-` is penalised
regardless of whether it overlaps with `S+`.

### Stage 4 — Align the visual mean to the text mean (Rotation H)

The modality gap means that raw image vectors `v_d` and text-derived subspaces like
`S+` live in different cones of the sphere. Before projecting `v_d` into `S+`, we need
to close this gap. CLAY's rotation `H` does this minimally:

```
μ_img  = normalize( mean(image_features, dim=0) )   # visual mean
μ_txt+ = μ_c+                                        # positive text mean
H      = _align_rotation(μ_img, μ_txt+)             # [d, d] rotation matrix
```

`H` is the minimal rotation in the 2-dimensional plane spanned by `μ_img` and `μ_c^+`
that maps `μ_img` to `μ_c^+`, leaving all directions orthogonal to both untouched.
The formula (`_align_rotation` in `tier1.py`) decomposes the rotation in the plane
`(μ_img, u2)` where `u2` is the unit vector in the direction of `μ_c^+ − (μ_img · μ_c^+)μ_img`:

```
c = μ_img · μ_c^+,   u2 = normalize(μ_c^+ − c·μ_img)
P_plane = outer(μ_img, μ_img) + outer(u2, u2)       # projector onto the 2D plane
R       = c·outer(μ_img, μ_img) + s·outer(u2, μ_img) − s·outer(μ_img, u2) + c·outer(u2, u2)
H       = I − P_plane + R                           # identity off-plane, 2D rotation on-plane
```

where `s = sqrt(1 − c²) = sin(arccos(c))`.

After rotation, each DB image `V = image_features @ H.T` (shape `[N, d]`) has its mean
aligned to `μ_c^+`, making the subsequent log-map at `μ_c^+` well-centred.

**Ablation note.** CLAY's own experiments and Tier-1's ablation in this project both show
rotation `H` helps (Tier-1: R@10 0.0313 without rotation vs 0.0351 with rotation). Track
S keeps it as a toggle (`use_rotation=True/False`) for the ablation table.

**Which mean to align to?** For Track S we have two tangent points, `μ_c^+` and `μ_c^-`.
Practically, align to `μ_c^+` (or the mean of `μ_c^+` and `μ_c^-` if both are non-empty)
— since the positive subspace is used for the primary similarity score and the negative
subspace is used for the penalty, primary alignment to the positive side is sensible.
For queries with empty `T+`, align to `μ_c^-`.

### Stage 5 — Project the frozen DB into S^+

After rotation, project every DB image vector into the positive subspace:

```
V_rot     = image_features @ H.T           # [N, d], rotated DB
coords_+  = _log_map(μ_c^+, V_rot) @ V_k+ # [N, k+], tangent coords in S+
D_+       = normalize_rows(coords_+)       # [N, k+], unit rows
```

`D_+[i]` is the k^+ -dimensional representation of DB image `i` in the positive
subspace, normalized so cosine in the subspace is a simple dot product. This is
*precomputed once per query* and reused for every source index — the key efficiency gain
(CLAY's design choice, inherited by Track S).

**For the negative penalty**, we also compute the raw (unnormalized) projection energy
of every DB image into the negative subspace:

```
coords_-  = _log_map(μ_c^-, V_rot) @ V_k-  # [N, k-], tangent coords in S-
neg_norms = coords_-.norm(dim=1)             # [N], ‖proj_{S-}(v_d)‖ for each DB image
```

`neg_norms[i]` measures how much energy DB image `i` has inside the negative subspace.
High energy there means the image strongly represents the negative attribute — it should
be penalised.

### Stage 6 — Score and rank

For each source index `src_idx`, the composite score for every DB image `v_d` is:

```
score(src_idx, d) = D_+[d] · D_+[src_idx]  −  λ · neg_norms[d]
```

- **First term** `D_+[d] · D_+[src_idx]` is the cosine similarity in the positive
  subspace between the DB image `d` and the reference image `src_idx`. This is the
  "preserve similarity while attending to positive attributes" step — exactly what CLAY
  does, but now restricted to the *positive-only* subspace so negative-attribute directions
  do not contaminate it.

- **Second term** `λ · neg_norms[d]` is the negation penalty: the Frobenius norm of
  image `d`'s projection into the negative subspace, scaled by the hyperparameter `λ`.
  Images with large energy in the negative subspace (i.e., images that look like they
  have the negative attribute) receive a large penalty and rank lower.

- **Source exclusion** (CONTRACT §5): `scores[src_idx] = -inf` before argsort.

```python
scores = D_plus @ D_plus[src_idx]  -  lam * neg_norms
scores[src_idx] = float('-inf')
ranking = torch.argsort(scores, descending=True).tolist()
```

**Why this works for negation.** A query like `-Male` produces `T- = ["Male"]`,
which builds `S-` capturing the geometric spread of all "Male" prompts. Images of
Male faces have large energy inside `S-` (their embedding is close to the directions
that explain the variance in "Male"-related prompts). The penalty `λ · ‖proj_{S-}(v_d)‖`
directly suppresses these images in the ranking. Female faces, children, and ambiguous
faces all have *low* energy in `S-` (they are geometrically far from the "Male" prompt
cluster), so they receive little or no penalty. The approach correctly identifies "not
Male" as "low energy in the Male subspace" — a region of the sphere, not a specific
point — which is exactly what SpaceVLM's geometric argument calls for.

**Why this is better than SpaceVLM's single-vector approach.** SpaceVLM encodes the
negative concept as a single point `e_n` and finds the centre direction of the spherical
cap complement. Track S uses a k^- -dimensional subspace `S-` for the negative concept.
A single text embedding `e_n = text_encoder("Male")` is a noisy, phrasing-specific estimate
of the "Male" direction. The SVD of 60 differently-worded "Male"-related prompts in tangent
space finds the *directions of maximum spread* of the "Male" concept distribution — a more
robust, lower-variance estimate that also captures multiple aspects of what "Male" means
visually (facial structure, hair length, presence of beard, etc.). The complement `(S-)^⊥`
is correspondingly richer than the complement of a single point.

### (Variant) Stage 6b — Per-condition subspaces and intersection

The headline Track S variant builds one subspace per condition (attribute) rather than
stacking all positive attributes into one combined `T+` matrix.

For a query like `"+Black_Hair, -Wavy_Hair, +Smiling"`, instead of building one `S+`
from all Black_Hair + Smiling prompts together, build:

```
S+_{Black_Hair}  from prompts of Black_Hair only
S+_{Smiling}     from prompts of Smiling only
S-_{Wavy_Hair}   from prompts of Wavy_Hair only
```

Then combine by intersection for positive and union of penalties for negative:

- **Positive score**: average (or min) of cosine similarities in each positive subspace:
  ```
  pos_score(d) = (1/|T+|) Σ_{a ∈ T+} [ D_a+ @ D_a+[src_idx] ]
  ```
  where `D_a+` is the projected DB for attribute `a`'s positive subspace.

- **Negative penalty**: sum of per-attribute projection norms:
  ```
  neg_penalty(d) = λ · Σ_{b ∈ T-} neg_norms_b[d]
  ```

- **Composite**: `pos_score(d) − neg_penalty(d)`

**Why this is better than stacking.** When Black_Hair and Smiling prompts are merged into
one `T+` matrix, the SVD finds directions that explain variance across both attributes.
If Black_Hair dominates the prompt stack numerically (more prompts or higher norm), the
SVD will be biased toward hair-related directions and underweight Smiling-related
directions. Per-condition subspaces give equal geometric weight to every condition
regardless of how many prompts describe it. This mirrors PoS-Subspaces' philosophy of
building one subspace per grammatical role rather than one merged subspace.

The stacked variant (one `T+` for all positive attributes) is kept as an **ablation
baseline** to quantify how much the per-condition split helps.

---

## 4. Mathematical Summary

### 4.1 Notation

| Symbol | Meaning |
|---|---|
| `d = 512` | CLIP embedding dimension (ViT-B/32) |
| `N = 19962` | Number of DB images (CelebA test split) |
| `S^{d-1}` | Unit hypersphere in R^d |
| `v_ref` | Reference image embedding, unit vector |
| `v_d` | DB image embedding `d`, unit vector |
| `T+, T-` | Positive/negative attribute names |
| `T+` (matrix) | Prompt embeddings for positive attributes, `[m+, d]`, unit rows |
| `T-` (matrix) | Prompt embeddings for negative attributes, `[m-, d]`, unit rows |
| `μ_c^+, μ_c^-` | Tangent points (normalized means of T+, T-), unit vectors |
| `L+, L-` | Log-mapped prompt matrices in tangent space, `[m±, d]` |
| `V_k^+, V_k^-` | Top-k± right singular vectors (basis of S+, S-), `[d, k±]` |
| `P+, P-` | Projection matrices `V_k V_k^T`, `[d, d]` |
| `H` | Modality-gap rotation (visual mean → text mean), `[d, d]` |
| `D_+` | Projected DB in S+, `[N, k+]`, unit rows |
| `neg_norms` | ‖proj_{S-}(v_d)‖ for each DB image, `[N]` |
| `λ` | Negation penalty weight (ablation hyperparameter) |
| `k+, k-` | Positive/negative subspace dimensionalities (ablation hyperparameters) |

### 4.2 Pipeline equations

**Step 1 — Polarity split:**
```
T_pos, T_neg = parse_query(query_str)
T+ = concat([ prompt_bank[j, :n_j] for j in T_pos ])    # [m+, d]
T- = concat([ prompt_bank[j, :n_j] for j in T_neg ])    # [m-, d]
```

**Step 2 — Positive subspace:**
```
μ_c^+ = normalize(mean(T+, dim=0))                       # [d]
L+    = Log_{μ_c^+}(T+)                                  # [m+, d]
_,_,Vh+ = svd(L+, full_matrices=False)
V_k+  = Vh+[:k+].T                                      # [d, k+]
P+    = V_k+ @ V_k+.T                                   # [d, d]
```

**Step 3 — Negative subspace:**
```
μ_c^- = normalize(mean(T-, dim=0))                       # [d]
L-    = Log_{μ_c^-}(T-)                                  # [m-, d]
_,_,Vh- = svd(L-, full_matrices=False)
V_k-  = Vh-[:k-].T                                      # [d, k-]
P-    = V_k- @ V_k-.T                                   # [d, d]
```

**Step 4 — Rotation (optional):**
```
μ_img = normalize(mean(image_features, dim=0))           # [d]
H     = align_rotation(μ_img, μ_c^+)                    # [d, d]
V_rot = image_features @ H.T                             # [N, d]
```

**Step 5 — Project DB:**
```
coords_+ = Log_{μ_c^+}(V_rot) @ V_k+                   # [N, k+]
D_+      = normalize_rows(coords_+)                      # [N, k+], unit rows
coords_- = Log_{μ_c^-}(V_rot) @ V_k-                   # [N, k-]
neg_norms = coords_-.norm(dim=1)                         # [N]
```

**Step 6 — Score:**
```
scores           = D_+ @ D_+[src_idx] - λ · neg_norms   # [N]
scores[src_idx]  = -inf
ranking          = argsort(scores, descending=True)      # [N]
```

### 4.3 Properties of the construction

- **Orthonormality.** `V_k^+ .T @ V_k^+ = I_{k+}` by construction (SVD produces
  orthonormal right singular vectors). The projection `P+ = V_k+ V_k+^T` is the unique
  orthogonal projector onto the span of V_k+: symmetric, idempotent (`P+² = P+`), and
  `‖P+ v‖ ≤ ‖v‖` for all `v` (projection is norm-non-increasing). Test in
  `test_tier2a_subspace.py`: `(P+ @ P+)` should equal `P+` to machine precision.

- **Complement orthogonality.** `I − P-` is the orthogonal projector onto `(S-)^⊥`:
  everything orthogonal to the negative subspace passes through unchanged, and everything
  in `S-` is zeroed out. `neg_norms[d] = ‖P- v_d‖ = ‖coords_-[d]‖`: smaller values
  mean the image has less "negative attribute" content.

- **Score decomposition.** The composite score can be written:
  ```
  score(d) = sim_{S+}(v_ref, v_d) − λ · sim_{S-}(v_d)
  ```
  where `sim_{S+}(a, b) = (P+ a) · (P+ b) / (‖P+ a‖ · ‖P+ b‖)` (already normalized in
  `D_+`) and `sim_{S-}(v_d) = ‖P- v_d‖` (unnormalized — we want to penalise the
  *amount* of negative-attribute energy, not just whether any exists).

- **Degenerate cases.** If `T+` is empty, skip Steps 2 and 5a; score = `−λ · neg_norms`.
  If `T-` is empty, skip Steps 3 and 5b; score = `D_+ @ D_+[src_idx]` (pure CLAY on
  positive subspace only). If both are non-empty, the full formula applies.

---

## 5. Paper-by-Paper Contribution Map

### 5.1 CLAY (Lim et al., CVPR 2026)

**Core contribution to Track S:** The manifold-aware subspace construction pipeline —
the exact sequence log-map → SVD → projection matrix → cosine-in-subspace — is Track S's
backbone, applied twice (once for `S+`, once for `S-`). Track S reuses `_log_map` and
`_align_rotation` from `tier1.py` (frozen, read-only).

**Specific items Track S inherits from CLAY:**
- Tangent-point computation `μ_c = normalize(mean(T_c))` (manifold mean approximation).
- Full SVD on the log-mapped matrix, keeping top-k right singular vectors.
- Projection matrix `P_c = V_k V_k^T`.
- Rotation `H` for modality-gap bridging (2D plane rotation, identity on complement).
- Symmetric formulation: project both `v_ref` and `v_d` into the subspace (not just `v_ref`).
- Pre-computation of the projected DB per query for efficiency.
- Ablation axis: `k` (subspace dimensionality) → maps to `k+` / `k-` in Track S.

**What Track S changes relative to CLAY:**
- Splits the naïve-stacked `T_c` into separate `T+` and `T-`, ending CLAY's polarity
  blindness.
- Adds the negation penalty term `λ · ‖proj_{S-}(v_d)‖` — absent from CLAY entirely.
- Optionally builds per-condition subspaces instead of one stacked subspace.

**CLAY's finding that the symmetric formulation (project both sides) beats query-only** is
relevant: Track S follows CLAY's lead and projects the full DB (not just the query), which
is also more efficient since the DB projection is pre-computed once per query.

### 5.2 SpaceVLM (Ranjbar et al., arXiv:2511.13231, November 2025)

**Core contribution to Track S:** The impossibility theorem (Section 3.1) establishes
*why* scalar-valued negation (Tier-0's subtraction) cannot work, providing the theoretical
justification for Track S's subspace-complement approach. The intersection of spherical
caps (`N(e_a) ∩ N^c(e_n)`) is the single-vector precursor to Track S's subspace
intersection (`S+ ∩ (S-)^⊥`).

**Specific items Track S borrows from SpaceVLM:**
- The composite score structure: affirmative term minus negation penalty term. Track S's
  `cos_{S+} − λ · ‖proj_{S-}‖` mirrors SpaceVLM's `e_I · normalize(d̂)` where `d̂`
  is a weighted difference of affirmative and negated concept vectors.
- The insight that the penalty term should scale with *energy* inside the negated region
  (SpaceVLM's `sin(α − θ/2)` coefficient; Track S's `λ`).
- The finding that the approach is backbone-agnostic (improves CLIP, SigLIP, AIMV2) →
  confidence that Track S will transfer across CLIP variants.
- The threshold `t ∈ [0.90, 0.95]` being robust → motivation for Track S's λ being in
  the mild range (0.1–0.5): too aggressive negation suppression hurts retrieval of valid
  targets that happen to share some features with the negated attribute.

**How Track S extends SpaceVLM:**
SpaceVLM operates on single text embeddings; Track S lifts the entire construction to
k-dimensional subspaces. The spherical cap `N(e_n) = { z : e_n · z ≥ t }` becomes the
subspace projection region `{ z : ‖P- z‖ ≥ ε }`. The central direction of the complement
region (SpaceVLM's `d̂`) becomes the entire subspace `(S-)^⊥` projected image space.

### 5.3 Parts-of-Speech-Grounded Subspaces (Oldfield et al., NeurIPS 2023)

**Core contribution to Track S:** The closed-form eigenvector construction for building
discriminative subspaces, and the explicit use of orthogonal-complement projection to
*remove* a visual mode. The joint objective:

```
C_i = (1−λ) X_i X_i^T − Σ_{j≠i} λ X_j X_j^T
```

maximizes variance of class `i` while suppressing variance of all other classes. Its
leading eigenvectors form the class-`i` subspace. Track S simplifies this (separate SVDs
on each polarity rather than the full contrastive objective), but the geometric philosophy
is identical: the positive subspace maximizes positive-attribute variance; the negative
subspace maximizes negative-attribute variance; their complement projections separate them.

**Specific items Track S borrows from PoS-Subspaces:**
- Manifold-aware tangent-space PGA (Principal Geodesic Analysis) — their "log → PCA →
  exp" construction is exactly Track S's log-map → SVD → projection.
- The orthogonal complement projector `Π_i^⊥(z) = I − Ŵ_i Ŵ_i^T` (projected into
  tangent space): this is `(I − P-)` in Track S, the operator that removes negative-
  attribute content.
- The λ trade-off (`λ=0.5` in their objective) balancing target-class variance and
  nuisance-class suppression → same intuition as Track S's `λ` negation weight.
- Per-condition subspace philosophy: one subspace per POS category rather than a merged
  subspace over all words → directly inspires Track S's per-condition variant.

**PoS-Subspaces' key finding for Track S:** "Projection onto the orthogonal complement
`Π_i^⊥(z)` reliably removes entire visual themes from CLIP-based generators" (applied
to removal of nouns / adjectives / verbs from generated images). If complement projection
removes visual themes from generators, it should equally penalize retrieval of images
with those themes — the exact semantics needed for negation.

### 5.4 VLMs Do Not Understand Negation (Alhamoud et al., CVPR 2025)

**Core contribution to Track S:** Diagnosis and quantification of the affirmation bias —
the empirical motivation for everything Track S does. The NegBench results confirm that
CLIP's 0.000 R@5 on `-Male, -Mustache` is not a quirk of the evaluation but a systematic,
well-documented failure across models and scales.

**Specific items Track S draws on:**
- **Affirmation bias evidence:** CLIP maps "a dog" and "no dog" to nearly identical
  embeddings; most models perform near chance (25%) on 4-way negation MCQ. This means
  `text_encoder("-Male")` and `text_encoder("+Male")` produce similar vectors — there is
  no useful negation signal in a single negated prompt embedding. Track S addresses this
  by working with *distributions* of positive/negative prompts and their subspaces, not
  single embeddings.
- **ConCLIP collapse:** ConCLIP collapses all negated embeddings into a single point —
  a degenerate failure mode that subspace methods avoid.
- **Evaluation protocol:** The MCQ Negation accuracy metric and the Hybrid MCQ template
  ("includes A but not B") are the right diagnostics for Track S. Track S's success
  condition is that `-Male, -Mustache` no longer scores 0.000.
- **Implicit definition of "correct" negation:** Retrieved images should satisfy all
  positive attributes and violate *none* of the negative ones — this is the criterion
  Track S's test queries should verify.

**What Track S must demonstrate (per this paper):** The negation queries that currently
score 0.000 must show non-zero R@5 with Track S. The paper's MCQ setup provides the
diagnostic language: affirmation accuracy (does it retrieve attribute-present images for
`+A` queries?) should be maintained, while negation accuracy (does it avoid attribute-
present images for `-A` queries?) should improve substantially.

### 5.5 When Negation Is a Geometry Problem (Sammani et al., arXiv:2603.20554, April 2026)

**Core contribution to Track S:** Independent geometric confirmation that CLIP's internal
representation contains a clean linear negation direction, validating that Track S's
SVD-based subspace construction will find meaningful, separable subspaces.

**Specific items Track S draws on:**
- **Negation direction exists at layer 4.** A linear classifier on 4K caption pairs
  achieves ≥99% accuracy → the positive/negative prompt distributions are linearly
  separable in CLIP's embedding space. This means SVD on `T+` and SVD on `T-` will
  produce subspaces that are genuinely geometrically separated (they span different
  directions), not numerically mixed.
- **Steering parameter α=0.13.** The mild value needed for effective negation steering
  (too large collapses semantics) translates to Track S's λ: the ablation should focus
  on the mild range (0.05–0.30) and avoid large values.
- **N-COCO finding.** Fine-tuned models overfit to common scene statistics; the
  training-free subspace approach avoids this — Track S should generalise better than
  any fine-tuned baseline on diverse query phrasings.
- **MLLM-as-judge evaluation.** Using Qwen3-VL to ask sequential yes/no questions
  removes false-negative contamination from standard Recall@K metrics. This is a stronger
  evaluation protocol that Track S's final write-up should consider adopting.
- **Layer 4 localization.** Negation information peaks in intermediate layers, not the
  final projection. Since CLIP's final `text_projection` aggregates across all layers,
  the final embedding contains attenuated negation signal — another reason why a single
  text embedding is a weak negation representation, and why the ensemble-then-SVD approach
  (which amplifies the shared semantic signal while averaging out noise) is necessary.

---

## 6. Hyperparameters and Ablation Plan

Track S has four primary hyperparameters. Each should be swept in the ablation table with
all others held at their default.

### 6.1 k+ and k- (subspace dimensionalities)

**What they control.** `k+` is the number of right singular vectors retained for `S+`;
`k-` for `S-`. Larger `k` captures more of the attribute's geometric spread but also
includes more noise (lower singular values correspond to directions with less explained
variance). Very small `k` (e.g., k=1) reduces the subspace to a single direction and loses
expressiveness. Very large `k` (approaching the number of prompts) includes near-zero
singular vectors and may introduce noise.

**Recommended sweep.** `k ∈ {5, 10, 20, 50}` for both `k+` and `k-` independently.
Based on Tier-1's ablation (R@10 plateaus around k=20 for the combined-prompts case), the
per-polarity sweet spot will likely be lower — around k=10–20 — since each polarity's
prompt stack is narrower (fewer prompts about one attribute compared to combined stacks
for multiple attributes).

**Default.** `k+ = k- = 20` as a starting point.

### 6.2 λ (negation penalty weight)

**What it controls.** The relative strength of the negation penalty vs. the positive-
subspace similarity. At λ=0, Track S reduces to pure CLAY on the positive subspace (no
negation). At very large λ, the score is dominated by the negation penalty and positive-
subspace similarity is ignored (everything that avoids the negative attribute ranks high,
regardless of whether it matches the positive conditions or the reference identity).

**Geometric intuition.** The `D_+ @ D_+[src_idx]` term has range `[−1, 1]` (cosine) and
is typically positive (images in the positive subspace). The `neg_norms` term has range
`[0, k-^{1/2}]` in principle but is typically small (most images have only moderate
projection in any given subspace). A λ in the range 0.05–0.30 gives roughly equal
weighting to the similarity and penalty terms.

**Literature guidance.** SpaceVLM's robust range for `t` (0.90–0.95) corresponds to mild
penalty strength. PoS-Subspaces uses `λ=0.5` in their contrastive objective. The steering
paper uses `α=0.13` for representation steering. All three point to mild values; Track S
should start with `λ ∈ {0.05, 0.1, 0.2, 0.5}`.

**Default.** `λ = 0.1` as a starting point.

### 6.3 Per-condition vs stacked subspaces

**Stacked variant (ablation baseline):** All positive-attribute prompts merged into one
`T+` matrix; one SVD; one `S+`. All negative-attribute prompts merged into one `T-`; one
SVD; one `S-`. This is faster and simpler but biased toward numerically dominant attributes.

**Per-condition variant (headline):** One SVD per attribute, separate `S+_a` and `S-_b`
for each condition `a ∈ T+`, `b ∈ T-`. Scores averaged (or min-pooled) over conditions.
This mirrors PoS-Subspaces' per-POS philosophy and gives equal geometric weight to every
condition.

Report both as separate ablation rows in the comparison table.

### 6.4 Rotation H ON vs OFF

As in Tier-1's ablation: running with and without the modality-gap rotation `H`. Based on
Tier-1 results (rotation helps: R@10 0.031 → 0.035), it is expected to help here too.
Keep it as a toggle and report both.

---

## 7. Implementation Notes

### 7.1 Reusing tier1.py helpers

`tier1.py` exports `_log_map(mu, X, eps=1e-6)` and `_align_rotation(a, b, eps=1e-6)`.
These are frozen (read-only) and implement the manifold math correctly. Track S should
import them:

```python
from tier1 import _log_map, _align_rotation
```

If CLAUDE.md's "no sideways imports between sibling peers" rule is enforced strictly, lift
both functions into a shared `src/manifold.py` module that both `tier1.py` and
`tier2a_subspace.py` import. This is a clean refactor but is not blocking — the import
is read-only (Tier-1 is frozen and Track S does not modify it). Flag the architectural
debt; implement if time allows.

### 7.2 Artifacts needed (already present)

- `artifacts/clip_image_features_test.pt` — `[N=19962, d=512]`, float32, unit rows.
  Loaded by `load_image_features()` from `clip_features.py`.
- `artifacts/clip_attr_prompt_bank.pt` — `[40, n, 512]`, float32, unit rows.
  Loaded by `load_prompt_bank()` from `clip_prompts.py`. `n` = max prompt count; shorter
  stacks are padded with duplicate rows (safe: duplicate rows add no new SVD span direction,
  as verified in `Alfonso.md`'s discussion of the prompt bank).

No new artifacts. Track S runs fully offline.

### 7.3 Edge cases to handle

1. **Empty T+ or T-.** For queries like `"+Smiling"` (no negative), skip the negative
   subspace entirely. For `"-Male, -Mustache"` (no positive), skip the positive subspace.
2. **Padding strips.** Use `build_prompts_for_attribute(name)` to get the true count `n_j`
   and slice `prompt_bank[j, :n_j]` before SVD — do not pass padded rows to the log-map.
3. **k clamping.** If `k > min(m, d)` (e.g., only 3 prompts for an attribute, k=20), clamp
   to `min(k, Vh.shape[0])` — same as `_build_subspace` in `tier1.py`.
4. **Numerical stability.** The log-map's `eps=1e-6` guard handles the `theta=0` case
   (a prompt at the exact tangent point). Normalize `D_+` rows with `eps=1e-12` to avoid
   division by zero for images with near-zero projection in S+.

### 7.4 Tests to write (test/test_tier2a_subspace.py)

Mirror `test/test_tier1.py`. Minimum required:

1. **Log/exp round-trip.** `Exp_{mu}(Log_{mu}(X)) ≈ X` to 1e-5 (uses functions from tier1).
2. **V_k+ orthonormality.** `V_k+.T @ V_k+ ≈ I_{k+}` to 1e-6.
3. **P+ idempotence.** `P+ @ P+ ≈ P+` to 1e-6.
4. **Complement orthogonality.** `(I − P-) @ P- ≈ 0` to 1e-6.
5. **neg_norms non-negative.** All elements ≥ 0.
6. **End-to-end ranking.** Source index not in top-1; ranking length == N-1.
7. **Empty T+ fallback.** Score with empty T+ does not crash; source excluded.
8. **Empty T- fallback.** Score with empty T- does not crash; source excluded.

---

## 8. Cross-Paper Synthesis Table

| Concept | Source paper(s) | How it maps to `tier2a_subspace.py` |
|---|---|---|
| Manifold-aware subspace (log → SVD → exp) | CLAY, PoS-Subspaces | `μ_c^±`, `_log_map`, SVD, `V_k^±` — same code path as `_build_subspace` in `tier1.py` |
| Asymmetric polarity subspaces | CLAY (condition-specific), PoS-Subspaces (per-POS) | Separate SVD for `T+` vs `T-` — eliminates CLAY's naïve pre-SVD merge |
| Negation as orthogonal complement | SpaceVLM (cap complement), PoS-Subspaces (`Π_i^⊥`) | `neg_norms = (log_{μ_c^-}(V_rot) @ V_k^-).norm(dim=1)` |
| Composite score (positive + negation penalty) | SpaceVLM (`d̂` formula) | `score = D_+ @ D_+[src] − λ · neg_norms` |
| Linear negation direction exists | When-Negation (steering) | Confirms `S^-` is geometrically meaningful; SVD will find real directions, not noise |
| Affirmation bias to overcome | VLMs-Do-Not-Understand-Negation | Defines the failure (`-Male, -Mustache` = 0.000 R@5) Track S must fix |
| Rotation H for modality gap | CLAY | `H = _align_rotation(μ_img, μ_c^+)`; toggle `use_rotation` |
| λ (negation penalty weight) | SpaceVLM (`t`), PoS-Subspaces (`λ`), steering (`α`) | Single ablation knob; all three papers suggest mild values (0.05–0.3) |
| Per-condition vs stacked subspaces | PoS-Subspaces (per-POS) | Headline variant: one SVD per attribute + score averaging; stacked as ablation |
| k+ / k- subspace dimensionality | CLAY (k ablation), PoS-Subspaces (k eigenvectors) | Ablation sweep: `k ∈ {5, 10, 20, 50}` for both |
| Evaluation protocol for negation | VLMs-Do-Not-Understand-Negation (MCQ), When-Negation (MLLM judge) | Report Recall@{1,5,10} + note negation-specific queries; MLLM judge if time allows |

---

## 9. Definition of Done

Track S is complete when:

1. `src/tier2a_subspace.py` exposes `make_get_ranking(query_str, …) → get_ranking(src_idx)`
   with the same interface as Tier-0 and Tier-1 (CONTRACT §7).
2. `output/tier2a_subspace_*.csv` files are written in the shared CSV schema (one file
   per ablation configuration: `_k{k+}_{k-}_{lam}_rot.csv` or similar).
3. **Negation queries no longer collapse:** `-Male, -Mustache` and `-Smiling, +Eyeglasses,
   +Wearing_Hat` both show R@5 > 0.000 — the concrete signal that the asymmetric-negation
   insight works.
4. **MEAN R@5 strictly beats Tier-0 (0.070)** on the 14-query benchmark.
5. `test/test_tier2a_subspace.py` passes all 8 checks (covering orthonormality,
   idempotence, complement, end-to-end ranking, and edge cases).
6. Report sub-section "Asymmetric conditional subspaces" is written.
