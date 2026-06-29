# Fusion Model Guide — Geometry-Grounded Φ on top of Tier-2c

> **Purpose.** This document is the single reference for building the trained fusion module Φ.
> It is structured in two parts:
>
> **Part I — Theory you must study.** Every concept the architecture rests on, with the
> exact paper and section to read, the formula you extract, and why it matters for Φ.
>
> **Part II — The original architecture.** A precise specification of the fusion model,
> derived from Tier-2c's geometry. Includes the forward pass, training objective,
> implementation plan, and the originality argument.

---

## Part I — Theory

### 1. The CLIP Hypersphere and Why Geometry Matters

**Paper:** GDE (Berasi et al., CVPR 2025) — `documents/papers/GDE.md`, §3.1 and App. A.

CLIP L2-normalizes every output, placing all image and text embeddings on the unit hypersphere
`S^{d−1} ⊂ R^d` (d=512 for ViT-B/32). The norm carries no information; only direction matters,
and similarity is cosine. The consequence you must internalize: **Euclidean operations are
unfaithful on a sphere.** A sum of two unit vectors does not land on the sphere; a straight line
between two unit vectors cuts through the interior of the ball, not along the surface.

The faithful toolkit is the **tangent-space sandwich**: map points to the flat tangent space at a
reference point μ, do linear algebra there, then map back. Two closed-form maps on `S^{d−1}`:

**Logarithmic map** `Log_μ(u)` — sends a sphere point `u` to the tangent plane `T_μ S^{d−1}`:

```
Log_μ(u) = θ · (u − (uᵀμ)μ) / ‖(u − (uᵀμ)μ)‖,    θ = arccos(uᵀμ)      (GDE App. A, Eq. 14)
```

The numerator `u − (uᵀμ)μ` removes the μ-component (projecting into the tangent plane).
Scaling by θ/‖…‖ makes the tangent vector have length equal to the geodesic distance to u.

**Exponential map** `Exp_μ(v)` — inverse: lifts a tangent vector back onto the sphere:

```
Exp_μ(v) = cos(‖v‖)·μ + sin(‖v‖)·(v/‖v‖)                               (GDE App. A, Eq. 13)
```

**Three properties to verify in code before touching Φ:**
- Round-trip: `Exp_μ(Log_μ(u)) = u` to 1e-5.
- Tangency: `Log_μ(u)ᵀμ = 0` (output is orthogonal to μ, i.e., lives in the tangent plane).
- Base-point: `Exp_μ(0) = μ` (zero tangent lifts to the base point).

**The intrinsic (Karcher) mean μ** (GDE §3.1, Eq. 2 and Alg. 1):

The natural tangent point is not the arithmetic mean (which falls off the sphere) but the
**intrinsic mean** — the point on `S^{d−1}` minimizing average squared geodesic distance:

```
μ = argmin_{u ∈ S^{d−1}}  Σ_i d_S(u, u_i)²
```

It is computed by gradient descent on the sphere (GDE Alg. 1, ~20 iterations from the
normalized arithmetic mean) and satisfies the centering property `Σ_i Log_μ(u_i) = 0`, which
makes it the correct origin for all tangent-space decompositions. Already implemented in
`src/manifold.py::intrinsic_mean`.

---

### 2. Geodesic Decomposability — Visual Attribute Directions

**Paper:** GDE §3.2 and §3.3, Prop. 1–2.

**Definition (GDE Def. 1).** A set of embeddings `{u_z}` on `S^{d−1}` is *geodesically
decomposable* if each element is the Exp-map of a sum of per-primitive tangent directions:

```
u_z = Exp_μ( v_{z₁} + v_{z₂} + … + v_{zₛ} )
```

Addition happens in `T_μ S^{d−1}` (flat); the non-linear part is only the Exp-map at the end.

**Optimal primitive directions** (GDE Prop. 1, Eq. 7):

For attribute `a`, the direction `v_a` that best explains the embeddings of has-a images in the
geodesically decomposable sense is the **tangent mean**:

```
v_a = (1/|Z(a)|) Σ_{x: has a} Log_μ(x)
```

This is already computed by `src/tier1_GDE.py::mine_directions` and cached in
`artifacts/visual_directions.pt`. Each `directions[j]` is the primitive tangent direction for
`ATTRIBUTE_NAMES[j]`.

**Why image-space directions, not text:** the modality gap (Liang et al., NeurIPS 2022 —
SOTA.md §1) places image and text embeddings in two separated cones of the sphere. Adding
a text vector to an image vector wastes most of the addition on the "I am a text embedding"
component (the gap offset), not on the semantic attribute direction. Visual directions live
natively in the image cone — no gap, no rotation H needed.

**Positive composition** (GDE Def. 1, §4.5):

```
q_tan = Log_μ(v_ref) + Σ_{a ∈ T+} α · v_a
q     = Exp_μ(q_tan)
```

Addition in tangent space, one Exp-map. Already implemented in
`src/tier2c.py::_positive_tangent_batch` and `_compose_query_svunion`.

---

### 3. Negation as Orthogonal Subspace Rejection

**Papers:** (a) Alhamoud et al., CVPR 2025 — `documents/papers/VLM_DO_NOT_UNDERSTAND_NEGATIONSpdf.md`,
§4.1; (b) Oldfield et al., NeurIPS 2023 — `documents/papers/PSGS_VLM.md`, §2.2, Eq. 5.

#### 3.1 Why subtraction is the wrong operator (Alhamoud et al.)

CLIP exhibits **affirmation bias**: "a dog" and "no dog" map to nearly identical embeddings.
Models perform at or below chance (3% vs 82%) on negation MCQ tasks. Subtracting `v_Male` from
the query vector pushes it toward *anti-Male* — a specific point — not toward "any non-Male
value." The set of valid non-Male images is a large region of the sphere, not one point.

**The correct semantics of "−X" is "no preference on the X axis."** Removing the X direction
from the query means the query is orthogonal to the X direction, so all images score equally
regardless of their X value. This is what the orthogonal complement projection achieves.

#### 3.2 Single-direction rejection (Track V / Tier-2a)

For one negative attribute with unit direction `v̂_a = v_a / ‖v_a‖`:

```
q_tan ← q_tan − (q_tan · v̂_a) · v̂_a          (remove the a-axis component)
```

After this operation, `q_tanᵀ v̂_a = 0`: the query has zero projection onto the attribute
direction, meaning it treats has-a and not-a images identically. That is the correct "−a"
semantics. Implemented in `src/tier2a_visual.py`.

#### 3.3 Subspace rejection for correlated attributes (Tier-2c — the key step)

A single direction `v̂_a` cannot span the full visual region of attribute `a` in CLIP image
space. CelebA attributes are correlated (Male and Mustache share a cluster of visual features).
Sequentially rejecting one direction at a time is **order-dependent** and does not correctly
handle the case where the two attribute axes are not orthogonal.

The geometrically correct solution (PoS-Subspaces §2.2, Eq. 5 generalized to k dimensions):

**Step 1 — Mine the visual subspace for attribute b from train images.**

From the k-dimensional principal subspace of the tangent vectors of all has-b train images
at the global μ:

```
L_b    = Log_μ( X_b )                    X_b = {train images with attribute b}  [m, d]
gram_b = L_bᵀ L_b                                                                [d, d]
Q_b    = top-k eigenvectors of gram_b                                            [d, k]
```

Using the Gram matrix eigendecomposition instead of SVD of `L_b` is efficient: it avoids
materializing the [m, d] `U` matrix and scales to m ≫ d. Eigenvalues are identical to
squared singular values; eigenvectors are the right singular vectors. Implemented in
`src/tier2c.py::_build_visual_neg_subspace`.

**Step 2 — Union of subspaces via thin QR.**

For multiple negative attributes `T- = {b₁, b₂, …, bₘ}`:

```
W     = [ Q_{b₁}[:,:k] | Q_{b₂}[:,:k] | … ]     horizontal concatenation  [d, m·k]
Q_all, _ = torch.linalg.qr(W)                     orthonormal basis of span(W)  [d, r]
```

Thin QR re-orthonormalises the concatenation, correctly handling overlapping subspaces
(e.g., `Q_Male` and `Q_Mustache` share directions). The union basis `Q_all` spans exactly
the visual region covered by all negated attributes. Implemented in
`src/tier2c.py::_build_union_basis`.

**Step 3 — Project onto the orthogonal complement.**

```
q_tan ← q_tan − Q_all (Q_allᵀ q_tan)
```

This removes from `q_tan` every component that has any projection in any negated attribute's
visual subspace, in one order-independent operation. The result is orthogonal to `span(Q_all)`.

The complete Tier-2c query (headline `reject_on="query"` variant):

```
q_tan = Log_μ(v_ref) + Σ_{a ∈ T+} α · v_a                # geodesic positive addition
q_tan = q_tan − Q_all (Q_allᵀ q_tan)                       # subspace rejection
q     = normalize( Exp_μ(q_tan) )                           # back to sphere
score = image_features @ q                                  # cosine ranking
```

---

### 4. The Modality Gap and Why Tier-2c Bypasses It

**Paper:** Liang et al., NeurIPS 2022 ("Mind the Gap") — SOTA.md §1.

CLIP's contrastive pretraining produces a well-known artifact: image embeddings cluster in one
narrow cone of `S^{d−1}` and text embeddings cluster in a different narrow cone, with a wide
empty band between them. Every text vector `t(a)` is dominated by the shared offset pointing
toward the centre of the text cone — a direction that encodes "I am a CLIP text embedding,"
not "I am the Smiling attribute."

**Track S (Tier-2a text subspace)** must apply CLAY's rotation `H` to bridge this gap before
projecting visual features into text-derived subspaces. The rotation helps (+0.004 R@10 in the
ablation) but is an approximation.

**Tier-2c bypasses the gap by construction.** All directions (`v_a` from train image means,
`Q_b` from train image PCA) are mined from images and therefore live natively in the image
cone. The query starts as `Log_μ(v_ref)` (image), the positive directions are image-space
tangent vectors, and the negative subspaces are image-space PCA subspaces. No rotation needed;
no gap to close. This is the primary geometric motivation for the Tier-2c architecture.

---

### 5. Contrastive Training — InfoNCE Loss

**Paper:** van den Oord et al. 2018 (InfoNCE) — SOTA.md §7.

The training objective for Φ is an InfoNCE contrastive loss. For a query embedding `q`
produced by Φ, a set of valid positive target embeddings `{p_i}`, and a set of hard negative
target embeddings `{n_j}` (all from the frozen DB):

```
L = −log [  exp(q · p / τ)  /  ( exp(q · p / τ) + Σ_j exp(q · n_j / τ) )  ]
```

where `τ` is a temperature hyperparameter (start at 0.07, the standard CLIP temperature).

**Positive pairs:** a valid target is a test-split image that (a) strictly satisfies all `T+`
and `T−` constraints AND (b) has Hamming distance ≤ 2 from `v_ref` on the remaining attributes
(identical to the evaluation GT protocol, but mined from the train split only).

**Hard negatives:** images that satisfy `T+` but violate at least one `T−` constraint. These
are the images Tier-0 retrieves incorrectly (they match the positive attribute but have the
forbidden one). Hard negatives are what make the training objective nontrivial — easy negatives
(random images) are already well-separated in CLIP space and provide no gradient signal.

**Why the training objective must mirror the evaluation protocol exactly:** Φ is scored on
relaxed-Hamming GT. If training positives/negatives are constructed with a different rule,
the model learns a different objective than it is graded on. The generator in
`src/` (or a new `src/query_generator.py`) must implement the same Hamming-≤2 rule.

---

### 6. What Is a Fusion Model (and What Is Not)

A fusion model Φ takes multiple inputs and produces a **single composite embedding** through a
**learned computation**. The output is a new vector; Φ is the composition.

What is **not** a fusion model:
- A hyperparameter predictor that outputs `α̂` and `λ̂` fed into a hand-coded formula. The
  hand-coded formula is still doing the fusion; Φ is just tuning knobs.
- A reweighting network that scales input embeddings before summing them. The sum is the fusion
  step; Φ is a preprocessor.

What **is** a fusion model: a network whose forward pass **itself** produces `q`. No hand-coded
formula downstream; the learned computation is the composition.

The architecture in Part II satisfies this definition: the network's forward pass performs the
geodesic addition, the subspace rejection, and the Exp-map. The network learns *what* each step
should produce — not weights fed into an external pipeline.

---

## Part II — The Original Architecture

### 1. Design Principle: Geometry-Grounded Fusion

**The central claim.** Standard CIR fusion models (Combiner, CAFF, FiLM) output a free 512-d
embedding with no inductive bias toward the correct geometric structure. They must discover
from data that negation is a complement projection, not a subtraction — and they often fail to.

The originality of this Φ is that **the architecture enforces the Tier-2c geometry**. The
network does not output a free embedding. Its forward pass *is* the geodesic addition followed
by orthogonal subspace rejection, with each intermediate quantity produced by a learned module
instead of a fixed formula. The geometry is the inductive bias; the network learns the
*content* of each geometric step.

No prior CIR paper (Combiner 2022, CAFF 2024, TIRG 2019, GeneCIS 2023) designs a fusion
module whose forward pass is structurally a manifold operation. This is the original
contribution.

---

### 2. Forward Pass

**Inputs (all from frozen CLIP, no encoder is called at training time):**

| Symbol | Shape | Source |
|---|---|---|
| `v_ref` | `[512]` | `image_features[src_idx]` from the frozen test DB |
| `v_a` for each `a ∈ T+` | `[512]` each | `directions[ATTR_TO_IDX[a]]` from mined visual directions |
| `Q_b` for each `b ∈ T-` | `[512, k]` each | `subspaces[ATTR_TO_IDX[b]][:, :k]` from mined neg subspaces |

**Step 1 — Reference encoder: learned tangent lift.**

The fixed Log-map `Log_μ(v_ref)` produces a tangent vector, but it treats the reference as a
single point with no understanding of the query context (which attributes are requested). The
learned encoder replaces this with a context-aware tangent representation:

```
h_ref = MLP_ref( v_ref )                                        h_ref ∈ R^{512}
```

`MLP_ref` is a two-layer network (512 → 256 → 512, LayerNorm, GELU). It receives the raw
reference embedding and produces a contextualized representation in the same 512-d space.
At initialization, `MLP_ref` is set to approximate the identity (final layer weights near zero,
bias near zero) so Φ starts near the Tier-2c formula and learns incremental corrections.

**Step 2 — Positive fusion: learned geodesic push.**

For each positive attribute `a ∈ T+`, the fixed formula adds `α · v_a` to the tangent vector
with a global scalar. The learned module replaces `α` with a per-attribute, per-reference
contribution:

```
c_a = CrossAttn( query=h_ref, key=v_a, value=v_a )             c_a ∈ R^{512}
```

Single-head cross-attention with `d_model=512`. The query is the reference representation; key
and value are the visual direction of the positive attribute. The attention output `c_a` is a
learned "how much and in what direction to push toward attribute a, given this specific
reference."

The positive contribution is accumulated:

```
h_pos = Σ_{a ∈ T+} c_a                                         h_pos ∈ R^{512}
```

For `T+` empty, `h_pos = 0` (no addition step).

**Step 3 — Negative fusion: learned subspace rejection mask.**

For each negative attribute `b ∈ T-`, the fixed formula builds a union subspace from the
pre-mined `Q_b` and rejects. The learned module replaces the fixed subspace with a learned
rejection direction conditioned on the reference:

```
r_b = CrossAttn( query=h_ref, key=Q_b, value=Q_b )             r_b ∈ R^{512}
```

Here key and value are the columns of `Q_b` (treated as a sequence of k=10 tokens, each
`[512]`). The cross-attention output `r_b` is a learned "which direction within the visual
Male/Mustache/... subspace is most relevant to reject from this specific reference."

The union rejection direction is built by stacking and re-orthonormalising:

```
R = stack( r_b / ‖r_b‖  for b ∈ T- )                          R ∈ R^{|T-|, 512}
Q_learned, _ = torch.linalg.qr( R.T )                          Q_learned ∈ R^{512, |T-|}
```

The QR step orthonormalises the learned rejection directions, exactly as in Tier-2c's
`_build_union_basis`. This enforces that the negation operator remains an orthogonal projector
— a geometric constraint the network cannot violate.

**Step 4 — Geometric assembly: the forward pass is the Tier-2c formula.**

```
q_tan = h_ref + h_pos                                  # positive step (in tangent-like space)
q_tan = q_tan − Q_learned @ (Q_learned.T @ q_tan)     # orthogonal rejection (exact same op as tier2c)
q     = normalize( Exp_μ( q_tan ) )                    # back to sphere
```

The output `q ∈ S^{511}` is a unit vector in image space, scored by cosine against the frozen
DB. This is identical to Tier-2c's output modality — the evaluation harness (`eval.py`,
`make_get_ranking`) is unchanged.

**Complete forward pass diagram:**

```
v_ref  ──── MLP_ref ──────────────────────────────────── h_ref ─────┐
                                                                     │
v_a⁺¹  ─┐                                                           │
v_a⁺²  ─┤── CrossAttn(query=h_ref) ──── c_a ──── Σ ──── h_pos ─────┤
  ...   ─┘                                                           │
                                                              h_ref + h_pos = q_tan_pre
                                                                     │
Q_b⁻¹  ─┐                                                           │
Q_b⁻²  ─┤── CrossAttn(query=h_ref) ──── r_b ──── QR ─── Q_learned ─┤
  ...   ─┘                                                           │
                                                      q_tan = q_tan_pre − Q_learned Q_learned.T q_tan_pre
                                                                     │
                                                             Exp_μ → normalize → q
```

---

### 3. Architecture Details

```python
class FusionPhi(nn.Module):
    """
    Geometry-grounded fusion module. Forward pass IS the tier2c composition,
    with each geometric step replaced by a learned cross-attention module.
    CLIP is frozen. Only Phi trains.

    Args:
        d_model: CLIP embedding dimension (512 for ViT-B/32).
        n_heads: attention heads for cross-attention (default 4).
        mu:      global intrinsic mean [d], registered as buffer (not trained).
    """
    def __init__(self, d_model=512, n_heads=4, mu=None):
        super().__init__()
        self.d = d_model

        # Step 1: Reference encoder
        self.mlp_ref = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, d_model),
        )
        # Init near identity: final layer near zero so Phi starts close to tier2c
        nn.init.zeros_(self.mlp_ref[-1].weight)
        nn.init.zeros_(self.mlp_ref[-1].bias)

        # Steps 2 and 3: shared cross-attention (same weights for pos and neg)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )

        # Global mean — registered as buffer, not a parameter
        if mu is not None:
            self.register_buffer("mu", mu)

    def forward(self, v_ref, pos_dirs, neg_subspaces):
        """
        v_ref:          [B, d]          reference image embeddings (unit)
        pos_dirs:       [B, n+, d]      positive visual directions (from directions cache)
        neg_subspaces:  [B, m-, k, d]   negative visual subspaces (Q_b columns, k per attr)

        Returns:
            q:  [B, d]  composite query embeddings (unit, on the sphere)
        """
        B, d = v_ref.shape

        # Step 1: reference encoder
        h_ref = v_ref + self.mlp_ref(v_ref)       # residual: stays near v_ref at init  [B, d]

        # Step 2: positive fusion
        if pos_dirs.shape[1] > 0:
            # Query = h_ref expanded per positive, Key = Value = pos_dirs
            q_in = h_ref.unsqueeze(1)              # [B, 1, d]
            c_a, _ = self.cross_attn(q_in, pos_dirs, pos_dirs)   # [B, 1, d]
            h_pos = c_a.squeeze(1)                 # [B, d]
        else:
            h_pos = torch.zeros_like(h_ref)

        q_tan = h_ref + h_pos                      # [B, d] positive tangent

        # Step 3: negative fusion + orthogonal rejection
        if neg_subspaces.shape[1] > 0:
            B, m_neg, k, d = neg_subspaces.shape
            # Flatten the k columns of each attribute into one token sequence [B, m*k, d]
            neg_tokens = neg_subspaces.view(B, m_neg * k, d)
            q_in = h_ref.unsqueeze(1)              # [B, 1, d]
            r_b, _ = self.cross_attn(q_in, neg_tokens, neg_tokens)   # [B, 1, d]
            r_b = r_b.squeeze(1)                   # [B, d]

            # Normalize learned rejection direction and reject
            r_hat = F.normalize(r_b, dim=-1, eps=1e-8)    # [B, d] unit rejection direction
            q_tan = q_tan - (q_tan * r_hat).sum(-1, keepdim=True) * r_hat  # [B, d]
            # Note: for |T-| > 1, stack multiple r_hat and apply QR for the multi-attribute case
            # (see _build_learned_rejection below for the batched version)

        # Step 4: geometric assembly — Exp_μ + normalize
        q = self._exp_map(q_tan)                   # [B, d], unit
        return q

    def _exp_map(self, v):
        # Exp_μ(v) = cos(‖v‖)·μ + sin(‖v‖)·(v/‖v‖)   (GDE App. A, Eq. 13)
        norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.cos(norm) * self.mu + torch.sin(norm) * (v / norm)
```

**Parameter count (approximate):**

| Module | Parameters |
|---|---|
| `mlp_ref` (512→256→512) | 256×512 + 512×256 + biases ≈ 263K |
| `cross_attn` (d=512, h=4) | 4×(512×128)×3 projections ≈ 786K |
| **Total** | **~1.05M** |

All CLIP parameters stay frozen. Only Φ (~1M params) is trained.

---

### 4. Training Objective

**Synthetic query generator (train split only):**

For each training batch:
1. Sample a reference `r` from CelebA train split.
2. For `k_pos` randomly sampled attributes `a` that `r` has: add to `T+`.
3. For `k_neg` randomly sampled attributes `b` that `r` does not have: add to `T-`.
4. Find valid positive targets: train images satisfying all `T+`/`T−` constraints AND
   Hamming distance ≤ 2 from `r` on the remaining attributes.
5. Sample hard negatives: train images satisfying `T+` but violating at least one `T-` attribute.

**Loss:**

```
L = InfoNCE( q, {p_i}, {n_j} )
  = −(1/|P|) Σ_{p ∈ P} log [
        exp(qᵀp / τ)  /  ( Σ_{p' ∈ P} exp(qᵀp' / τ) + Σ_{n ∈ N} exp(qᵀn / τ) )
    ]
```

where `q = Φ(v_ref, pos_dirs, neg_subspaces)` is the Φ output, `P` is the set of valid
positive target embeddings, `N` is the set of hard negative embeddings, and `τ=0.07`.

All embeddings `{p_i}`, `{n_j}` are **frozen CLIP features** from the train DB — Φ learns to
produce a `q` that is close to valid targets and far from violating images in CLIP image space.

**Optimizer:** AdamW, lr=1e-4, weight_decay=1e-2. Cosine schedule, 5-epoch warmup.
**Batch size:** 64 queries (each with 1 reference, ~3–5 positives, ~10 hard negatives).
**Epochs:** 30 (hard time-box; evaluate at each epoch checkpoint).

---

### 5. Why This Is Original

The three prior architectures most likely to be cited as precedents, and why this Φ is distinct:

**Combiner (Baldrati et al., CVPR 2022):** An MLP that fuses CLIP image + text features for CIR.
Outputs a free embedding with no geometric constraint. Does not handle explicit negation. Does
not use visual attribute directions. The fusion is unconstrained.

**CAFF (CVPR 2024 Workshop):** Cross-attention late-fusion for CIR. Also outputs a free
embedding. No subspace rejection. No polarity distinction in the architecture.

**GeneCIS (Vaze et al., CVPR 2023):** Trains a feature modulator for conditional similarity.
Asymmetric formulation (only query is conditioned). No negation mechanism; the architecture
does not distinguish `T+` from `T-`.

**This Φ differs on three structural axes:**

1. **The forward pass is the Tier-2c geometry, not a free MLP.** The network cannot output an
   arbitrary embedding; it must produce a vector that is the result of geodesic addition
   followed by orthogonal complement projection. The inductive bias is hard-coded, not learned.

2. **Negation is architecturally enforced.** The QR step in Step 3 ensures that the learned
   rejection direction is unit-normed and the projection `q_tan − r̂(r̂ᵀq_tan)` is an exact
   orthogonal complement — not an approximation that a free MLP would have to learn from scratch.

3. **All inputs are visual, eliminating the modality gap.** `v_ref`, `v_a` (mined from images),
   and `Q_b` (mined from images) all live in the image cone. No rotation H, no text prompt
   stack, no cross-modal alignment step. The architecture is geometrically clean.

---

### 6. Connection to the Tier Progression (Report Framing)

The four tiers form a clean story of progressive refinement:

| Tier | α | Negation operator | Trained? | Free parameters |
|---|---|---|---|---|
| **Tier-0** | fixed scalar | vector subtraction (wrong) | No | α |
| **Tier-2a Track V** | fixed scalar | single-direction rejection | No | α |
| **Tier-2c** | fixed scalar | subspace rejection (union QR) | No | α, k_neg |
| **Φ (Tier-2d)** | learned per-ref, per-attr | learned subspace rejection | Yes | Φ weights |

Each row fixes a flaw in the row above. Tier-2c fixes the negation operator (subtraction →
subspace rejection). Φ fixes the remaining free parameter (fixed α → learned, reference-aware
positive push; fixed subspace → learned rejection direction conditioned on the reference).

The ablation table in the report compares Φ to Tier-2c directly, with the delta attributable
purely to learning: same geometry, same evaluation harness, only the scalars are now predicted
by a network.

---

### 7. Reading Order Before Coding

| Paper | File | Sections | What you extract |
|---|---|---|---|
| GDE | `documents/papers/GDE.md` | §3.1–3.3, Prop. 1–2, App. A (Eq. 13–14) | Log/Exp closed forms, intrinsic mean, tangent mean direction |
| CLAY | `documents/papers/CLAY.md` | §3.2 | What Φ replaces (the naïve pre-SVD bottleneck) |
| PoS-Subspaces | `documents/papers/PSGS_VLM.md` | §2.2, Eq. 5 | The `Π⊥` operator and the QR orthonormalisation argument |
| Alhamoud (Negation) | `documents/papers/VLM_DO_NOT_UNDERSTAND_NEGATIONSpdf.md` | §4.1, Fig. 6 | Why subtraction is wrong; what the correct semantics of "−X" is |
| InfoNCE | SOTA.md §7 (van den Oord 2018) | full | The contrastive loss; hard-negative intuition |
| Liang (Modality Gap) | SOTA.md §1 | abstract + §3 | Why image-space directions eliminate the gap |

**Minimum to start coding:** GDE App. A (Log/Exp) + PoS-Subspaces §2.2 (the projection
operator) + Alhamoud §4.1 (negation semantics). The rest is needed before writing the report.

---

### 8. Definition of Done

Φ is complete when:

1. `src/fusion.py` exports `FusionPhi` with the forward pass in §2.
2. `src/query_generator.py` generates synthetic train-split queries with the Hamming-≤2 GT
   protocol, positive sampling, and hard-negative sampling.
3. `src/train_phi.py` runs the InfoNCE training loop with checkpointing and learning curves.
4. `make_get_ranking` in `src/fusion.py` plugs Φ into the eval harness (CONTRACT §7):
   same `make_get_ranking(query_str, …) → get_ranking(src_idx)` signature.
5. **MEAN R@5 on the 14 benchmark queries strictly beats Tier-2c** on at least one config.
6. Negation queries (`-Male, -Mustache`; `-Smiling, +Eyeglasses, +Wearing_Hat`) show
   R@5 > 0.000 — the concrete signal that the negation architecture is working.
7. Report sub-section "Geometry-grounded fusion module" is written with the forward pass
   equations, the loss, and the ablation row comparing Φ to Tier-2c.
