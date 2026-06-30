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

### 0. Inputs Are Literal Textual Constraints (Spec Compliance)

The project specification (§3, criterion 1) requires the architecture to **"natively accept and
process multiple textual conditions alongside a visual reference"**, and (criterion 2) to
replace **"the standard concatenation of textual ... embeddings prior to SVD"** with a learned
fusion mechanism that **"dynamically re-weight[s] features based on the provided text
conditions."** Both criteria are stated in terms of the textual embedding side of the problem —
not a closed-vocabulary lookup table of pre-mined visual prototypes.

This rules out grounding Φ in the mined visual directions/subspaces from Track V / Tier-2c
(`directions[a]`, `visual_neg_subspaces[b]`). Those are valid, well-motivated training-free
methods, but their forward pass never touches a CLIP text embedding, so a Φ built on them would
not satisfy "natively accept and process textual conditions," and would not engage with the
"naïve pre-SVD bottleneck" at all, since there is no SVD-over-text step in that pipeline to
replace.

**Φ is therefore grounded in Track S's machinery instead of Track V's.** The query format stays
exactly `(+Smiling, −Eyeglasses)` — attribute names with a polarity sign, never free text typed
by a user — but each attribute name is resolved to its **CLIP text embeddings** via the existing
prompt bank (`artifacts/clip_attr_prompt_bank.pt`, built by `src/clip_prompts.py`):

```
T_a = prompt_bank[ATTR_TO_IDX[a], :n_a]      # [n_a, 512], n_a ≈ 60 paraphrases of attribute a
```

This is the same object CLAY/Tier-1 and Track S (`tier2a_S.py`) already build their SVD
subspaces from. Φ's job is to **replace that SVD step with a learned, dynamically-reweighting
cross-attention**, exactly as criterion 2 names ("cross-attention layers ... that dynamically
re-weight features based on the provided text conditions").

---

### 1. Design Principle: Learned Fusion Over a Geometric Skeleton

**The central claim.** CLAY's fusion of multiple textual conditions is a **fixed, query-agnostic
linear operator**: stack every prompt embedding into one matrix, run one SVD, keep the top-k
right singular vectors. The same projection matrix `P_c = V_k V_k^T` is applied regardless of
which attributes are positive vs. negative, regardless of how many prompts each attribute
contributes, and regardless of the reference image. This is exactly the "naïve concatenation
... prior to SVD" the spec singles out as the bottleneck to overcome.

Φ replaces the SVD with a **learned cross-attention layer that plays the same structural role**
(producing a subspace basis from a stack of condition embeddings) but is no longer (a) fixed
across queries, (b) polarity-blind, or (c) reference-blind:

| CLAY's SVD step | Φ's cross-attention replacement |
|---|---|
| One SVD over `[T+; T−]` stacked together | Separate, **learned** attention over `T+` and over `T−` |
| Fixed basis `V_k`, independent of `v_ref` | Basis **conditioned on `v_ref`** via cross-attention query |
| No notion of polarity (CLAY has no `+/−`) | Positive and negative paths use the **same weights, opposite roles** (addition vs. rejection) — the network must learn why they differ, rather than the architecture hard-coding the formula |
| Static: same subspace for every reference image sharing a query string | Dynamic: **per-reference re-weighting** — a query's contribution to the fused direction changes with which reference it is composing against |

This is the literal "explore more advanced fusion mechanisms ... cross-attention layers ...
that dynamically re-weight features based on the provided text conditions" the spec asks for —
not a metaphor for it.

The geometric skeleton (geodesic addition in tangent space, orthogonal-complement rejection,
Exp-map back to the sphere) is retained from Tier-2c/Track S as the **inductive bias**: Φ does
not output a free 512-d embedding the way Combiner or CAFF do. It outputs a learned tangent
contribution that is composed through the same proven-correct manifold operations. The network
learns the *content* (which directions matter, how to combine multiple paraphrases of the same
attribute, how to weigh multiple attributes against the reference); the *structure* (addition is
additive, negation is rejection) stays geometrically guaranteed.

No prior CIR paper (Combiner 2022, CAFF 2024, TIRG 2019, GeneCIS 2023) designs a fusion module
whose attention layer is structurally a *replacement for SVD subspace construction*, conditioned
on the reference image, feeding a manifold-correct composition operator. This is the original
contribution, and it is the one that engages directly with the spec's two named criteria.

---

### 2. Forward Pass

**Inputs (per query, all from frozen CLIP, no new encoder calls beyond the existing prompt bank):**

| Symbol | Shape | Source |
|---|---|---|
| `v_ref` | `[512]` | `image_features[src_idx]` from the frozen test DB |
| `T_a` for each `a ∈ T+` | `[n_a, 512]` | `prompt_bank[ATTR_TO_IDX[a], :n_a]` — text embeddings, one stack per positive attribute |
| `T_b` for each `b ∈ T−` | `[n_b, 512]` | `prompt_bank[ATTR_TO_IDX[b], :n_b]` — text embeddings, one stack per negative attribute |

Each `T_a`/`T_b` is a stack of ~60 CLIP text embeddings (paraphrases of the same attribute) —
exactly CLAY's per-condition prompt stack, kept **per-attribute and per-polarity separate**
(never concatenated across attributes or across `+`/`−`, which is precisely the departure from
CLAY's naïve stacking).

**Step 0 — Modality-gap centering (fixed, not learned).**

Before any text embedding reaches the network, apply the closed-form correction already proven
in `tier0_enhanced.py` (FIX 1, +63% R@5 relative, the single largest lever found in this
project):

```
t̂ = normalize( t − μ_txt )      for every prompt embedding t in every T_a, T_b
```

`μ_txt` is the mean of all 40 attributes' text embeddings (precomputed once, frozen). This is a
deliberate non-learned step: the dominant text-cone offset has a known closed form, so Φ's
learned capacity is spent on attribute-specific and reference-specific reasoning, not on
rediscovering a mean-subtraction from scratch. This is the one place text and image modalities
need explicit reconciliation — and it is the cost of using literal text inputs that the
visual-prototype version of Φ did not have to pay (see §6, Note on Tier-2c).

**Step 1 — Reference encoder.**

```
h_ref = v_ref + MLP_ref( v_ref )                                h_ref ∈ R^{512}
```

Residual two-layer MLP (512 → 256 → 512, LayerNorm, GELU), initialized near-identity (final
layer weights/bias near zero) so Φ starts close to a sensible default and learns corrections.

**Step 2 — Positive fusion: cross-attention replaces the SVD, per attribute.**

For each positive attribute `a ∈ T+`, instead of computing `V_k` from a fixed SVD of `T̂_a`,
a learned cross-attention layer attends from the reference onto the attribute's paraphrase
stack, producing a single fused contribution:

```
c_a = CrossAttn( query=h_ref, key=T̂_a, value=T̂_a )             c_a ∈ R^{512}
```

This **is** the dynamic re-weighting the spec asks for: instead of SVD's fixed, data-independent
top-k truncation, the attention weights over the `n_a` paraphrases are a function of `h_ref` —
different references can emphasize different paraphrases of "Smiling" (e.g., one that better
matches a subtle smile vs. an open one). Multiple positive attributes are combined by summation
in tangent space, after each has been independently attended to — so one attribute with many
prompts cannot dominate another with few (the exact failure mode of CLAY's naïve concatenation,
Track S's S_plan.md §0.1):

```
h_pos = Σ_{a ∈ T+} c_a                                          h_pos ∈ R^{512}
```

**Step 3 — Negative fusion: cross-attention produces the rejection direction, per attribute.**

Symmetric construction, **same cross-attention weights** (shared module, different role) applied
to each negative attribute's prompt stack:

```
r_b = CrossAttn( query=h_ref, key=T̂_b, value=T̂_b )             r_b ∈ R^{512}
```

Sharing weights between Step 2 and Step 3 is a deliberate parameter-efficiency and inductive-bias
choice: the network learns one re-weighting operator over a paraphrase stack, conditioned on the
reference; what differs between positive and negative is only how the *output* is used
downstream (added vs. projected out), not how it is computed. This forces the network to learn a
**general-purpose attribute-fusion operator**, not two unrelated sub-networks — directly
answering the spec's "the model must learn ... how to dynamically weigh these inputs," for both
polarities with one mechanism.

**Step 4 — Orthogonal rejection (geometric, not learned).**

```
R = stack( r_b / ‖r_b‖  for b ∈ T− )                            R ∈ R^{|T−|, 512}
Q_learned, _ = torch.linalg.qr( R.T )                            Q_learned ∈ R^{512, |T−|}
q_tan = (h_ref + h_pos) − Q_learned @ (Q_learned.T @ (h_ref + h_pos))
```

The QR re-orthonormalisation and the projection are fixed manifold operations (identical
mechanism to Tier-2c's `_build_union_basis` / rejection step) — this is where "−X means any
value but X" (Alhamoud et al., §3 of Part I) is **architecturally guaranteed**, not hoped for.
A free MLP fusion model (Combiner-style) has no such guarantee; it must discover the correct
negation geometry purely from gradient signal, and may not.

**Step 5 — Geometric assembly.**

```
q = normalize( Exp_μ( q_tan ) )                                  q ∈ S^{511}
score = image_features @ q                                       cosine ranking, same harness
```

**Complete forward pass diagram:**

```
v_ref ── MLP_ref ──────────────────────────────────────── h_ref ────────┐
                                                                          │
T̂_a1 (text, n_a1×512) ─┐                                                │
T̂_a2 (text, n_a2×512) ─┤── CrossAttn(query=h_ref) per attr ── c_a ── Σ ── h_pos
  ...  T+               ─┘                                                │
                                                            h_ref+h_pos = q_tan_pre
                                                                          │
T̂_b1 (text, n_b1×512) ─┐  [SAME cross-attn weights as above]            │
T̂_b2 (text, n_b2×512) ─┤── CrossAttn(query=h_ref) per attr ── r_b ──┐    │
  ...  T−               ─┘                                          │    │
                                              normalize ── QR ── Q_learned
                                                                          │
                                      q_tan = q_tan_pre − Q_learned Q_learned.T q_tan_pre
                                                                          │
                                                          Exp_μ → normalize → q
```

---

### 3. Architecture Details

```python
class FusionPhi(nn.Module):
    """
    Geometry-grounded fusion module over LITERAL TEXTUAL CONSTRAINTS.
    Cross-attention replaces CLAY's per-condition SVD (spec criterion 2);
    the manifold composition (geodesic addition + orthogonal rejection)
    stays fixed as the inductive bias (spec criterion 1's "+/- interaction").
    CLIP is frozen; only Phi trains.

    Args:
        d_model:   CLIP embedding dimension (512 for ViT-B/32).
        n_heads:   attention heads for the shared cross-attention (default 4).
        mu:        global intrinsic mean [d], registered as buffer (not trained).
        mu_txt:    mean of all 40 attribute text embeddings [d], fixed centering offset.
    """
    def __init__(self, d_model=512, n_heads=4, mu=None, mu_txt=None):
        super().__init__()
        self.d = d_model

        self.mlp_ref = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, d_model),
        )
        nn.init.zeros_(self.mlp_ref[-1].weight)   # near-identity at init
        nn.init.zeros_(self.mlp_ref[-1].bias)

        # ONE shared cross-attention module for both polarities (Step 2 and Step 3)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )

        self.register_buffer("mu", mu)
        self.register_buffer("mu_txt", mu_txt)    # FIX-1-style centering offset

    def _center(self, T):
        # Step 0 — fixed modality-gap correction (Liang et al. 2022; tier0_enhanced FIX 1).
        # T: [n, d] raw text embeddings for one attribute's paraphrase stack.
        return F.normalize(T - self.mu_txt, dim=-1)

    def _attend(self, h_ref, T_hat):
        # Shared cross-attention: query = reference, key = value = centered prompt stack.
        # T_hat: [n, d] -> unsqueeze to [1, n, d] (batch handled by caller's loop/vmap).
        q_in = h_ref.unsqueeze(0).unsqueeze(0)     # [1, 1, d]
        kv = T_hat.unsqueeze(0)                    # [1, n, d]
        out, _ = self.cross_attn(q_in, kv, kv)      # [1, 1, d]
        return out.squeeze(0).squeeze(0)            # [d]

    def forward(self, v_ref, T_pos_stacks, T_neg_stacks):
        """
        v_ref:          [d]                reference image embedding (unit)
        T_pos_stacks:    list of [n_a, d]   one raw prompt stack per positive attribute
        T_neg_stacks:    list of [n_b, d]   one raw prompt stack per negative attribute

        Returns:
            q: [d]  composite query embedding (unit, on the sphere)

        (Batched/vmapped version used in training; single-query form shown for clarity —
        mirrors tier2c.py's single-vector seam vs. batched-driver split.)
        """
        h_ref = v_ref + self.mlp_ref(v_ref)                      # [d]

        h_pos = torch.zeros_like(h_ref)
        for T_a in T_pos_stacks:
            h_pos = h_pos + self._attend(h_ref, self._center(T_a))

        q_tan_pre = h_ref + h_pos                                  # [d]

        if T_neg_stacks:
            r_list = [F.normalize(self._attend(h_ref, self._center(T_b)), dim=-1, eps=1e-8)
                       for T_b in T_neg_stacks]
            R = torch.stack(r_list, dim=0)                          # [m-, d]
            Q_learned, _ = torch.linalg.qr(R.T)                     # [d, m-]
            q_tan = q_tan_pre - Q_learned @ (Q_learned.T @ q_tan_pre)
        else:
            q_tan = q_tan_pre

        return self._exp_map(q_tan)

    def _exp_map(self, v):
        # Exp_μ(v) = cos(‖v‖)·μ + sin(‖v‖)·(v/‖v‖)   (GDE App. A, Eq. 13)
        norm = v.norm().clamp(min=1e-8)
        return torch.cos(norm) * self.mu + torch.sin(norm) * (v / norm)
```

**Parameter count (approximate):**

| Module | Parameters |
|---|---|
| `mlp_ref` (512→256→512) | 256×512 + 512×256 + biases ≈ 263K |
| `cross_attn` (d=512, h=4, shared pos/neg) | 4×(512×128)×3 projections ≈ 786K |
| **Total** | **~1.05M** |

All CLIP parameters stay frozen. Only Φ (~1M params) is trained. The shared cross-attention
(vs. separate pos/neg modules) halves the attention parameter count relative to the earlier
visual-prototype design and forces a single general-purpose fusion operator.

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

where `q = Φ(v_ref, T_pos_stacks, T_neg_stacks)` is the Φ output — the literal prompt-bank text
stacks per attribute, not pre-mined visual prototypes (§0). `P` is the set of valid positive
**target image** embeddings, `N` is the set of hard negative **target image** embeddings, and
`τ=0.07`.

`{p_i}`, `{n_j}` are **frozen CLIP image features** from the train DB — only the *constraint*
side of the input is textual; the retrieval target was always images, exactly as the spec's
`v_target ∈ V` requires. Φ learns to produce a `q` that is close to valid target images and far
from constraint-violating images, using text as the conditioning signal and images as the
supervision signal.

**Optimizer:** AdamW, lr=1e-4, weight_decay=1e-2. Cosine schedule, 5-epoch warmup.
**Batch size:** 64 queries (each with 1 reference, ~3–5 positives, ~10 hard negatives).
**Epochs:** 30 (hard time-box; evaluate at each epoch checkpoint).

---

### 5. Why This Is Original

The three prior architectures most likely to be cited as precedents, and why this Φ is distinct:

**Combiner (Baldrati et al., CVPR 2022):** An MLP that fuses CLIP image + text features for CIR.
Outputs a free embedding with no geometric constraint. Operates on a single text embedding per
condition (no paraphrase stack, no per-condition subspace notion). Does not handle explicit
negation as a distinct operator.

**CAFF (CVPR 2024 Workshop):** Cross-attention late-fusion for CIR. Also outputs a free
embedding. No subspace/SVD-replacement framing, no polarity distinction in the architecture.

**GeneCIS (Vaze et al., CVPR 2023):** Trains a feature modulator for conditional similarity.
Asymmetric formulation (only query is conditioned). No negation mechanism; the architecture
does not distinguish `T+` from `T-`.

**This Φ differs on three structural axes:**

1. **Cross-attention is positioned as a literal SVD replacement, not a generic fusion block.**
   Φ attends over the *same per-attribute paraphrase stack* `T_a` that CLAY/Track S feed into
   SVD, conditioned on `v_ref`. No prior CIR paper frames cross-attention this way — as a
   learned, reference-conditioned substitute for a specific named bottleneck (the spec's
   "naïve concatenation ... prior to SVD"), rather than as an unconstrained fusion layer.

2. **Negation is architecturally enforced, not learned from scratch.** The QR + projection step
   (Step 4) guarantees `q_tan ⊥ span(Q_learned)` by construction — an exact orthogonal
   complement (Alhamoud's "any value but X" semantics, Part I §3.1) — regardless of what the
   cross-attention learns to output as `r_b`. A free MLP fusion model has no such guarantee.

3. **One shared cross-attention operator serves both polarities.** Positive and negative paths
   reuse the identical weights; only the downstream use (additive vs. rejected) differs. This
   directly answers the spec's "the model must learn ... how to dynamically weigh these inputs"
   for `T+` and `T−` with a single mechanism, rather than two independent sub-networks that
   might learn inconsistent notions of "relevance."

---

### 6. Connection to the Tier Progression (Report Framing)

| Tier | Constraint representation | Fusion mechanism | Negation operator | Trained? |
|---|---|---|---|---|
| **Tier-0** | single text embedding | vector addition | subtraction (wrong) | No |
| **Tier-1 CLAY** | stacked text prompts | one global SVD | none (polarity-blind) | No |
| **Tier-2a Track S** | per-polarity text prompts | separate SVD per polarity | subspace-norm penalty | No |
| **Tier-2c** | mined visual prototype | geodesic addition | subspace rejection (QR) | No |
| **Φ** | per-attribute text prompts (same source as Track S) | **learned cross-attention** (replaces SVD) | subspace rejection (QR, same as 2c) | **Yes** |

Φ is positioned as **Track S's natural successor**: same textual inputs (prompt-bank stacks per
attribute, never concatenated across polarity), same manifold-correct rejection mechanism as
Tier-2c, but the fixed SVD subspace-construction step is replaced by a learned, reference-
conditioned cross-attention. The ablation table compares Φ against **both** Track S (same
inputs, fixed fusion) and Tier-2c (same geometric rejection, different inputs) — isolating
exactly two deltas: *what changes when SVD becomes learned* (vs. Track S) and *what changes when
the constraint representation becomes textual instead of visual* (vs. Tier-2c).

**Note on Tier-2c / Track V.** The mined-visual-prototype design explored earlier in this
project remains a valid, fully training-free Tier-2a/2c contribution — it is kept and reported
on its own merits (no modality gap, no rotation needed, strong negation results). It is simply
not the basis for Φ, because the spec's fusion-module criteria are stated in terms of textual
conditioning and SVD replacement, which only Track S's input representation engages with
directly.

---

### 7. Reading Order Before Coding

| Paper | File | Sections | What you extract |
|---|---|---|---|
| GDE | `documents/papers/GDE.md` | §3.1–3.3, Prop. 1–2, App. A (Eq. 13–14) | Log/Exp closed forms, intrinsic mean, tangent mean direction |
| CLAY | `documents/papers/CLAY.md` | §3.2 | The exact SVD step Φ's cross-attention replaces |
| PoS-Subspaces | `documents/papers/PSGS_VLM.md` | §2.2, Eq. 5 | The `Π⊥` operator and the QR orthonormalisation argument |
| Alhamoud (Negation) | `documents/papers/VLM_DO_NOT_UNDERSTAND_NEGATIONSpdf.md` | §4.1, Fig. 6 | Why subtraction is wrong; what the correct semantics of "−X" is |
| Liang (Modality Gap) | SOTA.md §1 | abstract + §3 | Why the fixed centering step (Step 0) is needed for text inputs |
| InfoNCE | SOTA.md §7 (van den Oord 2018) | full | The contrastive loss; hard-negative intuition |
| `tier2a_S.py` / `S_plan.md` | `src/tier2a_S.py`, `documents/S_plan.md` | full | The exact prompt-bank input and SVD step Φ is built to outperform |

**Minimum to start coding:** GDE App. A (Log/Exp) + PoS-Subspaces §2.2 (the projection
operator) + Alhamoud §4.1 (negation semantics) + `tier2a_S.py` (the baseline Φ must beat using
the same inputs). The rest is needed before writing the report.

---

### 8. Definition of Done

Φ is complete when:

1. `src/fusion.py` exports `FusionPhi` with the forward pass in §2, consuming **prompt-bank
   text stacks** (`T_pos_stacks`, `T_neg_stacks`) as the literal positive/negative constraints —
   not mined visual prototypes.
2. `src/query_generator.py` generates synthetic train-split queries with the Hamming-≤2 GT
   protocol, positive sampling, and hard-negative sampling (target side stays images).
3. `src/train_phi.py` runs the InfoNCE training loop with checkpointing and learning curves.
4. `make_get_ranking` in `src/fusion.py` plugs Φ into the eval harness (CONTRACT §7):
   same `make_get_ranking(query_str, …) → get_ranking(src_idx)` signature.
5. **MEAN R@5 on the 14 benchmark queries strictly beats Track S (Tier-2a)** — the fixed-fusion
   baseline on the *same* textual inputs — isolating the gain attributable to learning.
6. Negation queries (`-Male, -Mustache`; `-Smiling, +Eyeglasses, +Wearing_Hat`) show
   R@5 > 0.000 — the concrete signal that the negation architecture is working.
7. Report sub-section "Learned cross-attention fusion over conditional text subspaces" is
   written with the forward pass equations, the loss, and two ablation rows: Φ vs. Track S
   (isolates the learned-SVD-replacement effect) and Φ vs. Tier-2c (isolates the
   text-vs-visual constraint representation effect).
