# Tier-3 Variations — Motivation, Design, and Expected Impact

**Date:** 2026-07-01  
**Author:** Alfonso Antognozzi  
**Status:** Implemented, pending VM training run

---

## 1. Background — Where We Stand After Tier-3 DGP (Φ)

### What we had

The project's training-based tier — `src/tier3_dgp.py` (formerly `fusion_dgp.py`) — is a
**Fusion-DGP Φ**: a learned cross-attention gate that replaces the closed-form softmax gate of
Tier-2d DGP. The architecture is:

```
Step 0 (fixed):   t̂_i = normalize(t_i − μ_txt)              # modality-gap centering (FIX-1)
Step 1 (LEARNED): h_ref = v_ref + MLP_ref(v_ref)             # zero-init residual → starts ≈ identity
                  c_a   = CrossAttn(h_ref, T̂_a, T̂_a)       # replaces Tier-2d's softmax gate
Step 2:           d_a   = normalize(c_a)                     # one direction per attribute
Step 3 (fixed):   q_tan = Log_μ(v_ref) + Σ_{a∈T+} α·d_a
                  for b∈T−: q_tan −= (q_tan·d_b) d_b         # orthogonal rejection
                  q     = normalize(Exp_μ(q_tan))             # back to S^{d-1}
```

~1.3M trainable parameters. CLIP fully frozen. Trained with multi-positive InfoNCE (τ=0.07),
AdamW lr=1e-4, 30 epochs, 64 queries/step on synthetic train-split queries.

### Results on the 14-query benchmark (test split)

| Method | P@1 | P@5 | P@10 | R@1 | R@5 | R@10 |
|---|---|---|---|---|---|---|
| Tier-0 enhanced (best training-free baseline) | 0.0393 | 0.0265 | 0.0224 | 0.0393 | 0.1073 | 0.1607 |
| Tier-2d DGP (closed-form gate) | 0.0252 | 0.0186 | 0.0155 | 0.0252 | 0.0731 | 0.1172 |
| **Tier-3 DGP Φ (InfoNCE, cross-attention)** | **0.0383** | **0.0281** | **0.0229** | **0.0383** | **0.1110** | **0.1607** |

Φ is the best result so far: +52% P@1 over Tier-2d DGP, and essentially tied with the
enhanced Tier-0 on R@10. It definitively proves the learned gate is better than the
closed-form gate.

### Why it is not yet decisively better than Tier-0 enhanced

Two structural bottlenecks remain:

1. **The loss does not directly optimise the ranking metrics.** InfoNCE is a mutual
   information surrogate — it pushes query embeddings toward positives and away from hard
   negatives in the inner-product sense, but P@K and R@K are rank-order quantities. A model
   that gets the ordering of 8 positives vs 16 hard negatives correct will score a low InfoNCE
   loss even if it fails to rank those positives above thousands of other test images.

2. **Negation is still a fixed hand-coded operator.** The orthogonal rejection step is
   inherited verbatim from Tier-2d DGP. Nothing in the cross-attention head or the loss
   explicitly trains the model to suppress negative attributes. The `-Male, -Mustache` query
   (and other negation-heavy queries) scores 0.000 across every tier, including Φ — the learned
   gate provides no signal for negation because negation happens after the gate, in a fixed
   geometric operation.

3. **The composition is still latent arithmetic.** The query is assembled as
   `q_tan = Log_μ(v_ref) + Σ α·d_a − rejection`, an explicit formula. The model can only
   improve the individual directions `d_a`; it cannot learn a fundamentally different way of
   combining reference and attributes.

4. **CLIP's image encoder is frozen.** All tiers compute `v_ref` from the same frozen
   ViT-B/32 encoder. The modality gap and the geometry of the embedding space are fixed; the
   model can only operate on top of a space that was not designed for compositional retrieval.

---

## 2. The Four Variations — Why and What

### 2.1 `tier3_contrastive.py` — Replace InfoNCE with a Ranking Loss

**Why:** InfoNCE is a self-supervised contrastive loss derived from the CLIP training paradigm.
It encourages the query to be close to its positives and far from a small set of hard negatives,
but it does not model the full ranking list. ListNet (Cao et al. 2007) directly optimises the
cross-entropy between the ideal permutation probability distribution and the model-scored one:

```
L = − Σ_i  y_i · log softmax_i( q·x_i / τ )
```

where `y_i = 1/|pos|` for positives and `0` for negatives. The gradient of ListNet flows through
the complete ranked list — including all `n_hard_neg=32` negatives — whereas InfoNCE's denominator
sees only 8 positives + 16 negatives in a log-sum-exp. ListNet is more directly metric-consistent
with P@K.

**What:** Identical `FusionPhi` architecture (same cross-attention gate, same manifold geometry,
same `QueryGenerator`). The only change is the loss function: `info_nce()` → `listnet_loss()`.
This is a pure ablation of the training objective with the architecture held constant.

**Expected gain:** Modest but consistent improvement across all metrics, especially on
multi-attribute queries where ranking the full list matters more than pair-wise push/pull. P@1
should improve the most because ListNet explicitly optimises the top of the list.

---

### 2.2 `tier3_negation.py` — Learned Negation Head

**Why:** Every tier negates by orthogonal rejection:

```
q_tan ← q_tan − (q_tan · d̂_b) · d̂_b
```

This is the principled geometric operator on the sphere (Oldfield et al. 2023, Alhamoud et al.
2025), but it has two hard assumptions baked in:

1. **The suppression coefficient is always exactly 1.0** — remove exactly the projection of
   `q_tan` onto `d_b`, no more, no less.
2. **The direction `d_b` is the right subspace to suppress** — the attribute direction for
   negation should be the same direction used for affirmation.

Both assumptions may be wrong in practice. An attribute like `-Male` may require suppressing
a different subspace than the one that `+Male` points toward, and the strength of suppression
may depend on how strongly the current query is already oriented toward that attribute.

**What:** A `NegationHead` module — a small MLP that takes `(q_tan, d_b)` as input and
outputs a learned coefficient `λ ∈ (0, 2)` (sigmoid-scaled, zero-init → λ=1.0 at init,
recovering orthogonal rejection as the starting point):

```
λ = 2 · sigmoid( MLP( [q_tan ; d_b ; q_tan⊙d_b ; q_tan·d_b] ) )
q_tan ← q_tan − λ · d_b
```

At λ=1 this is exact orthogonal rejection. λ>1 over-suppresses (useful when the attribute
is faint in `q_tan`). λ<1 under-suppresses (useful when the query should retain some of the
attribute). The affirmation path is unchanged (same cross-attention gate as `tier3_dgp`).

**Expected gain:** Primary target is negation-heavy queries (`-Male, -Mustache`, `-Young`,
`-Heavy_Makeup`). The `-Male, -Mustache` query scores 0.000 on every prior method — a learned
suppression coefficient may be what is needed to break that floor. Secondary gain: the λ signal
is conditioned on the current `q_tan`, so the model can learn context-dependent negation strength
(e.g. negate more strongly when the attribute is strongly present in the reference image).

---

### 2.3 `tier3_composer.py` — Full Learned Composer (Replace Latent Arithmetic)

**Why:** Every prior method assembles the query as explicit latent arithmetic on top of a
pre-defined geometric skeleton (log-map, geodesic addition, orthogonal rejection, exp-map).
The skeleton is principled but rigid — it forces the composition to follow a specific additive
structure that may not match the actual geometry of multi-attribute compositional retrieval in
CLIP's embedding space.

A small transformer encoder has no such structural constraint. Given the sequence of tokens
`[CLS, v_ref, t⁺₁, …, t⁺ₙ, t⁻₁, …, t⁻ₘ]` distinguished by learned polarity embeddings, the
encoder can learn any interaction pattern between the reference and the attribute directions —
including asymmetric affirmation/negation, cross-attribute interactions, and compositions that
the log/exp arithmetic cannot express.

**What:** `QueryComposer` — a 2-layer transformer encoder (`d=512, 8 heads, FFN=1024`) that
takes the token sequence, pools the CLS output, and adds a learned residual to `v_ref`:

```
tokens = [CLS, v_ref + pol(0), mean(T̂_a) + pol(+1), …, mean(T̂_b) + pol(−1), …]
out    = TransformerEncoder(tokens)            # [seq_len, 512]
q      = normalize( v_ref + MLP(out[CLS]) )   # residual: start from v_ref, adjust
```

Polarity embeddings (0=reference, 1=positive, 2=negative) are the only structural signal; the
model learns the rest. The residual connection ensures the untrained Composer starts close to
the identity (q ≈ v_ref) and the training signal bootstraps from there. No log/exp maps, no
rejection — completely data-driven composition. CLIP still frozen.

**Expected gain:** This is the architectural leap. If the latent arithmetic structure of prior
tiers is a binding constraint, the Composer should outperform all of them on multi-attribute
queries where composition is genuinely non-additive. The risk is that 30 epochs of training on
synthetic queries may not be enough to fully train a transformer from scratch — the Composer
may need more epochs or more data than the DGP-based tiers. Watch the training loss curve: if
it is still decreasing sharply at epoch 30, extend to 60 epochs.

---

## 4. How to Run on the VM

SSH in, activate the environment, open a tmux session, then run each in its own window. Run
them in the order below — cheaper first, so you have intermediate results early:

```bash
# Window 1
python src/tier3_contrastive.py

# Window 2 (start after window 1 is past epoch 5 to avoid GPU memory spikes)
python src/tier3_negation.py

# Window 3
python src/tier3_composer.py

```

Each script writes its CSV to `output/tier3_*/` and its weights to the same folder. Pull
results back with:

```bash
scp -P 5040 -r disi@lab-d2fdded3-9ba4-4188-80fa-16c27a4cd022.westeurope.cloudapp.azure.com:~/Deep-Learning-Project/output ./
```

---

## 5. Expected Outcome Ranking

From most to least likely to beat the current best (Tier-3 DGP Φ P@1=0.0383):

1. **tier3_composer** — removes the arithmetic constraint; may need more epochs but architecture
   is fundamentally more expressive.
2. **tier3_contrastive** — clean ablation; ListNet's metric-consistent gradient should give a
   consistent improvement over InfoNCE with no architectural cost.
3. **tier3_negation** — targeted fix for negation queries; will definitively answer whether
   orthogonal rejection is the bottleneck for the zero-scoring negation queries.

Any tier that beats **P@1=0.0393** (Tier-0 enhanced) will be the headline result. A tier that
breaks **P@1=0.05** would be a strong result for the project.
