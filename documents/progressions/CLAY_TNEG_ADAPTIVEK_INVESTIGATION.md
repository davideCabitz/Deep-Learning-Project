# CLAY T_neg Suppression: Adaptive-k Investigation

**Status:** Completed — negative result, approach retired  
**Dataset:** CelebA test split, 19,962 images, 40 binary attributes  
**Backbone:** CLIP ViT-B/32 (512-d embeddings)  
**Date:** 2026-06-29

---

## 1. Background and Motivation

### The confirmed failure mode

CLAY (Lim et al.) builds a manifold-aware textual subspace from condition prompts and measures reference↔DB similarity inside that subspace. For a query with positive condition T_pos and negative condition T_neg, it stacks all condition prompts naïvely into one matrix before SVD:

```
T_c = [ prompts(T_pos) ; prompts(T_neg) ]   ∈ R^{m × d}
μ_c = normalize( mean(T_c) )
L   = log_{μ_c}(T_c)                         # manifold log map, [m, d]
L   = U S Vᵀ  (thin SVD)
V_k = top-k cols of V                         # [d, k], the subspace basis
score(v_ref, v_db | c) = cos( V_kᵀ log_{μ_c}(H·v_ref),  V_kᵀ log_{μ_c}(H·v_db) )
```

where H is a minimal rotation closing the modality gap (CLAY §3.2). The paper uses k=50 and evaluates on synthetically disentangled attribute pairs where the positive and negative attributes are statistically independent by construction.

A three-batch diagnostic (see `CLAY_TNEG_DIAGNOSTIC_REPORT.md`) ran CLAY on a real 19,962-image CelebA split with **real-world correlated attribute pairs**. The finding was unambiguous: across 8 correlated pairs (φ = +0.379 to +0.761), T_neg **never significantly suppressed** the unwanted attribute, and in the highest-correlation case (Wearing_Lipstick / Heavy_Makeup, φ=+0.761) adding T_neg **increased** leakage by ΔR-Neg@10=+0.095 (p=0.003). Two low-correlation control pairs worked correctly (Young/Smiling: ΔR-Neg=−0.085, p=0.008; Mouth_Slightly_Open/Wavy_Hair: ΔR-Neg=−0.110, p=0.040), ruling out a wiring bug.

Additionally, a secondary finding from the diagnostic: the subspace overlap metric ‖V_posᵀ · V_{pos+neg}‖_F / k sat in a suspiciously tight band (0.1047–0.1153) across all 8 correlated pairs, with no relationship to φ strength or which specific attributes were involved.

### The hypothesis being tested

The uniform low overlap and the uniformly ineffective suppression suggested a specific geometric story: **k=50 might be too large for the actual intrinsic dimensionality of the attribute subspaces**, causing the subspace to include many noise directions that dilute the suppression signal.

Concretely: the prompt bank contains 60 prompts per attribute (12 sentence frames × 5 synonym phrases). After the log map, the tangent-space matrix L is [60, 512]. A thin SVD gives at most 60 singular values. With k=50, CLAY retains **83% of all available right singular vectors** — discarding only the bottom 10. If an attribute's prompts have low intrinsic semantic diversity in CLIP's text space (i.e. they all encode nearly the same visual concept with minor paraphrase variation), most of those 60 tangent vectors lie in a low-dimensional subspace. The singular values would drop sharply after the first few components, and k=50 would be including many noise directions with near-zero singular values.

**Prediction:** Failing-to-suppress attributes (Mouth_Slightly_Open, Heavy_Makeup, Wearing_Lipstick, High_Cheekbones, Attractive) would show low intrinsic rank — a sharp elbow in the singular value spectrum with k*(τ=0.90) ≪ 50. Working attributes (Smiling, Wavy_Hair) would show higher intrinsic rank with a more gradual decay. An adaptive-k rule:

```
k*(τ) = min { k : Σ_{i=1}^{k} S_i² / Σ_{i=1}^{n} S_i² ≥ τ }
```

applied to the combined prompt stack before extracting V_k would then select a more precise subspace for narrow-concept attributes, recovering suppression precision without requiring any training.

**For the reversal case specifically:** with φ=+0.761 (Wearing_Lipstick / Heavy_Makeup), both prompt stacks live in nearly the same region of CLIP text space. The combined [120-row] stack SVD at k=50 might capture directions that are *more central to the shared cosmetic cluster* than V_pos alone — pulling Heavy_Makeup-positive images closer to the query in the merged subspace. Adaptive k* on the combined stack would reduce to the few dominant shared directions, which are already well-represented in V_pos alone, and the marginal Heavy_Makeup contribution should shrink toward zero — attenuating the reversal.

---

## 2. Spectrum Inspection

### Method

For each of the 7 target attributes, the SVD step from `_build_subspace` was reproduced on that attribute's prompt stack alone (T_c = prompts(attr) only, padding stripped), retaining S instead of discarding it:

```python
T_c  = prompt_bank[j, :n_j]               # [n_j, 512], n_j=60 for all attrs
μ_c  = normalize(mean(T_c))
L    = log_{μ_c}(T_c)                     # [60, 512]
U, S, Vᵀ = svd(L, full_matrices=False)    # S ∈ R^{60}
```

k*(τ) was computed as the smallest k satisfying cumulative explained variance ≥ τ, for τ ∈ {0.80, 0.85, 0.90, 0.95, 0.99}.

### Results

| Attribute | Group | k*(0.85) | k*(0.90) | k*(0.95) | S[0]/S[1] |
|---|---|---|---|---|---|
| Mouth_Slightly_Open | FAILING | 9 | 11 | 14 | 1.05 |
| Heavy_Makeup | FAILING | 9 | 11 | 14 | 1.20 |
| Wearing_Lipstick | FAILING | 10 | 11 | 13 | 1.13 |
| High_Cheekbones | FAILING | 8 | 10 | 13 | 1.46 |
| Attractive | FAILING | 10 | 12 | 16 | 1.20 |
| Smiling | WORKING | 10 | 12 | 17 | 1.31 |
| Wavy_Hair | WORKING | 7 | 9 | 13 | 1.20 |

All seven attributes have near-identical spectrum shapes: flat tops (S[0]/S[1] ratios 1.05–1.46, no single dominant direction), gradual decay through the first 10–15 components, then rapid dropoff. Every attribute reaches 90% explained variance in 9–12 directions.

**The hypothesis is falsified.** Failing attributes do not have smaller k* than working ones. Wavy_Hair — the attribute with the *strongest* suppression result in the entire diagnostic (ΔR-Neg=−0.110, p=0.040) — has k*(0.90)=**9**, the smallest of all seven. Smiling (also working) has k*(0.90)=12, matching the failing attributes exactly. There is no spectral property that distinguishes the failing from the working group.

### What the uniform overlap now means

With k*(0.90) ≈ 9–12 for every attribute, but k=50 being applied uniformly, the first ~10 singular vectors of any attribute's prompt stack are nearly the same regardless of attribute content — they represent the dominant modes of variation introduced by the sentence frame structure (12 frames × 5 synonyms) in CLIP's text space. The overlap metric ‖V_posᵀ · V_{pos+neg}‖_F / k ≈ 0.11 is therefore a structural property of this prompt bank and CLIP's text geometry, not an attribute-specific signal. It is near-constant because it is measuring shared frame-structure variance, not semantic attribute variance.

---

## 3. Adaptive-k Experiment

### Setup

A new scorer (`tier1_CLAY_adaptivek.py`) was built as a non-destructive extension: it imports all geometry helpers from `tier1_CLAY.py` and replaces only the subspace construction to capture and use S. The diagnostic runner (`diagnostic_adaptivek.py`) ran both scorers on the same 8+5 pairs in a single pass for direct comparison.

```python
# Adaptive rule applied to the combined prompt stack T_c = [prompts(T_pos); prompts(T_neg)]
U, S, Vᵀ = svd(log_{μ_c}(T_c), full_matrices=False)
k* = min { k : Σ_{i<k} S_i² / Σ S_i² ≥ τ }
k_eff = min(k*, k_cap=50)
V_k = Vᵀ[:k_eff].T
```

τ=0.90, k_cap=50. Because the attribute spectra are indistinguishable, the combined stacks for two-condition queries (T_pos + T_neg) produce k_eff ≈ 11–16 across all pairs.

### Batch 1 results — correlated pairs

| Pair | φ | k_eff(+) | k_eff(+/−) | ΔR-Neg Fixed | ΔR-Neg Adaptive | Change |
|---|---|---|---|---|---|---|
| Wearing_Lipstick / Heavy_Makeup | +0.761 | 11 | 14 | **+0.095** | +0.055 | IMPROVED |
| Smiling / High_Cheekbones | +0.677 | 12 | 13 | +0.030 | +0.030 | NEUTRAL |
| Smiling / Mouth_Slightly_Open | +0.540 | 12 | 15 | −0.010 | −0.025 | NEUTRAL |
| No_Beard / Wearing_Lipstick | +0.431 | 12 | 15 | −0.015 | −0.040 | IMPROVED |
| Wearing_Lipstick / Attractive | +0.475 | 11 | 15 | +0.035 | **+0.105** | WORSE |
| Young / Attractive | +0.379 | 12 | 16 | +0.000 | +0.005 | NEUTRAL |
| Attractive / Heavy_Makeup | +0.467 | 12 | 16 | −0.035 | −0.040 | NEUTRAL |
| Mouth_Slightly_Open / High_Cheekbones | +0.417 | 11 | 12 | −0.005 | **+0.055** | WORSE |

The reversal (Wearing_Lipstick / Heavy_Makeup) attenuated from +0.095 to +0.055 — Criterion 1 technically passes — but the mechanism is accidental: reducing k from 50 to 14 weakens every directional signal in the subspace equally, including both the correct positive-attribute signal and the spurious Heavy_Makeup contamination. Two pairs got worse (Wearing_Lipstick/Attractive: +0.035 → +0.105; Mouth_Slightly_Open/High_Cheekbones: −0.005 → +0.055). No pair crossed into SUPPRESSION_WORKING.

### Batch 2 results — control pairs (critical)

| Pair | φ | Fixed verdict | Adaptive verdict | ΔR-Neg Fixed | ΔR-Neg Adaptive |
|---|---|---|---|---|---|
| Young / Smiling | −0.015 | **SUPPRESSION_WORKING** | SUPPRESSION_INEFFECTIVE | −0.085 | −0.060 |
| Young / Mouth_Slightly_Open | −0.006 | INEFFECTIVE | INEFFECTIVE | +0.050 | +0.015 |
| Attractive / Mouth_Slightly_Open | +0.002 | INEFFECTIVE | INEFFECTIVE | +0.025 | +0.030 |
| Young / High_Cheekbones | −0.013 | INEFFECTIVE | INEFFECTIVE | −0.010 | −0.065 |
| Mouth_Slightly_Open / Wavy_Hair | +0.015 | **SUPPRESSION_WORKING** | SUPPRESSION_INEFFECTIVE | −0.110 | −0.130 |

Both previously-working control pairs dropped to INEFFECTIVE under adaptive-k. Young/Smiling: ΔR-Neg −0.085 → −0.060, p=0.008 → p=0.248 (loses significance). Mouth_Slightly_Open/Wavy_Hair: ΔR-Neg −0.110 → −0.130 (the delta improves numerically, but p=0.040 → p=0.107, loses significance). The suppression signal is present in the right direction but the reduction in k throws away enough between-query variance to destroy the statistical power.

### Grand summary

| Category | Fixed-k | Adaptive-k (τ=0.90) |
|---|---|---|
| SUPPRESSION_WORKING | 2 | **0** |
| SUPPRESSION_INEFFECTIVE | 11 | 14 |
| SUPPRESSION_REVERSED | 1 | 0 |

Criterion 1 (reversal attenuated): **PASS** (mechanistically uncontrolled)  
Criterion 2 (working pairs preserved): **FAIL** — both drop to INEFFECTIVE

The adaptive-k scorer turns the 1 REVERSED result into INEFFECTIVE (a small gain) but simultaneously turns the 2 WORKING results into INEFFECTIVE (a clear regression). Net outcome: strictly worse than fixed-k on the only metric that matters — SUPPRESSION_WORKING count.

---

## 4. Why This Approach Failed: Mechanistic Account

### The core problem adaptive-k cannot solve

The diagnostic data and the spectrum analysis together point to the actual mechanism. When T_pos and T_neg are correlated (high φ), their prompt embeddings occupy overlapping regions of CLIP's text space. The combined stack T_c = [prompts(T_pos); prompts(T_neg)] therefore has a variance structure dominated by the **shared** cluster, not by the directions that distinguish the two attributes. The SVD of this combined stack will produce singular vectors that point along the shared cosmetic/facial cluster — regardless of how many (k) you retain. Keeping k=50 or k=12 makes no fundamental difference: the *first* singular vector already points at the shared cluster, and every subsequent one is also contaminated by the correlation.

Formally: let μ̂_pos and μ̂_neg be the normalized means of the two prompt stacks in tangent space. The angle between them is:

```
cos(θ) ≈ 1 − φ·correction   (for high φ, θ is small)
```

When θ is small, the mean of the combined stack μ̂_c lies between them, and the log-map of both stacks to μ̂_c projects both onto nearly the same tangent directions. The SVD of the combined L then finds singular vectors that span the union of both prompt clouds — but since the clouds nearly coincide, V_k is essentially the same as V_k computed from T_pos alone. This is precisely what the overlap metric measured: ‖V_posᵀ · V_{pos+neg}‖_F / k ≈ 0.11 uniformly, reflecting that adding T_neg prompts barely shifts the subspace because T_neg lives in the same text-space neighborhood as T_pos.

Reducing k does not change this geometry. Whether you take k=50 or k=12 of the combined SVD, the basis vectors you extract are all drawn from the shared-cluster directions. T_neg has contributed no unique basis direction that the suppression mechanism can act on.

### Why the spectrum hypothesis was wrong

The spectrum hypothesis assumed that failing attributes have lower intrinsic semantic dimensionality than working ones, causing k=50 to include irrelevant noise. The data show all seven attributes have nearly identical spectra (k*(0.90) = 9–12 across both failing and working groups). The working attribute Wavy_Hair has k*(0.90)=9 — lower than any failing attribute. The failure therefore has nothing to do with intrinsic attribute rank. The relevant variable is **inter-attribute correlation**, not intra-attribute dimensionality.

### Why the working control pairs lost significance under adaptive-k

When k is reduced from 50 to ~12 on the combined stack, the projected DB coordinates D = normalize(V_kᵀ log_{μ_c}(H·v)) become lower-dimensional. The reduced projection retains the dominant T_pos-cluster direction but the per-image variation in the T_neg-relevant directions shrinks. For working pairs (low φ, T_neg genuinely differs from T_pos in text space), the few T_neg-unique directions that carry the suppression signal are exactly the lower-ranked singular vectors — the ones that adaptive-k discards first. The result is that the raw score differences between T_neg-positive and T_neg-negative DB images shrink, reducing statistical power at N=20, even when the direction of the effect is correct (Mouth_Slightly_Open/Wavy_Hair: δ improved from −0.110 to −0.130 in magnitude but lost significance because the per-query variance also increased with the lower-dimensional projection).

---

## 5. What the Data Point Toward Instead

The combined evidence — spectrum uniformity, subspace overlap uniformity, adaptive-k failure — narrows the problem to a single architectural issue in CLAY:

**CLAY uses one SVD over the joint prompt stack. For correlated attributes, this SVD cannot separate the T_pos and T_neg contributions because they project onto the same tangent directions.**

The necessary fix requires computing T_pos and T_neg subspaces *separately*, and then explicitly orthogonalizing the T_neg basis relative to V_pos before using it for suppression. The idea:

```
V_pos = top-k right singular vectors of svd( log_{μ_c}(prompts(T_pos)) )
T_neg_residual = prompts(T_neg) - V_pos V_posᵀ prompts(T_neg)   # project out T_pos component
V_neg = top-k' right singular vectors of svd( log_{μ_c}(T_neg_residual) )
```

V_neg built this way is by construction orthogonal to V_pos and captures only the variation in T_neg's prompts that is *not already explained by the positive condition*. For a correlated pair with φ ≈ 0.76, most of the T_neg prompt variance lies in V_pos's span — the residual will be small but will point at whatever is genuinely distinct about T_neg. For an independent pair (φ ≈ 0), the residual is nearly unchanged from the original T_neg stack, preserving the working cases.

This is the direction of `tier2a_S` in the project track (Track S, separate positive/negative subspaces), and the diagnostic data confirm it is the right level at which to attack the problem. The adaptive-k experiment, by showing that a *single-SVD* approach cannot be fixed by tuning k, strengthens the case for Track S's architectural separation.

---

## 6. Files Created and Retired

The following files were created for this investigation and are now retired (deleted or to be deleted). They are documented here for reproducibility.

| File | Role | Status |
|---|---|---|
| `src/spectrum_inspect.py` | Standalone SVD spectrum analysis — prints S and k*(τ) for target attributes, no retrieval | Retired |
| `src/tier1_CLAY_adaptivek.py` | Adaptive-k scorer — imports helpers from tier1_CLAY, replaces _build_subspace to capture S, exposes tau parameter | Retired |
| `src/diagnostic_adaptivek.py` | Three-batch diagnostic runner using the adaptive-k scorer with side-by-side comparison output | Retired |

The benchmark output CSVs (`output/tier1_CLAY_adaptivek/`) document the parity check (tau=None, matching fixed-k) and the tau=0.90 run with the degraded MEAN R@10 (0.0223 vs 0.0541 for fixed-k).

`tier1_CLAY.py`, `manifold.py`, and `diagnostic_correlation.py` were not modified.
