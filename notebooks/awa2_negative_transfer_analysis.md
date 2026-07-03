# AwA2 Transfer Study — Why the CelebA Champion Loses to the Zero-Shot Baseline

**Status:** diagnosed and explained (negative-transfer result, intended for the final report)
**Artifacts:** `output/awa2/*.csv`, `output/awa2/tier5_diag_Aprime_lam*.csv`, `output/awa2/tier5_diag_log.txt`

---

## 1. The problem

We evaluated three methods on AwA2 (Animals with Attributes 2, spec §2) using the same
protocol as the CelebA benchmark: a frozen CLIP ViT-B/32 visual DB, a 12-query ±attribute
benchmark generated with the spec §3.1.1 relaxed-Hamming rule (strict ± satisfaction,
Hamming ≤ 2 on free attributes, ≥ 5 valid targets, 60 sources per query), and
Recall/Precision@{1,5,10}.

The ranking **inverted** relative to CelebA:

| Mean R@1 | CLIP zero-shot baseline | CLAY | tier5 `Bprime_listnet` (trained) |
|---|---|---|---|
| **CelebA** (13,3k-img test corpus) | 0.0397 | 0.0098 | **0.1104** ← champion |
| **AwA2** (7,468-img test corpus) | **0.8083** ← best | 0.7306 | 0.6667 ← worst |

On CelebA the learned Bayes fusion beats the baseline by **2.8×**; on AwA2 it loses to it
by **14 points**. Same architecture, same training recipe (BayesPhi + ListNet on the
Stage A′ core, seed 0), verified from the training logs — so the question was: bug,
training failure, or a real property of the method?

> Note on absolute scales: CelebA and AwA2 numbers are not comparable currencies. AwA2 is
> near-ceiling for *every* method because its ground truth reduces to nearest-class
> retrieval (see §3); CelebA's per-image rule is far harder. The signal is the ordering
> flip, not the magnitudes.

## 2. Ruling out the boring explanations

- **Wrong model?** No. The AwA2 run is the same `Bprime_listnet` arm as
  `output/tier5_cdbayes/tier5_Bprime_listnet_seed0_full.csv`: Stage-0 probes → C1
  temperature calibration → C2 Chow-Liu tree → BayesPhi (zero-init heads, κ=1) trained
  with ListNet on the Stage A′ core. Confirmed line-by-line in
  `output/awa2/tier5_train_log.txt`.
- **Training failure?** No — the diagnostic below shows training *improved* on its own
  base (+6 pts R@1). The loss decreased smoothly and κ stayed ≈ 0.99 (the init invariant
  held).
- **Broken ground truth?** No — the GT was validated semantically (e.g. `+stripes`
  targets are exactly the raccoon/skunk/tiger/zebra classes) and structurally
  (schema-identical to `celeba_evaluation.json`, consumed by the same `eval.py`).

## 3. The structural difference that matters

**CelebA attributes are per-image; AwA2 attributes are class-level.** Every image of a
class shares the class's 85-bit row of `predicate-matrix-binary.txt`. Two consequences:

1. **The ground-truth rule collapses to class retrieval.** "Strictly satisfy ± constraints
   and stay within Hamming ≤ 2 on free attributes" becomes a relation between *classes*;
   valid targets are all images of a handful of classes near the source's class. Raw CLIP
   cosine similarity ≈ class identity on animal photos, so the zero-shot baseline's query
   (`v_ref + α·delta`) is already close to the ideal scorer. There is no per-image
   attribute combinatorics left for a fusion module to exploit.

2. **Tier5's probe channel becomes a noise source.** The Bayes score routes through 85
   per-attribute posteriors and an exact Poisson-binomial Hamming tail over ~80 free
   attributes. On AwA2 the probes reach only **86.4% mean accuracy (min 75.0%,
   `spots`)** — many predicates are not visually groundable from a single photo
   (`smart`, `timid`, `fast`, `agility`, `newworld`, `oldworld`, ...), so their posteriors
   are systematic guesses. Errors compound multiplicatively across the tail DP. On CelebA
   the same probes hit ~88.5% on 40 *visual* attributes — and, crucially, there the CLIP
   cosine alternative was **adversarial** to the task (the `-Male,-Mustache` "female twin"
   failure: 0.000 R@K for every geometry tier), so noisy-but-rule-aware beat clean-but-wrong.

In short: **tier5's entire advantage is machinery for a problem AwA2 does not have, paid
for with a probe-noise cost AwA2 does not repay.**

## 4. The diagnostic experiment (mechanism evidence)

To separate "the training failed" from "the core is the ceiling" from "the identity
channel is missing", we ran a **training-free λ-sweep** of the Stage A′ closed-form score
(`python src/awa2/tier5_fusion.py --diagnose`):

```
S(x | ref) = constraint(x) + tail(x, ref) + λ · cos(v_x, v_ref)
```

λ = 0 is the exact base the trained Φ sat on; λ > 0 mixes in the raw CLIP identity
channel (the C4 mechanism of tier5). Results (mean over the 12 queries):

| λ | R@1 | R@5 |
|---|---|---|
| 0 (Bayes core alone) | 0.6069 | 0.7903 |
| 0.25 | 0.6167 | 0.8000 |
| 0.5 | 0.6306 | 0.8042 |
| 1 | 0.6458 | 0.8111 |
| 2 | 0.6653 | 0.8306 |
| 4 | 0.7097 | 0.8569 |
| 8 | 0.7653 | 0.8917 |
| *trained Φ (reference)* | *0.6667* | *0.9014* |
| *CLAY (reference)* | *0.7306* | *0.9111* |
| *CLIP baseline (reference)* | *0.8083* | *0.9361* |

Two findings:

1. **Training is not the problem.** The core alone scores 0.607; the trained Φ reached
   0.667. The 40-epoch ListNet run *worked* — it recovered ~6 points over its base
   (partly by learning a CLIP channel through the zero-initialized `q_v` head). The
   deficit vs the baseline lives in the core itself.

2. **The missing ingredient is exactly CLIP identity.** R@1 rises **monotonically** with
   λ — every single query improves, including the core's worst case (`-furry`,
   0.467 → 0.600) — converging toward the zero-shot baseline as the score trusts raw
   cosine over the probe channels. The more "tier5" the score, the worse it does; the
   more "tier0", the better.

> **Honesty caveat:** λ was swept directly on the test benchmark, so this is an *oracle
> diagnostic* that explains the mechanism. Cite the monotonic shape of the curve as
> evidence; do **not** report the best-λ point as a tuned competing result.

## 5. Takeaway for the report

The learned Bayes fusion **transfers in mechanism but not in advantage**. Its inductive
bias — score the ground-truth rule through per-attribute probe posteriors plus a
Hamming-tail identity term — is precisely matched to benchmarks where (a) attributes are
annotated **per image**, (b) the attributes are **visually groundable**, and (c) plain
CLIP cosine is **adversarial** to the compositional rule. CelebA satisfies all three;
AwA2 violates all three (class-level annotation, partly non-visual predicates, cosine ≈
the task). The result is a clean negative-transfer case: on AwA2 the method's probe
machinery only adds noise on top of a signal the zero-shot baseline already captures.

This also predicts *when* to prefer which method: per-image fine-grained attribute
retrieval → learned fusion (Φ); category-level retrieval where identity ≈ class →
zero-shot latent arithmetic, with CLAY in between.

## 6. Reproduction

```bash
# ground truth + benchmark (needs the raw dataset locally)
python src/awa2/generate_gt.py

# the three evaluations (need artifacts/awa2/ features; training on VM)
python src/awa2/tier0_baseline.py
python src/awa2/tier_clay.py
python src/awa2/tier5_fusion.py            # trains Φ (VM, ~40 min GPU)

# the diagnostic λ-sweep (training-free, ~seconds on GPU / minutes on CPU)
python src/awa2/tier5_fusion.py --diagnose

# statistical comparison (bootstrap CIs + Holm-corrected paired tests)
python src/awa2/compare.py
```
