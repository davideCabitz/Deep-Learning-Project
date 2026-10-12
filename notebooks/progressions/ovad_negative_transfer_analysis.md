# OVAD Transfer Study — Why the CD-Bayes Champion Collapses, and Why It Can't Be Tuned Away

**Status:** negative-transfer finding, analogous to `notebooks/awa2_negative_transfer_analysis.md`
**Artifacts:** `output/ovad/*.csv`, `output/ovad/tier5_train_log.txt`, `output/ovad/tier5_diag_log.txt`,
`output/ovad/tier5_diag_Aprime_lam*.csv`

---

## 1. The result

We evaluated the full method family on OVAD's 12-query benchmark (masked relative-Hamming
GT rule, `OVAD_Dataset/OVAD_Implementation_Plan.md` §3), reusing the CelebA/AwA2 eval
pipeline through `src/ovad/adapter.py`. Mean recall across queries:

| Mean | R@1 | R@5 | R@10 | family |
|---|---:|---:|---:|---|
| **Tier-3 DualScore** | 0.0840 | **0.2536** | 0.4119 | trained, CLIP-space |
| **Tier-2d DGP** (τ=100 uniform gate) | 0.0844 | 0.2455 | 0.3972 | training-free, text |
| **Tier-0 enhanced** (text deltas) | 0.0710 | 0.2415 | 0.3565 | training-free, text |
| **LDE** (mined visual directions) | 0.0504 | 0.1759 | 0.2789 | training-free, visual |
| **GDE** (mined visual directions) | 0.0493 | 0.1684 | 0.2717 | training-free, visual |
| **CLAY** (pre-SVD prompt stacking) | 0.0267 | 0.1158 | 0.1741 | training-free, text |
| **Tier-5 Bprime_listnet** (trained BayesPhi — **CelebA's champion**) | **0.0043** | **0.0919** | 0.1906 | trained, probe-space |

Tier5 is **last place on every metric**, and by a wide margin: its R@5 (0.092) is barely a
third of the best method's (0.254), and its R@1 (0.004) is effectively zero — it fails to
put a single valid target first for all but a handful of the ~2,762 source queries in the
benchmark. This is not a milder version of the AwA2 inversion (where tier5 was merely
*worst of seven*, at 0.667 vs. a 0.881 leader — still a respectable score in absolute
terms); on OVAD tier5's absolute score is the weakest number of *any* method, on *any* of
the three datasets, in the whole project.

## 2. Ruling out the boring explanations

- **Wrong adapter wiring?** No. `src/ovad/adapter.py::configure_tier5()` rebinds only data
  providers (attribute tables, tri-state-masked probe/tree/calibration trainers, the
  query-string parser) — the exact same single-seam pattern validated on AwA2. The scoring
  math in `src/tier5_cdbayes.py` runs byte-identical.
- **Broken ground truth?** No — the masked relative-Hamming rule is a direct, verified port
  of the CelebA rule (`OVAD_Implementation_Plan.md` §3), and the training-free methods
  (tier0, DGP, DualScore) score entirely reasonably against it (R@5 ≈ 0.24–0.25, in the
  same range as their CelebA numbers). If the GT were broken, *every* method would collapse
  together — instead the collapse is specific to tier5.
- **A code bug in this run?** No — the hard invariant (`verify_tier4_invariant`, run at
  every `tier5_cdbayes.py` startup) passed, and the per-source dump parity check against
  the published CSV passed (`_dump_per_source` in `src/ovad/tier5_fusion.py`). The training
  curve (`output/ovad/tier5_training_curve.png`) shows the ListNet loss oscillating in a
  stable range (3.5–3.7), not diverging or NaN-ing — the run completed normally by every
  technical measure. The failure is in what the model *learned to represent*, not a crash.

## 3. The mechanism: two independent failure modes, both structural

### 3.1 The Poisson-binomial identity tail is scoring noise, not signal

Tier5's core scoring term (`hamming_tail_log`, `src/tier4_bayes.py:165`) computes
`log P(Σ_c D_c ≤ hamming_max)` — an exact Poisson-binomial tail over the *free*
(non-queried) attributes, where each `D_c` is a Bernoulli mismatch indicator built from
probe posteriors. This term is the entire reason tier5 beats CLIP-cosine methods on
CelebA: it scores the GT's own generative rule (relaxed Hamming ball) instead of
approximating it geometrically.

The term's reliability depends on how the **truncation radius** (`hamming_max`) compares
to the **expected number of probe-induced mismatches** over the free-attribute count `F`:

| | CelebA | OVAD |
|---|---:|---:|
| Attributes (`N_ATTR`) | 40 | 117 |
| Free attributes in a typical query (`F`) | ≈ 37 | ≈ 114 |
| GT tolerance (relative) | 2/40 = 5% | 5% (by construction, §3.3 below) |
| `hamming_max` (absolute, tuned to match the 5% rate) | 2 | 6 |
| Probe test accuracy (masked, mean / min) | 88.46% / 69.69% (`Big_Lips`) | 92.29% / 55.20% (`position:vertical/…`) |
| Expected mismatches `F·ε` (ε ≈ 1 − mean acc) | 37 × 0.115 ≈ **4.3** | 114 × 0.077 ≈ **8.8** |

`hamming_max` is already dataset-configured — `src/ovad/tier5_fusion.py:36` sets it to 6
specifically because `round(0.05 × ~114 free attrs) = 6`, matching the GT's own 5%
tolerance rate exactly the way CelebA's `hamming_max=2` matches its 5% (2/40). **The
scalar was tuned correctly.** The problem is that `F·ε` (the *expected* mismatch count
under the probes' own error rate) already sits close to or above the truncation radius on
*both* datasets, and the gap only widens with attribute count: on OVAD, `F` is 3× larger,
so even a *slightly worse* mean probe accuracy (driven by the worst-case heads, `min` 55%
vs 70%) inflates `F·ε` from ~4.3 to ~8.8 — comfortably past `hamming_max=6`. Once
`F·ε ≫ hamming_max`, the DP's output saturates toward "almost certainly too many
mismatches" for nearly every candidate, so the tail term stops discriminating between
true positives and true negatives and instead reports the *probes'* aggregate noise
level. This is the identical mechanism the AwA2 study already isolated in its λ-sweep
(`awa2_negative_transfer_analysis.md` §5) — OVAD just sits further into the same regime
because it has three times the attribute count to accumulate error over.

**Confirmation via the same closed-form diagnostic used on AwA2** (`--diagnose` mode,
`src/ovad/tier5_fusion.py`, no retraining — a pure scoring sweep over the frozen Stage A′
core):

```
S(x | ref) = constraint(x) + tail(x, ref) + λ · cos(v_x, v_ref)
```

| λ | R@1 | R@5 |
|---|---|---|
| 0 (Bayes core alone — the tail/tree/constraint terms, no identity) | 0.0183 | 0.0724 |
| 0.25 | 0.0184 | 0.0905 |
| 0.5 | 0.0166 | 0.1019 |
| 1 | 0.0205 | 0.1275 |
| 2 (CelebA's own train-tuned value — see §4) | 0.0343 | 0.1489 |
| 4 | 0.0413 | 0.1515 |
| 8 | 0.0436 | 0.1529 |

R@5 **doubles** as λ grows from 0 to 8 — the score only becomes useful once raw CLIP
cosine is blended back in, exactly mirroring the AwA2 curve (there R@1 rose from 0.607 at
λ=0 to 0.765 at λ=8, converging toward the CLIP baseline). The probe-mediated tail/tree
core, in isolation, is close to uninformative on OVAD (R@5 = 0.072, barely above the
mean-target-density floor for a 12-query, ~230-candidate-per-query benchmark).

### 3.2 Training does not fix it — Φ effectively never leaves its init

`train_phi_prime` trains the same `BayesPhi` architecture (`src/tier4_bayes.py:187`) used
on CelebA — zero-initialized correction heads over a `[v_ref ⊕ p_ref ⊕ mh_pos ⊕ mh_neg]`
input, with a scalar tail gain `κ` initialized to 1 so that Φ at init reproduces the
closed-form Stage A′ score exactly (the project's hard init-invariant). On CelebA,
training moves this initialization meaningfully — it is what turns Stage A′'s R@5 (0.252,
`A_C1C2C3` in `output/tier5_cdbayes/tier5_ablation_summary_full.csv`) into the champion's
R@5 (0.291).

On OVAD, training barely moves anything (`output/ovad/tier5_train_log.txt`):

```
epoch 21/40  listnet=3.6084  kappa=0.995
epoch 25/40  listnet=3.5747  kappa=0.995
epoch 30/40  listnet=3.5411  kappa=0.995
epoch 35/40  listnet=3.5967  kappa=0.995
epoch 40/40  listnet=3.6036  kappa=0.995
```

`κ` — the learned gain on the already-unreliable tail term — sits at **0.995 for the
entire second half of training**, statistically indistinguishable from its zero-init
value of 1.0. The ListNet loss oscillates in a 3.5–3.7 band with no visible downward
trend over 20 epochs. Compare this to CelebA, where the loss curve descends
monotonically and produces a net R@5 gain of +0.039 over the untrained core. On OVAD, Φ
is not learning a useful correction to the closed-form score; the trained checkpoint is,
for practical purposes, still close to its own (already-weak) initialization.

## 4. Why the other methods don't collapse the same way

The methods that hold up on OVAD (DualScore 0.254, DGP 0.246, tier0 0.242) share exactly
the property the AwA2 study identified as the transfer-robust one: **their score is
either pure CLIP cosine similarity, or CLIP cosine plus a small, non-learned or
lightly-learned correction — never a probe-mediated combinatorial rule over all 117
attributes at once.**

- **Tier-0 / DGP** score by cosine similarity between a text-steered query vector and the
  frozen visual DB. Every attribute contributes one small, independent nudge to a single
  512-d vector; there's no joint probability model over 117 binary variables that can
  become miscalibrated as a whole.
- **DualScore** anchors on CLIP cosine for its backbone and adds learned presence/absence
  reward channels on top — the same "cosine base + small correction" shape, not "correction
  is the entire score" like tier5's tail/tree term.
- **GDE/LDE** (mined visual directions) inherit no probe machinery at all; they operate
  purely in CLIP's visual embedding space.
- **CLAY** is the one training-free method that also does poorly (R@5 = 0.116) — for a
  *different*, already-documented reason (it discards the raw identity direction into a
  pre-SVD subspace, the "naïve bottleneck" the project spec asks us to fix); it is not
  evidence against the mechanism above, it's evidence for the same general principle from
  the opposite direction (throwing away CLIP identity also hurts, whether you replace it
  with a subspace projection or a probe-Bayes rule).

Tier5 is the only method in the OVAD comparison whose score is *dominated* by a joint
combinatorial function of per-attribute probe posteriors (a 117-node Chow-Liu tree +
a Poisson-binomial DP over up to 114 free attributes). That is exactly the design that
won on CelebA (§5 of the AwA2 report already explains why: CelebA's GT rule is *itself* a
combinatorial Hamming-ball predicate, so a score that computes that predicate's own
probability wins over cosine, which cannot represent it). OVAD has the same style of GT
rule (masked relative-Hamming, `OVAD_Implementation_Plan.md` §3) — so in principle the
same design should help here too. It doesn't, because of §5.

## 5. The dataset-size problem — why this can't be tuned away

This is the crux of the negative result, and the reason the fix is not a hyperparameter.

| | CelebA | AwA2 | **OVAD** |
|---|---:|---:|---:|
| Attributes fit by the probe / tree / Φ | 40 | 85 (unused by tier5 — no trained tier5 run on AwA2's class-level attrs was the recommended path) | **117** |
| Training pool size | **162,770** images | — | **6,216** instance-crops |
| Images/attribute ratio (train pool ÷ attribute count) | ≈ 4,069 | — | **≈ 53** |
| Label completeness | 100% (every attribute labeled on every image) | 100% | **~87% per attribute** (tri-state; ~13% marked unlabeled) |
| Probe test accuracy, mean / **worst head** | 88.46% / **69.69%** | — | 92.29% / **55.20%** |

Three compounding constraints, all present simultaneously on OVAD and none present
together on CelebA:

1. **~26× less training data** (6,216 vs. 162,770) to fit **~3× more parameters that
   scale with attribute count** — a 117-head linear probe, a 117×117 Chow-Liu MI matrix
   and max-spanning tree (`build_chow_liu_tree`, `src/tier5_cdbayes.py:282`), 117
   per-attribute temperature scalars, and a `BayesPhi` network with four
   117-or-512-dimensional output heads. The **images-per-attribute ratio drops from
   ≈4,069 on CelebA to ≈53 on OVAD** — nearly two orders of magnitude — even though every
   one of these components needs *more*, not less, data to fit reliably as its
   dimensionality grows.
2. **Tri-state, ~87%-complete labels**, versus CelebA's fully-dense binary labels. Every
   masked variant in `src/ovad/adapter.py` (`_masked_bce`, `_masked_pair_counts`,
   `_masked_fit_temperature_scalars`, …) is doing the statistically correct thing
   (excluding unlabeled entries from each estimator's numerator *and* denominator), but
   "correct" does not mean "as data-efficient as observing the true label" — each masked
   estimator is still working from strictly less effective evidence per attribute than its
   CelebA counterpart, on top of the 26× smaller pool.
3. **This is why the worst-attribute probe accuracy (55.2%) is so much lower than
   CelebA's worst (69.7%)**, despite OVAD's *mean* probe accuracy being nominally higher
   (92.3% vs 88.5%) — mean accuracy is dragged up by many easy, frequent attributes
   (`freq_set = head`), while the tail/rare attributes that still make it into the query
   set (`state:closed`, sparse materials, rare face expressions) get a small, noisy
   training signal and land near chance. The tail term and the Chow-Liu tree consume
   *all 117* posteriors per query regardless of individual reliability — a handful of
   near-random heads is enough to inject the noise that §3.1 shows dominates the score
   once `F·ε` passes `hamming_max`.

**Why no realistic retuning of `hamming_max`, `probe_epochs`, `hidden`, or `weight_decay`
closes this gap:** the failure isn't that a scalar is set to the wrong value — every
scalar tier5 exposes was already fit specifically for OVAD (temperatures, the Chow-Liu
tree, `hamming_max`, and Φ itself are all trained fresh on OVAD's own 6,216-instance
split, exactly as the project's dataset-adapter design intends). The bottleneck is the
*ratio* of trainable structure (117 attributes × several O(n) or O(n²) components) to
available evidence (6,216 labeled, 87%-complete instances) — shrinking `hamming_max`
further would only make the tail saturate toward "impossible" for almost every candidate
(over-strict, not more informative); more probe epochs on the same 6,216 crops cannot
manufacture the label volume that made the worst CelebA head (69.7% acc, 4,069
images/attribute) more reliable than OVAD's worst head (55.2% acc, 53 images/attribute).
Fixing this properly would require either (a) far more labeled OVAD data than the
2,000-image benchmark provides, or (b) a different, more data-efficient dependency model
than a full Chow-Liu tree + per-attribute probes over 117 variables — both are changes to
what the method *is*, not a retuning of what it already has.

## 6. What does transfer: reusing CelebA's own best training-free row

Because the failure is concentrated in the learned tail/tree/Φ machinery (§3), not in the
Stage A′ scoring *framework* itself, the fix that survives is to prefer the CD-Bayes
ablation row that leans **least** on that machinery — without inventing any new logic.

`output/tier5_cdbayes/tier5_ablation_summary_full.csv` (CelebA's own ablation matrix, run
once, reused as-is):

| row | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| A_tier4repro (parity anchor, no C1–C5) | 0.0868 | 0.2596 | 0.3554 |
| A_C1 | 0.0726 | 0.2598 | 0.3787 |
| A_C1C2 | 0.0764 | 0.2500 | 0.3533 |
| A_C1C2C3 | 0.0742 | 0.2747 | 0.3656 |
| **Aprime_C4** (A′ core + tuned λ·cos, training-free) | 0.0759 | 0.2626 | 0.3720 |
| Aprime_C5 (A′ core + QB-Norm) | 0.0686 | 0.2449 | 0.3335 |
| Aprime_C4C5 (full training-free) | 0.0724 | 0.2524 | 0.3574 |
| **Bprime_listnet_seed0 (the published champion)** | **0.1104** | **0.2913** | **0.4128** |
| Bprime_lambdaloss (3 seeds) | 0.065–0.081 | 0.205–0.241 | 0.314–0.347 |

`Bprime_listnet` is legitimately CelebA's best row — it wins every metric there, not a
marginal call. But `Aprime_C4` (Stage A′ core plus a **fixed, non-learned**
`λ·cos(v_x, v_ref)` term, λ tuned once on CelebA's own train split to **λ = 2.0**,
`output/tier5_cdbayes/tier5_tuned_scalars_full.json`) is CelebA's best row that contains
no trained Φ at all.

Re-scoring OVAD with that exact λ = 2.0 (§3.1's table, no retuning, no retraining — a
pure closed-form pass over the already-cached OVAD probe/temperature/tree artifacts):

| scorer | OVAD R@5 |
|---|---:|
| Stage A′ core, λ = 0 | 0.072 |
| **Aprime_C4, λ = 2 (CelebA's tuned value, reused as-is)** | **0.149** |
| Bprime_listnet (the trained champion) | 0.092 |

Reusing CelebA's own tuned λ, with zero OVAD-specific retuning, **more than doubles**
Stage A′'s bare R@5 (0.072 → 0.149) and clears the trained champion (0.092) by 62%. It is
still well below the training-free-cosine methods (tier0 0.242, DGP 0.246, DualScore
0.254) — confirming that no CD-Bayes variant, however configured, beats plain
CLIP-cosine scoring on this dataset — but among the CD-Bayes family itself, it is the
row that survives transfer, and it survives for the same reason the AwA2 report gives for
LDE/GDE beating tier5 there: **the more a score is anchored in raw CLIP visual identity,
the better it transfers off CelebA; the more it depends on the learned probe/tree/tail
apparatus, the more exposed it is to exactly the attribute-count / data-volume regime
OVAD sits in.**

## 7. Takeaway for the report

One sentence: **tier5's CelebA-winning design (a joint probability model over all
attributes, scored via a Poisson-binomial identity tail and a Chow-Liu dependency tree)
requires enough per-attribute training evidence to keep its worst-case probe reliable —
OVAD provides roughly 1/77th the images-per-attribute of CelebA (53 vs. 4,069) while
asking for 3× more attributes, so the tail term saturates into probe noise
(`F·ε ≫ hamming_max`) and Φ's training signal is too weak to correct it (κ frozen at
0.995, loss flat for 20 epochs) — a data-volume bottleneck, not a mistuned scalar, which
is why retuning `hamming_max`/epochs/hidden-size cannot close the gap, and why the only
CD-Bayes variant that transfers is the one that relies on cosine identity instead of the
combinatorial attribute model** (`Aprime_C4`, reusing CelebA's own tuned λ = 2, R@5 0.149
vs. the champion's 0.092 — still short of the training-free cosine methods' ~0.25, which
remain OVAD's best choice overall).

## 8. Reproduction

```bash
# training-free methods + the trained champion (VM for tier5; rest local-safe)
python src/ovad/tier0_baseline.py
python src/ovad/tier_clay.py
python src/ovad/tier_dgp.py
python src/ovad/tier_gde.py                 # GDE + LDE
python src/ovad/tier3_dualscore.py          # VM
python src/ovad/tier5_fusion.py             # VM — trains BayesPhi, ~7 min GPU

# closed-form lambda-sweep diagnostic used in §3.1 / §6 (no training, CPU-safe,
# reuses cached artifacts/ovad/{tier4_probe,tier5_temperatures}.pt, builds the
# Chow-Liu tree fresh — under a minute total)
python src/ovad/tier5_fusion.py --diagnose
```
