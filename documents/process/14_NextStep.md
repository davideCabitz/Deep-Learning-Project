# NextStep — From Tier-4 Bayes-Φ to Project Completion

> **📜 Process document, kept as written.** Results ledger, diagnosis and plan by Davide Cabitza, July 2, 2026. Restored from git history.
>
> **Outcome:** "CD-Bayes" became **BAP+** and the final **A-BAP**. C1 (calibration), C2 (Chow–Liu tree) and C3 (conditioned tail) were kept. C4 (cosine channel) and C5 (QB-Norm) were tested and rejected. Stage B′ with ListNet became the champion (R@1 0.110, R@5 0.291, R@10 0.413), while LambdaLoss underperformed on all three seeds. The statistics, custom-query and AwA2 plans in Part IV were all carried out.
>
> **Correction:** Part I says six Tier-3 variants were "coded but never run". The git history shows results for all of them by July 2 (see the banners on [11](11_Tier3_Variations_Report.md) and [12](12_Negation_Fusion_Report.md)). File paths in this document refer to the layout before the July 5 refactor; see the [old → new file map](../README.md#old-file-names).

> **Role of this document.** Head-researcher retrospective + forward plan. Part I audits every
> result the project has produced (all numbers traced to CSVs in `output/` or tables in
> `documents/`). Part II diagnoses what the data actually says. Part III specifies the one
> method expected to beat both Tier-4 stages. Part IV defines the evaluation pipeline —
> statistical soundness, custom queries, AwA2. Part V is the priority-ordered roadmap to the
> deliverable. Everything here is constrained by
> [documents/project_specification.md](../project_specification.md): frozen CLIP
> ViT-B/32 (§3.3), DB built once & frozen (§3.2), GT rule §3.1.1, single-notebook deliverable
> (§4), from-scratch policy (§6). Ground-truth labels are used **only to define benchmarks,
> never inside scoring** — that invariant is restated wherever it is load-bearing.

---

## Part I — Complete results ledger (everything we have run)

MEAN over the 13 unique mandatory queries (the eval JSON runs 14 entries; `-Young` appears
twice — see Honest Calls). Best configuration per method; every row traces to a CSV in
`output/`.

| # | Method (best config) | R@1 | R@5 | R@10 | Source |
|---|---|---|---|---|---|
| 0 | Tier-0 CLIP arithmetic baseline | 0.0224 | 0.0699 | 0.1048 | `tier0_CLIP/tier0_baseline.csv` |
| 0+ | Tier-0 enhanced — **mean-centering only** | 0.0377 | 0.1142 | 0.1755 | `tier0_enhanced/tier0_enhanced_fix1_center.csv` |
| 0+ | Tier-0 enhanced — all fixes | 0.0393 | 0.1102 | 0.1651 | `tier0_enhanced/tier0_enhanced_all_fixes.csv` |
| 1 | Tier-1 GDE | 0.0221 | 0.0607 | 0.0910 | `tier1_GDE/tier1_GDE.csv` |
| 1 | Tier-1 CLAY (best: k50, rot H) | 0.0098 | 0.0368 | 0.0541 | `tier1_CLAY/tier1_CLAY_k50_rotH.csv` |
| 1 | Tier-1 CLAY (worst: k5, rot H) | 0.0017 | 0.0075 | 0.0156 | `tier1_CLAY/tier1_CLAY_k5_rotH.csv` |
| 2a | Stacked-SVD (best: k50/50, λ=0.1) | 0.0137 | 0.0454 | 0.0659 | `tier2a/…k50_50_lam0.1_rotH.csv` |
| 2b | LDE/GDE arithmetic (best: α=1.5) | 0.0248 | 0.0724 | 0.1085 | `tier2b/tier2b_gde_alpha1.5.csv` |
| 2c | Orthogonal rejection (best: kneg10, α=1.5) | 0.0200 | 0.0699 | 0.1038 | `tier2c/tier2c_query_kneg10_a1.5.csv` |
| 2d | DGP (best: τ=100, α=1, centered) | 0.0363 | 0.1152 | 0.1777 | `tier2d/tier2d_tau100.0_a1.0_center.csv` |
| 3 | DGP Φ (InfoNCE) | 0.0383 | 0.1110 | 0.1607 | `tier3_dgp/fusion_dgp_phi.csv` |
| 3 | Cross-attention Φ | 0.0311 | 0.1038 | 0.1607 | `tier3c/tier3c_phi.csv` |
| 3 | SymNeg | 0.0384 | 0.1090 | 0.1630 | `tier3_symneg/…k50_tau0.07_a1.csv` |
| 3 | Composer | 0.0052 | 0.0233 | 0.0375 | `tier3_composer/tier3_composer.csv` |
| 3 | **Combined** (best geometric method) | 0.0563 | 0.1576 | 0.2329 | `tier3_combined/tier3_combined.csv` |
| 4 | Hybrid (probe-penalty negation) | 0.0426 | 0.1256 | 0.1854 | `tier4_hybrid/tier4_hybrid.csv` |
| 4 | **Bayes Stage A** (closed-form, training-free) | **0.0868** | 0.2596 | 0.3554 | `tier4_bayes/tier4_bayes_stageA_full.csv` |
| 4 | **Bayes Stage B** (Φ, Bayes-init, ListNet) | 0.0783 | **0.2811** | **0.4009** | `tier4_bayes/tier4_bayes_stageB_full.csv` |

### Per-tier verdicts (what each taught, why each plateaued)

- **Tier-0 (floor).** Naïve latent arithmetic scores R@5 ≈ 0.07 — the modality gap makes raw
  `v_ref + Σt⁺ − Σt⁻` geometrically crude, exactly as the spec's roadmap (§3.2) anticipates.
  **Mean-centering the embedding space was the single most valuable geometric trick of the
  entire project** (+63% R@5 over raw, and `fix1_center` alone beats `all_fixes` at R@5/R@10):
  most of CLIP's image–text offset is a constant vector, and removing it helps more than any
  subsequent subspace surgery.
- **Tier-1 (assigned-literature reproductions).** Honest finding, worth stating plainly in
  the final report: **CLAY — the method the assignment builds on — is the worst performer of
  the whole ladder** (R@5 0.0075–0.0368, below the naive baseline at every configuration),
  and GDE also lands below Tier-0. This is not an indictment of those papers — it is evidence
  that their objective (semantic direction quality) is not this benchmark's objective
  (identity-constrained retrieval under a Hamming rule). It empirically validates the
  assignment's own premise that the pre-SVD stack is a bottleneck (§1) and legitimizes our
  departure from geometry.
- **Tier-2 (training-free geometric ladder).** Four families of subspace surgery (stacked
  SVD, LDE arithmetic, orthogonal rejection, DGP) and not one beat Tier-0-enhanced-centering
  meaningfully (best: 2d at R@5 0.1152 vs 0.1142). Conclusion: within single-vector cosine
  scoring, the training-free design space was exhausted. The ladder was worth building — it
  proves the plateau is structural, not a tuning failure.
- **Tier-3 (trained Φ over geometry).** Training on top of the same representation moved the
  needle only when methods were ensembled (Combined, R@5 0.1576 — the best geometric result).
  Composer collapsed globally (R@5 0.0233) yet was the first method to break the
  `-Male,-Mustache` floor (R@10 0.1111) — the first hint that negation needed a *different
  scoring space*, not a better loss. Six further variants (contrastive, dualscore, hybrid,
  negation, neggate, polaritycomposer) are coded but never run — see Honest Calls.
- **Tier-4 Hybrid.** Probe-penalty negation bolted onto cosine scoring: broke the negation
  floor weakly (`-Male,-Mustache` R@10 0.0370) but degraded the mean below Combined and
  zeroed the lipstick composite. Transitional evidence that probes help but the *fusion* was
  wrong.
- **Tier-4 Bayes (current best).** Scoring the GT rule itself in probability space:
  Stage A (training-free) R@5 0.2596, +65% over Combined, negation floor broken
  (`-Male,-Mustache` 0.000 → 0.222 R@5); Stage B (trained Φ, provably initialized at
  Stage A — init-sanity deviation 0.00e+00) R@5 0.2811, R@10 0.4009. Cost so far: ~1.3 of
  40 GPU-h. The pivot from geometric to probabilistic scoring is the project's central
  scientific finding.

### Honest calls (state these in the final report; hiding them costs more than they do)

1. **Single-seed status.** Every trained number above is one seed, one run. Stage B's +8%
   R@5 over Stage A is plausible but not yet variance-qualified.
2. **Small-n fragility.** `-Male,-Mustache` has 27 sources; the lipstick composite has 34.
   Stage B's R@10 = 0.5185 on the former is ~6 sources flipping. Wilson 95% CI on its R@5
   (0.259, n=27) is ≈ [0.13, 0.45] — Stage A's 0.222 sits comfortably inside. Per-query
   superiority claims on these queries are **not individually supported**; only aggregate
   and paired analyses are (Part IV fixes this).
3. **The R@1 regression.** Stage B trades R@1 (0.0868 → 0.0783, −10%) for R@5/R@10. The
   spec's rubric (§4) cites "e.g., R@1, R@5" — both will be looked at. Stage A remains the
   R@1-best operating point; Part III targets the recovery.
4. **Stage-B regressions are patterned, not noise.** Every per-query loss vs Stage A is a
   hair-color/makeup query (Black_Hair −33%, Heavy_Makeup −22%, Blond_Hair −15% at R@5);
   every gain is compositional/negation/rare (Chubby−Young +113%, Mustache +70%, Male +30%).
   A patterned trade-off has a cause; Part II names the suspect.
5. **The lipstick composite is still unsolved.** A: 0.0000, B: 0.0588 R@5. Off the floor,
   not off the problem.
6. **`-Young` appears twice in the eval JSON** (entries 05 and 08; identical results, and
   results dictionaries dedupe by query string so MEANs are over 13 unique queries).
   De-duplicate explicitly in all report tables and say so in a footnote.
7. **Six coded-but-never-run Tier-3 variants.** Recommendation: **archive as superseded.**
   They are all single-vector-cosine designs; the Bayes results invalidate their premise.
   Spending GPU-h on them now would be sunk-cost work that cannot alter the conclusion.

---

## Part II — Diagnosis: five failure modes the data shows

Each mode is named, evidenced from the ledger, and mapped to a fix in Part III.

**(i) The attribute-independence assumption.**
Both Bayes terms factorize over attributes: the constraint term multiplies marginal
posteriors, the identity tail assumes independent disagreement events. CelebA attributes are
strongly dependent — `Wearing_Lipstick` and `Heavy_Makeup` co-occur so tightly that their
probes co-fire. The query `+Wearing_Lipstick, -Heavy_Makeup, +Smiling` asks for a
correlation-violating combination; under independence, `p(Lipstick)·(1−p(Makeup))` is
structurally wrong because the true conditional `p(Lipstick=1, Makeup=0 | x)` is nothing like
the product. Evidence: the one query where Bayes loses to *every* geometric method (A: 0.000
vs Combined's 0.088 R@5). → Fix: **C2, C3**.

**(ii) Probe miscalibration.**
Probes are BCE-trained sigmoids used *as if* they were calibrated posteriors — but modern
networks' confidences are systematically miscalibrated (Guo et al., 2017), and class
imbalance makes it worse (Mustache ≈ 4% positive rate; probe floor 69.7% accuracy on
Big_Lips). Log-likelihood scoring amplifies calibration error: an overconfident wrong probe
contributes an unboundedly wrong log term. Evidence: Stage A's weakest wins concentrate on
rare/imbalanced attributes (+Mustache 0.133 vs Combined 0.206). → Fix: **C1**.

**(iii) ListNet's flat target mass.**
ListNet's target distribution is uniform over positives — rank 1 and rank 4 are equivalent to
the loss. The measured outcome is exactly the predicted one: mass shifted from R@1 (−10%)
into R@5/R@10 (+8%/+13%). → Fix: **Stage B′ LambdaLoss**.

**(iv) The learned CLIP channel reintroduces CLIP's bias (hypothesis — needs the ablation).**
Stage B's `q_v` head learns a per-query CLIP-space query. The regression pattern
(hair-color/makeup queries — precisely the attributes where CLIP's texture/color bias is
strongest) implicates it, but the evidence is correlational until the on/off ablation runs.
→ Fix: **Stage B′ q_v gate + C4**.

**(v) No score normalization anywhere in the ladder.**
Retrieval scores are used raw. Hubness — a few gallery items ranking high for many queries —
is endemic to high-dimensional retrieval and correctable training-free (QB-Norm, Bogolin et
al., CVPR 2022). Never tested in 19 methods. In probability space the analogue exists too:
candidates with confident, ref-like posterior profiles can dominate many references' tails.
→ Fix: **C5**.

---

## Part III — The next method: Tier-5 **CD-Bayes** (Correlation-aware Dependent Bayes)

One method, five independently-ablatable components plus a retrained Φ. Each component
targets exactly one diagnosed failure mode; each can be merged or dropped on evidence. The
design principle carried over from Tier-4 — *never regress from a proven floor* — is
preserved: every component defaults to a setting that reproduces Tier-4 exactly.

### C1 — Calibrated probes (fixes ii)

Per-attribute temperature scaling (Guo et al., 2017): hold out a slice of TRAIN, fit one
scalar `τ_a` per attribute minimizing NLL of `σ(z_a/τ_a)`, freeze. One parameter per
attribute, minutes of CPU, and both Bayes terms improve simultaneously because both consume
the posteriors. Report probe ECE (expected calibration error) before/after — a
report-quality diagnostic that costs nothing.

### C2 — Chow-Liu dependency tree for the constraint term (fixes i — the marquee fix)

Learn the maximum-spanning tree over the 40 attributes under pairwise mutual information,
estimated by counting on TRAIN labels (Chow & Liu, 1968; the TAN lineage, Friedman et al.,
1997). O(40²) counting, exact, from scratch in ~100 lines. The constraint term becomes the
tree-factorized joint conditioned on probe soft evidence:

```
constraint(x) = log p_tree( A_{T⁺} = 1, A_{T⁻} = 0 | evidence = probe posteriors of x )
```

computed by exact sum-product on the tree (40 binary nodes — trivial message passing, no
approximation). The tree will almost certainly place an edge Lipstick—Makeup with a learned
conditional table; `p(Lipstick=1, Makeup=0 | x)` is then computed *through the dependency*
instead of assumed independent. This is the principled version of what the lipstick query
demands. Spec fit: the tree's edge set and conditional tables are counted **once, offline,
from TRAIN labels** — DB-side artifacts exactly like the probe matrix (§3.2); no test-label
usage.

### C3 — Conditioned Hamming tail (fixes i, second-order)

The Poisson-binomial DP stays (it is exact given marginals). What changes is the marginals it
consumes: one belief-propagation pass conditions the free attributes' posteriors on the
constrained assignment (e.g., conditioning on `Male=0` shifts Beard/Mustache/Sideburns
priors), then the existing DP runs on the conditioned marginals. Keeps the DP's exactness,
captures first-order dependence, reuses `hamming_tail_log` unchanged.

### C4 — λ-cosine identity channel (targets the R@1 recovery, non-learned)

```
S(x|ref) += λ · cos(v_x, v_ref),   λ a single scalar tuned on synthetic TRAIN queries
```

The *non-learned* counterpart of Stage B's `q_v`. Rationale from evidence: identity
tie-breaking on tail plateaus is real signal (many candidates share near-identical Hamming
tails; CLIP cosine breaks the tie), but *learning* a per-query CLIP query appears to overfit
toward CLIP's color bias (failure iv). A single global λ captures the tie-breaking benefit
with no capacity to overfit. Directly targets R@1.

### C5 — QB-Norm score normalization (fixes v)

Querybank normalization with the Dynamic Inverted Softmax (Bogolin et al., CVPR 2022):
build a bank of synthetic-train-query score vectors once, normalize test-time scores against
it. Training-free, post-hoc, ~30 lines, ablatable in isolation. Spec fit: pure inference-time
re-normalization; DB untouched.

### Stage B′ — Φ retrained on the CD-Bayes floor (fixes iii, iv; carries the agreed next steps)

Same `BayesPhi` architecture and zero-init invariant (init ≡ Stage A′ exactly), three
committed changes:

1. **LambdaLoss replaces ListNet** (Wang et al., CIKM 2018). ListNet is rank-position-blind
   past the positive set; LambdaLoss weights each pairwise error by the ranking-metric change
   it causes, making top-of-list mistakes the most expensive — the standard cure for the
   measured R@1→R@5 mass shift.
2. **`q_v` gate ablation**: train with and without the CLIP head. Confirms or refutes
   hypothesis (iv) with one flag.
3. **3 seeds minimum**, reported mean±std (Part IV).

### Why CD-Bayes should beat both Tier-4 stages

- **vs Stage A**: C1–C3 lift the closed-form floor itself — calibration corrects every log
  term, the tree fixes the one query Stage A structurally cannot score, the conditioned tail
  sharpens identity. All closed-form: zero training risk, CPU-feasible, and the new Stage A′
  inherits Stage A's R@1 crown with headroom.
- **vs Stage B**: its two measured regressions are each directly removed — the R@1 dip by
  LambdaLoss + C4, the color-query losses by the q_v gate (if hypothesis iv holds) — while
  its gains (compositional/negation queries) came from the Bayes channels, which only get
  stronger under C1–C3.
- **Risk containment**: every component defaults to Tier-4 behavior (τ=1, no tree edges,
  unconditioned tail, λ=0, no normalization ⇒ bit-for-bit Stage A). The floor is never at
  risk; each component must *earn* its place in an ablation.

**Spec compliance.** Frozen CLIP ViT-B/32 throughout (§3.3). DB, probes, temperatures, tree
tables, querybank: all built once offline and frozen (§3.2). Training-free components are
explicitly sanctioned (§3: "training-free … strategies"); Φ remains the trained fusion module
the task centers on (§1). Everything from scratch; Chow-Liu, temperature scaling, QB-Norm and
LambdaLoss are implemented from their papers and cited (§6).

---

## Part IV — Evaluation pipeline

### IV.1 Statistical soundness (all post-hoc, ~0 GPU-h)

The central insight: R@K per query is a **sample proportion over a finite, small source
set**, and our two decisive uncertainties are of different kinds. Seeds capture *training*
variance; only intervals over sources capture *evaluation-sampling* variance — and on the
n=27/n=34 queries the latter dominates. No amount of seed-averaging fixes it.

**Prerequisite (one code change).** The eval driver additionally dumps a **per-source hit
matrix** — `query × source × hit@{1,5,10}` — per method and seed. Every statistic below is
then CPU-seconds on cached matrices; methods are never re-run for statistics.

| Instrument | What it answers | Why this one |
|---|---|---|
| **Wilson score interval** per query | "How uncertain is this R@K?" | Correct at the extremes where we live: at p̂=0 (Stage A lipstick, 0/34) the normal approximation collapses to zero width; Wilson gives [0, 0.10]. |
| **Paired bootstrap over sources** (B=10 000) | CI on MEAN R@K and on **method deltas** | Propagates small-n fragility into the headline number; the delta CI (resample sources, recompute mean_B − mean_A) is the honest test of "+8% R@5 is real". |
| **McNemar's test** per query | "Does B beat A *on this query*?" | Both methods score identical sources → the claim is paired. McNemar uses only discordant sources (one hit, one missed) — the correct, most powerful test; it can establish superiority even where marginal CIs overlap. |
| **Holm–Bonferroni** across the family | Multiple-comparisons control | 13 queries × methods × 3 K's is many tests; any "significant" flag must survive family-wise correction or be reported as exploratory. |
| **3–5 seeds** (trained components only) | Training variance, mean±std | Decide and state the probe protocol: resampled per seed (`--force-probes`, variance includes probe noise) or held fixed (Φ-conditional variance). Either is defensible; ambiguity is not. |

**Reporting rules.** De-duplicate `-Young` (footnote it). Flag every n<50 query as a wide-CI
point estimate, never a headline. Present Stage A′/B′ as a **Pareto pair** (R@1 champion /
R@5-R@10 champion) unless LambdaLoss collapses the trade-off — the spec's rubric cites both
R@1 and R@5, so both operating points stay in the report.

### IV.2 Custom queries (spec §3.1, explicitly encouraged)

**Prerequisite: a GT generator** (`src/generate_ground_truth.py`) implementing §3.1.1
verbatim over `celeba_attributes_test.pt`: strict T⁺/T⁻ satisfaction **and** Hamming ≤ 2 on
the free attributes vs the reference, keep sources with ≥ 5 valid targets, emit JSON in the
official file's exact shape (so `eval.py` and every scorer consume it unchanged —
`tier4_bayes.py` needs zero modification, and Φ is conditioned on T⁺/T⁻ masks at inference,
so it answers unseen attribute combinations without retraining).

**Integrity line, stated in the notebook**: test labels are read to *define* the benchmark —
the same operation the instructors performed to produce the official JSON — and never enter
scoring, which consumes only probe posteriors and CLIP features.

**Query families — chosen to stress the method's seams, not to pad the table:**

| Family | Example | What it probes |
|---|---|---|
| Deep negation | `-Male, -Mustache, -Goatee` | identity term under stacked negation |
| Correlated traps | `+Goatee, -Mustache`; `+Heavy_Makeup, -Wearing_Lipstick` | C2's dependency tree, the lipstick pathology inverted |
| Rare attributes | `+Bald`; `+Pale_Skin`; `+Gray_Hair` | probe calibration limits (C1) |
| Wide composition | `+Male, +Chubby, +Eyeglasses, -Young` | constraint-term degradation at 4 constraints |

Report each family with the same Wilson/bootstrap treatment; a family-level failure is a
*finding*, not an embarrassment.

### IV.3 AwA2 generalization (spec §2 names it)

The method is dataset-agnostic by construction — nothing in probes, tree, tail or Φ is
CelebA-specific; they consume `(features, labels, n_attr)`. Port via a small **dataset
strategy** object (names / features / labels / n_attr) so `N_ATTR=40` stops being a
constant; CelebA and AwA2 become two instances (85 attributes for AwA2). Pipeline: one
frozen-CLIP feature pass over AwA2 images, binary label tensor, same GT generator, same
scorers.

**Honest caveat to print next to the results**: AwA2 attributes are annotated **per class**,
not per image — all images of a class share one attribute vector, so within-class Hamming
distance is 0 and the identity ball collapses to (approximately) species identity. The
experiment still answers a real question — *does probability-space scoring transfer to a
different domain and attribute vocabulary?* — but its Hamming term is coarser than CelebA's,
and the report must say so. (OVAD is rejected: per-instance attributes in multi-object
scenes do not map onto a single-vector-per-image pipeline; the plumbing cost buys murkier
conclusions.)

---

## Part V — Priority-ordered roadmap (no dates; dependency-ordered)

| P | Work item | Depends on | GPU-h | Rationale for its slot |
|---|---|---|---|---|
| **P0** | Per-source hit-matrix dump + stats module (Wilson, paired bootstrap, McNemar+Holm); run on the **existing** Tier-4 A/B results | — | ~0 | Highest value per unit cost in the project: upgrades already-obtained results to report-grade claims before any new method work. |
| **P1** | **[implemented — pending run]** CD-Bayes Stage A′ = C1+C2+C3, closed-form; ablate each (τ on/off, tree vs independent, conditioned vs raw tail) — `src/tier5_cdbayes.py` rows `A_tier4repro…A_C1C2C3` | P0 (stats ready to judge it) | ~0 (CPU) | Attacks the lipstick pathology and the calibration debt with zero training risk; every later comparison stands on this floor. |
| **P2** | **[implemented — pending run]** C4 (λ-cosine) + C5 (QB-Norm), each ablated in isolation on Stage A′ — rows `Aprime_C4/C5/C4C5`, scalars grid-tuned on synthetic TRAIN queries | P1 | ~0 | Two training-free add-ons; C4 is the cheap R@1 play. |
| **P3** | **[implemented — pending run]** Stage B′: LambdaLoss Φ on Stage A′, q_v gate on/off, **3 seeds** — rows `Bprime_*` (stats pipeline still P0) | P1 (floor), P0 (stats) | ~5–8 | The committed trio (seeds / q_v ablation / top-heavy loss) in one batched VM job; single-run→multi-seed upgrade for every trained claim. |
| **P4** | GT generator + custom-query families; evaluate Stage A′/B′ (+ Tier-4 as baseline) | P0 | ~0–1 | Breadth the spec explicitly invites; zero retraining needed thanks to amortized Φ. |
| **P5** | AwA2 port (dataset strategy, features, labels, GT, eval) | P4 (generator) | ~1–2 | Generalization evidence; bounded scope, honest caveat pre-written. |
| **P6** | Notebook assembly: methodology math, ledger tables with CIs, learning curves, qualitative successes/failures, honest-calls section | everything | ~0 | Continuous — every P-item lands as a notebook section on completion, not in a final crunch. |

Budget: ~7–11 GPU-h allocated of ~38.7 remaining → **~28 GPU-h reserve** for re-runs and
surprises. If evidence forces triage, the cut order is P5 → extra P4 families → C5; P0–P3
are not cuttable — they carry the paper-grade claims.

### P1–P3 implementation status (2026-07-02, pending the VM run)

`src/tier5_cdbayes.py` implements the full matrix in one VM job (`python src/tier5_cdbayes.py`,
results bundled for `scripts/pull_tier5_cdbayes_results.ps1`; unit checks in
`test/test_tier5_cdbayes.py`). Each ablation row maps to the Part-II failure mode it tests:

| Ablation row | Components | Failure mode it tests / expected evidence |
|---|---|---|
| `A_tier4repro` | all off | parity anchor — must equal Tier-4 Stage A (hard invariant, asserted at startup) |
| `A_C1` | C1 | (ii) miscalibration — gains should concentrate on rare/imbalanced attrs (`+Mustache`, `-Young`) |
| `A_C1C2` | C1+C2 | (i) independence — the lipstick composite and `+Chubby,-Young` are the diagnostic queries |
| `A_C1C2C3` = A′ core | C1+C2+C3 | (i) 2nd order — negation identity (`-Male,-Mustache`) via conditioned tails |
| `Aprime_C4` | A′ core + λ·cos | R@1 recovery with a non-learned identity channel (λ TRAIN-tuned; λ=0 ⇒ C4 earns nothing) |
| `Aprime_C5` | A′ core + QB-Norm | (v) hubness — broad small gains rather than query-specific ones |
| `Aprime_C4C5` = full A′ | A′ core + C4 + C5 | the full training-free operating point |
| `Bprime_listnet` | Φ(ListNet) on A′ core | loss-change control (1 seed — direction, not variance) |
| `Bprime_lambdaloss` | Φ(LambdaLoss) on A′ core, 3 seeds | (iii) flat target mass — R@1 must recover vs Tier-4 Stage B |
| `Bprime_lambdaloss_noqv` | + `--no-qv`, 3 seeds | (iv) CLIP-channel color bias — hair/makeup queries must recover if the hypothesis holds |

Design notes locked into the script: B′ trains on the **A′ core** score (C1+C2+C3, no C4/C5)
so the q_v on/off comparison isolates the learned CLIP channel; seeds default to 3 on the
LambdaLoss rows and 1 on the ListNet control to stay ≤ 8 GPU-h (≈ 6.5–7.5 expected;
`--seeds 2` is the trim lever, `--full-seeds` the upgrade lever).

**Definition of done** (traces to the rubric, §4): one Colab notebook (§4) containing the
full tier ledger with intervals (empirical performance + thoroughness), the CD-Bayes
methodology with formulas and citations (originality + clarity), ablations and multi-seed
error bars (thoroughness), custom-query and AwA2 analyses with stated caveats (thoroughness),
and the honest-calls section (clarity — and credibility).

---

## References (beyond documents/SOTA.md)

- Chow, C. & Liu, C. (1968). *Approximating discrete probability distributions with
  dependence trees.* IEEE Trans. Information Theory. — the dependency-tree factorization (C2).
- Friedman, N., Geiger, D. & Goldszmidt, M. (1997). *Bayesian network classifiers.* Machine
  Learning. — TAN: the classifier lineage that relaxes naive-Bayes independence (C2).
- Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). *On calibration of modern neural
  networks.* ICML. https://proceedings.mlr.press/v70/guo17a.html — temperature scaling (C1).
- Bogolin, S.-V., Croitoru, I., Jin, H., Liu, Y. & Albanie, S. (2022). *Cross modal retrieval
  with querybank normalisation.* CVPR. https://arxiv.org/abs/2112.12777 — QB-Norm/DIS (C5).
- Wang, X., Li, C., Golbandi, N., Bendersky, M. & Najork, M. (2018). *The LambdaLoss
  framework for ranking metric optimization.* CIKM.
  https://dl.acm.org/doi/10.1145/3269206.3271784 — top-heavy listwise loss (Stage B′).
- Wilson, E. B. (1927). *Probable inference, the law of succession, and statistical
  inference.* JASA. — the score interval (IV.1).
- Cao, Z. et al. (2007). *Learning to rank: from pairwise approach to listwise approach.*
  ICML. — ListNet, the Stage-B loss being replaced.
