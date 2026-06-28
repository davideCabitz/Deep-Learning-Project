# Track S — Paper Notions Reference

> **Purpose.** Condensed reading notes for the five papers backing Track S
> (Asymmetric Conditional Subspaces). For each paper: summary → key findings →
> what / how / why → direct relevance to the implementation of `src/tier2a_subspace.py`.

---

## CLAY — Conditional Visual Similarity Modulation in Vision-Language Embedding Space

*Lim et al., KAIST, CVPR 2026*

**Short summary.**
CLAY reframes the fixed cosine-similarity space of a frozen VLM into a *text-conditional* similarity space. Given a user condition `c`, it builds a manifold-aware textual subspace from condition-related prompts and projects all image embeddings into it — no fine-tuning, no DB re-encoding.

**Key findings.**
- Symmetric formulation (both query and DB go through the projector) consistently outperforms asymmetric (query-only), despite the extra cost.
- Naive Euclidean SVD on text features ignores the hyperspherical geometry; log-mapping to the tangent space at the text-feature mean `µ_c` before SVD fixes this and measurably improves results.
- Pre-computing the projection matrix `P_c = V_k V_k^T` per condition decouples conditioning from DB re-encoding → inference is efficient even as condition count grows.
- A rotation `H` that aligns the mean of DB image features with `µ_c` is necessary to bridge the modality gap (cone effect) without distorting intra-DB relationships.

**What / How / Why.**
- *What:* A training-free conditional retrieval engine built entirely from SVD and rotation on frozen VLM features.
- *How:*
  1. Generate condition-related text prompts via LLM (`"a photo of {c}"`).
  2. Encode with text encoder → log-map at `µ_c` → stack as matrix → SVD → top-k right singular vectors `V_k`.
  3. Projection matrix `P_c = V_k V_k^T`.
  4. At inference: rotate DB features `H(v_d)` to align their mean with `µ_c` → log-map → project → cosine similarity in the subspace.
- *Why:* Static VLM similarity is blind to the current user condition; projecting into the condition's subspace makes only condition-relevant dimensions count.

**Relevance to Track S.**
- The manifold-aware subspace pipeline (log → SVD → projection matrix) is the *exact* construction Track S applies separately to positive and negative prompt sets to build `V_k⁺` and `V_k⁻`.
- The rotation `H` (aligning image-feature mean to text-feature mean) is the "rotation `H` ON/OFF" ablation row in Track S's evaluation table; CLAY demonstrates it matters for the modality gap.
- CLAY's symmetric formulation validates projecting *both* query and DB into the subspace rather than conditioning the query alone — Track S follows this.
- CLAY's `k` (subspace dimensionality) hyperparameter maps directly to `k⁺` / `k⁻` in Track S's ablation.

---

## SpaceVLM — Sub-Space Modeling of Negation in Vision-Language Models

*Ranjbar, Alhamoud & Ghassemi, arXiv 2511.13231, November 2025*

**Short summary.**
SpaceVLM proves (mathematically) that a single embedding vector cannot represent negation, then proposes a training-free scoring rule that treats negation as the intersection of two spherical caps: one around the affirmative concept and the complementary region around the negated concept.

**Key findings.**
- **Formal proof:** No unit vector `n` can separate cat-images from non-cat-images with a positive margin under dot-product scoring; negation is geometrically unrepresentable as a point.
- Modeling negation as `N(e_a) ∩ N^c(e_n)` (close to A, far from N) and using the central direction `d̂` of that region as the scoring vector improves MCQ accuracy by ~30% on average over prior methods.
- The threshold `t` (cosine-similarity cap radius) is robust in `[0.90, 0.95]`; no expensive per-dataset tuning needed.
- SpaceVLM is model-agnostic; it improves CLIP, SigLIP, AIMV2, ConCLIP, NegCLIP — even fine-tuned baselines — without touching weights.
- Zero-shot performance on affirmative queries is *unchanged* because the rule only activates when negation is detected.

**What / How / Why.**
- *What:* A training-free, inference-time scoring function that replaces `e_I ⊙ T(P)` with `e_I ⊙ d̂` for negated prompts.
- *How:*
  1. Parse `"A but not N"` → encode separately: `e_a = T(P_a)`, `e_n = T(P_n)`.
  2. Define spherical cap: `N(x) = {z : x⊙z ≥ t}`.
  3. Target region: `N(e_a) ∩ N^c(e_n)`.
  4. Central direction along the great-circle arc between `e_n` and `e_a`:
     ```
     d̂ = [sin(α + θ/2) / sin(θ)] · e_a  −  [sin(α − θ/2) / sin(θ)] · e_n
     ```
     where `α = arccos(t)`, `θ = arccos(e_a ⊙ e_n)`.
  5. Score: `s_neg(e_I, P) = e_I ⊙ normalize(d̂)`.
- *Why:* The feasible set "images of A that are not N" is a spherical region, not a point; using its centroid direction as the scoring vector is geometrically principled.

**Relevance to Track S.**
- SpaceVLM is the *direct theoretical parent* of Track S's negation step. The idea of "retrieve in `S⁺ ∩ (S⁻)^⊥`" is the subspace generalisation of SpaceVLM's spherical-cap intersection.
- The composite score `cos_{S⁺}(v_ref, v_d) − λ·‖proj_{S⁻}(v_d)‖` in Track S is the subspace analogue of SpaceVLM's `d̂` construction: both subtract a term proportional to proximity to the negated concept.
- SpaceVLM proves why subtraction alone (Tier-0 style) fails: the result is still a single point; using the complement of a *subspace* is the multi-dimensional extension that preserves all valid alternatives.
- The λ weight in Track S plays the role of SpaceVLM's threshold `t`: it controls how hard the negation penalty is applied.
- SpaceVLM's ablation on model sizes and backbones confirms the approach is backbone-agnostic, giving confidence that Track S will transfer across CLIP variants.

---

## Parts of Speech–Grounded Subspaces in Vision-Language Models

*Oldfield, Tzelepis, Panagakis, Nicolaou & Patras, NeurIPS 2023*

**Short summary.**
PoS-Grounded Subspaces disentangles different visual modes of variation (objects, appearances) inside CLIP's shared space by learning subspaces in which one part-of-speech (noun, adjective, verb…) has maximum variance while all others are suppressed. The solution is in closed form and extends to the hypersphere via tangent-space PGA.

**Key findings.**
- Noun subspace isolates object content; adjective subspace isolates visual appearance — projecting onto their orthogonal complements selectively removes that visual mode.
- The joint objective `C_i = (1−λ)X_i X_i^T − Σ_{j≠i} λ X_j X_j^T` is a trace-maximisation problem with a one-shot closed-form solution (leading eigenvectors of `C_i`).
- Manifold-aware version (tangent space at intrinsic mean `µ`) outperforms Euclidean PCA on 14/15 zero-shot classification benchmarks.
- Projection onto the orthogonal complement `Π_i^⊥(z) = Exp_µ((I_d − Ŵ_i Ŵ_i^T) Log_µ(z))` reliably removes entire visual themes from CLIP-based generators.
- `λ = 0.5` balances preserving target-class variance and suppressing nuisance-class variance across all tested CLIP architectures.

**What / How / Why.**
- *What:* A closed-form component analysis on the CLIP hypersphere that learns subspaces capturing variation unique to a target word class.
- *How:*
  1. Collect word embeddings for each POS from WordNet → map to tangent space at global `µ`.
  2. Build `C_i` as positive loading on target class minus negative loading on all other classes (scaled by `λ`).
  3. Eigen-decompose `C_i` → top-k eigenvectors `Ŵ_i` span the subspace.
  4. Project any embedding: log-map → multiply by `Ŵ_i Ŵ_i^T` → exp-map (or apply complement `(I_d − Ŵ_i Ŵ_i^T)` to remove that mode).
- *Why:* A single CLIP embedding entangles multiple visual modes; isolating them in closed-form subspaces enables disentangled retrieval and generation without fine-tuning.

**Relevance to Track S.**
- The closed-form eigenvector construction of `C_i` is the *blueprint* for building `V_k⁺` (leading eigenvectors of the positive-prompt covariance in tangent space) and `V_k⁻` (same for negative prompts).
- The orthogonal complement projection `(I_d − Ŵ_i Ŵ_i^T)` is exactly the "negation by complement" operator in Track S: projecting a DB image into `(S⁻)^⊥` removes its component that is explained by the negative-attribute subspace.
- The `λ` trade-off (maximise target variance, suppress nuisance variance) is directly applicable to Track S's subspace construction: building `V_k⁺` should suppress negative-prompt variance and vice versa, giving cleaner, more asymmetric subspaces.
- The manifold-aware tangent-space treatment (log → eigen → exp) is the geometry-correct procedure that Track S inherits from both CLAY and this paper.
- The per-condition subspace variant in Track S (one subspace per condition, intersect rather than stack) mirrors the per-POS subspace philosophy of this paper.

---

## VLMs Do Not Understand Negation

*Alhamoud, Alshammari, Tian, Li, Torr, Kim & Ghassemi (MIT / Oxford / OpenAI), CVPR 2025*

**Short summary.**
NegBench is the systematic benchmark for negation understanding in joint-embedding VLMs. The paper shows that CLIP-family models collapse affirmative and negated captions to nearly identical embeddings (*affirmation bias*), then demonstrates that fine-tuning on large-scale synthetic negation data can partially close the gap.

**Key findings.**
- CLIP maps "a dog" and "no dog" to almost identical embeddings; most models perform near chance (25%) on 4-way negation MCQ.
- Scaling model size (ViT-B → L → H) does *not* improve negation understanding.
- Fine-tuning on CC12M-NegCap: +10% recall, +8% MCQ. Adding CC12M-NegMCQ: +28% MCQ boost.
- ConCLIP collapses *all* negated embeddings into a single point — a degenerate failure mode.
- NegCLIP improves composition generally but suffers a 23% recall drop on hard negatives relative to its own affirmative performance.
- The affirmation bias stems from CLIP's pretraining: affirmative captions dominate; contrastive learning never sees paired (positive / negated) examples with the same content.

**What / How / Why.**
- *What:* NegBench — 79K examples, 18 task variations (image/video/medical), two task families (Retrieval-Neg, MCQ-Neg).
- *How:*
  - Retrieval-Neg: retrieve top-5 images matching a mixed affirmative + negated query.
  - MCQ-Neg: choose the correct caption (Affirmation / Negation / Hybrid templates) from 4 options; hard negatives differ only in what is affirmed or negated.
  - Fine-tuning pipeline: contrastive loss `L_CLIP` on CC12M-NegCap + MCQ cross-entropy loss `L_MCQ` on CC12M-NegMCQ, combined as `αL_CLIP + (1−α)L_MCQ`.
- *Why:* To quantify a fundamental gap and provide the community with a realistic benchmark; fine-tuning is shown as a partial remedy while the geometry of the problem is left open.

**Relevance to Track S.**
- NegBench is the *motivation* for Track S: the zero-valued negation-query metrics (`-Male, -Mustache` → 0.000 R@5) in the project are the exact affirmation-bias failure NegBench documents.
- Track S must demonstrate that negation queries no longer collapse — this paper's MCQ Negation accuracy metric is the right diagnostic to borrow.
- The finding that fine-tuning alone cannot fully close the gap (and that ConCLIP collapses) justifies the training-free subspace approach: a structured geometric operation avoids the collapse failure mode.
- The Hybrid template (e.g., "includes A but not B") is the query format closest to Track S's multi-condition retrieval scenario; performance on Hybrid MCQ should be the primary Track S check.
- The paper implicitly encodes what "correct" negation behaviour looks like: retrieved images should satisfy all positive attributes and violate *none* of the negative ones — this is the definition Track S's test queries should verify.

---

## When Negation Is a Geometry Problem in Vision-Language Models

*Sammani, Chamiti, Gavrikov & Deligiannis (VUB / imec), arXiv 2603.20554, April 2026*

**Short summary.**
Rather than fine-tuning, this paper asks whether a *negation direction* already exists inside CLIP's text-encoder hidden states and whether test-time *representation steering* along that direction can activate negation-aware behaviour. It also introduces an MLLM-as-judge evaluation protocol to correct false-negative bias in standard retrieval metrics.

**Key findings.**
- A linear binary classifier trained on 4K (affirmative, negated) caption-pair hidden states achieves ≥99% test accuracy at layer 4 of the CLIP text encoder across all tested backbones (ViT-B/32, B/16, L/14) → a negation direction provably exists in the space.
- Negation information peaks in *intermediate* layers, not early or late ones.
- Steering: `h^l = (1−α) h^l + α · W^l_dir · ‖h^l‖` (applied to all layers) outperforms fine-tuning baselines on the controlled SimpleNeg dataset using only 4K training samples vs. 12M for CC12M approaches.
- On the generalisation benchmark N-COCO (uncommon scenes, 25K images), steering achieves R@1 = 0.80 vs. 0.60 for base CLIP and 0.50 for NegCLIP / CLIP-CC12M — showing fine-tuned models *overfit* to common scene statistics.
- MLLM-as-judge (Qwen3-VL) reveals that most fine-tuned baselines (except NegCLIP) barely improve over vanilla CLIP once false negatives are accounted for — standard Recall@K metrics are misleading.

**What / How / Why.**
- *What:* A representation-engineering approach to negation, plus a more reliable MLLM-based evaluation framework.
- *How:*
  1. Extract `<eos>` hidden states from layer `l` of the CLIP text encoder for 4K affirmative + 4K negated captions.
  2. Train an L-BFGS linear classifier (no bias) → weights `W^l` define the negation direction.
  3. Steer at inference: shift `h^l` toward `W^l_dir` with strength `α = 0.13`, preserving norm.
  4. Repeat for all layers 1…L.
  5. Evaluate with MLLM judge asking two sequential yes/no questions: (a) is the image contextually correct? (b) is the negated object absent?
- *Why:* The geometric structure of CLIP's hidden space already encodes negation; activation is a matter of amplification, not re-learning.

**Relevance to Track S.**
- **Core geometric confirmation:** The existence of a clean linear negation direction at layer 4 confirms that CLIP's space *already* has the structure needed for subspace-based negation — Track S's complement projection is a data-driven method to locate and exploit that same structure.
- The fact that the negation direction is *linearly* separable from affirmative representations is the empirical basis for expecting that an SVD-derived negative subspace `S⁻` will also be linearly separable from `S⁺`.
- The MLLM-as-judge protocol is a superior evaluation strategy to add alongside Recall@K for Track S's final write-up — it removes false-negative contamination and tests genuine negation satisfaction.
- The N-COCO generalisation finding warns against over-fitting to specific query templates: Track S's per-condition subspaces should be evaluated on diverse negation phrasings, not just the 14 canonical project queries.
- The steering parameter `α = 0.13` offers an intuition for Track S's λ: too aggressive negation suppression collapses semantics; a mild penalty achieves the best balance.

---

## Cross-Paper Synthesis for Track S Implementation

| Concept | Source papers | How it maps to `tier2a_subspace.py` |
|---|---|---|
| Manifold-aware subspace (log → SVD → exp) | CLAY, PoS-Subspaces | Build `V_k⁺` and `V_k⁻` in tangent space at `µ` — same code as `_log_map` / `_align_rotation` in `tier1.py` |
| Asymmetric polarity subspaces | CLAY (condition-specific) + PoS-Subspaces | Separate SVD for positive prompts vs. negative prompts — eliminates the naïve pre-SVD merge that breaks CLAY |
| Negation as orthogonal complement | SpaceVLM (spherical cap complement), PoS-Subspaces (`Π_i^⊥`) | Score = `cos_{S⁺}(v_ref, v_d) − λ · ‖proj_{S⁻}(v_d)‖` |
| Linear negation direction exists | When-Negation (steering paper) | Confirms `S⁻` is geometrically meaningful even with simple SVD |
| Affirmation bias to overcome | VLMs-Do-Not-Understand-Negation | Defines the concrete failure (0.000 R@5 on `-Male`) Track S must fix |
| Rotation H for modality gap | CLAY | Optional ablation: rotate DB image means to `µ_{text}` before projecting |
| λ (negation penalty weight) | SpaceVLM (`t` threshold), PoS-Subspaces (`λ`), steering (`α`) | Single ablation knob; all papers suggest mild values (0.1–0.5 range) |
| Per-condition vs. stacked subspace | PoS-Subspaces (per-POS) | Headline variant: one SVD per condition + intersection; stacked baseline for ablation |
| Evaluation protocol | VLMs-Do-Not-Understand-Negation (MCQ), When-Negation (MLLM judge) | Use Recall@{1,5,10} + MCQ Negation accuracy; consider MLLM judge for rigour |
