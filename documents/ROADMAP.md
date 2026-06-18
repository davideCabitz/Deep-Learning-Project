# Roadmap — Dynamic & Hybrid Conditioning for Compositional Image Retrieval

## Context

This is a from-scratch Deep Learning course project (single Colab notebook deliverable) for a **team of 3, working full-time for exactly 2 weeks**, on a university VM. The goal is to **excel** against the grading rubric: originality, methodological thoroughness, report clarity, empirical performance vs. a baseline, and code quality.

**The problem in one paragraph.** CLIP (Radford et al., 2021) maps images and text into a shared, L2-normalized embedding space (a unit hypersphere). CLAY (Lim et al., 2026) reframes that space into a *text-conditional similarity space* so you can retrieve images that match a textual condition while keeping the visual database frozen. CLAY's weakness — and the entire point of this assignment — is that when you give it **multiple** conditions, it just **stacks/concatenates the text embeddings and runs one SVD** (the "naïve pre-SVD bottleneck"). That offers no dynamic control over how conditions are weighted, and no principled way to express **negative** ("must NOT have X") constraints. Our job is to build a **fusion module Φ** that takes a reference image embedding `v_ref`, a set of positive text constraints `T+`, and negative text constraints `T−`, and produces a composite query embedding `q` that retrieves targets which **keep the identity of `v_ref`** while **satisfying all + / − modifications**.

**Concrete task setup (fixed by the spec):**
- **Model:** `openai/clip-vit-base-patch32` (CLIP ViT-B/32) from HuggingFace. Frozen.
- **Dataset:** CelebA test split (40 binary face attributes). Evaluate **only** on source-image indices given in `celeba_evaluation.json`.
- **Ground truth:** a target is valid iff it *strictly* satisfies the +/− constraints AND has Hamming distance ≤ 2 on the *other* attributes vs. the source (identity preservation). Only sources with ≥ 5 valid targets are included.
- **Metrics:** Recall@K (hit rate, primary) and Precision@K, at **K = 1, 5, 10**, averaged over valid sources.
- **12 mandatory queries** (7 simple, 5 composed) listed in the spec; the JSON is authoritative.
- **CRITICAL indexing gotcha:** GT keys are **PyTorch dataset indices**, not filenames. Always fetch via `celeba[idx]`, never `Image.open(".../000013.jpg")`.

**Chosen direction: HYBRID** — build strong training-free baselines *and* a lightweight trained fusion module. This is the best fit because (a) the spec explicitly wants a baseline to beat, (b) it cleanly fills three people's time, and (c) it maximizes the originality + thoroughness scores. CLIP stays frozen; we precompute features once and only train a small Φ.

---

## Recommended technical approach

A single pipeline with **escalating method tiers**, all evaluated through one shared harness:

**Tier 0 — Vanilla zero-shot baseline (lower bound).**
`q = normalize( v_ref + Σ_i α·t⁺_i − Σ_j β·t⁻_j )`, score DB by cosine. Naïve latent arithmetic, no SVD, no learning. Establishes the floor and exercises the eval pipeline. (Note: because of CLIP's *modality gap*, adding raw text vectors to image vectors is geometrically crude — this motivates everything that follows.)

**Tier 1 — CLAY reproduction (the method-to-beat).**
Reproduce CLAY's manifold-aware textual subspace: log-map condition text features onto the tangent space at their mean, run SVD, keep top-k right singular vectors → projection `P_c = V_k V_kᵀ`; align the visual mean to the text mean with rotation `H(·)` (handles the modality gap / conic effect), project visual features, score by cosine. Multi-condition = **naïve stacking of all prompts into one SVD** — i.e., the exact bottleneck we are attacking. This is our principal comparison point.

**Tier 2 — Our contribution: the dynamic fusion module Φ.** Two complementary variants:
- **(2a) Training-free advanced:** treat positives and negatives asymmetrically. Project the query onto the span of the **positive** condition subspace and **orthogonally reject** the **negative** directions (so "−red hair" means "remove/penalize the red-hair direction," with any other hair color acceptable). This already fixes CLAY's "+/− are just stacked" flaw with zero training.
- **(2b) Trained lightweight Φ (the star):** a small module (recommended: **cross-attention** over tokens `[v_ref, t⁺_1…, t⁻_1…]` with a learned **sign/role embedding** marking each token as reference / positive / negative; *or* a **gating/FiLM** network that predicts per-condition weights). Output = composite query `q`. CLIP frozen; only Φ trains (target a few hundred K params). Trained with an **InfoNCE / triplet contrastive loss** where positives are valid targets and negatives are constraint-violating images.

**Why negatives are the gradeable insight:** "−X" is *negation* ("any value but X"), not vector subtraction. Subtracting `t_X` overshoots into "anti-X." The training-free rejection (2a) and the contrastive objective with violation-based negatives (2b) both model this correctly — make this distinction explicit in the report; it directly serves the *originality* and *thoroughness* criteria.

**Training data (key engineering piece):** generate synthetic queries from the **CelebA *train* split only** (never touch test). For a sampled reference, assert some of its present attributes as `+` and some absent ones as `−`, then mine valid positive targets with the **same relaxed-Hamming-≤2 rule** as the eval protocol, and sample violating images as hard negatives. Training the model on the *exact* objective it's graded on is what will push performance above the baselines.

**Efficiency backbone (from CLAY):** extract and cache **all** CLIP image features for the corpus **once, offline**, store as a single tensor indexed by dataset index; retrieval = one matrix-vector cosine over the frozen DB. This is what makes daily full-corpus iteration feasible on a modest VM.

---

## Two-week schedule (3 people, full-time)

**Phase A — Shared foundation (Days 1–3).** Everyone depends on this; build it together, fast.
- Day 1 (all-hands): agree on interfaces / data contracts (feature tensor layout, attribute tensor, query representation `(v_ref_idx, T+, T−)`, the `score(query) → ranking` signature). Set up the repo, the Colab notebook skeleton, the VM, and **verify VM specs** (GPU model, VRAM, disk). Download CelebA + `celeba_evaluation.json`.
- Days 2–3: deliver the three spine components in parallel (see ownership below): working **eval harness**, working **frozen feature DB + Tier-0 baseline**, and the **synthetic-query generator + Φ skeleton**. **Milestone M1 (end Day 3): Tier-0 baseline produces real Recall@K/Precision@K numbers on the 12 queries.**

**Phase B — Method development (Days 4–10).**
- Tier-1 CLAY reproduction working and scored (**Milestone M2, ~Day 6**).
- Tier-2a training-free advanced variant scored.
- Tier-2b trained Φ: first end-to-end training run, then iterate on architecture/loss/hyperparameters with learning curves (**Milestone M3, ~Day 10: trained Φ beats Tier-0 and is competitive with / beats Tier-1**).

**Phase C — Experiments, ablations, report (Days 11–14).**
- Full comparative table (all tiers × K∈{1,5,10} × 12 queries), ablations (k for SVD, α/β, Φ design choices, with/without rotation H, with/without manifold-aware log-map), qualitative success/failure retrieval grids, custom queries.
- Notebook becomes a clean report: methodology + math, experimental setup, results & discussion. Dry-run the full notebook top-to-bottom on Colab. **Milestone M4 (Day 14): submission-ready notebook.**

---

## Balanced 3-person workload division

The split is by **workstream ownership**, with deliberate load-balancing: the foundation owners (M1, M2) front-load in Phase A, then absorb extra experiment/ablation/report work in Phase C to offset the method owner (M3), whose component is heaviest and on the critical-risk path in Phase B.

### Member 1 — Data & Evaluation Infrastructure ("the harness")
*Owns the contract everyone is scored against.*
- CelebA loading via torchvision incl. the **index-vs-filename** handling; build the 40-dim attribute tensor indexed by dataset index.
- Data exploration: attribute frequencies, co-occurrence, and **validation that queries have viable targets** in the corpus.
- `celeba_evaluation.json` parser + **evaluation engine**: Recall@K, Precision@K at K=1,5,10, averaging over valid sources, source-self-exclusion handling.
- Qualitative visualization utilities (retrieval grids, success/failure cases).
- **Phase C:** run all comparative experiments through the harness, build final tables + charts, own the **Results & Discussion** section.

### Member 2 — Representation & Baselines ("foundation + method-to-beat")
*Owns CLIP, the frozen DB, and the reference methods.*
- CLIP ViT-B/32 wrapper (HF); attribute → text-prompt templating.
- **Offline feature extraction & caching** for the full test corpus + the train subset needed for Φ; the frozen retrieval DB.
- **Modality-gap analysis** (justifies design choices; cite CLIP §shared space).
- **Tier-0** vanilla latent-arithmetic baseline.
- **Tier-1** CLAY reproduction (tangent/log-map, SVD subspace, rotation `H`, naïve multi-condition stacking).
- **Phase C:** the **Tier-2a** training-free rejection variant as an ablation; own **Experimental-setup** + CLAY/CLIP **background** in the report.

### Member 3 — Novel Fusion Module Φ ("the contribution")
*Owns the originality surface.*
- Design Φ (recommend cross-attention with role/sign embeddings, with a gating/FiLM variant as ablation).
- **Synthetic query generator** from the train split mirroring the relaxed-Hamming GT protocol; positive/hard-negative sampling.
- **Training pipeline:** InfoNCE/triplet loss, optimizer, scheduler, checkpointing, learning curves.
- Architecture + hyperparameter tuning; ablations on Φ.
- **Phase C:** own the **Methodology (math + forward pass + loss)** section and the Φ ablations.

**Dependency management:** M3 needs only the *attribute tensor* (from M1) and the *feature extractor* (from M2) to start the generator by end of Day 2–3 — both are Phase-A deliverables, so M3 is unblocked early. All three converge on shared experiments in Phase C.

---

## Critical files / artifacts to create

- `notebook.ipynb` (the single deliverable; modular cells + markdown report).
- A small shared utilities module (mirrored into notebook cells): `data.py` (CelebA + indexing), `features.py` (CLIP extraction/caching), `eval.py` (metrics + GT parsing), `methods.py` (Tier 0/1/2a), `fusion.py` (Φ + training), `viz.py`.
- Cached artifacts on the VM: `clip_image_features_test.pt`, `clip_image_features_train.pt`, `attributes_test.pt`, `attributes_train.pt`.
- `celeba_evaluation.json` (downloaded; **authoritative** for queries + GT).

## Key risks & mitigations
- **Indexing bug** (filenames vs. dataset indices) → silently wrong scores. *Mitigation:* M1 writes an explicit assertion test (`celeba.filename[13] == "182651.jpg"`) before any scoring.
- **Modality gap** makes naïve arithmetic weak. *Mitigation:* expected — it's the motivation; the rotation `H` and trained Φ address it.
- **Φ overfits / doesn't beat Tier-1 in time.** *Mitigation:* Tier-2a (training-free) is a guaranteed-improvement fallback; keep Φ lightweight; checkpoint early.
- **VM specs unknown.** *Mitigation:* Day-1 spec check; the precompute-once design keeps us viable even on a single 16 GB GPU.
- **2-week scope creep.** *Mitigation:* AwA2/OVAD and SAE interpretability are explicitly *stretch goals only*.

## Verification (end-to-end)
1. **Sanity:** `celeba[13]`'s filename equals `182651.jpg`; feature DB row count == test-split size; every JSON source index has ≥ 5 GT targets.
2. **Pipeline:** Tier-0 produces non-trivial Recall@K on all 12 queries (M1).
3. **Improvement:** Tier-1 and Tier-2 strictly beat Tier-0 on Recall@1/@5 (the rubric's empirical-performance bar).
4. **Reproducibility:** restart-and-run-all on Colab completes top-to-bottom from cached features within session limits; fixed seeds; tables/plots regenerate.
5. **Coverage:** every mandatory query appears in the final comparison table at K = 1, 5, 10 for every method tier.
