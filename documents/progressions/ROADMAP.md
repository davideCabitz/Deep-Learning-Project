# Roadmap — Dynamic & Hybrid Conditioning for Compositional Image Retrieval

## Context

This is a from-scratch Deep Learning course project (single Colab notebook deliverable) for a **team of 2, working for exactly 10 days (today included)**, on a university VM. The repo and a few files are scaffolded, but **no real work has been done yet** — everything below is still to build. The goal is to **excel** against the grading rubric: originality, methodological thoroughness, report clarity, empirical performance vs. a baseline, and code quality.

**The problem in one paragraph.** CLIP (Radford et al., 2021) maps images and text into a shared, L2-normalized embedding space (a unit hypersphere). CLAY (Lim et al., 2026) reframes that space into a *text-conditional similarity space* so you can retrieve images that match a textual condition while keeping the visual database frozen. CLAY's weakness — and the entire point of this assignment — is that when you give it **multiple** conditions, it just **stacks/concatenates the text embeddings and runs one SVD** (the "naïve pre-SVD bottleneck"). That offers no dynamic control over how conditions are weighted, and no principled way to express **negative** ("must NOT have X") constraints. Our job is to build a **fusion module Φ** that takes a reference image embedding `v_ref`, a set of positive text constraints `T+`, and negative text constraints `T−`, and produces a composite query embedding `q` that retrieves targets which **keep the identity of `v_ref`** while **satisfying all + / − modifications**.

**Concrete task setup (fixed by the spec):**
- **Model:** `openai/clip-vit-base-patch32` (CLIP ViT-B/32) from HuggingFace. Frozen.
- **Dataset:** CelebA test split (40 binary face attributes). Evaluate **only** on source-image indices given in `celeba_evaluation.json`.
- **Ground truth:** a target is valid iff it *strictly* satisfies the +/− constraints AND has Hamming distance ≤ 2 on the *other* attributes vs. the source (identity preservation). Only sources with ≥ 5 valid targets are included.
- **Metrics:** Recall@K (hit rate, primary) and Precision@K, at **K = 1, 5, 10**, averaged over valid sources.
- **12 mandatory queries** (7 simple, 5 composed) listed in the spec; the JSON is authoritative.
- **CRITICAL indexing gotcha:** GT keys are **PyTorch dataset indices**, not filenames. Always fetch via `celeba[idx]`, never `Image.open(".../000013.jpg")`.

**Chosen direction: HYBRID, scoped to 10 days for 2 people** — build strong training-free baselines *and* a lightweight trained fusion module, but with the trained module kept deliberately small and the training-free advanced variant (2a) treated as the guaranteed deliverable. CLIP stays frozen; we precompute features once and only train a small Φ. The original 3-person stretch ambitions (extra ablations, second Φ architecture) become explicit "only if time remains" items.

---

## Recommended technical approach

A single pipeline with **escalating method tiers**, all evaluated through one shared harness:

**Tier 0 — Vanilla zero-shot baseline (lower bound).**
`q = normalize( v_ref + Σ_i α·t⁺_i − Σ_j β·t⁻_j )`, score DB by cosine. Naïve latent arithmetic, no SVD, no learning. Establishes the floor and exercises the eval pipeline. (Note: because of CLIP's *modality gap*, adding raw text vectors to image vectors is geometrically crude — this motivates everything that follows.)

**Tier 1 — CLAY reproduction (the method-to-beat).**
Reproduce CLAY's manifold-aware textual subspace: log-map condition text features onto the tangent space at their mean, run SVD, keep top-k right singular vectors → projection `P_c = V_k V_kᵀ`; align the visual mean to the text mean with rotation `H(·)` (handles the modality gap / conic effect), project visual features, score by cosine. Multi-condition = **naïve stacking of all prompts into one SVD** — i.e., the exact bottleneck we are attacking. This is our principal comparison point.

**Tier 2 — Our contribution: the dynamic fusion module Φ.** Two complementary variants:
- **(2a) Training-free advanced [PRIMARY contribution, must ship]:** treat positives and negatives asymmetrically. Project the query onto the span of the **positive** condition subspace and **orthogonally reject** the **negative** directions (so "−red hair" means "remove/penalize the red-hair direction," with any other hair color acceptable). This already fixes CLAY's "+/− are just stacked" flaw with zero training, and is our insurance policy under the tight timeline.
- **(2b) Trained lightweight Φ [STRETCH within scope]:** a small module (recommended: **cross-attention** over tokens `[v_ref, t⁺_1…, t⁻_1…]` with a learned **sign/role embedding** marking each token as reference / positive / negative; *or* a simpler **gating/FiLM** network that predicts per-condition weights — pick **one**, do not build both). Output = composite query `q`. CLIP frozen; only Φ trains (target a few hundred K params). Trained with an **InfoNCE / triplet contrastive loss** where positives are valid targets and negatives are constraint-violating images.

**Why negatives are the gradeable insight:** "−X" is *negation* ("any value but X"), not vector subtraction. Subtracting `t_X` overshoots into "anti-X." The training-free rejection (2a) and the contrastive objective with violation-based negatives (2b) both model this correctly — make this distinction explicit in the report; it directly serves the *originality* and *thoroughness* criteria. **Because 2a alone already demonstrates this insight, the project is gradeable even if 2b is cut.**

**Training data (key engineering piece, only needed for 2b):** generate synthetic queries from the **CelebA *train* split only** (never touch test). For a sampled reference, assert some of its present attributes as `+` and some absent ones as `−`, then mine valid positive targets with the **same relaxed-Hamming-≤2 rule** as the eval protocol, and sample violating images as hard negatives. Training the model on the *exact* objective it's graded on is what will push performance above the baselines.

**Efficiency backbone (from CLAY):** extract and cache **all** CLIP image features for the corpus **once, offline**, store as a single tensor indexed by dataset index; retrieval = one matrix-vector cosine over the frozen DB. This is what makes daily full-corpus iteration feasible on a modest VM.

---

## 10-day schedule (2 people)

The plan is **front-loaded**: get the shared spine and a real number on the board by Day 3, lock the guaranteed contribution (2a) by mid-project, and treat the trained Φ (2b) as the upside that we time-box hard. Report writing is woven in continuously, not left to the end.

**Phase A — Shared foundation (Days 1–3).** Everyone depends on this; build it together, fast.
- Day 1 (both): agree on interfaces / data contracts (feature tensor layout, attribute tensor, query representation `(v_ref_idx, T+, T−)`, the `score(query) → ranking` signature). Confirm repo + Colab notebook skeleton, **verify VM specs** (GPU model, VRAM, disk). Download/confirm CelebA + `celeba_evaluation.json`. Write the **indexing assertion test** immediately.
- Days 2–3: deliver the spine in parallel — **A owns** the eval harness (metrics + GT parsing); **B owns** the frozen feature DB + Tier-0 baseline. **Milestone M1 (end Day 3): Tier-0 baseline produces real Recall@K/Precision@K numbers on the 12 queries.**

**Phase B — Method development (Days 4–8).**
- **Tier-1 CLAY reproduction** working and scored (**Milestone M2, end Day 5**). — owner B.
- **Tier-2a training-free advanced variant** implemented and scored (**Milestone M3, end Day 6**). This is the guaranteed contribution; once it lands, the project is safe.
- **Tier-2b trained Φ (time-boxed):** synthetic-query generator + Φ skeleton (Days 5–6, owner A), first end-to-end training run by Day 7, iterate on architecture/loss/hyperparameters with learning curves through Day 8 (**Milestone M4, end Day 8: trained Φ beats Tier-0 and is competitive with / beats Tier-1**). **Hard stop:** if 2b is not beating Tier-0 by end of Day 8, freeze it as a documented negative result and lean on 2a.

**Phase C — Experiments, ablations, report (Days 9–10).**
- Full comparative table (all shipped tiers × K∈{1,5,10} × 12 queries), the ablations that are cheap to run (k for SVD, α/β, with/without rotation H), qualitative success/failure retrieval grids.
- Notebook becomes a clean report: methodology + math, experimental setup, results & discussion. **Dry-run the full notebook top-to-bottom on Colab.** **Milestone M5 (end Day 10): submission-ready notebook.**

> **Cut list (drop in this order if behind):** second Φ architecture → manifold-aware log-map ablation → 2b trained Φ entirely (keep 2a) → extra qualitative grids. Never cut: the eval harness, Tier-0, Tier-1, Tier-2a, the indexing assertion, and the top-to-bottom Colab dry-run.

---

## 2-person workload division

The split is by **workstream ownership**. Each person owns one half of the spine in Phase A, then the heavier method work (Tier-1 + 2a, and the trained Φ) is shared across Phase B with clear primary owners. Both converge on experiments and the report in Phase C.

### Member A — Data, Evaluation & the Fusion Module Φ
*Owns the contract everyone is scored against, and the trained contribution.*
- CelebA loading via torchvision incl. the **index-vs-filename** handling; build the 40-dim attribute tensor indexed by dataset index.
- `celeba_evaluation.json` parser + **evaluation engine**: Recall@K, Precision@K at K=1,5,10, averaging over valid sources, source-self-exclusion handling.
- The **indexing assertion test** and a quick data-viability check (every JSON source has ≥ 5 GT targets).
- **Synthetic query generator** from the train split mirroring the relaxed-Hamming GT protocol; positive/hard-negative sampling (for 2b).
- **Trained Φ (2b):** one architecture (cross-attention with role/sign embeddings *or* FiLM), InfoNCE/triplet loss, optimizer, scheduler, checkpointing, learning curves.
- Qualitative visualization utilities (retrieval grids, success/failure cases).
- **Report:** owns **Methodology (math + forward pass + loss)** and **Results & Discussion**.

### Member B — Representation, Baselines & Reference Methods
*Owns CLIP, the frozen DB, and the methods-to-beat.*
- CLIP ViT-B/32 wrapper (HF); attribute → text-prompt templating.
- **Offline feature extraction & caching** for the full test corpus + the train subset needed for Φ; the frozen retrieval DB.
- **Modality-gap analysis** (justifies design choices; cite CLIP §shared space).
- **Tier-0** vanilla latent-arithmetic baseline.
- **Tier-1** CLAY reproduction (tangent/log-map, SVD subspace, rotation `H`, naïve multi-condition stacking).
- **Tier-2a** training-free rejection variant (the guaranteed contribution).
- **Report:** owns **Experimental setup** + CLAY/CLIP **background**, and the SVD/α-β/rotation ablations.

**Dependency management:** Member A's generator + Φ work needs only the *attribute tensor* (A's own) and the *feature extractor* (from B) — both Phase-A deliverables — so A is unblocked for 2b by end of Day 3. Member B's tier sequence (0→1→2a) is self-contained on the frozen DB. The two only hard-sync on the shared `score(query)→ranking` interface (Day 1) and at each milestone.

---

## Critical files / artifacts to create

- `notebook.ipynb` (the single deliverable; modular cells + markdown report).
- A small shared utilities module (mirrored into notebook cells): `data.py` (CelebA + indexing), `features.py` (CLIP extraction/caching), `eval.py` (metrics + GT parsing), `methods.py` (Tier 0/1/2a), `fusion.py` (Φ + training), `viz.py`.
- Cached artifacts on the VM: `clip_image_features_test.pt`, `clip_image_features_train.pt`, `attributes_test.pt`, `attributes_train.pt`.
- `celeba_evaluation.json` (downloaded; **authoritative** for queries + GT).

## Key risks & mitigations
- **Indexing bug** (filenames vs. dataset indices) → silently wrong scores. *Mitigation:* Member A writes an explicit assertion test (`celeba.filename[13] == "182651.jpg"`) on **Day 1**, before any scoring.
- **10 days is tight for 2 people.** *Mitigation:* 2a (training-free) is the guaranteed contribution; 2b is hard time-boxed to end of Day 8 with an explicit cut list; report is written continuously.
- **Modality gap** makes naïve arithmetic weak. *Mitigation:* expected — it's the motivation; the rotation `H` and trained Φ address it.
- **Φ overfits / doesn't beat Tier-1 in time.** *Mitigation:* Tier-2a is the guaranteed-improvement fallback; keep Φ lightweight; checkpoint early; freeze as documented negative result if it stalls.
- **VM specs unknown.** *Mitigation:* Day-1 spec check; the precompute-once design keeps us viable even on a single 16 GB GPU.
- **Scope creep.** *Mitigation:* AwA2/OVAD, SAE interpretability, and a second Φ architecture are explicitly *stretch goals only* and first on the cut list.

## Verification (end-to-end)
1. **Sanity:** `celeba[13]`'s filename equals `182651.jpg`; feature DB row count == test-split size; every JSON source index has ≥ 5 GT targets.
2. **Pipeline:** Tier-0 produces non-trivial Recall@K on all 12 queries (M1).
3. **Improvement:** Tier-1 and Tier-2a strictly beat Tier-0 on Recall@1/@5 (the rubric's empirical-performance bar); Tier-2b beats Tier-0 if it ships.
4. **Reproducibility:** restart-and-run-all on Colab completes top-to-bottom from cached features within session limits; fixed seeds; tables/plots regenerate.
5. **Coverage:** every mandatory query appears in the final comparison table at K = 1, 5, 10 for every shipped method tier.
