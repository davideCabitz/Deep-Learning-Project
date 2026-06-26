# Alfonso — Progress Log

Tracking the data-half work (sections 1, 2, 4 of `CONTRACT.md`): loading CelebA,
building the attribute tensor, parsing the ground-truth JSON, and sanity-checking indices.

---

## Phase A — Data Preparation

### `project/data_loader.py` (frozen DB + shared loaders)

Single source of truth for the data half. Built once, never modified. Both the data
and model halves import from here.

**Constants**
- `ATTRIBUTE_NAMES` — the 40 attribute names in fixed order (per CONTRACT.md §1).
- `ATTR_TO_IDX` — name → column index (e.g. `"Smiling"` → 31).

**Functions**
- `setup_frozen_db(force=False)` — builds and saves the attribute tensor once.
  Reads `list_attr_celeba.txt`, filters to the test split, converts `-1/+1 → 0.0/1.0`,
  saves as `float32`. Skips if the file already exists (unless `force=True`).
- `load_attributes()` — loads the frozen tensor `[N, 40]`, `float32`, values in `{0.0, 1.0}`.
- `load_celeba_dataset()` — loads the CelebA test split with CLIP preprocessing
  (resize 224×224, ToTensor, CLIP normalization).

**Internals**
- `_get_project_root()` — project root via `__file__`.
- `_get_celeba_root()` — auto-detects CelebA location (project root or `~/Downloads/celeba`).

**One-time setup**
```bash
python project/data_loader.py
```

**Output artifact:** `artifacts/celeba_attributes_test.pt` (frozen, do not touch).

> **Path change (this session):** the frozen DB now lives in `artifacts/`, not the
> project root. `setup_frozen_db()` / `load_attributes()` read/write there via the new
> `_get_artifacts_dir()` helper (creates the dir if missing). This is also where Member B's
> `clip_image_features_*.pt` will go. Also replaced two `✓` chars in print statements with
> `[OK]` so the loader runs in a Windows console / headless `nbconvert`.

---

### `src/sanity.py` (Day-1 CRITICAL assertions)

The checks everything downstream depends on, written as **asserting** functions (not prints):
a broken assumption fails loudly here, before any retrieval method produces a silently-wrong
score. Replaces the old notebook Step 3, which only *printed* values and "passed" regardless.

**Functions**
- `assert_indexing(celeba)` — `celeba.filename[13] == "182651.jpg"` and `N == 19962`
  (CONTRACT.md §0). The indexing tripwire: catches a reshuffled split or a filename-based lookup.
- `assert_attributes(attributes, celeba=...)` — tensor is `[19962, 40]` `float32`, values ⊆ `{0.0, 1.0}`,
  row-aligned to the dataset (CONTRACT.md §2).
- `assert_gt_viability(eval_json)` — every source has ≥ 5 valid targets, all source/target indices
  in range, no source in its own target list (ROADMAP.md protocol + CONTRACT.md §5).
- `run_all_checks(verbose=True)` — the single entry point: runs all three, raises on the first
  failure, else returns a summary dict.

**Locked-in expected values** (verified against the real test split): `EXPECTED_FILENAME_AT_13 =
"182651.jpg"`, `EXPECTED_N_IMAGES = 19962`, `EXPECTED_N_ATTRS = 40`, `MIN_TARGETS_PER_SOURCE = 5`.

**Run standalone**
```bash
python src/sanity.py
```
Verified green against real data: `{n_images: 19962, n_attributes: 40, n_queries: 14, n_sources: 33052}`.

---

### `project/Phase_A_Data_Preparation.ipynb`

Loads and verifies the data contract. Thin notebook — all heavy lifting is in `data_loader.py`.

**Step 1 — Load CelebA dataset & attributes**
- Bootstraps `src/` onto `sys.path` by walking up to the project root (first ancestor
  containing both `src/` and `Evaluation/`), so the notebook imports cleanly **regardless of
  where Jupyter is launched**. (The old `notebook_dir.name == 'project'` logic was stale —
  the folder is `notebooks` — and silently mis-set the root.)
- Imports `load_celeba_dataset`, `load_attributes`, `setup_frozen_db` from `data_loader`.
- Calls `setup_frozen_db()` (no-op if already built), then loads the dataset and tensor.
- Verifies dataset size (~19,962 test images), attribute shape `[19962, 40]`, dtype `float32`.

**Step 2 — Load evaluation JSON**
- Reads `Evaluation/celeba_evaluation.json` (the authoritative ground-truth file).
- Prints per-query summary: query string, number of sources, avg targets per source.
- 14 queries total. **Resolved (CONTRACT.md §4):** the spec text says 12 but the JSON is
  authoritative, so we evaluate all 14 as-is (`-Young` listed twice + two composed queries
  differ); just note the 14-vs-12 mismatch in one sentence in the report. No code action.

**Step 3 — Sanity check (CRITICAL)**
- Now calls `run_all_checks()` from `sanity` — hard assertions that *raise* on any
  inconsistency, instead of the old prints that "passed" regardless. See `src/sanity.py`.
- Verified to run the full notebook top-to-bottom from a cold terminal (path bootstrap +
  ASCII output), satisfying the roadmap's restart-and-run-all requirement.

---

---

## Phase B — Baseline (Tier-0) + CLAY prerequisite (this session, 2026-06-26)

### Tier-0 — vanilla latent arithmetic (`src/tier0.py`)

The "method to beat" (spec §3.2.3): `query = v_ref + alpha * (Σ t(+attr) − Σ t(−attr))`,
ranked by cosine over the frozen CLIP image table. Plugs into the shared eval engine via
`make_get_ranking`. Run: `python src/tier0.py` → writes `output/tier0_alpha{alpha}.csv`.

**Result at alpha=1.0 (the literal "vanilla" from the spec):** MEAN R@1=0.0224,
R@5=0.0699, R@10=0.1048. Best single queries are concrete additive attributes
(`+Mustache` R@1=0.080, `+Smiling` R@1=0.063); negations and global attributes collapse
(`-Male,-Mustache` and `-Smiling,+Eyeglasses,+Wearing_Hat` → all zeros).

### What we demonstrated about the baseline (why the numbers are low — and that it's NOT a bug)

The low scores were investigated, not assumed. **Conclusion: the harness is correct; the
numbers are a genuine property of the method × benchmark.** Three pieces of evidence:

1. **alpha=0 sanity check (pure image→image retrieval, no text).** Ran a self-contained
   driver (loads cached `.pt`, reuses `eval.py`; avoids the `transformers` import that
   `clip_features.py` triggers — `transformers` is not installed locally). Result:
   MEAN **R@1=0.0217, R@5=0.0496, R@10=0.0727** — about the *same* as alpha=1.0, and
   slightly *worse* at R@10.
   - Interpretation: the GT requires a target to **both** satisfy the query constraints
     **and** sit within Hamming ≤2 of the source (spec §3.1.1). alpha=0 ignores the text,
     so it retrieves visual look-alikes that often keep the source's *original* attribute
     value and thus violate the query. alpha=1.0 does the opposite — the unit text vector
     plus the CLIP modality gap yanks the query into the text cone and washes out identity.
   - **Key insight for the report:** *neither endpoint wins.* "Ignore the text" (α=0) and
     "drown the identity" (α=1.0) both fail, for opposite reasons. That is exactly the
     fusion problem the project asks us to solve — and it's why a real Tier-1/Tier-2 method
     is needed, not just a better α. This makes Tier-0 a legitimate, honest lower bound.

2. **Image table is well-formed.** `clip_image_features_test.pt` is `[19962, 512]` float32,
   every row L2-norm = 1.0000.

3. **Self-retrieval works.** Row i is its own nearest neighbor at cos=1.0; true neighbors
   sit ~0.91, a random pair ~0.50 — sensible spread, and the source-exclusion + argsort
   mechanics behave correctly.

**Decision:** do NOT tune Tier-0 to inflate the baseline — that would raise the bar our own
method must clear and shrink the reported improvement. `alpha` is a documented hyperparameter
of the vanilla method, so an α-sweep is a legitimate *ablation* (optional, for the report),
but the headline baseline stays at the spec's vanilla α=1.0. The α=0 run is a sanity check,
not a baseline entry.

### CLAY prerequisite — per-attribute prompt bank (`src/clip_prompts.py`)

Built the raw material CLAY's subspace SVD needs. Tier-0 uses ONE text vector per attribute;
CLAY needs a *stack* of many same-meaning prompts per condition so SVD recovers a non-trivial
subspace (CLAY.md §3.2). With n=1 the stack is rank-1 and every CLAY accuracy lever
(adaptive-k, all-but-the-top, σ-weighting) has nothing to operate on.

- **Approach:** generate the n prompts from deterministic TEMPLATES (sentence FRAMES ×
  per-attribute synonym phrases) instead of an LLM (CLAY uses ChatGPT-5). Reproducible,
  from-scratch, course-policy compliant; the downstream geometry is identical — only the
  *source* of the sentences differs. Stated as a methodology choice in the report.
- **Output artifact:** `artifacts/clip_attr_prompt_bank.pt`, shape `[40, n, 512]` float32,
  each row L2-normalized, `bank[j]` == prompt stack for `ATTRIBUTE_NAMES[j]`. Padded to a
  common n with duplicates of the first prompt (a repeated row adds no new SVD direction).
- **Verified:** complete 40-attribute coverage (`_verify_coverage`), correct shape, n≥2,
  unit-normalized rows. Run: `python src/clip_prompts.py` (also has a Colab notebook path,
  since `transformers` isn't installed locally).
- **Note:** this `clip_attr_prompt_bank.pt` (the multi-prompt stack for CLAY) is distinct
  from `clip_attr_text_features.pt` (the single per-attribute vector Tier-0 uses).

### Ablation — Tier-0 on the prompt bank (prompt ensembling)

Ran Tier-0 unchanged but swapped `t(attr)`: instead of the single-prompt vector, use the
**mean of that attribute's n prompts** from `clip_attr_prompt_bank.pt`, re-normalized to the
unit sphere. Same α=1.0, same image table, same scoring/eval. Output:
`output/tier0_promptbank_alpha1.0.csv` (vs single-prompt `output/tier0_alpha1.0.csv`).

| Metric | Single-prompt | Prompt-bank (mean) | Δ |
|---|---|---|---|
| MEAN R@1  | 0.0224 | **0.0243** | +0.0019 |
| MEAN R@5  | 0.0699 | **0.0715** | +0.0016 |
| MEAN R@10 | 0.1048 | **0.1050** | +0.0002 |

**Why it's better (mechanism):** this is classic **CLIP prompt ensembling** (the standard
80-template ImageNet trick). A single text embedding mixes the *concept* with
*phrasing-specific noise* (wording/structure/token quirks). Averaging many same-meaning,
differently-worded prompts lets the shared concept add coherently while the per-phrasing
noise points in random directions and partially **cancels**; after re-normalizing, the
direction points more purely at the concept. So ranking improves slightly.

**Why the gain is small / not uniform:** the mean only denoises the *text term*. It does
**nothing** about the modality gap or the identity-vs-text trade-off — the actual structural
limits diagnosed above. So concrete single attributes improve most (`+Male` R@10
0.019→0.027, `+Eyeglasses,+Smiling` R@10 0.069→0.074), dead composite-negations stay dead
(`-Male,-Mustache` = 0.0000 in both), and a few rows are flat or fractionally *worse*
(`+Mustache` R@1 0.0797→0.0698). It wins **in aggregate**, not on every query; the +0.0002
at R@10 is effectively a tie.

**Status:** this is an **ablation row, not a baseline replacement.** `output/tier0_alpha1.0.csv`
(spec's vanilla single prompt) remains *the* official Tier-0 baseline. Takeaway for the
report: richer prompts are a real but minor lever inside latent arithmetic; closing the gap
needs CLAY/Tier-2, not better prompts. (Driver currently a scratchpad script — fold into
`src/tier0.py` as a `use_prompt_bank=True` option if we want it repo-tracked.)

---

## Key decisions / notes

- **Images are referenced by dataset INDEX, never by filename** (CONTRACT.md §0).
  `celeba[13]` ≠ `000013.jpg`.
- **Attribute tensor format** (CONTRACT.md §2): `[N, 40]`, `float32`, `+1.0` present /
  `0.0` absent. Raw CelebA uses `-1` for absent — we convert `-1 → 0` via `(x + 1) / 2`.
- **CelebA location**: currently in `~/Downloads/celeba/celeba/` (nested). `data_loader`
  auto-detects this and the project-root location.

## TODO

- [x] Confirm the "mandatory 12" queries vs. the 14 in the JSON — resolved: evaluate all 14
      (CONTRACT.md §4); note the mismatch in the report.
- [x] Day-1 CRITICAL sanity assertions (`src/sanity.py`, wired into the notebook).
- [x] Flag to Member B: frozen DB / features now live in `artifacts/`; `clip_image_features_test.pt` is present.
- [x] Model half (B): CLIP image features + Tier-0 `get_ranking` → meets the harness for **M1**.
- [x] Tier-0 baseline produced (`output/tier0_alpha1.0.csv`) and its low scores explained +
      verified (α=0 sanity check, well-formed image table, working self-retrieval).
- [x] CLAY prerequisite: per-attribute prompt bank (`artifacts/clip_attr_prompt_bank.pt`).
- [ ] (Optional, report only) α-sweep ablation curve for Tier-0 — characterization, not tuning.
- [ ] Next (Tier-1): build CLAY subspace projectors from the prompt bank (SVD per attribute).
