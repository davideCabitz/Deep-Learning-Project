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
- [ ] Flag to Member B: frozen DB / features now live in `artifacts/`; write `clip_image_features_*.pt` there.
- [ ] Model half (B): CLIP image features + Tier-0 `get_ranking` → meets the harness for **M1**.
