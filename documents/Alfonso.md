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

**Output artifact:** `celeba_attributes_test.pt` (frozen, do not touch).

---

### `project/Phase_A_Data_Preparation.ipynb`

Loads and verifies the data contract. Thin notebook — all heavy lifting is in `data_loader.py`.

**Step 1 — Load CelebA dataset & attributes**
- Imports `load_celeba_dataset`, `load_attributes`, `setup_frozen_db` from `data_loader`.
- Calls `setup_frozen_db()` (no-op if already built), then loads the dataset and tensor.
- Verifies dataset size (~19,962 test images), attribute shape `[19962, 40]`, dtype `float32`.

**Step 2 — Load evaluation JSON**
- Reads `Evaluation/celeba_evaluation.json` (the authoritative ground-truth file).
- Prints per-query summary: query string, number of sources, avg targets per source.
- 14 queries total (note: `-Young` appears twice; the "mandatory 12" still TBD per CONTRACT.md §4).

**Step 3 — Sanity check (CRITICAL)**
- Uses `test_idx = 13`.
- Check 1: `celeba.filename[13]` → filename.
- Check 2: `celeba_attrs[13]` shape and sample values (now `0.0/1.0`).
- Check 3: ground-truth structure — source 13 present, target list length and samples.

---

## Key decisions / notes

- **Images are referenced by dataset INDEX, never by filename** (CONTRACT.md §0).
  `celeba[13]` ≠ `000013.jpg`.
- **Attribute tensor format** (CONTRACT.md §2): `[N, 40]`, `float32`, `+1.0` present /
  `0.0` absent. Raw CelebA uses `-1` for absent — we convert `-1 → 0` via `(x + 1) / 2`.
- **CelebA location**: currently in `~/Downloads/celeba/celeba/` (nested). `data_loader`
  auto-detects this and the project-root location.

## TODO

- [ ] Confirm the "mandatory 12" queries vs. the 14 in the JSON (CONTRACT.md §4).
- [ ] Model half: CLIP image features (`clip_image_features_test.pt`, §6).
