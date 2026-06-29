# planSV.md — Implementation Plan for `src/tier2a_SVunion.py`

> **Purpose.** Single authoritative brief for implementing **Track SV-union**, a new
> training-free retrieval method. This document is self-contained: a reader who has *not*
> seen the design conversation should be able to implement the file from this alone. It
> states the idea, the exact geometry (with the one subtlety that makes or breaks it), the
> code to reuse verbatim, the config/ablation surface, the tests, and — honestly — what
> result to expect and why.
>
> **Owner module to create:** `src/tier2a_SVunion.py`
> **Tests to create:** `test/test_tier2a_SVunion.py`
> **CSV output dir:** `output/tier2a_SVunion/`

---

## 0. One-paragraph thesis

Track V (`tier1_GDE.py`) handles `−X` by **orthogonal rejection of a single mined direction**
`v̂_X` from the query's tangent vector. Track S (`tier2a_S.py`) handles `−X` by penalising DB
images for energy in a **text-derived subspace** `S⁻`. SV-union takes the better half of each:
keep V's **query-side rejection in image space** (no modality gap, no rotation `H`), but replace
its **single direction** with a **multi-dimensional visual subspace** mined by SVD from the
**train-split images that actually have attribute X**. "Not Male" becomes "delete the whole
*Male region* from the query," not "delete one average-Male axis." Everything else (positive
geodesic addition, cosine scoring on the frozen test DB) is inherited unchanged from Track V.

The name: **S**ubspace + **V**isual, **union** of the per-negative-attribute subspaces.

---

## 1. Why this is a distinct method (not a re-run of V or S)

| | negation representation | where it acts | source of the negative geometry |
|---|---|---|---|
| **Track V** (`tier1_GDE.py`) | single direction `v̂_X` | **query** tangent vector | tangent-mean of train has-X images |
| **Track S** (`tier2a_S.py`) | `k`-dim subspace `S⁻` | **DB** score penalty `λ·‖proj‖` | **text** prompt stack of "X" |
| **SV-union** (this file) | `k`-dim subspace `Q_X` | **query** tangent vector | **SVD of train has-X images** |

Key consequences, each a reportable point:

- **vs V:** SV-union is a strict generalisation. At `k_neg = 1` the rejected subspace is
  (approximately) V's single direction — see §3.4, the *nesting property*. `k_neg > 1` adds
  the spread of how the attribute actually appears across faces. The `k_neg` sweep directly
  measures whether that spread carries retrieval signal.
- **vs S:** SV-union's negative geometry is **visual** (train images), so it never crosses the
  modality gap. S builds `S⁻` from text and must trust that "close to the Male *text* subspace"
  identifies male *images* — the link the modality gap weakens. SV-union builds `S⁻` from male
  *images* directly.
- **vs NCS (Approach 2 in `documents/OriginalTrainingFreeApproaches.md`):** NCS projects the
  **DB** onto the complement of a **text** subspace. SV-union projects the **query** onto the
  complement of a **visual** subspace. Different side, different source. (DB-side projection is
  included here only as an ablation toggle, §5, `reject_on="db"`.)

**No label leakage.** All negative subspaces are mined from the **train** split
(`clip_image_features_train.pt` + `celeba_attributes_train.pt`). The test split and the GT JSON
are never inspected to build geometry — identical discipline to Track V, which already mines its
directions this way (`tier1_GDE.py:_load_train_features`).

---

## 2. Artifacts — all already exist, no Colab run needed

| Artifact | Shape | Loader (reuse) |
|---|---|---|
| `artifacts/clip_image_features_test.pt` | `[19962, 512]` unit | `clip_features.load_image_features()` |
| `artifacts/clip_image_features_train.pt` | `[N_train, 512]` unit | `tier1_GDE._load_train_features()` |
| `artifacts/celeba_attributes_train.pt` | `[N_train, 40]` `{0,1}` | `tier1_GDE._load_train_attributes()` |
| `artifacts/visual_directions.pt` | `{mu:[512], directions:[40,512]}` | `tier1_GDE.load_or_mine_directions()` |

The global tangent point `μ` and the positive directions `v_a` come straight from
`load_or_mine_directions()` — **do not recompute them.** SV-union only adds the *negative
visual subspaces*, which are new and must be mined + cached (§3.3).

---

## 3. The method, precisely

All notation on the unit hypersphere `S^{d−1}`, `d = 512`. `μ` is the global intrinsic mean of
the train corpus (from the directions cache). Everything happens in the **single tangent space
`T_μ`** — this is the load-bearing invariant (§3.2).

### 3.1 Positive side — inherited from Track V, unchanged

```
q_tan = Log_μ(v_ref) + Σ_{a ∈ T_pos} α · v_a            # v_a = directions[ATTR_TO_IDX[a]]
```

`v_ref = image_features[src_idx]` (test DB). `v_a` are the mined tangent-mean directions.
`α` is the push strength (Track V's knob; default `1.0`). This is verbatim the first loop of
`tier2a_visual_extension._compose_query_ext`.

### 3.2 ⚠ The one subtlety that makes or breaks it — log-map negatives at the GLOBAL μ

The query tangent vector `q_tan` lives in `T_μ` (tangent space at the **global** mean `μ`).
The negative subspace we reject **must live in the same `T_μ`**, or the rejection is
geometrically meaningless (projecting a vector onto a subspace defined in a *different*
tangent plane).

Therefore, to build the subspace for negative attribute `b`, log-map its train images **at the
global `μ`** — *not* at their own local mean:

```
mask_b   = train_attributes[:, idx_b] > 0.5            # train images that HAVE attribute b
X_b      = train_features[mask_b]                       # [m_b, 512] unit rows
L_b      = Log_μ(X_b)                                   # [m_b, 512] tangent vectors at GLOBAL μ
_,_,Vh_b = svd(L_b, full_matrices=False)
Q_b      = Vh_b[:k_neg].T                               # [512, k_eff] orthonormal cols, all ⊥ μ
```

> **Do NOT call `manifold.build_subspace` here.** That helper log-maps at the *local* mean
> `normalize(mean(X_b))` (correct for CLAY's text subspaces, wrong here). SV-union needs a new
> helper that takes `μ` as an explicit argument. This is the single most common way to get this
> file subtly wrong.

Because `Log_μ` outputs vectors orthogonal to `μ`, every column of `Q_b` is orthogonal to `μ`
— same hyperplane as `q_tan`. Good.

### 3.3 Mining + caching the negative subspaces

Precompute **all 40** attribute visual subspaces once at a max width `K_CACHE = 50`, cache as a
single tensor, then **slice `[..., :k_neg]`** per config (orthonormality of a column-subset is
preserved, so slicing a cached `k=50` basis to `k=10` is valid).

```
visual_neg_subspaces.pt :  [40, 512, K_CACHE]    # subspaces[j] = Q_j for ATTRIBUTE_NAMES[j]
```

Cache contract mirrors `load_or_mine_directions`: load if present (unless `force=True`), else
mine from train + save. Attributes with zero train examples (shouldn't happen for the 40 CelebA
attrs, but guard anyway) get a zero block and a printed warning.

> `K_CACHE = 50` is comfortably below `m_b` (thousands of images per attribute) and `d = 512`,
> so `k_eff = K_CACHE` for every attribute. The per-config `k_neg ∈ {1,5,10,20}` just slices.

### 3.4 Negation — reject the union of negative subspaces from the query

Stack the per-attribute bases for the query's negative attributes, orthonormalise their **union**
with a thin QR (handles overlap — e.g. Male and Mustache subspaces share directions), then
project `q_tan` onto the orthogonal complement:

```
cols   = [ subspaces[ATTR_TO_IDX[b]][:, :k_neg]  for b in T_neg ]   # each [512, k_neg]
W      = concat(cols, dim=1)                                          # [512, k_total]
Q_all,_= torch.linalg.qr(W)                                          # [512, k_total] orthonormal
q_tan  = q_tan − Q_all @ (Q_all.T @ q_tan)                           # reject union span
```

Then back to the sphere and score exactly as Track V:

```
q      = normalize( Exp_μ(q_tan) )
scores = image_features @ q          # [N] cosine, rows unit
scores[src_idx] = -inf               # CONTRACT §5 self-exclusion
ranking = argsort(scores, desc=True)
```

**Nesting property (a test, §6, and a report figure).** At `k_neg = 1`, `Q_b` is the top
singular direction of the log-mapped has-b images, which is approximately `v̂_b` (the normalised
tangent-mean Track V rejects), because the log-mapped points share a strong common component
along `v_b`. So **`SV-union @ k_neg=1 ≈ Track V negation`**. Verify the resulting query vectors
have cosine `> 0.99` on a negation query. This makes the `k_neg` sweep a clean, interpretable
ablation: it isolates exactly what the extra subspace dimensions buy over V.

### 3.5 Edge cases

- **`T_neg` empty** (`+Smiling`): `W` empty → rejection is identity → method == Track V positive
  composition. Implement by skipping the QR/reject block when `not T_neg`.
- **`T_pos` empty** (`-Male, -Mustache`): `q_tan = Log_μ(v_ref)` then reject. This is the headline
  stress case — the query is the reference with the negative visual regions deleted.
- **`k_neg` clamping:** `min(k_neg, K_CACHE)`; the cache is already `k_eff = K_CACHE`.

---

## 4. File structure — mirror the existing Track V files

Follow the shape of `tier1_GDE.py` / `tier2a_visual_extension.py` exactly (same imports, same
CONTRACT §5/§7 seam, same CSV plumbing). House comment style per `CLAUDE.md` (one dense leading
block per method: label · formula · approach · purpose · optional note).

```python
# src/tier2a_SVunion.py
from dataclasses import dataclass
import torch
import torch.nn.functional as F

from data_loader import ATTRIBUTE_NAMES, ATTR_TO_IDX, _get_artifacts_dir
from clip_features import load_image_features
from eval import parse_query, evaluate_all, format_results_table, load_eval_json, find_eval_json
from results_saver import save_results_csv, output_subdir
from manifold import log_map, exp_map
from tier1_GDE import (
    _load_train_features, _load_train_attributes, load_or_mine_directions,
)
```

### 4.1 Config

```python
@dataclass(frozen=True)
class SVunionConfig:
    k_neg: int = 10            # dim of each negative VISUAL subspace (sweep 1,5,10,20; k=1 ≈ Track V)
    alpha: float = 1.0         # positive push strength (Track V's α)
    reject_on: str = "query"   # "query" (headline) | "db" (ablation: project DB into complement)
    def tag(self) -> str:
        return f"{self.reject_on}_kneg{self.k_neg}_a{self.alpha}"
```

### 4.2 Functions to implement

| Function | Role |
|---|---|
| `_build_visual_neg_subspace(X_b, mu, k)` | §3.2 — log-map `X_b` at **global `mu`**, SVD, return `[d, k_eff]`. The new helper that must NOT reuse `build_subspace`. |
| `mine_neg_subspaces(train_features, train_attributes, mu, k=K_CACHE)` | Loop 40 attrs → `[40, d, K_CACHE]`. Zero-guard empty attrs. |
| `load_or_mine_neg_subspaces(force=False)` | Cache I/O for `visual_neg_subspaces.pt`; pulls `mu` from `load_or_mine_directions`. |
| `_reject_union(q_tan, subspaces, T_neg, k_neg)` | §3.4 — stack→QR→complement projection; identity when `T_neg` empty. |
| `_compose_query_svunion(v_ref, T_pos, T_neg, mu, directions, subspaces, cfg)` | §3.1 + §3.4 → unit query `q`. |
| `make_get_ranking(query_str, image_features, mu, directions, subspaces, cfg)` | CONTRACT §7 seam; `reject_on="db"` variant projects the DB instead (§5). |
| `_run_evaluate(cfg, ...)` / `evaluate_svunion(cfg)` / `run_ablation()` | Mirror `_run_evaluate_ext` / `__main__` in `tier2a_visual_extension.py`. |

`K_CACHE = 50` as a module constant with a one-line comment.

---

## 5. `reject_on="db"` ablation (cheap, include it)

Same visual subspaces, but instead of rejecting from the query, project **every** test vector
into the complement once per query and score there (this is the visual-source analogue of NCS
Option A, `documents/OriginalTrainingFreeApproaches.md` §Approach 2):

```
DB_perp = image_features − (image_features @ Q_all) @ Q_all.T     # [N, d], DO NOT renormalize
q       = normalize(Exp_μ(q_tan_positive_only))                   # positives only; negation is in the space
scores  = DB_perp @ q                                             # raw inner product in complement
```

**Critical:** do **not** renormalise `DB_perp` rows (the doc's §"Do Not Normalize After
Projecting" — renormalisation amplifies exactly the male-coded images you removed). Leave the
suppressed norm in place; cosine via the raw inner product uses it correctly.

Headline remains `reject_on="query"`. Report both; expect them to be close, with `"db"`
slightly stronger on pure-negation queries and slightly weaker on positive-heavy ones.

---

## 6. Tests — `test/test_tier2a_SVunion.py` (mirror `test/test_tier1.py`)

1. **Subspace orthonormality:** `Q_bᵀ Q_b ≈ I_k` (1e-5).
2. **Lives in `T_μ`:** `‖Q_bᵀ μ‖ ≈ 0` (1e-5) — guards the §3.2 subtlety (subspace in the
   global tangent plane, not a local one).
3. **Rejection idempotence:** rejecting `q_tan` twice == once (1e-6).
4. **Complement orthogonality:** `(I − Q_allQ_allᵀ)` output is ⊥ `span(Q_all)` (1e-6).
5. **Nesting (§3.4):** on `-Male`, cosine between SV-union(`k_neg=1`) query and Track V query
   `> 0.99`.
6. **Empty `T_neg`:** SV-union query == Track V positive composition (allclose).
7. **End-to-end:** source excluded from its own ranking; `len(ranking) == N-1`.
8. **Empty `T_pos` + negation:** runs without crashing; ranking valid.

---

## 7. Evaluation + what to compare against

Run on **all 14** JSON queries (authoritative). Save one CSV per config to
`output/tier2a_SVunion/tier2a_SVunion_{tag}.csv` via `save_results_csv` + `output_subdir`.

**Ablation grid:** `reject_on ∈ {query, db}` × `k_neg ∈ {1, 5, 10, 20}` at `alpha=1.0`, plus an
`alpha ∈ {0.5, 1.5}` sweep at the best `k_neg`.

**Baselines to print side-by-side** (numbers already on disk, MEAN row):

| Method | R@1 | R@5 | R@10 | source CSV |
|---|---|---|---|---|
| Tier-0 Enhanced (champion) | 0.0393 | **0.1102** | 0.1651 | `output/tier0_enhanced/tier0_enhanced_all_fixes.csv` |
| Track V (gde uniform α=1) | 0.0221 | 0.0607 | 0.0910 | `output/tier2a_visual_ext/tier2a_visual_ext_gde_alpha1.0.csv` |
| Track S (best, k50) | 0.0137 | 0.0454 | 0.0659 | `output/tier2a_S/tier2a_S_percond_anchor_k50_50_lam0.1_rotH.csv` |

The decisive per-query comparison is **SV-union vs Track V**, because they differ *only* in the
negation mechanism. Print a per-query R@5 table for the negation/composed queries:
`-Heavy_Makeup`, `-Young`, `+Eyeglasses,+Smiling`, `+Black_Hair,-Wavy_Hair`, `-Male,-Mustache`,
`+Chubby,-Young`, `-Smiling,+Eyeglasses,+Wearing_Hat`, `+Wearing_Lipstick,-Heavy_Makeup,+Smiling`.

---

## 8. Honest expected outcome (read before judging the result a failure)

- **Most likely:** SV-union **matches or slightly beats Track V** on negation/composed queries,
  and **does not beat Tier-0 Enhanced overall.** That is still a publishable, gradeable result:
  it isolates the value of *visual subspace* vs *single direction* negation, training-free.
- **`-Male, -Mustache` will probably stay at or near 0.000.** We verified why and it is NOT a
  bug in this method: all 27 sources are themselves `Male=1, Mustache=1`, and the GT additionally
  requires Hamming ≤ 2 on the other 38 attributes. So a valid target is the rare "female twin"
  of a specific male face. Deleting the Male/Mustache subspace from the query is necessary but
  not sufficient — you still need to rank those few twins above ~12k other valid-negation images,
  and the only signal for that is identity-to-`v_ref`, which (because `v_ref` is male) pulls the
  wrong way. This is the fundamental **identity-anchor vs negation** tension; closing it is what a
  *trained* Φ would be for. State this explicitly in the report; it is the gradeable insight.
- **What would count as a win:** (a) SV-union > Track V on the negation-bearing queries above at
  `k_neg > 1`, and (b) a clean monotone-or-plateau `k_neg` curve showing the subspace dimensions
  contribute (or honestly showing they don't past `k_neg=1`, which is itself a finding).

Do **not** tune toward beating Tier-0 Enhanced by inflating `alpha` or shrinking subspaces until
the method degenerates — the scientific value is the controlled V-vs-SV-union contrast, not a
leaderboard number.

---

## 9. Definition of done

1. `src/tier2a_SVunion.py` exposes `make_get_ranking(...)` with the CONTRACT §7 signature and
   runs `python src/tier2a_SVunion.py` to produce the full ablation grid offline (no Colab).
2. `artifacts/visual_neg_subspaces.pt` `[40, 512, 50]` mined once and cached; reused on rerun.
3. `output/tier2a_SVunion/*.csv` written, one per config, shared schema.
4. `test/test_tier2a_SVunion.py` — all 8 checks green (esp. #2 tangent-plane and #5 nesting).
5. A printed comparison table (SV-union best vs V vs S vs Tier-0 Enhanced) + the per-query
   negation R@5 table.
6. Two-to-three sentence written conclusion stating, honestly, whether visual-subspace rejection
   beat single-direction rejection, and the identity-anchor tension on `-Male,-Mustache`.

---

## 10. Reuse map (exact lines to lean on)

- Positive composition loop + seam shape: `tier2a_visual_extension.py:_compose_query_ext`
  (lines ~118–156) and `make_get_ranking_ext` (~163–186).
- `μ`, `directions`, caching pattern: `tier1_GDE.py:load_or_mine_directions` (~100–119).
- Train loaders: `tier1_GDE.py:_load_train_features` / `_load_train_attributes` (~44–65).
- `log_map` / `exp_map` (operate at an explicit `μ`): `manifold.py` (~18–40). **Reuse these;**
  write a *new* `_build_visual_neg_subspace` rather than `manifold.build_subspace` (§3.2).
- QR-complement rejection precedent (single-step, order-free): `tier2a_visual_extension.py`
  negation block (~137–152) — generalise its single `v̂` to the `Q_all` basis.
- CSV + eval driver: `save_results_csv`, `output_subdir`, `evaluate_all`, `format_results_table`,
  `_run_evaluate_ext` (~193–224).
- Test scaffold to copy: `test/test_tier1.py`.
