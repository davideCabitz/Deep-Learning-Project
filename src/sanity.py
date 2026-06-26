"""Sanity checks — the assertions everything downstream depends on.

These are the **Day-1 CRITICAL checks** from the roadmap and CONTRACT.md. They are
written as *asserting* functions (not prints) so that a broken assumption fails
loudly the moment this module runs — before any retrieval method produces a
silently-wrong score.

Three things are guarded:

1. **Indexing** (`CONTRACT.md` §0) — images are referred to by **dataset index**,
   never by filename. ``celeba[13]`` must be ``182651.jpg``. If torchvision ever
   reshuffles the split, or someone swaps in a filename-based lookup, this is the
   tripwire that catches it.
2. **Attribute tensor shape/format** (`CONTRACT.md` §2) — ``[N, 40]`` float32,
   values in ``{0.0, 1.0}``, one row per image, aligned to the dataset index.
3. **Ground-truth viability** (`ROADMAP.md` — "only sources with ≥5 valid targets")
   — every source in the eval JSON must have at least 5 valid targets, keys are
   strings, all indices are in range.

Run standalone (``python src/sanity.py``) for a green-light report, or import
``run_all_checks`` / the individual ``assert_*`` functions from the notebook so
Step 3 *proves* correctness instead of merely displaying it.
"""

from __future__ import annotations

from pathlib import Path

import torch

from data_loader import (
    ATTRIBUTE_NAMES,
    load_attributes,
    load_celeba_dataset,
)
from eval import find_eval_json, load_eval_json


# ---------------------------------------------------------------------------
# Locked-in expected values (verified against the real CelebA test split).
# ---------------------------------------------------------------------------
#: ``celeba.filename[13]`` for the torchvision CelebA *test* split. This is the
#: canonical indexing tripwire from CONTRACT.md §0 / ROADMAP.md. Do NOT change it
#: unless the dataset itself changes — that is exactly the failure it catches.
EXPECTED_FILENAME_AT_13 = "182651.jpg"

#: CelebA test split size and attribute count (CONTRACT.md §2).
EXPECTED_N_IMAGES = 19962
EXPECTED_N_ATTRS = 40

#: Eval protocol: a source is only included if it has at least this many valid
#: targets (ROADMAP.md). The JSON is authoritative; we assert the file honours it.
MIN_TARGETS_PER_SOURCE = 5


# ---------------------------------------------------------------------------
# 1. Indexing — the single most important check
# ---------------------------------------------------------------------------
def assert_indexing(celeba=None, *, idx: int = 13, expected: str = EXPECTED_FILENAME_AT_13):
    """Assert dataset index → filename mapping (CONTRACT.md §0).

    One failure here means every score downstream is wrong, so this runs first.
    """
    if celeba is None:
        celeba = load_celeba_dataset()

    actual = celeba.filename[idx]
    assert actual == expected, (
        f"INDEXING BROKEN: celeba.filename[{idx}] == {actual!r}, expected {expected!r}. "
        "Ground-truth indices are PyTorch dataset indices (CONTRACT.md §0); if this "
        "fails, the test split was reshuffled or a filename-based lookup crept in."
    )

    n = len(celeba)
    assert n == EXPECTED_N_IMAGES, (
        f"Test split has {n} images, expected {EXPECTED_N_IMAGES}. "
        "Wrong split or a changed dataset — downstream feature/attribute rows will misalign."
    )
    return celeba


# ---------------------------------------------------------------------------
# 2. Attribute tensor — shape, dtype, value range, alignment
# ---------------------------------------------------------------------------
def assert_attributes(attributes=None, celeba=None):
    """Assert the attribute tensor matches CONTRACT.md §2 exactly."""
    if attributes is None:
        attributes = load_attributes()

    assert isinstance(attributes, torch.Tensor), (
        f"attributes must be a torch.Tensor, got {type(attributes)!r}."
    )
    assert attributes.dtype == torch.float32, (
        f"attributes dtype is {attributes.dtype}, expected float32 (CONTRACT.md §2)."
    )

    n, d = attributes.shape
    assert d == EXPECTED_N_ATTRS == len(ATTRIBUTE_NAMES), (
        f"attributes has {d} columns; expected {EXPECTED_N_ATTRS} "
        f"(len(ATTRIBUTE_NAMES) == {len(ATTRIBUTE_NAMES)})."
    )
    assert n == EXPECTED_N_IMAGES, (
        f"attributes has {n} rows, expected {EXPECTED_N_IMAGES} (one per test image)."
    )

    # Values must be a clean 0/1 mask (CONTRACT.md §2: we converted -1 → 0).
    uniq = torch.unique(attributes)
    assert set(uniq.tolist()) <= {0.0, 1.0}, (
        f"attributes contains values outside {{0.0, 1.0}}: {uniq.tolist()}. "
        "Raw CelebA uses -1/+1; the loader must map -1 → 0."
    )

    # Row count must match the dataset (same index space as everything else).
    if celeba is not None:
        assert n == len(celeba), (
            f"attributes rows ({n}) != dataset size ({len(celeba)}); index spaces disagree."
        )
    return attributes


# ---------------------------------------------------------------------------
# 3. Ground-truth viability — the eval JSON honours the protocol
# ---------------------------------------------------------------------------
def assert_gt_viability(eval_json=None, *, n_images: int = EXPECTED_N_IMAGES):
    """Assert every source has ≥5 valid targets and all indices are in range.

    Trusts the JSON as authoritative (CONTRACT.md §4) but verifies it obeys the
    protocol the rest of the code assumes.
    """
    if eval_json is None:
        eval_json = load_eval_json(find_eval_json())

    assert isinstance(eval_json, list) and eval_json, (
        "Eval JSON must be a non-empty list of query objects (CONTRACT.md §4)."
    )

    n_queries = len(eval_json)
    offenders = []
    n_sources = 0

    for entry in eval_json:
        assert "query" in entry and "ground_truth" in entry, (
            f"Query object missing 'query'/'ground_truth' keys: {list(entry.keys())}."
        )
        gt = entry["ground_truth"]
        for src, targets in gt.items():
            n_sources += 1

            # Keys are strings that must cast cleanly to in-range dataset indices.
            src_idx = int(src)  # raises if not an int-string
            assert 0 <= src_idx < n_images, (
                f"Source index {src_idx} out of range [0, {n_images}) for query "
                f"{entry['query']!r}."
            )

            # Every target is an in-range dataset index, distinct from its source.
            for t in targets:
                t = int(t)
                assert 0 <= t < n_images, (
                    f"Target index {t} out of range [0, {n_images}) "
                    f"(source {src_idx}, query {entry['query']!r})."
                )
                assert t != src_idx, (
                    f"Source {src_idx} appears in its own target list "
                    f"(query {entry['query']!r}) — it must be excluded (CONTRACT.md §5)."
                )

            if len(targets) < MIN_TARGETS_PER_SOURCE:
                offenders.append((entry["query"], src_idx, len(targets)))

    assert not offenders, (
        f"{len(offenders)} source(s) have < {MIN_TARGETS_PER_SOURCE} targets, violating "
        f"the eval protocol (ROADMAP.md). First few: {offenders[:5]}."
    )
    return n_queries, n_sources


# ---------------------------------------------------------------------------
# 4. Run everything — the one entry point the notebook / CLI calls
# ---------------------------------------------------------------------------
def run_all_checks(*, verbose: bool = True) -> dict:
    """Run all sanity checks; raise on the first failure, else return a summary."""
    def _say(msg: str):
        if verbose:
            print(msg)

    celeba = load_celeba_dataset()
    attributes = load_attributes()
    eval_json = load_eval_json(find_eval_json())

    assert_indexing(celeba)
    _say(f"[OK] Indexing: celeba.filename[13] == {EXPECTED_FILENAME_AT_13!r}, "
         f"N == {len(celeba)} images")

    assert_attributes(attributes, celeba=celeba)
    _say(f"[OK] Attributes: {tuple(attributes.shape)} {attributes.dtype}, values in {{0,1}}")

    n_queries, n_sources = assert_gt_viability(eval_json, n_images=len(celeba))
    _say(f"[OK] Ground truth: {n_queries} queries, {n_sources} sources, "
         f"all >= {MIN_TARGETS_PER_SOURCE} targets, all indices in range")

    _say("[OK] ALL SANITY CHECKS PASSED -- safe to score.")
    return {
        "n_images": len(celeba),
        "n_attributes": attributes.shape[1],
        "n_queries": n_queries,
        "n_sources": n_sources,
    }


if __name__ == "__main__":
    run_all_checks()
