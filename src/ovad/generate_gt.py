"""
Ground-truth generator for OVAD — masked relative-Hamming rule over tri-state labels.

Reproduces the INSTRUCTOR PROTOCOL verified in src/generate_custom_gt.py (exact
reconstruction of the official celeba_evaluation.json) and mirrored in
src/awa2/generate_gt.py, adapted for OVAD's structural differences (see
OVAD/OVAD_Implementation_Plan.md §3):

  For query (T+, T-):
    source s is valid  iff  (0) s VIOLATES every +/- constraint — a source that already
                                satisfies the query is a twin-retrieval task, not a
                                compositional EDIT (the instructor policy; see
                                src/generate_custom_gt.py's docstring and
                                src/awa2/generate_gt.py's _source_class_mask for the
                                CelebA/AwA2 analogues this reproduces).
    target x is valid  iff  (1) x strictly satisfies every +/- constraint, AND
                             (2) D_Hrel(s, x) <= HAMMING_RATE on the free attributes.

OVAD-specific adaptations to that shared protocol:

  1. Labels are TRI-STATE {1=positive, 0=negative, -1=unlabeled}. "Satisfies" and
     "violates" both require the instance to be LABELED on that attribute (an unlabeled
     -1 entry can prove neither); an unlabeled candidate/source is excluded from both
     roles on that query. Comparing two instances on a FREE attribute for the Hamming
     rate is only meaningful if BOTH are labeled — unlabeled entries are masked OUT of
     the Hamming sum entirely (not counted as a match or a mismatch).
  2. OVAD has ~3x CelebA's attribute count (117 vs 40) with variable per-pair valid-
     comparison counts (some attributes are only ~45% labeled), so a FIXED integer
     Hamming radius (CelebA's "<=2") does not transfer. We instead fix the RATE:

       D_Hrel(s, t) = (1/|V(s,t)|) * sum_{i in V(s,t)} 1[attr(s,i) != attr(t,i)]
       V(s,t) = {i : attr(s,i) != -1 AND attr(t,i) != -1, i not in query attributes}

       valid target  iff  D_Hrel(s, t) <= HAMMING_RATE   (0.05, matching CelebA's 2/40)

Retrieval corpus = query-source pool = the TEST split of the synthetic image-level
train/test split (data_loader.build_split). Emitted indices are test-split-local rows —
the exact index space every tier scores in via load_image_features("test"). The split
exists so trained/mined methods (GDE/LDE, DualScore, tier5) have a leak-free train pool;
if the smaller test corpus kills a curated query's viability, main() rebuilds the split
at FALLBACK_TRAIN_FRACTION and reports which ratio was frozen. (The earlier full-corpus
benchmark is archived as ovad_evaluation_fullcorpus.json, superseded by this one.)

No source/target caps — exact parity with generate_custom_gt.py and the corrected
src/awa2/generate_gt.py: every valid source is emitted with its COMPLETE target set,
so Precision@K / Recall@K are exact (a subsampled GT understates |R_K ∩ G| and makes
both metrics unreal).

Output: Evaluation/ovad_evaluation.json, schema-identical to celeba_evaluation.json,
so eval.py/vm_pipeline.evaluate_scorer consume it unchanged.

One-time build (no GPU):  python src/ovad/generate_gt.py
"""

import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from eval import parse_query
from ovad.data_loader import (ATTR_TO_IDX, ATTRIBUTE_NAMES, N_ATTRS,
                              _get_project_root, build_split, load_attribute_meta,
                              load_attributes, load_split_meta)

HAMMING_RATE = 0.05      # relative tolerance, matches CelebA's 2/40 = 5% exactly
MIN_TARGETS = 5          # spec §3.1.2 — statistical-significance floor per source
MAX_DISCRIMINATIVE = 0.30    # reject queries where > this fraction of corpus is a target
FALLBACK_TRAIN_FRACTION = 0.7   # used iff a curated query dies on the 80/20 test corpus

# Curated benchmark: single-attribute queries drawn from head/medium freq_set (avoids
# vacuous GT from tail-rare attributes), plus composed queries mixing parent_type
# families. All 12 verified viable on the frozen 70/30 test split (probed 2026-07-05;
# earlier composed picks — "+blue & -wood", "+striped & -black & +big", "+wood & +big" —
# produced ZERO valid sources under the violate-every-constraint source policy on
# progressively smaller corpora). The triple pairs ± within clothes color (+blue, -black)
# plus a face-expression axis: a conflicting-condition edit in the spirit of CelebA's
# "+Wearing Lipsticks & -Heavy Makeup & +Smiling" — the only viable 3-attr composition
# on this corpus. Underscore, not space, before the colon — matches eval.parse_query's
# ' ' -> '_' normalization (ATTR_TO_IDX keys use the raw name with spaces converted the
# same way at load time, see _to_query_name below).
CURATED_QUERIES = [
    "+clothes_color:blue",
    "-clothes_color:black",
    "+material:wood/wooden/bamboo/hardwood",
    "+pattern:striped/lined/pinstriped",
    "+state:closed",
    "+face_expression:happy/smiling/laughing/grinning/joyful",
    "-cleanliness:unclean/dirt/dirty/muddy",
    "+size:big/large/giant/huge",
    "-clothes_color:black & +face_expression:happy/smiling/laughing/grinning/joyful",
    "+state:closed & -cleanliness:unclean/dirt/dirty/muddy",
    "+material:wood/wooden/bamboo/hardwood & -cleanliness:unclean/dirt/dirty/muddy",
    "+clothes_color:blue & -clothes_color:black & +face_expression:happy/smiling/laughing/grinning/joyful",
]


def _to_query_name(attr_name: str) -> str:
    # ATTR_TO_IDX keys use the RAW attribute name (spaces intact, e.g. "clothes color:blue").
    # Query strings use underscores (eval.parse_query normalizes ' ' -> '_'); invert that
    # here so a query token maps back to the raw key.
    return attr_name.replace('_', ' ')


def query_indices(query_str: str) -> tuple[list[int], list[int], list[int]]:
    # OVAD variant of vm_pipeline.query_indices: parse into (positive, negative, free)
    # column indices over the 117 attributes, undoing the underscore normalization.
    pos, neg = parse_query(query_str)
    pi = [ATTR_TO_IDX[_to_query_name(a)] for a in pos]
    ni = [ATTR_TO_IDX[_to_query_name(b)] for b in neg]
    free = [c for c in range(N_ATTRS) if c not in set(pi) | set(ni)]
    return pi, ni, free


def _masked_hamming_rate_batch(src_rows: torch.Tensor, cand_rows: torch.Tensor) -> torch.Tensor:
    # D_Hrel for ALL (source, candidate) pairs at once: mismatches / valid-comparisons
    # over free_cols, pairs where either side is unlabeled (-1) excluded from both
    # numerator and denominator. src_rows [S, F], cand_rows [M, F] (both pre-restricted
    # to free_cols). Returns [S, M], +inf where a pair has zero valid comparisons.
    # Vectorized over sources (not just candidates) — the source loop was the O(N) Python
    # bottleneck; this turns the whole viability sweep into a handful of GEMM-scale ops.
    src = src_rows.to(torch.int16)                                # [S, F]
    cand = cand_rows.to(torch.int16)                              # [M, F]
    valid = (src.unsqueeze(1) != -1) & (cand.unsqueeze(0) != -1)   # [S, M, F]
    mismatch = (src.unsqueeze(1) != cand.unsqueeze(0)) & valid     # [S, M, F]
    n_valid = valid.sum(dim=2).float()                             # [S, M]
    n_mismatch = mismatch.sum(dim=2).float()                       # [S, M]
    return torch.where(n_valid > 0, n_mismatch / n_valid.clamp(min=1),
                       torch.full_like(n_valid, float("inf")))


def _strictly_satisfies(attrs: torch.Tensor, pi: list[int], ni: list[int]) -> torch.Tensor:
    # Target rule clause 1 — boolean mask [N]: instance is LABELED and correct on every
    # queried attribute. An unlabeled (-1) instance on a queried attribute cannot verify
    # the constraint, so it is excluded from the target pool.
    mask = torch.ones(attrs.shape[0], dtype=torch.bool)
    for a in pi:
        mask &= attrs[:, a] == 1
    for b in ni:
        mask &= attrs[:, b] == 0
    return mask


def _source_violates(attrs: torch.Tensor, pi: list[int], ni: list[int]) -> torch.Tensor:
    # Source rule clause 0 — boolean mask [N]: instance VIOLATES every queried
    # constraint (the exact complement of _strictly_satisfies per axis, tri-state aware:
    # "violates +a" needs attrs[:,a]==0 labeled-negative, not merely != 1, since -1 proves
    # nothing). Mirrors src/generate_custom_gt.py's _source_mask / src/awa2/generate_gt.py's
    # _source_class_mask — an already-satisfying source is twin retrieval, not an edit.
    mask = torch.ones(attrs.shape[0], dtype=torch.bool)
    for a in pi:
        mask &= attrs[:, a] == 0
    for b in ni:
        mask &= attrs[:, b] == 1
    return mask


SOURCE_CHUNK = 256   # bounds the [chunk, M, F] intermediate to a few hundred MB


def build_ground_truth(query_str, attrs):
    # {src_idx: [complete target list]} for one query over the full filtered instance
    # corpus — the REAL ground truth, no source/target caps (parity with
    # generate_custom_gt.build_ground_truth). Source pool (clause 0, violates) and target
    # pool (clause 1, satisfies) are DISJOINT by construction (an instance cannot both
    # violate and satisfy the same constraint), so no self-exclusion term is needed —
    # unlike a same-pool design, a valid source can never appear in its own target list.
    # Sources are chunked (not looped one at a time) so the O(S*M*F) mask/mismatch
    # tensors stay GEMM-scale.
    pi, ni, free = query_indices(query_str)
    free_cols = torch.tensor(free, dtype=torch.long) if free else torch.empty(0, dtype=torch.long)

    target_mask = _strictly_satisfies(attrs, pi, ni)
    cand_idx = target_mask.nonzero(as_tuple=True)[0]
    if len(cand_idx) < MIN_TARGETS:
        return {}
    cand_free = attrs[cand_idx][:, free_cols] if len(free_cols) else attrs[cand_idx][:, :0]

    source_mask = _source_violates(attrs, pi, ni)
    src_idx = source_mask.nonzero(as_tuple=True)[0]
    if len(src_idx) == 0:
        return {}

    ground_truth = {}
    for lo in range(0, len(src_idx), SOURCE_CHUNK):
        src_idx_chunk = src_idx[lo:lo + SOURCE_CHUNK]
        src_free_chunk = attrs[src_idx_chunk][:, free_cols] if len(free_cols) else \
            attrs[src_idx_chunk][:, :0]
        rate = _masked_hamming_rate_batch(src_free_chunk, cand_free)   # [chunk, M]
        keep = rate <= HAMMING_RATE
        for j, s in enumerate(src_idx_chunk.tolist()):
            tgt = cand_idx[keep[j]]
            if len(tgt) < MIN_TARGETS:
                continue
            ground_truth[str(s)] = sorted(tgt.tolist())
    return ground_truth


def _candidate_queries():
    # Auto-generated candidates for the viability report: every single +/-attribute
    # restricted to head/medium freq_set (tail attributes are too sparsely labeled to
    # reliably produce >=5-target sources).
    meta = load_attribute_meta()
    cands = []
    for m in meta:
        if m['freq_set'] == 'tail':
            continue
        name = _to_query_name(ATTRIBUTE_NAMES[m['id']]).replace(' ', '_')
        cands.append(f"+{name}")
        cands.append(f"-{name}")
    return cands


def viability_report(attrs):
    n = attrs.shape[0]
    rows = []
    for q in _candidate_queries():
        gt = build_ground_truth(q, attrs)
        if not gt:
            rows.append((q, 0, 0.0, 0.0))
            continue
        sizes = [len(v) for v in gt.values()]
        mean_tgt = sum(sizes) / len(sizes)
        rows.append((q, len(gt), mean_tgt, mean_tgt / n))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def _freeze_curated(attrs):
    # Build the curated benchmark on one corpus; returns (entries, dead queries).
    gt_list, dead = [], []
    for q in CURATED_QUERIES:
        gt = build_ground_truth(q, attrs)
        if not gt:
            dead.append(q)
            print(f"  [!] DEAD {q}: no source passes the policy + >={MIN_TARGETS}-target floor")
            continue
        disc = (sum(len(v) for v in gt.values()) / len(gt)) / attrs.shape[0]
        if disc > MAX_DISCRIMINATIVE:
            print(f"  [!] WARN {q}: discriminativeness {disc:.3f} > {MAX_DISCRIMINATIVE}")
        gt_list.append({"query": q, "ground_truth": gt})
        print(f"  {q:<60s} sources={len(gt):3d}  disc={disc:.3f}")
    return gt_list, dead


def _archive_fullcorpus_json(eval_dir: Path) -> None:
    # One-time: the pre-split full-corpus benchmark is superseded, not deleted.
    current, archive = eval_dir / "ovad_evaluation.json", eval_dir / "ovad_evaluation_fullcorpus.json"
    if current.exists() and not archive.exists():
        current.rename(archive)
        print(f"[OK] Archived full-corpus benchmark -> {archive}")


def main():
    # 1. Curated benchmark on the 80/20 test split; fall back to 70/30 if a query dies.
    build_split()
    attrs = load_attributes("test")
    print(f"Test corpus: {attrs.shape[0]} instances "
          f"(train_fraction={load_split_meta()['train_fraction']})")
    gt_list, dead = _freeze_curated(attrs)
    if dead:
        print(f"[!] {len(dead)} curated queries died at 80/20 — rebuilding split at "
              f"{FALLBACK_TRAIN_FRACTION:.0%} train")
        build_split(force=True, train_fraction=FALLBACK_TRAIN_FRACTION)
        attrs = load_attributes("test")
        print(f"Test corpus: {attrs.shape[0]} instances (fallback)")
        gt_list, dead = _freeze_curated(attrs)
        for q in dead:
            print(f"  [!] SKIP {q}: still dead at the fallback ratio")

    # 2. Viability report over auto-generated candidates on the FINAL test corpus.
    report = viability_report(attrs)
    out_dir = _get_project_root() / "OVAD"
    report_path = out_dir / "query_viability.csv"
    with open(report_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", "n_sources", "mean_targets_per_source", "discriminativeness"])
        for q, n_src, mean_tgt, disc in report:
            w.writerow([q, n_src, f"{mean_tgt:.2f}", f"{disc:.4f}"])
    print(f"[OK] Viability report: {report_path} ({len(report)} candidates)")

    eval_dir = _get_project_root() / "Evaluation"
    eval_dir.mkdir(exist_ok=True)
    _archive_fullcorpus_json(eval_dir)
    json_path = eval_dir / "ovad_evaluation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(gt_list, f)
    meta = load_split_meta()
    print(f"[OK] Benchmark JSON: {json_path} ({len(gt_list)} queries, "
          f"test split @ train_fraction={meta['train_fraction']}, seed={meta['seed']})")


if __name__ == "__main__":
    main()
