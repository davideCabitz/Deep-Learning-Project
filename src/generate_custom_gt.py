"""
Ground-truth generator for CUSTOM CelebA queries (spec §3.1: "additional, custom queries").

The official celeba_evaluation.json is authoritative for the mandatory benchmark and is
never modified (CONTRACT §4). This module reproduces the instructor's protocol EXACTLY —
no subsampling, no caps — so additional queries can be frozen in a SEPARATE JSON with the
identical schema. The full protocol, reverse-engineered from and verified against the
official JSON (see verify_official):

  For query (T⁺, T⁻):
    source s is valid    iff  (0) s VIOLATES every ± constraint (each axis must demand a
                                  real modification — else the task is twin retrieval),
                          AND (3) s has ≥ MIN_TARGETS valid targets (spec §3.1.2).
    target x is valid    iff  (1) x strictly satisfies every ± constraint,
                          AND (2) Hamming(A(s), A(x)) ≤ HAMMING_MAX on non-queried attrs.

Unlike AwA2 (class-level rows → a 50×50 relation, src/awa2/generate_gt.py), CelebA
attributes are per-image, so the rule is a genuine [N, N] image relation. It is computed
with the popcount matmul identity ham(s,x) = ‖a_s‖₁ + ‖a_x‖₁ − 2·a_s·a_xᵀ, chunked over
sources, never materializing the full matrix.

`--verify-official` is the credibility gate: the generator must RECONSTRUCT the official
JSON identically — same source sets, same target sets, every query. Exact equality against
the instructor's data is what licenses trusting the custom GT produced by this code path.

Outputs:
  Evaluation/celeba_custom_evaluation.json  — official schema, consumed unchanged by eval.py
  output/custom/query_viability.csv         — the candidate report

One-time build (no GPU, no model):   python src/generate_custom_gt.py
Exact check vs the instructor:       python src/generate_custom_gt.py --verify-official
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import ATTRIBUTE_NAMES, load_attributes
from eval import find_eval_json, load_eval_json
from results_saver import output_subdir
from vm_pipeline import query_indices

HAMMING_MAX = 2               # spec §3.1.1 — identity-preservation radius on free attrs
MIN_TARGETS = 5               # spec §3.1.2 — statistical-significance floor per source
MAX_DISCRIMINATIVE = 0.30     # warn on queries where > this corpus fraction is a target
SOURCE_CHUNK = 4096           # block size — bounds the [chunk, M] Hamming slab

# Curated custom benchmark — every query tests one named claim of the report
# (§10.11 details the hypothesis table). Frozen after inspecting the viability report.
CURATED_QUERIES = [
    "-Young, -Smiling",                        # pure-negation replication beyond -Male,-Mustache
    "-No_Beard",                               # double negation ("not beardless")
    "+Heavy_Makeup, -Wearing_Lipstick",        # anti-correlated pair (Chow-Liu stress)
    "+Gray_Hair, -Young",                      # correlation-consistent contrast to the above
    "+Big_Lips",                               # probe-weak attribute (probe acc 69.6%)
    "+Pale_Skin",                              # probe-weak / low-support attribute
    "+Bald",                                   # rare attribute (~2% prevalence)
    "+Smiling, +Young, -Male, +Heavy_Makeup",  # depth-4 composite, corpus-mode attributes
    "+Male, -Young, +Eyeglasses, -Smiling",    # depth-4 composite, hard variant
]


def _strict_mask(A: torch.Tensor, pi: list[int], ni: list[int]) -> torch.Tensor:
    # Target rule clause 1 — strict ± satisfaction over the whole corpus at once.
    strict = torch.ones(A.shape[0], dtype=torch.bool)
    if pi:
        strict &= (A[:, pi] == 1).all(dim=1)
    if ni:
        strict &= (A[:, ni] == 0).all(dim=1)
    return strict


def _source_mask(A: torch.Tensor, pi: list[int], ni: list[int]) -> torch.Tensor:
    # Source rule clause 0 — a source must VIOLATE every constraint, i.e. the exact
    # complement of _strict_mask per axis. (Instructor policy, proven by verify_official's
    # exact reconstruction: a +a query from an already-a source is twin retrieval, not a
    # compositional edit.)
    viol = torch.ones(A.shape[0], dtype=torch.bool)
    if pi:
        viol &= (A[:, pi] == 0).all(dim=1)
    if ni:
        viol &= (A[:, ni] == 1).all(dim=1)
    return viol


def _free_hamming(src_free: torch.Tensor, cand_free: torch.Tensor) -> torch.Tensor:
    # Target rule clause 2 — [S, M] Hamming on free attrs via ‖a_s‖₁ + ‖a_x‖₁ − 2·a_s·a_xᵀ
    # (exact for 0/1 values in float32; one GEMM instead of an [S, M, F] broadcast).
    if src_free.shape[1] == 0:
        return torch.zeros(src_free.shape[0], cand_free.shape[0])
    return (src_free.sum(1, keepdim=True) + cand_free.sum(1).unsqueeze(0)
            - 2.0 * (src_free @ cand_free.T))


def target_counts(A: torch.Tensor, pi: list[int], ni: list[int],
                  free: list[int]) -> torch.Tensor:
    # Per-source count of valid targets, chunked — viability needs counts, not lists.
    # (No self-exclusion term: under the source policy a valid source violates the
    # constraints, so it can never be strict-satisfying, i.e. never its own target.)
    strict = _strict_mask(A, pi, ni)
    cand_free = A[strict][:, free]
    counts = torch.zeros(A.shape[0], dtype=torch.long)
    if cand_free.shape[0] == 0:
        return counts
    for lo in range(0, A.shape[0], SOURCE_CHUNK):
        hi = min(lo + SOURCE_CHUNK, A.shape[0])
        ham = _free_hamming(A[lo:hi][:, free], cand_free)
        counts[lo:hi] = (ham <= HAMMING_MAX).sum(dim=1)
    counts[strict] -= 1          # only relevant for diagnostics on policy-violating rows
    return counts


def targets_for_sources(A: torch.Tensor, pi: list[int], ni: list[int], free: list[int],
                        sources: list[int]) -> dict[str, list[int]]:
    # Explicit valid-target lists for the given sources — the FULL sets, uncapped.
    strict_idx = _strict_mask(A, pi, ni).nonzero(as_tuple=True)[0]
    out: dict[str, list[int]] = {}
    for lo in range(0, len(sources), SOURCE_CHUNK):
        chunk = sources[lo:lo + SOURCE_CHUNK]
        src_t = torch.as_tensor(chunk, dtype=torch.long)
        ham = _free_hamming(A[src_t][:, free], A[strict_idx][:, free])
        for row, s in enumerate(chunk):
            tgt = strict_idx[ham[row] <= HAMMING_MAX]
            out[str(s)] = sorted(tgt[tgt != s].tolist())
    return out


def build_ground_truth(query_str: str, A: torch.Tensor) -> dict[str, list[int]]:
    # {src_idx: [target_idx, ...]} for one query — the REAL ground truth: every source
    # passing clause 0 + the ≥ MIN_TARGETS floor, with its complete target set.
    pi, ni, free = query_indices(query_str)
    viable = (target_counts(A, pi, ni, free) >= MIN_TARGETS) & _source_mask(A, pi, ni)
    return targets_for_sources(A, pi, ni, free, viable.nonzero(as_tuple=True)[0].tolist())


def _candidate_queries() -> list[str]:
    # Viability-report candidates: every single ±attribute plus the curated composites.
    singles = [f"{sign}{name}" for name in ATTRIBUTE_NAMES for sign in ("+", "-")]
    return singles + [q for q in CURATED_QUERIES if q not in singles]


def viability_report(A: torch.Tensor) -> list[tuple[str, int, float, float]]:
    # Per candidate: n_valid_sources, mean targets/source, discriminativeness (mean
    # fraction of corpus that is a valid target). Basis for freezing CURATED_QUERIES:
    # reject the empty (floor) and the too-easy (high discriminativeness) ends.
    rows = []
    for q in _candidate_queries():
        pi, ni, free = query_indices(q)
        counts = target_counts(A, pi, ni, free)
        kept = counts[(counts >= MIN_TARGETS) & _source_mask(A, pi, ni)]
        if kept.numel() == 0:
            rows.append((q, 0, 0.0, 0.0))
            continue
        mean_tgt = kept.float().mean().item()
        rows.append((q, kept.numel(), mean_tgt, mean_tgt / A.shape[0]))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def verify_official(A: torch.Tensor, limit: int | None = None) -> int:
    # Credibility gate — EXACT RECONSTRUCTION of the instructor's JSON from the attribute
    # DB: for every official query our generator must produce the identical source set and,
    # per source, the identical target set. Zero inference, zero tolerance; a passing run
    # proves generator ≡ instructor protocol. Returns the number of mismatched queries.
    gt_list = load_eval_json(find_eval_json())
    if limit is not None:
        gt_list = gt_list[:limit]
    mismatched = 0
    for entry in gt_list:
        ours = build_ground_truth(entry["query"], A)
        theirs = {s: sorted(int(t) for t in ts) for s, ts in entry["ground_truth"].items()}
        missing = theirs.keys() - ours.keys()
        extra = ours.keys() - theirs.keys()
        tgt_diff = sum(1 for s in theirs.keys() & ours.keys() if ours[s] != theirs[s])
        ok = not (missing or extra or tgt_diff)
        mismatched += 0 if ok else 1
        print(f"  [{'OK ' if ok else 'FAIL'}] {entry['query']:<46s} "
              f"sources ours={len(ours):5d} theirs={len(theirs):5d}  "
              f"missing={len(missing)} extra={len(extra)} target_mismatch={tgt_diff}")
    print(f"[{'OK' if mismatched == 0 else 'FAIL'}] exact reconstruction: "
          f"{len(gt_list) - mismatched}/{len(gt_list)} queries identical to the official JSON")
    return mismatched


def main() -> None:
    A = load_attributes()

    # 1. Viability report over auto-generated candidates + curated composites.
    report = viability_report(A)
    out_dir = output_subdir("custom")
    report_path = out_dir / "query_viability.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query", "n_sources", "mean_targets_per_source", "discriminativeness"])
        for q, n_src, mean_tgt, disc in report:
            w.writerow([q, n_src, f"{mean_tgt:.2f}", f"{disc:.4f}"])
    print(f"[OK] Viability report: {report_path} ({len(report)} candidates)")

    # 2. Freeze the curated benchmark JSON (official schema, separate file — CONTRACT §4).
    gt_list = []
    for q in CURATED_QUERIES:
        gt = build_ground_truth(q, A)
        if not gt:
            print(f"  [!] SKIP {q}: no source passes the policy + >={MIN_TARGETS}-target floor")
            continue
        disc = (sum(len(v) for v in gt.values()) / len(gt)) / A.shape[0]
        if disc > MAX_DISCRIMINATIVE:
            print(f"  [!] WARN {q}: discriminativeness {disc:.3f} > {MAX_DISCRIMINATIVE} (too easy)")
        gt_list.append({"query": q, "ground_truth": gt})
        print(f"  {q:<44s} sources={len(gt):5d}  disc={disc:.3f}")

    json_path = find_eval_json().parent / "celeba_custom_evaluation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(gt_list, f)
    print(f"[OK] Custom benchmark JSON: {json_path} ({len(gt_list)} queries)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Custom CelebA GT: freeze benchmark or verify rule")
    ap.add_argument("--verify-official", action="store_true",
                    help="exact reconstruction check of the instructor JSON from the DB")
    ap.add_argument("--limit", type=int, default=None,
                    help="verify only the first N official queries (default: all)")
    args = ap.parse_args()
    if args.verify_official:
        raise SystemExit(1 if verify_official(load_attributes(), args.limit) else 0)
    main()
