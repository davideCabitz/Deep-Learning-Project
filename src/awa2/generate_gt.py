"""
Ground-truth generator for AwA2 — the capability CelebA got for free (instructor JSON).

Reproduces the spec §3.1.1 relaxed-Hamming protocol on AwA2. Because AwA2 attributes are
CLASS-level, ground truth collapses to a 50×50 class relation, then expands to images:

  For query (T⁺, T⁻) and source image with class c_s:
    valid SOURCE class c_s  iff  (0) c_s VIOLATES every ± constraint — each axis must demand
                                     a real modification (matches the CelebA official policy,
                                     proven by generate_custom_gt.py's exact reconstruction;
                                     without it, satisfying-class sources are twin retrieval).
    valid target CLASS c_t  iff  (1) c_t strictly satisfies every ± constraint, AND
                                  (2) freeHam(c_s, c_t) ≤ min_ham(c_s) + HAMMING_SLACK on the
                                      NON-queried attributes — the ADAPTIVE radius: the spec's
                                      fixed ≤2 is impossible at class level (min inter-class
                                      free-Hamming is 3–21), so identity preservation becomes
                                      "the closest satisfying classes to the source".
    valid target IMAGES = all test images whose class is a valid target class.

Viability (spec §3.1.2): keep a source only if it has ≥ MIN_TARGETS valid target images.
No source/target caps — exact parity with generate_custom_gt.py: every valid source is
emitted with its COMPLETE target set, so Precision@K / Recall@K are exact.

Query set: auto-generate candidates, score their viability + discriminativeness, emit a
report CSV, and freeze a CURATED list (CURATED_QUERIES below, chosen from that report).

Outputs:
  Evaluation/awa2_evaluation.json   — CelebA schema, consumed unchanged by eval.py
  output/awa2/query_viability.csv   — the auto-generated candidate report

One-time build (no GPU):  python src/awa2/generate_gt.py
"""

import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / "Baselines"))
sys.path.insert(0, str(Path(__file__).parents[1] / "Training-Free"))
sys.path.insert(0, str(Path(__file__).parents[1] / "Trained"))

from eval import parse_query
from awa2.data_loader import (ATTR_TO_IDX, ATTRIBUTE_NAMES, N_ATTRS,
                              _get_project_root, load_class_attribute_matrix,
                              load_class_labels, load_class_names)

# Class-level generalization of spec §3.1.1's "Hamming ≤ 2": targets are the satisfying
# classes within min_ham(source) + slack. (A fixed radius of 2 is structurally empty —
# min free-attr Hamming from any violating source class to any satisfying class is 3–21.)
HAMMING_SLACK = 2
MIN_TARGETS = 5          # spec §3.1.2 — statistical-significance floor per source
MAX_DISCRIMINATIVE = 0.30    # reject queries where > this fraction of corpus is a target

# Curated benchmark, chosen from the viability report (see __main__). Mix of single- and
# multi-attribute, spanning color/texture, body parts, locomotion, habitat, diet.
CURATED_QUERIES = [
    "+stripes",
    "-furry",
    "+spots",
    "+swims",
    "+horns",
    "+longneck",
    "-quadrapedal",
    "+swims & +ocean",
    "+big & -meat",
    "+forest & -arctic",
    "+furry & -water & +ground",
    "+meat & -domestic & +fierce",
]


def query_indices(query_str: str) -> tuple[list[int], list[int], list[int]]:
    # AwA2 variant of vm_pipeline.query_indices (which is bound to CelebA's 40 names):
    # parse into (positive, negative, free) column indices over the 85 predicates.
    pos, neg = parse_query(query_str)
    pi = [ATTR_TO_IDX[a] for a in pos]
    ni = [ATTR_TO_IDX[b] for b in neg]
    free = [c for c in range(N_ATTRS) if c not in set(pi) | set(ni)]
    return pi, ni, free


def _source_class_mask(class_matrix, pi, ni):
    # Source policy clause 0 — a source class must VIOLATE every ± constraint (mirror of
    # generate_custom_gt._source_mask, whose exact reconstruction of the official CelebA
    # JSON proved this is the instructor protocol). Without it, images of satisfying
    # classes are twin-retrieval sources: their own class is a valid target at Hamming 0.
    viol = torch.ones(class_matrix.shape[0], dtype=torch.bool)
    for a in pi:
        viol &= class_matrix[:, a] == 0
    for b in ni:
        viol &= class_matrix[:, b] == 1
    return viol


def _valid_target_classes(class_matrix, pi, ni, free, src_class):
    # Boolean mask over the 50 classes: strict ± satisfaction + ADAPTIVE Hamming radius
    # (freeHam ≤ min over satisfying classes + HAMMING_SLACK) relative to src_class.
    # class_matrix is [50, 85] in {0,1}.
    strict = torch.ones(class_matrix.shape[0], dtype=torch.bool)
    for a in pi:
        strict &= class_matrix[:, a] == 1
    for b in ni:
        strict &= class_matrix[:, b] == 0
    if not strict.any():
        return strict
    if free:
        free_cols = torch.tensor(free, dtype=torch.long)
        ham = (class_matrix[:, free_cols] != class_matrix[src_class, free_cols]).sum(dim=1)
        strict &= ham <= ham[strict].min() + HAMMING_SLACK
    return strict


def build_ground_truth(query_str, class_matrix, test_labels):
    # {src_test_idx: [complete target list]} for one query, over the TEST split — the REAL
    # ground truth, no source/target caps (parity with generate_custom_gt.build_ground_truth).
    # Indices are split-local (rows of the test feature/attribute tables) — the same
    # index space every AwA2 tier scores against.
    pi, ni, free = query_indices(query_str)
    n_classes = class_matrix.shape[0]

    # Per-class list of test-image indices (split-local).
    class_to_imgs = {c: (test_labels == c).nonzero(as_tuple=True)[0] for c in range(n_classes)}

    source_ok = _source_class_mask(class_matrix, pi, ni)
    ground_truth = {}
    for src_class in range(n_classes):
        if not source_ok[src_class]:
            continue
        valid_mask = _valid_target_classes(class_matrix, pi, ni, free, src_class)
        target_imgs = torch.cat([class_to_imgs[c] for c in range(n_classes) if valid_mask[c]]) \
            if valid_mask.any() else torch.empty(0, dtype=torch.long)
        if len(target_imgs) < MIN_TARGETS:
            continue

        sources = class_to_imgs[src_class]
        # Safety only: under clause 0 a source class violates the constraints, so it can
        # never be a valid target class — the filter below is a no-op kept as a guard.
        for s in sources.tolist():
            tgt = target_imgs[target_imgs != s]
            if len(tgt) < MIN_TARGETS:
                continue
            ground_truth[str(s)] = sorted(tgt.tolist())
    return ground_truth


def _candidate_queries():
    # Auto-generated candidates for the viability report: every single ±attribute.
    cands = []
    for name in ATTRIBUTE_NAMES:
        cands.append(f"+{name}")
        cands.append(f"-{name}")
    return cands


def viability_report(class_matrix, test_labels):
    # For each candidate: n_valid_sources, mean targets/source, discriminativeness
    # (fraction of test corpus that is a valid target on average). Used to hand-pick
    # CURATED_QUERIES — reject the too-easy (high discriminativeness) and empty ones.
    n_test = len(test_labels)
    rows = []
    for q in _candidate_queries():
        gt = build_ground_truth(q, class_matrix, test_labels)
        if not gt:
            rows.append((q, 0, 0.0, 0.0))
            continue
        sizes = [len(v) for v in gt.values()]
        mean_tgt = sum(sizes) / len(sizes)
        rows.append((q, len(gt), mean_tgt, mean_tgt / n_test))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def main():
    class_matrix = load_class_attribute_matrix()
    test_labels = load_class_labels("test")

    # 1. Viability report over auto-generated candidates.
    report = viability_report(class_matrix, test_labels)
    out_dir = _get_project_root() / "output" / "awa2"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "query_viability.csv"
    with open(report_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", "n_sources", "mean_targets_per_source", "discriminativeness"])
        for q, n_src, mean_tgt, disc in report:
            w.writerow([q, n_src, f"{mean_tgt:.2f}", f"{disc:.4f}"])
    print(f"[OK] Viability report: {report_path} ({len(report)} candidates)")

    # 2. Freeze the curated benchmark JSON (CelebA schema).
    gt_list = []
    for q in CURATED_QUERIES:
        gt = build_ground_truth(q, class_matrix, test_labels)
        if not gt:
            print(f"  [!] SKIP {q}: no source passes the policy + >={MIN_TARGETS}-target floor")
            continue
        disc = (sum(len(v) for v in gt.values()) / len(gt)) / len(test_labels)
        if disc > MAX_DISCRIMINATIVE:
            print(f"  [!] WARN {q}: discriminativeness {disc:.3f} > {MAX_DISCRIMINATIVE} (too easy)")
        gt_list.append({"query": q, "ground_truth": gt})
        print(f"  {q:<34s} sources={len(gt):3d}  disc={disc:.3f}")

    eval_dir = _get_project_root() / "Evaluation"
    eval_dir.mkdir(exist_ok=True)
    json_path = eval_dir / "awa2_evaluation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(gt_list, f)
    print(f"[OK] Benchmark JSON: {json_path} ({len(gt_list)} queries)")


if __name__ == "__main__":
    # One-time build (no GPU): python src/awa2/generate_gt.py
    main()
