"""
Ground-truth generator for AwA2 — the capability CelebA got for free (instructor JSON).

Reproduces the spec §3.1.1 relaxed-Hamming protocol on AwA2. Because AwA2 attributes are
CLASS-level, ground truth collapses to a 50×50 class relation, then expands to images:

  For query (T⁺, T⁻) and source image with class c_s:
    valid target CLASS c_t  iff  (1) c_t strictly satisfies every ± constraint, AND
                                  (2) Hamming(class_attr[c_s], class_attr[c_t]) ≤ 2 on the
                                      NON-queried attributes (identity preserved).
    valid target IMAGES = all test images whose class is a valid target class.

Viability (spec §3.1.2): keep a source only if it has ≥ MIN_TARGETS valid target images.
To keep the JSON and eval tractable we subsample sources and cap targets per source.

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

from eval import parse_query
from awa2.data_loader import (ATTR_TO_IDX, ATTRIBUTE_NAMES, N_ATTRS,
                              _get_project_root, load_class_attribute_matrix,
                              load_class_labels, load_class_names)

HAMMING_MAX = 2          # spec §3.1.1 — identity-preservation radius on free attributes
MIN_TARGETS = 5          # spec §3.1.2 — statistical-significance floor per source
MAX_SOURCES_PER_QUERY = 60   # subsample cap (fixed seed) — keeps eval tractable
MAX_TARGETS_PER_SOURCE = 500 # cap target list length per source
MAX_DISCRIMINATIVE = 0.30    # reject queries where > this fraction of corpus is a target
SUBSAMPLE_SEED = 7

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
    "+swims & -ocean",
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


def _valid_target_classes(class_matrix, pi, ni, free, src_class):
    # Boolean mask over the 50 classes: strict ± satisfaction + Hamming ≤ 2 on free attrs
    # relative to src_class. class_matrix is [50, 85] in {0,1}.
    strict = torch.ones(class_matrix.shape[0], dtype=torch.bool)
    for a in pi:
        strict &= class_matrix[:, a] == 1
    for b in ni:
        strict &= class_matrix[:, b] == 0
    if free:
        free_cols = torch.tensor(free, dtype=torch.long)
        ham = (class_matrix[:, free_cols] != class_matrix[src_class, free_cols]).sum(dim=1)
        strict &= ham <= HAMMING_MAX
    return strict


def build_ground_truth(query_str, class_matrix, test_labels, gen):
    # {src_test_idx: [target_test_idx, ...]} for one query, over the TEST split.
    # Indices are split-local (rows of the test feature/attribute tables) — the same
    # index space every AwA2 tier scores against.
    pi, ni, free = query_indices(query_str)
    n_classes = class_matrix.shape[0]

    # Per-class list of test-image indices (split-local).
    class_to_imgs = {c: (test_labels == c).nonzero(as_tuple=True)[0] for c in range(n_classes)}

    ground_truth = {}
    for src_class in range(n_classes):
        valid_mask = _valid_target_classes(class_matrix, pi, ni, free, src_class)
        target_imgs = torch.cat([class_to_imgs[c] for c in range(n_classes) if valid_mask[c]]) \
            if valid_mask.any() else torch.empty(0, dtype=torch.long)
        if len(target_imgs) < MIN_TARGETS:
            continue

        sources = class_to_imgs[src_class]
        # Source self-exclusion happens at scoring time (§5); a source may sit inside the
        # target class, so remove it from its own target list here for a clean GT set.
        for s in sources.tolist():
            tgt = target_imgs[target_imgs != s]
            if len(tgt) < MIN_TARGETS:
                continue
            if len(tgt) > MAX_TARGETS_PER_SOURCE:
                perm = torch.randperm(len(tgt), generator=gen)[:MAX_TARGETS_PER_SOURCE]
                tgt = tgt[perm]
            ground_truth[str(s)] = sorted(tgt.tolist())

    # Subsample sources for tractability (deterministic).
    if len(ground_truth) > MAX_SOURCES_PER_QUERY:
        keys = list(ground_truth.keys())
        pick = torch.randperm(len(keys), generator=gen)[:MAX_SOURCES_PER_QUERY].tolist()
        ground_truth = {keys[i]: ground_truth[keys[i]] for i in sorted(pick)}
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
        gen = torch.Generator().manual_seed(SUBSAMPLE_SEED)
        gt = build_ground_truth(q, class_matrix, test_labels, gen)
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
        gen = torch.Generator().manual_seed(SUBSAMPLE_SEED)
        gt = build_ground_truth(q, class_matrix, test_labels, gen)
        disc = 0.0
        if gt:
            disc = (sum(len(v) for v in gt.values()) / len(gt)) / len(test_labels)
        if not gt:
            print(f"  [!] SKIP {q}: no source has ≥{MIN_TARGETS} targets")
            continue
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
