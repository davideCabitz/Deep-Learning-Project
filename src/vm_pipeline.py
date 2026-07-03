"""
vm_pipeline — single owner of the VM-run operational seams shared by the tier pipelines.

Extracted from tier4_bayes.py so tier5_cdbayes reuses the timestamped logging, the chunked
evaluation driver, the training-curve plot and the results-bundle mechanics by import
instead of copy-paste. Pure ops, no science: scoring math stays in the tier modules.
"""

from __future__ import annotations

import tarfile
import time
from pathlib import Path

import torch

from data_loader import ATTRIBUTE_NAMES, ATTR_TO_IDX
from eval import evaluate_query, parse_query

try:                                        # optional: only for the training-curve PNG
    import matplotlib
    matplotlib.use("Agg")                   # headless: render to file, never open a window
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ModuleNotFoundError:
    HAVE_MPL = False


TOP_K_KEEP = 100    # rankings cached to top-100: metrics need K ≤ 10 (spec §3.1.3)

_LOG_FILE: Path | None = None


def set_log_file(path: Path | None) -> None:
    # Route log() into a run-specific file (each pipeline's out_dir/train_log.txt).
    global _LOG_FILE
    _LOG_FILE = path


def log(msg: str) -> None:
    # Timestamped, flushed, tee'd to the run log — progress must be visible live over SSH.
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _LOG_FILE is not None:
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def query_indices(query_str: str) -> tuple[list[int], list[int], list[int]]:
    # Parse a query into (positive idx, negative idx, free idx) over the 40 attributes.
    pos, neg = parse_query(query_str)
    pi = [ATTR_TO_IDX[a] for a in pos]
    ni = [ATTR_TO_IDX[b] for b in neg]
    free = [c for c in range(len(ATTRIBUTE_NAMES)) if c not in set(pi) | set(ni)]
    return pi, ni, free


def evaluate_scorer(name: str, score_chunk, gt_list: list[dict], chunk_size: int,
                    dump: dict | None = None) -> dict:
    # Score every JSON query through eval.evaluate_query (the project ruler), computing
    # score matrices in source-chunks for GPU efficiency. score_chunk(srcs, pi, ni, free)
    # → [S, N]. Rankings cached to top-100 (metrics need K ≤ 10). Self-exclusion per §5.
    # Pass `dump={}` to also collect per-source outcomes (per_source.py) from the SAME
    # rankings — the statistics suite needs the pre-average atoms, not just the means.
    results = {}
    t0 = time.time()
    for qi, entry in enumerate(gt_list):
        pi, ni, free = query_indices(entry["query"])
        srcs = [int(s) for s in entry["ground_truth"].keys()]
        rankings: dict[int, list[int]] = {}
        for lo in range(0, len(srcs), chunk_size):
            chunk = srcs[lo:lo + chunk_size]
            scores = score_chunk(chunk, pi, ni, free)                    # [S, N]
            rows = torch.arange(len(chunk), device=scores.device)
            cols = torch.as_tensor(chunk, device=scores.device)
            scores[rows, cols] = float("-inf")                           # §5 self-exclusion
            top = torch.topk(scores, TOP_K_KEEP, dim=1).indices.cpu()
            rankings.update({s: top[j].tolist() for j, s in enumerate(chunk)})
        results[entry["query"]] = evaluate_query(entry["ground_truth"], rankings.__getitem__)
        if dump is not None:
            from per_source import per_source_outcomes  # lazy: numpy-only consumers stay light
            dump[entry["query"]] = per_source_outcomes(entry["ground_truth"], rankings.__getitem__)
        r = results[entry["query"]]
        log(f"  [{name}] {qi + 1:02d}/{len(gt_list)}  {entry['query']:<46s}"
            f"  R@5={r['recall@5']:.4f}  ({time.time() - t0:.0f}s)")
    return results


def save_training_curve(curve: list[float], path: Path, title: str, ylabel: str) -> None:
    # Headless per-epoch loss plot; silently skipped when matplotlib is absent on the VM.
    if not HAVE_MPL or not curve:
        return
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(curve) + 1), curve)
    plt.xlabel("epoch"), plt.ylabel(ylabel), plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def bundle_results(root: Path, bundle_path: Path, payload: list[Path],
                   pull_hints: list[str]) -> Path:
    # One tar.gz with repo-root-relative arcnames, so extracting at any clone's root
    # restores results in place. payload items may be files or directories.
    with tarfile.open(bundle_path, "w:gz") as tar:
        for item in payload:
            if item.is_dir():
                files = sorted(p for p in item.rglob("*") if p.is_file())
            else:
                files = [item] if item.exists() else []
            for p in files:
                tar.add(p, arcname=str(p.relative_to(root)))
    log(f"Bundle written: {bundle_path}")
    for line in pull_hints:
        log(line)
    return bundle_path
