"""Evaluation engine — compositional image retrieval.

This module is the **ruler** of the project: every method (Tier-0 baseline, CLAY,
our fusion module) is scored through the functions here, so they all get measured the
same way.

Metric definitions are taken **verbatim** from the assignment spec
(`documents/project_specification.md`, section 3.1.3):

- **Recall@K (hit rate)** — the *primary* metric. ``1`` if the top-K share at least one
  image with the ground-truth set ``G``, else ``0``. (This is a hit-rate, **not**
  textbook recall ``|hits|/|relevant|``.)
- **Precision@K** — secondary. ``|top-K ∩ G| / K``.

Both are computed per source image, then **averaged over all valid sources** of a query.

All image identifiers are **dataset indices** (ints), per `CONTRACT.md` §0. A *ranking*
is a ``list[int]`` of dataset indices, best match first, with the source image already
removed (`CONTRACT.md` §5).

A retrieval *method* plugs in through a ``get_ranking(source_idx) -> ranking`` callback.
"""

from __future__ import annotations

import json
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Locating the evaluation JSON
# ---------------------------------------------------------------------------
def find_eval_json(start: Path | None = None) -> Path:
    # Walk upward from this file (default) to find Evaluation/celeba_evaluation.json,
    # so imports resolve regardless of the caller's CWD. Pass `start` to override.
    here = (start or Path(__file__).parent).resolve()
    for folder in [here, *here.parents]:
        candidate = folder / 'Evaluation' / 'celeba_evaluation.json'
        if candidate.exists():
            return candidate
    raise FileNotFoundError('Could not find Evaluation/celeba_evaluation.json; set the path manually.')


# ---------------------------------------------------------------------------
# 2. Per-source metrics — the two atomic functions. Everything else averages these.
# ---------------------------------------------------------------------------
def recall_at_k(ranking: list[int], ground_truth: set[int], k: int) -> float:
    # Recall@K (primary metric) — hit rate: 1.0 if top-K ∩ G ≠ ∅, else 0.0.
    # (NOT textbook recall |hits|/|relevant| — spec §3.1.3 defines it as a hit rate.)
    topk = ranking[:k]
    return 1.0 if (set(topk) & ground_truth) else 0.0


def precision_at_k(ranking: list[int], ground_truth: set[int], k: int) -> float:
    # Precision@K (secondary) — |top-K ∩ G| / K. Denominator is K, not |G|.
    topk = ranking[:k]
    hits = len(set(topk) & ground_truth)
    return hits / k


# ---------------------------------------------------------------------------
# 3. Per-query and full-benchmark scoring
# ---------------------------------------------------------------------------
def evaluate_query(ground_truth: dict, get_ranking, ks=(1, 5, 10)) -> dict:
    # Score one query, averaging each metric over all of its valid source images.
    sums = {f'recall@{k}': 0.0 for k in ks}
    sums.update({f'precision@{k}': 0.0 for k in ks})

    sources = list(ground_truth.keys())
    for src in sources:
        src_idx = int(src)                       # keys may be strings -> int
        targets = {int(t) for t in ground_truth[src]}
        ranking = get_ranking(src_idx)
        for k in ks:
            sums[f'recall@{k}'] += recall_at_k(ranking, targets, k)
            sums[f'precision@{k}'] += precision_at_k(ranking, targets, k)

    n = len(sources)
    results = {m: (total / n if n else 0.0) for m, total in sums.items()}
    results['num_sources'] = n
    return results


def evaluate_all(gt_list: list[dict], make_get_ranking, ks=(1, 5, 10)) -> dict:
    # Score every query in the eval JSON. `make_get_ranking(query_str)` yields the
    # per-query get_ranking callback — this is the seam every method plugs in through.
    out = {}
    for entry in gt_list:
        query_str = entry['query']
        get_ranking = make_get_ranking(query_str)
        out[query_str] = evaluate_query(entry['ground_truth'], get_ranking, ks)
    return out


# ---------------------------------------------------------------------------
# 4. Helpers — load JSON, parse queries, pretty table
# ---------------------------------------------------------------------------
def load_eval_json(path) -> list[dict]:
    # Load the authoritative evaluation JSON (a list of query dicts).
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_query(query_str: str) -> tuple[list[str], list[str]]:
    # Parse a query string into (positive_attrs, negative_attrs).
    # Accepts JSON style ('+Black_Hair, -Wavy_Hair') and spec-table style
    # ('+ Black Hair & - Wavy Hair'); names normalize to underscores ('Black_Hair').
    pos, neg = [], []
    for piece in query_str.replace('&', ',').split(','):
        piece = piece.strip()
        if not piece:
            continue
        sign, rest = piece[0], piece[1:].strip()
        name = rest.replace(' ', '_')
        if sign == '+':
            pos.append(name)
        elif sign == '-':
            neg.append(name)
        else:
            raise ValueError(f'Query piece must start with + or -: {piece!r}')
    return pos, neg


def format_results_table(all_results: dict, ks=(1, 5, 10)) -> str:
    # Render evaluate_all() output as a column-aligned console table.
    cols = ['query'] + [f'R@{k}' for k in ks] + [f'P@{k}' for k in ks] + ['#src']
    rows = [cols]
    for query, res in all_results.items():
        rows.append(
            [query]
            + [f"{res[f'recall@{k}']:.3f}" for k in ks]
            + [f"{res[f'precision@{k}']:.3f}" for k in ks]
            + [str(res['num_sources'])]
        )
    widths = [max(len(r[i]) for r in rows) for i in range(len(cols))]
    lines = []
    for r, row in enumerate(rows):
        lines.append('  '.join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
        if r == 0:
            lines.append('  '.join('-' * w for w in widths))
    return '\n'.join(lines)
