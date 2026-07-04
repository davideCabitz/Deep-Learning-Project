"""
per_source — single owner of PER-SOURCE outcome dumps for the statistical evaluation.

The canonical CSVs (results_saver.py) keep only per-query macro-averages, which is enough
to rank methods but not to attach uncertainty to that ranking: bootstrap confidence
intervals and paired significance tests need the underlying per-source random variables
hit@K ∈ {0,1} and precision@K ∈ [0,1] (spec §3.1.3 metrics, evaluated per source image).
This module recomputes exactly those atoms through eval.recall_at_k / eval.precision_at_k
— the same functions evaluate_query() averages — so a dump can never disagree with the
published CSV, and enforces that with an explicit parity gate.

One .npz per model in output/per_source/. Per query, the arrays stored are:
    sources [n] int64, hit{K} [n] uint8, prec{K} [n] float32, top{TOP_KEEP} [n,10] int64
(top-10 retrieved indices feed the qualitative retrieval grids; K ≤ 10 per spec).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eval import precision_at_k, recall_at_k
from results_saver import output_subdir

KS = (1, 5, 10)          # spec §3.1.3 — the only K values reported anywhere
TOP_KEEP = 10            # retained ranking prefix; max(KS), enough for grids too
_FIELD_SEP = "::"        # npz key layout "<query>::<field>"


# Per-source atoms for one query: hit@K / precision@K per source + top-10 prefix.
# Same get_ranking seam and same atomic metric functions as eval.evaluate_query —
# the dump is the *pre-average* view of the identical computation, nothing new.
def per_source_outcomes(ground_truth: dict, get_ranking, ks=KS) -> dict[str, np.ndarray]:
    # Keys may be str (JSON) or int; normalize once, keep a handle to the original key.
    by_idx = {int(s): targets for s, targets in ground_truth.items()}
    sources = sorted(by_idx)
    if not sources:
        raise ValueError("ground_truth has no source images")
    n = len(sources)
    out: dict[str, np.ndarray] = {
        "sources": np.asarray(sources, dtype=np.int64),
        f"top{TOP_KEEP}": np.empty((n, TOP_KEEP), dtype=np.int64),
    }
    for k in ks:
        out[f"hit{k}"] = np.empty(n, dtype=np.uint8)
        out[f"prec{k}"] = np.empty(n, dtype=np.float32)

    for i, src in enumerate(sources):
        targets = {int(t) for t in by_idx[src]}
        ranking = get_ranking(src)
        if len(ranking) < TOP_KEEP:
            raise ValueError(f"ranking for source {src} shorter than {TOP_KEEP}")
        out[f"top{TOP_KEEP}"][i] = ranking[:TOP_KEEP]
        for k in ks:
            out[f"hit{k}"][i] = recall_at_k(ranking, targets, k)
            out[f"prec{k}"][i] = precision_at_k(ranking, targets, k)
    return out


# Full-benchmark dump: sweep every JSON query through per_source_outcomes.
# make_get_ranking(query_str) is the same seam eval.evaluate_all uses.
def dump_all(gt_list: list[dict], make_get_ranking, ks=KS) -> dict[str, dict[str, np.ndarray]]:
    return {
        entry["query"]: per_source_outcomes(entry["ground_truth"], make_get_ranking(entry["query"]), ks)
        for entry in gt_list
    }


# Parity gate — a dump is only trustworthy if its means reproduce the published CSV.
# Recomputes each query's macro metrics from the per-source arrays and compares against
# the CSV row (written at 4 decimals, so tolerance is half an ulp of that format —
# INCLUSIVE: a mean landing exactly on .xxxx5 is formatted half-to-even, putting the
# true error at exactly 5e-5, so the bound carries a float-eps margin).
def check_parity_with_csv(dump: dict[str, dict[str, np.ndarray]], csv_path: Path,
                          ks=KS, atol: float = 5.5e-5) -> None:
    import csv as _csv
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = {r["query"]: r for r in _csv.DictReader(f)}
    for query, arrays in dump.items():
        if query not in rows:
            raise ValueError(f"query {query!r} missing from {csv_path}")
        row = rows[query]
        for k in ks:
            pairs = ((f"R@{k}", arrays[f"hit{k}"].mean()),
                     (f"P@{k}", arrays[f"prec{k}"].mean()))
            for col, recomputed in pairs:
                published = float(row[col])
                if abs(recomputed - published) > atol:
                    raise AssertionError(
                        f"parity FAIL {csv_path.name} [{query}] {col}: "
                        f"csv={published:.4f} vs dump={recomputed:.6f}")
    print(f"  Parity OK vs {csv_path.name} ({len(dump)} queries)")


def per_source_dir() -> Path:
    return output_subdir("per_source")


def save_per_source_npz(model_name: str, dump: dict[str, dict[str, np.ndarray]],
                        path: Path | None = None) -> Path:
    # Flatten to "<query>::<field>" keys; query list kept in a JSON manifest entry so
    # loading restores the exact benchmark order without parsing key strings' order.
    path = path or per_source_dir() / f"{model_name}.npz"
    flat: dict[str, np.ndarray] = {}
    for query, arrays in dump.items():
        if _FIELD_SEP in query:
            raise ValueError(f"query string contains reserved separator: {query!r}")
        for field, arr in arrays.items():
            flat[f"{query}{_FIELD_SEP}{field}"] = arr
    flat["__manifest__"] = np.frombuffer(
        json.dumps({"model": model_name, "queries": list(dump)}).encode(), dtype=np.uint8)
    np.savez_compressed(path, **flat)
    print(f"  Saved per-source dump: {path}")
    return path


def load_per_source(path: Path) -> dict[str, dict[str, np.ndarray]]:
    with np.load(path) as z:
        manifest = json.loads(bytes(z["__manifest__"]).decode())
        dump: dict[str, dict[str, np.ndarray]] = {q: {} for q in manifest["queries"]}
        for key in z.files:
            if key == "__manifest__":
                continue
            query, field = key.split(_FIELD_SEP, 1)
            dump[query][field] = z[key]
    return dump
