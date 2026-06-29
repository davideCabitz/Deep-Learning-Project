"""
results_io — single owner of evaluation-result persistence (shared seam).

Every tier produces the same evaluate_all() dict and must write the same CSV layout so
output/*.csv files compare row-for-row. That responsibility lives here ONCE: tiers depend
on this module, never on each other. Pulling it out of tier0.py removes the DRY-violating
copies in tier0/tier0_enhanced and the sideways `from tier0 import …` coupling in tier1.
"""

import csv
from pathlib import Path


def output_dir():
    # Project-root output/ folder for saved CSVs (created on demand).
    out = Path(__file__).parent.parent / "output"
    out.mkdir(exist_ok=True)
    return out


def output_subdir(name):
    # A named subfolder of output/ (created on demand) so a method's many ablation CSVs
    # group under one folder instead of scattering across the shared output/ root.
    sub = output_dir() / name
    sub.mkdir(exist_ok=True)
    return sub


def save_results_csv(results, path, ks=(1, 5, 10)):
    # Persist evaluate_all() output: one row per query + a macro-MEAN row.
    # Columns query, R@k…, P@k…, num_sources — the one canonical layout so every
    # tier's CSV compares directly. MEAN weights each query equally (macro-average).
    cols = ["query"] + [f"R@{k}" for k in ks] + [f"P@{k}" for k in ks] + ["num_sources"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for query, res in results.items():
            writer.writerow(
                [query]
                + [f"{res[f'recall@{k}']:.4f}" for k in ks]
                + [f"{res[f'precision@{k}']:.4f}" for k in ks]
                + [res["num_sources"]]
            )
        n = len(results)
        mean_row = ["MEAN"]
        for metric in [f"recall@{k}" for k in ks] + [f"precision@{k}" for k in ks]:
            mean_row.append(f"{sum(r[metric] for r in results.values()) / n:.4f}")
        mean_row.append("")
        writer.writerow(mean_row)
    print(f"  Saved: {path}")
