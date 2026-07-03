"""
AwA2 three-way statistical comparison — champion vs the two references.

Loads the per-source NPZ dumps written by the tier runners and reports, per metric:
  - bootstrap 95% CI on each model's macro-mean (stratified_bootstrap);
  - paired-bootstrap Δ + two-sided p for champion − baseline and champion − CLAY,
    Holm–Bonferroni-corrected across the {metrics × pairs} family.
Reuses src/stats_eval.py unchanged — the exact protocol as the CelebA study.

Run (after the three tier runners have produced their NPZs):
  python src/awa2/compare.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from per_source import load_per_source
from results_saver import output_subdir
from stats_eval import (holm_bonferroni, metric_label, paired_bootstrap_diff,
                        stratified_bootstrap)

MODELS = {
    "CLIP baseline": "awa2_tier0_baseline.npz",
    "CLAY": "awa2_tier_clay.npz",
    "Bprime_listnet": "awa2_tier5_bprime_listnet.npz",
}
CHAMPION = "Bprime_listnet"


def _load_dumps():
    ps_dir = output_subdir("awa2") / "per_source"
    dumps = {}
    for name, fname in MODELS.items():
        path = ps_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing per-source dump for {name}: {path}\n"
                                    f"Run the corresponding tier runner first.")
        dumps[name] = load_per_source(path)
    return dumps


def main():
    dumps = _load_dumps()

    print("=" * 72)
    print("Per-model macro means with bootstrap 95% CI")
    print("=" * 72)
    for name, dump in dumps.items():
        boot = stratified_bootstrap(dump)
        print(f"\n{name}")
        for field, res in boot.items():
            print(f"  {metric_label(field):<8s} {res.point:.4f}  "
                  f"[{res.ci_lo:.4f}, {res.ci_hi:.4f}]")

    # Paired significance: champion vs each reference; Holm across the whole family.
    pvals, diffs = {}, {}
    for ref in (n for n in MODELS if n != CHAMPION):
        d = paired_bootstrap_diff(dumps[CHAMPION], dumps[ref])
        diffs[ref] = d
        for field, res in d.items():
            pvals[f"{CHAMPION}-{ref}::{field}"] = res.p_value
    adj = holm_bonferroni(pvals)

    print("\n" + "=" * 72)
    print(f"Paired Δ vs {CHAMPION} (Holm-corrected p)")
    print("=" * 72)
    for ref, d in diffs.items():
        print(f"\n{CHAMPION} − {ref}")
        for field, res in d.items():
            p_adj = adj[f"{CHAMPION}-{ref}::{field}"]
            sig = "*" if p_adj < 0.05 else " "
            print(f"  {metric_label(field):<8s} Δ={res.delta:+.4f}  "
                  f"[{res.ci_lo:+.4f}, {res.ci_hi:+.4f}]  "
                  f"rel={res.rel_gain:+.1%}  p={p_adj:.4f} {sig}")


if __name__ == "__main__":
    main()
