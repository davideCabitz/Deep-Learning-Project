"""
OVAD multi-model statistical comparison — champion vs every reference arm.

Loads the per-source NPZ dumps written by the tier runners and reports, per metric:
  - bootstrap 95% CI on each model's macro-mean (stratified_bootstrap);
  - paired-bootstrap Δ + two-sided p for champion − each reference,
    Holm–Bonferroni-corrected across the {metrics × pairs} family.
Reuses src/stats_eval.py unchanged — the exact protocol as the CelebA/AwA2 studies.

Run (after the tier runners have produced their NPZs):
  python src/ovad/compare.py [--champion NAME]
"""

import argparse
import sys
from pathlib import Path

# Force UTF-8 stdout so the Δ/− glyphs print on a Windows cp1252 console too.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / "Baselines"))
sys.path.insert(0, str(Path(__file__).parents[1] / "Training-Free"))
sys.path.insert(0, str(Path(__file__).parents[1] / "Trained"))

from per_source import load_per_source
from results_saver import output_subdir
from stats_eval import (holm_bonferroni, metric_label, paired_bootstrap_diff,
                        stratified_bootstrap)

# All OVAD comparison models. Champion defaults to LDE (the AwA2 winner) pending OVAD
# results — override with --champion once the numbers land.
MODELS = {
    "LDE": "ovad_tier_lde.npz",
    "GDE": "ovad_tier_gde.npz",
    "CLIP baseline": "ovad_tier0_baseline.npz",
    "DGP tau100": "ovad_tier2d_tau100.0_a1.0_center.npz",
    "DualScore": "ovad_tier3_dualscore_seed0.npz",
    "CLAY": "ovad_tier_clay.npz",
    "Bprime_listnet": "ovad_tier5_bprime_listnet.npz",
}
DEFAULT_CHAMPION = "LDE"


def _load_dumps(champion):
    # Missing dumps are skipped with a warning (e.g. a VM-trained model's NPZ not yet
    # pulled) so the comparison over the available arms still runs. The champion is
    # mandatory — without it there is nothing to test against.
    ps_dir = output_subdir("ovad") / "per_source"
    dumps = {}
    for name, fname in MODELS.items():
        path = ps_dir / fname
        if not path.exists():
            print(f"[!] Skipping {name}: missing {path}")
            continue
        dumps[name] = load_per_source(path)
    if champion not in dumps:
        raise FileNotFoundError(f"Champion dump missing: {MODELS[champion]} — "
                                f"run its tier runner first.")
    return dumps


def main():
    ap = argparse.ArgumentParser(description="OVAD statistical comparison")
    ap.add_argument("--champion", default=DEFAULT_CHAMPION, choices=list(MODELS))
    args = ap.parse_args()
    champion = args.champion
    dumps = _load_dumps(champion)

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
    for ref in (n for n in dumps if n != champion):
        d = paired_bootstrap_diff(dumps[champion], dumps[ref])
        diffs[ref] = d
        for field, res in d.items():
            pvals[f"{champion}-{ref}::{field}"] = res.p_value
    adj = holm_bonferroni(pvals)

    print("\n" + "=" * 72)
    print(f"Paired Δ vs {champion} (Holm-corrected p)")
    print("=" * 72)
    for ref, d in diffs.items():
        print(f"\n{champion} − {ref}")
        for field, res in d.items():
            p_adj = adj[f"{champion}-{ref}::{field}"]
            sig = "*" if p_adj < 0.05 else " "
            print(f"  {metric_label(field):<8s} Δ={res.delta:+.4f}  "
                  f"[{res.ci_lo:+.4f}, {res.ci_hi:+.4f}]  "
                  f"rel={res.rel_gain:+.1%}  p={p_adj:.4f} {sig}")


if __name__ == "__main__":
    main()
