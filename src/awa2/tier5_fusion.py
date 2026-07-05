"""
AwA2 tier-5 champion — Bprime_listnet (BayesPhi Φ + ListNet on the Stage A′ core).

Retrains the CelebA champion (tier5_Bprime_listnet_seed0_full) on AwA2 and evaluates it
against the AwA2 CLIP baseline and CLAY. VM-ONLY: this trains Φ and reads ~30k train
features — never run locally.

Reuse strategy — Adapter, not fork. The tier4/tier5 modules are hard-bound to CelebA
(module constants N_ATTR=40, ATTRIBUTE_NAMES, ATTR_TO_IDX; CelebA train/test loaders in
gde and a CelebA test-attr path in the diagnostic). Forking hundreds of lines of
calibration + Chow-Liu + Stage-A′ math would massively duplicate; editing the constants
in place would break the CelebA pipeline. Instead `configure_awa2()` injects AwA2's
attribute count, names, index map, and data providers into those modules through ONE
explicit, documented seam, then the existing champion training/eval path runs unchanged.

Requires (VM): artifacts/awa2/clip_image_features_{train,test}.pt.

Run (VM):  python src/awa2/tier5_fusion.py            # full champion run
           python src/awa2/tier5_fusion.py --quick    # CPU smoke (tiny epochs, 2 queries)
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / "Baselines"))
sys.path.insert(0, str(Path(__file__).parents[1] / "Training-Free"))
sys.path.insert(0, str(Path(__file__).parents[1] / "Trained"))

from awa2 import data_loader as awa2_data
from awa2.adapter import configure_tier5 as configure_awa2
from awa2.data_loader import _get_artifacts_dir


def _awa2_gt():
    from eval import find_eval_json, load_eval_json
    return load_eval_json(find_eval_json().parent / "awa2_evaluation.json")


# λ sweep for the --diagnose mode. λ=0 is the Stage A′ core alone (isolates probe-noise
# deficit vs training deficit); λ>0 adds the C4 identity channel λ·cos(v_x, v_ref).
DIAG_LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def run_diagnosis(cfg, feats, P, tree, gt_list, out_dir) -> None:
    # ORACLE DIAGNOSTIC — training-free. Evaluates the Stage A′ core at each λ directly on
    # the benchmark to locate tier5's AwA2 deficit: if λ=0 ≈ the trained Φ score, training
    # is not the problem (the Bayes core is); if λ>0 closes the gap to the CLIP baseline,
    # the missing ingredient is the identity channel. λ is swept on TEST, so cite the
    # CURVE as mechanism evidence in the report, never the argmax as a tuned result.
    from eval import format_results_table
    from results_saver import save_results_csv
    from cdbayes import StageAPrime
    from vm_pipeline import evaluate_scorer, log

    summary = []
    for lam in DIAG_LAMBDAS:
        scorer = StageAPrime(feats, P, tree, conditioned_tail=True,
                             lam=lam, hamming_max=cfg.tier4.hamming_max)
        name = f"Aprime_lam{lam:g}"
        results = evaluate_scorer(name, scorer.chunk, gt_list, cfg.tier4.chunk)
        print(f"\n=== Stage A' core + {lam:g}*cos ===")
        print(format_results_table(results))
        save_results_csv(results, out_dir / f"tier5_diag_{name}.csv")
        mean_r1 = sum(r["recall@1"] for r in results.values()) / len(results)
        mean_r5 = sum(r["recall@5"] for r in results.values()) / len(results)
        summary.append((lam, mean_r1, mean_r5))

    log("DIAGNOSIS SUMMARY — Stage A' core + lambda*cos (oracle sweep, cite the curve)")
    log(f"  {'lambda':>8s}  {'R@1':>7s}  {'R@5':>7s}")
    for lam, r1, r5 in summary:
        log(f"  {lam:8g}  {r1:7.4f}  {r5:7.4f}")
    log("  reference points: trained Bprime_listnet R@1=0.6667 | CLAY 0.7306 | "
        "CLIP baseline 0.8083 (output/awa2/*.csv)")


def main() -> None:
    ap = argparse.ArgumentParser(description="AwA2 tier-5 champion (Bprime_listnet)")
    ap.add_argument("--quick", action="store_true", help="CPU smoke run (tiny epochs, 2 queries)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--force-probes", action="store_true")
    ap.add_argument("--diagnose", action="store_true",
                    help="training-free λ-sweep of the Stage A' core (locates the AwA2 "
                         "deficit: probe noise vs missing identity channel); no Φ training")
    args = ap.parse_args()

    configure_awa2()

    # Imported AFTER configure_awa2 so their module-level reads see AwA2 bindings.
    from eval import format_results_table
    from results_saver import output_subdir, save_results_csv
    from fusion_train_utils import assert_db_frozen
    from bap import (BAPConfig, load_or_train_probes,
                             probe_accuracy_diagnostic)
    from cdbayes import (StageAPrime, CDBayesConfig, build_chow_liu_tree,
                               calibrated_posteriors, load_or_fit_temperatures,
                               make_stage_b_prime_chunk, posteriors, train_phi_prime)
    from vm_pipeline import evaluate_scorer, log, save_training_curve, set_log_file

    cfg = CDBayesConfig()
    if args.epochs is not None:
        cfg = replace(cfg, tier4=replace(cfg.tier4, epochs=args.epochs))
    if args.quick:
        cfg = replace(cfg, tier4=replace(cfg.tier4, probe_epochs=2, epochs=2, chunk=64))

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = output_subdir("awa2")
    set_log_file(out_dir / ("tier5_diag_log.txt" if args.diagnose else "tier5_train_log.txt"))
    log(f"Device: {dev}")

    # Stage 0 — probes; C1 — temperature calibration; C2 — Chow-Liu tree (all on AwA2 train).
    probe = load_or_train_probes(cfg.tier4, dev, force=args.force_probes)
    feats = awa2_data.load_image_features("test").to(dev)
    probe_accuracy_diagnostic(posteriors(probe, feats))
    tau = load_or_fit_temperatures(probe, cfg, dev)
    tree = build_chow_liu_tree(awa2_data.load_attributes("train"), cfg.laplace_alpha).to(dev)

    gt_list = _awa2_gt()
    if args.quick:
        gt_list = gt_list[:2]

    # Stage A′ core scorer (C1+C2+C3, λ=0) — the exact base Φ trains on. P carries C1.
    P = calibrated_posteriors(probe, feats, tau)

    if args.diagnose:
        run_diagnosis(cfg, feats, P, tree, gt_list, out_dir)
        return

    scorer = StageAPrime(feats, P, tree, conditioned_tail=True,
                         lam=0.0, hamming_max=cfg.tier4.hamming_max)

    # Stage B′ — BayesPhi + ListNet (the champion arm), seed 0.
    log("Stage B' — BayesPhi + ListNet (Bprime_listnet, seed 0)")
    model, curve = train_phi_prime(cfg, probe, tau, tree, dev, loss_name="listnet",
                                   use_qv=True, seed=0,
                                   max_steps=5 if args.quick else None)
    torch.save(model.state_dict(),
               _get_artifacts_dir() / "tier5_bprime_listnet_seed0.pt")
    save_training_curve(curve, out_dir / "tier5_training_curve.png",
                        "AwA2 tier-5 Bprime_listnet", "ListNet loss")

    score_chunk = make_stage_b_prime_chunk(model, scorer, use_qv=True)
    assert_db_frozen(feats, lambda s: _rank(score_chunk, s, gt_list[0]))
    results = evaluate_scorer("Bprime_listnet", score_chunk, gt_list, cfg.tier4.chunk)
    print("\n" + format_results_table(results))
    csv_path = out_dir / "tier5_bprime_listnet.csv"
    save_results_csv(results, csv_path)

    _dump_per_source(score_chunk, gt_list, csv_path, cfg.tier4.chunk)


def _rank(score_chunk, src_idx, entry):
    # Single-source ranking adapter for the frozen-DB invariant check.
    from awa2.generate_gt import query_indices
    pi, ni, free = query_indices(entry["query"])
    scores = score_chunk([src_idx], pi, ni, free).squeeze(0)
    scores[src_idx] = float("-inf")
    return torch.argsort(scores, descending=True).tolist()


def _dump_per_source(score_chunk, gt_list, csv_path, chunk):
    # Per-source NPZ for the stats suite, parity-gated against the CSV via evaluate_scorer's
    # own dump path (identical rankings by construction).
    from per_source import check_parity_with_csv, save_per_source_npz
    from results_saver import output_subdir
    dump: dict = {}
    from vm_pipeline import evaluate_scorer
    evaluate_scorer("Bprime_listnet(dump)", score_chunk, gt_list, chunk, dump=dump)
    check_parity_with_csv(dump, csv_path)
    ps_dir = output_subdir("awa2") / "per_source"
    ps_dir.mkdir(exist_ok=True)
    save_per_source_npz("awa2_tier5_bprime_listnet", dump,
                        ps_dir / "awa2_tier5_bprime_listnet.npz")


if __name__ == "__main__":
    # VM-only: python src/awa2/tier5_fusion.py
    main()
