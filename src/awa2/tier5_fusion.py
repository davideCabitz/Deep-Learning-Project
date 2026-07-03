"""
AwA2 tier-5 champion — Bprime_listnet (BayesPhi Φ + ListNet on the Stage A′ core).

Retrains the CelebA champion (tier5_Bprime_listnet_seed0_full) on AwA2 and evaluates it
against the AwA2 CLIP baseline and CLAY. VM-ONLY: this trains Φ and reads ~30k train
features — never run locally.

Reuse strategy — Adapter, not fork. The tier4/tier5 modules are hard-bound to CelebA
(module constants N_ATTR=40, ATTRIBUTE_NAMES, ATTR_TO_IDX; CelebA train/test loaders in
tier1_GDE and a CelebA test-attr path in the diagnostic). Forking hundreds of lines of
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

from awa2 import data_loader as awa2_data
from awa2.data_loader import ATTRIBUTE_NAMES as AWA2_NAMES
from awa2.data_loader import N_ATTRS, _get_artifacts_dir


def configure_awa2() -> None:
    # THE adapter seam: rebind the CelebA-bound globals in tier4_bayes / tier5_cdbayes /
    # tier1_GDE to AwA2's providers BEFORE any of their functions run. Centralized here so
    # the coupling is visible in one place rather than scattered across call sites.
    import tier1_GDE
    import tier3_dgp
    import tier4_bayes
    import tier5_cdbayes as t5
    import vm_pipeline

    attr_to_idx = {name: i for i, name in enumerate(AWA2_NAMES)}

    # 1. Attribute cardinality + name/index tables (used for free-set + multi-hot builds,
    #    QueryGenerator's index→name mapping in tier3_dgp, and vm_pipeline.query_indices
    #    which evaluate_scorer calls to parse each benchmark query).
    for mod in (tier3_dgp, tier4_bayes, t5, vm_pipeline):
        mod.ATTRIBUTE_NAMES = AWA2_NAMES
    for mod in (tier4_bayes, t5, vm_pipeline):
        mod.ATTR_TO_IDX = attr_to_idx
    for mod in (tier4_bayes, t5):
        mod.N_ATTR = N_ATTRS

    # 2. Train-split providers (tier1_GDE's CelebA loaders → AwA2 split tensors). Both
    #    modules imported these by-reference; rebind on every namespace that holds them.
    def _train_feats():
        return awa2_data.load_image_features("train")

    def _train_attrs():
        return awa2_data.load_attributes("train")

    for mod in (tier1_GDE, tier4_bayes, t5):
        mod._load_train_features = _train_feats
        mod._load_train_attributes = _train_attrs

    # 3. Cache dir → artifacts/awa2/ so AwA2 probes/temperatures never clobber the CelebA
    #    caches (both modules write tier4_probe.pt / tier5_temperatures.pt by that helper).
    tier4_bayes._get_artifacts_dir = _get_artifacts_dir
    t5._get_artifacts_dir = _get_artifacts_dir

    # 4. Test-feature loader + test-attr diagnostic path (tier4 reads celeba_attributes_test.pt).
    tier4_bayes.load_image_features = lambda: awa2_data.load_image_features("test")

    def _awa2_probe_diag(P_test):
        attrs = awa2_data.load_attributes("test")
        acc = ((P_test.cpu() > 0.5).float() == attrs.float()).float().mean(0)
        worst = int(acc.argmin())
        tier4_bayes.log(f"  probe test acc: mean {acc.mean():.4f}  "
                        f"min {acc[worst]:.4f} ({AWA2_NAMES[worst]})")

    tier4_bayes.probe_accuracy_diagnostic = _awa2_probe_diag


def _awa2_gt():
    from eval import find_eval_json, load_eval_json
    return load_eval_json(find_eval_json().parent / "awa2_evaluation.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="AwA2 tier-5 champion (Bprime_listnet)")
    ap.add_argument("--quick", action="store_true", help="CPU smoke run (tiny epochs, 2 queries)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--force-probes", action="store_true")
    args = ap.parse_args()

    configure_awa2()

    # Imported AFTER configure_awa2 so their module-level reads see AwA2 bindings.
    from eval import format_results_table
    from results_saver import output_subdir, save_results_csv
    from tier3_train_utils import assert_db_frozen
    from tier4_bayes import (Tier4Config, load_or_train_probes,
                             probe_accuracy_diagnostic)
    from tier5_cdbayes import (StageAPrime, Tier5Config, build_chow_liu_tree,
                               calibrated_posteriors, load_or_fit_temperatures,
                               make_stage_b_prime_chunk, posteriors, train_phi_prime)
    from vm_pipeline import evaluate_scorer, log, save_training_curve, set_log_file

    cfg = Tier5Config()
    if args.epochs is not None:
        cfg = replace(cfg, tier4=replace(cfg.tier4, epochs=args.epochs))
    if args.quick:
        cfg = replace(cfg, tier4=replace(cfg.tier4, probe_epochs=2, epochs=2, chunk=64))

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = output_subdir("awa2")
    set_log_file(out_dir / "tier5_train_log.txt")
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
