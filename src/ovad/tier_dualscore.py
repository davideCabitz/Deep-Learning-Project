"""
OVAD tier-3 DualScore — CLIP-space trained challenger on the instance-crop benchmark.

Same architecture/training as src/tier3_dualscore.py (two-stage: masked-BCE probes, then
Φ⁺+FusionHead with composite ListNet), rebound to OVAD via the adapter. OVAD deltas all
live in the seams, not here: tri-state labels flow through the masked probe loss and the
TriStateQueryGenerator (adapter); attribute lookup uses the dual raw/underscored key map
(training uses raw names from the sampler, eval uses parse_query's underscored tokens —
raw_stacks is dual-keyed below for the same reason).

Requires (VM): artifacts/ovad/clip_image_features.pt, clip_attr_prompt_bank.pt, and
visual_directions.pt (run src/ovad/tier_gde.py first — DualScore's Φ⁺ shares GDE's μ).

Run (VM):  python src/ovad/tier_dualscore.py
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from ovad import data_loader as ovad_data
from ovad.adapter import configure_dualscore
from ovad.data_loader import N_ATTRS


def main() -> None:
    ap = argparse.ArgumentParser(description="OVAD tier-3 DualScore (two-stage training)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None, help="override Stage-2 epochs")
    args = ap.parse_args()

    configure_dualscore()

    # Imported AFTER configure_dualscore so by-value imports see the OVAD bindings.
    from dataclasses import replace
    import tier3_dualscore as ds
    from eval import evaluate_all, find_eval_json, format_results_table, load_eval_json
    from results_saver import output_subdir, save_results_csv
    from tier2d_dgp import compute_mu_txt
    from tier3_dgp import build_raw_stacks
    from tier3_hybrid import log, train_probes
    from ovad.clip_prompts import load_prompt_bank
    from ovad.tier_gde import load_or_mine_directions

    # tier3_dualscore's own name tables were bound at ITS import — rebind to OVAD.
    from ovad.adapter import OVAD_ATTR_TO_IDX_DUAL
    ds.ATTR_TO_IDX = OVAD_ATTR_TO_IDX_DUAL

    cfg = ds.DualScoreConfig(seed=args.seed)
    if args.epochs is not None:
        cfg = replace(cfg, epochs=args.epochs)
    torch.manual_seed(cfg.seed)
    dev = ds.DEVICE
    log(f"OVAD DualScore  device={dev}  seed={cfg.seed}  phi_epochs={cfg.epochs}")

    train_feats = ovad_data.load_image_features("train").to(dev)
    train_attrs = ovad_data.load_attributes("train")
    prompt_bank = load_prompt_bank()
    mu, _ = load_or_mine_directions()          # GDE cache (artifacts/ovad, tri-state mining)
    mu_txt = compute_mu_txt(prompt_bank)
    raw_stacks = {k: v.to(dev) for k, v in build_raw_stacks(prompt_bank).items()}
    assert train_attrs.shape[1] == N_ATTRS and len(raw_stacks) == N_ATTRS
    # Training indexes stacks by RAW names, eval by parse_query's underscored tokens.
    raw_stacks.update({k.replace(" ", "_"): v for k, v in list(raw_stacks.items())})

    log("STAGE 1 - attribute probes (masked BCE on frozen OVAD train features)")
    probes, _ = train_probes(train_feats, train_attrs, cfg)

    model = ds.DualScoreRetriever(cfg, mu.to(dev), mu_txt.to(dev)).to(dev)
    log("STAGE 2 - positive fusion + FusionHead (composite ListNet, probes frozen)")
    ds.train_dualscore(model, probes, train_feats, train_attrs, raw_stacks, cfg)

    out_dir = output_subdir("ovad")
    tag = f"dualscore_seed{cfg.seed}"
    torch.save(model.state_dict(), out_dir / f"tier3_{tag}_phi.pt")
    torch.save(probes.state_dict(), out_dir / f"tier3_{tag}_probes.pt")

    # Evaluate on the OVAD benchmark (tier3_dualscore.evaluate_dualscore is bound to the
    # CelebA GT path, so the eval loop lives here; the ranking seam is reused as-is).
    image_features = ovad_data.load_image_features("test").to(dev)
    probes.eval()
    with torch.no_grad():
        db_probs = torch.sigmoid(probes(image_features))
    gt_list = load_eval_json(find_eval_json().parent / "ovad_evaluation.json")

    def make(query_str):
        return ds.make_get_ranking(query_str, image_features, db_probs, model, raw_stacks)

    results = evaluate_all(gt_list, make)
    print(f"\nOVAD DualScore (seed {cfg.seed}) - {len(gt_list)} queries\n")
    print(format_results_table(results))
    csv_path = out_dir / f"tier3_{tag}.csv"
    save_results_csv(results, csv_path)

    from per_source import check_parity_with_csv, dump_all, save_per_source_npz
    dump = dump_all(gt_list, make)
    check_parity_with_csv(dump, csv_path)
    ps_dir = out_dir / "per_source"
    ps_dir.mkdir(exist_ok=True)
    save_per_source_npz(f"ovad_tier3_{tag}", dump, ps_dir / f"ovad_tier3_{tag}.npz")


if __name__ == "__main__":
    # VM-only (trains Φ): python src/ovad/tier_dualscore.py
    main()
