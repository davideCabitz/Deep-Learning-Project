"""
AwA2 tier-3 DualScore — CLIP-space trained challenger for the transfer study.

Why DualScore for AwA2: unlike tier5 (probe-mediated Bayes score — shown by the λ-sweep
diagnostic to be structurally disadvantaged here), DualScore's backbone score IS CLIP
cosine on a fused query q⁺, with probe channels only as query-conditioned additive
rewards (β, γ learned per query). At worst the head learns β,γ→0 and recovers the
identity-anchored fusion; at best it reweights direction steering on the queries where
text deltas are weak. Same architecture/training as src/tier3_dualscore.py (two-stage:
BCE probes, then Φ⁺+FusionHead with composite ListNet), rebound to AwA2 via the adapter.

Requires (VM): artifacts/awa2/clip_image_features_{train,test}.pt,
clip_attr_prompt_bank.pt, and visual_directions.pt (run src/awa2/tier_gde.py first —
DualScore's Φ⁺ shares GDE's mined μ).

Run (VM):  python src/awa2/tier_dualscore.py
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / "Baselines"))
sys.path.insert(0, str(Path(__file__).parents[1] / "Training-Free"))
sys.path.insert(0, str(Path(__file__).parents[1] / "Trained"))

from awa2 import data_loader as awa2_data
from awa2.adapter import configure_dualscore
from awa2.data_loader import N_ATTRS


def main() -> None:
    ap = argparse.ArgumentParser(description="AwA2 tier-3 DualScore (two-stage training)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None, help="override Stage-2 epochs")
    args = ap.parse_args()

    configure_dualscore()

    # Imported AFTER configure_dualscore so by-value imports see the AwA2 bindings.
    from dataclasses import replace
    import tier3_dualscore as ds
    from eval import find_eval_json, format_results_table, load_eval_json
    from results_saver import output_subdir, save_results_csv
    from tier1_GDE import load_or_mine_directions
    from tier2d_dgp import compute_mu_txt
    from tier3_dgp import build_raw_stacks
    from tier3_hybrid import log, train_probes
    from awa2.clip_prompts import load_prompt_bank

    # tier3_dualscore's own name tables were bound at ITS import — rebind to AwA2.
    from awa2.adapter import AWA2_ATTR_TO_IDX
    ds.ATTR_TO_IDX = AWA2_ATTR_TO_IDX

    cfg = ds.DualScoreConfig(seed=args.seed)
    if args.epochs is not None:
        cfg = replace(cfg, epochs=args.epochs)
    torch.manual_seed(cfg.seed)
    dev = ds.DEVICE
    log(f"AwA2 DualScore  device={dev}  seed={cfg.seed}  phi_epochs={cfg.epochs}")

    train_feats = awa2_data.load_image_features("train").to(dev)
    train_attrs = awa2_data.load_attributes("train")
    prompt_bank = load_prompt_bank()
    mu, _ = load_or_mine_directions()          # GDE cache (artifacts/awa2, via adapter)
    mu_txt = compute_mu_txt(prompt_bank)
    raw_stacks = {k: v.to(dev) for k, v in build_raw_stacks(prompt_bank).items()}
    assert train_attrs.shape[1] == N_ATTRS and len(raw_stacks) == N_ATTRS

    log("STAGE 1 - attribute probes (BCE on frozen AwA2 train features)")
    probes, _ = train_probes(train_feats, train_attrs, cfg)

    model = ds.DualScoreRetriever(cfg, mu.to(dev), mu_txt.to(dev)).to(dev)
    log("STAGE 2 - positive fusion + FusionHead (composite ListNet, probes frozen)")
    ds.train_dualscore(model, probes, train_feats, train_attrs, raw_stacks, cfg)

    out_dir = output_subdir("awa2")
    tag = f"dualscore_seed{cfg.seed}"
    torch.save(model.state_dict(), out_dir / f"tier3_{tag}_phi.pt")
    torch.save(probes.state_dict(), out_dir / f"tier3_{tag}_probes.pt")

    # Evaluate on the AwA2 benchmark (tier3_dualscore.evaluate_dualscore is bound to the
    # CelebA GT path, so the eval loop lives here; the ranking seam is reused as-is).
    image_features = awa2_data.load_image_features("test").to(dev)
    probes.eval()
    with torch.no_grad():
        db_probs = torch.sigmoid(probes(image_features))
    gt_list = load_eval_json(find_eval_json().parent / "awa2_evaluation.json")

    def make(query_str):
        return ds.make_get_ranking(query_str, image_features, db_probs, model, raw_stacks)

    from eval import evaluate_all
    results = evaluate_all(gt_list, make)
    print(f"\nAwA2 DualScore (seed {cfg.seed}) - {len(gt_list)} queries\n")
    print(format_results_table(results))
    csv_path = out_dir / f"tier3_{tag}.csv"
    save_results_csv(results, csv_path)

    from per_source import check_parity_with_csv, dump_all, save_per_source_npz
    dump = dump_all(gt_list, make)
    check_parity_with_csv(dump, csv_path)
    ps_dir = out_dir / "per_source"
    ps_dir.mkdir(exist_ok=True)
    save_per_source_npz(f"awa2_tier3_{tag}", dump, ps_dir / f"awa2_tier3_{tag}.npz")


if __name__ == "__main__":
    # VM-only (trains Φ): python src/awa2/tier_dualscore.py
    main()
