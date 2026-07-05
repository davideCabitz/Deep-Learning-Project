"""
OVAD tier-2d DGP — dynamic gated projection, CelebA's best training-free config.

Replicates the CelebA champion configuration (τ=100 → near-uniform gate, α=1.0,
modality-gap centering on). The DGP math (softmax gate over prompt stacks, geodesic
composition, per-reference rejection negation) is reused from src/tier2d_dgp.py via the
adapter.

μ (the tangent-space base point) comes from the TRAIN split's intrinsic mean, exactly as
on AwA2/CelebA — the synthetic image-level split (data_loader.build_split) restored the
train pool, so the earlier corpus-mean substitute is gone and the retrieval DB is the
held-out test split only.

Training-free — local-safe.

Run:  python src/ovad/tier_dgp.py                    # champion config (tau=100)
      python src/ovad/tier_dgp.py --tau 0.07         # sharp-gate ablation
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from ovad import data_loader as ovad_data
from ovad.adapter import configure_tier_dgp

OVAD_EVAL_JSON = "ovad_evaluation.json"


def main() -> None:
    ap = argparse.ArgumentParser(description="OVAD tier-2d DGP (training-free)")
    ap.add_argument("--tau", type=float, default=100.0,
                    help="gate temperature (default 100 = CelebA champion, uniform gate)")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--no-center", action="store_true", help="FIX-1 centering ablation")
    args = ap.parse_args()

    configure_tier_dgp()

    # Imported AFTER configure_tier_dgp so module-level tables see OVAD bindings.
    import tier2d_dgp as dgp
    from eval import evaluate_all, find_eval_json, format_results_table, load_eval_json
    from manifold import intrinsic_mean
    from results_saver import output_subdir, save_results_csv
    from ovad.clip_prompts import load_prompt_bank

    cfg = dgp.DGPConfig(tau=args.tau, alpha=args.alpha, center=not args.no_center)
    image_features = ovad_data.load_image_features("test")
    mu = intrinsic_mean(ovad_data.load_image_features("train"))   # train μ, as on AwA2/CelebA
    prompt_bank = load_prompt_bank()
    mu_txt = dgp.compute_mu_txt(prompt_bank)
    gt_list = load_eval_json(find_eval_json().parent / OVAD_EVAL_JSON)

    def make(query_str):
        return dgp.make_get_ranking(query_str, image_features, mu, prompt_bank, mu_txt, cfg)

    results = evaluate_all(gt_list, make)
    print(f"\nOVAD tier-2d DGP ({cfg.tag()}) - {len(gt_list)} queries\n")
    print(format_results_table(results))

    csv_path = output_subdir("ovad") / f"tier2d_{cfg.tag()}.csv"
    save_results_csv(results, csv_path)

    from per_source import check_parity_with_csv, dump_all, save_per_source_npz
    dump = dump_all(gt_list, make)
    check_parity_with_csv(dump, csv_path)
    ps_dir = output_subdir("ovad") / "per_source"
    ps_dir.mkdir(exist_ok=True)
    save_per_source_npz(f"ovad_tier2d_{cfg.tag()}", dump,
                        ps_dir / f"ovad_tier2d_{cfg.tag()}.npz")


if __name__ == "__main__":
    # Training-free, local-safe: python src/ovad/tier_dgp.py
    main()
