"""
AwA2 tier-2d DGP — dynamic gated projection, CelebA's best training-free config.

Replicates the CelebA champion configuration (τ=100 → near-uniform gate, α=1.0,
modality-gap centering on) on AwA2. The DGP math (softmax gate over prompt stacks,
geodesic composition, per-reference rejection negation) is reused from src/tier2d_dgp.py
via the adapter; μ comes from the GDE directions cache (artifacts/awa2/visual_directions.pt
— run src/awa2/tier_gde.py first).

Training-free — local-safe.

Run:  python src/awa2/tier_dgp.py                    # champion config (tau=100)
      python src/awa2/tier_dgp.py --tau 0.07         # sharp-gate ablation
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / "Baselines"))
sys.path.insert(0, str(Path(__file__).parents[1] / "Training-Free"))
sys.path.insert(0, str(Path(__file__).parents[1] / "Trained"))

from awa2 import data_loader as awa2_data
from awa2.adapter import configure_tier2d


def main() -> None:
    ap = argparse.ArgumentParser(description="AwA2 tier-2d DGP (training-free)")
    ap.add_argument("--tau", type=float, default=100.0,
                    help="gate temperature (default 100 = CelebA champion, uniform gate)")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--no-center", action="store_true", help="FIX-1 centering ablation")
    args = ap.parse_args()

    configure_tier2d()

    # Imported AFTER configure_tier2d so module-level tables see AwA2 bindings.
    import tier2d_dgp as dgp
    from eval import (evaluate_all, find_eval_json, format_results_table,
                      load_eval_json)
    from results_saver import output_subdir, save_results_csv
    from tier1_GDE import load_or_mine_directions
    from awa2.clip_prompts import load_prompt_bank

    cfg = dgp.DGPConfig(tau=args.tau, alpha=args.alpha, center=not args.no_center)
    image_features = awa2_data.load_image_features("test")
    mu, _ = load_or_mine_directions()          # GDE cache (artifacts/awa2, via adapter)
    prompt_bank = load_prompt_bank()
    mu_txt = dgp.compute_mu_txt(prompt_bank)
    gt_list = load_eval_json(find_eval_json().parent / "awa2_evaluation.json")

    def make(query_str):
        return dgp.make_get_ranking(query_str, image_features, mu, prompt_bank, mu_txt, cfg)

    results = evaluate_all(gt_list, make)
    print(f"\nAwA2 tier-2d DGP ({cfg.tag()}) - {len(gt_list)} queries\n")
    print(format_results_table(results))

    csv_path = output_subdir("awa2") / f"tier2d_{cfg.tag()}.csv"
    save_results_csv(results, csv_path)

    from per_source import check_parity_with_csv, dump_all, save_per_source_npz
    dump = dump_all(gt_list, make)
    check_parity_with_csv(dump, csv_path)
    ps_dir = output_subdir("awa2") / "per_source"
    ps_dir.mkdir(exist_ok=True)
    save_per_source_npz(f"awa2_tier2d_{cfg.tag()}", dump,
                        ps_dir / f"awa2_tier2d_{cfg.tag()}.npz")


if __name__ == "__main__":
    # Training-free, local-safe: python src/awa2/tier_dgp.py
    main()
