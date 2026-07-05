"""
run_custom_eval — model roster on the CUSTOM CelebA benchmark (spec §3.1 custom queries).

One orchestrator that rebinds the tiers' existing public seams to
Evaluation/celeba_custom_evaluation.json (frozen by generate_custom_gt.py). No scoring
math lives here, and NOTHING trains: every model loads its cached artifacts/checkpoints
(`force=False` everywhere) and is inference-only; a model whose artifacts are absent is
announced and skipped. Per model this writes the canonical CSV plus a parity-gated
per-source NPZ under output/custom/, feeding the report's §10 statistics unchanged.

Run:  python src/run_custom_eval.py [--models CLIP,CLAY,...] [--device cpu|cuda]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "Baselines"))
sys.path.insert(0, str(Path(__file__).parent / "Training-Free"))
sys.path.insert(0, str(Path(__file__).parent / "Trained"))

from eval import evaluate_all, find_eval_json, format_results_table, load_eval_json
from per_source import check_parity_with_csv, dump_all, save_per_source_npz
from results_saver import output_subdir, save_results_csv


def load_custom_gt() -> list[dict]:
    # The frozen custom benchmark — a hard prerequisite, never regenerated silently.
    path = find_eval_json().parent / "celeba_custom_evaluation.json"
    if not path.exists():
        raise SystemExit(f"{path} not found — run `python src/generate_custom_gt.py` first.")
    return load_eval_json(path)


# ---------------------------------------------------------------------------
# Model builders — each returns make(query_str) -> get_ranking over cached artifacts.
# Import inside the builder so a missing artifact only disables ITS model.
# ---------------------------------------------------------------------------

def _make_clip():
    # CLIP (§4.1) — vanilla latent arithmetic, α=1 (published baseline config).
    from clip_baseline import load_attribute_text_features, load_image_features, make_get_ranking
    feats, txt = load_image_features(), load_attribute_text_features()
    return lambda q: make_get_ranking(q, feats, txt, alpha=1.0)


def _make_clip_enhanced():
    # CLIP Enhanced (§4.2) — all three geometric fixes on (published champion config).
    from clip_enhanced import build_attr_directions, load_image_features, make_get_ranking
    feats = load_image_features()
    dirs, mu_img = build_attr_directions(use_prompt_bank=True, center=True)
    return lambda q: make_get_ranking(q, feats, dirs, mu_img,
                                      alpha=1.0, norm_delta=True, center=True)


def _make_clay():
    # CLAY (§4.3) — faithful reproduction, k=50 + rotation (published config).
    from clip_features import load_image_features
    from clip_prompts import load_prompt_bank
    from clay import make_get_ranking
    feats, bank = load_image_features(), load_prompt_bank()
    return lambda q: make_get_ranking(q, feats, bank, k=50, use_rotation=True)


def _make_dgp():
    # DGP (§4.6) — uniform gate, centered (τ=100, α=1: the published tier-2d champion).
    from clip_features import load_image_features
    from clip_prompts import load_prompt_bank
    from gde import load_or_mine_directions
    from dgp import DGPConfig, compute_mu_txt, make_get_ranking
    cfg = DGPConfig(tau=100.0, alpha=1.0)
    feats, bank = load_image_features(), load_prompt_bank()
    mu, _ = load_or_mine_directions()
    mu_txt = compute_mu_txt(bank)
    return lambda q: make_get_ranking(q, feats, mu, bank, mu_txt, cfg)


def _make_dualscore():
    # Dual Score (§5.4) — trained weights live on the VM only; explicit skip keeps the
    # roster honest locally without blocking on a VM pull.
    raise FileNotFoundError("Dual Score checkpoint is VM-only — rerun there or pull "
                            "its weights, then wire the seam here.")


def _run_abap(gt_list: list[dict], dev: torch.device) -> tuple[dict, dict]:
    # A-BAP champion (§8) — inference-only replay of tier5's dump_per_source_champions
    # recipe on the custom benchmark: cached probe/τ/tree + saved ListNet-seed0 Φ over
    # the λ=0 A′ core (the exact base it was trained on). No tuning, no training.
    from clip_features import load_image_features
    from cdbayes import (BayesPhi, StageAPrime, CDBayesConfig, calibrated_posteriors,
                               load_or_build_tree, load_or_fit_temperatures,
                               load_or_train_probes, make_stage_b_prime_chunk)
    from vm_pipeline import evaluate_scorer

    cfg = CDBayesConfig()
    ckpt = output_subdir("cdbayes") / "tier5_Bprime_listnet_seed0.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} not found — pull the tier-5 bundle from the VM.")

    probe = load_or_train_probes(cfg.tier4, dev, force=False)
    tau = load_or_fit_temperatures(probe, cfg, dev, force=False)
    tree = load_or_build_tree(cfg, force=False).to(dev)
    feats = load_image_features().to(dev)
    core = StageAPrime(feats, calibrated_posteriors(probe, feats, tau), tree,
                       conditioned_tail=True, lam=0.0, hamming_max=cfg.tier4.hamming_max)
    model = BayesPhi(cfg.tier4).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev))

    dump: dict = {}
    results = evaluate_scorer("A-BAP", make_stage_b_prime_chunk(model, core, use_qv=True),
                              gt_list, cfg.tier4.chunk, dump=dump)
    return results, dump


MODELS = {  # display key -> (output stem, get_ranking builder | None for the chunked path)
    "CLIP":          ("custom_CLIP", _make_clip),
    "CLIP_Enhanced": ("custom_CLIP_Enhanced", _make_clip_enhanced),
    "CLAY":          ("custom_CLAY", _make_clay),
    "DGP":           ("custom_DGP", _make_dgp),
    "Dual_Score":    ("custom_Dual_Score", _make_dualscore),
    "A-BAP":         ("custom_A-BAP", None),
}


def _persist(stem: str, results: dict, dump: dict, out_dir: Path) -> None:
    # CSV + parity-gated NPZ — the dump can never disagree with the published table.
    csv_path = out_dir / f"{stem}.csv"
    save_results_csv(results, csv_path)
    check_parity_with_csv(dump, csv_path)
    ps_dir = out_dir / "per_source"
    ps_dir.mkdir(exist_ok=True)
    save_per_source_npz(stem, dump, path=ps_dir / f"{stem}.npz")


def run_model(name: str, gt_list: list[dict], dev: torch.device, out_dir: Path) -> None:
    stem, builder = MODELS[name]
    try:
        if builder is None:
            results, dump = _run_abap(gt_list, dev)
        else:
            make = builder()
            results = evaluate_all(gt_list, make)
            dump = dump_all(gt_list, make)
    except FileNotFoundError as e:
        print(f"[skip] {name}: {e}")
        return
    print(f"\n{name} — custom benchmark ({len(gt_list)} queries)\n")
    print(format_results_table(results))
    _persist(stem, results, dump, out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the roster on the custom benchmark")
    ap.add_argument("--models", type=str, default=None,
                    help=f"comma-separated subset of: {','.join(MODELS)}")
    ap.add_argument("--device", type=str, default=None, help="cuda | cpu (default: auto)")
    args = ap.parse_args()

    wanted = list(MODELS) if args.models is None else [m.strip() for m in args.models.split(",")]
    unknown = [m for m in wanted if m not in MODELS]
    if unknown:
        raise SystemExit(f"unknown --models {unknown}; valid: {list(MODELS)}")

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    gt_list = load_custom_gt()
    out_dir = output_subdir("custom")
    for name in wanted:
        run_model(name, gt_list, dev, out_dir)


if __name__ == "__main__":
    main()
