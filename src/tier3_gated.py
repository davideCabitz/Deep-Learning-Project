"""
Tier-3 Gated — FusionPhiNeg composition + additive attribute-classifier gate.

score(d) = cos(q, v_d) + λ_pos·Σ_{a∈T+} log P(a|v_d) + λ_neg·Σ_{b∈T−} log(1−P(b|v_d))
where q = FusionPhiNeg(v_ref, T+, T−) and P(a|v_d) is a train-fit attribute probe.

Separates the two jobs the composition model conflates: FusionPhiNeg steers identity
toward the desired attributes (a soft, geometric push), while the gate enforces the
HARD constraint "does this DB image actually satisfy each ±attribute" via a calibrated
per-attribute log-probability. The probe is fit ONLY on frozen TRAIN features/labels
(the same regime GDE mines its directions from), then evaluated once over the frozen
test DB — the DB is never re-encoded, so this is spec-compliant (§3.2 step 2). The gate
is additive, not multiplicative: a wrong-side image takes a large negative log-penalty
and sinks, without collapsing the identity ordering among valid images.

(Note: the `-Male,-Mustache` query is near CLIP ViT-B/32's frozen-DB ceiling — an ORACLE
gate with true test labels + identity cosine reaches only R@10≈0.074, because the valid
female targets are not the females CLIP considers closest to a male reference (+0.0135
cosine over random). The gate's real contribution there is getting the query OFF zero
with a cited mechanism; the ceiling is a property of the embedding space, documented not
hidden.)

Run (load combined weights, don't retrain):
  python src/tier3_gated.py --weights output/tier3_combined/tier3_combined_phi.pt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import ATTRIBUTE_NAMES, ATTR_TO_IDX, _get_artifacts_dir
from clip_features import load_image_features
from clip_prompts import load_prompt_bank
from eval import parse_query, evaluate_query, format_results_table, load_eval_json, find_eval_json
from results_saver import save_results_csv, output_subdir
from tier1_GDE import load_or_mine_directions, _load_train_features, _load_train_attributes
from tier2d_dgp import compute_mu_txt
from tier3_dgp import build_raw_stacks
from tier3_negation import FusionPhiNeg
from tier3_combined import CombinedConfig, train_combined


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GatedConfig:
    # Tier-3 Gated — FusionPhiNeg composition + additive attribute-classifier gate.
    lambda_neg: float = 4.0        # negated-attribute log-penalty strength (primary lever)
    lambda_pos: float = 0.0        # positive-attribute log-bonus (0 = off; opt-in ablation)
    probe: str = "linear"          # probe backend: "linear" | "mlp"
    gate_wd: float = 1e-3          # L2 regularisation for the probe fit
    probe_epochs: int = 200
    seed: int = 0


# ---------------------------------------------------------------------------
# Attribute probes — strategy pattern (linear | mlp), fit on frozen TRAIN
# ---------------------------------------------------------------------------

def _fit_linear_probe(
    Xtr: torch.Tensor, ytr: torch.Tensor, wd: float, epochs: int, seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Logistic regression P(a|v) = sigmoid(w·v + b), fit by Adam+BCE on frozen train features.
    # Linear suffices: probes hit 96–99% test accuracy, so a heavier model earns nothing.
    torch.manual_seed(seed)
    w = torch.zeros(Xtr.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.5, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(Xtr @ w + b, ytr).backward()
        opt.step()
    return w.detach(), b.detach()


class _MLPProbe(nn.Module):
    # Small 2-layer classifier P(a|v) = sigmoid(MLP(v)) — the nonlinear probe backend.
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _fit_mlp_probe(
    Xtr: torch.Tensor, ytr: torch.Tensor, wd: float, epochs: int, seed: int,
) -> _MLPProbe:
    # Fit the nonlinear MLP probe (same BCE objective) on frozen train features.
    torch.manual_seed(seed)
    model = _MLPProbe(Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(model(Xtr), ytr).backward()
        opt.step()
    return model.eval()


@torch.no_grad()
def _score_db_logits(probe, image_features: torch.Tensor) -> torch.Tensor:
    # Evaluate a fitted probe over the frozen DB → raw logits [N_db]. Dispatches on backend.
    if isinstance(probe, tuple):                       # linear: (w, b)
        w, b = probe
        return image_features @ w + b
    return probe(image_features)                       # mlp module


# ---------------------------------------------------------------------------
# Gate builder — fit once on TRAIN, cache log-prob tables over the frozen DB
# ---------------------------------------------------------------------------

def build_attr_gate(cfg: GatedConfig = GatedConfig(), force: bool = False) -> dict:
    # Per-attribute probe fit on frozen TRAIN, scored on frozen TEST DB, cached as
    # log P / log(1−P) tables [40, N_db] aligned to ATTRIBUTE_NAMES and DB row order.
    # Load-or-build idiom mirroring tier1_GDE.load_or_mine_directions.
    cache_path = _get_artifacts_dir() / f"attr_gate_{cfg.probe}.pt"
    image_features = load_image_features()
    n_db = image_features.shape[0]

    if cache_path.exists() and not force:
        ckpt = torch.load(cache_path, weights_only=True)
        if ckpt["logp"].shape == (len(ATTRIBUTE_NAMES), n_db) and ckpt["probe"] == cfg.probe:
            print(f"[OK] Loading cached attribute gate: {cache_path}")
            return ckpt
        print(f"[!] Cached gate at {cache_path} mismatched (shape/probe) — rebuilding.")

    if cfg.probe not in ("linear", "mlp"):
        raise ValueError(f"probe must be 'linear' or 'mlp', got {cfg.probe!r}")

    print(f"Fitting {cfg.probe} attribute probes on frozen train features…")
    Xtr = _load_train_features()                        # [N_train, d] unit
    Atr = _load_train_attributes()                      # [N_train, 40] in {0,1}
    if Xtr.shape[0] != Atr.shape[0]:
        raise ValueError(f"train feature/label row mismatch: {Xtr.shape[0]} vs {Atr.shape[0]}")

    logp = torch.empty(len(ATTRIBUTE_NAMES), n_db)
    log1mp = torch.empty(len(ATTRIBUTE_NAMES), n_db)
    fit = _fit_linear_probe if cfg.probe == "linear" else _fit_mlp_probe
    for j, name in enumerate(ATTRIBUTE_NAMES):
        y = (Atr[:, j] > 0.5).float()
        probe = fit(Xtr, y, cfg.gate_wd, cfg.probe_epochs, cfg.seed)
        z = _score_db_logits(probe, image_features)     # [N_db] logits over frozen DB
        logp[j] = F.logsigmoid(z)                       # log P(a|v_d)
        log1mp[j] = F.logsigmoid(-z)                    # log(1 − P(a|v_d))
        print(f"  [{j+1:02d}/40] {name}", end="\r")

    if torch.isnan(logp).any() or torch.isinf(logp).any():
        raise ValueError("gate produced NaN/Inf log-probabilities — check probe fit.")

    gate = {"logp": logp, "log1mp": log1mp, "probe": cfg.probe}
    torch.save(gate, cache_path)
    print(f"\n  Saved: {cache_path}")
    return gate


# ---------------------------------------------------------------------------
# Retrieval seam — CONTRACT §5/§7
# ---------------------------------------------------------------------------

def make_get_ranking(
    query_str: str,
    image_features: torch.Tensor,
    model: FusionPhiNeg,
    raw_stacks: dict[str, torch.Tensor],
    gate: dict,
    cfg: GatedConfig,
) -> callable:
    # Compose q via FusionPhiNeg, cosine-score the frozen DB, then ADD the log-prob gate:
    # +λ_pos·log P for each T+ attr, +λ_neg·log(1−P) for each T− attr. The gate term is
    # source-independent (shared across all sources of the query), computed once here.
    T_pos, T_neg = parse_query(query_str)
    model.eval()

    gate_term = torch.zeros(image_features.shape[0])
    for a in T_pos:
        gate_term = gate_term + cfg.lambda_pos * gate["logp"][ATTR_TO_IDX[a]]
    for b in T_neg:
        gate_term = gate_term + cfg.lambda_neg * gate["log1mp"][ATTR_TO_IDX[b]]

    @torch.no_grad()
    def get_ranking(src_idx: int) -> list[int]:
        q = model(
            image_features[src_idx].unsqueeze(0),
            [raw_stacks[nm] for nm in T_pos],
            [raw_stacks[nm] for nm in T_neg],
        ).squeeze(0)
        scores = (image_features @ q) + gate_term
        scores[src_idx] = float("-inf")
        return torch.argsort(scores, descending=True).tolist()

    return get_ranking


# ---------------------------------------------------------------------------
# Evaluation entry point
# ---------------------------------------------------------------------------

def evaluate_gated(
    model: FusionPhiNeg,
    gate: dict,
    cfg: GatedConfig = GatedConfig(),
    ks=(1, 5, 10),
    save: bool = True,
    tag: str = "gated",
) -> dict:
    # Score on the 14-query benchmark, print table, save CSV. Mirrors evaluate_combined.
    image_features = load_image_features()
    prompt_bank = load_prompt_bank()
    raw_stacks = build_raw_stacks(prompt_bank)
    gt_list = load_eval_json(find_eval_json())

    results = {}
    for entry in gt_list:
        get_ranking = make_get_ranking(entry["query"], image_features, model, raw_stacks, gate, cfg)
        results[entry["query"]] = evaluate_query(entry["ground_truth"], get_ranking, ks)

    print(f"\nTier-3 Gated ({tag}, probe={cfg.probe}, λ_neg={cfg.lambda_neg}, λ_pos={cfg.lambda_pos}) "
          f"— {len(gt_list)} queries\n")
    print(format_results_table(results, ks=ks))

    if save:
        save_results_csv(results, output_subdir("tier3_gated") / f"tier3_{tag}.csv", ks=ks)
    return results


# ---------------------------------------------------------------------------
# Model loading — reuse combined weights, don't retrain by default
# ---------------------------------------------------------------------------

def _load_combined_model(weights_path: str) -> FusionPhiNeg:
    # Instantiate FusionPhiNeg and load tier3_combined weights (same setup as train_combined).
    prompt_bank = load_prompt_bank()
    mu, _ = load_or_mine_directions()
    mu_txt = compute_mu_txt(prompt_bank)
    model = FusionPhiNeg(CombinedConfig().as_phi_config(), mu, mu_txt)
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    return model.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to FusionPhiNeg weights (.pt); load without retraining.")
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain the combined model from scratch instead of loading.")
    parser.add_argument("--lambda-neg", type=float, default=GatedConfig.lambda_neg)
    parser.add_argument("--lambda-pos", type=float, default=GatedConfig.lambda_pos)
    parser.add_argument("--probe", type=str, default=GatedConfig.probe, choices=["linear", "mlp"])
    parser.add_argument("--force-gate", action="store_true", help="Rebuild the gate cache.")
    args = parser.parse_args()

    cfg = GatedConfig(lambda_neg=args.lambda_neg, lambda_pos=args.lambda_pos, probe=args.probe)

    if args.retrain or not args.weights:
        model = train_combined(CombinedConfig())
        weights_path = output_subdir("tier3_gated") / "tier3_gated_phi.pt"
        torch.save(model.state_dict(), weights_path)
        print(f"Weights saved: {weights_path}")
    else:
        print(f"Loading weights from: {args.weights}")
        model = _load_combined_model(args.weights)

    gate = build_attr_gate(cfg, force=args.force_gate)
    tag = f"gated_{cfg.probe}_ln{cfg.lambda_neg:g}_lp{cfg.lambda_pos:g}"
    evaluate_gated(model, gate, cfg, tag=tag)
