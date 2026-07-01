"""
Tier-3E — Text-space negation steering (training-free, DGP backbone).

Negation via a latent direction in CLIP's TEXT space, not the visual attribute mean.
    Ŵ_neg = normalize( linear-classifier weights: affirmative(0) vs negated(1) prompts )
    steer:  d̃_c = normalize( (1−α)·d_c + α·Ŵ_neg·‖d_c‖ )        # norm-preserving (Sammani Eq.1)
    reject: q_tan −= (q_tan·d̃_c)·d̃_c                            # DGP orthogonal rejection

All prior negation subtracts a VISUAL attribute direction mined at the global mean μ; those
directions are entangled (mined Male/Mustache cos=0.83) and blunt for a cross-cluster shift.
This tier instead borrows the finding of Sammani et al. 2026 ("When Negation is a Geometry
Problem in VLMs"): affirmative and negated captions are linearly separable in CLIP's text
embedding, so a single classifier weight vector IS a negation axis. We steer the gated
condition direction toward that axis (norm-preserving, α≈0.13) before rejecting it, giving a
cleaner "not-this-attribute" operator than raw rejection.

Affirmation and identity are the unchanged DGP composition (tier2d_dgp); only the negation
operator differs. Training-free (the classifier fits on 40-attr affirmative/negated PROMPTS,
never on images or the test DB), frozen DB, spec-compliant (§3.2 step 2). The negation axis is
a cached artifact; CLIP is invoked only once to encode the negated prompts.

Run:  python src/tier3_negsteer.py
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import ATTRIBUTE_NAMES, ATTR_TO_IDX, _get_artifacts_dir
from clip_features import CLIP_MODEL_NAME, FEATURE_DIM, _pick_device, load_image_features
from clip_prompts import load_prompt_bank, build_prompts_for_attribute, FRAMES, PREDICATIVE_FRAMES, ATTR_PHRASES
from eval import parse_query, evaluate_query, format_results_table, load_eval_json, find_eval_json
from results_saver import save_results_csv, output_subdir
from manifold import log_map, exp_map
from tier1_GDE import load_or_mine_directions
from tier2d_dgp import DGPConfig, compute_mu_txt, _centered_stack, _gated_directions_batch


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NegSteerConfig:
    # Tier-3E — DGP composition + text-space negation steering.
    tau: float = 0.07              # DGP gate temperature (inherited)
    alpha: float = 1.0             # DGP positive push strength (inherited)
    center: bool = True            # DGP modality-gap centering (inherited)
    steer: float = 0.13            # negation-axis steering strength (Sammani et al. best α)
    clf_wd: float = 1e-3           # L2 reg for the negation classifier
    clf_epochs: int = 300
    seed: int = 0

    def as_dgp(self) -> DGPConfig:
        return DGPConfig(tau=self.tau, alpha=self.alpha, center=self.center)

    def tag(self) -> str:
        return f"steer{self.steer:g}_tau{self.tau:g}_a{self.alpha:g}"


# ---------------------------------------------------------------------------
# Negation-axis mining — a linear classifier over affirmative vs negated prompts
# ---------------------------------------------------------------------------

_NEG_TEMPLATES = {
    # For each phrase kind, how to render its NEGATED form. noun → "no/without <phrase>";
    # adj → "not <phrase>". Kept plain — CLIP ViT-B/32 keys on content words, not syntax.
    "noun": ["no {phrase}", "without {phrase}"],
    "adj": ["not {phrase}", "that is not {phrase}"],
}


def _strip_article(phrase: str) -> str:
    # Drop a leading article so negation templates read "no mustache", not "no a mustache".
    for art in ("a ", "an ", "the "):
        if phrase.startswith(art):
            return phrase[len(art):]
    return phrase


def _build_prompt_pairs() -> tuple[list[str], list[str]]:
    # Affirmative and negated prompt lists across all 40 attributes (frames × phrases × neg-forms).
    # Affirmative reuses the exact clip_prompts builder; negated swaps the phrase into a negation
    # template inside the same frames — so only the +/− polarity differs, isolating the axis.
    affirmative, negated = [], []
    for name in ATTRIBUTE_NAMES:
        kind, phrases = ATTR_PHRASES[name]
        frames = PREDICATIVE_FRAMES if kind == "adj" else FRAMES
        affirmative.extend(build_prompts_for_attribute(name))
        for phrase in phrases:
            base = _strip_article(phrase) if kind == "noun" else phrase
            for neg_tmpl in _NEG_TEMPLATES[kind]:
                neg_phrase = neg_tmpl.format(phrase=base)
                for frame in frames:
                    negated.append(frame.format(phrase=neg_phrase))
    return affirmative, negated


@torch.no_grad()
def _encode_prompts(prompts: list[str], batch: int = 256) -> torch.Tensor:
    # Encode a prompt list with frozen CLIP text tower → [n, 512] unit rows. (One-time.)
    from transformers import CLIPModel, AutoTokenizer
    device = _pick_device()
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(CLIP_MODEL_NAME)
    out = []
    for start in range(0, len(prompts), batch):
        toks = tokenizer(prompts[start:start + batch], padding=True, return_tensors="pt").to(device)
        feats = model.text_projection(model.text_model(**toks).pooler_output)
        out.append(F.normalize(feats, dim=1).cpu().to(torch.float32))
    return torch.cat(out, dim=0)


def build_negation_axis(cfg: NegSteerConfig = NegSteerConfig(), force: bool = False) -> torch.Tensor:
    # Negation axis Ŵ_neg [512] — normalized weights of a linear classifier separating
    # affirmative(0) from negated(1) CLIP text embeddings (Sammani et al. 2026 §4). Cached.
    # Load-or-build idiom mirroring tier1_GDE.load_or_mine_directions.
    cache_path = _get_artifacts_dir() / "negation_axis.pt"
    if cache_path.exists() and not force:
        ckpt = torch.load(cache_path, weights_only=True)
        if ckpt["axis"].shape == (FEATURE_DIM,):
            print(f"[OK] Loading cached negation axis: {cache_path}")
            return ckpt["axis"]

    print("Encoding affirmative/negated prompt pairs for the negation axis…")
    affirmative, negated = _build_prompt_pairs()
    X_aff = _encode_prompts(affirmative)
    X_neg = _encode_prompts(negated)
    X = torch.cat([X_aff, X_neg], dim=0)
    y = torch.cat([torch.zeros(len(X_aff)), torch.ones(len(X_neg))])

    torch.manual_seed(cfg.seed)
    w = torch.zeros(FEATURE_DIM, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.1, weight_decay=cfg.clf_wd)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(cfg.clf_epochs):
        opt.zero_grad()
        loss_fn(X @ w + b, y).backward()
        opt.step()

    with torch.no_grad():
        acc = (((X @ w + b) > 0).float() == y).float().mean().item()
    print(f"  negation classifier train accuracy: {acc:.4f}  "
          f"(aff={len(X_aff)}, neg={len(X_neg)})")
    axis = F.normalize(w.detach(), dim=0)                    # Ŵ_neg direction only
    torch.save({"axis": axis, "acc": acc}, cache_path)
    print(f"  Saved: {cache_path}")
    return axis


# ---------------------------------------------------------------------------
# Steered composition — DGP affirmation, text-space-steered negation
# ---------------------------------------------------------------------------

def _compose_batch_steered(
    V_ref: torch.Tensor,
    T_pos: list[str],
    T_neg: list[str],
    mu: torch.Tensor,
    stacks: dict[str, torch.Tensor],
    neg_axis: torch.Tensor,
    cfg: NegSteerConfig,
) -> torch.Tensor:
    # DGP query with text-space-steered negation. Affirmation identical to tier2d_dgp; each
    # negated attribute's gated direction is steered toward Ŵ_neg (norm-preserving) before the
    # orthogonal rejection, so we reject a "negation-aware" axis rather than the raw attribute one.
    q_tan = log_map(mu, V_ref)                               # [m, d] tangent at μ

    for name in T_pos:
        d_c = _gated_directions_batch(V_ref, stacks[name], cfg.tau)
        q_tan = q_tan + cfg.alpha * d_c

    for name in T_neg:
        d_c = _gated_directions_batch(V_ref, stacks[name], cfg.tau)          # [m, d] unit
        norms = d_c.norm(dim=1, keepdim=True)                                # ‖d_c‖ (≈1, kept for fidelity)
        steered = (1 - cfg.steer) * d_c + cfg.steer * neg_axis.unsqueeze(0) * norms
        d_tilde = F.normalize(steered, dim=1)                               # [m, d]
        coeff = (q_tan * d_tilde).sum(dim=1, keepdim=True)
        q_tan = q_tan - coeff * d_tilde

    return F.normalize(exp_map(mu, q_tan), dim=1)


# ---------------------------------------------------------------------------
# Retrieval seam — CONTRACT §5/§7
# ---------------------------------------------------------------------------

def make_get_ranking(
    query_str: str,
    image_features: torch.Tensor,
    mu: torch.Tensor,
    prompt_bank: torch.Tensor,
    mu_txt: torch.Tensor,
    neg_axis: torch.Tensor,
    cfg: NegSteerConfig = NegSteerConfig(),
) -> callable:
    # Curry one query into get_ranking(src_idx): steered composition, cosine-score the frozen DB.
    T_pos, T_neg = parse_query(query_str)
    stacks = {name: _centered_stack(name, prompt_bank, mu_txt, cfg.center)
              for name in (*T_pos, *T_neg)}

    def get_ranking(src_idx: int) -> list[int]:
        q = _compose_batch_steered(
            image_features[src_idx].unsqueeze(0), T_pos, T_neg, mu, stacks, neg_axis, cfg,
        ).squeeze(0)
        scores = image_features @ q
        scores[src_idx] = float("-inf")
        return torch.argsort(scores, descending=True).tolist()

    return get_ranking


# ---------------------------------------------------------------------------
# Evaluation entry point
# ---------------------------------------------------------------------------

def evaluate_negsteer(cfg: NegSteerConfig = NegSteerConfig(), ks=(1, 5, 10), save: bool = True) -> dict:
    # Score text-space negation steering on the 14-query benchmark, print table, save CSV.
    image_features = load_image_features()
    prompt_bank = load_prompt_bank()
    mu, _ = load_or_mine_directions()
    mu_txt = compute_mu_txt(prompt_bank)
    neg_axis = build_negation_axis(cfg)
    gt_list = load_eval_json(find_eval_json())

    results = {}
    for entry in gt_list:
        get_ranking = make_get_ranking(
            entry["query"], image_features, mu, prompt_bank, mu_txt, neg_axis, cfg,
        )
        results[entry["query"]] = evaluate_query(entry["ground_truth"], get_ranking, ks)

    print(f"\nTier-3E NegSteer ({cfg.tag()}) — {len(gt_list)} queries\n")
    print(format_results_table(results, ks=ks))

    if save:
        save_results_csv(results, output_subdir("tier3_negsteer") / f"tier3_{cfg.tag()}.csv", ks=ks)
    return results


if __name__ == "__main__":
    # Default steer, plus a small sweep of the steering strength (global, not per-query).
    for s in (0.0, 0.13, 0.3):
        evaluate_negsteer(NegSteerConfig(steer=s))
