"""
Tier-2d — DGP: Dynamic Gated Projection (training-free).

Attacks the spec's named bottleneck head-on: CLAY/Track-S build a per-condition subspace by
running ONE SVD over the attribute's prompt stack with every paraphrase row weighted EQUALLY —
query-agnostic and reference-blind (project_specification.md §3.2, "naïve … pre-SVD"). DGP
replaces that fixed SVD with a CLOSED-FORM, REFERENCE-CONDITIONED gate over the same rows:

    Step 0  center each prompt row (modality gap, tier0_enhanced FIX 1, the +63% lever):
              t̂_i = normalize(t_i − μ_txt)
    Step 1  reference-conditioned weights over the n paraphrases (the SVD replacement):
              a_i = softmax_i( ⟨v_ref, t̂_i⟩ / τ )          # each ref emphasises matching prompts
    Step 2  conditional direction from the GATED stack (rank-1) — a dynamic, per-query direction:
              d_c = normalize( Σ_i a_i · t̂_i )
    Step 3  compose on the manifold (geodesic add for T+, orthogonal rejection for T−):
              q_tan = Log_μ(v_ref) + Σ_{c∈T+} α·d_c ;  reject span{d_c : c∈T−} ;  q = Exp_μ(q_tan)

This is the training-free SHADOW of cross-attention: the gate a_i is exactly what a learned
cross-attention over the prompt stack would produce, hand-coded. It therefore (a) satisfies the
spec — replaces the pre-SVD stack with dynamic per-condition re-weighting, processes multiple
textual conditions, defines +/− interaction; and (b) is the closed-form ablation rung directly
beneath the learned fusion model Φ (FusionModelGuide.md), which swaps a_i for learned attention.

Frozen-DB discipline (CLAY, spec §3.2): CLIP is never invoked here. Every input — the image DB,
the prompt bank, μ — is a pre-built cached tensor; the gate is dot products + a weighted mean +
a matmul against the read-only DB. No re-encoding, no DB mutation.

No label leakage: the gate uses only v_ref and the prompt bank; the test DB / GT are never
inspected to build a direction. μ is the train-mined global mean (shared with every other tier).

Run:  python src/tier2d_dgp.py
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from data_loader import ATTR_TO_IDX
from clip_features import load_image_features
from clip_prompts import load_prompt_bank, build_prompts_for_attribute
from eval import parse_query, evaluate_query, format_results_table, load_eval_json, find_eval_json
from results_saver import save_results_csv, output_subdir
from manifold import log_map, exp_map
from tier1_GDE import load_or_mine_directions


# ---------------------------------------------------------------------------
# Configuration — the ablation surface (bundled so call sites stay short).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DGPConfig:
    """DGP hyperparameters. Defaults are the headline starting point."""
    tau: float = 0.07          # gate temperature; small = peaked, large → uniform (≈ centered-mean baseline)
    alpha: float = 1.0         # positive push strength (Track-V's α)
    center: bool = True        # FIX-1 modality-gap centering of prompt rows (off = ablation)

    def __post_init__(self):
        # Fail loudly at construction — an illegal config must never reach the scoring loop.
        if self.tau <= 0:
            raise ValueError(f"tau must be > 0, got {self.tau}")

    def tag(self) -> str:
        # Filename-safe descriptor, e.g. "tau0.07_a1.0_center".
        c = "center" if self.center else "nocenter"
        return f"tau{self.tau}_a{self.alpha}_{c}"


# ---------------------------------------------------------------------------
# Prompt-stack preparation  (centering + padding strip)
# ---------------------------------------------------------------------------

def _attr_stack(name: str, prompt_bank: torch.Tensor) -> torch.Tensor:
    # One attribute's UNPADDED prompt stack [n_a, d]. The cached bank pads every stack to a
    # common width with repeated rows (clip_prompts.py); keeping them would let duplicated
    # directions dominate the softmax gate, so strip to the true count from the deterministic
    # prompt builder (no CLIP call — just len()).
    j = ATTR_TO_IDX[name]
    n_a = len(build_prompts_for_attribute(name))
    return prompt_bank[j, :n_a]                              # [n_a, d]


def compute_mu_txt(prompt_bank: torch.Tensor) -> torch.Tensor:
    # μ_txt — the common-mode text-cone offset (modality gap, Liang et al.; tier0_enhanced FIX 1).
    # Mean of the 40 per-attribute prompt-stack means, then the rows are centered on it. Shared,
    # query-independent, computed once. (Self-consistent: centre on the mean of exactly the
    # directions we later add/reject, the same construction tier0_enhanced uses.)
    attr_means = torch.stack([
        prompt_bank[ATTR_TO_IDX[name], :len(build_prompts_for_attribute(name))].mean(dim=0)
        for name in ATTR_TO_IDX
    ])                                                       # [40, d]
    return attr_means.mean(dim=0)                            # [d]


def _centered_stack(name: str, prompt_bank: torch.Tensor, mu_txt: torch.Tensor, center: bool) -> torch.Tensor:
    # Centered, unit-row prompt stack for one attribute: t̂_i = normalize(t_i − μ_txt).
    # Centering strips the dominant "I am a text embedding" offset so the gate weights reflect
    # attribute match, not shared text-cone direction. center=False is the FIX-1 ablation.
    T = _attr_stack(name, prompt_bank)                       # [n_a, d]
    if center:
        T = T - mu_txt
    return F.normalize(T, dim=1)


# ---------------------------------------------------------------------------
# The gate — the SVD replacement  (Step 1–2)
# ---------------------------------------------------------------------------

def _gated_directions_batch(
    V_ref: torch.Tensor, T_hat: torch.Tensor, tau: float,
) -> torch.Tensor:
    # Reference-conditioned condition direction for a BATCH of references (the SVD replacement).
    # a = softmax(V_ref · T̂ᵀ / τ) over the n paraphrases;  d = normalize(a @ T̂).  [m, d]
    # Each reference re-weights the SAME prompt stack differently — the dynamic, per-condition
    # re-weighting the spec asks for, replacing SVD's fixed equal-weight top-k truncation.
    # (Note: the gate score is the RAW reference·prompt dot product, i.e. cosine since both are
    # unit — geometry in CLIP's native ambient space, not the tangent plane; only the COMPOSED
    # query is lifted to the manifold in Step 3. This keeps the gate a faithful cross-attention
    # analogue and is exactly what learned Φ attends with.)
    logits = (V_ref @ T_hat.T) / tau                         # [m, n_a]
    a = F.softmax(logits, dim=1)                             # [m, n_a] per-reference weights
    return F.normalize(a @ T_hat, dim=1)                     # [m, d] gated condition directions


# ---------------------------------------------------------------------------
# Query composition  (Step 3 — geodesic add + orthogonal rejection)
# ---------------------------------------------------------------------------

def _compose_batch(
    V_ref: torch.Tensor,
    T_pos: list[str],
    T_neg: list[str],
    mu: torch.Tensor,
    stacks: dict[str, torch.Tensor],
    cfg: DGPConfig,
) -> torch.Tensor:
    # DGP query for a batch of references — gated directions composed on the manifold.
    # q_tan = Log_μ(V_ref) + Σ_{c∈T+} α·d_c(V_ref);  reject onto span{d_c(V_ref) : c∈T−};  Exp_μ.
    # Negation is orthogonal rejection (Alhamoud/Oldfield "−X = any value but X"), not subtraction.
    # The negative basis is per-reference (each row has its own gated d_c), so rejection is done
    # row-wise via a batched einsum rather than one shared QR — correct because the gate makes the
    # rejection subspace reference-dependent (the very dynamism that distinguishes DGP from 2b).
    q_tan = log_map(mu, V_ref)                               # [m, d] tangent at μ

    for name in T_pos:
        d_c = _gated_directions_batch(V_ref, stacks[name], cfg.tau)   # [m, d]
        q_tan = q_tan + cfg.alpha * d_c

    for name in T_neg:
        d_c = _gated_directions_batch(V_ref, stacks[name], cfg.tau)   # [m, d] unit per row
        # Row-wise orthogonal rejection: q_tan -= (q_tan·d̂_c) d̂_c, each reference its own axis.
        coeff = (q_tan * d_c).sum(dim=1, keepdim=True)       # [m, 1]
        q_tan = q_tan - coeff * d_c

    return F.normalize(exp_map(mu, q_tan), dim=1)            # [m, d] unit queries


# ---------------------------------------------------------------------------
# Retrieval seam — CONTRACT §5/§7
# ---------------------------------------------------------------------------

def _build_stacks(T_pos, T_neg, prompt_bank, mu_txt, cfg) -> dict[str, torch.Tensor]:
    # Centered prompt stacks for exactly the attributes this query touches (built once per query).
    return {
        name: _centered_stack(name, prompt_bank, mu_txt, cfg.center)
        for name in (*T_pos, *T_neg)
    }


def make_get_ranking(
    query_str: str,
    image_features: torch.Tensor,
    mu: torch.Tensor,
    prompt_bank: torch.Tensor,
    mu_txt: torch.Tensor,
    cfg: DGPConfig = DGPConfig(),
) -> callable:
    # CONTRACT §7 seam — curry one query string into get_ranking(src_idx). Per-source: compose the
    # gated query, score the frozen DB by cosine. Single-source adapter over the batched composer.
    T_pos, T_neg = parse_query(query_str)
    stacks = _build_stacks(T_pos, T_neg, prompt_bank, mu_txt, cfg)

    def get_ranking(src_idx: int) -> list[int]:
        q = _compose_batch(
            image_features[src_idx].unsqueeze(0), T_pos, T_neg, mu, stacks, cfg,
        ).squeeze(0)
        scores = image_features @ q                          # cosine [N] (rows unit, q unit)
        scores[src_idx] = float("-inf")                      # source never ranks itself (CONTRACT §5)
        return torch.argsort(scores, descending=True).tolist()

    return get_ranking


# ---------------------------------------------------------------------------
# Evaluation entry points
# ---------------------------------------------------------------------------

def _load_artifacts():
    # The four read-only cached inputs every config shares (frozen DB, μ, prompt bank, μ_txt, GT).
    image_features = load_image_features()
    mu, _          = load_or_mine_directions()              # shared global mean μ (Track-V cache)
    prompt_bank    = load_prompt_bank()
    mu_txt         = compute_mu_txt(prompt_bank)
    gt_list        = load_eval_json(find_eval_json())
    return image_features, mu, prompt_bank, mu_txt, gt_list


def _batched_rankings(
    T_pos, T_neg, image_features, mu, prompt_bank, mu_txt, cfg, sources, top_k, chunk=512,
) -> dict:
    # All of a query's sources ranked at once → {src_idx: top-`top_k` ranking}. Memory-bounded by `chunk`.
    # The centered prompt stacks are query-level (source-independent) so they are built ONCE; only the
    # cheap per-source gate + composition runs inside the chunk loop. Batches the per-source queries into
    # one Exp_μ + GEMM (image_features @ Q.T), turning thousands of mat-vec products into a few BLAS GEMMs.
    stacks = _build_stacks(T_pos, T_neg, prompt_bank, mu_txt, cfg)
    rankings = {}
    for start in range(0, len(sources), chunk):
        cols = torch.as_tensor(sources[start:start + chunk], dtype=torch.long)
        Q = _compose_batch(image_features[cols], T_pos, T_neg, mu, stacks, cfg)   # [m, d] unit queries
        scores = image_features @ Q.T                        # [N, m]
        scores[cols, torch.arange(len(cols))] = float("-inf")    # source never ranks itself (§5)
        top = torch.topk(scores, top_k, dim=0).indices       # [top_k, m] — metrics read only the prefix
        for j, src in enumerate(cols.tolist()):
            rankings[src] = top[:, j].tolist()
    return rankings


def _evaluate_one(cfg, image_features, mu, prompt_bank, mu_txt, gt_list, ks, save) -> dict:
    # Score one config over the full benchmark, print the table, optionally save the CSV.
    # Single owner of the eval→print→save block so evaluate_dgp and run_ablation can't diverge.
    top_k = max(ks)
    results = {}
    for entry in gt_list:
        T_pos, T_neg = parse_query(entry["query"])
        sources = [int(s) for s in entry["ground_truth"]]
        rankings = _batched_rankings(
            T_pos, T_neg, image_features, mu, prompt_bank, mu_txt, cfg, sources, top_k,
        )
        results[entry["query"]] = evaluate_query(entry["ground_truth"], rankings.__getitem__, ks)

    print(f"\nTier-2d DGP ({cfg.tag()}) — {len(gt_list)} queries\n")
    print(format_results_table(results, ks=ks))

    if save:
        save_results_csv(results, output_subdir("tier2d") / f"tier2d_{cfg.tag()}.csv", ks=ks)
    return results


def evaluate_dgp(cfg: DGPConfig = DGPConfig(), ks=(1, 5, 10), save=True) -> dict:
    # DGP — reference-conditioned gated projection over prompt stacks (the main method).
    image_features, mu, prompt_bank, mu_txt, gt_list = _load_artifacts()
    return _evaluate_one(cfg, image_features, mu, prompt_bank, mu_txt, gt_list, ks, save)


def _ablation_configs() -> list[DGPConfig]:
    # α sweep at the default τ, plus the two gate-isolation ablations:
    #   τ→large ("uniform" weights) must collapse toward a centered-mean baseline ≈ tier0_enhanced
    #     — proving the GATE, not the centering, is what moves the metric;
    #   center=False isolates the modality-gap FIX-1 contribution.
    configs = [DGPConfig(tau=0.07, alpha=a) for a in (0.5, 1.0, 1.5)]
    configs += [DGPConfig(tau=100.0, alpha=1.0)]             # ≈ uniform gate (SVD-mean baseline)
    configs += [DGPConfig(tau=0.07, alpha=1.0, center=False)]
    return configs


def run_ablation(ks=(1, 5, 10)):
    # Run the full ablation grid once, reusing the loaded DB / μ / bank / μ_txt across configs.
    image_features, mu, prompt_bank, mu_txt, gt_list = _load_artifacts()
    for cfg in _ablation_configs():
        _evaluate_one(cfg, image_features, mu, prompt_bank, mu_txt, gt_list, ks, save=True)


if __name__ == "__main__":
    run_ablation()
