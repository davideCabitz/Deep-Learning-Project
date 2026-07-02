"""
Tier-5 LingoCIR — Compositional Textual-Inversion Querying (CLIP frozen).

Every prior tier composes the query as vector arithmetic in CLIP's shared embedding
space (latent addition, tangent-space geodesics, learned cross-attention gates).
LingoCIR routes composition through CLIP's own text encoder instead: a small MLP
`g: v_ref → S*` inverts the reference image into one pseudo-token embedding, which
is spliced into a natural-language template —

    "a photo of [S*], wearing eyeglasses, with a serious expression"

— and encoded by the frozen CLIP text tower. The text transformer's self-attention
*is* Φ (FinalDestination.md §4.C): zero fusion parameters beyond `g`. Negation is
handled by antonym rewriting (ANTONYM_MAP) instead of vector subtraction, sidestepping
CLIP's documented negation blindness; a fixed Slerp anchors the composed text query
back toward the reference's image-manifold point, mitigating the modality gap that
capped every geometric tier (FinalDestination.md §1.3).

Technical note (the one genuinely risky piece): HuggingFace's CLIPTextModel.forward
hard-requires input_ids — it cannot be called with only inputs_embeds, and its built-in
EOS pooling silently breaks once the vocab is extended by one token (it starts pooling
at the new token instead of EOS, since legacy pooling is `argmax(input_ids)`). This
module never calls `model.text_model(...)` as a black box; encode_composed_query
reimplements CLIPTextTransformer's forward pass manually (verified bit-for-bit
equivalent to the black box, see test/test_tier5_lingocir.py spike 1) with explicit
EOS-id search for pooling (spike 4) and byte-identical frozen embedding rows after
the vocab resize (spike 3).

Trainable parameters: InversionMLP only. CLIP (image tower, text tower, projections,
including the resized embedding table's pre-existing rows) never receives gradient —
freeze_clip() is called once, before any optimizer is built, and only g.parameters()
is ever passed to an optimizer (spike 5 operationalizes this as a test).

VM-ready: timestamped progress + ETA, headless training curve, final tar.gz bundle
that scripts/pull_tier5_lingocir_results.ps1 copies back into any clone.

Run (from repo root):  python src/tier5_lingocir.py            # full pipeline
                       python src/tier5_lingocir.py --quick    # CPU smoke run
See FinalDestination.md §4.C for the method's derivation and roadmap.
"""

from __future__ import annotations

import argparse
import tarfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

try:                                        # optional: only for the training-curve PNG
    import matplotlib
    matplotlib.use("Agg")                   # headless: render to file, never open a window
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ModuleNotFoundError:
    HAVE_MPL = False

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, AutoTokenizer
from transformers.masking_utils import create_causal_mask

from data_loader import ATTRIBUTE_NAMES, _get_artifacts_dir
from clip_features import CLIP_MODEL_NAME, FEATURE_DIM, load_image_features
from clip_prompts import ATTR_PHRASES
from eval import parse_query, evaluate_query, format_results_table, load_eval_json, find_eval_json
from results_saver import save_results_csv, output_subdir
from tier1_GDE import _load_train_features, _load_train_attributes
from tier3_dgp import PhiConfig, QueryGenerator, info_nce
from tier3_train_utils import assert_db_frozen


PLACEHOLDER_TOKEN = "<sstar>"

_LOG_FILE: Path | None = None


def log(msg: str) -> None:
    # Timestamped, flushed, tee'd to the run log — progress must be visible live over SSH.
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _LOG_FILE is not None:
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Phrase maps — positive side reuses clip_prompts.ATTR_PHRASES, negative side is net-new
# ---------------------------------------------------------------------------

def positive_phrase(attr_name: str) -> str:
    # First canonical synonym for +attr, wired as "with {noun}" or a bare adjective.
    kind, phrases = ATTR_PHRASES[attr_name]
    phrase = phrases[0]
    return phrase if kind == "adj" else f"with {phrase}"


# ANTONYM_MAP: CelebA attribute name -> natural-language phrase for the NEGATED (-attr) case.
# Net-new (no existing table covers negation phrasing). Every entry must read grammatically
# after "a photo of [S*], ..."; -Male/-Young are full noun-phrase replacements, not "with"
# clauses, matching FinalDestination.md's own antonym examples verbatim.
ANTONYM_MAP: dict[str, str] = {
    "5_o_Clock_Shadow": "with a clean-shaven face", "Arched_Eyebrows": "with flat, unarched eyebrows",
    "Attractive": "with an unremarkable appearance", "Bags_Under_Eyes": "with smooth skin under the eyes",
    "Bald": "with a full head of hair", "Bangs": "with hair swept back off the forehead",
    "Big_Lips": "with thin lips", "Big_Nose": "with a small nose",
    "Black_Hair": "without black hair", "Blond_Hair": "without blond hair",
    "Blurry": "in sharp focus", "Brown_Hair": "without brown hair",
    "Bushy_Eyebrows": "with thin, sparse eyebrows", "Chubby": "with a slim face",
    "Double_Chin": "with a well-defined jawline", "Eyeglasses": "not wearing eyeglasses",
    "Goatee": "with no goatee", "Gray_Hair": "without gray hair",
    "Heavy_Makeup": "with little to no makeup", "High_Cheekbones": "with low, soft cheekbones",
    "Male": "a woman", "Mouth_Slightly_Open": "with the mouth fully closed",
    "Mustache": "with no mustache", "Narrow_Eyes": "with wide open eyes",
    "No_Beard": "with a beard", "Oval_Face": "with an angular face shape",
    "Pale_Skin": "with a tanned complexion", "Pointy_Nose": "with a rounded nose",
    "Receding_Hairline": "with a full, low hairline", "Rosy_Cheeks": "with pale, even-toned cheeks",
    "Sideburns": "with no sideburns", "Smiling": "with a serious expression",
    "Straight_Hair": "with curly or wavy hair", "Wavy_Hair": "with straight hair",
    "Wearing_Earrings": "not wearing earrings", "Wearing_Hat": "not wearing a hat",
    "Wearing_Lipstick": "not wearing lipstick", "Wearing_Necklace": "not wearing a necklace",
    "Wearing_Necktie": "not wearing a necktie", "Young": "an elderly person",
}


def _verify_antonym_coverage() -> None:
    # Tripwire (mirrors clip_prompts._verify_coverage): ANTONYM_MAP covers exactly the 40 attrs.
    missing = [n for n in ATTRIBUTE_NAMES if n not in ANTONYM_MAP]
    extra = [n for n in ANTONYM_MAP if n not in ATTRIBUTE_NAMES]
    assert not missing, f"ANTONYM_MAP missing attributes: {missing}"
    assert not extra, f"ANTONYM_MAP has unknown attributes: {extra}"


_verify_antonym_coverage()


def compose_template(pos_names: list[str], neg_names: list[str]) -> str:
    # Build "a photo of [S*], phrase1, phrase2, ..." — positives via positive_phrase(),
    # negatives via ANTONYM_MAP. Order is fixed (positives then negatives) for
    # reproducibility; the text transformer's self-attention integrates them regardless
    # of order — that integration IS Φ, so no hand-coded polarity weighting is applied.
    clauses = [positive_phrase(a) for a in pos_names] + [ANTONYM_MAP[b] for b in neg_names]
    if not clauses:
        return f"a photo of {PLACEHOLDER_TOKEN}"
    return f"a photo of {PLACEHOLDER_TOKEN}, " + ", ".join(clauses)


# ---------------------------------------------------------------------------
# Pseudo-token injection — the core mechanic
# ---------------------------------------------------------------------------

def build_clip_and_tokenizer(device: torch.device) -> tuple[CLIPModel, AutoTokenizer, int]:
    # Load frozen CLIP + tokenizer, extend the vocab by one trainable-at-forward-time row
    # for <sstar>, freeze every CLIP parameter. Extending the vocab (vs. reusing an existing
    # rare token as a placeholder) is provably safe — resize_token_embeddings leaves every
    # pre-existing row byte-identical (test/test_tier5_lingocir.py spike 3) — and avoids a
    # real token's own pretrained (semantically loaded) embedding biasing S*'s init.
    tokenizer = AutoTokenizer.from_pretrained(CLIP_MODEL_NAME)
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    tokenizer.add_tokens([PLACEHOLDER_TOKEN])
    model.text_model.resize_token_embeddings(len(tokenizer))     # CLIPModel-level raises NotImplementedError
    freeze_clip(model)
    placeholder_id = tokenizer.convert_tokens_to_ids(PLACEHOLDER_TOKEN)
    return model, tokenizer, placeholder_id


def freeze_clip(clip_model: CLIPModel) -> None:
    # Zero gradient on every CLIP parameter, including the resized embedding table as a
    # whole tensor. The new <sstar> row is never a requires_grad=True leaf read directly —
    # it enters the graph only via a tensor-value substitution (encode_composed_query step 2),
    # whose gradient flows to g, never to the frozen nn.Embedding.weight.
    for p in clip_model.parameters():
        p.requires_grad_(False)
    clip_model.eval()


def encode_composed_query(
    clip_model: CLIPModel,
    tokenizer: AutoTokenizer,
    placeholder_id: int,
    template_strs: str | list[str],
    s_star: torch.Tensor,             # [B, hidden] — g(v_ref), one row per template instance
    device: torch.device,
) -> torch.Tensor:
    # Encode composed template(s) with the <sstar>-row overridden by a learned embedding,
    # replicating CLIPTextTransformer.forward's internals directly (embeddings -> causal
    # mask -> encoder -> final_layer_norm -> EOS pooling -> text_projection) because
    # CLIPTextModel.forward hard-requires input_ids and doesn't expose inputs_embeds as a
    # top-level parameter (transformers 5.12.1, modeling_clip.py:520-593).
    # (Note: pooling searches explicitly for tokenizer.eos_token_id — never plain
    # argmax(input_ids) — because extending the vocab makes <sstar>'s id the new max,
    # which would make the library's own legacy pooling path silently pool at the wrong
    # position; see test/test_tier5_lingocir.py spike 4.)
    if isinstance(template_strs, str):
        template_strs = [template_strs]
    text_model = clip_model.text_model
    ids = tokenizer(template_strs, padding=True, return_tensors="pt").to(device)
    input_ids, attention_mask = ids["input_ids"], ids["attention_mask"]

    tok_emb = text_model.embeddings.token_embedding(input_ids).clone()   # clone: avoid an
    mask = (input_ids == placeholder_id)                                  # in-place write
    assert (mask.sum(dim=1) == 1).all(), "each template must contain <sstar> exactly once"
    tok_emb[mask] = s_star.to(tok_emb.dtype)

    hidden = text_model.embeddings(inputs_embeds=tok_emb)
    causal_mask = create_causal_mask(
        config=text_model.config, inputs_embeds=hidden,
        attention_mask=attention_mask, past_key_values=None,
    )
    enc_out = text_model.encoder(inputs_embeds=hidden, attention_mask=causal_mask, is_causal=True)
    last = text_model.final_layer_norm(enc_out.last_hidden_state)

    eos_pos = (input_ids == tokenizer.eos_token_id).int().argmax(dim=-1)
    pooled = last[torch.arange(input_ids.shape[0], device=device), eos_pos]
    return clip_model.text_projection(pooled)                            # [B, 512], un-normalized


# ---------------------------------------------------------------------------
# Inversion MLP g: v_ref -> S*
# ---------------------------------------------------------------------------

class InversionMLP(nn.Module):
    # g: v_ref (512-d unit CLIP image embedding) -> S* (CLIP text hidden-size token).
    # Pic2Word-style pseudo-token inversion, no cross-attention of its own — the composed
    # sentence's self-attention over [S*, attribute clauses] IS the fusion (FinalDestination
    # §4.C: "the text encoder's self-attention IS Φ"). Output is NOT unit-normalized: S*
    # lives in text hidden space pre-causal-attention, not on the retrieval sphere.
    def __init__(self, d_in: int = FEATURE_DIM, hidden: int = 512, d_out: int | None = None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, d_out or d_in),
        )

    def forward(self, v_ref: torch.Tensor) -> torch.Tensor:
        return self.net(v_ref)


def cycle_loss(
    clip_model: CLIPModel, tokenizer: AutoTokenizer, placeholder_id: int,
    g: InversionMLP, v_ref: torch.Tensor, device: torch.device,
) -> torch.Tensor:
    # L_cycle = 1 - cos(v_ref, encode("a photo of [S*]")) — the bare-template round trip.
    # Forces g to place S* such that CLIP's text tower, with no attribute clauses appended,
    # reconstructs the reference identity it was given. The only signal tying S* to v_ref's
    # SPECIFIC identity (info_nce alone only constrains the full composed query).
    s_star = g(v_ref)
    q_bare = encode_composed_query(
        clip_model, tokenizer, placeholder_id, f"a photo of {PLACEHOLDER_TOKEN}", s_star, device,
    )
    return (1.0 - F.cosine_similarity(F.normalize(v_ref, dim=-1), F.normalize(q_bare, dim=-1))).mean()


# ---------------------------------------------------------------------------
# Slerp text-anchoring — pulls the composed text query back toward the image manifold
# ---------------------------------------------------------------------------

def slerp(a: torch.Tensor, b: torch.Tensor, t: float, eps: float = 1e-6) -> torch.Tensor:
    # slerp(a,b,t) = sin((1-t)Ω)/sin(Ω)·a + sin(tΩ)/sin(Ω)·b, Ω = arccos(a·b). Falls back to
    # linear interpolation when Ω≈0 (0/0 guard, same pattern as manifold.exp_map's eps).
    # a = composed text query, b = v_ref, t fixed (not learned — keeps g the only trainable
    # module). Mitigates the modality gap (FinalDestination.md §1.3): pure-text queries sit
    # offset from the image manifold the frozen DB lives on; slerping partway toward v_ref
    # pulls the query back onto it.
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    dot = (a * b).sum(-1, keepdim=True).clamp(-1 + eps, 1 - eps)
    omega = torch.arccos(dot)
    sin_omega = torch.sin(omega)
    small = sin_omega.abs() < eps
    coef_a = torch.where(small, torch.full_like(sin_omega, 1 - t), torch.sin((1 - t) * omega) / sin_omega)
    coef_b = torch.where(small, torch.full_like(sin_omega, t), torch.sin(t * omega) / sin_omega)
    return F.normalize(coef_a * a + coef_b * b, dim=-1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LingoCIRConfig:
    """Tier-5 LingoCIR hyperparameters. Defaults are the headline VM run."""
    # Inversion MLP
    hidden: int = 512

    # Training
    lr: float = 1e-4
    weight_decay: float = 1e-2
    tau: float = 0.07             # InfoNCE temperature (cosine-bound scores, same as tier3)
    lambda_cycle: float = 1.0
    epochs: int = 20
    batch_queries: int = 64
    steps_div: int = 50           # steps_per_epoch = N_train // (batch_queries · steps_div)
    k_pos_max: int = 3
    k_neg_max: int = 2
    n_pos_targets: int = 8
    n_hard_neg: int = 16
    hamming_max: int = 2
    seed: int = 0

    # Slerp text-anchoring
    slerp_t: float = 0.3          # 0.0 disables anchoring (ablation)

    # Evaluation
    chunk: int = 256              # sources per batch, matches tier4_bayes chunking pattern
    max_seq_len: int = 24         # template strings are short; keeps padding tight

    def phi_gen_config(self) -> PhiConfig:
        # Borrow PhiConfig ONLY to drive QueryGenerator (identical sampling protocol as tier3/4).
        return PhiConfig(
            k_pos_max=self.k_pos_max, k_neg_max=self.k_neg_max,
            n_pos_targets=self.n_pos_targets, n_hard_neg=self.n_hard_neg,
            hamming_max=self.hamming_max, seed=self.seed,
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_lingocir(
    cfg: LingoCIRConfig, dev: torch.device, max_steps: int | None = None,
) -> tuple[InversionMLP, CLIPModel, AutoTokenizer, int, list[float]]:
    # Train g with InfoNCE + cycle loss on synthetic train-split queries (QueryGenerator
    # mirrors the eval GT rule; no leakage). CLIP is frozen throughout — only g.parameters()
    # is ever passed to the optimizer. Returns (g on CPU, clip_model, tokenizer,
    # placeholder_id, loss curve).
    torch.manual_seed(cfg.seed)
    clip_model, tokenizer, placeholder_id = build_clip_and_tokenizer(dev)
    train_feats = _load_train_features().to(dev)
    train_attrs = _load_train_attributes()

    g = InversionMLP(hidden=cfg.hidden, d_out=clip_model.text_model.config.hidden_size).to(dev)
    gen = QueryGenerator(train_attrs, cfg.phi_gen_config())
    opt = torch.optim.AdamW(g.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    steps_per_epoch = max(1, gen.N // (cfg.batch_queries * cfg.steps_div))
    if max_steps is not None:
        steps_per_epoch = min(steps_per_epoch, max_steps)
    total_steps = cfg.epochs * steps_per_epoch
    log(f"Stage: {sum(p.numel() for p in g.parameters()):,} params (g only), "
        f"{cfg.epochs} epochs x {steps_per_epoch} steps x {cfg.batch_queries} queries")

    curve, step, t0 = [], 0, time.time()
    for epoch in range(cfg.epochs):
        g.train()
        running, seen = 0.0, 0
        for _ in range(steps_per_epoch):
            opt.zero_grad()
            batch_loss, n = torch.zeros((), device=dev), 0
            for _ in range(cfg.batch_queries):
                sample = gen.sample_query()
                if sample is None:
                    continue
                r, pos_names, neg_names, pos_ids, hard_ids = sample
                v_ref = train_feats[r].unsqueeze(0)

                s_star = g(v_ref)
                template = compose_template(pos_names, neg_names)
                q_raw = encode_composed_query(clip_model, tokenizer, placeholder_id, template, s_star, dev)
                q = slerp(q_raw, v_ref, cfg.slerp_t).squeeze(0)

                pos = train_feats[pos_ids.to(dev)]
                neg = train_feats[hard_ids.to(dev)] if hard_ids.numel() else train_feats[:0]
                loss_contrastive = info_nce(q, pos, neg, cfg.tau)
                loss_cycle = cycle_loss(clip_model, tokenizer, placeholder_id, g, v_ref, dev)
                batch_loss = batch_loss + loss_contrastive + cfg.lambda_cycle * loss_cycle
                n += 1
            if n == 0:
                continue
            loss = batch_loss / n
            loss.backward()
            torch.nn.utils.clip_grad_norm_(g.parameters(), 5.0)
            opt.step()
            running += loss.item() * n
            seen += n
            step += 1
        sched.step()
        curve.append(running / max(seen, 1))
        eta = (time.time() - t0) / max(step, 1) * (total_steps - step) / 60
        log(f"  epoch {epoch + 1:02d}/{cfg.epochs}  loss={curve[-1]:.4f}  eta={eta:.1f}min")

    return g.cpu(), clip_model, tokenizer, placeholder_id, curve


def load_or_train_g(
    cfg: LingoCIRConfig, dev: torch.device, force: bool = False,
) -> tuple[InversionMLP, CLIPModel, AutoTokenizer, int, list[float]]:
    # Cache contract mirrors tier4_bayes.load_or_train_probes: load if present, else train + save.
    path = _get_artifacts_dir() / "tier5_lingocir_g.pt"
    if path.exists() and not force:
        clip_model, tokenizer, placeholder_id = build_clip_and_tokenizer(dev)
        g = InversionMLP(hidden=cfg.hidden, d_out=clip_model.text_model.config.hidden_size)
        g.load_state_dict(torch.load(path, map_location="cpu"))
        log(f"[OK] Loaded cached g: {path}")
        return g.to(dev), clip_model, tokenizer, placeholder_id, []
    log("Training InversionMLP g...")
    g, clip_model, tokenizer, placeholder_id, curve = train_lingocir(cfg, dev)
    torch.save(g.state_dict(), path)
    log(f"  Saved: {path}")
    return g.to(dev), clip_model, tokenizer, placeholder_id, curve


# ---------------------------------------------------------------------------
# Retrieval seam — CONTRACT §5/§7
# ---------------------------------------------------------------------------

def make_lingocir_chunk(
    clip_model: CLIPModel, tokenizer: AutoTokenizer, placeholder_id: int,
    g: InversionMLP, feats: torch.Tensor, cfg: LingoCIRConfig, query_str: str, dev: torch.device,
    slerp_t: float,
):
    # Curry one query into score_chunk(srcs) -> [S, N] cosine scores, batching the text
    # encode over all `srcs` in one forward pass (same template text, varying S* per source).
    pos_names, neg_names = parse_query(query_str)
    template = compose_template(pos_names, neg_names)
    g.eval()

    @torch.no_grad()
    def score_chunk(srcs: list[int]) -> torch.Tensor:
        v_ref = feats[srcs]                                          # [S, 512]
        s_star = g(v_ref)                                            # [S, 512]
        templates = [template] * len(srcs)                           # same text, batched S*
        q_raw = encode_composed_query(clip_model, tokenizer, placeholder_id, templates, s_star, dev)
        q = slerp(q_raw, v_ref, slerp_t)
        return q @ feats.T                                            # [S, N]

    return score_chunk


def make_get_ranking(
    clip_model: CLIPModel, tokenizer: AutoTokenizer, placeholder_id: int,
    g: InversionMLP, feats: torch.Tensor, cfg: LingoCIRConfig, query_str: str, dev: torch.device,
    slerp_t: float | None = None,
):
    # CONTRACT §5/§7 seam — curry one query into get_ranking(src_idx) (single-source path;
    # bulk evaluation goes through the chunked driver in main() instead).
    chunk_fn = make_lingocir_chunk(
        clip_model, tokenizer, placeholder_id, g, feats, cfg, query_str, dev,
        cfg.slerp_t if slerp_t is None else slerp_t,
    )

    @torch.no_grad()
    def get_ranking(src_idx: int) -> list[int]:
        scores = chunk_fn([src_idx]).squeeze(0)
        scores[src_idx] = float("-inf")
        return torch.argsort(scores, descending=True).tolist()

    return get_ranking


# ---------------------------------------------------------------------------
# Chunked evaluation driver (official harness underneath, mirrors tier4_bayes._evaluate_scorer)
# ---------------------------------------------------------------------------

def _evaluate_lingocir(
    clip_model: CLIPModel, tokenizer: AutoTokenizer, placeholder_id: int, g: InversionMLP,
    feats: torch.Tensor, cfg: LingoCIRConfig, gt_list: list[dict], dev: torch.device, slerp_t: float,
) -> dict:
    # Score every JSON query in source-chunks for throughput (one batched text encode per
    # chunk, not per source). Self-exclusion per CONTRACT §5.
    results = {}
    t0 = time.time()
    for qi, entry in enumerate(gt_list):
        chunk_fn = make_lingocir_chunk(
            clip_model, tokenizer, placeholder_id, g, feats, cfg, entry["query"], dev, slerp_t,
        )
        srcs = [int(s) for s in entry["ground_truth"].keys()]
        rankings: dict[int, list[int]] = {}
        for lo in range(0, len(srcs), cfg.chunk):
            chunk = srcs[lo:lo + cfg.chunk]
            scores = chunk_fn(chunk)                                          # [S, N]
            rows = torch.arange(len(chunk), device=scores.device)
            cols = torch.as_tensor(chunk, device=scores.device)
            scores[rows, cols] = float("-inf")                               # §5 self-exclusion
            top = torch.topk(scores, min(100, scores.shape[1]), dim=1).indices.cpu()
            rankings.update({s: top[j].tolist() for j, s in enumerate(chunk)})
        results[entry["query"]] = evaluate_query(entry["ground_truth"], rankings.__getitem__)
        r = results[entry["query"]]
        log(f"  [slerp_t={slerp_t}] {qi + 1:02d}/{len(gt_list)}  {entry['query']:<46s}"
            f"  R@5={r['recall@5']:.4f}  ({time.time() - t0:.0f}s)")
    return results


# ---------------------------------------------------------------------------
# Results bundle — carry everything back from the VM clone to any local clone
# ---------------------------------------------------------------------------

def bundle_results(out_dir: Path) -> Path:
    # One tar.gz with repo-root-relative arcnames (output/tier5_lingocir/*, the g cache),
    # so extracting at any clone's root restores results in place. The tokenizer's added
    # token needs no caching: add_tokens(["<sstar>"]) in the same order deterministically
    # reproduces the same id, only g.state_dict() matters.
    root = Path(__file__).parent.parent
    bundle = root / "output" / "tier5_lingocir_bundle.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for p in sorted(out_dir.rglob("*")):
            if p.is_file():
                tar.add(p, arcname=str(p.relative_to(root)))
        g_path = _get_artifacts_dir() / "tier5_lingocir_g.pt"
        if g_path.exists():
            tar.add(g_path, arcname=str(g_path.relative_to(root)))
    log(f"Bundle written: {bundle}")
    log("Pull it back to your local clone with (from local repo root):")
    log("  powershell scripts/pull_tier5_lingocir_results.ps1")
    log("  # or manually:")
    log("  scp -P 5040 disi@lab-d2fdded3-9ba4-4188-80fa-16c27a4cd022.westeurope.cloudapp.azure.com:"
        "~/Deep-Learning-Project/output/tier5_lingocir_bundle.tar.gz .")
    log('  tar -xzf tier5_lingocir_bundle.tar.gz   # extracts output/tier5_lingocir/* in place')
    return bundle


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global _LOG_FILE
    ap = argparse.ArgumentParser(description="Tier-5 LingoCIR: pseudo-token textual inversion")
    ap.add_argument("--quick", action="store_true", help="CPU smoke run (tiny epochs, 2 queries)")
    ap.add_argument("--skip-eval", action="store_true", help="train g only, skip benchmark eval")
    ap.add_argument("--force-train", action="store_true", help="retrain g even if cached")
    ap.add_argument("--epochs", type=int, default=None, help="override training epochs")
    ap.add_argument("--slerp-t", type=float, default=None, help="override Slerp anchoring weight")
    ap.add_argument("--device", type=str, default=None, help="cuda | cpu (default: auto)")
    args = ap.parse_args()

    cfg = LingoCIRConfig()
    if args.epochs is not None:
        cfg = replace(cfg, epochs=args.epochs)
    if args.slerp_t is not None:
        cfg = replace(cfg, slerp_t=args.slerp_t)
    if args.quick:
        cfg = replace(cfg, epochs=2, chunk=64)

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = output_subdir("tier5_lingocir")
    _LOG_FILE = out_dir / "train_log.txt"
    tag = "quick" if args.quick else "full"
    log(f"Device: {dev}" + (f"  GPU: {torch.cuda.get_device_name(0)}" if dev.type == "cuda" else ""))

    g, clip_model, tokenizer, placeholder_id, curve = load_or_train_g(cfg, dev, force=args.force_train)

    if HAVE_MPL and curve:
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(curve) + 1), curve)
        plt.xlabel("epoch"), plt.ylabel("loss (InfoNCE + λ·cycle)"), plt.title("Tier-5 LingoCIR training")
        plt.tight_layout()
        plt.savefig(out_dir / "tier5_training_curve.png", dpi=120)
        plt.close()

    if args.skip_eval:
        bundle_results(out_dir)
        return

    feats = load_image_features().to(dev)
    gt_list = load_eval_json(find_eval_json())
    if args.quick:
        gt_list = gt_list[:2]

    probe_query = gt_list[0]["query"]
    probe_src = next(int(s) for s in gt_list[0]["ground_truth"].keys())
    assert_db_frozen(
        feats,
        make_get_ranking(clip_model, tokenizer, placeholder_id, g, feats, cfg, probe_query, dev),
        probe_src=probe_src,
    )

    for slerp_t in sorted({0.0, cfg.slerp_t}):
        log(f"Evaluating — slerp_t={slerp_t}")
        results = _evaluate_lingocir(clip_model, tokenizer, placeholder_id, g, feats, cfg, gt_list, dev, slerp_t)
        print("\n" + format_results_table(results))
        save_results_csv(results, out_dir / f"tier5_lingocir_{tag}_slerp{slerp_t}.csv")
        mean_r5 = sum(r["recall@5"] for r in results.values()) / len(results)
        log(f"MEAN R@5 (slerp_t={slerp_t}) = {mean_r5:.4f}")

    bundle_results(out_dir)


if __name__ == "__main__":
    main()
