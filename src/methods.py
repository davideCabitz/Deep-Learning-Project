"""Retrieval methods — Tier-0 vanilla latent-arithmetic baseline.

This is the **floor** of the project (ROADMAP.md, Tier-0). It builds a composite
query embedding by naive vector arithmetic in CLIP space and scores the frozen
image DB by cosine similarity::

    q = normalize( v_ref + alpha * sum(t_pos) - beta * sum(t_neg) )

No SVD, no learning, no modality-gap correction — that crudeness is the point:
it establishes the lower bound that Tier-1 (CLAY) and Tier-2 (our fusion module Φ)
must beat, and it exercises the shared evaluation harness end-to-end (Milestone M1).

Everything here speaks the data contract verbatim:

- **Image features** are the cached `[N, 512]` L2-normalized table (CONTRACT.md §6),
  built in Colab and dropped into `artifacts/` (see `data_loader.load_image_features`).
- **Text features** for the 40 attributes are injected as a dict
  ``{attr_name: [512] tensor}`` so this module never needs `transformers` locally
  (CLIP lives in Colab). Build the dict there, or use
  `build_text_features(...)` below if `transformers` is available.
- The public entry point matches the **one shared signature** (CONTRACT.md §7):
  ``score(T_pos, T_neg, v_ref_idx, image_features, attributes, **kwargs) -> ranking``
  where a *ranking* is a ``list[int]`` of dataset indices, best-first, with the
  **source removed** (CONTRACT.md §5).

`make_get_ranking` adapts this into the ``get_ranking(source_idx)`` callback that
`eval.evaluate_all` consumes, so scoring all 14 queries is a one-liner.
"""

from __future__ import annotations

from typing import Callable

import torch

from eval import parse_query


# ---------------------------------------------------------------------------
# 1. The core baseline — build the composite query, score the DB
# ---------------------------------------------------------------------------
def tier0_query_vector(
    v_ref: torch.Tensor,
    t_pos: list[torch.Tensor],
    t_neg: list[torch.Tensor],
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """Compose the Tier-0 query: ``normalize(v_ref + α·Σt⁺ − β·Σt⁻)``.

    All inputs are 1-D ``[512]`` tensors; ``v_ref`` is the reference image's row
    from the feature table. Returns a unit-length ``[512]`` query vector.
    """
    q = v_ref.clone()
    for t in t_pos:
        q = q + alpha * t
    for t in t_neg:
        q = q - beta * t
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def score(
    T_pos: list[str],
    T_neg: list[str],
    v_ref_idx: int,
    image_features: torch.Tensor,
    attributes: torch.Tensor | None = None,
    *,
    text_features: dict[str, torch.Tensor],
    alpha: float = 1.0,
    beta: float = 1.0,
) -> list[int]:
    """Tier-0 retrieval — the shared method signature (CONTRACT.md §7).

    Args:
        T_pos, T_neg: attribute-name lists from ``eval.parse_query`` (e.g. ``["Smiling"]``).
        v_ref_idx: source image's **dataset index** (CONTRACT.md §0).
        image_features: ``[N, 512]`` L2-normalized table (CONTRACT.md §6).
        attributes: unused by Tier-0; accepted to honor the shared signature.
        text_features: dict ``{attr_name: [512] tensor}`` (L2-normalized). Required.
        alpha, beta: positive / negative condition weights (ablation knobs).

    Returns:
        ``list[int]`` of dataset indices, **best match first**, with ``v_ref_idx``
        removed (CONTRACT.md §5). Length = corpus − 1.
    """
    feats = image_features
    v_ref = feats[v_ref_idx]

    try:
        t_pos = [text_features[name] for name in T_pos]
        t_neg = [text_features[name] for name in T_neg]
    except KeyError as e:
        raise KeyError(
            f"No text feature for attribute {e.args[0]!r}. text_features must cover "
            f"every attribute in the queries (keys = CONTRACT.md §1 names)."
        ) from None

    # Match dtype/device of the feature table so the matmul stays on one device.
    t_pos = [t.to(feats.dtype).to(feats.device) for t in t_pos]
    t_neg = [t.to(feats.dtype).to(feats.device) for t in t_neg]

    q = tier0_query_vector(v_ref, t_pos, t_neg, alpha=alpha, beta=beta)

    # Cosine == dot product, since both q and every row are L2-normalized.
    scores = feats @ q  # [N]

    # Sort all rows best-first, then drop the source from its own ranking
    # (CONTRACT.md §5) so the result length is corpus − 1.
    ranking = torch.argsort(scores, descending=True).tolist()
    ranking.remove(v_ref_idx)
    return ranking


# ---------------------------------------------------------------------------
# 2. Adapter to the eval harness — make a get_ranking(source_idx) per query
# ---------------------------------------------------------------------------
def make_get_ranking_factory(
    image_features: torch.Tensor,
    attributes: torch.Tensor | None,
    text_features: dict[str, torch.Tensor],
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Callable[[str], Callable[[int], list[int]]]:
    """Build the ``make_get_ranking`` callable that ``eval.evaluate_all`` expects.

    Returns a function ``make_get_ranking(query_str) -> get_ranking(source_idx)``.
    Each query string is parsed once; the resulting ``get_ranking`` closes over the
    parsed ``T_pos``/``T_neg`` and the frozen tables.
    """
    def make_get_ranking(query_str: str) -> Callable[[int], list[int]]:
        T_pos, T_neg = parse_query(query_str)

        def get_ranking(source_idx: int) -> list[int]:
            return score(
                T_pos, T_neg, source_idx,
                image_features, attributes,
                text_features=text_features, alpha=alpha, beta=beta,
            )

        return get_ranking

    return make_get_ranking


# ---------------------------------------------------------------------------
# 3. Optional: build text features locally (only if transformers is installed)
# ---------------------------------------------------------------------------
def attribute_to_prompt(attr_name: str) -> str:
    """Default prompt template: ``Black_Hair`` → ``"a photo of a person with black hair"``."""
    return f"a photo of a person with {attr_name.replace('_', ' ').lower()}"


def build_text_features(
    attribute_names: list[str],
    *,
    prompt_fn: Callable[[str], str] = attribute_to_prompt,
    model_name: str = "openai/clip-vit-base-patch32",
    device: str | None = None,
) -> dict[str, torch.Tensor]:
    """Encode the 40 attribute prompts with CLIP into ``{name: [512]}`` (L2-normalized).

    Convenience for when ``transformers`` is available (e.g. in Colab, or locally).
    The Tier-0 baseline does not require this — you can build the dict however you
    like, as long as the vectors are unit-length and share the image feature space.
    """
    from transformers import CLIPModel, CLIPProcessor  # local import: optional dep

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device).eval()

    feats: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name in attribute_names:
            inputs = processor(text=[prompt_fn(name)], return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            t = model.get_text_features(**inputs)
            t = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            feats[name] = t.squeeze(0).cpu()
    return feats


# ---------------------------------------------------------------------------
# 4. CLI — run the full Tier-0 benchmark (Milestone M1)
# ---------------------------------------------------------------------------
def run_tier0_benchmark(*, alpha: float = 1.0, beta: float = 1.0, verbose: bool = True) -> dict:
    """Score all queries with Tier-0 and return ``eval.evaluate_all`` output.

    Needs the cached image features in ``artifacts/`` (built in Colab). Text
    features are loaded from cache if present, else computed locally via CLIP.
    """
    from data_loader import (
        ATTRIBUTE_NAMES,
        load_attributes,
        load_image_features,
        load_text_features,
    )
    from eval import (
        evaluate_all,
        find_eval_json,
        format_results_table,
        load_eval_json,
    )

    image_features = load_image_features()
    attributes = load_attributes()

    text_features = load_text_features()
    if text_features is None:
        if verbose:
            print("No cached text features; encoding 40 attribute prompts with CLIP...")
        text_features = build_text_features(ATTRIBUTE_NAMES)

    gt_list = load_eval_json(find_eval_json())
    make_get_ranking = make_get_ranking_factory(
        image_features, attributes, text_features, alpha=alpha, beta=beta,
    )
    results = evaluate_all(gt_list, make_get_ranking)

    if verbose:
        print(format_results_table(results))
    return results


if __name__ == "__main__":
    run_tier0_benchmark()
