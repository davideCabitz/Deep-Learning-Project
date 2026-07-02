"""
Tier-5 LingoCIR verification.

Five load-bearing correctness spikes for the pseudo-token injection mechanic — this is
the highest-risk part of the method (HuggingFace's CLIPTextModel.forward hard-requires
input_ids and cannot be driven by inputs_embeds alone; extending the vocab silently
breaks the library's built-in EOS pooling). Spikes 1/3/4 gate that encode_composed_query's
manual CLIPTextTransformer reimplementation is correct BEFORE any learned g is involved;
spike 5 operationalizes "CLIP stays frozen" as an executable gradient check. Only after
all five pass should g be wired into the full training loop (tier5_lingocir.train_lingocir).

CPU-only, seconds-scale (loads real CLIP weights, no CelebA/artifacts needed).

Run:  python test/test_tier5_lingocir.py   (or: pytest test/test_tier5_lingocir.py)
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transformers import CLIPModel, AutoTokenizer
from transformers.masking_utils import create_causal_mask

from clip_features import CLIP_MODEL_NAME
from tier5_lingocir import (
    InversionMLP,
    PLACEHOLDER_TOKEN,
    build_clip_and_tokenizer,
    compose_template,
    encode_composed_query,
    freeze_clip,
)


def _manual_encode_no_override(clip_model, tokenizer, template: str, device) -> torch.Tensor:
    # Same 6-step body as encode_composed_query's steps 3-8, but WITHOUT the placeholder
    # row override — isolates "is the hand-written CLIPTextTransformer reimplementation
    # correct" from "does the S*-injection mechanic work" (spike 1 vs. spike 2).
    text_model = clip_model.text_model
    ids = tokenizer([template], return_tensors="pt").to(device)
    input_ids, attention_mask = ids["input_ids"], ids["attention_mask"]

    tok_emb = text_model.embeddings.token_embedding(input_ids)
    hidden = text_model.embeddings(inputs_embeds=tok_emb)
    causal_mask = create_causal_mask(
        config=text_model.config, inputs_embeds=hidden,
        attention_mask=attention_mask, past_key_values=None,
    )
    enc_out = text_model.encoder(inputs_embeds=hidden, attention_mask=causal_mask, is_causal=True)
    last = text_model.final_layer_norm(enc_out.last_hidden_state)
    eos_pos = (input_ids == tokenizer.eos_token_id).int().argmax(dim=-1)
    pooled = last[torch.arange(input_ids.shape[0], device=device), eos_pos]
    return clip_model.text_projection(pooled)


# ---------------------------------------------------------------------------
# 1. Manual reimplementation ≡ black box (no S*-injection, no vocab resize)
# ---------------------------------------------------------------------------

def test_manual_reimplementation_matches_black_box():
    dev = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(CLIP_MODEL_NAME)
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(dev).eval()

    text = "a photo of a cat, wearing eyeglasses"
    ids = tokenizer([text], return_tensors="pt").to(dev)
    with torch.no_grad():
        ref = model.get_text_features(**ids).pooler_output   # projected features (transformers 5.x)
        manual = _manual_encode_no_override(model, tokenizer, text, dev)

    assert torch.allclose(ref, manual, atol=1e-5), f"max diff {(ref - manual).abs().max()}"
    print("[OK] 1. manual CLIPTextTransformer reimplementation matches get_text_features exactly")


# ---------------------------------------------------------------------------
# 2. A REAL known word substituted at the S* slot reproduces the plain encode
# ---------------------------------------------------------------------------

def test_injection_matches_real_word_encode():
    dev = torch.device("cpu")
    model, tokenizer, placeholder_id = build_clip_and_tokenizer(dev)

    # "cat" must be a single BPE sub-token for exact-length comparison to be valid.
    assert len(tokenizer.tokenize("cat")) == 1, "test assumes 'cat' is one BPE token"
    cat_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("cat")[0])
    cat_row = model.text_model.embeddings.token_embedding.weight[cat_id].unsqueeze(0)

    plain_text = "a photo of cat, wearing eyeglasses"
    template = f"a photo of {PLACEHOLDER_TOKEN}, wearing eyeglasses"

    with torch.no_grad():
        ref = model.get_text_features(**tokenizer([plain_text], return_tensors="pt").to(dev)).pooler_output
        injected = encode_composed_query(model, tokenizer, placeholder_id, template, cat_row, dev)

    assert torch.allclose(ref, injected, atol=1e-4), f"max diff {(ref - injected).abs().max()}"
    print("[OK] 2. real-word injection at the S* slot reproduces the plain-text encode")


# ---------------------------------------------------------------------------
# 3. resize_token_embeddings leaves every pre-existing row byte-identical
# ---------------------------------------------------------------------------

def test_resize_preserves_pretrained_rows():
    dev = torch.device("cpu")
    base = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(dev)
    old_weight = base.text_model.embeddings.token_embedding.weight.detach().clone()
    old_vocab_size = old_weight.shape[0]

    model, _, placeholder_id = build_clip_and_tokenizer(dev)
    new_weight = model.text_model.embeddings.token_embedding.weight.detach()

    assert placeholder_id == old_vocab_size, "  <sstar> should land immediately after the old vocab"
    assert torch.equal(old_weight, new_weight[:old_vocab_size]), (
        "resize_token_embeddings perturbed a pre-existing (frozen) embedding row"
    )
    print("[OK] 3. vocab resize leaves every pre-existing embedding row byte-identical")


# ---------------------------------------------------------------------------
# 4. EOS-pooling correctness post-resize (the silent legacy-path bug + its fix)
# ---------------------------------------------------------------------------

def test_eos_pooling_survives_vocab_extension():
    dev = torch.device("cpu")
    model, tokenizer, placeholder_id = build_clip_and_tokenizer(dev)
    template = f"a photo of {PLACEHOLDER_TOKEN}, wearing eyeglasses"
    ids = tokenizer([template], return_tensors="pt").to(dev)
    input_ids = ids["input_ids"]

    naive_pos = input_ids.argmax(dim=-1)                                   # legacy (broken) path
    correct_pos = (input_ids == tokenizer.eos_token_id).int().argmax(dim=-1)
    assert naive_pos.item() != correct_pos.item(), (
        "expected <sstar>'s id to exceed eos_token_id post-resize, breaking naive argmax pooling"
    )

    text_model = model.text_model
    tok_emb = text_model.embeddings.token_embedding(input_ids)
    hidden = text_model.embeddings(inputs_embeds=tok_emb)
    causal_mask = create_causal_mask(
        config=text_model.config, inputs_embeds=hidden,
        attention_mask=ids["attention_mask"], past_key_values=None,
    )
    with torch.no_grad():
        enc_out = text_model.encoder(inputs_embeds=hidden, attention_mask=causal_mask, is_causal=True)
        last = text_model.final_layer_norm(enc_out.last_hidden_state)
    pooled_correct = last[0, correct_pos.item()]
    pooled_naive = last[0, naive_pos.item()]
    assert not torch.allclose(pooled_correct, pooled_naive, atol=1e-4), (
        "EOS and <sstar> positions coincide — test template must place <sstar> before EOS"
    )
    print("[OK] 4. EOS-id search pooling diverges from (and correctly replaces) naive argmax pooling")


# ---------------------------------------------------------------------------
# 5. Gradient isolation — CLIP frozen, only g trainable
# ---------------------------------------------------------------------------

def test_gradient_isolation():
    dev = torch.device("cpu")
    model, tokenizer, placeholder_id = build_clip_and_tokenizer(dev)
    g = InversionMLP(hidden=64, d_out=model.text_model.config.hidden_size).to(dev)

    v_ref = torch.nn.functional.normalize(torch.randn(1, 512), dim=-1)
    s_star = g(v_ref)
    template = compose_template(["Eyeglasses"], ["Smiling"])
    q = encode_composed_query(model, tokenizer, placeholder_id, template, s_star, dev)
    q.sum().backward()

    assert all(p.grad is None for p in model.parameters()), "CLIP received gradient — frozen-CLIP violation"
    assert all(p.grad is not None for p in g.parameters()), "g received no gradient — injection graph is broken"
    print("[OK] 5. gradient isolation — CLIP frozen (no grad), g fully trainable")


if __name__ == "__main__":
    test_manual_reimplementation_matches_black_box()
    test_injection_matches_real_word_encode()
    test_resize_preserves_pretrained_rows()
    test_eos_pooling_survives_vocab_extension()
    test_gradient_isolation()
    print("\nAll Tier-5 LingoCIR checks passed.")
