"""
Per-attribute prompt bank for OVAD — CLAY/DGP's subspace/gate raw material.

Same role as src/clip_prompts.py (CelebA) / src/awa2/clip_prompts.py (AwA2), adapted for
OVAD's 117 object-instance attributes and its OWN naming convention: every attribute name
is already "type:phrase1/phrase2/...", so synonym phrases come FOR FREE from the name
itself (unlike AwA2, which needed hand-curated synonym lists) — only the sentence frame
(is-a vs has-a) and the object noun need to vary.

The object is a generic "object"/"thing" placeholder in most frames (OVAD spans 80
categories, unlike CelebA's fixed "person"), which the `is_has_att` field of each
attribute record selects between two frame sets, mirroring CelebA's ATTR_PHRASES
noun/adj split.

Output artifact:  artifacts/ovad/clip_attr_prompt_bank.pt   [117, n, 512], rows L2-norm.

One-time build:  python src/ovad/clip_prompts.py
"""

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, CLIPModel

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / "Baselines"))
sys.path.insert(0, str(Path(__file__).parents[1] / "Training-Free"))
sys.path.insert(0, str(Path(__file__).parents[1] / "Trained"))

from ovad.data_loader import (ATTRIBUTE_NAMES, _get_artifacts_dir,
                              load_attribute_meta)

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
FEATURE_DIM = 512

# "has" frames: the phrase is a NOUN describing a part/feature the object HAS
# (e.g. "clothes color:black" -> "an object with black clothes").
HAS_FRAMES = [
    "a photo of an object with {p}",
    "a photo of a thing that has {p}",
    "a picture of an object with {p}",
    "a close-up of an object with {p}",
    "an image showing an object with {p}",
]

# "is" frames: the phrase is an ADJECTIVE/STATE describing the object itself
# (e.g. "state:closed" -> "a photo of a closed object").
IS_FRAMES = [
    "a photo of a {p} object",
    "a photo of an object that is {p}",
    "a picture of a {p} thing",
    "a close-up of a {p} object",
    "an image of an object that looks {p}",
]


def _phrases(attr_name: str) -> list[str]:
    # "type:phrase1/phrase2/..." -> [phrase1, phrase2, ...]. Multi-word phrases (e.g.
    # "stainless steel") keep their spaces; only the "/"-joined list is split.
    _, rhs = attr_name.split(":", 1)
    return rhs.split("/")


def build_prompts_for_attribute(name: str, is_has_att: str) -> list[str]:
    # Cross this attribute's synonym phrases (from its own name) x the matching frame
    # set -> de-duplicated prompt list. is_has_att comes from the attribute record, not
    # inferred from the name, since parent_type alone does not determine it (e.g. "color"
    # and "pattern" each appear under both "is" and "has" — verified against the raw
    # attribute list).
    frames = HAS_FRAMES if is_has_att == "has" else IS_FRAMES
    prompts, seen = [], set()
    for phrase in _phrases(name):
        for frame in frames:
            text = frame.format(p=phrase)
            if text not in seen:
                seen.add(text)
                prompts.append(text)
    return prompts


def _verify_coverage(meta):
    names = {m["name"] for m in meta}
    assert names == set(ATTRIBUTE_NAMES), "attribute metadata does not match ATTRIBUTE_NAMES"
    assert all(m["is_has_att"] in ("is", "has") for m in meta), "unexpected is_has_att value"


def _pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("[!] No CUDA device found - running on CPU. Encoding 117 short prompt sets is still fast.")
    return torch.device("cpu")


@torch.no_grad()
def extract_prompt_bank(force=False):
    # Encode every attribute's prompt bank with frozen CLIP -> cache [117, n, 512].
    # Stacks padded to a common n (padding repeats the first prompt — SVD-safe, DGP-safe).
    out_path = _get_artifacts_dir() / "clip_attr_prompt_bank.pt"
    if out_path.exists() and not force:
        print(f"[OK] Prompt bank already exists: {out_path}")
        return

    meta = load_attribute_meta()
    _verify_coverage(meta)

    device = _pick_device()
    print(f"Loading frozen CLIP: {CLIP_MODEL_NAME}")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(CLIP_MODEL_NAME)

    per_attr = [build_prompts_for_attribute(m["name"], m["is_has_att"]) for m in meta]
    n = max(len(p) for p in per_attr)
    print(f"  Prompts per attribute: min={min(len(p) for p in per_attr)}, max={n}")

    bank = torch.empty(len(ATTRIBUTE_NAMES), n, FEATURE_DIM, dtype=torch.float32)
    for j, prompts in enumerate(per_attr):
        padded = prompts + [prompts[0]] * (n - len(prompts))
        tokens = tokenizer(padded, padding=True, return_tensors="pt").to(device)
        text_outputs = model.text_model(**tokens)
        feats = model.text_projection(text_outputs.pooler_output)
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        bank[j] = feats.cpu().to(torch.float32)
        print(f"  encoded {j + 1}/{len(ATTRIBUTE_NAMES)} ({ATTRIBUTE_NAMES[j]})", end="\r")

    print()
    _verify(bank)
    torch.save(bank, out_path)
    print(f"  Saved: {out_path}  shape={tuple(bank.shape)}")


def _verify(bank):
    assert bank.shape[0] == len(ATTRIBUTE_NAMES), bank.shape
    assert bank.shape[2] == FEATURE_DIM, bank.shape
    assert bank.shape[1] >= 2, f"Only {bank.shape[1]} prompt(s)/attr — CLAY/DGP need n>1."
    norms = bank.norm(p=2, dim=2)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), "Bank rows not L2-normalized."
    print(f"  [OK] verified: [{bank.shape[0]}, {bank.shape[1]}, {FEATURE_DIM}], unit-normalized")


def load_prompt_bank():
    path = _get_artifacts_dir() / "clip_attr_prompt_bank.pt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt bank not found. Run extract_prompt_bank() first.\n"
                                f"Expected: {path}")
    return torch.load(path)


if __name__ == "__main__":
    # One-time build: python src/ovad/clip_prompts.py
    extract_prompt_bank()
