"""
Per-attribute prompt bank — the raw material CLAY's subspace needs.

Tier-0 (latent arithmetic) uses ONE text vector per attribute. CLAY does NOT: it
builds a *subspace* per condition by running SVD on a STACK of many prompts that all
describe the same attribute (CLAY.md §3.2):

    T_c = [t_c_1, ..., t_c_n]^T  in R^{n x d}      # n DIFFERENT sentences, one condition
    log_{mu_c}(T_c) = U S V^T                       # SVD on the manifold-mapped stack
    P_c = V_k V_k^T                                  # top-k subspace projector

With n = 1 (the existing clip_attr_text_features.pt) that stack is rank-1: SVD gives a
single direction, k=50 is meaningless, and every accuracy lever (adaptive-k,
all-but-the-top, soft sigma-weighting) has nothing to operate on. So CLAY is undefined
until this file exists.

CLAY generates the n prompts with an LLM (CLAY.md §5.1, ChatGPT-5). We instead generate
them with deterministic TEMPLATES — a reproducible, from-scratch substitute (course
policy, project_specification.md §6). The geometry downstream is identical; only the
*source* of the n sentences differs. This is stated as a methodology choice in the report.

Output artifact (new):  artifacts/clip_attr_prompt_bank.pt
    Shape [40, n, 512], float32, each [j, i] row L2-normalized.
    bank[j] is the prompt stack T_c for attribute ATTRIBUTE_NAMES[j].

One-time build:  python src/clip_prompts.py
"""

import torch
from transformers import CLIPModel, AutoTokenizer

from data_loader import ATTRIBUTE_NAMES, _get_artifacts_dir
from clip_features import CLIP_MODEL_NAME, FEATURE_DIM, _pick_device


# ---------------------------------------------------------------------------
# Sentence frames — the "shape" of the description. Each {phrase} slot is filled
# with one synonym phrase (below). Frames x synonyms = the prompt bank for an attr.
# Kept varied (subject wording, sentence structure) so CLIP places the encodings in
# genuinely different spots — that spread IS the subspace SVD will recover.
# ---------------------------------------------------------------------------
FRAMES = [
    "a photo of a person with {phrase}",
    "a photo of a person who has {phrase}",
    "a portrait of someone with {phrase}",
    "a close-up of a face with {phrase}",
    "an image of a person showing {phrase}",
    "a headshot of an individual with {phrase}",
    "a picture of a face with {phrase}",
    "a cropped photo of a person with {phrase}",
    "a selfie of someone with {phrase}",
    "a candid shot of a person having {phrase}",
    "a frontal portrait of a face with {phrase}",
    "a clear photograph of a person with {phrase}",
]

# A few attributes read better as "a person <who-is>" than "a person with <noun>".
# These frames are used for attributes flagged predicative (see ATTR_PHRASES).
PREDICATIVE_FRAMES = [
    "a photo of a {phrase} person",
    "a photo of someone who is {phrase}",
    "a portrait of a {phrase} person",
    "a close-up of a {phrase} face",
    "an image of a {phrase} individual",
    "a headshot of a {phrase} person",
    "a picture of a {phrase} face",
    "a cropped photo of a {phrase} person",
    "a selfie of a {phrase} person",
    "a candid shot of a {phrase} individual",
    "a frontal portrait of a {phrase} person",
    "a clear photograph of a {phrase} person",
]


# ---------------------------------------------------------------------------
# Per-attribute synonym phrases. Each attribute maps to:
#   ("noun" | "adj", [phrase, phrase, ...])
# "noun" -> filled into FRAMES ("...with <phrase>"); "adj" -> PREDICATIVE_FRAMES.
# Multiple synonyms give lexical spread; the frames give structural spread. Together
# they produce ~ (#frames x #synonyms) prompts per attribute, all meaning the same thing.
#
# Phrases are deliberately plain and unambiguous — CLIP ViT-B/32 is not a strong
# language model, so concrete visual words beat clever paraphrase.
# ---------------------------------------------------------------------------
ATTR_PHRASES = {
    "5_o_Clock_Shadow": ("noun", ["a five o'clock shadow", "light stubble", "a faint beard shadow", "short stubble on the face", "a day's worth of stubble"]),
    "Arched_Eyebrows": ("noun", ["arched eyebrows", "curved eyebrows", "highly arched brows", "sharply arched eyebrows", "elegantly curved brows"]),
    "Attractive": ("adj", ["attractive", "good looking", "beautiful", "handsome", "striking-looking"]),
    "Bags_Under_Eyes": ("noun", ["bags under the eyes", "puffy under-eyes", "eye bags", "swollen under-eye skin", "dark circles under the eyes"]),
    "Bald": ("adj", ["bald", "hairless on the head", "with no hair", "completely bald", "with a shaved head"]),
    "Bangs": ("noun", ["bangs", "a fringe of hair over the forehead", "hair covering the forehead", "a straight fringe", "front bangs across the brow"]),
    "Big_Lips": ("noun", ["big lips", "full lips", "large lips", "plump lips", "thick full lips"]),
    "Big_Nose": ("noun", ["a big nose", "a large nose", "a prominent nose", "a wide nose", "a broad nose"]),
    "Black_Hair": ("noun", ["black hair", "dark black hair", "jet-black hair", "deep black hair", "raven-black hair"]),
    "Blond_Hair": ("noun", ["blond hair", "blonde hair", "light golden hair", "pale yellow hair", "bright blonde hair"]),
    "Blurry": ("adj", ["blurry", "out of focus", "unsharp", "blurred and hazy", "low in sharpness"]),
    "Brown_Hair": ("noun", ["brown hair", "brunette hair", "dark brown hair", "chestnut-brown hair", "medium brown hair"]),
    "Bushy_Eyebrows": ("noun", ["bushy eyebrows", "thick eyebrows", "dense brows", "heavy bushy eyebrows", "full thick brows"]),
    "Chubby": ("adj", ["chubby", "plump", "round-faced", "full-cheeked", "heavyset in the face"]),
    "Double_Chin": ("noun", ["a double chin", "a second chin", "extra fold under the chin", "a fleshy under-chin", "a sagging under-chin"]),
    "Eyeglasses": ("noun", ["eyeglasses", "glasses", "spectacles", "a pair of glasses", "reading glasses"]),
    "Goatee": ("noun", ["a goatee", "a goatee beard", "a small pointed beard", "a chin goatee", "a trimmed goatee"]),
    "Gray_Hair": ("noun", ["gray hair", "grey hair", "silver hair", "white-gray hair", "salt-and-pepper hair"]),
    "Heavy_Makeup": ("noun", ["heavy makeup", "a lot of makeup", "thick makeup", "bold heavy makeup", "heavily applied makeup"]),
    "High_Cheekbones": ("noun", ["high cheekbones", "prominent cheekbones", "sharp high cheekbones", "well-defined cheekbones", "raised cheekbones"]),
    "Male": ("noun", ["a male face", "the face of a man", "masculine features", "a man's face", "a masculine appearance"]),
    "Mouth_Slightly_Open": ("noun", ["a slightly open mouth", "lips parted slightly", "a mouth slightly open", "barely parted lips", "a softly open mouth"]),
    "Mustache": ("noun", ["a mustache", "a moustache", "hair on the upper lip", "a thick mustache", "a trimmed mustache"]),
    "Narrow_Eyes": ("noun", ["narrow eyes", "squinted eyes", "thin eyes", "slightly closed eyes", "narrowed eyes"]),
    "No_Beard": ("adj", ["clean-shaven", "without a beard", "with a beardless face", "smooth-shaven", "free of facial hair"]),
    "Oval_Face": ("noun", ["an oval face", "an oval-shaped face", "a long oval face", "a softly oval face", "an egg-shaped face"]),
    "Pale_Skin": ("noun", ["pale skin", "fair skin", "light-toned skin", "very pale skin", "porcelain-pale skin"]),
    "Pointy_Nose": ("noun", ["a pointy nose", "a sharp nose", "a pointed nose", "a narrow pointed nose", "a thin pointy nose"]),
    "Receding_Hairline": ("noun", ["a receding hairline", "a hairline pulled back", "thinning hair at the temples", "a balding hairline", "a hairline receding at the front"]),
    "Rosy_Cheeks": ("noun", ["rosy cheeks", "pink cheeks", "flushed cheeks", "reddish rosy cheeks", "blushing cheeks"]),
    "Sideburns": ("noun", ["sideburns", "long sideburns", "hair down the sides of the face", "thick sideburns", "sideburns along the cheeks"]),
    "Smiling": ("adj", ["smiling", "with a smile", "grinning", "smiling broadly", "happily smiling"]),
    "Straight_Hair": ("noun", ["straight hair", "sleek straight hair", "perfectly straight hair", "smooth straight hair", "flat straight hair"]),
    "Wavy_Hair": ("noun", ["wavy hair", "curly wavy hair", "loosely wavy hair", "soft wavy hair", "gently waving hair"]),
    "Wearing_Earrings": ("noun", ["earrings", "ear jewelry", "studs in the ears", "dangling earrings", "a pair of earrings"]),
    "Wearing_Hat": ("noun", ["a hat", "a cap on the head", "headwear", "a brimmed hat", "a hat on the head"]),
    "Wearing_Lipstick": ("noun", ["lipstick", "colored lipstick", "lip color", "bright lipstick", "bold lipstick"]),
    "Wearing_Necklace": ("noun", ["a necklace", "a chain around the neck", "neck jewelry", "a pendant necklace", "a beaded necklace"]),
    "Wearing_Necktie": ("noun", ["a necktie", "a tie", "a knotted tie", "a formal necktie", "a tie around the collar"]),
    "Young": ("adj", ["young", "youthful", "young-looking", "youthful-faced", "in their youth"]),
}


def build_prompts_for_attribute(name: str) -> list[str]:
    # Cross sentence frames × synonym phrases → de-duplicated prompt list for one attr.
    # Frame kind (noun/adj) picks FRAMES vs PREDICATIVE_FRAMES; the spread IS the subspace.
    kind, phrases = ATTR_PHRASES[name]
    frames = PREDICATIVE_FRAMES if kind == "adj" else FRAMES

    prompts = []
    seen = set()
    for phrase in phrases:
        for frame in frames:
            text = frame.format(phrase=phrase)
            if text not in seen:
                seen.add(text)
                prompts.append(text)
    return prompts


def _verify_coverage():
    # Tripwire: assert ATTR_PHRASES covers exactly the 40-attr master list — no drift.
    missing = [n for n in ATTRIBUTE_NAMES if n not in ATTR_PHRASES]
    extra = [n for n in ATTR_PHRASES if n not in ATTRIBUTE_NAMES]
    assert not missing, f"ATTR_PHRASES missing attributes: {missing}"
    assert not extra, f"ATTR_PHRASES has unknown attributes: {extra}"


@torch.no_grad()
def extract_prompt_bank(force=False):
    # Encode every attribute's prompt bank with frozen CLIP → cache [40, n, 512].
    # Per-attr prompt counts differ, so each stack is PADDED to a common n to keep
    # one dense tensor with row j == attribute j. `force` rebuilds even if cached.
    # (Note: padding rows duplicate the attr's first prompt — SVD-safe, since a
    # repeated row adds no new span direction; its singular value folds into the existing one.)
    out_path = _get_artifacts_dir() / "clip_attr_prompt_bank.pt"

    if out_path.exists() and not force:
        print(f"[OK] Prompt bank already exists: {out_path}")
        return

    _verify_coverage()

    device = _pick_device()
    print(f"Loading frozen CLIP: {CLIP_MODEL_NAME}")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(CLIP_MODEL_NAME)

    # Build every attribute's prompt list, then pad to the max length so all stacks align.
    per_attr_prompts = [build_prompts_for_attribute(name) for name in ATTRIBUTE_NAMES]
    n = max(len(p) for p in per_attr_prompts)
    counts = [len(p) for p in per_attr_prompts]
    print(f"  Prompts per attribute: min={min(counts)}, max={n} (padded to n={n})")

    bank = torch.empty(len(ATTRIBUTE_NAMES), n, FEATURE_DIM, dtype=torch.float32)

    for j, prompts in enumerate(per_attr_prompts):
        padded = prompts + [prompts[0]] * (n - len(prompts))
        tokens = tokenizer(padded, padding=True, return_tensors="pt").to(device)
        text_outputs = model.text_model(**tokens)
        feats = model.text_projection(text_outputs.pooler_output)
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)  # rows on the unit sphere
        bank[j] = feats.cpu().to(torch.float32)
        print(f"  encoded {j + 1}/{len(ATTRIBUTE_NAMES)} ({ATTRIBUTE_NAMES[j]})", end="\r")

    print()
    _verify(bank)

    torch.save(bank, out_path)
    print(f"  Saved: {out_path}")
    print(f"  Shape: {tuple(bank.shape)} = [num_attrs, prompts_per_attr, dim]")
    print("[OK] Prompt bank ready — CLAY's subspace SVD can now be built per attribute.")


def _verify(bank):
    # Tripwire: assert the bank is [40, ≥2, 512] and unit-normalized (CLAY needs n>1).
    assert bank.shape[0] == len(ATTRIBUTE_NAMES), (
        f"Expected {len(ATTRIBUTE_NAMES)} attributes, got {bank.shape[0]}."
    )
    assert bank.shape[2] == FEATURE_DIM, f"Expected {FEATURE_DIM}-d vectors, got {bank.shape[2]}."
    assert bank.shape[1] >= 2, (
        f"Only {bank.shape[1]} prompt(s) per attribute — CLAY needs n>1 for a non-trivial "
        "subspace. Add more synonyms/frames."
    )
    norms = bank.norm(p=2, dim=2)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), "Bank rows are not L2-normalized."
    print(f"  [OK] verified: [{bank.shape[0]}, {bank.shape[1]}, {FEATURE_DIM}], unit-normalized")


def load_prompt_bank():
    # Load the cached [40, n, 512] prompt bank: bank[j] is the stack T_c for
    # ATTRIBUTE_NAMES[j], each row L2-normalized. Raises if absent.
    path = _get_artifacts_dir() / "clip_attr_prompt_bank.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt bank not found. Run extract_prompt_bank() first.\n"
            f"Expected: {path}"
        )
    return torch.load(path)


if __name__ == "__main__":
    # One-time setup: python src/clip_prompts.py
    extract_prompt_bank()
