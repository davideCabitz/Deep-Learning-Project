"""
Per-attribute prompt bank for AwA2 — CLAY's subspace raw material (transfer study).

Same role as src/clip_prompts.py but for AwA2's 85 predicates and an animal subject.
CLAY builds a per-condition subspace by SVD over a STACK of paraphrases (n>1); Tier-0
just averages the stack. Both need this [85, n, 512] bank.

The 85 predicates fall into grammatical GROUPS (adjective / has-noun / verb / habitat /
diet). Rather than hand-write 85 phrasings, each predicate is assigned a group and a
short phrase; the group picks a frame set. Frames × the predicate's phrase variants give
the structural+lexical spread SVD recovers. (Course policy §6: deterministic templates,
not an LLM — a reproducible from-scratch substitute; stated as a methodology choice.)

Output artifact:  artifacts/awa2/clip_attr_prompt_bank.pt   [85, n, 512], rows L2-norm.

One-time build:  python src/awa2/clip_prompts.py
"""

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, CLIPModel

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / "Baselines"))
sys.path.insert(0, str(Path(__file__).parents[1] / "Training-Free"))
sys.path.insert(0, str(Path(__file__).parents[1] / "Trained"))

from awa2.clip_features import CLIP_MODEL_NAME, FEATURE_DIM, _pick_device
from awa2.data_loader import ATTRIBUTE_NAMES, _get_artifacts_dir


# Frame sets per grammatical group. {p} is the predicate phrase. Variety in subject
# wording and structure spreads the encodings — that spread IS the subspace.
FRAMES = {
    "adj": [  # "a <p> animal"  (black, big, fast, fierce, nocturnal, ...)
        "a photo of a {p} animal",
        "a photo of an animal that is {p}",
        "a picture of a {p} creature",
        "a close-up of a {p} animal",
        "an image of a {p} animal in the wild",
        "a wildlife photo of a {p} animal",
    ],
    "has": [  # "an animal with <p>"  (spots, stripes, horns, tail, flippers, ...)
        "a photo of an animal with {p}",
        "a photo of an animal that has {p}",
        "a picture of a creature with {p}",
        "a close-up of an animal with {p}",
        "an image of an animal with {p} in the wild",
        "a wildlife photo of an animal with {p}",
    ],
    "verb": [  # "an animal that <p>s"  (swims, flys, hops, walks, hunts, grazes, ...)
        "a photo of an animal that {p}",
        "a photo of an animal which {p}",
        "a picture of a creature that {p}",
        "a close-up of an animal that {p}",
        "an image of an animal that {p} in the wild",
        "a wildlife photo of an animal that {p}",
    ],
    "habitat": [  # "an animal that lives in the <p>"  (arctic, ocean, forest, ...)
        "a photo of an animal that lives in the {p}",
        "a photo of an animal found in the {p}",
        "a picture of a creature living in the {p}",
        "a close-up of an animal in the {p}",
        "an image of an animal native to the {p}",
        "a wildlife photo of an animal in the {p}",
    ],
    "diet": [  # "an animal that eats <p>"  (fish, meat, plankton, vegetation, insects)
        "a photo of an animal that eats {p}",
        "a photo of an animal feeding on {p}",
        "a picture of a creature that eats {p}",
        "a close-up of an animal eating {p}",
        "an image of an animal whose diet is {p}",
        "a wildlife photo of an animal that eats {p}",
    ],
}


# predicate -> (group, [phrase variants]). Predicate names come from predicates.txt.
ATTR_SPEC = {
    # colors / texture (adjective)
    "black": ("adj", ["black", "black-colored", "dark black"]),
    "white": ("adj", ["white", "white-colored", "pale white"]),
    "blue": ("adj", ["blue", "blue-colored", "bluish"]),
    "brown": ("adj", ["brown", "brown-colored", "brownish"]),
    "gray": ("adj", ["gray", "grey", "grayish"]),
    "orange": ("adj", ["orange", "orange-colored", "orangeish"]),
    "red": ("adj", ["red", "red-colored", "reddish"]),
    "yellow": ("adj", ["yellow", "yellow-colored", "yellowish"]),
    "patches": ("has", ["patches", "colored patches", "patchy markings"]),
    "spots": ("has", ["spots", "spotted markings", "a spotted coat"]),
    "stripes": ("has", ["stripes", "striped markings", "a striped coat"]),
    "furry": ("adj", ["furry", "covered in fur", "fur-covered"]),
    "hairless": ("adj", ["hairless", "without fur", "bare-skinned"]),
    "toughskin": ("adj", ["tough-skinned", "with tough skin", "thick-skinned"]),
    # size / build (adjective)
    "big": ("adj", ["big", "large", "large-bodied"]),
    "small": ("adj", ["small", "tiny", "small-bodied"]),
    "bulbous": ("adj", ["bulbous", "round-bodied", "rotund"]),
    "lean": ("adj", ["lean", "slender", "thin-bodied"]),
    # body parts (has)
    "flippers": ("has", ["flippers", "flipper limbs", "swimming flippers"]),
    "hands": ("has", ["hands", "grasping hands", "hand-like paws"]),
    "hooves": ("has", ["hooves", "hoofed feet", "hard hooves"]),
    "pads": ("has", ["paw pads", "soft foot pads", "padded feet"]),
    "paws": ("has", ["paws", "soft paws", "clawed paws"]),
    "longleg": ("has", ["long legs", "elongated legs", "tall legs"]),
    "longneck": ("has", ["a long neck", "an elongated neck", "a tall neck"]),
    "tail": ("has", ["a tail", "a long tail", "a bushy tail"]),
    "chewteeth": ("has", ["chewing teeth", "grinding teeth", "flat chewing teeth"]),
    "meatteeth": ("has", ["meat-eating teeth", "sharp carnivore teeth", "flesh-tearing teeth"]),
    "buckteeth": ("has", ["buck teeth", "prominent front teeth", "large incisors"]),
    "strainteeth": ("has", ["straining teeth", "filter-feeding teeth", "baleen-like teeth"]),
    "horns": ("has", ["horns", "a pair of horns", "curved horns"]),
    "claws": ("has", ["claws", "sharp claws", "curved claws"]),
    "tusks": ("has", ["tusks", "long tusks", "protruding tusks"]),
    # behavior traits (adjective)
    "smelly": ("adj", ["smelly", "foul-smelling", "strong-smelling"]),
    # locomotion (verb)
    "flys": ("verb", ["flies", "can fly", "flies through the air"]),
    "hops": ("verb", ["hops", "moves by hopping", "hops around"]),
    "swims": ("verb", ["swims", "can swim", "swims in water"]),
    "tunnels": ("verb", ["tunnels", "digs tunnels", "burrows underground"]),
    "walks": ("verb", ["walks", "moves by walking", "walks on land"]),
    "fast": ("adj", ["fast", "fast-moving", "quick"]),
    "slow": ("adj", ["slow", "slow-moving", "sluggish"]),
    "strong": ("adj", ["strong", "powerful", "muscular"]),
    "weak": ("adj", ["weak", "frail", "feeble"]),
    "muscle": ("adj", ["muscular", "heavily muscled", "well-muscled"]),
    "bipedal": ("adj", ["bipedal", "two-legged", "walking on two legs"]),
    "quadrapedal": ("adj", ["quadrupedal", "four-legged", "walking on four legs"]),
    "active": ("adj", ["active", "energetic", "lively"]),
    "inactive": ("adj", ["inactive", "sedentary", "still"]),
    "nocturnal": ("adj", ["nocturnal", "active at night", "night-active"]),
    "hibernate": ("verb", ["hibernates", "sleeps through winter", "enters hibernation"]),
    "agility": ("adj", ["agile", "nimble", "quick and agile"]),
    # diet (diet)
    "fish": ("diet", ["fish", "fish and seafood", "small fish"]),
    "meat": ("diet", ["meat", "flesh", "other animals"]),
    "plankton": ("diet", ["plankton", "tiny plankton", "krill and plankton"]),
    "vegetation": ("diet", ["vegetation", "plants", "leaves and plants"]),
    "insects": ("diet", ["insects", "bugs", "small insects"]),
    # foraging strategy (adjective / noun-ish, treated as adj role)
    "forager": ("adj", ["a foraging", "food-foraging", "forage-feeding"]),
    "grazer": ("adj", ["a grazing", "grass-grazing", "pasture-grazing"]),
    "hunter": ("adj", ["a hunting", "predatory", "prey-hunting"]),
    "scavenger": ("adj", ["a scavenging", "carrion-eating", "scavenger"]),
    "skimmer": ("adj", ["a skimming", "surface-skimming", "water-skimming"]),
    "stalker": ("adj", ["a stalking", "prey-stalking", "ambush-hunting"]),
    # biogeography (adjective)
    "newworld": ("adj", ["a New World", "native to the Americas", "found in the Americas"]),
    "oldworld": ("adj", ["an Old World", "native to Africa or Asia", "found in Africa or Asia"]),
    # habitat (habitat)
    "arctic": ("habitat", ["arctic", "frozen arctic", "icy arctic"]),
    "coastal": ("habitat", ["coast", "coastal shore", "seashore"]),
    "desert": ("habitat", ["desert", "dry desert", "sandy desert"]),
    "bush": ("habitat", ["bush", "bushland", "shrubland"]),
    "plains": ("habitat", ["plains", "open plains", "grassy plains"]),
    "forest": ("habitat", ["forest", "dense forest", "woodland"]),
    "fields": ("habitat", ["fields", "open fields", "grassy fields"]),
    "jungle": ("habitat", ["jungle", "tropical jungle", "dense jungle"]),
    "mountains": ("habitat", ["mountains", "rocky mountains", "high mountains"]),
    "ocean": ("habitat", ["ocean", "open ocean", "deep sea"]),
    "ground": ("habitat", ["ground", "bare ground", "open ground"]),
    "water": ("habitat", ["water", "rivers and lakes", "bodies of water"]),
    "tree": ("habitat", ["trees", "treetops", "tree canopy"]),
    "cave": ("habitat", ["cave", "dark cave", "underground cave"]),
    # temperament / social (adjective)
    "fierce": ("adj", ["fierce", "aggressive", "ferocious"]),
    "timid": ("adj", ["timid", "shy", "skittish"]),
    "smart": ("adj", ["smart", "intelligent", "clever"]),
    "group": ("adj", ["group-living", "living in groups", "social and group-dwelling"]),
    "solitary": ("adj", ["solitary", "living alone", "solitary-living"]),
    "nestspot": ("has", ["a nest", "a nesting spot", "a nest den"]),
    "domestic": ("adj", ["domestic", "domesticated", "a tame domestic"]),
}


def build_prompts_for_attribute(name: str) -> list[str]:
    # Cross the predicate's frame set × phrase variants → de-duplicated prompt list.
    group, phrases = ATTR_SPEC[name]
    frames = FRAMES[group]
    prompts, seen = [], set()
    for phrase in phrases:
        for frame in frames:
            text = frame.format(p=phrase)
            if text not in seen:
                seen.add(text)
                prompts.append(text)
    return prompts


def _verify_coverage():
    # Tripwire: ATTR_SPEC covers exactly the 85 predicates — no drift.
    missing = [n for n in ATTRIBUTE_NAMES if n not in ATTR_SPEC]
    extra = [n for n in ATTR_SPEC if n not in ATTRIBUTE_NAMES]
    assert not missing, f"ATTR_SPEC missing predicates: {missing}"
    assert not extra, f"ATTR_SPEC has unknown predicates: {extra}"


@torch.no_grad()
def extract_prompt_bank(force=False):
    # Encode every predicate's prompt bank with frozen CLIP → cache [85, n, 512].
    # Stacks padded to a common n (padding repeats the first prompt — SVD-safe).
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

    per_attr = [build_prompts_for_attribute(name) for name in ATTRIBUTE_NAMES]
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
    print("[OK] Prompt bank ready — CLAY's subspace SVD can now be built per attribute.")


def _verify(bank):
    assert bank.shape[0] == len(ATTRIBUTE_NAMES), bank.shape
    assert bank.shape[2] == FEATURE_DIM, bank.shape
    assert bank.shape[1] >= 2, f"Only {bank.shape[1]} prompt(s)/attr — CLAY needs n>1."
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
    # One-time build: python src/awa2/clip_prompts.py
    extract_prompt_bank()
