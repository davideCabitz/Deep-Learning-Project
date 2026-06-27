"""
Frozen database loader for CelebA dataset and attributes.
Builds once, never modifies. Both data and model halves import from here.
"""

import torch
from pathlib import Path
from torchvision import datasets, transforms


# Master list of 40 attribute names (from CONTRACT.md section 1)
ATTRIBUTE_NAMES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
    "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
    "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
    "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young",
]
ATTR_TO_IDX = {name: i for i, name in enumerate(ATTRIBUTE_NAMES)}


def _get_project_root():
    return Path(__file__).parent.parent


def _get_artifacts_dir():
    # Cached-artifacts dir (frozen DB + CLIP feature tensors); created on demand.
    artifacts = _get_project_root() / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    return artifacts


def _get_celeba_root():
    # Locate the CelebA dataset, probing the known install paths in order.
    project_root = _get_project_root()

    celeba_paths = [
        project_root / 'celeba',
        Path.home() / 'Downloads' / 'celeba',
    ]

    for path in celeba_paths:
        celeba_subdir = path / 'celeba'
        if (celeba_subdir / 'img_align_celeba').exists() and (celeba_subdir / 'list_attr_celeba.txt').exists():
            return path

    raise FileNotFoundError(f"CelebA dataset not found in any expected location:\n" +
                          "\n".join(str(p / 'celeba') for p in celeba_paths))


def setup_frozen_db(force=False):
    # Build the frozen [N, 40] attribute tensor for the CelebA test split.
    # Reads list_attr_celeba.txt, keeps test-split rows, maps -1/+1 → 0.0/1.0,
    # caches as float32. Built once, never modified — every method reads it as-is.
    # `force` rebuilds even if the cache exists.
    attr_cache_path = _get_artifacts_dir() / 'celeba_attributes_test.pt'

    if attr_cache_path.exists() and not force:
        print(f"[OK] Frozen DB already exists: {attr_cache_path}")
        return

    print("Building frozen attribute tensor...")

    # Need the test-split filenames so we can filter the raw attribute file to them.
    celeba_root = _get_celeba_root()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711])
    ])

    celeba = datasets.CelebA(
        root=celeba_root,
        split='test',
        download=False,
        transform=transform
    )
    test_filenames = set(celeba.filename)
    print(f"  Test split: {len(test_filenames)} images")

    attributes = []
    attr_file = celeba_root / 'celeba' / 'list_attr_celeba.txt'

    with open(attr_file, 'r') as f:
        f.readline()  # line 1: total count
        f.readline()  # line 2: column header

        for line in f:
            parts = line.strip().split()
            filename = parts[0]

            if filename in test_filenames:
                attrs = [int(x) for x in parts[1:]]
                attributes.append(attrs)

    # Map raw -1/+1 labels onto a clean 0/1 mask: (x+1)/2.
    celeba_attrs = torch.tensor(attributes, dtype=torch.float32)
    celeba_attrs = (celeba_attrs + 1) / 2

    torch.save(celeba_attrs, attr_cache_path)
    print(f"  Saved: {attr_cache_path}")
    print(f"  Shape: {celeba_attrs.shape}")
    print(f"  dtype: {celeba_attrs.dtype}")
    print(f"  Range: [{celeba_attrs.min():.1f}, {celeba_attrs.max():.1f}]")
    print("[OK] Frozen DB ready (do not modify)")


def load_attributes():
    # Load the frozen [N, 40] float32 attribute mask (values in {0.0, 1.0}).
    # Raises if absent — run setup_frozen_db() first.
    attr_cache_path = _get_artifacts_dir() / 'celeba_attributes_test.pt'

    if not attr_cache_path.exists():
        raise FileNotFoundError(
            f"Attributes not found. Run setup_frozen_db() first.\n"
            f"Expected: {attr_cache_path}"
        )

    return torch.load(attr_cache_path)


def load_image_features():
    # Load the cached [N, 512] CLIP image-feature table (CONTRACT §6).
    # Built once in Colab, dropped into artifacts/; row i = celeba[i], L2-normalized.
    path = _get_artifacts_dir() / 'clip_image_features_test.pt'

    if not path.exists():
        raise FileNotFoundError(
            f"CLIP image features not found. Build them in Colab and place at:\n"
            f"  {path}\n"
            f"Expected: [N, 512] float32, L2-normalized (CONTRACT.md section 6)."
        )

    return torch.load(path)


def load_text_features():
    # Load the cached attribute text table if present, else None.
    # Optional: encoding 40 short prompts is cheap, so callers may recompute instead.
    # When cached, it's {attr_name: [512] L2-normalized tensor}.
    path = _get_artifacts_dir() / 'clip_text_features_test.pt'

    if not path.exists():
        return None

    return torch.load(path)


def load_celeba_dataset():
    # CelebA test split with CLIP preprocessing (resize 224 + CLIP-mean/std normalize).
    celeba_root = _get_celeba_root()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711])
    ])

    return datasets.CelebA(
        root=celeba_root,
        split='test',
        download=False,
        transform=transform
    )


if __name__ == '__main__':
    # One-time setup: python src/data_loader.py
    setup_frozen_db()
