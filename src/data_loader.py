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
    """Get the project root directory."""
    return Path(__file__).parent.parent


def _get_artifacts_dir():
    """Directory holding cached artifacts (frozen DB, CLIP feature tensors)."""
    artifacts = _get_project_root() / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    return artifacts


def _get_celeba_root():
    """Find CelebA dataset location (check multiple paths)."""
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
    """
    Build and save frozen attribute tensor.

    Reads from CelebA's list_attr_celeba.txt, filters to test split,
    converts -1/+1 → 0.0/1.0, saves as float32 tensor.

    Args:
        force: if True, rebuild even if file exists
    """
    attr_cache_path = _get_artifacts_dir() / 'celeba_attributes_test.pt'

    if attr_cache_path.exists() and not force:
        print(f"[OK] Frozen DB already exists: {attr_cache_path}")
        return

    print("Building frozen attribute tensor...")

    # Load CelebA to get test split filenames
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

    # Read attributes from file and filter to test split
    attributes = []
    attr_file = celeba_root / 'celeba' / 'list_attr_celeba.txt'

    with open(attr_file, 'r') as f:
        f.readline()  # Skip total count
        f.readline()  # Skip header

        for line in f:
            parts = line.strip().split()
            filename = parts[0]

            if filename in test_filenames:
                attrs = [int(x) for x in parts[1:]]
                attributes.append(attrs)

    # Convert to float32 and map -1 → 0.0, +1 → 1.0
    celeba_attrs = torch.tensor(attributes, dtype=torch.float32)
    celeba_attrs = (celeba_attrs + 1) / 2

    # Save
    torch.save(celeba_attrs, attr_cache_path)
    print(f"  Saved: {attr_cache_path}")
    print(f"  Shape: {celeba_attrs.shape}")
    print(f"  dtype: {celeba_attrs.dtype}")
    print(f"  Range: [{celeba_attrs.min():.1f}, {celeba_attrs.max():.1f}]")
    print("[OK] Frozen DB ready (do not modify)")


def load_attributes():
    """
    Load the frozen attribute tensor.

    Returns:
        torch.Tensor: [N, 40] float32 tensor, values in {0.0, 1.0}
    """
    attr_cache_path = _get_artifacts_dir() / 'celeba_attributes_test.pt'

    if not attr_cache_path.exists():
        raise FileNotFoundError(
            f"Attributes not found. Run setup_frozen_db() first.\n"
            f"Expected: {attr_cache_path}"
        )

    return torch.load(attr_cache_path)


def load_celeba_dataset():
    """
    Load CelebA test split with CLIP preprocessing.

    Returns:
        torchvision.datasets.CelebA: test split dataset
    """
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
