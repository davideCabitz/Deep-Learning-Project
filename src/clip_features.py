"""
Frozen CLIP image-feature table for the CelebA test split.

Turns every test image into a 512-d CLIP ViT-B/32 vector, L2-normalizes it, and
caches the [N, 512] tensor to disk. Built once, never modified — every retrieval
method (Tier-0/1/2a, trained Φ) ranks the corpus against this table.

One-time build:  python src/clip_features.py
"""

import torch
from torch.utils.data import DataLoader
from transformers import CLIPModel

from data_loader import load_celeba_dataset, load_attributes, _get_artifacts_dir


CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
FEATURE_DIM = 512


def _pick_device():
    """Prefer GPU; fall back to CPU (with a heads-up — CPU is slow for ~20k images)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("⚠️  No CUDA device found — running on CPU. Encoding ~20k images will be slow.")
    return torch.device("cpu")


@torch.no_grad()
def extract_image_features(batch_size=256, force=False):
    """
    Encode the whole CelebA test split with frozen CLIP and cache the feature table.

    Iterates the dataset in strict index order (shuffle=False) so row i of the output
    is celeba[i]. Images arrive ALREADY preprocessed by load_celeba_dataset() (CLIP
    resize + normalize), so they go straight into the vision encoder — we do NOT run
    them through CLIPProcessor again (that would normalize twice).

    Args:
        batch_size: images per forward pass.
        force: if True, rebuild even if the cache already exists.
    """
    out_path = _get_artifacts_dir() / "clip_image_features_test.pt"

    if out_path.exists() and not force:
        print(f"✓ Feature table already exists: {out_path}")
        return

    device = _pick_device()

    print(f"Loading frozen CLIP: {CLIP_MODEL_NAME}")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    model.eval()  # frozen — no training, no dropout

    dataset = load_celeba_dataset()
    print(f"  Test split: {len(dataset)} images")

    # shuffle=False is load-bearing: it keeps row i == celeba[i] (CONTRACT §0).
    # We only need the image (CelebA yields (image, attrs)); drop attrs in collate.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: torch.stack([img for img, _ in batch]),
    )

    feats = []
    total = len(dataset)
    done = 0
    for pixel_values in loader:
        pixel_values = pixel_values.to(device)
        # Vision tower -> pooled [B,768], then CLIP's visual_projection -> [B,512].
        # (This is exactly what get_image_features does internally; we call the two
        # steps explicitly because get_image_features's return type varies across
        # transformers versions — some return an output object, not a tensor.)
        # Inputs are already CLIP-normalized by load_celeba_dataset().
        vision_outputs = model.vision_model(pixel_values=pixel_values)
        batch_feats = model.visual_projection(vision_outputs.pooler_output)
        # L2-normalize rows so cosine similarity == dot product downstream (CONTRACT §6).
        batch_feats = torch.nn.functional.normalize(batch_feats, p=2, dim=1)
        feats.append(batch_feats.cpu())
        done += pixel_values.shape[0]
        print(f"  encoded {done}/{total}", end="\r")

    image_features = torch.cat(feats, dim=0).to(torch.float32)
    print()  # newline after the progress counter

    _verify(image_features)

    torch.save(image_features, out_path)
    print(f"  Saved: {out_path}")
    print(f"  Shape: {tuple(image_features.shape)}")
    print(f"  dtype: {image_features.dtype}")
    print("✓ CLIP feature table ready (do not modify)")


def _verify(image_features):
    """Fail loudly if the table is misaligned or not unit-normalized."""
    n_attrs = load_attributes().shape[0]
    n_feats = image_features.shape[0]
    assert n_feats == n_attrs, (
        f"Row count mismatch: {n_feats} features vs {n_attrs} attribute rows. "
        "Index alignment is broken (CONTRACT §0)."
    )
    assert image_features.shape[1] == FEATURE_DIM, (
        f"Expected {FEATURE_DIM}-d vectors, got {image_features.shape[1]}."
    )
    norms = image_features.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), (
        "Rows are not L2-normalized."
    )
    print(f"  ✓ verified: {n_feats} rows, {FEATURE_DIM}-d, unit-normalized, aligned with attributes")


def load_image_features():
    """
    Load the frozen CLIP feature table.

    Returns:
        torch.Tensor: [N, 512] float32, each row L2-normalized.
    """
    path = _get_artifacts_dir() / "clip_image_features_test.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature table not found. Run extract_image_features() first.\n"
            f"Expected: {path}"
        )
    return torch.load(path)


if __name__ == "__main__":
    # One-time setup: python src/clip_features.py
    extract_image_features()
