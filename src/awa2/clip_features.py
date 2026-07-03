"""
Frozen CLIP image-feature tables for the AwA2 splits.

Encodes every AwA2 image with frozen CLIP ViT-B/32 → L2-normalized [N_split, 512]
tables, one per split, row order == load_split(split) order. Built once (GPU job,
~37k images — run on the VM), never modified.

Resume-by-chunk: progress is checkpointed every CHUNK_SIZE images so an interrupted
run continues instead of restarting.

One-time build (VM):  python src/awa2/clip_features.py
"""

import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import CLIPModel

sys.path.insert(0, str(Path(__file__).parents[1]))

from awa2.data_loader import (_get_artifacts_dir, _get_awa2_root,
                              load_image_index, load_split)

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
FEATURE_DIM = 512
CHUNK_SIZE = 2048  # images per resume checkpoint

# Same preprocessing as the CelebA pipeline (src/data_loader.py): CLIP mean/std.
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])


def _pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("[!] No CUDA device found - running on CPU. Encoding ~37k images will be slow.")
    return torch.device("cpu")


class _Awa2Images(Dataset):
    # Yields CLIP-preprocessed tensors for a fixed list of global image indices.
    def __init__(self, global_indices: torch.Tensor):
        self.root = _get_awa2_root()
        paths = load_image_index()["paths"]
        self.paths = [paths[i] for i in global_indices.tolist()]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.root / self.paths[i]).convert("RGB")
        return TRANSFORM(img)


@torch.no_grad()
def extract_split(split: str, batch_size=256, force=False):
    # Encode one split with frozen CLIP → cache [N_split, 512] unit-norm table.
    # Chunk-checkpointed: partial results land in *.partial.pt and are resumed.
    out_path = _get_artifacts_dir() / f"clip_image_features_{split}.pt"
    partial_path = _get_artifacts_dir() / f"clip_image_features_{split}.partial.pt"

    if out_path.exists() and not force:
        print(f"[OK] Feature table already exists: {out_path}")
        return

    indices = load_split(split)
    dataset = _Awa2Images(indices)
    total = len(dataset)

    done_rows = 0
    chunks = []
    if partial_path.exists() and not force:
        prev = torch.load(partial_path)
        chunks = [prev]
        done_rows = prev.shape[0]
        print(f"  Resuming {split}: {done_rows}/{total} already encoded")

    device = _pick_device()
    print(f"Loading frozen CLIP: {CLIP_MODEL_NAME}")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    model.eval()

    remaining = torch.utils.data.Subset(dataset, range(done_rows, total))
    loader = DataLoader(remaining, batch_size=batch_size, shuffle=False, num_workers=0)

    buf, buf_rows = [], 0
    for pixel_values in loader:
        pixel_values = pixel_values.to(device)
        # Vision tower + visual_projection (version-robust, same as src/clip_features.py).
        vision_outputs = model.vision_model(pixel_values=pixel_values)
        feats = model.visual_projection(vision_outputs.pooler_output)
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        buf.append(feats.cpu().to(torch.float32))
        buf_rows += feats.shape[0]
        done_rows += feats.shape[0]
        print(f"  encoded {done_rows}/{total}", end="\r")

        if buf_rows >= CHUNK_SIZE:
            chunks.append(torch.cat(buf, dim=0))
            torch.save(torch.cat(chunks, dim=0), partial_path)
            buf, buf_rows = [], 0

    if buf:
        chunks.append(torch.cat(buf, dim=0))
    image_features = torch.cat(chunks, dim=0)
    print()

    _verify(image_features, total)
    torch.save(image_features, out_path)
    partial_path.unlink(missing_ok=True)
    print(f"  Saved: {out_path}  shape={tuple(image_features.shape)}")
    print(f"[OK] {split} CLIP feature table ready (do not modify)")


def _verify(image_features, expected_rows):
    # Tripwire: row count, 512-d, unit-normalized.
    assert image_features.shape == (expected_rows, FEATURE_DIM), (
        f"Expected ({expected_rows}, {FEATURE_DIM}), got {tuple(image_features.shape)}."
    )
    norms = image_features.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), (
        "Rows are not L2-normalized."
    )
    print(f"  [OK] verified: {expected_rows} rows, {FEATURE_DIM}-d, unit-normalized")


if __name__ == "__main__":
    # One-time build (VM): python src/awa2/clip_features.py
    extract_split("test")
    extract_split("train")
