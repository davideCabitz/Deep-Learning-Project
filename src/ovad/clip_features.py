"""
Frozen CLIP image-feature table for OVAD instance CROPS.

Unlike CelebA/AwA2 (whole-image encoding), OVAD's retrieval unit is an object instance:
each row of the frozen table is the CLIP encoding of one bbox CROP (cut from its parent
COCO image), not the whole photo. Row order == load_instance_index() order (frozen
contract, src/ovad/data_loader.py).

One-time build:  python src/ovad/clip_features.py
"""

import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import CLIPModel

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / "Baselines"))
sys.path.insert(0, str(Path(__file__).parents[1] / "Training-Free"))
sys.path.insert(0, str(Path(__file__).parents[1] / "Trained"))

from ovad.data_loader import _get_artifacts_dir, _get_ovad_root, load_instance_index

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
FEATURE_DIM = 512

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])


def _pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("[!] No CUDA device found - running on CPU. Encoding ~9k crops will be slow-ish.")
    return torch.device("cpu")


class _OvadCrops(Dataset):
    # Yields CLIP-preprocessed crops for every row of the frozen instance index.
    def __init__(self):
        self.root = _get_ovad_root() / "val2017"
        self.index = load_instance_index()

    def __len__(self):
        return len(self.index["ann_ids"])

    def __getitem__(self, i):
        img = Image.open(self.root / self.index["file_names"][i]).convert("RGB")
        x, y, w, h = self.index["bboxes"][i]
        crop = img.crop((x, y, x + w, y + h))
        return TRANSFORM(crop)


@torch.no_grad()
def extract_features(batch_size=256, force=False):
    # Encode every instance crop with frozen CLIP -> cache [N, 512] unit-norm table.
    out_path = _get_artifacts_dir() / "clip_image_features.pt"
    if out_path.exists() and not force:
        print(f"[OK] Feature table already exists: {out_path}")
        return

    dataset = _OvadCrops()
    total = len(dataset)
    print(f"  Instance crops: {total}")

    device = _pick_device()
    print(f"Loading frozen CLIP: {CLIP_MODEL_NAME}")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    model.eval()

    # shuffle=False is load-bearing: keeps row i == instance i (frozen index contract).
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    feats, done = [], 0
    for pixel_values in loader:
        pixel_values = pixel_values.to(device)
        vision_outputs = model.vision_model(pixel_values=pixel_values)
        batch_feats = model.visual_projection(vision_outputs.pooler_output)
        batch_feats = torch.nn.functional.normalize(batch_feats, p=2, dim=1)
        feats.append(batch_feats.cpu().to(torch.float32))
        done += pixel_values.shape[0]
        print(f"  encoded {done}/{total}", end="\r")

    image_features = torch.cat(feats, dim=0)
    print()

    _verify(image_features, total)
    torch.save(image_features, out_path)
    print(f"  Saved: {out_path}  shape={tuple(image_features.shape)}")
    print("[OK] CLIP feature table ready (do not modify)")


def _verify(image_features, expected_rows):
    assert image_features.shape == (expected_rows, FEATURE_DIM), (
        f"Expected ({expected_rows}, {FEATURE_DIM}), got {tuple(image_features.shape)}."
    )
    norms = image_features.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), "Rows are not L2-normalized."
    print(f"  [OK] verified: {expected_rows} rows, {FEATURE_DIM}-d, unit-normalized")


if __name__ == "__main__":
    # One-time build: python src/ovad/clip_features.py
    extract_features()
