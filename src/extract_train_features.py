"""
Build the two train-split artifacts that fusion_dgp.py needs:

  artifacts/celeba_attributes_train.pt   — [N_train, 40]  float32 {0,1}
  artifacts/clip_image_features_train.pt — [N_train, 512] float32 L2-normalized

Mirrors notebooks/colab_extract_train_features.ipynb exactly (same model,
same normalization, same row ordering) so train and test vectors are
in the same CLIP image cone.

Run: python src/extract_train_features.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPModel

from data_loader import _get_artifacts_dir, _get_celeba_root


CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
FEATURE_DIM     = 512
BATCH_SIZE      = 256


_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275,  0.40821073],
        std =[0.26862954, 0.26130258, 0.27577711],
    ),
])


class _TrainDataset(Dataset):
    def __init__(self, img_dir: Path, filenames: list[str]):
        self.img_dir   = img_dir
        self.filenames = filenames

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.img_dir / self.filenames[idx]).convert("RGB")
        return _TRANSFORM(img)


def _build_attributes(celeba_dir: Path, artifacts: Path) -> torch.Tensor:
    # [N_train, 40] float32 {0,1} attribute mask for the train split.
    out_path = artifacts / "celeba_attributes_train.pt"
    if out_path.exists():
        print(f"[OK] attributes already exist: {out_path}")
        return torch.load(out_path, weights_only=True)

    partition_file = celeba_dir / "list_eval_partition.txt"
    attr_file      = celeba_dir / "list_attr_celeba.txt"
    assert partition_file.exists(), f"Missing: {partition_file}"
    assert attr_file.exists(),      f"Missing: {attr_file}"

    partition: dict[str, int] = {}
    with open(partition_file) as f:
        for line in f:
            parts = line.strip().split()
            partition[parts[0]] = int(parts[1])

    train_total = sum(v == 0 for v in partition.values())
    print(f"Partition: train={train_total}, "
          f"val={sum(v==1 for v in partition.values())}, "
          f"test={sum(v==2 for v in partition.values())}")

    rows: list[list[int]] = []
    filenames: list[str]  = []
    with open(attr_file) as f:
        f.readline()
        f.readline()
        for line in f:
            parts = line.strip().split()
            if partition.get(parts[0]) == 0:
                rows.append([int(x) for x in parts[1:]])
                filenames.append(parts[0])

    attr_tensor = torch.tensor(rows, dtype=torch.float32)
    attr_tensor = (attr_tensor + 1) / 2

    torch.save(attr_tensor, out_path)
    print(f"Saved: {out_path}  shape={tuple(attr_tensor.shape)}")

    # cache filenames for the feature builder
    _build_attributes._filenames = filenames  # type: ignore[attr-defined]
    return attr_tensor


def _build_features(
    celeba_dir: Path,
    artifacts: Path,
    filenames: list[str],
    n_attrs: int,
    device: torch.device,
) -> torch.Tensor:
    # [N_train, 512] float32 L2-normalized CLIP vectors for the train split.
    out_path = artifacts / "clip_image_features_train.pt"
    if out_path.exists():
        print(f"[OK] features already exist: {out_path}")
        return torch.load(out_path, weights_only=True)

    img_dir = celeba_dir / "img_align_celeba"
    assert img_dir.exists(), f"Missing image dir: {img_dir}"

    print(f"Loading frozen CLIP: {CLIP_MODEL_NAME}")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    model.eval()

    dataset = _TrainDataset(img_dir, filenames)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    total   = len(dataset)
    print(f"Train images to encode: {total}")

    feats: list[torch.Tensor] = []
    done = 0
    with torch.no_grad():
        for pixel_values in loader:
            vision_out  = model.vision_model(pixel_values=pixel_values.to(device))
            batch_feats = model.visual_projection(vision_out.pooler_output)
            batch_feats = F.normalize(batch_feats, p=2, dim=1)
            feats.append(batch_feats.cpu())
            done += pixel_values.shape[0]
            print(f"  encoded {done}/{total}", end="\r", flush=True)
    print()

    image_features = torch.cat(feats, dim=0).to(torch.float32)

    assert image_features.shape[0] == n_attrs, (
        f"Row mismatch: {image_features.shape[0]} feature rows vs {n_attrs} attribute rows."
    )
    assert image_features.shape[1] == FEATURE_DIM
    norms = image_features.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), "Rows not unit-normalized."
    print(f"[OK] verified: {image_features.shape[0]} rows, {FEATURE_DIM}-d, unit-normalized")

    torch.save(image_features, out_path)
    print(f"Saved: {out_path}  shape={tuple(image_features.shape)}")
    return image_features


def main() -> None:
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    artifacts  = _get_artifacts_dir()
    celeba_dir = _get_celeba_root() / "celeba"

    attr_tensor = _build_attributes(celeba_dir, artifacts)
    filenames   = getattr(_build_attributes, "_filenames", None)

    if filenames is None:
        # attributes were cached; rebuild the filename list from the partition file
        partition_file = celeba_dir / "list_eval_partition.txt"
        attr_file      = celeba_dir / "list_attr_celeba.txt"
        partition: dict[str, int] = {}
        with open(partition_file) as f:
            for line in f:
                parts = line.strip().split()
                partition[parts[0]] = int(parts[1])
        filenames = []
        with open(attr_file) as f:
            f.readline(); f.readline()
            for line in f:
                fname = line.strip().split()[0]
                if partition.get(fname) == 0:
                    filenames.append(fname)

    _build_features(celeba_dir, artifacts, filenames, attr_tensor.shape[0], device)
    print("\n[DONE] Both train-split artifacts ready in artifacts/")


if __name__ == "__main__":
    main()
