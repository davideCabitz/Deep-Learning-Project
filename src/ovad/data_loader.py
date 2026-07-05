"""
Frozen database loader for OVAD (Bravo et al., CVPR 2023) — object-instance retrieval.

Unlike CelebA/AwA2, the retrieval unit here is an OBJECT INSTANCE (a bounding box inside
a COCO val2017 image), not a whole image — OVAD's ov ad2000.json annotates 14,300
instances across 2,000 images with 117 tri-state attributes each. This module freezes
three contracts:
  1. Instance index — every annotation with min(bbox side) >= MIN_SIDE_PX becomes one
     retrieval-corpus row, in annotation-id order; row i is THE index every artifact
     aligns to (mirrors AwA2's image_index.json contract).
  2. Attribute label matrix — [N, 117] int8 in {1, 0, -1} (1=positive, 0=negative,
     -1=unlabeled). VERIFIED against OVAD's own ovad_evaluator.py (line ~976:
     `gt[gt == -1] = 2` before treating 2 as "unlabeled") — do not flip this convention.
  3. Synthetic train/test split — OVAD ships test-only (2,000 images), but trained/mined
     methods (GDE/LDE, DualScore, tier5) need a train pool, so we freeze an 80/20 split
     BY image_id (seed 42): same-image instances never straddle the split. Loaders take
     an optional split arg; None keeps the full-corpus view for corpus-level artifacts.
Built once, never modified — every method reads these as-is.

One-time setup:  python src/ovad/data_loader.py
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))  # src/ on path for peer imports
sys.path.insert(0, str(Path(__file__).parents[1] / "Baselines"))
sys.path.insert(0, str(Path(__file__).parents[1] / "Training-Free"))
sys.path.insert(0, str(Path(__file__).parents[1] / "Trained"))


MIN_SIDE_PX = 32     # CLIP ViT-B/32 patch size — smaller crops are pure upsampling noise
N_ATTRS = 117
N_CATEGORIES = 80
SPLIT_SEED = 42
TRAIN_FRACTION = 0.8   # generate_gt falls back to 0.7 if the test-split GT loses viability


def _get_project_root():
    return Path(__file__).parents[2]


def _get_artifacts_dir():
    artifacts = _get_project_root() / 'artifacts' / 'ovad'
    artifacts.mkdir(parents=True, exist_ok=True)
    return artifacts


def _get_ovad_root():
    root = _get_project_root() / 'OVAD_Dataset'
    ann_path = root / 'val2017' / 'ovad2000.json'
    if not ann_path.exists():
        raise FileNotFoundError(f"OVAD annotations not found at {ann_path}.")
    return root


def load_raw_annotation() -> dict:
    # The single source of truth: OVAD_Dataset/val2017/ovad2000.json, loaded once per call.
    with open(_get_ovad_root() / 'val2017' / 'ovad2000.json', 'r') as f:
        return json.load(f)


def load_attribute_names() -> list[str]:
    # 117 names, id-ordered, e.g. "clothes color:blue" (type:phrase1/phrase2/...).
    raw = load_raw_annotation()
    by_id = {a['id']: a['name'] for a in raw['attributes']}
    return [by_id[i] for i in range(len(by_id))]


ATTRIBUTE_NAMES = load_attribute_names()
ATTR_TO_IDX = {name: i for i, name in enumerate(ATTRIBUTE_NAMES)}


def load_attribute_meta() -> list[dict]:
    # Full attribute records (id, name, type, parent_type, is_has_att, freq_set), id-ordered.
    raw = load_raw_annotation()
    by_id = {a['id']: a for a in raw['attributes']}
    return [by_id[i] for i in range(len(by_id))]


def load_category_names() -> dict[int, str]:
    # {category_id: name} — COCO category ids are not contiguous, so a dict, not a list.
    raw = load_raw_annotation()
    return {c['id']: c['name'] for c in raw['categories']}


def build_instance_index(force: bool = False) -> None:
    # Freeze the global instance index: annotations sorted by id, filtered to
    # min(bbox side) >= MIN_SIDE_PX. Saved manifest is the index->(image, bbox) contract:
    # row i of every artifact == instance_index["annotations"][i]. `force` rebuilds even
    # if cached.
    manifest_path = _get_artifacts_dir() / 'instance_index.json'
    if manifest_path.exists() and not force:
        print(f"[OK] Instance index already exists: {manifest_path}")
        return

    raw = load_raw_annotation()
    images_by_id = {im['id']: im['file_name'] for im in raw['images']}

    kept = []
    n_total = len(raw['annotations'])
    for ann in sorted(raw['annotations'], key=lambda a: a['id']):
        x, y, w, h = ann['bbox']
        if min(w, h) < MIN_SIDE_PX:
            continue
        kept.append({
            'ann_id': ann['id'],
            'image_id': ann['image_id'],
            'file_name': images_by_id[ann['image_id']],
            'category_id': ann['category_id'],
            'bbox': ann['bbox'],
            'att_vec': ann['att_vec'],
        })

    manifest = {
        'ann_ids': [e['ann_id'] for e in kept],
        'image_ids': [e['image_id'] for e in kept],
        'file_names': [e['file_name'] for e in kept],
        'category_ids': [e['category_id'] for e in kept],
        'bboxes': [e['bbox'] for e in kept],
        'att_vecs': [e['att_vec'] for e in kept],
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f)

    n_images = len(set(manifest['image_ids']))
    print(f"[OK] Instance index frozen: {len(kept)}/{n_total} instances survive "
          f"min-side>={MIN_SIDE_PX}px filter, spanning {n_images} images")
    print(f"  Saved: {manifest_path}")


def load_instance_index() -> dict:
    # {"ann_ids", "image_ids", "file_names", "category_ids", "bboxes", "att_vecs"} —
    # frozen contract; row i == instance i for every downstream artifact.
    path = _get_artifacts_dir() / 'instance_index.json'
    if not path.exists():
        raise FileNotFoundError(f"Instance index not found. Run build_instance_index() first.\n"
                                f"Expected: {path}")
    with open(path, 'r') as f:
        return json.load(f)


def build_split(force: bool = False, train_fraction: float = TRAIN_FRACTION) -> None:
    # Synthetic train/test split BY image_id (seeded shuffle, deterministic forever),
    # propagated to instance rows — instances of one image never straddle the split
    # (bbox crops of the same photo share background/context; splitting per-instance
    # would leak near-duplicates from train into the eval corpus).
    train_path = _get_artifacts_dir() / 'split_train_idx.pt'
    test_path = _get_artifacts_dir() / 'split_test_idx.pt'
    meta_path = _get_artifacts_dir() / 'split_meta.json'
    if train_path.exists() and test_path.exists() and not force:
        print(f"[OK] Split already exists: {train_path}")
        return

    image_ids = torch.tensor(load_instance_index()['image_ids'], dtype=torch.long)
    unique_ids = image_ids.unique(sorted=True)
    gen = torch.Generator().manual_seed(SPLIT_SEED)
    perm = unique_ids[torch.randperm(len(unique_ids), generator=gen)]
    n_train = int(round(len(unique_ids) * train_fraction))
    train_images = perm[:n_train]

    is_train = torch.isin(image_ids, train_images)
    train_idx = is_train.nonzero(as_tuple=True)[0]
    test_idx = (~is_train).nonzero(as_tuple=True)[0]
    assert len(train_idx) + len(test_idx) == len(image_ids)
    assert not set(train_idx.tolist()) & set(test_idx.tolist())

    torch.save(train_idx, train_path)
    torch.save(test_idx, test_path)
    with open(meta_path, 'w') as f:
        json.dump({'train_fraction': train_fraction, 'seed': SPLIT_SEED}, f)
    print(f"[OK] Split frozen (seed={SPLIT_SEED}, frac={train_fraction}): "
          f"{len(train_idx)} train / {len(test_idx)} test instances "
          f"({n_train}/{len(unique_ids) - n_train} images)")


def load_split(split: str) -> torch.Tensor:
    # Instance row indices for 'train' or 'test' — sorted, disjoint, frozen.
    if split not in ('train', 'test'):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    path = _get_artifacts_dir() / f'split_{split}_idx.pt'
    if not path.exists():
        raise FileNotFoundError(f"Split not found. Run build_split() first.\nExpected: {path}")
    return torch.load(path)


def load_split_meta() -> dict:
    # {'train_fraction', 'seed'} of the frozen split — reported alongside GT/benchmark runs.
    with open(_get_artifacts_dir() / 'split_meta.json', 'r') as f:
        return json.load(f)


def _maybe_slice(tensor: torch.Tensor, split: str | None) -> torch.Tensor:
    return tensor if split is None else tensor[load_split(split)]


def load_attributes(split: str | None = None) -> torch.Tensor:
    # [N, 117] int8 in {1, 0, -1} (1=positive, 0=negative, -1=unlabeled). Row order ==
    # load_instance_index() order; split='train'/'test' row-slices by load_split(split),
    # None = full corpus (corpus-level artifacts keep their original view).
    index = load_instance_index()
    return _maybe_slice(torch.tensor(index['att_vecs'], dtype=torch.int8), split)


def load_category_labels(split: str | None = None) -> torch.Tensor:
    # [N] long — COCO category id per instance, aligned with load_attributes(split).
    index = load_instance_index()
    return _maybe_slice(torch.tensor(index['category_ids'], dtype=torch.long), split)


def load_image_features(split: str | None = None) -> torch.Tensor:
    # [N, 512] float32, L2-normalized; one canonical full-corpus tensor on disk (row
    # order == load_instance_index() order), sliced on load for split views.
    path = _get_artifacts_dir() / 'clip_image_features.pt'
    if not path.exists():
        raise FileNotFoundError(
            f"CLIP features not found. Run src/ovad/clip_features.py first.\nExpected: {path}"
        )
    return _maybe_slice(torch.load(path), split)


if __name__ == '__main__':
    # One-time setup: python src/ovad/data_loader.py
    build_instance_index()
    build_split()
    attrs = load_attributes()
    labeled = (attrs != -1)
    print(f"  Attribute count: {len(ATTRIBUTE_NAMES)}")
    print(f"  Attributes shape: {tuple(attrs.shape)}")
    print(f"  Mean labeled/instance: {labeled.sum(dim=1).float().mean():.1f}")
    print(f"  Mean positive/instance: {(attrs == 1).sum(dim=1).float().mean():.1f}")
