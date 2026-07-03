"""
Frozen database loader for the AwA2 dataset (transfer study, spec §2).

AwA2 attributes are CLASS-level: every image of a class shares the class's 85-bit
row from predicate-matrix-binary.txt. This module freezes three contracts:
  1. Image index — classes in classes.txt order, filenames sorted within class;
     global index 0..N-1 is THE index every artifact row aligns to.
  2. Per-image attributes — [N, 85] float32 {0,1} via class-row lookup.
  3. Split — stratified-by-class 80/20 train/test, fixed seed. Random per-image
     (not the ZSL 40/10 class split): retrieval needs every attribute pattern
     present in the test corpus.
Built once, never modified — every method reads these as-is.

One-time setup:  python src/awa2/data_loader.py
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))  # src/ on path for peer imports


SPLIT_SEED = 42
TRAIN_FRACTION = 0.8
N_CLASSES = 50
N_ATTRS = 85


def _get_project_root():
    return Path(__file__).parents[2]


def _get_artifacts_dir():
    # Cached-artifacts dir for AwA2 (index manifest, splits, tensors); created on demand.
    artifacts = _get_project_root() / 'artifacts' / 'awa2'
    artifacts.mkdir(parents=True, exist_ok=True)
    return artifacts


def _get_awa2_root():
    root = _get_project_root() / 'Animals_with_Attributes2'
    if not (root / 'classes.txt').exists() or not (root / 'JPEGImages').exists():
        raise FileNotFoundError(
            f"AwA2 dataset not found at {root} (expected classes.txt + JPEGImages/)."
        )
    return root


def _read_indexed_list(path: Path) -> list[str]:
    # Parse AwA2's "  <idx>\t<name>" list files, order-preserving.
    names = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.split()
            if parts:
                names.append(parts[-1])
    return names


# Metadata (class names, predicate names, class→attribute matrix) is cached as small
# artifacts so downstream tiers run on a machine that has ONLY the extracted features and
# not the 13GB raw dataset (e.g. the training VM). The raw dataset is the source of truth
# when present; the cache is written on first read and used as the fallback otherwise.
def _load_cached_or_raw(cache_name: str, raw_reader):
    cache = _get_artifacts_dir() / cache_name
    if cache.exists():
        with open(cache, 'r') as f:
            return json.load(f)
    root = _get_project_root() / 'Animals_with_Attributes2'
    if not (root / 'classes.txt').exists():
        raise FileNotFoundError(
            f"AwA2 metadata cache missing ({cache}) and raw dataset absent ({root}).\n"
            "Run src/awa2/data_loader.py once on the machine that HAS the dataset to build "
            "the caches, then copy artifacts/awa2/*.json to this machine.")
    value = raw_reader(root)
    with open(cache, 'w') as f:
        json.dump(value, f)
    return value


def load_class_names() -> list[str]:
    return _load_cached_or_raw(
        'class_names.json', lambda root: _read_indexed_list(root / 'classes.txt'))


def load_predicate_names() -> list[str]:
    return _load_cached_or_raw(
        'predicate_names.json', lambda root: _read_indexed_list(root / 'predicates.txt'))


ATTRIBUTE_NAMES = load_predicate_names()
ATTR_TO_IDX = {name: i for i, name in enumerate(ATTRIBUTE_NAMES)}


def load_class_attribute_matrix() -> torch.Tensor:
    # [50, 85] float32 {0,1}; row order == classes.txt, column order == predicates.txt.
    def _read(root):
        path = root / 'predicate-matrix-binary.txt'
        return [[float(x) for x in line.split()] for line in open(path) if line.strip()]

    matrix = torch.tensor(_load_cached_or_raw('class_attribute_matrix.json', _read),
                          dtype=torch.float32)
    if matrix.shape != (N_CLASSES, N_ATTRS):
        raise ValueError(f"class-attribute matrix has shape {tuple(matrix.shape)}, "
                         f"expected ({N_CLASSES}, {N_ATTRS}).")
    return matrix


def build_image_index(force=False):
    # Freeze the global image index: classes in classes.txt order, sorted filenames
    # within each class. Saved manifest is the index→file contract (CONTRACT §0 analog):
    # row i of every artifact == image_index[i]. `force` rebuilds even if cached.
    manifest_path = _get_artifacts_dir() / 'image_index.json'
    if manifest_path.exists() and not force:
        print(f"[OK] Image index already exists: {manifest_path}")
        return

    root = _get_awa2_root()
    classes = load_class_names()
    entries = []  # (relative_path, class_idx)
    for c_idx, c_name in enumerate(classes):
        class_dir = root / 'JPEGImages' / c_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        files = sorted(p.name for p in class_dir.glob('*.jpg'))
        if not files:
            raise ValueError(f"Class folder has no jpg images: {class_dir}")
        entries.extend((f"JPEGImages/{c_name}/{fname}", c_idx) for fname in files)

    manifest = {
        "paths": [e[0] for e in entries],
        "class_idx": [e[1] for e in entries],
        "classes": classes,
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f)
    print(f"[OK] Image index frozen: {len(entries)} images, {len(classes)} classes")
    print(f"  Saved: {manifest_path}")


def load_image_index() -> dict:
    # {"paths": [rel_path], "class_idx": [int], "classes": [name]} — frozen contract.
    path = _get_artifacts_dir() / 'image_index.json'
    if not path.exists():
        raise FileNotFoundError(f"Image index not found. Run build_image_index() first.\n"
                                f"Expected: {path}")
    with open(path, 'r') as f:
        return json.load(f)


def build_split(force=False):
    # Stratified 80/20 per-image split, fixed generator seed — deterministic forever.
    # Per-class shuffle keeps every class (hence every attribute pattern) in both splits.
    train_path = _get_artifacts_dir() / 'split_train_idx.pt'
    test_path = _get_artifacts_dir() / 'split_test_idx.pt'
    if train_path.exists() and test_path.exists() and not force:
        print(f"[OK] Split already exists: {train_path}")
        return

    index = load_image_index()
    class_idx = torch.tensor(index['class_idx'], dtype=torch.long)
    gen = torch.Generator().manual_seed(SPLIT_SEED)

    train_parts, test_parts = [], []
    for c in range(len(index['classes'])):
        members = (class_idx == c).nonzero(as_tuple=True)[0]
        perm = members[torch.randperm(len(members), generator=gen)]
        n_train = int(round(len(members) * TRAIN_FRACTION))
        train_parts.append(perm[:n_train])
        test_parts.append(perm[n_train:])

    train_idx = torch.cat(train_parts).sort().values
    test_idx = torch.cat(test_parts).sort().values
    assert len(train_idx) + len(test_idx) == len(class_idx)
    assert not set(train_idx.tolist()) & set(test_idx.tolist())

    torch.save(train_idx, train_path)
    torch.save(test_idx, test_path)
    print(f"[OK] Split frozen (seed={SPLIT_SEED}): "
          f"{len(train_idx)} train / {len(test_idx)} test")


def load_split(split: str) -> torch.Tensor:
    # Global image indices for 'train' or 'test' — sorted, disjoint, frozen.
    if split not in ('train', 'test'):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    path = _get_artifacts_dir() / f'split_{split}_idx.pt'
    if not path.exists():
        raise FileNotFoundError(f"Split not found. Run build_split() first.\nExpected: {path}")
    return torch.load(path)


def load_attributes(split: str) -> torch.Tensor:
    # [N_split, 85] float32 {0,1} — class-row lookup expanded per image, split-local
    # row order == load_split(split) order (which is global-index order).
    matrix = load_class_attribute_matrix()
    index = load_image_index()
    class_idx = torch.tensor(index['class_idx'], dtype=torch.long)
    return matrix[class_idx[load_split(split)]]


def load_class_labels(split: str) -> torch.Tensor:
    # [N_split] long — class index per image, aligned with load_attributes(split).
    index = load_image_index()
    class_idx = torch.tensor(index['class_idx'], dtype=torch.long)
    return class_idx[load_split(split)]


def load_image_features(split: str) -> torch.Tensor:
    # [N_split, 512] float32, L2-normalized; row order == load_split(split) order.
    path = _get_artifacts_dir() / f'clip_image_features_{split}.pt'
    if not path.exists():
        raise FileNotFoundError(
            f"CLIP features not found. Run src/awa2/clip_features.py first.\nExpected: {path}"
        )
    return torch.load(path)


if __name__ == '__main__':
    # One-time setup: python src/awa2/data_loader.py
    build_image_index()
    build_split()
    print(f"  Attribute count: {len(ATTRIBUTE_NAMES)}")
    print(f"  Test attributes shape: {tuple(load_attributes('test').shape)}")
