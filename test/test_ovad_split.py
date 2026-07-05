"""
OVAD synthetic split verification.

Unit checks build_split on a synthetic instance index in a tmp artifacts dir (seed
determinism, image-level integrity), plus integrity checks on the REAL frozen split if
present (disjointness, full coverage, no image_id straddling, loader alignment).

Run:  pytest test/test_ovad_split.py
"""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ovad import data_loader as dl


def _fake_index(tmp_path, monkeypatch, n_images=10, per_image=3):
    # Synthetic instance index: n_images × per_image instances, in a tmp artifacts dir.
    monkeypatch.setattr(dl, "_get_artifacts_dir", lambda: tmp_path)
    image_ids = [100 + i for i in range(n_images) for _ in range(per_image)]
    manifest = {
        "ann_ids": list(range(len(image_ids))),
        "image_ids": image_ids,
        "file_names": [f"{i}.jpg" for i in image_ids],
        "category_ids": [1] * len(image_ids),
        "bboxes": [[0, 0, 64, 64]] * len(image_ids),
        "att_vecs": [[1, 0, -1]] * len(image_ids),
    }
    with open(tmp_path / "instance_index.json", "w") as f:
        json.dump(manifest, f)
    return image_ids


def test_build_split_deterministic_and_image_level(tmp_path, monkeypatch):
    image_ids = _fake_index(tmp_path, monkeypatch)
    dl.build_split(force=True)
    tr1, te1 = dl.load_split("train"), dl.load_split("test")
    dl.build_split(force=True)
    tr2, te2 = dl.load_split("train"), dl.load_split("test")
    assert torch.equal(tr1, tr2) and torch.equal(te1, te2)   # seed determinism

    ids = torch.tensor(image_ids)
    assert len(tr1) + len(te1) == len(ids)                   # full coverage
    assert not set(tr1.tolist()) & set(te1.tolist())         # disjoint
    # No image_id straddles the split.
    assert not set(ids[tr1].tolist()) & set(ids[te1].tolist())
    # 80/20 at image level: 8 train images × 3 instances.
    assert len(tr1) == 8 * 3 and len(te1) == 2 * 3
    assert dl.load_split_meta() == {"train_fraction": dl.TRAIN_FRACTION, "seed": dl.SPLIT_SEED}


def test_split_aware_loaders_slice_by_split(tmp_path, monkeypatch):
    _fake_index(tmp_path, monkeypatch)
    dl.build_split(force=True)
    full = dl.load_attributes()
    test_view = dl.load_attributes("test")
    assert torch.equal(test_view, full[dl.load_split("test")])
    assert torch.equal(dl.load_category_labels("train"),
                       dl.load_category_labels()[dl.load_split("train")])


def test_real_frozen_split_integrity():
    # Integrity of the ACTUAL frozen split (skipped where artifacts aren't built).
    try:
        train_idx, test_idx = dl.load_split("train"), dl.load_split("test")
        image_ids = torch.tensor(dl.load_instance_index()["image_ids"])
    except FileNotFoundError:
        pytest.skip("real OVAD split/index not built on this machine")
    assert len(train_idx) + len(test_idx) == len(image_ids)
    assert not set(train_idx.tolist()) & set(test_idx.tolist())
    assert not set(image_ids[train_idx].tolist()) & set(image_ids[test_idx].tolist())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
