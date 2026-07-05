"""
AwA2 data_loader verification — reads the frozen dataset (no CLIP, no GPU).

Load-bearing checks: predicate matrix shape, attribute lookup against a known class
(zebra ⇒ stripes=1, furry=0), split determinism + disjointness + class stratification,
and the image-index manifest round-trip (index → file path contract).

Run:  pytest test/test_awa2_data_loader.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "Baselines"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "Training-Free"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "Trained"))

from awa2 import data_loader as dl


def test_class_attribute_matrix_shape():
    m = dl.load_class_attribute_matrix()
    assert m.shape == (dl.N_CLASSES, dl.N_ATTRS)
    assert set(m.unique().tolist()) <= {0.0, 1.0}


def test_known_class_attributes():
    # Ground-truth anchors on the matrix wiring (values verified against the binary matrix).
    m = dl.load_class_attribute_matrix()
    classes = dl.load_class_names()
    zebra = classes.index("zebra")
    assert m[zebra, dl.ATTR_TO_IDX["stripes"]] == 1.0
    assert m[zebra, dl.ATTR_TO_IDX["swims"]] == 0.0
    whale = classes.index("blue+whale")
    assert m[whale, dl.ATTR_TO_IDX["swims"]] == 1.0
    assert m[whale, dl.ATTR_TO_IDX["flippers"]] == 1.0
    assert m[whale, dl.ATTR_TO_IDX["furry"]] == 0.0


def test_attribute_names_cover_85():
    assert len(dl.ATTRIBUTE_NAMES) == dl.N_ATTRS == 85
    assert dl.ATTR_TO_IDX["black"] == 0
    assert dl.ATTR_TO_IDX["domestic"] == 84


def test_split_disjoint_and_deterministic():
    tr, te = dl.load_split("train"), dl.load_split("test")
    assert not (set(tr.tolist()) & set(te.tolist()))
    total = len(dl.load_image_index()["paths"])
    assert len(tr) + len(te) == total
    # Determinism: reloading yields identical index tensors.
    assert torch.equal(tr, dl.load_split("train"))


def test_split_is_class_stratified():
    # Every class must appear in BOTH splits — the whole reason for the per-image split.
    idx = dl.load_image_index()
    class_idx = torch.tensor(idx["class_idx"])
    for split in ("train", "test"):
        present = class_idx[dl.load_split(split)].unique()
        assert len(present) == dl.N_CLASSES, f"{split} missing classes"


def test_attributes_align_with_split():
    attrs = dl.load_attributes("test")
    labels = dl.load_class_labels("test")
    assert attrs.shape == (len(labels), dl.N_ATTRS)
    # Row i attributes must equal its class row (class-level attribute contract).
    m = dl.load_class_attribute_matrix()
    for i in (0, len(labels) // 2, len(labels) - 1):
        assert torch.equal(attrs[i], m[labels[i]])


def test_image_index_roundtrip():
    idx = dl.load_image_index()
    assert len(idx["paths"]) == len(idx["class_idx"])
    assert len(idx["classes"]) == dl.N_CLASSES
    # Path encodes its class folder — the index→file invariant.
    p0, c0 = idx["paths"][0], idx["class_idx"][0]
    assert idx["classes"][c0] in p0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
