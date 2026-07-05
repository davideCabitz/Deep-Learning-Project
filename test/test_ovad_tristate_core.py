"""
Masked tri-state cores in src/ovad/adapter.py — binary parity + correctness.

The OVAD adapter rebinds the CelebA label-consuming functions (BCE probe trainers,
Chow-Liu counts, temperature/ECE fit) with masked variants that treat -1 as "no
evidence". The CelebA modules are untouched, so each masked variant is pinned here:
(a) EXACT parity with the original on binary {0,1} labels, and (b) -1 entries ignored
on tri-state labels.

Run:  pytest test/test_ovad_tristate_core.py
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import tier4_bayes
import tier5_cdbayes as t5
from ovad import adapter


# ---------------------------------------------------------------------------
# Chow-Liu pairwise-complete counts
# ---------------------------------------------------------------------------

def test_pair_counts_binary_parity_with_original():
    torch.manual_seed(0)
    A = (torch.rand(64, 6) > 0.5).float()
    assert torch.equal(adapter._masked_pair_counts(A), t5._pair_counts(A))


def test_pair_counts_tristate_pairwise_complete():
    # Hand-checked 2-attr case: rows with a -1 on either attribute drop out of that pair.
    A = torch.tensor([
        [1.0, 1.0],    # counted: (1,1)
        [1.0, 0.0],    # counted: (1,0)
        [0.0, -1.0],   # NOT counted for the (0,1) pair
        [-1.0, 1.0],   # NOT counted
        [0.0, 0.0],    # counted: (0,0)
    ])
    c = adapter._masked_pair_counts(A)[0, 1]
    assert c[1, 1] == 1 and c[1, 0] == 1 and c[0, 0] == 1 and c[0, 1] == 0
    assert c.sum() == 3   # only the 3 jointly-labeled rows


def test_mutual_information_binary_parity_with_original():
    torch.manual_seed(1)
    A = (torch.rand(200, 5) > 0.6).float()
    assert torch.allclose(adapter._masked_mutual_information_matrix(A, 1.0),
                          t5.mutual_information_matrix(A, 1.0))


def test_chow_liu_tree_binary_parity_with_original():
    torch.manual_seed(1)
    A = (torch.rand(200, 5) > 0.6).float()
    ours, orig = adapter._masked_build_chow_liu_tree(A, 1.0), t5.build_chow_liu_tree(A, 1.0)
    assert ours.parent == orig.parent and ours.order == orig.order
    assert torch.allclose(ours.log_marginal, orig.log_marginal)
    assert torch.allclose(ours.log_cpt, orig.log_cpt)
    assert torch.allclose(ours.mi, orig.mi)


def test_chow_liu_tree_tristate_runs_and_is_proper():
    # Tri-state labels: tree builds, CPT rows and marginals stay proper distributions.
    torch.manual_seed(2)
    A = torch.randint(-1, 2, (300, 6)).float()
    tree = adapter._masked_build_chow_liu_tree(A, alpha=1.0)
    assert torch.allclose(tree.log_marginal.exp().sum(1), torch.ones(6))
    assert torch.isfinite(tree.log_cpt).all() and torch.isfinite(tree.mi).all()


# ---------------------------------------------------------------------------
# Temperature fit + ECE
# ---------------------------------------------------------------------------

def test_fit_temperature_binary_parity_with_original():
    torch.manual_seed(3)
    z = torch.randn(128, 4)
    y = (torch.rand(128, 4) > 0.5).float()
    tau_orig = t5.fit_temperature_scalars(z, y, steps=50, lr=0.05)
    tau_masked = adapter._masked_fit_temperature_scalars(z, y, steps=50, lr=0.05)
    assert torch.allclose(tau_orig, tau_masked)


def test_ece_binary_parity_and_tristate_masking():
    torch.manual_seed(4)
    p = torch.rand(100, 1)
    y = (torch.rand(100, 1) > 0.5).float()
    # Binary parity with the original.
    assert torch.allclose(adapter._masked_expected_calibration_error(p, y, 5),
                          t5.expected_calibration_error(p, y, 5))
    # Masked ECE over padded data == unmasked ECE over the valid subset alone.
    y_pad = y.clone()
    y_pad[60:] = -1.0
    masked = adapter._masked_expected_calibration_error(p, y_pad, 5)
    subset = t5.expected_calibration_error(p[:60], y[:60], 5)
    assert torch.allclose(masked, subset)


# ---------------------------------------------------------------------------
# Masked-BCE probes (linear variant; the MLP variant uses the identical _masked_bce)
# ---------------------------------------------------------------------------

def test_masked_bce_binary_parity_with_plain_bce():
    torch.manual_seed(6)
    logits = torch.randn(64, 3)
    y = (torch.rand(64, 3) > 0.5).float()
    assert torch.allclose(adapter._masked_bce(logits, y),
                          torch.nn.functional.binary_cross_entropy_with_logits(logits, y))


def test_masked_probe_training_ignores_unlabeled(monkeypatch):
    # Attribute 0: half positives, half UNLABELED (-1). Masked BCE sees only positive
    # evidence -> probe pushes p >> 0.5. If -1 leaked in as a negative-ish target, the
    # logits would be dragged down instead — this discriminates the mask behaviorally.
    torch.manual_seed(5)
    n, d = 256, 8
    feats = torch.randn(n, d)
    labels = torch.full((n, 2), -1.0)
    labels[: n // 2, 0] = 1.0                                 # attr0: 1 or -1 only
    labels[:, 1] = (torch.rand(n) > 0.5).float()              # attr1: ordinary binary

    monkeypatch.setattr(tier4_bayes, "FEATURE_DIM", d)
    monkeypatch.setattr(tier4_bayes, "N_ATTR", 2)
    monkeypatch.setattr(tier4_bayes, "_load_train_features", lambda: feats)
    monkeypatch.setattr(tier4_bayes, "_load_train_attributes", lambda: labels)

    # Default probe_batch (8192) gives 1 Adam step/epoch on 256 samples — shrink the
    # batch and raise epochs so the linear probe actually converges in-test.
    cfg = tier4_bayes.Tier4Config(probe_epochs=150, probe_batch=64)
    probe = adapter._masked_train_probes_linear(cfg, torch.device("cpu"))
    with torch.no_grad():
        p0 = torch.sigmoid(probe(feats)[:, 0]).mean()
    assert p0 > 0.8, f"masked probe should saturate positive on attr0, got {p0:.3f}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
