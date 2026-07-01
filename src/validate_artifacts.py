"""
Artifact validation — run on the VM before any Tier-3 training to confirm all
required .pt files are present, correctly shaped, typed, and normalized.

Required by tier3_*.py:
  clip_image_features_test.pt   — eval DB (test split)
  clip_image_features_train.pt  — training features
  celeba_attributes_train.pt    — training attribute labels
  clip_attr_prompt_bank.pt      — per-attribute prompt stacks
  visual_directions.pt          — global mean μ + GDE directions

Run:  python src/validate_artifacts.py
Exits with code 1 if any check fails.
"""

from __future__ import annotations

import sys
import torch
from pathlib import Path

from data_loader import _get_artifacts_dir

FEATURE_DIM   = 512
N_ATTRS       = 40
N_TEST        = 19962
N_TRAIN       = 162770


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok(name: str, msg: str = "") -> bool:
    print(f"  [PASS] {name}" + (f"  ({msg})" if msg else ""))
    return True


def _fail(name: str, reason: str) -> bool:
    print(f"  [FAIL] {name}  ← {reason}")
    return False


def _load(path: Path, name: str) -> torch.Tensor | None:
    if not path.exists():
        _fail(name, f"not found: {path}")
        return None
    try:
        return torch.load(path, weights_only=True)
    except Exception as e:
        _fail(name, f"load error: {e}")
        return None


def _check(cond: bool, name: str, reason: str) -> bool:
    return True if cond else _fail(name, reason)


def _unit_normalized(t: torch.Tensor, name: str, atol: float = 1e-3) -> bool:
    norms = t.reshape(-1, t.shape[-1]).norm(p=2, dim=1)
    bad = (~torch.isclose(norms, torch.ones_like(norms), atol=atol)).sum().item()
    return _check(bad == 0, name, f"{bad} rows not unit-normalized")


def _no_nan_inf(t: torch.Tensor, name: str) -> bool:
    if torch.isnan(t).any():
        return _fail(name, "contains NaN")
    if torch.isinf(t).any():
        return _fail(name, "contains Inf")
    return True


def _binary(t: torch.Tensor, name: str) -> bool:
    sample = t.flatten()[torch.randperm(t.numel())[:2000]]
    bad = ((sample != 0.0) & (sample != 1.0)).sum().item()
    return _check(bad == 0, name, f"{bad} values not in {{0, 1}}")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_test_features(artifacts: Path) -> bool:
    name = "clip_image_features_test.pt"
    t = _load(artifacts / name, name)
    if t is None:
        return False
    ok = all([
        _check(t.shape == (N_TEST, FEATURE_DIM), name, f"shape {tuple(t.shape)} != ({N_TEST}, {FEATURE_DIM})"),
        _check(t.dtype == torch.float32,          name, f"dtype {t.dtype} != float32"),
        _no_nan_inf(t, name),
        _unit_normalized(t, name),
    ])
    return _ok(name, f"{tuple(t.shape)}, float32, unit-normalized") if ok else False


def check_train_features(artifacts: Path) -> bool:
    name = "clip_image_features_train.pt"
    t = _load(artifacts / name, name)
    if t is None:
        return False
    ok = all([
        _check(t.shape == (N_TRAIN, FEATURE_DIM), name, f"shape {tuple(t.shape)} != ({N_TRAIN}, {FEATURE_DIM})"),
        _check(t.dtype == torch.float32,           name, f"dtype {t.dtype} != float32"),
        _no_nan_inf(t, name),
        _unit_normalized(t, name),
    ])
    return _ok(name, f"{tuple(t.shape)}, float32, unit-normalized") if ok else False


def check_train_attributes(artifacts: Path) -> bool:
    name = "celeba_attributes_train.pt"
    t = _load(artifacts / name, name)
    if t is None:
        return False
    ok = all([
        _check(t.shape == (N_TRAIN, N_ATTRS), name, f"shape {tuple(t.shape)} != ({N_TRAIN}, {N_ATTRS})"),
        _check(t.dtype == torch.float32,      name, f"dtype {t.dtype} != float32"),
        _no_nan_inf(t, name),
        _binary(t, name),
    ])
    return _ok(name, f"{tuple(t.shape)}, float32, values in {{0,1}}") if ok else False


def check_prompt_bank(artifacts: Path) -> bool:
    name = "clip_attr_prompt_bank.pt"
    t = _load(artifacts / name, name)
    if t is None:
        return False
    ok = True
    if t.ndim != 3:
        return _fail(name, f"expected 3-d, got {t.ndim}-d")
    n_attrs, n_prompts, d = t.shape
    ok = all([
        _check(n_attrs == N_ATTRS,    name, f"dim-0 {n_attrs} != {N_ATTRS} attributes"),
        _check(d == FEATURE_DIM,      name, f"dim-2 {d} != {FEATURE_DIM}"),
        _check(n_prompts >= 2,        name, f"only {n_prompts} prompt(s) per attr — need >= 2"),
        _check(t.dtype == torch.float32, name, f"dtype {t.dtype} != float32"),
        _no_nan_inf(t, name),
        _unit_normalized(t, name),
    ])
    return _ok(name, f"{tuple(t.shape)}, float32, unit-normalized") if ok else False


def check_visual_directions(artifacts: Path) -> bool:
    name = "visual_directions.pt"
    path = artifacts / name
    if not path.exists():
        return _fail(name, f"not found: {path}")
    try:
        ckpt = torch.load(path, weights_only=True)
    except Exception as e:
        return _fail(name, f"load error: {e}")

    ok = True
    for key, expected in [("mu", (FEATURE_DIM,)), ("directions", (N_ATTRS, FEATURE_DIM))]:
        if key not in ckpt:
            _fail(name, f"missing key '{key}'")
            ok = False
            continue
        t = ckpt[key]
        ok = ok and _check(tuple(t.shape) == expected, name, f"['{key}'] shape {tuple(t.shape)} != {expected}")
        ok = ok and _no_nan_inf(t, name)

    if ok:
        mu_norm = ckpt["mu"].norm().item()
        ok = ok and _check(abs(mu_norm - 1.0) < 1e-3, name, f"mu not unit-normalized (norm={mu_norm:.4f})")

    return _ok(name, f"mu {tuple(ckpt['mu'].shape)}, directions {tuple(ckpt['directions'].shape)}") if ok else False


def check_train_alignment(artifacts: Path) -> bool:
    name = "train row alignment"
    fp = artifacts / "clip_image_features_train.pt"
    ap = artifacts / "celeba_attributes_train.pt"
    if not fp.exists() or not ap.exists():
        return _fail(name, "skipped — one or both train files missing")
    n_feat = torch.load(fp, weights_only=True).shape[0]
    n_attr = torch.load(ap, weights_only=True).shape[0]
    ok = _check(n_feat == n_attr, name, f"features {n_feat} rows != attributes {n_attr} rows")
    return _ok(name, f"{n_feat} rows aligned") if ok else False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    artifacts = _get_artifacts_dir()
    print(f"Artifacts dir: {artifacts}\n")

    checks = [
        ("Test features (eval DB)",   lambda: check_test_features(artifacts)),
        ("Train features",            lambda: check_train_features(artifacts)),
        ("Train attributes",          lambda: check_train_attributes(artifacts)),
        ("Prompt bank",               lambda: check_prompt_bank(artifacts)),
        ("Visual directions (μ)",     lambda: check_visual_directions(artifacts)),
        ("Train row alignment",       lambda: check_train_alignment(artifacts)),
    ]

    results: list[tuple[str, bool]] = []
    for label, fn in checks:
        print(label)
        results.append((label, fn()))
        print()

    passed = sum(ok for _, ok in results)
    total  = len(results)
    print("=" * 50)
    print(f"  {passed}/{total} passed")
    if passed == total:
        print("  All artifacts valid — safe to run Tier-3 training.")
    else:
        failed = [label for label, ok in results if not ok]
        print(f"  Fix before training: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
