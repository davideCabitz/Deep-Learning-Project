"""
AwA2 adapter — the single seam that rebinds CelebA-bound tier modules to AwA2 providers.

The tier3/4/5 modules hard-bind CelebA at import time (module constants ATTRIBUTE_NAMES /
ATTR_TO_IDX / N_ATTR, train/test loaders, artifact paths, prompt phrasing). Forking their
calibration / Chow-Liu / fusion math would massively duplicate; editing the constants
would break the CelebA pipeline. Instead each AwA2 trainer calls ONE configure_* function
here BEFORE importing the tier entry points, and the existing code runs unchanged.
(Adapter pattern; runtime module rebinding is deliberate and centralized HERE only.)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / "Baselines"))
sys.path.insert(0, str(Path(__file__).parents[1] / "Training-Free"))
sys.path.insert(0, str(Path(__file__).parents[1] / "Trained"))

from awa2 import data_loader as awa2_data
from awa2.data_loader import ATTRIBUTE_NAMES as AWA2_NAMES
from awa2.data_loader import N_ATTRS, _get_artifacts_dir

AWA2_ATTR_TO_IDX = {name: i for i, name in enumerate(AWA2_NAMES)}


def _train_feats():
    return awa2_data.load_image_features("train")


def _train_attrs():
    return awa2_data.load_attributes("train")


def _bind_tables(*mods):
    # Attribute name/index tables — free-set sizes, multi-hot builds, index→name maps.
    for mod in mods:
        mod.ATTRIBUTE_NAMES = AWA2_NAMES
        if hasattr(mod, "ATTR_TO_IDX"):
            mod.ATTR_TO_IDX = AWA2_ATTR_TO_IDX


def _bind_train_loaders(*mods):
    # CelebA train-split loaders (defined in gde, re-imported by value elsewhere).
    for mod in mods:
        mod._load_train_features = _train_feats
        mod._load_train_attributes = _train_attrs


def configure_tier5() -> None:
    # Bindings for the tier5 Bprime_listnet path (bap + cdbayes + deps).
    import gde
    import phi_gate
    import bap
    import cdbayes as t5
    import vm_pipeline

    _bind_tables(phi_gate, bap, t5, vm_pipeline)
    bap.ATTR_TO_IDX = AWA2_ATTR_TO_IDX
    t5.ATTR_TO_IDX = AWA2_ATTR_TO_IDX
    vm_pipeline.ATTR_TO_IDX = AWA2_ATTR_TO_IDX
    for mod in (bap, t5):
        mod.N_ATTR = N_ATTRS
        mod._get_artifacts_dir = _get_artifacts_dir   # probe/temperature caches → artifacts/awa2/
    _bind_train_loaders(gde, bap, t5)

    bap.load_image_features = lambda: awa2_data.load_image_features("test")

    def _awa2_probe_diag(P_test):
        attrs = awa2_data.load_attributes("test")
        acc = ((P_test.cpu() > 0.5).float() == attrs.float()).float().mean(0)
        worst = int(acc.argmin())
        bap.log(f"  probe test acc: mean {acc.mean():.4f}  "
                        f"min {acc[worst]:.4f} ({AWA2_NAMES[worst]})")

    bap.probe_accuracy_diagnostic = _awa2_probe_diag


def configure_tier2d() -> None:
    # Bindings for the tier2d DGP path (training-free): attribute tables, AwA2 prompt
    # counts for padding-strip, and gde's cache dir so μ comes from
    # artifacts/awa2/visual_directions.pt (built by src/awa2/tier_gde.py).
    import gde
    import dgp

    from awa2.clip_prompts import build_prompts_for_attribute as awa2_prompts

    _bind_tables(gde, dgp)
    _bind_train_loaders(gde)
    gde._get_artifacts_dir = _get_artifacts_dir
    dgp.build_prompts_for_attribute = awa2_prompts


def configure_dualscore() -> None:
    # Bindings for the tier3 DualScore path (attribute_probes + phi_gate + dgp deps).
    # MUST run before importing dual_score, so its by-value imports see these bindings.
    import gde
    import dgp
    import phi_gate
    import attribute_probes

    from awa2.clip_prompts import build_prompts_for_attribute as awa2_prompts

    _bind_tables(gde, dgp, phi_gate, attribute_probes)
    _bind_train_loaders(gde, attribute_probes)
    gde._get_artifacts_dir = _get_artifacts_dir  # visual_directions cache → artifacts/awa2/

    # Prompt padding is stripped by true per-attribute prompt counts — AwA2's, not CelebA's.
    dgp.build_prompts_for_attribute = awa2_prompts

    attribute_probes.load_image_features = lambda: awa2_data.load_image_features("test")

    # AttributeProbes defaults n_attr=40 and train_probes constructs it without the arg;
    # subclass with the AwA2 default so every construction site gets 85 heads.
    _CelebAProbes = attribute_probes.AttributeProbes

    class Awa2AttributeProbes(_CelebAProbes):
        def __init__(self, d_model, hidden, n_attr=N_ATTRS):
            super().__init__(d_model, hidden, n_attr=n_attr)

    attribute_probes.AttributeProbes = Awa2AttributeProbes
