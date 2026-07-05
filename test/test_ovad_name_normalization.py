from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'Baselines'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'Training-Free'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'Trained'))

from ovad import tier_clay, tier_dgp as ovad_tier_dgp
from ovad.adapter import configure_tier_dgp
from ovad.tier0_baseline import score as baseline_score


def test_baseline_accepts_query_normalized_attribute_names():
    image_features = torch.eye(3, 4)
    attr_directions = torch.zeros(117, 4)
    mu_img = torch.zeros(4)

    ranking = baseline_score(['clothes_color:blue'], [], 0, image_features,
                             attr_directions, mu_img)

    assert ranking == [1, 2, 0]


def test_clay_stack_accepts_query_normalized_attribute_names(monkeypatch):
    monkeypatch.setattr(tier_clay, '_true_prompt_count', lambda name: 1)
    prompt_bank = torch.zeros(117, 1, 2)

    stacked = tier_clay._stack_condition_prompts(['clothes_color:blue'], [], prompt_bank)

    assert stacked.shape == (1, 2)


def test_dgp_adapter_binds_dual_attribute_index():
    import dgp

    configure_tier_dgp()

    assert dgp.ATTR_TO_IDX['clothes color:blue'] == dgp.ATTR_TO_IDX['clothes_color:blue']


def test_dgp_runner_normalizes_query_strings():
    assert ovad_tier_dgp._normalize_query_string('+ clothes_color:blue') == '+ clothes color:blue'
