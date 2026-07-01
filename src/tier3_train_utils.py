"""
Tier-3 shared training utilities — single owner of the online-mining and
disentanglement helpers reused across the trained fusion tiers.

Extracted from tier3_combined.py so that tier3_combined and the newer negation
tiers (tier3_neggate) depend on ONE copy instead of reaching sideways into each
other (CLAUDE.md §imports). Pure functions over a FusionPhi-shaped model — no I/O,
no config ownership; callers pass tensors and the model.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Frozen-DB invariant (spec §3.2 — "construct once, keep frozen")
# ---------------------------------------------------------------------------

def assert_db_frozen(image_features: torch.Tensor, get_ranking, probe_src: int = 0) -> None:
    # Tripwire — the frozen CLIP DB must be byte-identical before and after a query.
    # Snapshots the DB tensor, runs one ranking, and fails loudly if any scoring path
    # mutated it in place (the spec forbids re-encoding / per-query DB rewriting). Cached
    # side-tensors like probe matrices are separate and read-only, so this only guards the
    # CLIP feature DB itself. Cheap enough to call once per model in the eval harness.
    before = image_features.clone()
    ranking = get_ranking(probe_src)
    assert torch.equal(image_features, before), (
        "FROZEN-DB VIOLATION: the CLIP image-feature DB changed during a query. "
        "Scoring must never mutate or re-encode the frozen database (spec §3.2)."
    )
    assert isinstance(ranking, list) and ranking, "get_ranking must return a non-empty list of indices."
    assert ranking[0] != probe_src, "source index must be excluded from its own ranking (§5)."


# ---------------------------------------------------------------------------
# Distance-based hard-negative reranking (online curriculum mining)
# ---------------------------------------------------------------------------

def distance_mine(
    q: torch.Tensor,
    hard_ids: torch.Tensor,
    train_feats: torch.Tensor,
    k: int,
) -> torch.Tensor:
    # Hardest-negative reranking — among attribute-selected hard negatives keep the k
    # highest-cosine to the current query q (the images the model confuses most right now).
    # Standard curriculum mining (Schroff et al. 2015, FaceNet); train-split only, no GT leak.
    # q [d] unit; hard_ids [M]; returns [min(k, M)] indices into train_feats.
    if hard_ids.numel() <= k:
        return hard_ids
    cands = train_feats[hard_ids]              # [M, d]
    sims = cands @ q                           # [M] cosine similarities
    top = torch.topk(sims, k).indices          # [k] positions into hard_ids
    return hard_ids[top]


# ---------------------------------------------------------------------------
# Attribute disentanglement auxiliary loss
# ---------------------------------------------------------------------------

def disentanglement_loss(
    model,
    train_feats: torch.Tensor,
    raw_stacks: dict[str, torch.Tensor],
    n_pairs: int,
    rng: random.Random,
    dev: torch.device,
) -> torch.Tensor:
    # L_dis = (1/|pairs|) Σ_{(a,b)} |cos(d_a, d_b)| over sampled attribute pairs.
    # Penalises collinear attribute directions to fight CLIP's attribute entanglement
    # (e.g. Male↔Mustache), which makes negation geometrically cleaner. One shared random
    # reference per call keeps it to a single log_map (cheap). `model` is FusionPhi-shaped:
    # it exposes .phi.mlp_ref / .phi._center / .phi._direction / .phi.attn_pos.
    attr_names = list(raw_stacks.keys())
    r_idx = rng.randint(0, train_feats.shape[0] - 1)
    v_ref = train_feats[r_idx].unsqueeze(0)               # [1, d]

    h_ref = v_ref + model.phi.mlp_ref(v_ref)              # [1, d]
    directions: dict[str, torch.Tensor] = {}
    sampled_attrs = rng.sample(attr_names, min(n_pairs * 2, len(attr_names)))
    for name in sampled_attrs:
        T_hat = model.phi._center(raw_stacks[name])
        d = model.phi._direction(h_ref, T_hat, model.phi.attn_pos)   # [1, d]
        directions[name] = F.normalize(d.squeeze(0), dim=0)          # [d]

    names = list(directions.keys())
    if len(names) < 2:
        return torch.zeros((), device=dev)

    pairs_tried, loss = 0, torch.zeros((), device=dev)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            cos_sim = (directions[names[i]] * directions[names[j]]).sum()
            loss = loss + cos_sim.abs()
            pairs_tried += 1
            if pairs_tried >= n_pairs:
                break
        if pairs_tried >= n_pairs:
            break

    return loss / max(pairs_tried, 1)
