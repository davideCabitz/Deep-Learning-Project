"""
Tier-5 CD-Bayes verification.

Cheap, self-contained checks of tier5_cdbayes.py on synthetic tensors — no CLIP, no real
feature DB, no artifacts. The load-bearing checks are #2 (tree sum-product is exact vs
brute-force enumeration), #4/#5 (the tier4-parity invariants the whole ablation matrix
rests on) and #8 (Φ′ at init reproduces Stage A′ core).

Run:  python test/test_tier5_cdbayes.py   (or: pytest test/test_tier5_cdbayes.py)
"""

import itertools
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tier4_bayes import EPS, N_ATTR, BayesPhi, Tier4Config, make_stage_a_chunk
from tier5_cdbayes import (
    DependencyTree,
    StageAPrime,
    build_chow_liu_tree,
    clamp_evidence,
    evidence_log_ratio,
    fit_temperature_scalars,
    lambda_loss,
    make_stage_b_prime_chunk,
    maximum_spanning_tree,
    mutual_information_matrix,
    qbnorm_rescore,
    tree_constraint_log,
    tree_log_partition,
    tree_marginals,
)
from vm_pipeline import query_indices


def _fake_db(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Synthetic unit features + valid posteriors in (EPS, 1−EPS).
    g = torch.Generator().manual_seed(seed)
    feats = F.normalize(torch.randn(n, 512, generator=g), dim=1)
    P = torch.rand(n, N_ATTR, generator=g).clamp(EPS, 1 - EPS)
    return feats, P


def _synthetic_labels(n_img: int, n_attr: int, seed: int) -> torch.Tensor:
    # Correlated binary labels: half the columns copy a noisy version of their predecessor,
    # so the MI matrix has real structure for the tree to find.
    g = torch.Generator().manual_seed(seed)
    A = (torch.rand(n_img, n_attr, generator=g) > 0.5).float()
    for c in range(1, n_attr, 2):
        flip = (torch.rand(n_img, generator=g) > 0.8).float()
        A[:, c] = A[:, c - 1] * (1 - flip) + (1 - A[:, c - 1]) * flip
    return A


def _brute_force_log_joint(tree: DependencyTree, log_u: torch.Tensor) -> torch.Tensor:
    # log[ p_T(A)·Π_a exp(u_a(A_a)) ] for every assignment — the exhaustive reference the
    # sum-product results are checked against. Returns [N, 2^n].
    n = len(tree.parent)
    rows = []
    for bits in itertools.product([0, 1], repeat=n):
        lp = torch.zeros(log_u.shape[0])
        for a, s in enumerate(bits):
            p = tree.parent[a]
            lp += (tree.log_marginal[a, s] if p < 0 else tree.log_cpt[a, bits[p], s])
            lp += log_u[:, a, s]
        rows.append(lp)
    return torch.stack(rows, dim=1)


# ---------------------------------------------------------------------------
# 1. Chow-Liu: Prim's MST matches brute force over ALL labeled trees (Prüfer)
# ---------------------------------------------------------------------------

def test_chow_liu_is_max_spanning_tree():
    n = 5
    A = _synthetic_labels(300, n, seed=0)
    mi = mutual_information_matrix(A, alpha=1.0)
    assert torch.allclose(mi, mi.T, atol=1e-6) and (mi.diagonal() == 0).all()

    parent = maximum_spanning_tree(mi)
    got = sum(mi[c, p].item() for c, p in enumerate(parent) if p >= 0)

    best = float("-inf")                     # enumerate all n^{n-2} trees via Prüfer decode
    for seq in itertools.product(range(n), repeat=n - 2):
        degree = [1] * n
        for v in seq:
            degree[v] += 1
        seq_list, weight = list(seq), 0.0
        for v in seq_list:
            leaf = min(i for i in range(n) if degree[i] == 1)
            weight += mi[leaf, v].item()
            degree[leaf] -= 1
            degree[v] -= 1
        u, v = [i for i in range(n) if degree[i] == 1]
        weight += mi[u, v].item()
        best = max(best, weight)
    assert abs(got - best) < 1e-6, f"Prim weight {got} != brute-force max {best}"
    print("[OK] 1. Chow-Liu tree is the maximum spanning tree (Prüfer brute force)")


# ---------------------------------------------------------------------------
# 2. Tree sum-product (logZ, constraint, marginals) is EXACT vs enumeration
# ---------------------------------------------------------------------------

def test_sum_product_vs_enumeration():
    n = 7
    tree = build_chow_liu_tree(_synthetic_labels(400, n, seed=1), alpha=1.0)
    g = torch.Generator().manual_seed(2)
    q = torch.rand(3, n, generator=g).clamp(0.05, 0.95)
    log_u = evidence_log_ratio(q, tree.log_marginal)

    joint = _brute_force_log_joint(tree, log_u)                        # [3, 2^n]
    want_z = torch.logsumexp(joint, dim=1)
    assert torch.allclose(tree_log_partition(tree, log_u), want_z, atol=1e-5)

    pi, ni = [1, 4], [2]
    keep = [j for j, bits in enumerate(itertools.product([0, 1], repeat=n))
            if all(bits[a] == 1 for a in pi) and all(bits[b] == 0 for b in ni)]
    want_c = torch.logsumexp(joint[:, keep], dim=1) - want_z
    got_c = tree_constraint_log(tree, log_u, want_z, pi, ni)
    assert torch.allclose(got_c, want_c, atol=1e-5), "constraint term mismatch"

    probs = torch.softmax(joint, dim=1)
    want_m = torch.stack(
        [sum(probs[:, j] for j, bits in
             enumerate(itertools.product([0, 1], repeat=n)) if bits[a] == 1)
         for a in range(n)], dim=1)
    assert torch.allclose(tree_marginals(tree, log_u), want_m, atol=1e-5), "marginal mismatch"

    cond = tree_marginals(tree, clamp_evidence(log_u, pi, ni))         # conditioned marginals
    cond_want = torch.stack(
        [sum(probs[:, j] for j, bits in
             enumerate(itertools.product([0, 1], repeat=n)) if bits[a] == 1 and j in set(keep))
         for a in range(n)], dim=1) / probs[:, keep].sum(dim=1, keepdim=True)
    assert torch.allclose(cond, cond_want, atol=1e-5), "conditioned marginal mismatch"
    print("[OK] 2. tree sum-product (Z, constraint, marginals) matches brute force")


# ---------------------------------------------------------------------------
# 3. Temperature fitting strictly reduces held-out NLL on a miscalibrated probe
# ---------------------------------------------------------------------------

def test_temperature_reduces_nll():
    g = torch.Generator().manual_seed(3)
    z_true = 1.5 * torch.randn(4000, 6, generator=g)
    y = (torch.rand(4000, 6, generator=g) < torch.sigmoid(z_true)).float()
    z_obs = 3.0 * z_true                                               # overconfident ×3
    tau = fit_temperature_scalars(z_obs[:2000], y[:2000], steps=400, lr=0.05)
    assert ((tau > 2.0) & (tau < 4.0)).all(), f"tau should recover ≈3, got {tau}"
    nll_before = F.binary_cross_entropy_with_logits(z_obs[2000:], y[2000:])
    nll_after = F.binary_cross_entropy_with_logits(z_obs[2000:] / tau, y[2000:])
    assert nll_after < nll_before, "calibration must strictly reduce held-out NLL"
    print("[OK] 3. temperature scaling recovers tau≈3 and reduces held-out NLL")


# ---------------------------------------------------------------------------
# 4. Edgeless forest: C2 collapses to the independent product, C3 to the raw tail
# ---------------------------------------------------------------------------

def test_empty_tree_reduces_to_independence():
    n = 6
    g = torch.Generator().manual_seed(4)
    marg = torch.rand(n, generator=g).clamp(0.2, 0.8)
    empty = DependencyTree(parent=(-1,) * n, order=tuple(range(n)),
                           log_cpt=torch.zeros(n, 2, 2),
                           log_marginal=torch.stack([(1 - marg).log(), marg.log()], dim=1),
                           mi=torch.zeros(n, n))
    q = torch.rand(50, n, generator=g).clamp(EPS, 1 - EPS)
    log_u = evidence_log_ratio(q, empty.log_marginal)
    pi, ni, free = [0], [3], [1, 2, 4, 5]

    z_free = tree_log_partition(empty, log_u)
    assert z_free.abs().max() < 1e-5, "edgeless evidence-ratio Z must be exactly 1"
    got = tree_constraint_log(empty, log_u, z_free, pi, ni)
    want = q[:, pi].log().sum(1) + (1 - q[:, ni]).log().sum(1)
    assert torch.allclose(got, want, atol=1e-5), "edgeless constraint != independent product"

    cond = tree_marginals(empty, clamp_evidence(log_u, pi, ni))[:, free]
    assert torch.allclose(cond, q[:, free], atol=1e-5), \
        "edgeless conditioned marginals must equal the raw posteriors (C3 == raw tail)"
    print("[OK] 4. edgeless forest: C2 == independent product, C3 == unconditioned tail")


# ---------------------------------------------------------------------------
# 5. HARD INVARIANT — all components off ≡ tier4 Stage A
# ---------------------------------------------------------------------------

def test_all_off_equals_tier4_stage_a():
    cfg = Tier4Config()
    feats, P = _fake_db(200, seed=5)
    pi, ni, free = query_indices("+Wearing_Lipstick, -Heavy_Makeup, +Smiling")
    want = make_stage_a_chunk(feats, P, cfg)([3, 7, 11], pi, ni, free)
    got = StageAPrime(feats, P, tree=None, conditioned_tail=False, lam=0.0,
                      hamming_max=cfg.hamming_max).chunk([3, 7, 11], pi, ni, free)
    diff = (want - got).abs().max().item()
    assert diff < 1e-4, f"all-off scorer deviates from tier4 Stage A: {diff}"
    print("[OK] 5. hard invariant: all-off CD-Bayes == tier4 Stage A")


# ---------------------------------------------------------------------------
# 6. LambdaLoss — direction + hand-computed 2-item and 4-item lists
# ---------------------------------------------------------------------------

def test_lambda_loss():
    # Direction: raising the positive's score (rank order unchanged) lowers the loss.
    l0 = lambda_loss(torch.tensor([1.0, 0.5, 0.4, 0.3]), n_pos=1, sigma=1.0)
    l1 = lambda_loss(torch.tensor([2.0, 0.5, 0.4, 0.3]), n_pos=1, sigma=1.0)
    assert l1 < l0, "raising the positive's score must reduce LambdaLoss"

    # Hand case, 2 items: positive s=1 at rank 2, negative s=2 at rank 1.
    # |ΔNDCG| = |1/maxDCG|·|D2 − D1| = 1·|1/log2(3) − 1| ; loss = w·log(1+e^{−(1−2)}).
    w = abs(1 / math.log2(3) - 1.0)
    want = w * math.log(1 + math.e)
    got = lambda_loss(torch.tensor([1.0, 2.0]), n_pos=1, sigma=1.0).item()
    assert abs(got - want) < 1e-5, f"2-item LambdaLoss {got} != hand value {want}"

    # Hand case, 4 items via an independent scalar loop (positives first by convention).
    scores = [3.0, 1.0, 2.0, 0.0]
    n_pos, ranks = 2, {0: 1, 2: 2, 1: 3, 3: 4}                          # by descending score
    D = {i: 1 / math.log2(1 + r) for i, r in ranks.items()}
    max_dcg = 1 / math.log2(2) + 1 / math.log2(3)
    want = sum((1 / max_dcg) * abs(D[i] - D[j]) * math.log(1 + math.exp(-(scores[i] - scores[j])))
               for i in range(n_pos) for j in range(n_pos, 4))
    got = lambda_loss(torch.tensor(scores), n_pos=n_pos, sigma=1.0).item()
    assert abs(got - want) < 1e-5, f"4-item LambdaLoss {got} != hand value {want}"
    print("[OK] 6. LambdaLoss direction + hand-computed values")


# ---------------------------------------------------------------------------
# 7. QB-Norm — hand-computed DIS case incl. the dynamic activation gate
# ---------------------------------------------------------------------------

def test_qbnorm_dis():
    bank = torch.tensor([[10.0, 0.0, 0.0], [0.0, 5.0, 0.0]])           # activation set {0, 1}
    scores = torch.tensor([[3.0, 2.0, 1.0],                            # top-1 = 0 → renormalized
                           [1.0, 2.0, 9.0]])                           # top-1 = 2 → untouched
    out = qbnorm_rescore(scores, bank, beta=1.0)

    denom = torch.logsumexp(bank, dim=0)                               # per-item hub mass
    assert torch.allclose(out[0], scores[0] - denom, atol=1e-5), "DIS renormalization wrong"
    assert int(out[0].argmax()) == 2, "the hub item must lose its top-1 after DIS"
    assert torch.equal(out[1], scores[1]), "non-activated query must pass through untouched"
    print("[OK] 7. QB-Norm DIS matches the hand-computed case and gates dynamically")


# ---------------------------------------------------------------------------
# 8. Φ′ at init ≡ Stage A′ core (the B′ zero-init invariant)
# ---------------------------------------------------------------------------

def test_phi_prime_init_equals_stage_a_prime():
    torch.manual_seed(6)
    cfg = Tier4Config()
    feats, P = _fake_db(150, seed=7)
    tree = build_chow_liu_tree(_synthetic_labels(500, N_ATTR, seed=8), alpha=1.0)
    core = StageAPrime(feats, P, tree, conditioned_tail=True, lam=0.0,
                       hamming_max=cfg.hamming_max)
    pi, ni, free = query_indices("+Eyeglasses, -Smiling")
    a = core.chunk([2, 9], pi, ni, free)
    b = make_stage_b_prime_chunk(BayesPhi(cfg), core, use_qv=True)([2, 9], pi, ni, free)
    diff = (a - b).abs().max().item()
    assert diff < 1e-4, f"Phi' at init deviates from Stage A' core: {diff}"
    print("[OK] 8. BayesPhi at init reproduces the Stage A' core score")


if __name__ == "__main__":
    test_chow_liu_is_max_spanning_tree()
    test_sum_product_vs_enumeration()
    test_temperature_reduces_nll()
    test_empty_tree_reduces_to_independence()
    test_all_off_equals_tier4_stage_a()
    test_lambda_loss()
    test_qbnorm_dis()
    test_phi_prime_init_equals_stage_a_prime()
    print("\nAll Tier-5 CD-Bayes checks passed.")
