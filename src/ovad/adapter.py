"""
OVAD adapter — attribute-table rebinding + the is_has_att prompt-builder wrapper.

tier1_CLAY.py needs only its pure manifold math (_build_subspace, _project_db), which
src/ovad/tier_clay.py imports directly and pairs with a locally re-implemented prompt
stacker — no rebinding needed there (mirrors src/awa2/tier_clay.py).

tier2d_dgp.py is different: its DGP-specific functions (_attr_stack, compute_mu_txt)
read the module-level ATTR_TO_IDX (CelebA's 40-name dict) and call the CelebA
single-arg build_prompts_for_attribute(name) INTERNALLY, not just at the runner's call
site. Forking _gated_directions_batch/_compose_batch to avoid this would duplicate the
real DGP math; instead configure_tier_dgp() rebinds those two names before
src/ovad/tier_dgp.py imports the module, following the src/awa2/adapter.py pattern.

One OVAD-specific wrinkle: build_prompts_for_attribute here needs BOTH the attribute
name and its is_has_att field (src/ovad/clip_prompts.py), but tier2d_dgp calls it with
a single `name` argument — _single_arg_prompt_builder closes over the name->is_has_att
lookup so the rebound function matches that call signature exactly.

Trained tiers (DualScore, tier5) get three extra seams the training-free path doesn't need:
  - TriStateQueryGenerator: the synthetic-query sampler with OVAD's tri-state label
    semantics (never query an unlabeled reference attribute; positives by the masked
    relative-Hamming RATE, the exact GT rule) — rebinds tier3_dgp.QueryGenerator.
  - Dual-keyed ATTR_TO_IDX: tier3_dualscore/tier5 index attributes with RAW names during
    training (from the QueryGenerator) but with parse_query's UNDERSCORED tokens at eval;
    OVAD names contain spaces, so the bound dict carries both key forms (never colliding).
  - Masked tri-state cores: the CelebA label-consuming functions (BCE probe trainers,
    Chow-Liu counts, temperature/ECE fit) assume binary {0,1} labels; their masked
    variants below treat -1 as "no evidence" and are rebound at configure time. The
    CelebA modules themselves are NEVER edited — deliberate hard isolation; the cost is
    a controlled near-duplicate of each function living here, with binary-parity tests
    (test_ovad_tristate_core.py) pinning them to the originals.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from ovad import data_loader as ovad_data
from ovad.data_loader import ATTRIBUTE_NAMES as OVAD_NAMES
from ovad.data_loader import N_ATTRS, _get_artifacts_dir, load_attribute_meta

OVAD_ATTR_TO_IDX = {name: i for i, name in enumerate(OVAD_NAMES)}
# Both raw ("clothes color:blue") and query-normalized ("clothes_color:blue") keys.
OVAD_ATTR_TO_IDX_DUAL = {**OVAD_ATTR_TO_IDX,
                         **{name.replace(' ', '_'): i for i, name in enumerate(OVAD_NAMES)}}
IS_HAS_BY_NAME = {m["name"]: m["is_has_att"] for m in load_attribute_meta()}


def _single_arg_prompt_builder():
    from ovad.clip_prompts import build_prompts_for_attribute as ovad_prompts

    def _wrapped(name: str) -> list[str]:
        return ovad_prompts(name, IS_HAS_BY_NAME[name])

    return _wrapped


def configure_tier_dgp() -> None:
    # Bindings for the OVAD tier2d DGP path (training-free).
    import tier1_GDE
    import tier2d_dgp

    tier2d_dgp.ATTR_TO_IDX = OVAD_ATTR_TO_IDX
    tier2d_dgp.build_prompts_for_attribute = _single_arg_prompt_builder()
    tier1_GDE._get_artifacts_dir = _get_artifacts_dir


def _train_feats():
    return ovad_data.load_image_features("train")


def _train_attrs():
    return ovad_data.load_attributes("train")


def _bind_tables(*mods):
    # Attribute name/index tables — free-set sizes, multi-hot builds, index→name maps.
    for mod in mods:
        mod.ATTRIBUTE_NAMES = OVAD_NAMES
        if hasattr(mod, "ATTR_TO_IDX"):
            mod.ATTR_TO_IDX = OVAD_ATTR_TO_IDX_DUAL


def _bind_train_loaders(*mods):
    # CelebA train-split loaders (defined in tier1_GDE, re-imported by value elsewhere).
    # Attributes stay int8 tri-state — the masked cores (train_probes, Chow-Liu,
    # temperature fit) and TriStateQueryGenerator handle the -1 entries themselves.
    for mod in mods:
        mod._load_train_features = _train_feats
        mod._load_train_attributes = _train_attrs


# ---------------------------------------------------------------------------
# Masked tri-state cores — OVAD variants of the CelebA label-consuming functions.
# Each treats -1 as "no evidence" and reduces EXACTLY to its CelebA original on
# binary {0,1} labels (pinned by test/test_ovad_tristate_core.py).
# ---------------------------------------------------------------------------

def _masked_bce(logits: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
    # Mean BCE over LABELED entries only; -1 targets carry zero weight.
    valid = (yb >= 0).float()
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits, yb.clamp(min=0), weight=valid, reduction="sum") / valid.sum().clamp(min=1)


def _masked_train_probes_linear(cfg, dev):
    # Replaces tier4_bayes.train_probes — identical loop, masked loss (tri-state labels).
    import tier4_bayes as t4
    train_feats = t4._load_train_features().to(dev)
    train_attrs = t4._load_train_attributes().float().to(dev)
    probe = torch.nn.Linear(t4.FEATURE_DIM, t4.N_ATTR).to(dev)
    opt = torch.optim.Adam(probe.parameters(), lr=cfg.probe_lr, weight_decay=cfg.probe_wd)
    n = train_feats.shape[0]
    for epoch in range(cfg.probe_epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, cfg.probe_batch):
            idx = perm[i:i + cfg.probe_batch]
            opt.zero_grad()
            loss = _masked_bce(probe(train_feats[idx]), train_attrs[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * idx.numel()
        t4.log(f"  probe epoch {epoch + 1:02d}/{cfg.probe_epochs}  BCE={tot / n:.4f}")
    return probe


def _masked_train_probes_mlp(train_feats, train_attrs, cfg):
    # Replaces tier3_hybrid.train_probes (DualScore Stage 1) — identical two-way split
    # and loop, masked loss + masked val-accuracy readout.
    import tier3_hybrid as th
    torch.manual_seed(cfg.seed)
    N = train_feats.shape[0]
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(cfg.seed))
    n_val = int(N * cfg.val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    probes = th.AttributeProbes(cfg.d_model, cfg.probe_hidden).to(th.DEVICE)
    opt = torch.optim.Adam(probes.parameters(), lr=cfg.probe_lr)

    Xtr, Ytr = train_feats[tr_idx], train_attrs[tr_idx].float()
    Xva, Yva = train_feats[val_idx].to(th.DEVICE), train_attrs[val_idx].float().to(th.DEVICE)
    Vva = Yva >= 0

    history = []
    for epoch in range(cfg.probe_epochs):
        probes.train()
        order = torch.randperm(Xtr.shape[0])
        running, seen = 0.0, 0
        for start in range(0, Xtr.shape[0], cfg.probe_batch):
            sel = order[start:start + cfg.probe_batch]
            xb, yb = Xtr[sel].to(th.DEVICE), Ytr[sel].to(th.DEVICE)
            opt.zero_grad()
            loss = _masked_bce(probes(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.shape[0]
            seen += xb.shape[0]
        history.append(running / max(seen, 1))

        probes.eval()
        with torch.no_grad():
            acc = ((probes(Xva) > 0).float() == Yva)[Vva].float().mean().item()
        th.log(f"  [probes] epoch {epoch + 1:02d}/{cfg.probe_epochs}  "
               f"BCE={history[-1]:.4f}  val_acc={acc:.4f}")

    return probes, history


def _masked_pair_counts(A: torch.Tensor) -> torch.Tensor:
    # Replaces tier5_cdbayes._pair_counts — pairwise-complete observations: a row counts
    # for pair (a, b) only if labeled (!= -1) on BOTH. Binary labels ⇒ bit-identical.
    M = (A != -1).float()
    A1 = (A == 1).float()
    c11 = A1.T @ A1
    counts = torch.empty(A.shape[1], A.shape[1], 2, 2)
    counts[:, :, 1, 1] = c11
    counts[:, :, 1, 0] = A1.T @ M - c11
    counts[:, :, 0, 1] = M.T @ A1 - c11
    counts[:, :, 0, 0] = M.T @ M - counts[:, :, 1, 1] - counts[:, :, 1, 0] - counts[:, :, 0, 1]
    return counts


def _masked_mutual_information_matrix(A: torch.Tensor, alpha: float) -> torch.Tensor:
    # Replaces tier5_cdbayes.mutual_information_matrix — same MI, but each pair is
    # normalized by ITS OWN jointly-labeled total (= N for binary ⇒ identical).
    counts = _masked_pair_counts(A)
    n_pair = counts.sum(dim=(2, 3), keepdim=True)
    joint = (counts + alpha) / (n_pair + 4 * alpha)
    ma = joint.sum(dim=3)
    mb = joint.sum(dim=2)
    mi = (joint * (joint.log() - ma.unsqueeze(3).log() - mb.unsqueeze(2).log())).sum(dim=(2, 3))
    mi.fill_diagonal_(0.0)
    return mi


def _masked_build_chow_liu_tree(A: torch.Tensor, alpha: float):
    # Replaces tier5_cdbayes.build_chow_liu_tree — same MST/CPT construction over the
    # masked sufficient statistics; marginals over each attribute's own labeled count.
    import tier5_cdbayes as t5
    A = A.float().cpu()
    n = A.shape[1]
    counts = _masked_pair_counts(A)
    mi = _masked_mutual_information_matrix(A, alpha)
    parent = t5.maximum_spanning_tree(mi)
    order = t5._topological_order(parent)

    n_lab = (A != -1).float().sum(0)
    n1 = (A == 1).float().sum(0)
    log_marginal = torch.stack(
        [(n_lab - n1 + alpha), (n1 + alpha)], dim=1).div((n_lab + 2 * alpha).unsqueeze(1)).log()
    log_cpt = torch.zeros(n, 2, 2)
    for c, p in enumerate(parent):
        if p < 0:
            continue
        pair = counts[p, c] + alpha
        log_cpt[c] = (pair / pair.sum(dim=1, keepdim=True)).log()
    return t5.DependencyTree(tuple(parent), tuple(order), log_cpt, log_marginal, mi)


def _masked_fit_temperature_scalars(logits: torch.Tensor, labels: torch.Tensor,
                                    steps: int, lr: float) -> torch.Tensor:
    # Replaces tier5_cdbayes.fit_temperature_scalars — validity-weighted NLL; each τ's
    # term is scaled independently, so masking never shifts another attribute's fit.
    w = (labels >= 0).float()
    y = labels.clamp(min=0)
    log_tau = torch.zeros(logits.shape[1], device=logits.device, requires_grad=True)
    opt = torch.optim.Adam([log_tau], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits / log_tau.exp(), y, weight=w)
        loss.backward()
        opt.step()
    return log_tau.detach().exp()


def _masked_expected_calibration_error(p: torch.Tensor, y: torch.Tensor,
                                       n_bins: int) -> torch.Tensor:
    # Replaces tier5_cdbayes.expected_calibration_error — -1 entries carry zero weight
    # in bin counts, gaps, and the per-attribute denominator.
    w = (y >= 0).float()
    y = y.clamp(min=0)
    ece = torch.zeros(p.shape[1], device=p.device)
    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=p.device)
    for b in range(n_bins):
        in_bin = ((p > edges[b]) & (p <= edges[b + 1]) if b else (p <= edges[1])).float() * w
        n_b = in_bin.sum(0)
        gap = ((y * in_bin).sum(0) - (p * in_bin).sum(0)).abs()
        ece += torch.where(n_b > 0, gap / w.sum(0).clamp(min=1), torch.zeros_like(gap))
    return ece


def _make_tristate_query_generator():
    # Deferred import: tier3_dgp pulls the whole CelebA stack at import time; only the
    # trained-tier configure functions pay that cost.
    import tier3_dgp
    from ovad.generate_gt import HAMMING_RATE, _masked_hamming_rate_batch

    class TriStateQueryGenerator(tier3_dgp.QueryGenerator):
        # OVAD synthetic-query sampler — tri-state semantics on every mask:
        #   query axes only where the reference is LABELED (==1 has / ==0 lacks, never -1);
        #   candidate constraints ==1 / ==0 (a -1 candidate proves nothing -> excluded);
        #   positives by the masked relative-Hamming RATE <= 0.05 on free attrs (the exact
        #   GT rule, replacing the integer radius that doesn't transfer to 117 attrs);
        #   hard negatives labeled-1 on a violated negative attribute.
        def sample_query(self):
            r = self._rand(self.N)
            has_idx = (self.A[r] == 1).nonzero(as_tuple=True)[0].tolist()
            lacks_idx = (self.A[r] == 0).nonzero(as_tuple=True)[0].tolist()
            if not has_idx or not lacks_idx:
                return None

            k_pos = 1 + self._rand(self.cfg.k_pos_max)
            k_neg = self._rand(self.cfg.k_neg_max + 1)
            pos_attrs = self._choice(has_idx, min(k_pos, len(has_idx)))
            neg_attrs = self._choice(lacks_idx, min(k_neg, len(lacks_idx)))

            sat = torch.ones(self.N, dtype=torch.bool)
            for a in pos_attrs:
                sat &= self.A[:, a] == 1
            for b in neg_attrs:
                sat &= self.A[:, b] == 0

            constrained = set(pos_attrs) | set(neg_attrs)
            free = torch.tensor([c for c in range(self.n_attr) if c not in constrained],
                                dtype=torch.long)
            rate = _masked_hamming_rate_batch(self.A[r].unsqueeze(0)[:, free],
                                              self.A[:, free]).squeeze(0)      # [N]
            pos_mask = sat & (rate <= HAMMING_RATE)
            pos_mask[r] = False
            pos_ids = pos_mask.nonzero(as_tuple=True)[0]
            if pos_ids.numel() < 1:
                return None

            sat_pos = torch.ones(self.N, dtype=torch.bool)
            for a in pos_attrs:
                sat_pos &= self.A[:, a] == 1
            viol_neg = torch.zeros(self.N, dtype=torch.bool)
            for b in neg_attrs:
                viol_neg |= self.A[:, b] == 1
            hard_mask = sat_pos & viol_neg
            hard_mask[r] = False
            hard_ids = hard_mask.nonzero(as_tuple=True)[0]
            if neg_attrs and hard_ids.numel() < 1:
                return None

            pos_sel = self._take(pos_ids, self.cfg.n_pos_targets)
            hard_sel = self._take(hard_ids, self.cfg.n_hard_neg) if hard_ids.numel() else hard_ids
            pos_names = [OVAD_NAMES[a] for a in pos_attrs]
            neg_names = [OVAD_NAMES[b] for b in neg_attrs]
            return r, pos_names, neg_names, pos_sel, hard_sel

    return TriStateQueryGenerator


def configure_dualscore() -> None:
    # Bindings for the tier3 DualScore path (tier3_hybrid + tier3_dgp + tier2d_dgp deps).
    # MUST run before importing tier3_dualscore, so its by-value imports see these bindings.
    import tier1_GDE
    import tier2d_dgp
    import tier3_dgp
    import tier3_hybrid

    _bind_tables(tier1_GDE, tier2d_dgp, tier3_dgp, tier3_hybrid)
    _bind_train_loaders(tier1_GDE, tier3_hybrid)
    tier1_GDE._get_artifacts_dir = _get_artifacts_dir  # visual_directions cache → artifacts/ovad/
    tier2d_dgp.build_prompts_for_attribute = _single_arg_prompt_builder()
    # Rebind on every module holding a by-value copy (their imports already ran above).
    qg = _make_tristate_query_generator()
    tier3_dgp.QueryGenerator = qg
    tier3_hybrid.QueryGenerator = qg
    tier3_hybrid.train_probes = _masked_train_probes_mlp   # tri-state Stage 1
    tier3_hybrid.load_image_features = lambda: ovad_data.load_image_features("test")

    # AttributeProbes defaults n_attr=40 and train_probes constructs it without the arg;
    # subclass with the OVAD default so every construction site gets 117 heads.
    _CelebAProbes = tier3_hybrid.AttributeProbes

    class OvadAttributeProbes(_CelebAProbes):
        def __init__(self, d_model, hidden, n_attr=N_ATTRS):
            super().__init__(d_model, hidden, n_attr=n_attr)

    tier3_hybrid.AttributeProbes = OvadAttributeProbes


def configure_tier5() -> None:
    # Bindings for the tier5 Bprime_listnet path (tier4_bayes + tier5_cdbayes + deps).
    import tier1_GDE
    import tier3_dgp
    import tier4_bayes
    import tier5_cdbayes as t5
    import vm_pipeline

    from ovad.generate_gt import query_indices as ovad_query_indices

    _bind_tables(tier3_dgp, tier4_bayes, t5, vm_pipeline)
    for mod in (tier4_bayes, t5):
        mod.N_ATTR = N_ATTRS
        mod._get_artifacts_dir = _get_artifacts_dir   # probe/temperature caches → artifacts/ovad/
    _bind_train_loaders(tier1_GDE, tier4_bayes, t5)
    # Rebind on every module holding a by-value copy (their imports already ran above).
    qg = _make_tristate_query_generator()
    tier3_dgp.QueryGenerator = qg
    tier4_bayes.QueryGenerator = qg
    t5.QueryGenerator = qg

    # Tri-state label consumers → masked variants (module-global lookups inside
    # load_or_train_probes / load_or_fit_temperatures / the tree builder see these).
    tier4_bayes.train_probes = _masked_train_probes_linear
    t5._pair_counts = _masked_pair_counts
    t5.mutual_information_matrix = _masked_mutual_information_matrix
    t5.build_chow_liu_tree = _masked_build_chow_liu_tree
    t5.fit_temperature_scalars = _masked_fit_temperature_scalars
    t5.expected_calibration_error = _masked_expected_calibration_error

    tier4_bayes.load_image_features = lambda: ovad_data.load_image_features("test")

    # Benchmark query strings use underscore-normalized names with ':' and '/' — the
    # CelebA query_indices lookup would KeyError; OVAD's own parser owns that mapping.
    vm_pipeline.query_indices = ovad_query_indices
    t5.query_indices = ovad_query_indices

    def _ovad_probe_diag(P_test):
        # Masked accuracy: only labeled (-1-free) entries count.
        attrs = ovad_data.load_attributes("test")
        valid = attrs != -1
        correct = (P_test.cpu() > 0.5) == (attrs == 1)
        acc = ((correct & valid).sum(0).float() / valid.sum(0).clamp(min=1))
        worst = int(acc.argmin())
        tier4_bayes.log(f"  probe test acc (masked): mean {acc.mean():.4f}  "
                        f"min {acc[worst]:.4f} ({OVAD_NAMES[worst]})")

    tier4_bayes.probe_accuracy_diagnostic = _ovad_probe_diag
