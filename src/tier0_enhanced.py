"""
Tier-0 ENHANCED — vanilla latent arithmetic, but with correct CLIP geometry.

Same family as tier0.py (no SVD, no subspaces, no learning), but fixes three purely
*geometric* mistakes the naive baseline makes. Each fix is a mean-subtraction or a
renormalization — defensible as "still vanilla latent arithmetic, done right" — so
Tier-1 (CLAY) and Tier-2 keep all their headroom.

    Naive (tier0.py):
        q = normalize( v_ref + alpha * ( Σ t(+a) − Σ t(−a) ) )

    Enhanced (this file), with all three fixes on:
        t_bank(a) = normalize( mean_k bank[a, k] )        # FIX 3: prompt ensembling
        t_hat(a)  = normalize( t_bank(a) − μ_txt )         # FIX 1: de-bias modality gap
        d         = normalize( Σ t_hat(+a) − Σ t_hat(−a) ) # FIX 2: unit delta direction
        q         = normalize( (v_ref − μ_img) + alpha * d )  (+ μ_img re-added)

The three fixes (all training-free, all well-cited):
  FIX 1  Modality-gap centering (Liang et al. 2022, "Mind the Gap"). Image and text
         embeddings live in two separated cones; the common text mean μ_txt is a
         constant offset carrying NO attribute signal. Subtract it so alpha buys
         attribute direction, not "text-ness". (toggle: center)
  FIX 2  Delta normalization. ||Σt+ − Σt−|| scales with constraint COUNT, so a fixed
         alpha interpolates a different angular distance per query. Normalize the delta
         so alpha is one honest angle on the sphere for all queries. (toggle: norm_delta)
  FIX 3  Prompt ensembling (Radford et al. 2021, CLIP). One sentence is a noisy estimate
         of the attribute direction; the mean of n paraphrases (the prompt bank) has
         lower variance. NOT SVD — we keep the centroid, discard the spread. (toggle:
         use_prompt_bank)

ORDER MATTERS: ensemble the bank FIRST, then subtract μ_txt, then normalize. Averaging
paraphrases does not remove the common-mode modality offset — that is what FIX 1 is for.

CSV output matches tier0.py exactly, so output/tier0_enhanced_*.csv drops straight into
the same comparison as tier0_alpha1.0.csv and tier0_promptbank_alpha1.0.csv.

Run:  python src/tier0_enhanced.py
"""

import torch

from data_loader import ATTR_TO_IDX, _get_artifacts_dir
from eval import parse_query, evaluate_all, format_results_table, load_eval_json, find_eval_json
from results_saver import save_results_csv, output_subdir


# ---------------------------------------------------------------------------
# Direct cached-tensor loaders.
#
# We deliberately do NOT import clip_features / clip_prompts: those modules import
# `transformers` at top level (needed only to *extract* features), which need not be
# installed wherever we merely *score* from the pre-built .pt cache. Loading the cached
# tensors here keeps tier0_enhanced.py runnable on any machine that has the artifacts.
# Shapes/normalization are guaranteed by the extraction step that wrote them.
# ---------------------------------------------------------------------------
def load_image_features():
    # [N, 512] L2-normalized CLIP image table (artifacts/clip_image_features_test.pt).
    return torch.load(_get_artifacts_dir() / "clip_image_features_test.pt")


def load_attribute_text_features():
    # [40, 512] L2-normalized single-prompt attribute table (row j == attribute j).
    return torch.load(_get_artifacts_dir() / "clip_attr_text_features.pt")


def load_prompt_bank():
    # [40, n, 512] L2-normalized per-attribute prompt bank (bank[j] == attribute j stack).
    return torch.load(_get_artifacts_dir() / "clip_attr_prompt_bank.pt")


def build_attr_directions(use_prompt_bank=True, center=True):
    # Per-attribute unit-direction table — applies FIX 3 then FIX 1, in that order.
    # t̂(a) = normalize( normalize(meanₖ bank[a,k]) − μ_txt ).
    # FIX 3 ensembles paraphrases (lower-variance direction); FIX 1 strips the
    # common-mode text offset μ_txt. Returns ([40,512] unit dirs, μ_img) so the
    # caller can center the image side with the SAME μ_img.
    # (Order is load-bearing: averaging paraphrases does not remove the modality
    # offset — that is FIX 1's job, and it must come after the ensemble.)
    image_features = load_image_features()                 # [N, 512] unit
    mu_img = image_features.mean(dim=0)                     # image-cone center

    if use_prompt_bank:
        bank = load_prompt_bank()                          # [40, n, 512] unit rows
        # FIX 3: mean over paraphrases, then renormalize -> one direction per attr.
        attr = torch.nn.functional.normalize(bank.mean(dim=1), p=2, dim=1)  # [40, 512]
    else:
        attr = load_attribute_text_features().clone()      # [40, 512] unit (single prompt)

    if center:
        # FIX 1: mu_txt is the common-mode text offset shared by every attribute. It is
        # the centroid of the SAME directions we use downstream, so the de-biased space
        # is self-consistent (subtract the mean of exactly what we add/subtract later).
        mu_txt = attr.mean(dim=0)
        attr = torch.nn.functional.normalize(attr - mu_txt, p=2, dim=1)

    return attr, mu_img


def score(
    T_pos, T_neg, v_ref_idx, image_features, attr_directions, mu_img,
    alpha=1.0, beta=None, norm_delta=True, center=True,
):
    # Tier-0 ENHANCED scorer — vanilla latent arithmetic in CORRECT CLIP geometry.
    # q = normalize( (v_ref − μ_img) + α·d + μ_img ),  d = normalize(Σ t̂⁺ − Σ t̂⁻).
    # Three independent toggles over the naive baseline: FIX 1 (center) does the
    # arithmetic in the image-centered frame, FIX 2 (norm_delta) makes α one honest
    # angle regardless of constraint count, FIX 3 lives upstream in attr_directions.
    # Returns dataset indices best-first, source excluded. β defaults to α (symmetric).
    if beta is None:
        beta = alpha

    v_ref = image_features[v_ref_idx].clone()

    # Build the +/- delta from the (already corrected) attribute directions.
    delta = torch.zeros_like(v_ref)
    for name in T_pos:
        delta += attr_directions[ATTR_TO_IDX[name]]
    for name in T_neg:
        delta -= attr_directions[ATTR_TO_IDX[name]]

    # FIX 2: make alpha an honest angle — strip the constraint-count-dependent magnitude.
    if norm_delta and delta.norm() > 1e-8:
        delta = torch.nn.functional.normalize(delta, p=2, dim=0)

    # FIX 1 (image side): do the arithmetic in the centered frame, then re-add mu_img so
    # the query lands back in the image cone where the database lives.
    if center:
        query = (v_ref - mu_img) + alpha * delta + mu_img
    else:
        query = v_ref + alpha * delta

    # Unit-normalize so the corpus dot product IS cosine similarity.
    query = torch.nn.functional.normalize(query, p=2, dim=0)
    scores = image_features @ query

    scores[v_ref_idx] = float("-inf")        # never rank a source against itself

    return torch.argsort(scores, descending=True).tolist()


def make_get_ranking(
    query_str, image_features, attr_directions, mu_img,
    alpha=1.0, beta=None, norm_delta=True, center=True,
):
    # Curry one parsed query into the get_ranking(src_idx) callback eval expects.
    T_pos, T_neg = parse_query(query_str)
    return lambda src_idx: score(
        T_pos, T_neg, src_idx, image_features, attr_directions, mu_img,
        alpha=alpha, beta=beta, norm_delta=norm_delta, center=center,
    )


def evaluate_enhanced(
    alpha=1.0, beta=None, ks=(1, 5, 10), save=True, tag=None,
    use_prompt_bank=True, center=True, norm_delta=True,
):
    # Run enhanced Tier-0 over the full benchmark: print table, save CSV.
    # The three fixes are independent toggles so each ablates cleanly:
    # use_prompt_bank (FIX 3), center (FIX 1), norm_delta (FIX 2). `tag`
    # overrides the CSV suffix (defaults to an auto fix-flag string).
    image_features = load_image_features()
    attr_directions, mu_img = build_attr_directions(use_prompt_bank=use_prompt_bank, center=center)
    gt_list = load_eval_json(find_eval_json())

    def make(query_str):
        return make_get_ranking(
            query_str, image_features, attr_directions, mu_img,
            alpha=alpha, beta=beta, norm_delta=norm_delta, center=center,
        )

    results = evaluate_all(gt_list, make, ks=ks)

    flags = f"bank={int(use_prompt_bank)} center={int(center)} normdelta={int(norm_delta)}"
    print(f"\nTier-0 ENHANCED (alpha={alpha}, beta={beta if beta is not None else alpha}; {flags}) "
          f"- {len(gt_list)} queries\n")
    print(format_results_table(results, ks=ks))

    if save:
        if tag is None:
            tag = f"bank{int(use_prompt_bank)}_center{int(center)}_normdelta{int(norm_delta)}_alpha{alpha}"
        save_results_csv(results, output_subdir("tier0_enhanced") / f"tier0_enhanced_{tag}.csv", ks=ks)
    return results


def _mean_recall(results, k=5):
    return sum(r[f"recall@{k}"] for r in results.values()) / len(results)


if __name__ == "__main__":
    # Full ablation: naive-equivalent -> add fixes one at a time -> all three on.
    # Each run writes its own CSV; the console summary shows the marginal R@5 gain.
    print("=" * 70)
    print("Tier-0 ENHANCED — fix ablation (all CSVs land in output/)")
    print("=" * 70)

    configs = [
        # tag,                 bank,  center, norm_delta   -- what it isolates
        ("naive",              False, False, False),   # ~ reproduces tier0.py
        ("fix3_bank",          True,  False, False),   # + prompt ensembling
        ("fix1_center",        False, True,  False),   # + modality-gap centering
        ("fix2_normdelta",     False, False, True),    # + delta normalization
        ("all_fixes",          True,  True,  True),    # all three together
    ]

    summary = []
    for tag, bank, center, normd in configs:
        res = evaluate_enhanced(
            alpha=1.0, use_prompt_bank=bank, center=center, norm_delta=normd, tag=tag,
        )
        summary.append((tag, _mean_recall(res, 5)))

    print("\n" + "=" * 70)
    print("SUMMARY — mean Recall@5 by configuration")
    print("=" * 70)
    base = summary[0][1]
    for tag, r5 in summary:
        delta = r5 - base
        print(f"  {tag:18s}  R@5 = {r5:.4f}   ({'+' if delta >= 0 else ''}{delta:.4f} vs naive)")
