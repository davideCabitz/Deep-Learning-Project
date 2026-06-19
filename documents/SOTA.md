# State of the Art — Dynamic & Hybrid Conditioning for Compositional Image Retrieval

> Curated literature for the project's "Related Work / State of the Art" section.
> Grouped by the role each paper plays relative to *our* goal: a dynamic +/−
> fusion module Φ on top of frozen CLIP ViT-B/32, for compositional retrieval on
> CelebA. ⭐ marks the must-cite backbone.
>
> **Suggested write-up structure (funnel):** broad foundations (§1) → compositional
> structure in the latent space (§2) → the retrieval task itself (§3, CIR) → the two
> pieces we innovate on (§4 fusion + §5 negation), positioning CLAY/GDE as the
> immediate prior work we extend and the negation papers as the gap we fill.

---

## 1. Foundations (frozen backbone + the core problem)

- ⭐ **CLIP — Learning Transferable Visual Models From Natural Language Supervision**
  (Radford et al., 2021) — https://arxiv.org/abs/2103.00020
  Our frozen model. Cite for the shared image-text embedding space and the
  contrastive pretraining objective everything else builds on.

- ⭐ **Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive
  Learning** (Liang et al., 2022) — https://arxiv.org/abs/2203.02053
  *The* justification for why naïve `v_ref + t` arithmetic is geometrically weak,
  and why CLAY needs a rotation H. Central to our motivation. (Code:
  https://github.com/Weixin-Liang/Modality-Gap)

- **Explaining and Mitigating the Modality Gap in Contrastive Multimodal Learning**
  (2024) — https://openreview.net/pdf?id=2sThreW73a
  More recent, deeper treatment of the gap; optional second citation.

## 2. Compositionality in the embedding space (direct lineage of our assigned papers)

- ⭐ **Linear Spaces of Meanings: Compositional Structures in VLMs** (Trager et al.,
  ICCV 2023) — https://arxiv.org/abs/2302.14383
  The "ideal words" paper: composite concepts as linear combinations in CLIP text
  space. GDE explicitly generalizes this from linear to manifold geometry — we
  must position our work relative to it.

- ⭐ **Not Only Text: Exploring Compositionality of Visual Representations in VLMs
  (GDE)** (Berasi et al., CVPR 2025) —
  https://openaccess.thecvf.com/content/CVPR2025/papers/Berasi_Not_Only_Text_Exploring_Compositionality_of_Visual_Representations_in_Vision-Language_CVPR_2025_paper.pdf
  One of the two assigned papers (local copy in `documents/`). Log-map / tangent-space
  geodesic decomposition of visual embeddings; the geometric backbone we build on.

- **CLAY: Conditional visual similarity modulation in vision-language embedding
  space** (Lim et al., CVPR 2026) — *no public link found via search; cite from the
  copy provided with the assignment.*
  The method-to-beat: reframes CLIP space as a text-conditional similarity space,
  uses SVD over a textual subspace + rotation H. Its naïve multi-condition stacking
  (pre-SVD bottleneck) is exactly what we attack.

- **Parts of Speech–Grounded Subspaces in Vision-Language Models** (Oldfield et al.,
  NeurIPS 2023) — https://arxiv.org/abs/2305.14053
  Principal Geodesic Analysis to learn lower-dimensional submanifolds of the CLIP
  sphere. Directly relevant to our SVD-subspace projection (Tier 1 / Tier 2a).

- **SpLiCE: Interpreting CLIP with Sparse Linear Concept Embeddings** (Bhalla et al.,
  2024) — https://arxiv.org/abs/2402.10376
  Decomposes CLIP embeddings into sparse, human-interpretable concept directions.
  Useful framing if we discuss attribute directions / interpretability.

## 3. Composed / Compositional Image Retrieval — the task family (CIR)

The field our task belongs to; the spec just adds explicit negatives and identity
preservation.

- ⭐ **A Comprehensive Survey on Composed Image Retrieval** (2025) —
  https://arxiv.org/abs/2502.18495
  Best single entry point. Cite to frame the field and to mine further references.

- **Composing Text and Image for Image Retrieval (TIRG)** (Vo et al., CVPR 2019) —
  https://arxiv.org/abs/1812.07119
  Origin of the "image + text modification → target" task. Historical anchor.

- ⭐ **Effective Conditioned and Composed Image Retrieval (Combiner)** (Baldrati et
  al., CVPR 2022) —
  https://openaccess.thecvf.com/content/CVPR2022/papers/Baldrati_Effective_Conditioned_and_Composed_Image_Retrieval_Combining_CLIP-Based_Features_CVPR_2022_paper.pdf
  A trained network that fuses CLIP image+text features — direct precedent for our
  trained Φ.

- ⭐ **Pic2Word: Mapping Pictures to Words for Zero-Shot Composed Image Retrieval**
  (Saito et al., CVPR 2023) — https://arxiv.org/abs/2302.03084
  Textual-inversion zero-shot baseline; maps an image to a pseudo-word token.

- ⭐ **Zero-Shot Composed Image Retrieval with Textual Inversion (SEARLE)** (Baldrati
  & Agnolucci et al., ICCV 2023) — https://arxiv.org/abs/2303.15247
  (Code: https://github.com/miccunifi/SEARLE)
  Two-stage optimization + distillation textual inversion. Key zero-shot reference.
  - **iSEARLE** (extension, 2024) — https://arxiv.org/abs/2405.02951

- ⭐ **Vision-by-Language for Training-Free Compositional Image Retrieval (CIReVL)**
  (Karthik et al., ICLR 2024) — https://arxiv.org/abs/2310.09291
  (Code: https://github.com/ExplainableML/Vision_by_Language)
  Training-free VLM+LLM pipeline (caption → edit caption → retrieve). Strong
  training-free baseline; good contrast to our trained Φ.

## 4. Fusion mechanisms (architectural options for Φ)

- **FiLM: Visual Reasoning with a General Conditioning Layer** (Perez et al., 2018) —
  https://arxiv.org/abs/1709.07871
  The gating/FiLM variant named in our roadmap (per-condition feature modulation).

- **Attention Is All You Need** (Vaswani et al., 2017) —
  https://arxiv.org/abs/1706.03762
  For the cross-attention Φ over `[v_ref, t⁺…, t⁻…]` tokens.

- **Cross-modal Feature Alignment and Fusion for Composed Image Retrieval (CAFF)**
  (CVPR 2024 Workshop) —
  https://openaccess.thecvf.com/content/CVPR2024W/CVFAD/papers/Wan_Cross-modal_Feature_Alignment_and_Fusion_for_Composed_Image_Retrieval_CVPRW_2024_paper.pdf
  Recent cross-attention late-fusion design to compare against.

## 5. Negation in VLMs — our "−" constraints are the gradeable insight (study carefully)

- ⭐ **Vision-Language Models Do Not Understand Negation** (Alhamoud et al., 2025) —
  https://arxiv.org/abs/2501.09425
  Documents *why* "−X" is hard for CLIP. Directly motivates asymmetric +/− handling.

- **Know "No" Better: A Data-Driven Approach for Enhancing Negation Awareness in
  CLIP** (2025) — https://arxiv.org/abs/2501.10913
  Data-centric strategy (synthetic negated captions) to improve negation awareness.

- ⭐ **SpaceVLM: Sub-Space Modeling of Negation in Vision-Language Models** (2025) —
  https://arxiv.org/abs/2511.12331
  Models negation as subspace intersection, training-free — almost exactly our
  Tier-2a "orthogonal rejection" idea. Critical comparison point.

- **When Negation Is a Geometry Problem in Vision-Language Models** (2026) —
  https://arxiv.org/abs/2603.20554
  Newest; geometric framing of negation in the embedding space.

## 6. Compositionality benchmarks (to frame "why this is hard")

- **When and Why VLMs Behave like Bags-of-Words (ARO)** (Yuksekgonul et al., ICLR
  2023) — https://arxiv.org/abs/2210.01936
  Shows CLIP often ignores word order / attribute binding.

- **Winoground: Probing VLMs for Visio-Linguistic Compositionality** (Thrush et al.,
  CVPR 2022) — https://arxiv.org/abs/2204.03162

- **SugarCrepe: Fixing Hackable Benchmarks for Vision-Language Compositionality**
  (Hsieh et al., NeurIPS 2023) — https://arxiv.org/abs/2306.14610

## 7. Training objective (for the trained Φ)

- **Representation Learning with Contrastive Predictive Coding (InfoNCE)** (van den
  Oord et al., 2018) — https://arxiv.org/abs/1807.03748
  The contrastive loss family for training Φ.

- **A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)**
  (Chen et al., 2020) — https://arxiv.org/abs/2002.05709
  Contrastive-loss design and hard-negative intuition.

## 8. Stretch / interpretability angle (only if pursuing the SAE direction)

- **Interpreting CLIP with Hierarchical Sparse Autoencoders** (2025) —
  https://arxiv.org/abs/2502.20578

- **Steering CLIP's Vision Transformer with Sparse Autoencoders** (2025) —
  https://arxiv.org/abs/2504.08729

---

### Notes

- **CLAY** (Lim et al., CVPR 2026) has no public link surfaced via search — it is the
  very recent paper provided directly with the assignment. Cite it from the local
  copy; everything in §2–§3 is the literature it builds on.
- For more references, GDE's own related-work section and the CIR survey (§3) are the
  fastest sources.
