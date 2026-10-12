# Literature

- **[SOTA.md](SOTA.md)**: the curated state-of-the-art review written for the project (Davide Cabitza, June 19 – July 2, 2026), grouped by the role each paper plays.
- **Reading list (below)**: the papers we studied in depth. During the project we kept full-text copies in `documents/papers/` for close reading. Those are other authors' work, so they are not published in this repository. Follow the links instead.

## Papers studied in depth

| Paper | Link | How it was used in this project |
|---|---|---|
| Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), ICML 2021 | [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) | The frozen ViT-B/32 backbone that every method is built on. |
| Lim et al., *CLAY: Conditional Visual Similarity Modulation in Vision-Language Embedding Space*, CVPR 2026 | [arXiv:2604.11539](https://arxiv.org/abs/2604.11539) | Assigned paper. Its naïve pre-SVD stacking of conditions is the bottleneck the fusion module Φ replaces. Our reproduction scored below the vanilla baseline. |
| Berasi et al., *Not Only Text: Exploring Compositionality of Visual Representations in Vision-Language Models* (GDE), CVPR 2025 | [arXiv:2503.17142](https://arxiv.org/abs/2503.17142) | Assigned paper. Geodesic composition on the sphere, the basis of Track V / GDE-VP. |
| Liang et al., *Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning*, NeurIPS 2022 | [arXiv:2203.02053](https://arxiv.org/abs/2203.02053) | Motivated modality-gap centering, the largest training-free gain (+63% R@5). |
| Trager et al., *Linear Spaces of Meanings: Compositional Structures in Vision-Language Models*, ICCV 2023 | [arXiv:2302.14383](https://arxiv.org/abs/2302.14383) | The flat (linear) special case of GDE, used as the LDE ablation. |
| Kim et al., *Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors* (TCAV), ICML 2018 | [arXiv:1711.11279](https://arxiv.org/abs/1711.11279) | Justifies mining an attribute as a direction from has-X vs not-X images (Track V). |
| Oldfield et al., *Parts of Speech–Grounded Subspaces in Vision-Language Models*, NeurIPS 2023 | [arXiv:2305.14053](https://arxiv.org/abs/2305.14053) | Projection and orthogonal-complement operators behind subspace negation (Track S, Subspace Union). |
| Alhamoud et al., *Vision-Language Models Do Not Understand Negation* (NegBench), CVPR 2025 | [arXiv:2501.09425](https://arxiv.org/abs/2501.09425) | Documents CLIP's affirmation bias, the cause of the 0.000 floor on `-Male, -Mustache`. Its discriminative-data idea motivated the probe channel of Dual Score. |
| Kazemi Ranjbar et al., *SpaceVLM: Sub-Space Modeling of Negation in Vision-Language Models*, 2025 | [arXiv:2511.12331](https://arxiv.org/abs/2511.12331) | Negation as a region (subspace) rather than a point; informed Track S's complement scoring. |
| Sammani et al., *When Negation Is a Geometry Problem in Vision-Language Models*, 2026 | [arXiv:2603.20554](https://arxiv.org/abs/2603.20554) | Evidence that a linear negation direction exists in CLIP space. |
| van den Oord et al., *Representation Learning with Contrastive Predictive Coding*, 2018 | [arXiv:1807.03748](https://arxiv.org/abs/1807.03748) | The InfoNCE loss used to train the first fusion modules (Φ-Gate). |

The probabilistic methods (BAP, BAP+, A-BAP) also draw on calibration (Guo et al., 2017), Chow–Liu trees (1968) and ListNet (Cao et al., 2007). These are cited in the [README](../../README.md#references) and in [process/14_NextStep.md](../process/14_NextStep.md).
