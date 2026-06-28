# SpaceVLM: Sub-Space Modeling of Negation in Vision-Language Models

**Authors:** Sepehr Kazemi Ranjbar*¹, Kumail Alhamoud*², Marzyeh Ghassemi²

**Affiliations:** ¹Independent Researcher · ²MIT

**Contacts:** sepehrkazemi9@gmail.com · kumail@mit.edu · mghassem@mit.edu

*Equal contribution.

**Preprint:** arXiv:2511.13231v1 [cs.CV], 15 November 2025

---

## Abstract

Vision-Language Models (VLMs) struggle with negation. Given a prompt like "retrieve (or generate) a street scene without pedestrians," they often fail to respect the "not." Existing methods address this limitation by fine-tuning on large negation datasets, but such retraining often compromises the model's zero-shot performance on affirmative prompts.

We show that the embedding space of VLMs, such as CLIP, can be divided into semantically consistent subspaces. Based on this property, we propose a training-free framework that models negation as a subspace in the joint embedding space rather than as a single point. To find the matching image for a caption such as "A but not N," we construct two spherical caps around the embeddings of A and N, and we score images by the central direction of the region that is close to A and far from N.

Across retrieval, MCQ, and text-to-image tasks, our method improves negation understanding by about 30% on average over prior methods. It closes the gap between affirmative and negated prompts while preserving the zero-shot performance that fine-tuned models fail to maintain. Code will be released upon publication.

---

## 1. Introduction

Joint embedding-based Vision-Language Models (VLMs) [15, 26, 41], such as CLIP [26], have become strong foundations for visual understanding. These models consist of an image encoder and a text encoder that map visual and textual inputs into a shared embedding space, where similarity is measured by dot product. When pretrained on massive image–text datasets, they exhibit strong generalization and are widely used for classification, retrieval, and text-to-image generation [12, 42, 46], with successful applications in specialized domains such as healthcare [21].

However, they struggle with inputs that require logical reasoning [10, 13, 27], particularly those involving negation [1, 30]. Consider the query "retrieve an image with a dog but not a cat." A model processing this input must correctly exclude images containing cats, while retaining valid alternatives that include dogs. Yet, as shown by prior work [1], CLIP-like models [26, 30, 39, 41] fail to interpret negation in their standard inference setup. Previous studies attributed this weakness to the lack of negation-rich captions in the training data; to address this, they generated large synthetic datasets and fine-tuned VLMs on negation-enriched image–text pairs [25, 30, 39]. Yet these fine-tuning methods face two limitations: (i) they fail to fully close the performance gap between affirmative and negated queries, and (ii) they often reduce the model's zero-shot generalization on tasks unrelated to negation. This raises a central question: *can negation be modeled effectively without any fine-tuning?*

First, we motivate why fine-tuning alone cannot fully solve negation. The key issue is that "not a cat" excludes cat, but leaves open many alternatives, such as dog or apple. Representing this with a single embedding vector — following the dot-product scoring used in joint embedding-based VLMs such as CLIP [26], SigLIP [41], or LiT-tuned AIMV2 [7, 40] — is inherently insufficient. To account for infinitely many valid possibilities, negation cannot be modeled by a single point in the VLM embedding space.

In contrast, we verify that CLIP's embedding space can be divided into semantically consistent subspaces [4, 45]. We then model negation as the intersection between an affirmative and a complementary subspace, and derive a simple, training-free scoring rule. For a caption "A but not N," we compute two spherical caps centered at the embeddings of A and N, and use the central direction of the region that is close to A and far from N to score images. Because this scoring operates purely at inference time, it leaves the model's behavior unchanged on queries without negation, ensuring no degradation on unrelated tasks. Importantly, our **SpaceVLM** framework is model-agnostic and applicable to any joint embedding-based VLM.

We validate SpaceVLM across more than 40 experimental settings spanning combinations of VLM backbones, image and video datasets, and diverse negation tasks including multimodal retrieval, Multiple Choice Question (MCQ), and text-to-image generation. Following NegBench [1], we use the COCO [18], VOC-2007 [6], and MSR-VTT [38] datasets for general-domain retrieval and MCQ, and CheXpert [8] for medical diagnostics with negation. Our training-free framework consistently improves negation understanding for every joint embedding-based model tested — CLIP [26], SigLIP [35], NegCLIP [39], ConCLIP [30], AimV2 [7], BiomedCLIP [44], and others — while preserving zero-shot performance on affirmative queries. Despite requiring no training or architectural modification, SpaceVLM outperforms fine-tuned baselines such as CLIP-NegFull [1], ConCLIP [30], NegCLIP [39], and NegationCLIP [25], and it even surpasses the recent geometric approach DCSM [11].

Ablation studies show that the cosine-similarity threshold, the main hyperparameter in SpaceVLM, is robust within a practical range, making it easy to apply to new downstream applications. We also provide a visual inspection study to confirm that SpaceVLM retrieves diverse images consistent with negated prompts. We hope the effectiveness of this subspace perspective on VLM embeddings encourages future geometric methods for broader VLM logical reasoning tasks.

---

## 2. Related Work

### 2.1 Joint Embedding-based Vision-Language Models

Joint embedding-based VLMs align visual and textual representations in a shared embedding space. A representative example is CLIP [26], which trains an image encoder $I: x \to \mathbb{R}^d$ and a text encoder $T: y \to \mathbb{R}^d$ on 400 million image-caption pairs using a contrastive objective. The two encoders map inputs to the surface of a unit sphere, and image-text similarity is measured by what we call the **CLIP dot-product scoring** $I(x) \odot T(y)$. Given a caption $y$, the corresponding image is retrieved by:

$$\hat{x} = \arg\max_x \; I(x) \odot T(y).$$

CLIP's pretrained encoders are widely used across tasks, from multimodal retrieval [3, 22, 23] to multimodal LLMs [20, 36] and text-to-image generation [28, 31]. Several follow-up variants adopt similar principles: SigLIP [41] replaces the softmax contrastive loss with a sigmoid loss; AIMV2 [7] replaces the contrastive loss with a multimodal autoregressive loss, but its vision and text encoders can be aligned via Locked-Image Text Tuning [40], making it applicable to the CLIP dot-product scoring. We build on this family of models, improving their handling of negation at inference time without modifying their pretrained parameters.

### 2.2 Fine-tuning for Negation Understanding in VLMs

VLMs struggle with logical reasoning in prompts involving conjunction, disjunction, negation, contrast, comparison, condition, causality, and temporality [11, 14, 16, 24, 47]. Most relevant to this work is NegBench [1], which evaluates negation understanding via text-to-image retrieval and multiple-choice (MCQ) tasks. Most proposed solutions [25, 30, 39] address these problems by constructing logically rich datasets and fine-tuning VLMs on them. NegCLIP [39] fine-tunes CLIP to improve sensitivity to logical structure, including negation. Singh et al. [30] propose ConCLIP, a CLIP model finetuned for negation understanding, and Alhamoud et al. [1] extend this research by finetuning CLIP and NegCLIP on CC12M-NegFull, an extension of CC12M [5] with synthetically augmented negated captions.

While these methods improve negation performance, their reliance on fine-tuning has two drawbacks: (i) degraded zero-shot generalization, and (ii) the fundamental inability of joint embedding-based models to represent negation with a single embedding vector, regardless of the scale of fine-tuning data. Our zero-shot method eliminates both drawbacks by modeling negation geometrically, without any parameter updates.

### 2.3 Towards Training-free Solutions

Concurrent to our work, Kang et al. [11] note that joint embedding-based models cannot geometrically represent negation. They propose DCSM, a modification to the CLIP scoring function that retains all image patch embeddings and text token embeddings, computes cosine similarities across all pairs, and trains a convolutional projection head to aggregate this information. DCSM differs from our approach in two aspects. First, SpaceVLM explicitly models negation as a logical operation through the intersection of subspaces, whereas DCSM does not directly encode logical operators. Second, DCSM trains a lightweight scoring network on top of the frozen CLIP features for each dataset, while our method requires no additional training and operates entirely at inference time.

Few works address negation in text-to-image generation without fine-tuning [17, 37]. They use large language models to parse the negated prompt, construct an intermediate image layout, and feed it to Stable Diffusion with negative prompts to suppress excluded concepts. While effective for generative pipelines, these methods are orthogonal to our goal: they target a specific application and do not improve negation understanding in the underlying vision–language encoder. Other general work [25] also incorporates negation into text–image modeling, but it does so by fine-tuning the CLIP text encoder on specific datasets, which can introduce task-specific specialization and may affect generalization across other negation tasks, as we show in our experiments.

---

## 3. Method

We first state the key premise that motivates our approach.

> **Statement 1.** Negation cannot be modeled by a single point (vector) in the joint embedding VLM space.

*Proof sketch.* Let $I: x \to \mathbb{R}^d$ and $T: y \to \mathbb{R}^d$ be the CLIP image/text encoders with $\|I(x)\| = \|T(y)\| = 1$. We want to prove that there is no unit vector $n \in \mathbb{R}^d$ that separates cat from non-cat images with a positive margin under the CLIP dot-product scoring; i.e., there do not exist $\beta \in \mathbb{R}$ and $\delta > 0$ such that:

$$\inf_{x_{\text{non-cat}}} n \odot I(x) \geq \beta + \delta \quad \text{and} \quad \sup_{x_{\text{cat}}} n \odot I(x) \leq \beta.$$

Suppose for contradiction that such a unit $n$ and margin $\delta > 0$ exist. Then every non-cat image $x$ satisfies $n \odot I(x) \geq \beta + \delta$. Pick $m$ non-cat images with unit embeddings $u_1, \dots, u_m$ that are pairwise weakly correlated: $u_i \odot u_j \leq \gamma$ for $i \neq j$, for some $\gamma \geq 0$ (in high dimension we can choose $\gamma$ arbitrarily small by sampling unrelated classes). Summing the non-cat lower bound gives:

$$m(\beta + \delta) \leq \sum_{i=1}^m n \odot u_i = n \odot \left(\sum_{i=1}^m u_i\right) \leq \left\|\sum_{i=1}^m u_i\right\|.$$

By expanding the norm and using the pairwise bound:

$$\left\|\sum_{i=1}^m u_i\right\|^2 = \sum \|u_i\|^2 + 2\sum_{1 \leq i < j \leq m} u_i \odot u_j \leq m + \gamma m(m-1),$$

hence $m(\beta + \delta) \leq \sqrt{m + \gamma m(m-1)}$. Letting $\gamma \to 0$ yields $m(\beta + \delta) \leq \sqrt{m}$, i.e., $\beta + \delta \leq 1/\sqrt{m}$. As $m \to \infty$, this forces $\beta + \delta \leq 0$, contradicting $\delta > 0$. Thus, no unit vector $n$ can separate cat from non-cat with any positive margin under the CLIP dot-product scoring. $\blacksquare$

### 3.1 Empirical Divisibility of the Embedding Space

To model negation more effectively, we first examine the geometric structure of the CLIP embedding space. CLIP aligns images and captions by maximizing cosine similarity, and with $\ell_2$-normalized embeddings, all representations lie on the surface of a $d$-dimensional unit sphere. Empirically, embeddings that refer to the same visual concept (e.g., "dog") occupy compact, well-separated regions on this sphere [4, 45].

The distribution of pairwise cosine similarities between images within the same CIFAR-100 class, and similarities between the textual prompt "A photo of a \<class\>" and images of that class, both indicate that intra-class samples form high-similarity clusters that are distinct from other concepts. When such clusters are tight and sufficiently separated, we say that the space is *divisible*: a single cosine-similarity threshold can determine whether a new embedding belongs to a concept region or lies outside it. This divisibility property provides the geometric basis for our approach.

### 3.2 Problem Formulation

Given the divisibility property, we can represent complex textual compositions by reasoning over the regions induced by their constituent concepts. Consider the input:

$$P = \text{"A photo of } \langle a \rangle \text{ but not } \langle n \rangle\text{".}$$

We split $P$ into an affirmative part $P_a = \text{"A photo of } \langle a \rangle\text{"}$ and a negated part $P_n = \text{"A photo of } \langle n \rangle\text{"}$, so that:

$$P \equiv P_a + \text{"but not"} + P_n. \tag{1}$$

Let:

$$e_a = T(P_a), \quad e_n = T(P_n) \tag{2}$$

be the corresponding normalized text embeddings, and let $e_I$ be the normalized image embedding, all produced by CLIP. In standard CLIP inference, the image-text similarity is computed as the dot product $s = e_I \odot e_P$, where $e_P$ is the text embedding of the full caption $P$. However, when $P$ contains negation, this score becomes unreliable (Statement 1). Instead, our goal is to define a training-free scoring rule that leverages the compact regions around $e_a$ and $e_n$ to compute a more faithful score.

### 3.3 SpaceVLM: Sub-Space Modeling of Negation

We now define the training-free scoring rule that models negation as a subspace. We start with the affirmative and negated embeddings $e_a = T(P_a)$ and $e_n = T(P_n)$. In practice, a language processor such as a lightweight LLM is used to split input text into its affirmative and negative parts $P_a$ and $P_n$. Note that both $P_a$ and $P_n$ are phrased as affirmative captions. We denote the neighborhood (spherical cap) of a normalized point $x$ in the VLM space as:

$$\mathcal{N}(x) = \{z \in \mathbb{R}^d \mid x \odot z \geq t\}, \quad t \in [-1, 1],$$

where $t$ is a cosine-similarity threshold. We associate $P_a$ and $P_n$ with their subspaces $\mathcal{N}(e_a)$ and $\mathcal{N}(e_n)$. The target subspace for $P$ is the region that is close to the affirmative concept but outside the neighborhood of the negated one:

$$\mathcal{N}(P) = \mathcal{N}(e_a) \cap \mathcal{N}^c(e_n),$$

where $\mathcal{N}^c(e_n) = \{z \in \mathbb{R}^d \mid z \notin \mathcal{N}(e_n)\}$.

To perform image-text matching, we need a similarity score between an image embedding $e_I$ and this region $\mathcal{N}(P)$. Because embeddings lie on a unit sphere and cosine similarity is rotationally symmetric, a representative direction for $\mathcal{N}(P)$ provides a natural scoring vector. We choose the direction $\hat{d}$ at the angular "center" of the feasible region:

$$\hat{d} = \frac{\sin\!\left(\alpha + \frac{\theta}{2}\right)}{\sin(\theta)}\, e_a - \frac{\sin\!\left(\alpha - \frac{\theta}{2}\right)}{\sin(\theta)}\, e_n \tag{3}$$

where:

$$\alpha = \arccos(t), \quad \theta = \arccos(e_a \odot e_n). \tag{4}$$

Intuitively, $\theta$ is the angle between $e_a$ and $e_n$, and $\alpha$ defines the cap radius induced by the threshold $t$. The vector $\hat{d}$ points to the center of the intersection region $\mathcal{N}(e_a) \cap \mathcal{N}^c(e_n)$ along the great-circle arc joining $e_n$ and $e_a$.

The final score uses the standard CLIP dot-product form with this direction (optionally normalized):

$$\tilde{d} = \frac{\hat{d}}{\|\hat{d}\|}, \quad s_{\text{neg}}(e_I, P) = e_I \odot \tilde{d}. \tag{5}$$

**Algorithm 1** provides PyTorch-style pseudocode for SpaceVLM, computing negation-aware text embeddings for a generic VLM:

```python
# Inputs: caption, text_encoder, LLM, threshold t in [-1, 1]
# Output: d_hat — negation-aware embedding of input caption

# 1. Split into affirmative and negative parts     (Eq. 1)
aff_cap, neg_cap = LLM(caption)

# 2. Encode using the original VLM encoder          (Eq. 2)
e_a = text_encoder(aff_cap)
e_n = text_encoder(neg_cap)

# 3. Compute angular distances                      (Eq. 4)
alpha = arccos(threshold)
theta = arccos(dot_product(e_a, e_n))

# 4. Compute negation-aware embedding               (Eq. 3)
d_hat  = e_a * sin(alpha + theta/2) / sin(theta)
d_hat -= e_n * sin(alpha - theta/2) / sin(theta)

# 5. Normalize                                      (Eq. 5)
d_hat = d_hat / norm(d_hat)
```

---

## 4. Experiments

We evaluate the effectiveness of our approach for enhancing negation understanding across multiple VLMs.

### 4.1 Evaluation Protocol

**Tasks.** Following NegBench [1], we assess negation understanding on two tasks: (i) Image/Video Retrieval with negated queries, and (ii) Text Retrieval (MCQ) with negated captions. The negated retrieval task measures coarse-grained reasoning: given a negated query such as "A photo of a dog not on grass," the model must retrieve relevant images or videos. The MCQ task measures fine-grained reasoning: given an image, the model selects the correct caption among four closely related candidates drawn from Affirmation, Negation, and Hybrid templates. For medical VLMs, NegBench includes a simplified MCQ task providing a binary choice between negated and affirmative captions. We later extend this evaluation to a new text-to-image generation (T2I) task.

**Datasets.** For negated Image/Video Retrieval, we use the negated extensions of COCO [18] and MSR-VTT [38] provided by NegBench [1]. For MCQ, samples are drawn from COCO, VOC-2007 [6], and MSR-VTT. For MCQ in the medical domain, we use negated CheXpert [1, 8].

**Metrics.** For retrieval, we report Recall@K (R@K for $K \in \{1, 5, 10\}$), measuring the fraction of queries where at least one relevant image or video appears in the top-$K$ results. We report performance for both standard (affirmative) and negated queries. For MCQ, we report accuracy decomposed by the template of the correct answer (Affirmation, Negation, Hybrid), to expose performance gaps between affirmative and negated captions. For text-to-image generation, accuracy measures whether the generated image successfully excludes the object negated in the input prompt.

**Hyperparameters.** The similarity threshold $t$ is tuned per dataset on validation splits. Optimal $t$ values lie in $[0.90, 0.95]$ across all datasets, and performance is robust within this range, enabling simple hand-set choices of $t$ in new downstream applications without expensive tuning. To decompose input queries into affirmative and negated parts, we use a lightweight language processor based on Mistral-7B-v0.3 [9], fine-tuned on small subsets of COCO (Image Neg-Retrieval) and VOC-2007 (MCQ) [1]. This module does not modify the VLM and is used solely for query decomposition.

**Baselines.** We evaluate our method added to nine models spanning both pretrained and fine-tuned VLMs. Pretrained VLMs include CLIP [26], AIMV2 [7], and SigLIP-2 [35]. Fine-tuned variants include CLIP-NegFull (fine-tuned on CC12M-NegFull [1]), ConCLIP (fine-tuned on CC-Neg [30]), NegCLIP (fine-tuned on COCO with hard negative captions), and NegCLIP-NegFull (fine-tuned on CC12M-NegFull). For the medical MCQ task, we use BiomedCLIP [44]. We apply our training-free and model-agnostic method directly to each baseline and report results with and without our modification to isolate its effect. Unless otherwise stated, all models use the ViT-B/32 backbone for consistency.

### 4.2 Evaluation on NegBench

**Image/Video Retrieval.** SpaceVLM improves retrieval across all baselines and datasets, substantially closing the gap between affirmative and negated queries. In some cases, retrieval performance on negated queries even exceeds that of the base model on standard queries, as the additional negation information helps disambiguate similar images. Across all models and datasets, performance on affirmative queries remains unchanged, confirming that our scoring rule preserves the original model behavior on non-negated prompts. Representative average improvements on negated recall (pp) are:

| Dataset | CLIP | CLIP-NegFull | ConCLIP | NegCLIP | AIMV2 | SigLIP-2 |
|---------|------|-------------|---------|---------|-------|----------|
| COCO | +6.5% | +3.0% | +4.2% | +4.4% | +5.0% | +11.1% |
| MSR-VTT | +4.1% | +0.7% | +4.0% | +2.7% | +3.2% | +7.2% |

**MCQ.** Across all models and datasets, SpaceVLM achieves large gains, especially when the correct caption follows a Negation template. More surprisingly, it also improves accuracy when the correct caption follows an Affirmation template, because it reduces confusion with other templates. For example, vanilla CLIP maps both captions "a photo of a fish and coral" and "a photo of a fish but not coral" to nearly identical embeddings, which causes the model to select them interchangeably. With our geometric scoring, these captions become clearly separable. Notably, when applied to vanilla CLIP, our method outperforms several fine-tuned baselines trained specifically for negation understanding, such as CLIP-NegFull. MCQ average accuracy (AVG over Affirmation / Negation / Hybrid templates):

| Dataset | CLIP | +Ours | ConCLIP | +Ours | NegCLIP | +Ours | AIMV2 | +Ours | SigLIP-2 | +Ours |
|---------|------|-------|---------|-------|---------|-------|-------|-------|----------|-------|
| COCO | 39.2 | 66.3 (+27.1%) | 24.4 | 58.2 (+28.8%) | 26.8 | 64.2 (+37.4%) | 33.5 | 66.4 (+32.9%) | 29.8 | 63.6 (+33.8%) |
| VOC-2007 | 37.9 | 81.1 (+44.8%) | 38.2 | 79.5 (+41.3%) | 30.2 | 85.3 (+55.1%) | 31.4 | 76.3 (+44.9%) | 26.8 | 73.7 (+46.9%) |
| MSR-VTT | 31.6 | 58.0 (+27.6%) | 42.0 | 62.3 (+20.3%) | 25.8 | 62.5 (+36.7%) | 28.4 | 59.2 (+30.8%) | 27.2 | 59.3 (+32.1%) |

**Binary MCQs and medical VLMs.** We apply our method to improve the accuracy of BiomedCLIP given medical negations in the CheXpert MCQ task [1]:

| Model | CheXpert-Control | CheXpert-Negation |
|-------|-----------------|-------------------|
| BiomedCLIP | 66.8 | 45.5 |
| → +Ours | 66.8 | **67.4** (↑21.9%) |

**Comparison with concurrent works.** Using the same CLIP ViT-B/16 backbone, SpaceVLM substantially outperforms both DCSM [11] and NegationCLIP [25] on COCO and VOC-2007 MCQ tasks with no training required:

| Method | COCO MCQ | VOC-2007 MCQ |
|--------|----------|-------------|
| DCSM [11] | 48.6 | 49.0 |
| NegationCLIP [25] | 29.8 | 38.8 |
| **SpaceVLM (ours)** | **68.1** | **78.5** |

### 4.3 Application to Text-to-Image Generation (T2I)

We test whether SpaceVLM scoring improves negation adherence in T2I generation systems. We apply our method to GALIP [31], a GAN-based generator that uses a CLIP text encoder and produces image quality comparable to Stable Diffusion [28]. We evaluate on 107 negated prompts from [25] using Gemma-3-27B-it [32] as an automatic evaluator for presence/absence checks. Aff-Acc measures correctly generating the positive concept; Neg-Acc measures omitting the negated concept; Acc requires both to be satisfied simultaneously.

| Model | Aff-Acc ↑ | Neg-Acc ↑ | Acc ↑ |
|-------|----------|----------|-------|
| CLIP | 97.3 | 28.5 | 27.3 |
| → +Ours | 98.8 | 60.9 | **59.7** (+32.4) |
| CLIP-NegFull | 98.1 | 40.7 | 39.7 |
| → +Ours | 97.4 | 64.0 | **61.8** (+22.1) |
| ConCLIP | 27.7 | 68.3 | 11.0 |
| → +Ours | 86.6 | 57.8 | **48.2** (+37.2) |
| NegCLIP | 98.8 | 24.5 | 23.7 |
| → +Ours | 98.9 | 60.6 | **59.8** (+36.1) |
| NegCLIP-NegFull | 98.6 | 35.5 | 34.8 |
| → +Ours | 98.0 | 63.9 | **62.3** (+27.5) |
| NegationCLIP | 98.8 | 45.2 | 44.5 |
| → +Ours | 97.1 | 60.7 | **58.6** (+14.1) |

Our method substantially improves negation adherence with up to 37% higher accuracy over baselines. The subspace formulation explicitly removes the negated concept from the CLIP text embedding, enabling the generator to condition on representations that better match the intended prompt semantics.

### 4.4 Ablation Studies

**Varying VLM Complexity.** We evaluate how VLM complexity affects our method by testing three CLIP backbones of increasing size. Across all model sizes, our method consistently improves performance:

| Model | ViT-B/32 | ViT-B/16 | ViT-L/14 |
|-------|----------|----------|----------|
| CLIP | 39.2 | 41.4 | 38.5 |
| → +Ours | 66.3 (↑27.1%) | 67.4 (↑26.0%) | 65.9 (↑27.4%) |

**Sensitivity to the Threshold $t$.** We analyze performance sensitivity to $t$ on the MCQ benchmark by varying $t \in [0.90, 0.95]$. The maximum accuracy drop is 3.09% (COCO), indicating robustness and enabling practical, hand-set choices of $t$ in many applications. For the highest accuracy, we recommend cross-validation on the target task.

**Language Pre-processor.** We ablate across several LLMs of varying scale: SmolLM-360M-Instruct [2], TinyLlama-1.1B-Chat-v1.0 [43], Qwen2.5-3B-Instruct [34], and Mistral-7B [33]. TinyLlama-1B provides a favorable balance between accuracy and inference speed relative to the other models, making it a practical choice in real-world settings. We evaluate performance by averaging accuracy on the NegBench MCQ tasks across COCO, VOC-2007, and MSR-VTT; inference time is measured for a 32-input batch on a single H100 GPU.

### 4.5 Visualization

We compare SpaceVLM with vanilla CLIP by conducting an image-retrieval study on CIFAR-100 to evaluate both exclusion (retrieving images outside a negated category) and diversity among the retrieved results. We consider two settings: (i) vanilla prompting with "Not a photo of a \<category\>", (ii) our subspace-based negation using an affirmative prompt "This is a photo" combined with a negation prompt "A photo of a \<category\>". To quantify diversity, we compute the Shannon entropy [29] over the CIFAR-100 categories of the top-5 retrieved images.

Our method consistently yields higher entropy than vanilla CLIP, indicating more diverse results. Moreover, decreasing the threshold $t$ increases entropy, consistent with a larger feasible region in the embedding space. The retrieved images for the prompt "Not a photo of a mountain" confirm that our approach retrieves diverse images not labeled "mountain," whereas vanilla CLIP often fails to account for negation. As $t$ decreases, the retrieved images further diverge from the "mountain" category, reflecting the expanded subspace. To retrieve concepts related to the negated one (e.g., sky to mountain), the threshold should be kept in the range $[0.9, 0.95]$; lower thresholds lead to totally irrelevant retrieved concepts.

---

## 5. Conclusions and Limitations

We have presented a training-free geometric framework, **SpaceVLM**, for modeling negation in vision-language models. It treats negation as a subspace rather than a single embedding vector, allowing joint-embedding VLMs to handle negated prompts effectively without fine-tuning. The framework depends on a lightweight language module for query decomposition, which adds minor latency but works effectively even with small models such as TinyLlama-1B. Our study focuses on joint-embedding architectures; extending the subspace formulation to sequence-conditioned models such as LLaVA [20] is left for future work. The consistent gains across diverse backbones and tasks suggest that subspace reasoning is a natural mechanism for representing logical structure in vision–language spaces. We hope this geometric perspective will inspire further research on broader forms of logical and compositional reasoning.

---

## References

1. Kumail Alhamoud, Shaden Alshammari, Yonglong Tian, Guohao Li, Philip HS Torr, Yoon Kim, and Marzyeh Ghassemi. *Vision-language models do not understand negation.* In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 29612–29622, 2025.

2. Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Leandro von Werra, and Thomas Wolf. *SmolLM — blazingly fast and remarkably powerful,* 2024.

3. Tayfun Alpay, Sven Magg, Philipp Broze, and Daniel Speck. *Multimodal video retrieval with CLIP: a user study.* Information Retrieval Journal, 26(1):6, 2023.

4. Usha Bhalla, Alexander X. Oesterling, Suraj Srinivas, Flávio du Pin Calmon, and Himabindu Lakkaraju. *Interpreting CLIP with sparse linear concept embeddings (SpLiCE).* arXiv:2402.10376, 2024.

5. Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. *Conceptual 12M: pushing web-scale image-text pre-training to recognize long-tail visual concepts.* In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3558–3568, 2021.

6. Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. *The PASCAL visual object classes (VOC) challenge.* International Journal of Computer Vision, 88(2):303–338, 2010.

7. Enrico Fini, Mustafa Shukor, Xiujun Li, Philipp Dufter, Michal Klein, David Haldimann, Sai Aitharaju, Victor GT Turrisi da Costa, Louis Béthune, Zhe Gan, et al. *Multimodal autoregressive pre-training of large vision encoders.* In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 9641–9654, 2025.

8. Jeremy Irvin, Pranav Rajpurkar, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, Henrik Marklund, Behzad Haghgoo, Robyn Ball, Katie Shpanskaya, et al. *CheXpert: a large chest radiograph dataset with uncertainty labels and expert comparison.* In Proceedings of the AAAI Conference on Artificial Intelligence, pages 590–597, 2019.

9. Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. *Mistral 7B,* 2023.

10. Amita Kamath, Jack Hessel, and Kai-Wei Chang. *Text encoders bottleneck compositionality in contrastive vision-language models.* In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4933–4944, Singapore, 2023.

11. Raphi Kang, Yue Song, Georgia Gkioxari, and Pietro Perona. *Is CLIP ideal? No. Can we fix it? Yes!* ICCV, 2025.

12. Konstantin Klemmer, Esther Rolf, Caleb Robinson, Lester Mackey, and Marc Rußwurm. *SatCLIP: global, general-purpose location embeddings with satellite imagery.* In Proceedings of the AAAI Conference on Artificial Intelligence, pages 4347–4355, 2025.

13. Martha Lewis, Nihal V. Nayak, Peilin Yu, Qinan Yu, Jack Merullo, Stephen H. Bach, and Ellie Pavlick. *Does CLIP bind concepts? Probing compositionality in large image models.* arXiv:2212.10537, 2022.

14. Baiqi Li, Zhiqiu Lin, Deepak Pathak, Jiayao Li, Yixin Fei, Kewen Wu, Tiffany Ling, Xide Xia, Pengchuan Zhang, Graham Neubig, and Deva Ramanan. *GenAI-Bench: evaluating and improving compositional text-to-visual generation.* arXiv:2406.13743, 2024.

15. Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. *BLIP: bootstrapping language-image pre-training for unified vision-language understanding and generation.* In International Conference on Machine Learning, pages 12888–12900. PMLR, 2022.

16. Siting Li, Pang Wei Koh, and Simon Shaolei Du. *Exploring how generative MLLMs perceive more than CLIP with the same vision encoder.* In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10101–10119, 2025.

17. Long Lian, Boyi Li, Adam Yala, and Trevor Darrell. *LLM-grounded diffusion: enhancing prompt understanding of text-to-image diffusion models with large language models.* Transactions on Machine Learning Research, 2024.

18. Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. *Microsoft COCO: common objects in context.* In European Conference on Computer Vision, pages 740–755. Springer, 2014.

19. Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. *Flow matching for generative modeling.* arXiv:2210.02747, 2022.

20. Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. *Visual instruction tuning.* Advances in Neural Information Processing Systems, 36:34892–34916, 2023.

21. Ming Y. Lu, Bowen Chen, Drew FK Williamson, Richard J. Chen, Ivy Liang, Tong Ding, Guillaume Jaume, Igor Odintsov, Long Phi Le, Georg Gerber, et al. *A visual-language foundation model for computational pathology.* Nature Medicine, 30(3):863–874, 2024.

22. Christian Lülf, Denis Mayr Lima Martins, Marcos Antonio Vaz Salles, Yongluan Zhou, and Fabian Gieseke. *CLIP-branches: interactive fine-tuning for text-image retrieval.* In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2719–2723, 2024.

23. Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen, Wen Lei, Nan Duan, and Tianrui Li. *CLIP4Clip: an empirical study of CLIP for end-to-end video clip retrieval.* arXiv:2104.08860, 2021.

24. Zixian Ma, Jerry Hong, Mustafa Omer Gul, Mona Gandhi, Irena Gao, and Ranjay Krishna. *CREPE: can vision-language foundation models reason compositionally?* In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10910–10921, 2023.

25. Junsung Park, Jungbeom Lee, Jongyoon Song, Sangwon Yu, Dahuin Jung, and Sungroh Yoon. *Know "no" better: a data-driven approach for enhancing negation awareness in CLIP.* arXiv:2501.10913, 2025.

26. Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. *Learning transferable visual models from natural language supervision.* In International Conference on Machine Learning, pages 8748–8763. PMLR, 2021.

27. Ali Rasekh, Sepehr Kazemi Ranjbar, and Simon Gottschalk. *Multi-rationale explainable object recognition via contrastive conditional inference.* arXiv:2508.14280, 2025.

28. Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. *High-resolution image synthesis with latent diffusion models.* In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022.

29. Claude E. Shannon. *A mathematical theory of communication.* Bell Syst. Tech. J., 27:623–656, 1948.

30. Jaisidh Singh, Ishaan Shrivastava, Mayank Vatsa, Richa Singh, and Aparna Bharati. *Learning the power of "no": foundation models with negations.* In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 8002–8012. IEEE, 2025.

31. Ming Tao, Bing-Kun Bao, Hao Tang, and Changsheng Xu. *GALIP: generative adversarial CLIPs for text-to-image synthesis.* In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14214–14223, 2023.

32. Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, et al. *Gemma 3 technical report.* arXiv:2503.19786, 2025.

33. Mistral AI Team. *Mistral-7B-Instruct-v0.3.* https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3, 2024.

34. Qwen Team. *Qwen2.5: a party of foundation models,* 2024.

35. Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. *SigLIP 2: multilingual vision-language encoders with improved semantic understanding, localization, and dense features.* arXiv:2502.14786, 2025.

36. Jianfeng Wang, Zhengyuan Yang, Xiaowei Hu, Linjie Li, Kevin Lin, Zhe Gan, Zicheng Liu, Ce Liu, and Lijuan Wang. *GIT: a generative image-to-text transformer for vision and language.* arXiv:2205.14100, 2022.

37. Tsung-Han Wu, Long Lian, Joseph E. Gonzalez, Boyi Li, and Trevor Darrell. *Self-correcting LLM-controlled diffusion models.* In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6327–6336, 2024.

38. Jun Xu, Tao Mei, Ting Yao, and Yong Rui. *MSR-VTT: a large video description dataset for bridging video and language.* In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 5288–5296, 2016.

39. Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, and James Zou. *When and why vision-language models behave like bags-of-words, and what to do about it?* arXiv:2210.01936, 2022.

40. Xiaohua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner, Daniel Keysers, Alexander Kolesnikov, and Lucas Beyer. *LiT: zero-shot transfer with locked-image text tuning.* In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18123–18133, 2022.

41. Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. *Sigmoid loss for language image pre-training.* In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11975–11986, 2023.

42. Jingyi Zhang, Jiaxing Huang, Sheng Jin, and Shijian Lu. *Vision-language models for vision tasks: a survey.* IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(8):5625–5644, 2024.

43. Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, and Wei Lu. *TinyLlama: an open-source small language model.* arXiv:2401.02385, 2024.

44. Sheng Zhang, Yanbo Xu, Naoto Usuyama, Hanwen Xu, Jaspreet Bagga, Robert Tinn, Sam Preston, Rajesh Rao, Mu Wei, Naveen Valluri, et al. *BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs.* arXiv:2303.00915, 2023.

45. Jitian Zhao, Chenghui Li, Frederic Sala, and Karl Rohe. *Quantifying structure in CLIP embeddings: a statistical framework for concept interpretation.* arXiv:2506.13831, 2025.

46. Zihao Zhao, Yuxiao Liu, Han Wu, Mei Wang, Yonghao Li, Sheng Wang, Lin Teng, Disheng Liu, Zhiming Cui, Qian Wang, et al. *CLIP in medical imaging: a comprehensive survey.* arXiv:2312.07353, 2023.

47. Yuchen Zhou, Jiayu Tang, Shuo Yang, Xiaoyan Xiao, Yuqin Dai, Wenhao Yang, Chao Gou, Xiaobo Xia, and Tat-Seng Chua. *Logic unseen: revealing the logical blindspots of vision-language models.* arXiv:2508.11317, 2025.
