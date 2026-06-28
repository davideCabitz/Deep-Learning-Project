# When Negation Is a Geometry Problem in Vision-Language Models

**Authors:** Fawaz Sammani*¹˒², Tzoulio Chamiti*¹˒², Paul Gavrikov³, Nikos Deligiannis¹˒²

**Affiliations:** ¹ETRO Department, Vrije Universiteit Brussel, Belgium · ²imec, Kapeldreef 75, B-3001 Leuven, Belgium · ³Independent Researcher

*Equal contribution.

**Preprint:** arXiv:2603.20554v2 [cs.CV], 3 April 2026

**Code:** <https://github.com/fawazsammani/negation-steering>

---

## Abstract

Joint Vision-Language Embedding models such as CLIP typically fail at understanding negation in text queries — for example, failing to distinguish "no" in the query "a plain blue shirt with no logos." Prior work has largely addressed this limitation through data-centric approaches, fine-tuning CLIP on large-scale synthetic negation datasets. However, these efforts are commonly evaluated using retrieval-based metrics that cannot reliably reflect whether negation is actually understood.

In this paper, we identify two key limitations of such evaluation metrics and investigate an alternative evaluation framework based on Multimodal LLMs-as-a-judge, which typically excel at understanding simple yes/no questions about image content, providing a fair evaluation of negation understanding in CLIP models. We then ask whether there already exists a direction in the CLIP embedding space associated with negation. We find evidence that such a direction exists, and show that it can be manipulated through test-time intervention via representation engineering to steer CLIP toward negation-aware behavior without any fine-tuning. Finally, we test negation understanding on non-common image-text samples to evaluate generalization under distribution shifts.

---

## 1. Introduction

Contrastive Language–Image Pretraining (CLIP) [20] learns a joint embedding space that aligns images and text, where cosine similarity between their embeddings serves as a measure of semantic correspondence, with higher similarity indicating a stronger match. Despite their effectiveness, CLIP models struggle with understanding negation, an equally critical aspect of language. Many real-world applications require robust negation handling, particularly in search and recommendation systems where CLIP models often serve as the primary building block. For instance, a user may search for images of "lion sightings with no vehicles in frame", a city planner for "high-traffic intersections with no pedestrian crossings", or a real-estate agent for "apartments with a balcony and no busy market street." In each case, a single negation cue fundamentally alters the expected result. This limitation impacts a wide range of multimodal applications built on CLIP, including text-to-image generation [18, 21], multimodal large language models [10, 11, 13], open-vocabulary object detection [15, 25], referring image segmentation [8, 9, 14], and more.

To counteract this issue, existing approaches adopt data-centric strategies, generating synthetic negated captions either through template-based methods [22] or by leveraging a range of foundation models [1, 5, 17, 23]. While scalable and efficient, this creates a massive amount of false negatives (i.e., images satisfying the query but not explicitly labeled as ground truth by the data construction pipeline). This issue is particularly relevant when evaluating with the commonly used COCO dataset [12], where key themes are often similar. We further observe that some baseline models collapse after being finetuned and result in retrieved images that are unrelated to the query.

These observations motivate us to investigate alternative methods for the evaluation of retrieval and negation understanding using MLLM-as-a-judge, separately evaluating each component of the query and the retrieved image. Since this approach is unsupervised, it can be applied to any image database of any size.

Finally, we investigate whether a "negation" direction vector already exists within the CLIP embedding space. Specifically, we ask whether CLIP embeddings (or a subset of their dimensions) encode information related to negation, and if so, whether the embedding space can be manipulated at inference time by steering it towards the discovered "negation" direction when a query contains negation. In our work, we find evidence of this directional vector in the CLIP space, and we show that CLIP models can be geometrically steered to understand negation without any finetuning.

### Contributions

- We highlight two fundamental limitations in the evaluation of negated retrieval systems and explore alternative evaluation methods based on MLLM-as-a-judge.
- We investigate whether a negation direction already exists in the CLIP embedding space and whether it can be extracted.
- We find evidence that this direction already exists, and we use it to steer CLIP representations at inference-time for negation-aware understanding.

---

## 2. Related Work

### 2.1 Negation in Vision-Language Models

Several works [1, 5, 17, 23] have investigated why joint embedding VLMs such as CLIP typically collapse affirmative (e.g., "a dog on the grass") and negated (e.g., "a dog not on the grass") queries into indistinguishable embeddings. The reasons are two-fold: (1) CLIP behaves like a bag of words [7, 23], primarily contextualizing the prompt as a whole rather than understanding fine-grained linguistic structures, relying heavily on the presence of content words rather than the syntactic and relational structure of the sentence. This limitation largely stems from the nature of CLIP's pretraining data: captions involving negation are severely underrepresented [17]. For instance, it is far less common to encounter an internet caption of "a dog not on the grass." Even when such captions appear in the training data, contrastive learning — despite very large batch sizes — rarely provides paired positive and negative samples in the same batch that share the same content but differ in relational structure or word order. As a result, models seldom observe contrasts (e.g., "a dog on the grass" vs. "a dog not on the grass"). A bag-of-words strategy can be high-reward and sufficient to get a low loss. (2) In addition to the training data, the geometry of a single embedding vector from CLIP makes it unreliable at handling basic matching, attribute binding, spatial relationships, and negation altogether. This problem occurs even in large and better-performing CLIP models such as CLIP ViT-L/H [23], SigLIP ViT-L/14 [24].

### 2.2 Data-Centric Approaches to Negation

To address these limitations, existing approaches adopt data-centric strategies of generating synthetic negated captions and finetuning CLIP on them [1, 5, 22]. While this strategy enables large-scale training, the synthetic data generation process can introduce a substantial number of false negatives, where multiple images may satisfy the same negated caption, but only a single image is treated as a positive example. As a result, improvements measured under standard retrieval-based metrics mix semantic matching with genuine negation understanding. Apart from this, we also show that some baseline models collapse when being finetuned on synthetic negation data, leading to unrelated image associations to the text query.

These observations motivate us to investigate an alternative evaluation metric based on MLLM-as-a-judge, which jointly assesses retrieval and negation understanding. Moreover, in contrast to prior approaches that rely on fine-tuning CLIP with large volumes of synthetic data, we instead examine whether a latent "negation" direction already exists within CLIP's embedding space, and whether CLIP can be manipulated with this direction via representation engineering to improve its handling of negation.

---

## 3. Negation Evaluation in Image Retrieval

All existing approaches to negation-aware vision–language learning [1, 5, 17, 22, 23] rely on the same underlying strategy: they generate synthetic negated captions and fine-tune CLIP on this data. For example, NegBench [1] constructs a large-scale synthetic dataset of 12 million image–text pairs by combining the OWL-ViT open-vocabulary object detection model [16] with a Large Language Model (LLM). While this pipeline is scalable and efficient, it introduces two fundamental problems:

**(1) False Negatives.** The automatic data annotation pipeline introduces a substantial amount of false negatives. These are images that are correct matches for a caption, but are not labeled as ground truth by the automatic process. The COCO dataset contains many similar images with overlapping object compositions. Since the annotation pipeline operates in isolation per image–caption pair, it is highly likely that multiple images satisfy the same (negated) caption. When only a single image is treated as the positive example, all other valid matches are implicitly considered incorrect during evaluation. Notably, this problem is also well-studied in conventional image-text retrieval [3], motivating researchers to re-annotate data.

**(2) Collapse After Finetuning.** We also observe that ConCLIP [22], one of the baseline models, collapses completely when handling negated queries. Specifically, it retrieves images that are entirely unrelated to the negated text query and assigns them lower cosine similarity scores than any non-negated query. This finding is consistent with [1], which observed that ConCLIP collapses all negated captions (independent of the object or action being negated) into the same point in the embedding space. Such behavior also raises concerns about the reliability of the evaluation metric used in [22].

Motivated by these two observations, we explore alternative evaluation strategies for assessing retrieval and negation understanding. Multimodal Large Language Models (MLLMs) achieve remarkable performance on complex visual reasoning benchmarks. This implies that they can reliably judge simpler yes/no questions about image content. Drawing inspiration from the success of LLM-as-a-judge [4], we propose to use MLLM-as-a-judge to evaluate text-to-image negated retrieval performance. This also allows us to use any image database of any size, simulating real-world scenarios where database sizes may be prohibitively large. Unlike existing negation benchmarks, we use a relatively large image database of 25K images (5× larger than current benchmarks).

**MLLM-as-a-judge evaluation protocol.** Let $c_n$ denote a text query that contains negation. We use an LLM to formulate two evaluation questions, $q_r$ and $q_n$. The question $q_r$ assesses whether the retrieved image is contextually and semantically correct, mitigating the collapse issue discussed earlier. The question $q_n$ evaluates negation (i.e., whether the negated object is present in the image). Evaluation proceeds sequentially. We first ask $q_r$, whose ground-truth answer is "yes", indicating that the retrieved image is contextually and semantically correct. If the judge answers "yes" (i.e., $q_r$ is correct), we then ask $q_n$, whose ground-truth answer is "no", indicating that the negated object is not present in the image. For negation benchmarks, both $q_r$ and $q_n$ must be answered correctly. In our work, we use Qwen3-VL-32B, Qwen3-VL-8B, and Qwen3-VL-4B [2] as MLLM judges.

**Results on NegBench.** We report results on the NegBench benchmark [1]. NegBench consists of a database of 5K images from the COCO 2017 validation set [12], each annotated with 5 synthetic negated captions. Results are shown in Table 1 using CLIP ViT-B/32. We compare the original CLIP to three baselines: ConCLIP [22], NegCLIP [23], and CLIP-CC12M [1]. We report three metrics: Top-1, Avg.5, and Top-5, both for retrieval performance (correctness of $q_r$) and for joint retrieval and negation performance (correctness of $q_r$ and $q_n$).

**Table 1.** Retrieval and Negation performance of baseline models with ViT-B/32 on NegBench across two MLLMs-as-a-judge.

| | Qwen3-VL-8B Retrieval | | | Qwen3-VL-8B Ret.+Neg. | | | Qwen3-VL-32B Retrieval | | | Qwen3-VL-32B Ret.+Neg. | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Model | Top-1 | Avg.5 | Top-5 | Top-1 | Avg.5 | Top-5 | Top-1 | Avg.5 | Top-5 | Top-1 | Avg.5 | Top-5 |
| CLIP [20] | 46.7 | 30.4 | 75.2 | 39.4 | 25.1 | 68.4 | 47.2 | 30.6 | 76.3 | 40.3 | 25.4 | 69.9 |
| ConCLIP [22] | 46.6 | 30.7 | 76.4 | 40.8 | 26.1 | 70.7 | 47.2 | 30.7 | 77.7 | 41.6 | 26.3 | 72.2 |
| NegCLIP [23] | **60.9** | **37.9** | **84.3** | **51.7** | **31.5** | **77.9** | **62.0** | **38.3** | **85.8** | **53.1** | **32.1** | **79.8** |
| CLIP-CC12M [1] | 49.0 | 31.2 | 76.4 | 42.7 | 26.7 | 70.6 | 49.9 | 31.4 | 78.0 | 43.8 | 27.0 | 72.5 |

As shown, our evaluation protocol indicates that all baselines, except NegCLIP, perform only marginally better than the original CLIP. NegCLIP [23] is the only method that significantly outperforms all other baselines. By analyzing the training data of both NegCLIP and CLIP-CC12M, we find that NegCLIP was trained on contrasting data covering (1) attribute binding, (2) spatial relationships, and (3) negation. On the other hand, the training data of CLIP-CC12M is largely dominated by negation data. This explains why NegCLIP outperforms CLIP-CC12M.

We also report standard retrieval performance on NegBench using Recall@K in Table 2. Under these metrics, all negation-aware models appear to improve smoothly over the CLIP baseline, with NegCLIP achieving the highest recall. However, such an evaluation does not distinguish between semantic retrieval correctness and negation satisfaction, nor does it account for the presence of multiple valid matches that are not annotated as ground truth. MLLM-as-a-judge evaluation alleviates this problem.

**Table 2.** Retrieval performance (Recall@K) of baseline models with ViT-B/32 on NegBench.

| Model | R@1 | R@5 | R@10 |
|-------|-----|-----|------|
| CLIP [20] | 0.24 | 0.47 | 0.59 |
| ConCLIP [22] | 0.26 | 0.51 | 0.62 |
| NegCLIP [23] | **0.37** | **0.64** | **0.74** |
| CLIP-CC12M [1] | 0.28 | 0.51 | 0.62 |

---

## 4. A Negation Direction in the CLIP Space

Although current works address the negation issue using data-centric approaches, an earlier work [19] observed that CLIP already possesses an inherent ability to process negation. This observation led us to hypothesize that a directional vector for negation may already exist within the CLIP embedding space. In this work, we draw inspiration from TCAV [6], an established work in the field of Explainable AI, to verify this hypothesis.

Specifically, we take 4,000 captions from the COCO dataset [12] and use an LLM to negate them, obtaining two subsets of the original captions and their negated counterparts, each of equal size. We also instruct the LLM to vary the negation cues to avoid overfitting to a single style. We extract the hidden representations of all captions and their negations from the residual stream of a layer $l$ of the CLIP text encoder. Specifically, we use the hidden state corresponding to the \<eos\> token, denoted as $h^l \in \mathbb{R}^d$, as it serves as a summary representation of the entire caption. This choice is further backed by [19], which demonstrates that the final token position has the strongest influence on the negation signal. We split the dataset into training and test sets and train a linear binary classifier to distinguish between original captions (label 0, no negation) and their negated counterparts (label 1, negated).

The classifier achieves a test accuracy of 99% or above at layer 4 for all three CLIP models tested (ViT-B/32, ViT-B/16, ViT-L/14), indicating that negation-related information is clearly encoded in the hidden states and that the representations of original and negated captions are linearly separable. We further observe that negation information is better encoded in intermediate layers of the CLIP Text Encoder, rather than early or late layers.

After training the linear classifier, its coefficients (weights) $W^l \in \mathbb{R}^d$ define a direction in the latent space. This vector is aligned with $h^l$ and points toward the direction associated with negation, since the dot product $W^l h^l$ is 1 when $h^l$ corresponds to the latent representation of a negated caption. We first isolate the magnitude of $W^l$ from its direction by performing a unit normalization: $W^l_{\text{dir}} = W^l / \|W^l\|$. We then steer $h^l$ in the direction of negation by:

$$h^l = (1 - \alpha)\, h^l + \alpha\, W^l_{\text{dir}}\, \|h^l\| \tag{1}$$

where $\alpha$ is a hyperparameter that controls how much to steer the representations, and $\|h^l\|$ preserves the norm of the representations after steering, preventing them from being shifted into an out-of-distribution latent. We perform this for all layers; that is, $l = [1 \ldots L]$ where $L$ is the total number of layers in the CLIP Text Encoder.

**Controlled evaluation with SimpleNeg.** To objectively evaluate this capability, we curated a simple, controlled dataset of negation queries containing at most two objects, adjectives, or actions, and a negation (e.g., "a supermarket scene with a shopping cart but no cashier"). We used an LLM in combination with human-in-the-loop to create this benchmark, which we term **SimpleNeg** of 900 samples. Our primary goal is not to maximize task complexity, but to rigorously test whether the negation direction can be explicitly identified and manipulated within the embedding space — a unit-test approach rather than a stress-test one. Furthermore, in practical retrieval scenarios, user queries are typically concise and focused, making this setup representative of real-world use.

**Results on SimpleNeg.** Results are presented in Table 3 for 3 MLLMs-as-a-judge. Results demonstrate that steering the representations with the negation direction achieves performance surpassing all baselines on the SimpleNeg dataset. This approach demonstrates that (1) there exists a negation direction in the CLIP text embedding space that can be used to geometrically steer CLIP so that it captures and understands negation; (2) this can be achieved without any fine-tuning of CLIP; and (3) it requires only a relatively small dataset (4K samples) to train the linear classifier, which can be obtained completely independently of images, with any text-only LLM. Specifically, the linear classifier requires 0.03% of the amount of data used in previous approaches [1].

**Table 3.** Intervention using the discovered negation direction on SimpleNeg with ViT-B/32, across three MLLMs-as-a-judge.

| Method | # Neg. Training Data | Without Finetuning? | Qwen3-VL-32B | | | Qwen3-VL-8B | | | Qwen3-VL-4B | | |
|--------|---------------------|---------------------|---|---|---|---|---|---|---|---|---|
| | | | Top-1 | Avg.5 | Top-5 | Top-1 | Avg.5 | Top-5 | Top-1 | Avg.5 | Top-5 |
| Baseline [20] | – | – | 44.9 | 44.9 | 84.7 | 46.0 | 45.2 | 84.6 | 46.9 | 45.0 | 85.4 |
| ConCLIP [22] | 228K | ✗ | 30.7 | 30.2 | 63.8 | 31.7 | 30.5 | 64.6 | 30.9 | 30.3 | 63.9 |
| NegCLIP [23] | 120K | ✗ | 46.0 | 45.7 | 83.1 | 47.7 | 46.7 | 84.8 | 48.1 | 46.8 | 84.2 |
| CLIP-CC12M [1] | 12M | ✗ | 53.1 | 49.8 | 88.0 | 55.9 | 50.8 | 89.3 | 55.0 | 50.0 | 88.3 |
| **Steering (Ours)** | **4K** | **✓** | **54.3** | **53.3** | **98.0** | **53.7** | **51.7** | **94.0** | **54.0** | **51.5** | **94.1** |

**Implementation Details.** We use the L-BFGS solver to train the binary classifier with no bias (intercept) for a maximum of 1000 iterations. We use the original CLIP model from OpenAI. For SimpleNeg, we set the database size to 25K images sourced from the COCO 2014 Validation Set [12]. We use gpt-5-mini-2025-08-07 from OpenAI as the LLM.

**Embedding Analysis.** We visualize the embeddings of the captions using Principal Component Analysis (PCA) on 197 samples, covering original (affirmative) captions, their negated counterparts, and the negated embeddings after steering. The affirmative and negated representations remain somewhat separable even in a simple 3D space, supporting our hypothesis. However, additional refinement is still required to achieve a complete separation from the affirmative representations.

**Ablation on Steering Strength.** We conduct ablation studies on the $\alpha$ parameter used in Eq. (1). Larger values of $\alpha$ shift the distribution excessively, leading to a degradation in performance, highlighting the trade-off between amplifying the negation signal and preserving the original semantic structure of the embedding. The best-performing value we identified is $\alpha = 0.13$, which we use in our experiments. This finding is consistent across all MLLM judges.

**Non-Common Objects in Context (N-COCO).** We also evaluate how well the baselines and our steering method generalize to non-common image–text pairs. To this end, we construct a synthetic benchmark termed **N-COCO**, of 200 images spanning 10 negated queries describing uncommon scenes (e.g., "a book not made of paper", "a giraffe but not outdoors"). Images are generated using GPT-5 Image with a human-in-the-loop; every generated image is manually reviewed by a human. For each query, the benchmark includes 10 positive images that satisfy the negated caption and 10 distractor images that violate it. Unlike traditional retrieval benchmarks that assume a single ground-truth image per query, our setup includes multiple semantically valid matches, allowing evaluation of negation understanding under semantic ambiguity. R@1/R@3/R@5 measure whether the respective top-$K$ retrieved images are among those 10 positives.

**Results on N-COCO.** As shown in Table 4, the original CLIP model maintains strong performance under this distribution shift, achieving R@1 of 0.60. In contrast, both NegCLIP and CLIP-CC12M report a drop in performance (R@1 = 0.50). This is particularly notable because these models were explicitly finetuned to improve negation understanding. Our steering approach achieves the highest R@1 of 0.80, without modifying model weights or requiring additional training data. The degradation of finetuned models suggests overfitting to specific image-text patterns in the training data, reducing robustness under distribution shift. Representation steering operates directly on the latent structure, enabling improved negation handling without compromising generalization.

**Table 4.** Retrieval performance (Recall@K) of baseline models with ViT-B/32 on the N-COCO controlled benchmark. ConCLIP achieved 0.00 on all Recall@K metrics.

| Model | R@1 | R@3 | R@5 |
|-------|-----|-----|-----|
| CLIP [20] | 0.60 | 0.60 | 0.90 |
| NegCLIP [23] | 0.50 | 0.50 | 0.80 |
| CLIP-CC12M [1] | 0.50 | 0.60 | 0.90 |
| **Steering (Ours)** | **0.80** | **0.80** | **0.90** |

---

## 5. Conclusion

In this work, we investigate negation in vision–language embedding models and identify critical flaws in existing conventional metrics that make them unreliable for assessing negation understanding. By introducing a MLLM-as-a-judge framework, we address these issues and reduce the need for data annotation, providing a more reliable way to evaluate negation understanding in CLIP models. We then find that negation information is already encoded in CLIP's text latent space, just not well activated. Motivated by this observation, we identified the negation direction in representations of the text encoder layers and showed that test-time steering along this direction enables negation-aware understanding. Finally, we introduced a synthetic benchmark N-COCO to evaluate CLIP's handling of negation in uncommon scenes, and used it to assess both baseline methods and our steering approach under distribution shifts. Extending such representation-level interventions to more complex queries involving multiple interacting objects and actions remains an important direction for future work. Overall, our findings highlight the potential of lightweight representation engineering in multimodal models as an alternative to large-scale retraining.

---

## Acknowledgments

Fawaz Sammani is funded by the Fonds Wetenschappelijk Onderzoek (FWO) (PhD fellowship strategic basic research 1SH7W24N). T. Chamiti and N. Deligiannis acknowledge the "Onderzoeksprogramma Artificiële Intelligentie (AI) Vlaanderen" programme and the ERC Consolidator Grant IONIAN (No. 101171240, DOI: 10.3030/101171240). Funded by the European Union.

---

## References

1. Kumail Alhamoud, Shaden S. Alshammari, Yonglong Tian, Guohao Li, Philip Torr, Yoon Kim, and Marzyeh Ghassemi. *Vision-language models do not understand negation.* 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 29612–29622, 2025.

2. Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. *Qwen3-VL technical report.* arXiv:2511.21631, 2025.

3. Sanghyuk Chun, Wonjae Kim, Song Park, Minsuk Chang Chang, and Seong Joon Oh. *ECCV caption: correcting false negatives by collecting machine-and-human-verified image-caption associations for MS-COCO.* In European Conference on Computer Vision (ECCV), 2022.

4. Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, et al. *A survey on LLM-as-a-judge.* arXiv:2411.15594, 2024.

5. Raphi Kang, Yue Song, Georgia Gkioxari, and Pietro Perona. *Is CLIP ideal? No. Can we fix it? Yes!* ICCV, 2025.

6. Been Kim, Martin Wattenberg, Justin Gilmer, Carrie J. Cai, James Wexler, Fernanda B. Viégas, and Rory Sayres. *Interpretability beyond feature attribution: quantitative testing with concept activation vectors (TCAV).* In International Conference on Machine Learning, 2017.

7. Darina Koishigarina, Arnas Uselis, and Seong Joon Oh. *CLIP behaves like a bag-of-words model cross-modally but not uni-modally.* In The Fourteenth International Conference on Learning Representations, 2026.

8. Jungbeom Lee, Sungjin Lee, Jinseok Nam, Seunghak Yu, Jaeyoung Do, and Tara Taghavi. *Weakly supervised referring image segmentation with intra-chunk and inter-chunk consistency.* pages 21813–21824, 2023.

9. Jungbeom Lee, Sanghyuk Chun, and Sangdoo Yun. *Toward interactive regional understanding in vision-large language models.* In North American Chapter of the Association for Computational Linguistics, 2024.

10. Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. *Otter: a multi-modal model with in-context instruction tuning.* IEEE Transactions on Pattern Analysis and Machine Intelligence, 47:7543–7557, 2023.

11. Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. *LLaVA-onevision: easy visual task transfer.* Transactions on Machine Learning Research, 2025.

12. Tsung-Yi Lin, Michael Maire, Serge J. Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. *Microsoft COCO: common objects in context.* In European Conference on Computer Vision, 2014.

13. Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. *Improved baselines with visual instruction tuning.* 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 26286–26296, 2024.

14. Timo Lüddecke and Alexander S. Ecker. *Image segmentation using text and image prompts.* 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7076–7086, 2022.

15. Neil Houlsby, Matthias Minderer, and Alexey Gritsenko. *Scaling open-vocabulary object detection.* NeurIPS, 2023.

16. Matthias Minderer, Alexey A. Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, Xiao Wang, Xiaohua Zhai, Thomas Kipf, and Neil Houlsby. *Simple open-vocabulary object detection with vision transformers.* ECCV, 2022.

17. Junsung Park, Jungbeom Lee, Jongyoon Song, Sangwon Yu, Dahuin Jung, and Sungroh Yoon. *Know "no" better: a data-driven approach for enhancing negation awareness in CLIP.* ICCV, 2025.

18. Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. *SDXL: improving latent diffusion models for high-resolution image synthesis.* In The Twelfth International Conference on Learning Representations, 2024.

19. Vincent Quantmeyer, Pablo Mosteiro, and Albert Gatt. *How and where does CLIP process negation?* In Proceedings of the 3rd Workshop on Advances in Language and Vision Research (ALVR), pages 59–72, Bangkok, Thailand, 2024.

20. Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. *Learning transferable visual models from natural language supervision.* In International Conference on Machine Learning, 2021.

21. Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. *High-resolution image synthesis with latent diffusion models.* 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10674–10685, 2022.

22. Jaisidh Singh, Ishaan Shrivastava, Mayank Vatsa, Richa Singh, and Aparna Bharati. *Learning the power of "no": foundation models with negations.* 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 8002–8012, 2025.

23. Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, and James Zou. *When and why vision-language models behave like bags-of-words, and what to do about it?* In The Eleventh International Conference on Learning Representations, 2023.

24. Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. *Sigmoid loss for language image pre-training.* 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 11941–11952, 2023.

25. Chuhan Zhang, Chaoyang Zhu, Pingcheng Dong, Long Chen, and Dong Zhang. *Cyclic contrastive knowledge transfer for open-vocabulary object detection.* ICLR, 2025.
