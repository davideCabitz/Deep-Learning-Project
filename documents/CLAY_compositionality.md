# Not Only Text: Exploring Compositionality of Visual Representations in Vision-Language Models

Davide Berasi, Matteo Farina, Massimiliano Mancini, Elisa Ricci, Nicola Strisciuglio

Fondazione Bruno Kessler — University of Trento — University of Twente

## Abstract

Vision-Language Models (VLMs) learn a shared feature space for text and images, enabling the comparison of inputs of different modalities. While prior works demonstrated that VLMs organize natural language representations into regular structures encoding composite meanings, it remains unclear if compositional patterns also emerge in the visual embedding space. In this work, we investigate compositionality in the image domain, where the analysis of compositional properties is challenged by noise and sparsity of visual data. We address these problems and propose a framework, called **Geodesically Decomposable Embeddings (GDE)**, that approximates image representations with geometry-aware compositional structures in the latent space. We demonstrate that visual embeddings of pre-trained VLMs exhibit a compositional arrangement, and evaluate the effectiveness of this property in the tasks of compositional classification and group robustness. GDE achieves stronger performance in compositional classification compared to its counterpart method that assumes linear geometry of the latent space. Notably, it is particularly effective for group robustness, where we achieve higher results than task-specific solutions. Our results indicate that VLMs can automatically develop a human-like form of compositional reasoning in the visual domain, making their underlying processes more interpretable.

## 1. Introduction

Compositionality is the principle by which cognitive and computational systems create meaning of a complex expression by combining the meaning of its (simpler) parts [50, 51]. Humans leverage compositionality instinctively, combining known elements to interpret novel situations. In machine intelligence, efforts were made to replicate this capability by developing models that imitate compositional processes, e.g., solving complex tasks via sub-goals [9, 31, 49, 61], modeling objects as compositions of their parts [13, 14, 47, 60, 62], encoding concept hierarchies [11, 17, 33, 55], explicitly learning compositional representations [1, 18, 38, 44], or architectures [20, 24, 36, 63, 68].

With the rise of modern Vision-Language Models (VLMs) [26, 54, 74] jointly trained on large-scale image-text pairs, there has been growing interest in investigating whether these models exhibit intrinsic compositional behaviors [45, 52, 71]. In particular, Trager et al. [67] investigated latent compositional structures within the CLIP [54] text embedding space, demonstrating that composite concepts can be represented as linear combinations of embedding vectors corresponding to various factors. These vectors, called **ideal words**, can be used to compose new concepts in the embedding space. Their work focuses on finding compositional structures in the text embedding space of CLIP, motivated by the fact that the structured and symbolic nature of language may facilitate the study of computational approaches to capture compositional meaning. However, cognitive studies show that language itself is used to describe and interpret the visual world and directly affects visual perception [6]. Hence, similar to text, human visual representations exhibit a compositional structure [19], made of simpler components systematically combined. Despite this connection, compositional properties of visual embeddings of VLMs have remained so far mostly unexplored.

To fill this gap, in this paper, we introduce **GEODESICALLY DECOMPOSABLE EMBEDDINGS (GDE)**, a framework grounded in differential geometry and designed to investigate compositional structures of pre-trained embeddings within Riemannian manifolds. Visual embeddings exhibit unique challenges not present in compositional analysis of text embeddings, namely data sparsity in the compositional space and noise and ambiguity in images. Specifically, we deal with the sparsity of composite concepts, as certain combinations of elementary primitives may not appear in real image collections (e.g., focusing on objects and attributes, "blue dog" images are unlikely to exist). Noise and ambiguity concern additional visual cues and information present in images, e.g. background, context, etc., that do not correspond to the composite concepts.

We evaluated the compositional representations computed with the proposed approach in two relevant applications, namely compositional classification and group robustness, considering publicly available datasets, showing that it better captures visual compositional structures than the alternatives (e.g., [10]). GDE is particularly effective for group robustness, where we achieve better debiasing results than task-specific methods. Furthermore, we show that GDE can be successfully used in combination with state-of-the-art generative models to synthesize images of compositional concepts. Our contributions are:

i) We study compositional structures within visual embeddings for VLMs and demonstrate that the latent representations of visual signals also exhibit a degree of compositionality.
ii) We show that, unlike for text embeddings, linear structures are insufficient to (de)compose visual concepts; thus, the manifold geometry must be considered.
iii) We propose a framework that deals with the sparsity and noise of composite concepts in images, enabling the compositional analysis of visual embeddings.

## 2. Related Work

### Compositionality in Vision

Compositionality is considered a cornerstone of perception [34], and compositional representations offer an effective tool to represent real-world phenomena [12]. The primary benefit of compositionality is the possibility of combining the representation of simpler concepts to understand and reason on complex ones, allowing for generalization to new unseen combinations of concepts [31, 44, 60]. In computer vision, early efforts focused on recognizing objects as composition of parts [13, 14, 47, 48] and evolved into architectures that can recognize and model objects in a compositional fashion [60], compositional generation [49, 64, 76], and interpretable representations [4, 63]. Compositionality has also lead to progress in various tasks, such as human-object interaction detection, model spatial/semantic relationships [21, 22, 28], and compositional zero-shot learning, where the goal is to recognize unseen compositions of training primitives [37, 41–44]. While these works focus on specific applications, in this paper we aim to study whether there exists an underlying compositional structure in the visual embeddings of VLMs.

### Compositionality in VLMs

Modern Vision-Language Models (VLMs) like CLIP [54] are trained to extract meaningful representations from complex visual scenes guided by textual inputs without a priori imposing any form of compositionality. In this context, a natural question is: Does compositional behavior emerge automatically in VLMs?

Previous works already showed how VLMs are more suitable for tasks such as compositional zero-shot learning [40, 45, 52], and how their representations allow for cross-modal compositions, such as visual editing [5, 29, 75] and compositional retrieval [3, 23, 27, 59]. At the same time, works studied the challenges of VLMs in modeling compositional inputs, e.g., at the level of word order, object-attribute bindings, spatial relationships and other compositional challenges [23, 65, 66, 70].

In this paper, we study the compositional structure in the visual embeddings extracted from VLMs. Close to our goal is [35], studying the compositional properties of the CLIP text encoder through compositional distributional semantics models in synthetic test scenarios. Similarly, [67] show that the textual embeddings of VLMs can be well approximated by linear compositions of smaller sets of ideal vectors. Motivated by the cross-modal alignment of VLMs, we investigate whether the embeddings of visual inputs exhibit an analogous compositional property. We achieve this by constructing a geometry-aware decomposition framework, following ideas similar to [46], where Principal Geodesic Analysis (PGA) [15] is applied to learn lower-dimensional submanifolds of the CLIP sphere associated to distinct parts-of-speech. To the best of our knowledge, this is the first work that investigates the emergence of compositional structures in the visual embeddings of VLMs.

## 3. Method

We propose a framework to analyze the compositional properties of image embeddings of neural encoders. We start by reviewing the fundamentals of the CLIP model along with key concepts from differential geometry (Sec. 3.1). We then formalize the concept of geodesic decomposability (Sec. 3.2) and discuss our methodology for dealing with visual inputs (Sec. 3.3).

### 3.1 Preliminaries

**Contrastive Language-Image Pretraining (CLIP)** consists of a pre-trained image encoder φ_im : X → R^d and a text encoder φ_t : Y → R^d that represent multi-modal text-visual inputs in a shared vision-language space. The latent representations of an image x ∈ X and text y ∈ Y are compared by cosine similarity, which is the scalar product u_x^⊤ u_y of their normalized versions u_x = φ_im(x)/||φ_im(x)||, u_y = φ_t(y)/||φ_t(y)||. The weights of the encoders are trained to optimize a contrastive objective on a huge collection of paired image-text samples. Since the norm of CLIP embeddings does not carry any meaningful information, spherical geometry applies to their post-hoc analysis.

**Riemannian Manifolds** are geometric spaces where intrinsic distances can be measured. For a generic manifold M ⊂ R^d with intrinsic distance d_M : M × M → [0, ∞), we recall the notions of exponential map and intrinsic mean. These tools permit operating with non-linear data, like the spherical normalized CLIP embeddings, while respecting their intrinsic shape. Let µ be a point on M and let T_µ M be the tangent space in µ. The **exponential map** projects a tangent vector v ∈ T_µ M onto the manifold by moving along the geodesic segment it defines. Formally, if γ_v : [0, 1] → M is the unique geodesic path starting from γ_v(0) = µ with initial velocity γ̇_v(0) = v, then Exp_µ(v) := γ_v(1). This function is locally invertible and its inverse is the **logarithmic map** Log_µ = Exp_µ^{-1}.

The exponential and logarithmic maps send straight lines of the tangent plane into geodesic curves of the manifold, and vice-versa. Moreover, they approximately preserve distances between elements close to the point of tangency µ:

```
d_M(u, u') ≈ ||Log_µ(u) − Log_µ(u')||,  u, u' ∈ M    (1)
```

Note that in (1) equality holds if u = µ or u' = µ. When applying the logarithmic map to a set of points {u_i}_{i=1}^N ⊂ M, the natural choice for the point of tangency µ is the **intrinsic mean**, i.e., the element of M minimizing the average squared distance to the given points. Each point u_i is associated to a scalar weight w_i belonging to a probability-simplex vector ∆ and the (weighted) intrinsic mean is:

```
µ = argmin_{u ∈ M} Σ_i w_i d_M(u, u_i)²    (2)
```

This distance-minimizing element µ guarantees that the images of the points through the logarithmic map are centered in the origin of the tangent space: Σ_i w_i Log_µ(u_i) = 0.

### 3.2 Geodesically Decomposable Embeddings

We now formalize our proposed notion of compositional embeddings. We consider a set of composite meanings Z = Z_1 × ··· × Z_s, defined as the Cartesian product between finite lists of primitive concepts, and refer to the Z_i (i = 1,..., s) as the **dimensions** of Z. For example, Z = {red, blue} × {car, dress, flower} combines primitives from an attribute dimension and an object dimension.

We then consider an embedding map φ : Z → M representing the composite concepts as points on a manifold M ⊂ R^d. Intuitively, the set φ(Z) = {u_z | z ∈ Z} is compositional if it has a regular structure reflecting the composite nature of the inputs, i.e., if one can compose primitive concepts within the geometric space to obtain embeddings of complex meanings. In this paper, we associate compositionality to the notion of **geodesic decomposability** which accounts for the intrinsic geometry of the manifold.

**Definition 1 (Geodesically decomposable embeddings).** A set of embeddings φ(Z) = {u_z | z ∈ Z} ⊂ M with intrinsic mean µ is geodesically decomposable if there exist v_{z_i} ∈ T_µ M for all z_i ∈ Z_i (i = 1,..., s) such that:

```
u_z = Exp_µ(v_{z_1} + ··· + v_{z_s})  ∀z = (z_1,..., z_s)    (3)
```

Note that in a decomposable set φ(Z) a new valid decomposition is obtained by adding the same tangent vector to all v_{z_i} and subtracting it from all v_{z_j}, for any i ≠ j. However, we can guarantee the uniqueness of the factorization by imposing a centering constraint.

**Lemma 1.** Let φ(Z) be a geodesically decomposable set. Then there exist unique vectors v_{z_i} ∈ T_µ M for all z_i ∈ Z_i such that Σ_{z_i ∈ Z_i} v_{z_i} = 0 for all i = 1,..., s and Eq. (3) holds.

For an intuitive interpretation, the intrinsic mean µ of a decomposable set can be seen as the context of the decomposition, and each unique direction v_{z_i} represents the "universal meaning" of the primitive concept z_i relative to µ. These "universal directions" are combined by addition on the tangent space T_µ M. The exponential map of the resulting tangent vector defines the geodesic segment on the manifold M from µ to the corresponding composite meaning.

Our notion of geodesic decomposability is general and applicable to manifolds of any shape. It generalizes that of [67], which is equivalent to ours in the special case M = R^n, where the intrinsic mean is the arithmetic mean and the exponential and logarithmic maps behave like the identity function. Our manifold formalization agrees with the fact that lower-dimensional semantic subspaces in CLIP latent space are captured by submanifolds better than linear subspaces [46].

**Best decomposable approximation.** Decomposable sets live in a lower-dimension subspace of their manifold M. The dimension of Span({v_{z_i}}_{z_i ∈ Z_i}) is at most |Z_i| − 1 for all i = 1,..., s, implying the additive combinations of the primitive directions belong to a subspace of dimension at most Σ_i (|Z_i| − 1). This suggests that a generic set of embeddings {u_z} is unlikely to be perfectly decomposable. We thus search for its best decomposable approximation, that is the set {ũ_z} that minimizes the error:

```
Σ_{z ∈ Z} d_M(u_z, ũ_z)²    (4)
```

In general, this is a hard problem to solve. Similarly to the standard solution to Principal Geodesic Analysis [15], we use Eq. (1) to approximate the objective in the "simpler" Euclidean space T_µ M, and rewrite Eq. (4) as:

```
Σ_{z ∈ Z} ||Log_µ(u_z) − Log_µ(ũ_z)||²    (5)
```

The solution to the approximate problem is obtained by computing vector means in T_µ M. For a fixed primitive concept z_i ∈ Z_i, let Z(z_i) = {(z_1',..., z_r') ∈ Z | z_i' = z_i} denote the slice of Z containing all tuples with the i-th component equal to z_i.

**Proposition 1.** Given a set φ(Z) = {u_z | z ∈ Z} ⊂ M with intrinsic mean µ, the minimization problem

```
argmin_{{ũ_z}} Σ_{z ∈ Z} ||Log_µ(u_z) − Log_µ(ũ_z)||²    (6)
s.t. {ũ_z} is geodesically decomposable
```

is solved by ũ_z = Exp_µ(v_{z_1} + ··· + v_{z_r}), where

```
v_{z_i} = (1/|Z(z_i)|) Σ_{z ∈ Z(z_i)} Log_µ(u_z)    (7)
```

Moreover, Σ_{z_i ∈ Z_i} v_{z_i} = 0 for all i = 1,..., s.

This result tells us that each vector v_{z_i} in the optimal decomposition is the tangent mean of all the input compositions including the primitive z_i. Moreover, the choice of the intrinsic mean as the point of tangency guarantees the uniqueness constraint is satisfied.

### 3.3 Decomposable Embeddings of Visual Inputs

Our framework holds for arbitrary manifolds and for any embedding map, hence being independent of the input modality. However, collections of natural visual data contain noise and are sparse. We account for these properties in our framework as presented in the following.

#### 3.3.1 Removing Noise from Finite Image Sets

We refer to **noise** as information carried by images in addition to the composite concept of interest. For example, an image from the tuple z = (red, car) likely contains non-negligible extra information, e.g. a driver, a road, or a blue sky in the background. This stems from the inherent ambiguity and non-uniqueness of visual signals. Most importantly, it is absent in text, for which it is easier to manually craft the string "a red car" ensuring no extra information.

**Problem formulation.** Since images contain noise in addition to represented concepts, we consider an input set φ(Z × E) = {u_{(z,e)} | (z, e) ∈ Z × E} where each z is represented by k = |E| different image embeddings varying along the unknown noise dimension E. For each fixed z, we model this aspect with a probability distribution {p_{(z,e)}}_{e ∈ E} describing how well the elements in {u_{(z,e)}}_{e ∈ E} represent their label z. In this setting, we want the decomposable set {ũ_z}_{z ∈ Z} minimizing the objective:

```
Σ_{(z,e) ∈ Z × E} p_{(z,e)} d_M(u_{(z,e)}, ũ_z)²    (8)
```

where the importance given to the approximation error for each input embedding is weighted according to the noise distribution.

**Proposition 2.** Let p_{(z,e)}, (z, e) ∈ Z × E, be non-negative scalars such that Σ_{e ∈ E} p_{(z,e)} = 1 for each z ∈ Z, and let φ(Z × E) = {u_{(z,e)} | (z, e) ∈ Z × E} ⊂ M be a set of embeddings with weighted intrinsic mean µ w.r.t. the weights w_{(z,e)} = p_{(z,e)} / Σ_{(z,e)} p_{(z,e)}. The minimization problem

```
argmin_{{ũ_z}} Σ_{(z,e) ∈ Z × E} p_{(z,e)} ||Log_µ(u_{(z,e)}) − Log_µ(ũ_z)||²    (9)
s.t. {ũ_z} is geodesically decomposable
```

is solved by ũ_z = Exp_µ(v_{z_1} + ··· + v_{z_s}), where

```
v_{z_i} = (1/|Z(z_i)|) Σ_{z ∈ Z(z_i)} v_z,   v_z = Σ_{e ∈ E} p_{(z,e)} Log_µ(u_{(z,e)})    (10)
```

Moreover, Σ_{z_i ∈ Z_i} v_{z_i} = 0 for all i = 1,..., s.

The vectors v_z can be seen as denoised tangent representations of the tuples in Z. The solution {ũ_z}_{z ∈ Z} corresponds to the decomposable approximation given by Proposition 1 applied to the denoised embeddings {u_z := Exp_µ(v_z)}_{z ∈ Z}.

**Lemma 2.** Using the notation of Proposition 2, the set {u_z := Exp_µ(v_z)}_{z ∈ Z} has intrinsic mean µ.

**Noise distribution.** The described setup requires the noise scores p_{(z,e)}. Given a collection of visual inputs T representing each label z ∈ Z' with k_z > 0 elements, a simple choice is using uniform scores p_{(z,e)} = 1/k_z. Alternatively, we propose using the CLIP image-to-text distribution p_{(z,e)} = P((z,e)|y(z)), where y(z) is a text prompt for label z ∈ Z'. This is the softmax of the scaled similarities:

```
P((z,e)|y(z)) = exp(u_{(z,e)}^⊤ u_{y(z)} / t) / Σ_{e'} exp(u_{(z,e')}^⊤ u_{y(z)} / t)    (12)
```

The temperature parameter t is learned during training, but it can be tweaked to smooth or sharpen the distribution.

#### 3.3.2 Dealing with Sparsity in Finite Image Sets

The previously described setup assumes that every z ∈ Z is represented by k > 0 images. This requirement can be too restrictive in practice, because some combinations of primitives may not occur in real image collections. For example, if Z = {red, blue} × {car, apple}, there will probably be no pictures of a (blue, apple). We refer to the absence of composite concepts as **sparsity**. Once more, please note that sparsity is not an issue with text, since strings can be manually crafted for any z ∈ Z.

**Problem Formulation.** In general, in a labeled image collection, only a subset T ⊂ Z × E is available, and only a subgroup Z' ⊂ Z of composite concepts is represented by at least one element in T. In this scenario, we obtain a decomposable approximation of φ(T) by approximating the vector means in Eq. (10) with the mean of the available elements. The only requirement is that every primitive z_i ∈ Z_i (i = 1,..., s) appears in at least one tuple of Z'. Precisely, we first compute the weighted intrinsic mean µ of φ(T) with weights w_{(z,e)} = p_{(z,e)} / Σ_{(z,e) ∈ T} p_{(z,e)}, and then consider ũ_z = Exp_µ(v_{z_1} + ··· + v_{z_s}), where:

```
v_{z_i} = (1/|Z'(z_i)|) Σ_{z ∈ Z'(z_i)} v_z,   v_z = Σ_{e: (z,e) ∈ T} p_{(z,e)} Log_µ(u_{(z,e)})    (11)
```

Moreover, Σ_{z_i ∈ Z_i} v_{z_i} = 0 for all i = 1,..., s.

Note that the obtained decomposable set contains vector representations of all concepts in Z, including the unseen elements of Z \ Z'. The formulation in (11) deals with all aspects mentioned so far: the manifold M, noise, and sparsity.

**Attribute-object decomposition.** We usually deal with sparse collections of visual inputs T where only a subset Z' of labels present at least one image. Thus, we compute the embedding decomposition according to Eq. (11): the optimal vectors ũ_{(a,o)} = Exp_µ(v_a + v_o) are the combinations of the attribute directions v_a = (1/|Z'(a)|) Σ_o v_{(a,o)} and the object directions v_o = (1/|Z'(o)|) Σ_a v_{(a,o)}, where the denoised representations v_{(a,o)}, (a, o) ∈ Z', are the mean tangent vectors within pairs. For compositional classification and group robustness, we use the CLIP image-to-text probabilities as the noise distribution discussed in Sec. 3.3.2. We finetune the temperature parameter (see Appendix for details). In the other experiments, we utilize uniform scores.

## 4. Experimental Validation

We carry out experiments to analyze the decomposable properties of visual embeddings of VLMs. When not specified differently, we use the pre-trained CLIP ViT-L/14 [54]. We also consider CLIP ResNet50 [54] and SigLIP [72]. All considered models are from the OpenCLIP repository [8]. We use images with attribute-object labels to represent sets of composite concepts of the form Z = Z_attr × Z_obj.

In this setup, we first assess the decomposable nature of small sets of embeddings inspecting their geometric arrangement according to Proposition 2 (Sec. 4.1). Then, we leverage the structured nature of the decomposed embeddings and experiment on the tasks of compositional classification (Sec. 4.2) and group robustness (Sec. 4.3). Finally, we visualize the approximate decomposable embeddings using a diffusion model (Stable Diffusion v2.1 [57]) with the unCLIP technique [56] (Sec. 4.4).

**Datasets.** We represent composite concepts with images from the training sets of diverse compositional datasets. We test compositional classification on the typical benchmark datasets UT-Zappos [69] and MIT-states [25] with the splits from [53]. UT-Zappos contains images of shoes centered on a white background all sharing the same orientation. There are 12 object classes referring to the footwear type and 16 attribute categories referring to the material. MIT-states is a collection of natural objects in different states. The dataset contains 115 attribute categories and 245 object categories, generating a large number of possible combinations.

We test group robustness on the Waterbirds and CelebA datasets with the splits in [58]. These contain objects with spuriously correlated attributes, making them suitable for debiasing tasks. Waterbirds contains images of two bird species Z_obj = {waterbird, landbird} on two types of background Z_attr = {land, water}. We use the version of CelebA from [58] that contains close-up photos of celebrities labeled with hair-color Z_obj = {blonde, dark} and gender Z_attr = {male, female}. The data distribution over the four groups is highly unbalanced in the train sets of these two datasets, implying spurious correlations.

### 4.1 Visualizing Compositional Embeddings

We evaluate the decomposability of the embeddings from a geometric perspective. We visualize lower-dimensional PCA projections of the tangent vectors {v_{(a,o)}}, considering that the denoised representations u_{(a,o)} := Exp_µ(v_{(a,o)}) are geodesically decomposable if and only if their tangent directions are the vertices of a geometric shape with parallel faces. For example, decomposable sets of size |Z| = 2×2 and |Z| = 2×3 correspond to a parallelogram and a triangular prism, respectively.

By increasing the number k of images per pair, the noise is successfully removed and the resulting representations define shapes with parallel faces, indicating approximate geodesical decomposability. This highlights the importance of the denoising step and demonstrates compositional regularities of visual embeddings.

### 4.2 Compositional Classification

We perform compositional classification on the UT-Zappos and MIT-states datasets using the decomposable approximation of the train data as classifiers. This task serves to evaluate the generalization capabilities towards novel compositions of objects and states. Specifically, we follow the standard generalized zero-shot evaluation protocol in both closed-world and open-world scenarios [41]. We compute decomposable embeddings on a subset Z' ⊂ Z of seen pairs from the training set, while not all labels in the test set are in Z'. In the closed-world setting, the set of target labels Z_test ⊂ Z contains only the pairs appearing in the dataset, while in the open-world framework, no prior knowledge is assumed and all attribute-object combinations in Z_test = Z are considered. Both settings require generalizing the prior knowledge about the primitives to understand the unseen compositions in Z_test \ Z'. This operation is particularly challenging in the open-world scenario, where the more numerous novel compositions in the test set are a distraction for the predictor.

Our framework provides a straightforward solution to the complex problem of compositional classification. The geodesically decomposable set {ũ_{(a,o)}} computed with the full train data T represents all the pairs, including the unseen ones. Thus we classify an image x as argmax_{(a,o) ∈ Z_test} ũ_{(a,o)}^⊤ u_x. We evaluate the prediction with the standard metrics [7, 53]: attribute accuracy (ATTR), object accuracy (OBJ), best seen accuracy (SEEN), best unseen accuracy (UNSEEN), best harmonic mean (HM) between the seen and unseen accuracy, and area under the seen-unseen curve (AUC).

**Baselines.** Our primary goal is to examine if the Geodesically Decomposable Embeddings (GDE) approximating the train data contain semantically meaningful information about the composite concepts they represent. We evaluate the relative performance ρ (AUC ratio) obtained with decomposed embeddings w.r.t. the results achieved with the standard zero-shot baseline (CLIP) using the full-state embeddings (attribute-object labels are represented by the text embedding of "An image of a {a} {o}").

We investigate the importance of complying with data geometry and compare with the **Linearly Decomposable Embeddings (LDE)** proposed in [67], which we compute by setting M = R^n in our method, for both text and image modalities. We indicate the modality by adding "(TEXT)" or "(IMAGE)" next to method names. Decomposed text-embeddings are given by Proposition 1, as noise and sparsity belong only to visual data.

**Results.** In general, GDEs of visual data perform closely to the zero-shot full-state baseline, demonstrating they encode semantically meaningful information about the labels. Interestingly, on the UT-Zappos dataset, they improve the standard zero-shot approach by a large margin. We attribute this gap to the fact that in UT-Zappos numerous representations are used for the computation of each primitive direction on average (~1400 per attribute, ~1900 per object). In contrast, the MIT-states dataset contains noisy annotations [2] and on average fewer representations to compute the primitives (~260 per attribute, ~120 per object). The decomposition shows robustness to sparsity, as indicated by the good open-world unseen accuracy on the MIT-states datasets, for which seen pairs are less than 5% of the total. LDE for visual data performs much worse than GDE on both datasets and when ablating the VLM backbone. This indicates that image embeddings are not closely linearly decomposable, and highlights the importance of respecting the data geometry when dealing with the extra complexity given by noise and sparsity.

### 4.3 Group Robustness

Pre-trained VLMs produce biased representations, leading to zero-shot classifiers not robust to group shifts [73]. Our framework offers a training-free method to compute unbiased embeddings. We evaluate it on the group robustness benchmark presented in [58], which requires classifying an image without leveraging spurious correlations. In this setting, a set of target classes Z_obj has spurious correlations with a set of attributes Z_attr due to the highly unbalanced data distribution over the groups G = Z_attr × Z_obj. The goal is to obtain an object classifier that does not exploit spurious correlations, improving the average accuracy over all the groups (AVG) while keeping the (GAP) on the worst group accuracy (WG) small. We use the object embeddings ũ_o := Exp_µ(v_o), o ∈ Z_obj computed with our method to evaluate the group robustness performance. Intuitively, these embed only object representations that are not correlated with attribute-related spurious features. We thus predict the object class of an image x as argmax_{o ∈ Z_obj} ũ_o^⊤ u_x.

**Baselines.** In addition to the zero-shot CLIP and LDE method, we include two standard baselines that use labeled data, namely Empirical Risk Minimization (ERM) with linear probing [32] and ERM with feature adapters [16]. Furthermore, we compare with two recent methods improving the performance of VLMs, **Deep Feature Reweighting (DFR)** [30] and **Contrastive Adapters (CA)** [73], and with **FairerCLIP** [10] that performs debiasing of the frozen CLIP representations in the training-free setting like our method.

**Results.** GDE considerably outperforms CLIP and LDE, with an increase of WG accuracy on the Waterbirds and CelebA datasets of about 42 and 21.8, respectively. This indicates that our method effectively decomposes the embeddings of object and attribute primitives, producing robust classifiers. Notably, it achieves state-of-the-art WG accuracy and smaller Gap compared to all other methods that use labeled data, including the task-specific FairerCLIP. GDE is thus an effective training-free solution to compute unbiased embeddings. Furthermore, GDE demonstrates remarkable data-efficiency performance, achieving high results using limited amount of data. For example, when using 25% of the full train samples (randomly selected keeping group ratios fixed) the WG decreases less than 1% on both the Waterbirds and CelebA datasets.

### 4.4 Visualize Decomposable Approximations

We visualize the decomposed visual embeddings using a diffusion model implementing the unCLIP mechanism (Stable Diffusion v2.1) [56, 57], trained to invert the CLIP image encoder by conditioning the generative process with the image embeddings. We invert the decomposable vectors ũ_{(a,o)} obtained in previous experiments to qualitatively examine the information they contain.

The generated images well represent both the object and the attribute of the label, with no difference in the quality of the outputs from seen (two leftmost columns) and unseen pairs (two rightmost columns). This emphasizes the generalization properties of our decomposable image embeddings, with potential to be applied in practical tasks like augmenting compositional sparse datasets.

The modularity of the decomposable structures allows representing the composition of two objects o_1, o_2 ∈ Z_obj as Exp_µ(v_{o_1} + v_{o_2}). Inspired by [39], we experiment with blending different animal species. The generated images portray photorealistic creatures with features of the two input species. This further highlights the power and versatility of the proposed framework.

## 5. Conclusion

We investigated the emergence of compositional structures within the image latent space of vision-language models and demonstrated that visual embeddings also exhibit a degree of compositionality similar to that of textual representations. We proposed a training-free framework, **Geodesically Decomposable Embeddings (GDE)**, designed to address the noisy and sparse nature of image data. GDE decomposes visual representations as a geometry-aware combination of optimal directions representing primitive concepts. We demonstrated that these composed representations encode complex concepts and are effective in several tasks, including compositional classification and group robustness. Notably, GDE presents more robust abilities to perform compositionality than existing approaches based on linear decomposition of latent spaces, contributing to higher results in group robustness than existing task-specific methods. We believe this work contributes to achieving better interpretability and controllability of modern VLMs.

**Acknowledgements.** This work was sponsored by the ERJU project, the EU Horizon project ELIAS (No. 101120237), Ministero delle Imprese e del Made in Italy (IPCEI Cloud DM 27 giugno 2022 – IPCEI-CL-0000007), and the FAIR - Future AI Research (PE00000013), funded by NextGeneration EU. The authors acknowledge the CINECA award under the ISCRA initiative for the availability of high-performance computing resources and support.

## References

[1] Jacob Andreas. Measuring compositionality in representation learning. In ICLR, 2019.

[2] Yuval Atzmon, Felix Kreuk, Uri Shalit, and Gal Chechik. A causal view of compositional zero-shot recognition. NeurIPS, 33:1462–1473, 2020.

[3] Alberto Baldrati, Lorenzo Agnolucci, Marco Bertini, and Alberto Del Bimbo. Zero-shot composed image retrieval with textual inversion. In ICCV, 2023.

[4] Moritz Böhle, Mario Fritz, and Bernt Schiele. B-cos networks: Alignment is all we need for interpretability. In CVPR, 2022.

[5] Duygu Ceylan, Chun-Hao P Huang, and Niloy J Mitra. Pix2video: Video editing using image diffusion. In ICCV, 2023.

[6] Sarah Chabal and Viorica Marian. Speakers of different languages process the visual world differently. Journal of Experimental Psychology: General, 144(3):539–550, 2015.

[7] Wei-Lun Chao, Soravit Changpinyo, Boqing Gong, and Fei Sha. An empirical study and analysis of generalized zero-shot learning for object recognition in the wild. In ECCV, pages 52–68. Springer, 2016.

[8] Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In CVPR, pages 2818–2829, 2023.

[9] Konrad Czechowski, Tomasz Odrzygozdz, Marek Zbysinski, Michał Zawalski, Krzysztof Olejnik, Yuhuai Wu, Łukasz Kucinski, and Piotr Miłos. Subgoal search for complex reasoning tasks. NeurIPS, 2021.

[10] Sepehr Dehdashtian, Lan Wang, and Vishnu Naresh Boddeti. FairerCLIP: Debiasing zero-shot predictions of CLIP in RKHSS. In ICLR, 2024.

[11] Karan Desai, Maximilian Nickel, Tanmay Rajpurohit, Justin Johnson, and Shanmukha Ramakrishna Vedantam. Hyperbolic image-text representations. In ICML, pages 7694–7731. PMLR, 2023.

[12] Jacob Feldman. Probabilistic origins of compositional mental representations. Psychological Review, 131(3):599–624, 2024.

[13] Pedro Felzenszwalb, David McAllester, and Deva Ramanan. A discriminatively trained, multiscale, deformable part model. In CVPR, 2008.

[14] Martin A Fischler and Robert A Elschlager. The representation and matching of pictorial structures. IEEE Transactions on Computers, 100(1):67–92, 1973.

[15] P Thomas Fletcher, Conglin Lu, Stephen M Pizer, and Sarang Joshi. Principal geodesic analysis for the study of nonlinear statistics of shape. IEEE TMI, 23(8):995–1005, 2004.

[16] Peng Gao, Shijie Geng, Renrui Zhang, Teli Ma, Rongyao Fang, Yongfeng Zhang, Hongsheng Li, and Yu Qiao. Clip-adapter: Better vision-language models with feature adapters. IJCV, 132(2):581–595, 2024.

[17] Songwei Ge, Shlok Mishra, Simon Kornblith, Chun-Liang Li, and David Jacobs. Hyperbolic contrastive learning for visual representations beyond objects. In CVPR, pages 6840–6849, 2023.

[18] Yunye Gong, Srikrishna Karanam, Ziyan Wu, Kuan-Chuan Peng, Jan Ernst, and Peter C Doerschuk. Learning compositional visual concepts with mutual consistency. In CVPR, 2018.

[19] Alon Hafri, E.J. Green, and Chaz Firestone. Compositionality in visual perception. Behavioral and Brain Sciences, 46:e277, 2023.

[20] Irina Higgins, Nicolas Sonnerat, Loic Matthey, Arka Pal, Christopher P Burgess, Matko Bosnjak, Murray Shanahan, Matthew Botvinick, Demis Hassabis, and Alexander Lerchner. SCAN: Learning hierarchical compositional visual concepts. In ICLR, 2018.

[21] Zhi Hou, Xiaojiang Peng, Yu Qiao, and Dacheng Tao. Visual compositional learning for human-object interaction detection. In ECCV, 2020.

[22] Zhi Hou, Baosheng Yu, and Dacheng Tao. Discovering human-object interaction concepts via self-compositional learning. In ECCV, 2022.

[23] Cheng-Yu Hsieh, Jieyu Zhang, Zixian Ma, Aniruddha Kembhavi, and Ranjay Krishna. SugarCrepe: Fixing hackable benchmarks for vision-language compositionality. NeurIPS, 2023.

[24] Drew A Hudson and Christopher D Manning. Compositional attention networks for machine reasoning. In ICLR, 2018.

[25] Phillip Isola, Joseph J. Lim, and Edward H. Adelson. Discovering states and transformations in image collections. In CVPR, 2015.

[26] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In ICML, pages 4904–4916, 2021.

[27] Shyamgopal Karthik, Karsten Roth, Massimiliano Mancini, and Zeynep Akata. Vision-by-language for training-free compositional image retrieval. In ICLR, 2024.

[28] Keizo Kato, Yin Li, and Abhinav Gupta. Compositional learning for human object interaction. In ECCV, 2018.

[29] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In CVPR, 2023.

[30] Polina Kirichenko, Pavel Izmailov, and Andrew Gordon Wilson. Last layer re-training is sufficient for robustness to spurious correlations. In ICLR, 2023.

[31] Jayanth Koushik, Hiroaki Hayashi, and Devendra Singh Sachan. Compositional reasoning for visual question answering. In ICML, 2017.

[32] Ananya Kumar, Aditi Raghunathan, Robbie Matthew Jones, Tengyu Ma, and Percy Liang. Fine-tuning can distort pre-trained features and underperform out-of-distribution. In ICLR, 2022.

[33] Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham Kakade, Prateek Jain, et al. Matryoshka representation learning. NeurIPS, 35:30233–30249, 2022.

[34] Kevin J. Lande. Compositionality in perception: A framework. WIREs Cognitive Science, 15(6):e1691, 2024.

[35] Martha Lewis, Nihal Nayak, Peilin Yu, Jack Merullo, Qinan Yu, Stephen Bach, and Ellie Pavlick. Does CLIP bind concepts? probing compositionality in large image models. In Findings of EACL 2024, pages 1487–1500, 2024.

[36] Xilai Li, Xi Song, and Tianfu Wu. AOGNets: Compositional grammatical architectures for deep learning. In CVPR, 2019.

[37] Yong-Lu Li, Yue Xu, Xiaohan Mao, and Cewu Lu. Symmetry and group in attribute-object compositions. In CVPR, 2020.

[38] Renjie Liao, Alex Schwing, Richard Zemel, and Raquel Urtasun. Learning deep parsimonious representations. NeurIPS, 2016.

[39] Giorgio Longari, Lorenzo Olearo, Simone Melzi, Rafael Penaloza, and Alessandro Raganato. How to blend concepts in diffusion models. arXiv preprint arXiv:2407.14280, 2024.

[40] Xiaocheng Lu, Song Guo, Ziming Liu, and Jingcai Guo. Decomposed soft prompt guided fusion enhancing for compositional zero-shot learning. In CVPR, 2023.

[41] Massimiliano Mancini, Muhammad Ferjad Naeem, Yongqin Xian, and Zeynep Akata. Open world compositional zero-shot learning. In CVPR, pages 5222–5230, 2021.

[42] Massimiliano Mancini, Muhammad Ferjad Naeem, Yongqin Xian, and Zeynep Akata. Learning graph embeddings for open world compositional zero-shot learning. IEEE TPAMI, 46(3):1545–1560, 2022.

[43] Ishan Misra, Abhinav Gupta, and Martial Hebert. From red wine to red tomato: Composition with context. In CVPR, 2017.

[44] Tushar Nagarajan and Kristen Grauman. Attributes as operators: factorizing unseen attribute-object compositions. In ECCV, 2018.

[45] Nihal V. Nayak, Peilin Yu, and Stephen Bach. Learning to compose soft prompts for compositional zero-shot learning. In ICLR, 2023.

[46] James Oldfield, Christos Tzelepis, Yannis Panagakis, Mihalis Nicolaou, and Ioannis Patras. Parts of speech–grounded subspaces in vision-language models. NeurIPS, 36:2700–2724, 2023.

[47] Bjorn Ommer and Joachim Buhmann. Learning the compositional nature of visual object categories for recognition. IEEE TPAMI, 32(3):501–516, 2009.

[48] Bjorn Ommer and Joachim M Buhmann. Learning the compositional nature of visual objects. In CVPR, 2007.

[49] Dim P Papadopoulos, Youssef Tamaazousti, Ferda Ofli, Ingmar Weber, and Antonio Torralba. How to make a pizza: Learning a compositional layer-based GAN model. In CVPR, 2019.

[50] Barbara Partee et al. Lexical semantics and compositionality. An invitation to cognitive science: Language, 1:311–360, 1995.

[51] Barbara H Partee. Compositionality in formal semantics: Selected papers. John Wiley & Sons, 2008.

[52] Pramuditha Perera, Matthew Trager, Luca Zancato, Alessandro Achille, and Stefano Soatto. Prompt algebra for task composition. arXiv preprint arXiv:2306.00310, 2023.

[53] Senthil Purushwalkam, Maximilian Nickel, Abhinav Gupta, and Marc'Aurelio Ranzato. Task-driven modular networks for zero-shot compositional learning. In ICCV, 2019.

[54] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763. PMLR, 2021.

[55] Sameera Ramasinghe, Violetta Shevchenko, Gil Avraham, and Ajanthan Thalaiyasingam. Accept the modality gap: An exploration in the hyperbolic space. In CVPR, pages 27263–27272, 2024.

[56] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with CLIP latents. arXiv preprint arXiv:2204.06125, 2022.

[57] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, pages 10684–10695, 2022.

[58] Shiori Sagawa, Pang Wei Koh, Tatsunori B. Hashimoto, and Percy Liang. Distributionally robust neural networks. In ICLR, 2020.

[59] Kuniaki Saito, Kihyuk Sohn, Xiang Zhang, Chun-Liang Li, Chen-Yu Lee, Kate Saenko, and Tomas Pfister. Pic2word: Mapping pictures to words for zero-shot composed image retrieval. In CVPR, 2023.

[60] Sascha Saralajew, Lars Holdijk, Maike Rees, Ebubekir Asan, and Thomas Villmann. Classification-by-components: Probabilistic modeling of reasoning over a set of components. NeurIPS, 2019.

[61] Jürgen Schmidhuber. Towards compositional learning in dynamic networks. Technical University of Munich (Technical Report FKI-129-90), 1990.

[62] Zhangzhang Si and Song-Chun Zhu. Learning and-or templates for object recognition and detection. IEEE TPAMI, 35(9):2189–2205, 2013.

[63] Austin Stone, Huayan Wang, Michael Stark, Yi Liu, D Scott Phoenix, and Dileep George. Teaching compositionality to CNNs. In CVPR, 2017.

[64] Fuwen Tan, Song Feng, and Vicente Ordonez. Text2scene: Generating compositional scenes from textual descriptions. In CVPR, 2019.

[65] Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet Singh, Adina Williams, Douwe Kiela, and Candace Ross. Winoground: Probing vision and language models for visio-linguistic compositionality. In CVPR, 2022.

[66] Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? Exploring the visual shortcomings of multimodal LLMs. In CVPR, 2024.

[67] Matthew Trager, Pramuditha Perera, Luca Zancato, Alessandro Achille, Parminder Bhatia, and Stefano Soatto. Linear spaces of meanings: compositional structures in vision-language models. In ICCV, pages 15395–15404, 2023.

[68] Jianyu Wang and Alan L Yuille. Semantic part segmentation using compositional model combining shape and appearance. In CVPR, 2015.

[69] Aron Yu and Kristen Grauman. Fine-grained visual comparisons with local learning. In CVPR, pages 192–199, 2014.

[70] Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, and James Zou. When and why vision-language models behave like bags-of-words, and what to do about it? In ICLR, 2023.

[71] Tian Yun, Usha Bhalla, Ellie Pavlick, and Chen Sun. Do vision-language pretrained models learn composable primitive concepts? TMLR, 2023.

[72] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In ICCV, 2023.

[73] Michael Zhang and Christopher Ré. Contrastive adapters for foundation model group robustness. NeurIPS, 35:21682–21697, 2022.

[74] Yizhen Zhang, Minkyu Choi, Kuan Han, and Zhongming Liu. Explainable semantic space by grounding language to vision with cross-modal contrastive learning. Advances in Neural Information Processing Systems, 34:18513–18526, 2021.

[75] Zhixing Zhang, Ligong Han, Arnab Ghosh, Dimitris N Metaxas, and Jian Ren. SINE: Single image editing with text-to-image diffusion models. In CVPR, 2023.

[76] Bo Zhao, Bo Chang, Zequn Jie, and Leonid Sigal. Modular generative adversarial networks. In ECCV, 2018.
