# Not Only Text: Exploring Compositionality of Visual Representations in Vision-Language Models

**Authors:** Davide Berasi¹, Matteo Farina², Massimiliano Mancini², Elisa Ricci¹˒², Nicola Strisciuglio³

**Affiliations:** ¹Fondazione Bruno Kessler · ²University of Trento · ³University of Twente

**Corresponding author:** dberasi@fbk.eu

**Preprint:** arXiv:2503.17142v1 [cs.CV], 21 March 2025

**Code:** <https://github.com/BerasiDavide/vlm_image_compositionality>

---

## Abstract

Vision-Language Models (VLMs) learn a shared feature space for text and images, enabling the comparison of inputs of different modalities. While prior works demonstrated that VLMs organize natural language representations into regular structures encoding composite meanings, it remains unclear if compositional patterns also emerge in the visual embedding space.

In this work, we investigate compositionality in the image domain, where the analysis of compositional properties is challenged by noise and sparsity of visual data. We address these problems and propose a framework, called **Geodesically Decomposable Embeddings (GDE)**, that approximates image representations with geometry-aware compositional structures in the latent space.

We demonstrate that visual embeddings of pre-trained VLMs exhibit a compositional arrangement, and evaluate the effectiveness of this property in the tasks of compositional classification and group robustness. GDE achieves stronger performance in compositional classification compared to its counterpart method that assumes linear geometry of the latent space. Notably, it is particularly effective for group robustness, where we achieve higher results than task-specific solutions.

Our results indicate that VLMs can automatically develop a human-like form of compositional reasoning in the visual domain, making their underlying processes more interpretable.

---

## 1. Introduction

Compositionality is the principle by which cognitive and computational systems create the meaning of a complex expression by combining the meaning of its (simpler) parts. Humans leverage compositionality instinctively, combining known elements to interpret novel situations. In machine intelligence, efforts were made to replicate this capability by developing models that imitate compositional processes, e.g.:

- solving complex tasks via sub-goals,
- modeling objects as compositions of their parts,
- encoding concept hierarchies,
- explicitly learning compositional representations, or
- designing compositional architectures.

With the rise of modern Vision-Language Models (VLMs) jointly trained on large-scale image-text pairs, there has been growing interest in investigating whether these models exhibit intrinsic compositional behaviors. In particular, Trager et al. investigated latent compositional structures within the CLIP text embedding space, demonstrating that composite concepts can be represented as linear combinations of embedding vectors corresponding to various factors. These vectors, called *ideal words*, can be used to compose new concepts in the embedding space. Their work focuses on finding compositional structures in the **text** embedding space of CLIP, motivated by the fact that the structured and symbolic nature of language may facilitate the study of computational approaches to capture compositional meaning.

However, cognitive studies show that language itself is used to describe and interpret the visual world and directly affects visual perception. Hence, similar to text, human visual representations exhibit a compositional structure, made of simpler components systematically combined. Despite this connection, compositional properties of visual embeddings of VLMs have remained so far mostly unexplored.

To fill this gap, we introduce **Geodesically Decomposable Embeddings (GDE)**, a framework grounded in differential geometry and designed to investigate compositional structures of pre-trained embeddings within Riemannian manifolds. Visual embeddings exhibit unique challenges not present in the compositional analysis of text embeddings, namely:

- **Sparsity** of composite concepts: certain combinations of elementary primitives may not appear in real image collections (e.g., focusing on objects and attributes, "blue dog" images are unlikely to exist).
- **Noise and ambiguity**: additional visual cues and information present in images (e.g., background, context) that do not correspond to the composite concepts.

We evaluated the compositional representations computed with the proposed approach in two relevant applications, namely **compositional classification** and **group robustness**, on publicly available datasets, showing that it better captures visual compositional structures than the alternatives. GDE is particularly effective for group robustness, where we achieve better debiasing results than task-specific methods. Furthermore, we show that GDE can be successfully used in combination with state-of-the-art generative models to synthesize images of compositional concepts.

### Contributions

1. We study compositional structures within visual embeddings for VLMs and demonstrate that the latent representations of visual signals also exhibit a degree of compositionality.
2. We show that, unlike for text embeddings, linear structures are insufficient to (de)compose visual concepts; thus, the manifold geometry must be considered.
3. We propose a framework that deals with the sparsity and noise of composite concepts in images, enabling the compositional analysis of visual embeddings.

---

## 2. Related Work

### 2.1 Compositionality in Vision

Compositionality is considered a cornerstone of perception, and compositional representations offer an effective tool to represent real-world phenomena. The primary benefit of compositionality is the possibility of combining the representation of simpler concepts to understand and reason on complex ones, allowing for generalization to new unseen combinations of concepts.

In computer vision, early efforts focused on recognizing objects as a composition of parts and evolved into architectures that can recognize and model objects in a compositional fashion, compositional generation, and interpretable representations. Compositionality has also led to progress in various tasks, such as human-object interaction detection, modeling spatial/semantic relationships, and compositional zero-shot learning, where the goal is to recognize unseen compositions of training primitives. While these works focus on specific applications, in this paper we aim to study whether there exists an underlying compositional structure in the visual embeddings of VLMs.

### 2.2 Compositionality in VLMs

Modern Vision-Language Models like CLIP are trained to extract meaningful representations from complex visual scenes guided by textual inputs, without a priori imposing any form of compositionality. In this context, a natural question is: *Does compositional behavior emerge automatically in VLMs?*

Previous works already showed how VLMs are more suitable for tasks such as compositional zero-shot learning, and how their representations allow for cross-modal compositions, such as visual editing and compositional retrieval. At the same time, works studied the challenges of VLMs in modeling compositional inputs, e.g., at the level of word order, object-attribute bindings, spatial relationships, and other compositional challenges.

In this paper, we study the compositional structure in the **visual** embeddings extracted from VLMs. Close to our goal is the work studying the compositional properties of the CLIP text encoder through compositional distributional semantics models in synthetic test scenarios. Similarly, Trager et al. show that the textual embeddings of VLMs can be well approximated by linear compositions of smaller sets of *ideal vectors*. Motivated by the cross-modal alignment of VLMs, we investigate whether the embeddings of visual inputs exhibit an analogous compositional property. We achieve this by constructing a geometry-aware decomposition framework, following ideas similar to Principal Geodesic Analysis (PGA), which is applied to learn lower-dimensional submanifolds of the CLIP sphere that are associated with distinct parts-of-speech. To the best of our knowledge, this is the first work that investigates the emergence of compositional structures in the visual embeddings of VLMs.

---

## 3. Method

We propose a framework to analyze the compositional properties of image embeddings of neural encoders. We start by reviewing the fundamentals of the CLIP model along with key concepts from differential geometry (Sec. 3.1). We then formalize the concept of geodesic decomposability (Sec. 3.2) and discuss our methodology for dealing with visual inputs (Sec. 3.3).

### 3.1 Preliminaries

#### Contrastive Language-Image Pretraining (CLIP)

CLIP consists of a pre-trained image encoder $\phi_{im}: \mathcal{X} \to \mathbb{R}^d$ and a text encoder $\phi_t: \mathcal{Y} \to \mathbb{R}^d$ that represent multi-modal text-visual inputs in a shared vision-language space. The latent representations of an image $x \in \mathcal{X}$ and text $y \in \mathcal{Y}$ are compared by cosine similarity, which is the scalar product $u_x^\top u_y$ of their normalized versions:

$$u_x = \frac{\phi_{im}(x)}{\lVert \phi_{im}(x) \rVert}, \qquad u_y = \frac{\phi_t(y)}{\lVert \phi_t(y) \rVert}$$

The weights of the encoders are trained to optimize a contrastive objective on a huge collection of paired image-text samples. Since the norm of CLIP embeddings does not carry any meaningful information, **spherical geometry** applies to their post-hoc analysis.

#### Riemannian Manifolds

Riemannian manifolds are geometric spaces where intrinsic distances can be measured. For a generic manifold $\mathcal{M} \subset \mathbb{R}^d$ with intrinsic distance $d_\mathcal{M}: \mathcal{M} \times \mathcal{M} \to [0, \infty)$, we recall the notions of *exponential map* and *intrinsic mean*. These tools permit operating with non-linear data, like the spherical normalized CLIP embeddings, while respecting their intrinsic shape.

Let $\mu$ be a point on $\mathcal{M}$ and let $T_\mu \mathcal{M}$ be the tangent space at $\mu$. The **exponential map** projects a tangent vector $v \in T_\mu \mathcal{M}$ onto the manifold by moving along the geodesic segment it defines. Formally, if $\gamma_v : [0,1] \to \mathcal{M}$ is the unique geodesic path starting from $\gamma_v(0) = \mu$ with initial velocity $\dot\gamma_v(0) = v$, then $\mathrm{Exp}_\mu(v) := \gamma_v(1)$. This function is locally invertible and its inverse is the **logarithmic map** $\mathrm{Log}_\mu = \mathrm{Exp}_\mu^{-1}$.

The exponential and logarithmic maps send straight lines of the tangent plane into geodesic curves of the manifold, and vice-versa. Moreover, they approximately preserve distances between elements close to the point of tangency $\mu$:

$$d_\mathcal{M}(u, u') \approx \lVert \mathrm{Log}_\mu(u) - \mathrm{Log}_\mu(u') \rVert, \qquad u, u' \in \mathcal{M} \tag{1}$$

Note that in Eq. (1) the equality holds if $u = \mu$ or $u' = \mu$. When applying the logarithmic map to a set of points $\{u_i\}_{i=1}^N \subset \mathcal{M}$, the natural choice for the point of tangency $\mu$ is the **intrinsic mean**, i.e., the element of $\mathcal{M}$ minimizing the average squared distance to the given points. In a more general definition, each point $u_i$ is associated with a scalar weight $w_i$ belonging to a probability-simplex vector $\Delta$, and the (weighted) intrinsic mean is:

$$\mu = \arg\min_{u \in \mathcal{M}} \sum_{i=1}^N w_i \, d_\mathcal{M}(u, u_i)^2 \tag{2}$$

This distance-minimizing element $\mu$ guarantees that the images of the points through the logarithmic map are centered in the origin of the tangent space: $\sum_i w_i \mathrm{Log}_\mu(u_i) = 0$.

### 3.2 Geodesically Decomposable Embeddings

We consider a set of composite meanings $\mathcal{Z} = \mathcal{Z}_1 \times \cdots \times \mathcal{Z}_s$, defined as the Cartesian product between finite lists of primitive concepts, and refer to the $\mathcal{Z}_i$ ($i = 1, \dots, s$) as the **dimensions** of $\mathcal{Z}$. For example, $\mathcal{Z} = \{\text{red}, \text{blue}\} \times \{\text{car}, \text{dress}, \text{flower}\}$ combines primitives from an attribute dimension and an object dimension.

We then consider an embedding map $\phi: \mathcal{Z} \to \mathcal{M}$ representing the composite concepts as points on a manifold $\mathcal{M} \subset \mathbb{R}^d$. Intuitively, the set $\phi(\mathcal{Z}) = \{u_z \mid z \in \mathcal{Z}\}$ is **compositional** if it has a regular structure reflecting the composite nature of the inputs, i.e., if one can compose primitive concepts within the geometric space to obtain embeddings of complex meanings. In this paper, we associate compositionality with the notion of **geodesic decomposability**, which accounts for the intrinsic geometry of the manifold.

> **Definition 1 (Geodesically decomposable embeddings).**
> A set of embeddings $\phi(\mathcal{Z}) = \{u_z \mid z \in \mathcal{Z}\} \subset \mathcal{M}$ with intrinsic mean $\mu$ is *geodesically decomposable* if there exist $v_{z_i} \in T_\mu \mathcal{M}$ for all $z_i \in \mathcal{Z}_i$ ($i = 1, \dots, s$) such that
> $$u_z = \mathrm{Exp}_\mu\!\left(v_{z_1} + \cdots + v_{z_s}\right) \qquad \forall\, z = (z_1, \dots, z_s) \tag{3}$$

Note that in a decomposable set $\phi(\mathcal{Z})$ a new valid decomposition is obtained by adding the same tangent vector to all $v_{z_i}$ and subtracting it from all $v_{z_j}$, for any $i \neq j$. However, we can guarantee the uniqueness of the factorization by imposing a centering constraint.

> **Lemma 1.**
> Let $\phi(\mathcal{Z})$ be a geodesically decomposable set. Then there exist unique vectors $v_{z_i} \in T_\mu \mathcal{M}$ for all $z_i \in \mathcal{Z}_i$ such that $\sum_{z_i \in \mathcal{Z}_i} v_{z_i} = 0$ for all $i = 1, \dots, s$ and Eq. (3) holds.

For an intuitive interpretation, the intrinsic mean $\mu$ of a decomposable set can be seen as the **context** of the decomposition, and each unique direction $v_{z_i}$ relative to $\mu$ represents the **meaning** of the primitive concept $z_i$. These "universal directions" are combined by addition on the tangent space $T_\mu \mathcal{M}$. The exponential map of the resulting tangent vector defines the geodesic segment on the manifold $\mathcal{M}$ from $\mu$ to the corresponding composite meaning.

Our notion of geodesic decomposability is general and applicable to manifolds of any shape. It generalizes the linear formulation of Trager et al., which is equivalent to ours in the special case $\mathcal{M} = \mathbb{R}^n$, where the intrinsic mean is the arithmetic mean and the exponential and logarithmic maps behave like the identity function. Our manifold formalization agrees with the fact that lower-dimensional semantic subspaces in the CLIP latent space are captured by submanifolds better than by linear subspaces.

#### Best decomposable approximation

Decomposable sets live in a lower-dimension subspace of their manifold $\mathcal{M}$. The dimension of $\mathrm{Span}(\{v_{z_i}\}_{z_i \in \mathcal{Z}_i})$ is indeed at most $|\mathcal{Z}_i| - 1$ for all $i = 1, \dots, s$, implying the additive combinations of the primitive directions belong to a subspace of dimension at most $\sum_i (|\mathcal{Z}_i| - 1)$. This suggests that a generic set of embeddings $\{u_z\}$ is unlikely to be perfectly decomposable. We thus search for its **best decomposable approximation**, that is, the set $\{\tilde u_z\}$ that minimizes the error:

$$\sum_{z \in \mathcal{Z}} d_\mathcal{M}(u_z, \tilde u_z)^2 \tag{4}$$

In general, this is a hard problem to solve. Similarly to the standard solution to Principal Geodesic Analysis, we use Eq. (1) to approximate the objective in the "simpler" Euclidean space $T_\mu \mathcal{M}$, and rewrite Eq. (4) as:

$$\sum_{z \in \mathcal{Z}} \lVert \mathrm{Log}_\mu(u_z) - \mathrm{Log}_\mu(\tilde u_z) \rVert^2 \tag{5}$$

The solution to the approximate problem is obtained by computing vector means in $T_\mu \mathcal{M}$, as described in the next proposition. For a fixed primitive concept $z_i \in \mathcal{Z}_i$, let $\mathcal{Z}(z_i) = \{(z_1', \dots, z_r') \in \mathcal{Z} \mid z_i' = z_i\}$ denote the **slice** of $\mathcal{Z}$ containing all tuples with the $i$-th component equal to $z_i$.

> **Proposition 1.**
> Given a set $\phi(\mathcal{Z}) = \{u_z \mid z \in \mathcal{Z}\} \subset \mathcal{M}$ with intrinsic mean $\mu$, the minimization problem
> $$\arg\min_{\{\tilde u_z\}} \sum_{z \in \mathcal{Z}} \lVert \mathrm{Log}_\mu(u_z) - \mathrm{Log}_\mu(\tilde u_z) \rVert^2 \quad \text{s.t. } \{\tilde u_z\} \text{ is geodesically decomposable} \tag{6}$$
> is solved by $\tilde u_z = \mathrm{Exp}_\mu(v_{z_1} + \cdots + v_{z_r})$, where
> $$v_{z_i} = \frac{1}{|\mathcal{Z}(z_i)|} \sum_{z \in \mathcal{Z}(z_i)} \mathrm{Log}_\mu(u_z) \tag{7}$$
> Moreover, $\sum_{z_i \in \mathcal{Z}_i} v_{z_i} = 0$ for all $i = 1, \dots, s$.

This result tells us that each vector $v_{z_i}$ in the optimal decomposition is the **tangent mean** of all the input compositions including the primitive $z_i$. Moreover, the choice of the intrinsic mean as the point of tangency guarantees the uniqueness constraint is satisfied (see Appendix B).

### 3.3 Decomposable Embeddings of Visual Inputs

Our framework holds for arbitrary manifolds and for any embedding map, hence being independent of the input modality. However, collections of natural visual data contain noise and are sparse. We account for these properties in our framework as presented in the following.

#### 3.3.1 Removing noise from finite image sets

We refer to **noise** as information carried by images in addition to the composite concept of interest. For example, an image from the tuple $z = (\text{red}, \text{car})$ likely contains non-negligible extra information, e.g., a driver, a road, or a blue sky in the background. This stems from the inherent ambiguity and non-uniqueness of visual signals. Most importantly, it is absent in text, for which it is easier to manually craft the string "a red car" ensuring no extra information.

**Problem formulation.** Since images contain noise in addition to the represented concepts, we consider an input set $\phi(\mathcal{Z} \times \mathcal{E}) = \{u_{(z,e)} \mid (z,e) \in \mathcal{Z} \times \mathcal{E}\}$ where each $z \in \mathcal{Z}$ is represented by $k = |\mathcal{E}|$ different image embeddings varying along the unknown noise dimension $\mathcal{E}$. Also, different images may contain different amounts of noise. For each fixed $z$, we model this aspect with a probability distribution $\{p_{(z,e)}\}_{e \in \mathcal{E}}$ describing how well the elements in $\{u_{(z,e)}\}_{e \in \mathcal{E}}$ represent their label $z$. In this setting, we want the decomposable set $\{\tilde u_z\}_{z \in \mathcal{Z}}$ minimizing the objective:

$$\sum_{(z,e) \in \mathcal{Z} \times \mathcal{E}} p_{(z,e)} \, d_\mathcal{M}(u_{(z,e)}, \tilde u_z)^2 \tag{8}$$

where the importance given to the approximation error for each input embedding is weighted according to the noise distribution. The next result generalizes Proposition 1 (which addresses the special case $k = 1$) and provides an easy-to-compute approximate solution.

> **Proposition 2.**
> Let $p_{(z,e)}$, $(z,e) \in \mathcal{Z} \times \mathcal{E}$, be non-negative scalars such that $\sum_{e \in \mathcal{E}} p_{(z,e)} = 1$ for each $z \in \mathcal{Z}$, and let $\phi(\mathcal{Z} \times \mathcal{E}) = \{u_{(z,e)} \mid (z,e) \in \mathcal{Z} \times \mathcal{E}\} \subset \mathcal{M}$ be a set of embeddings with weighted intrinsic mean $\mu$ w.r.t. the weights $w_{(z,e)} = p_{(z,e)} / \sum_{(z,e)} p_{(z,e)}$. The minimization problem
> $$\arg\min_{\{\tilde u_z\}} \sum_{(z,e) \in \mathcal{Z} \times \mathcal{E}} p_{(z,e)} \lVert \mathrm{Log}_\mu(u_{(z,e)}) - \mathrm{Log}_\mu(\tilde u_z) \rVert^2 \quad \text{s.t. } \{\tilde u_z\} \text{ is geodesically decomposable} \tag{9}$$
> is solved by $\tilde u_z = \mathrm{Exp}_\mu(v_{z_1} + \cdots + v_{z_s})$, where
> $$v_{z_i} = \frac{1}{|\mathcal{Z}(z_i)|} \sum_{z \in \mathcal{Z}(z_i)} v_z, \qquad v_z = \sum_{e \in \mathcal{E}} p_{(z,e)} \, \mathrm{Log}_\mu(u_{(z,e)}) \tag{10}$$
> Moreover, $\sum_{z_i \in \mathcal{Z}_i} v_{z_i} = 0$ for all $i = 1, \dots, s$.

Using the same notation, the vectors $v_z$ can be seen as a **denoised tangent representation** of the tuples in $\mathcal{Z}$, and the solution $\{\tilde u_z\}_{z \in \mathcal{Z}}$ to the weighted optimization problem corresponds to the decomposable approximation given by Proposition 1 applied to the denoised embeddings $\{u_z := \mathrm{Exp}_\mu(v_z)\}$. Indeed, these have intrinsic mean equal to the weighted intrinsic mean $\mu$.

> **Lemma 2.**
> Using the notation of Proposition 2, the set $\{u_z := \mathrm{Exp}_\mu(v_z)\}_{z \in \mathcal{Z}}$ has intrinsic mean $\mu$.

#### 3.3.2 Dealing with sparsity in finite image sets

The previously described setup assumes that every $z \in \mathcal{Z}$ is represented by $k > 0$ images. This requirement can be too restrictive in practice, because some combinations of primitives may not occur in real image collections. For example, if $\mathcal{Z} = \{\text{red}, \text{blue}\} \times \{\text{car}, \text{apple}\}$, there will probably be no pictures of a (blue, apple). We refer to the absence of composite concepts as **sparsity**. Once more, note that sparsity is not an issue with text, since strings can be manually crafted for any $z \in \mathcal{Z}$.

**Problem formulation.** In general, in a labeled image collection, only a subset $\mathcal{T} \subset \mathcal{Z} \times \mathcal{E}$ is available, and only a subgroup $\mathcal{Z}' \subset \mathcal{Z}$ of composite concepts is represented by at least one element in $\mathcal{T}$. In this scenario, we obtain a decomposable approximation of $\phi(\mathcal{T})$ by approximating the mean of the available vectors. The only requirement is that every primitive $z_i \in \mathcal{Z}_i$ ($i = 1, \dots, s$) appears in at least one tuple of $\mathcal{Z}'$.

Precisely, we first compute the weighted intrinsic mean $\mu$ representation of $\phi(\mathcal{T})$ with weights $w_{(z,e)} = p_{(z,e)} / \sum_{(z,e)} p_{(z,e)}$, and then consider $\tilde u_z = \mathrm{Exp}_\mu(v_{z_1} + \cdots + v_{z_s})$, where:

$$v_{z_i} = \frac{1}{|\mathcal{Z}'(z_i)|} \sum_{z \in \mathcal{Z}'(z_i)} v_z, \qquad v_z = \sum_{\substack{e \in \mathcal{E} \\ \text{s.t. } (z,e) \in \mathcal{T}}} p_{(z,e)} \, \mathrm{Log}_\mu(u_{(z,e)}) \tag{11}$$

Note that the obtained decomposable set contains vector representations of **all** the concepts in $\mathcal{Z}$, including the unseen elements of $\mathcal{Z} \setminus \mathcal{Z}'$. The formulation in Eq. (11) deals with all aspects mentioned so far: the manifold $\mathcal{M}$, noise, and sparsity. In the next section, we use it to evaluate the compositional structure of real visual embeddings.

**Noise distribution.** The described setup requires the noise scores $p_{(z,e)}$. Given a collection of visual inputs $\mathcal{T}$ representing each label $z \in \mathcal{Z}'$ with $k_z > 0$ elements, a simple choice is using uniform scores $p_{(z,e)} = 1/k_z$. Alternatively, we propose using the CLIP image-to-text distribution $p_{(z,e)} = P((z,e) \mid y(z))$, where $y(z)$ is a text prompt for label $z \in \mathcal{Z}'$. This is the softmax of the scaled similarities:

$$P((z,e) \mid y(z)) = \frac{\exp\!\left(u_{(z,e)}^\top u_{y(z)} / t\right)}{\sum_e \exp\!\left(u_{(z,e)}^\top u_{y(z)} / t\right)} \tag{12}$$

The temperature parameter $t$ is learned during training, but it can be tweaked to smooth or sharpen the distribution.

---

## 4. Experimental Validation

We carry out experiments to analyze the decomposable properties of visual embeddings of VLMs. When not specified differently, we use the pre-trained **CLIP ViT-L/14**. We also consider **CLIP ResNet50** and **SigLIP**. All considered models are from the OpenCLIP repository. We use images with attribute-object labels to represent sets of composite concepts of the form $\mathcal{Z} = \mathcal{Z}_{\text{attr}} \times \mathcal{Z}_{\text{obj}}$.

In this setup, we first assess the decomposable nature of small sets of embeddings by inspecting their geometric arrangement (Sec. 4.1). Then, we leverage the structured nature of the decomposed embeddings and experiment on the tasks of compositional classification (Sec. 4.2) and group robustness (Sec. 4.3). Finally, we visualize the approximate decomposable embeddings using a diffusion model (Stable Diffusion v2.1) with the unCLIP technique (Sec. 4.4).

**Attribute-object decomposition.** We usually deal with sparse collections of visual inputs $\mathcal{T}$ where only a subset $\mathcal{Z}'$ of labels presents at least one image. Thus, we compute the embedding decomposition according to Eq. (11): the optimal vectors are the combinations $v_{(a,o)} = v_a + v_o$ of the attribute directions $v_a = \frac{1}{|\mathcal{Z}'(a)|} \sum_o v_{(a,o)}$ and the object directions $v_o = \frac{1}{|\mathcal{Z}'(o)|} \sum_a v_{(a,o)}$, where the denoised vectors with pairs $(a,o) \in \mathcal{Z}'$ are the mean tangent representations. For compositional classification and group robustness, we use the CLIP image-to-text probabilities as the noise distribution (Sec. 3.3.2), finetuning the temperature parameter (see Appendix C.2). In the other experiments, we use uniform scores.

### 4.1 Datasets

We represent composite concepts with images from the training sets of diverse compositional datasets.

- **UT-Zappos** (compositional classification): images of shoes centered on a white background, all sharing the same orientation. There are 12 object classes referring to the footwear type and 16 attribute categories referring to the material.
- **MIT-states** (compositional classification): a collection of natural objects in different states. The dataset contains 115 attribute categories and 245 object categories, generating a large number of possible combinations.
- **Waterbirds** (group robustness): images of two bird species $\mathcal{Z}_{\text{obj}} = \{\text{waterbird}, \text{landbird}\}$ on two types of background $\mathcal{Z}_{\text{attr}} = \{\text{land}, \text{water}\}$.
- **CelebA** (group robustness): close-up photos of celebrities labeled with hair color $\mathcal{Z}_{\text{obj}} = \{\text{blonde}, \text{dark}\}$ and gender $\mathcal{Z}_{\text{attr}} = \{\text{male}, \text{female}\}$.

For UT-Zappos and MIT-states we use the splits commonly adopted in the literature. The Waterbirds and CelebA datasets contain objects with spuriously correlated attributes, making them suitable for debiasing tasks; the data distribution over the four different groups is highly unbalanced in their training sets, implying spurious correlations.

### 4.2 Visualizing Compositional Embeddings

We evaluated the decomposability of the embeddings from a geometric perspective. We visualize lower-dimensional PCA projections of the tangent vectors $\{v_{(a,o)}\}$, considering that the denoised representations $u_{(a,o)} := \mathrm{Exp}_\mu(v_{(a,o)})$ are geodesically decomposable if and only if their tangent directions are the **vertices of a geometric shape with parallel faces**. For example, decomposable sets of size $|\mathcal{Z}| = 2 \times 2$ and $|\mathcal{Z}| = 2 \times 3$ correspond to a parallelogram and a triangular prism, respectively.

Considering the four compositions of two attributes and two objects from Waterbirds, and the two-by-three concepts in the set $\{\text{leather}, \text{suede}\} \times \{\text{boots ankle}, \text{boots knee high}, \text{shoes flats}\}$ from UT-Zappos, we observe that by increasing the number $k$ of images per pair, the noise is successfully removed and the resulting representations define shapes with parallel faces, indicating approximate geodesic decomposability. This highlights the importance of the denoising step and demonstrates the compositional regularities of visual embeddings.

### 4.3 Compositional Classification

We perform compositional classification on UT-Zappos and MIT-states using the decomposable approximation of the train data as classifiers. This task evaluates the generalization capabilities towards novel compositions of objects and states. We follow the standard generalized zero-shot evaluation protocol in both **closed-world** and **open-world** scenarios.

We compute decomposable embeddings on a subset $\mathcal{Z}' \subset \mathcal{Z}$ of seen pairs from the training set, while not all labels in the test set are in $\mathcal{Z}'$.

- In the **closed-world** setting, the set of target labels $\mathcal{Z}_{\text{test}} \subset \mathcal{Z}$ contains only the pairs appearing in the dataset.
- In the **open-world** framework, no prior knowledge is assumed and all the attribute-object combinations in $\mathcal{Z}_{\text{test}} = \mathcal{Z}$ are considered.

Both settings require generalizing the prior knowledge about the primitives to understand the unseen compositions in $\mathcal{Z}_{\text{test}} \setminus \mathcal{Z}'$. This is particularly challenging in the open-world scenario, where the more numerous novel compositions in the test set are a distraction for the predictor.

Our framework provides a straightforward solution. The geodesically decomposable set $\{\tilde u_{(a,o)}\}$ computed with the full train data $\mathcal{T}$ represents all the pairs, including the unseen ones. Thus we classify an image $x$ as $\arg\max_{(a,o) \in \mathcal{Z}_{\text{test}}} \tilde u_{(a,o)}^\top u_x$. We evaluate the prediction with the standard metrics: attribute accuracy (ATTR), object accuracy (OBJ), best seen accuracy (SEEN), best unseen accuracy (UNSEEN), best harmonic mean (HM) between seen and unseen accuracy, and area under the seen-unseen curve (AUC).

#### Baselines

Our primary goal is to examine if the Geodesically Decomposable Embeddings (GDE) approximating the train data contain semantically meaningful information about the composite concepts they represent. We evaluate the relative performance $\rho$ (AUC ratio) obtained with decomposed embeddings w.r.t. the standard zero-shot baseline (CLIP) using the full-state embeddings (an attribute-object label $(a,o)$ is represented by the text embedding of *"An image of a {a} {o}"*).

We investigate the importance of complying with data geometry and compare with the **Linearly Decomposable Embeddings (LDE)**, which we compute by setting $\mathcal{M} = \mathbb{R}^n$ in our method, for both text and image modalities. We indicate the modality by adding "(TEXT)" or "(IMAGE)" next to the method names. Decomposed text embeddings are given by Proposition 1, as noise and sparsity belong only to visual data.

#### Results

GDEs of visual data perform closely to the zero-shot full-state baseline, demonstrating they encode semantically meaningful information about the labels. Key observations:

- **UT-Zappos.** GDE improves the standard zero-shot approach by a large margin. In the closed-world scenario, GDE (IMAGE) reaches an AUC of 13.9 versus 4.4 for the zero-shot CLIP baseline (a relative performance $\rho \approx 318\%$). We attribute this gap to the fact that in UT-Zappos numerous representations are used to compute each primitive direction on average (~1400 per attribute, ~1900 per object).
- **MIT-states.** This dataset contains noisy annotations and on average fewer representations to compute the primitives (~260 per attribute, ~120 per object). The decomposition still shows robustness to sparsity, as indicated by the good open-world unseen accuracy, for which seen pairs are less than 5% of the total.
- **GDE vs. LDE.** LDE for visual data performs much worse than GDE on both datasets and across VLM backbones (ResNet50, ViT-L/14, SigLIP). This indicates that image embeddings are **not** closely linearly decomposable, and highlights the importance of respecting the data geometry when dealing with the extra complexity given by noise and sparsity. This finding also holds on other geometries (see Appendix D.2).

### 4.4 Group Robustness

Pre-trained VLMs produce biased representations, leading to zero-shot classifiers that are not robust to group shifts. Our framework offers a **training-free** method to compute unbiased embeddings. We evaluate it on a standard group robustness benchmark, which requires classifying an image without leveraging spurious correlations.

In this setting, a set of target classes $\mathcal{Z}_{\text{obj}}$ has spurious correlations with a set of attributes $\mathcal{Z}_{\text{attr}}$ due to the highly unbalanced data distribution over the groups in $\mathcal{G} = \mathcal{Z}_{\text{attr}} \times \mathcal{Z}_{\text{obj}}$. The goal is to obtain an object classifier that does not exploit spurious correlations, improving the average accuracy over all groups (**AVG**) while keeping the gap (**GAP**) on the worst-group accuracy (**WG**) small.

We use the object embeddings $\tilde u_o := \mathrm{Exp}_\mu(v_o)$, $o \in \mathcal{Z}_{\text{obj}}$, computed with our method. Intuitively, these embed only object representations that are not correlated with attribute-related spurious features. We thus predict the object class of an image $x$ as $\arg\max_{o \in \mathcal{Z}_{\text{obj}}} \tilde u_o^\top u_x$.

#### Baselines

In addition to the zero-shot CLIP and the LDE method, we include two standard baselines that use labeled data: Empirical Risk Minimization (ERM) with linear probing and ERM with feature adapters. Furthermore, we compare with two recent methods improving the performance of VLMs — Deep Feature Reweighting (DFR) and Contrastive Adapters (CA) — and with FairerCLIP, which performs debiasing of the frozen CLIP representations in the training-free setting like our method.

#### Results

GDE considerably outperforms CLIP and LDE, with an increase in WG accuracy on Waterbirds and CelebA of about **42** and **21.8** points, respectively. This indicates that our method effectively decomposes the embeddings of object and attribute primitives, producing robust classifiers. Notably, it achieves **state-of-the-art WG accuracy and a smaller GAP** compared to all other methods that use labeled data, including the task-specific FairerCLIP. GDE is thus an effective training-free solution to compute unbiased embeddings.

Furthermore, GDE demonstrates remarkable data efficiency, achieving high results using a limited amount of data. For example, when using 25% of the full train samples (randomly selected, keeping group ratios fixed), the WG accuracy decreases by less than 1% on both Waterbirds and CelebA.

### 4.5 Visualizing Decomposable Approximations

We visualize the decomposed visual embeddings using a diffusion model implementing the unCLIP mechanism (Stable Diffusion v2.1), trained to invert the CLIP image encoder by conditioning the generative process with the image embeddings. We invert the decomposable vectors $\tilde u_{(a,o)}$ obtained in previous experiments, allowing us to qualitatively examine the information they contain.

We generate images for object-attribute pairs where the attribute is **not** the most common state of the object (i.e., we avoid common pairs like "green broccoli" or "big elephant"), observing whether the generated image correctly represents the full label and not just the attribute/object. The generated images well represent both the object and the attribute of the label, with no difference in the quality of the outputs from seen versus unseen pairs. This emphasizes the generalization properties of our decomposable image embeddings, with the potential to be applied to practical tasks like augmenting compositional sparse datasets.

The modularity of the decomposable structures also allows representing the composition of two objects $o_1, o_2 \in \mathcal{Z}_{\text{obj}}$ as $\mathrm{Exp}_\mu(v_{o_1} + v_{o_2})$. Experimenting with blending different animal species, the generated images portray photorealistic creatures with features of the two input species. This further highlights the power and versatility of the proposed framework (more generated images are in Appendix D.4).

---

## 5. Conclusion

We investigated the emergence of compositional structures within the image latent space of vision-language models and demonstrated that visual embeddings also exhibit a degree of compositionality similar to that of textual representations. We proposed a training-free framework, **Geodesically Decomposable Embeddings (GDE)**, designed to address the noisy and sparse nature of image data. GDE decomposes visual representations as a geometry-aware combination of optimal directions representing primitive concepts.

We demonstrated that these composed representations encode complex concepts and are effective in several tasks, including compositional classification and group robustness. Notably, GDE presents more robust compositional abilities than existing approaches based on linear decomposition of latent spaces, contributing to higher results in group robustness than existing task-specific methods. We believe this work contributes to achieving better interpretability and controllability of modern VLMs.

### Acknowledgements

This work was sponsored by the ERJU project, the EU Horizon project ELIAS (No. 101120237), Ministero delle Imprese e del Made in Italy (IPCEI Cloud DM 27 giugno 2022 – IPCEI-CL-0000007), and the FAIR – Future AI Research (PE00000013), funded by NextGeneration EU. The authors acknowledge the CINECA award under the ISCRA initiative for the availability of high-performance computing resources and support.

---

## Supplementary Material

This Supplementary Material provides additional details on Riemannian manifolds (Appendix A), proves the theoretical results the framework builds upon (Appendix B), describes extra implementation details (Appendix C), and presents further experimental results (Appendix D).

### Appendix A. Details on the Riemannian Manifold

We discuss some details of the tools used to deal with the geometry of a data manifold $\mathcal{M}$. In the following, we focus on the spherical case $\mathcal{M} = S^{d-1}$, which applies to the case with normalized embeddings.

#### A.1 Closed-form solutions

The exponential and logarithmic maps can be expressed in closed form on the unit sphere $S^{d-1}$. For any point of tangency $\mu \in S^{d-1}$, we have:

$$\mathrm{Exp}_\mu(v) = \cos(\lVert v \rVert)\,\mu + \sin(\lVert v \rVert)\,\frac{v}{\lVert v \rVert}, \qquad v \in T_\mu S^{d-1} \tag{13}$$

and

$$\mathrm{Log}_\mu(u) = \theta\,\frac{(I_d - \mu\mu^\top)(u - \mu)}{\lVert (I_d - \mu\mu^\top)(u - \mu) \rVert}, \qquad u \in S^{d-1} \tag{14}$$

where $\theta = \arccos(u^\top \mu)$ and $I_d \in \mathbb{R}^{d \times d}$ is the identity matrix.

#### A.2 Intrinsic mean

**Existence, uniqueness, and characterization.** The (weighted) intrinsic mean $\mu$ of a set of points $\{u_i\}_{i=1}^N$, defined as the solution of a minimization problem, is not necessarily unique. For example, on $S^2$ all the points on the equator minimize the average distance from the north and south poles. However, existence and uniqueness are guaranteed if the points live inside the same **geodesic ball** $\mathcal{B}_o(r) := \{u \in \mathcal{M} \mid d_\mathcal{M}(o, u) < r\}$ of radius $r > 0$ small enough. Under the same condition, $\mu$ is the unique point on $\mathcal{M}$ centering the logarithmic map of the input points, i.e., such that $\sum_{i=1}^N w_i \mathrm{Log}_\mu(u_i) = 0$. We refer to this property as the **characterization of the intrinsic mean**. For the unit sphere $S^{d-1}$, the closeness assumption is satisfied for any $r < \pi/2$. We can expect this condition to be verified by the normalized embeddings of a neural encoder because of the cone effect.

**Computation by gradient descent.** Computing the intrinsic mean $\mu$ of a weighted set of points requires minimizing the objective function:

$$f(u) = \frac{1}{2} \sum_{i=1}^N w_i \, d_\mathcal{M}(u, u_i)^2, \qquad u \in \mathcal{M} \tag{15}$$

This can be done with a gradient descent algorithm. It can be shown that Eq. (15) has gradient:

$$\nabla f(u) = -\sum_{i=1}^N w_i \, \mathrm{Log}_u(u_i), \qquad u \in \mathcal{M} \tag{16}$$

At each iteration, the new approximation $\mu_{j+1}$ is obtained by first moving in the opposite direction of the gradient and then mapping back onto the manifold with the exponential map centered in $\mu_j$. The cycle stops when the norm of the update is smaller than a fixed small value $\epsilon > 0$. Usually, the starting value $\mu_0 \in \mathcal{M}$ is chosen among the input points, which live on the manifold. Otherwise, in the special case $\mathcal{M} = S^{d-1}$, a good choice is the normalized (weighted) arithmetic mean $\mu_0 = \sum_{i=1}^N w_i u_i / \lVert \sum_{i=1}^N w_i u_i \rVert$. The learning rate $\eta$ must be carefully chosen to guarantee convergence; it has been shown that setting $\eta = 1$ is sufficient for spheres.

**Algorithm 1 — Intrinsic mean (gradient descent):**

```
Input:  u_1, ..., u_N ∈ M,  weights w_1, ..., w_N ∈ Δ_N,  initial μ_0 ∈ M
Output: the intrinsic mean μ ∈ M

repeat
    δ_μ   = η · Σ_{i=1}^N w_i · Log_{μ_j}(u_i)
    μ_{j+1} = Exp_{μ_j}(δ_μ)
until  ‖δ_μ‖ < ε
```

> *Note on the cut locus:* the logarithmic map is defined on $\mathcal{M} \setminus C_\mu$, where $C_\mu$ is the cut locus of $\mu$. This detail is usually not stressed because $C_\mu$ has measure zero on $\mathcal{M}$. On the unit sphere $S^{d-1}$, the cut locus of any point $\mu$ is its antipode $-\mu$.

### Appendix B. Proofs

We provide the proofs of the theoretical results stated in the methodology section. We omit the proof of Proposition 1 because it is the same as the more general Proposition 2 in the special case $|\mathcal{E}| = 1$. In the following, we assume that a given composite concept $z \in \mathcal{Z}$ is the tuple $z = (z_1, \dots, z_s)$.

#### Proof of Lemma 1

Let $\phi(\mathcal{Z}) = \{u_z\}$ be a geodesically decomposable set with tangent projections $v_z = \mathrm{Log}_\mu(u_z)$ decomposed as $v_z = v'_{z_1} + \cdots + v'_{z_s}$. Indicating $\bar v_{\mathcal{Z}_i} = \frac{1}{|\mathcal{Z}_i|} \sum_{z_i \in \mathcal{Z}_i} v'_{z_i}$, we show that the searched directions are $v_{z_i} = v'_{z_i} - \bar v_{\mathcal{Z}_i}$ ($i = 1, \dots, s$). The centering constraint $\sum_{z_i \in \mathcal{Z}_i} v_{z_i} = 0$ then follows immediately from the definition.

Observing that $\sum_i \bar v_{\mathcal{Z}_i} = \frac{1}{|\mathcal{Z}|} \sum_{z \in \mathcal{Z}} v_z = 0$ by the characterization of the intrinsic mean, this implies Eq. (3) is satisfied:

$$v_z = v'_{z_1} + \cdots + v'_{z_s} = (\bar v_{\mathcal{Z}_1} + \cdots + \bar v_{\mathcal{Z}_s}) + v_{z_1} + \cdots + v_{z_s} = v_{z_1} + \cdots + v_{z_s} \tag{17}$$

To show uniqueness, we demonstrate the $v_{z_i}$ are uniquely determined by the original vectors $v_z$:

$$v_{z_i} = v'_{z_i} - \bar v_{\mathcal{Z}_i} = \frac{1}{|\mathcal{Z}(z_i)|} \sum_{z \in \mathcal{Z}(z_i)} (v'_{z_1} + \cdots + v'_{z_s}) = \frac{1}{|\mathcal{Z}(z_i)|} \sum_{z \in \mathcal{Z}(z_i)} v_z \tag{18}$$

$\blacksquare$

#### Proof of Proposition 2

We start by observing that if $\{\tilde u_z\}$ is a geodesically decomposable set with intrinsic mean $\mu'$, then, following the proof of Lemma 1, we can write its tangent projection $\tilde v_z = \mathrm{Log}_\mu(\tilde u_z) \in T_\mu \mathcal{M}$ as $\tilde v_z = v_0 + v_{z_1} + \cdots + v_{z_s}$ where $\sum_{z_i \in \mathcal{Z}_i} v_{z_i} = 0$ and $v_0 = \frac{1}{|\mathcal{Z}|} \sum_z \tilde v_z$. Note that $\mu' = \mu$ if and only if $v_0 = 0$.

In the setting of the statement, we indicate $v_{(z,e)} = \mathrm{Log}_\mu(u_{(z,e)})$ and rephrase the objective in Eq. (9) as finding the vectors $v_0, v_{z_i} \in T_\mu \mathcal{M}$, $z_i \in \mathcal{Z}_i$ ($i = 1, \dots, s$) minimizing:

$$\frac{1}{2} \sum_{(z,e) \in \mathcal{Z} \times \mathcal{E}} p_{(z,e)} \lVert v_{(z,e)} - (v_0 + v_{z_1} + \cdots + v_{z_s}) \rVert^2 \tag{19}$$

Deriving Eq. (19) with respect to $v_0$ and observing that $\sum_{z_i \in \mathcal{Z}_i} v_{z_i} = 0$, the derivative is:

$$\sum_{z \in \mathcal{Z}} \sum_{e \in \mathcal{E}} p_{(z,e)} \big(v_{(z,e)} - (v_0 + v_{z_1} + \cdots + v_{z_s})\big) = \sum_{z \in \mathcal{Z}} (v_z - v_0) \tag{20}$$

where $v_z = \sum_{e \in \mathcal{E}} p_{(z,e)} v_{(z,e)}$. Setting this equal to zero gives $v_0 = \frac{1}{|\mathcal{Z}|} \sum_z v_z = \sum_{(z,e)} w_{(z,e)} v_{(z,e)} = 0$. The last equality follows from the characterization of the intrinsic mean and implies the intrinsic mean of the solution is $\mu$. The derivative with respect to a fixed $v_{z_i}$ is:

$$\sum_{z \in \mathcal{Z}(z_i)} \sum_{e \in \mathcal{E}} p_{(z,e)} \big(v_{(z,e)} - (v_{z_1} + \cdots + v_{z_s})\big) = \sum_{z \in \mathcal{Z}(z_i)} (v_z - v_{z_i}) \tag{21}$$

Setting this equal to zero gives $v_{z_i} = \frac{1}{|\mathcal{Z}(z_i)|} \sum_{z \in \mathcal{Z}(z_i)} v_z$. $\blacksquare$

#### Proof of Lemma 2

As observed in the proof of Proposition 2, we have $\frac{1}{|\mathcal{Z}|} \sum_z v_z = 0$, implying the weighted mean $\mu$ is the intrinsic mean of $\{u_z := \mathrm{Exp}_\mu(v_z)\}_{z \in \mathcal{Z}}$. $\blacksquare$

### Appendix C. Experimental Details

#### C.1 Closeness assumption

We numerically verify the closeness assumption (Appendix A.2), which guarantees the existence and uniqueness of the intrinsic mean. Given a set of points on $S^{d-1}$, a good guess for the center $o \in \mathcal{M}$ of a small geodesic ball $\mathcal{B}_o(r)$ containing them is their normalized arithmetic mean $\mu_0$. For all the sets of embeddings used in our experiments, we verify that their maximum intrinsic distance (i.e., angle) from $\mu$ is smaller than $\pi/2 \approx 1.57$.

Using the embeddings from the default model (CLIP ViT-L/14), the closeness assumption is satisfied for **all** datasets — UT-Zappos, MIT-states, Waterbirds, and CelebA — for both image and text embeddings: in every case the maximum distance to the normalized arithmetic mean stays below the $r < \pi/2$ threshold.

#### C.2 Noise distribution

**Temperature selection.** When performing compositional classification and group robustness, we use the image-to-text distribution $P((z,e) \mid y(z))$ defined by the VLM as the noise distribution. For CLIP, this is given by the softmax activations described in the main paper and depends on the temperature parameter $t \in (0, +\infty)$. For each dataset, we select $t$ by performing a grid search on the validation set, optimizing the AUC metric for compositional classification and the WG accuracy for group robustness.

**SigLIP sigmoid probabilities.** Differently from the original CLIP, SigLIP uses a sigmoid-based loss processing every image-text pair independently, and it defines the pair-specific probabilities:

$$P((z,e) \mid y(z)) = \frac{1}{1 + \exp\!\left(-u_{(z,e)}^\top u_{y(z)} / t - b\right)} \tag{22}$$

When considering SigLIP embeddings, we use the noise distribution $p_{(z,e)} \propto P((z,e) \mid y(z))$, proportional to the pair-specific sigmoid probabilities. We select the temperature parameter $t$ as described for the CLIP model while keeping the logit bias $b$ equal to the learned value ($b \approx -16.5$).

#### C.3 Text prompts

For UT-Zappos and MIT-states, we use the same text prompts as Trager et al. An attribute-object pair $(a,o)$ is described by $y(a,o) = $ *"an image of a {a} {o}"*, where {a} and {o} are the lower-case original category names. For UT-Zappos, every dot character is substituted with a space (e.g., "Synthetic Boots.Ankle" → "synthetic boots ankle"). We use these prompts both when decomposing text embeddings and when computing the image-to-text probabilities defining the noise distribution.

For Waterbirds and CelebA, we use the text prompts from prior work. These are obtained by representing each spurious attribute and each target class with the captions listed below. Prepending the spurious prompts to the class prompts produces $k = 4$ and $k = 3$ textual descriptions for each composite group in Waterbirds and CelebA, respectively. We compute the image-to-text probabilities for the noise distribution using the decomposable text embeddings $\tilde u_{y(z)}$, $y(z) \in \mathcal{Z}$, given by Proposition 2 applied to the input embeddings. Note that they can be written as $\{u_{(z,e)} \mid (z,e) \in \mathcal{Z} \times \mathcal{E}\}$, where $\mathcal{E}$ is a "prompt template" dimension.

**Waterbirds prompts:**

- *Class prompts:* "This is a picture of a landbird." / "This is a picture of a waterbird."
- *Spurious attribute prompts:*
  - "This is a land background." / "This is a water background."
  - "This is a picture of a forest." / "This is a picture of a beach."
  - "This is a picture of a mountain." / "This is a picture of an ocean."
  - "This is a picture of a wood." / "This is a picture of a port."

**CelebA prompts:**

- *Class prompts:* "A photo of a celebrity with dark hair." / "A photo of a celebrity with blond hair."
- *Spurious attribute prompts:*
  - "A photo of a male." / "A photo of a female."
  - "A photo of a male celebrity." / "A photo of a female celebrity."
  - "A photo of a man." / "A photo of a woman."

### Appendix D. Additional Results

#### D.1 Ablation: noise distribution

Our decomposition method (GDE) computes the noise distribution using CLIP scores with a custom temperature parameter. We compare GDE against the decomposition obtained when using a uniform noise distribution (denoted GDE$_u$) in the task of compositional classification. While the simpler GDE$_u$ already performs well compared to the zero-shot baseline (e.g., on UT-Zappos closed-world, GDE$_u$ reaches AUC 13.6 / $\rho \approx 311\%$ versus GDE's 13.9 / $\rho \approx 318\%$; on MIT-states, 8.2 / $\rho \approx 74\%$ versus 8.6 / $\rho \approx 78\%$), leveraging the non-uniform noise distribution from the CLIP scores **always improves** performance.

#### D.2 Decomposing hyperbolic representations

We investigate the compositional properties of visual representations on geometries different from CLIP's hypersphere. Specifically, we perform compositional classification of the pre-trained **MERU ViT-L-16** embeddings, which are points on the **Lorentz model**:

$$\mathcal{L}^d = \{u \in \mathbb{R}^{d+1} \mid \langle u, u \rangle_{\mathcal{L}} = -1/c\} \tag{23}$$

Here $\langle \cdot, \cdot \rangle_{\mathcal{L}}$ is the Lorentzian inner product and the parameter $c > 0$ is learned during pre-training. The exponential and logarithmic maps have a closed-form solution for the hyperboloid $\mathcal{L}^d$, enabling a simple application of the GDE framework in this setting.

Results show that, as observed for CLIP spherical embeddings, the GDEs of MERU's hyperbolic representations contain semantically meaningful information about the concepts they represent. Moreover, the significantly lower performance of LDE highlights the importance of GDE's geometry awareness also in this non-spherical setup.

#### D.3 Runtime

A potential limitation of our framework is the additional computational cost it requires for mapping embeddings to and from the tangent space. Suppose we compute a decomposable set for $M = |\mathcal{Z}|$ composite concepts using $N = |\mathcal{T}|$ visual embeddings on the sphere $S^{d-1} \subset \mathbb{R}^d$. Compared to LDE, GDE additionally computes $\mathrm{Log}_\mu$ for the $N$ inputs and $\mathrm{Exp}_\mu$ for the $M$ tangent compositions. The computational complexity of these operations is $O(Nd)$ and $O(Md)$, respectively.

Note that the orthogonal projection in Eq. (14) can be rewritten as $(I_d - \mu\mu^\top)w = w - (\mu^\top w)\mu$, avoiding the explicit computation of the $d \times d$ matrix. Calculating $\mu$ with Algorithm 1 is $O(Nd)$ per gradient step, keeping the extra compute linear in $N$, $M$, $d$.

Both methods are fast on the relatively small datasets used for our analysis (runtimes measured on a Titan Xp GPU, averaged over 5 runs, with tolerance $\epsilon = 10^{-5}$ for $\mu$). GDE is significantly slower than LDE, with most of its extra runtime being spent on the computation of $\mu$. However, we argue that approximating $\mu$ with a smaller subset of $N' < N$ input embeddings could be sufficient and drastically improve efficiency when the number of inputs is large.

#### D.4 Generated images

We show extra images generated using Stable Diffusion with the unCLIP module to invert composite embeddings, including attribute-object pairs from all datasets used in our experiments. Similarly to the animal-animal pairs shown in the main document, we identify other high-level categories within the MIT-states objects (items, environments, and materials) and visualize animal-environment and item-material compositions.

We also observe that the modularity of the compositions allows finer control over the composite embeddings. By inverting embeddings of the form $\mathrm{Exp}_\mu(\alpha v_a + v_o)$, where the attribute direction is scaled by a scalar $\alpha \in \mathbb{R}$, changing the value of $\alpha$ modifies the **intensity** of the attribute, resulting in a weaker or stronger appearance of it in the generated images. This further demonstrates that the primitive vectors resulting from solving the proposed optimization problem are interpretable directions of the latent space.

Our initial goal for the generative experiments was to qualitatively inspect the GDE compositions. However, the good quality of the results suggests that our framework could be useful for **augmenting compositional datasets**. To support this, we compute the average CLIP-score of 500 outputs (five generations of 100 random unseen concepts) to assess how a CLIP model perceives composite concepts in generated images. As a baseline, we consider the default text-to-image (T2I) version of the generative model. GDE-inverted images obtain a higher average CLIP-score than the T2I baseline on both datasets (UT-Zappos: $0.68 \pm 0.06$ vs. $0.62 \pm 0.10$; MIT-states: $0.58 \pm 0.08$ vs. $0.55 \pm 0.10$).

#### D.5 Failure cases

We investigate failure cases in Stable Diffusion visualizations and note that the decomposable embeddings may encode spurious correlations of the input data or produce ambiguous compositions. For instance, generated images suggest that the *inflated* and *boat* primitive directions are respectively linked to *round object* and *water*, and the *tiger* + *horse* and *dog* + *forest* compositions are respectively close to *zebra* and *bear*.

---

## References

1. Bijan Afsari. *Riemannian $L^p$ center of mass: existence, uniqueness, and convexity.* Proceedings of the American Mathematical Society, 139(2):655–673, 2011.
2. Bijan Afsari, Roberto Tron, and René Vidal. *On the convergence of gradient descent for finding the Riemannian center of mass.* SIAM Journal on Control and Optimization, 51(3):2230–2260, 2013.
3. Jacob Andreas. *Measuring compositionality in representation learning.* ICLR, 2019.
4. Yuval Atzmon, Felix Kreuk, Uri Shalit, and Gal Chechik. *A causal view of compositional zero-shot recognition.* NeurIPS, 33:1462–1473, 2020.
5. Alberto Baldrati, Lorenzo Agnolucci, Marco Bertini, and Alberto Del Bimbo. *Zero-shot composed image retrieval with textual inversion.* ICCV, 2023.
6. Moritz Böhle, Mario Fritz, and Bernt Schiele. *B-cos networks: Alignment is all we need for interpretability.* CVPR, 2022.
7. Duygu Ceylan, Chun-Hao P. Huang, and Niloy J. Mitra. *Pix2video: Video editing using image diffusion.* ICCV, 2023.
8. Sarah Chabal and Viorica Marian. *Speakers of different languages process the visual world differently.* Journal of Experimental Psychology: General, 144(3):539–550, 2015.
9. Wei-Lun Chao, Soravit Changpinyo, Boqing Gong, and Fei Sha. *An empirical study and analysis of generalized zero-shot learning for object recognition in the wild.* ECCV, pages 52–68, Springer, 2016.
10. Ching-Yao Chuang, Varun Jampani, Yuanzhen Li, Antonio Torralba, and Stefanie Jegelka. *Debiasing vision-language models via biased prompts.* arXiv:2302.00070, 2023.
11. Konrad Czechowski, Tomasz Odrzygóźdź, Marek Zbysiński, Michał Zawalski, Krzysztof Olejnik, Yuhuai Wu, Łukasz Kuciński, and Piotr Miłoś. *Subgoal search for complex reasoning tasks.* NeurIPS, 2021.
12. Sepehr Dehdashtian, Lan Wang, and Vishnu Naresh Boddeti. *FairerCLIP: Debiasing zero-shot predictions of CLIP in RKHSs.* ICLR, 2024.
13. Karan Desai, Maximilian Nickel, Tanmay Rajpurohit, Justin Johnson, and Shanmukha Ramakrishna Vedantam. *Hyperbolic image-text representations.* ICML, pages 7694–7731, PMLR, 2023.
14. Jacob Feldman. *Probabilistic origins of compositional mental representations.* Psychological Review, 131(3):599–624, 2024.
15. Pedro Felzenszwalb, David McAllester, and Deva Ramanan. *A discriminatively trained, multiscale, deformable part model.* CVPR, 2008.
16. Martin A. Fischler and Robert A. Elschlager. *The representation and matching of pictorial structures.* IEEE Transactions on Computers, 100(1):67–92, 1973.
17. P. Thomas Fletcher, Conglin Lu, Stephen M. Pizer, and Sarang Joshi. *Principal geodesic analysis for the study of nonlinear statistics of shape.* IEEE TMI, 23(8):995–1005, 2004.
18. Peng Gao, Shijie Geng, Renrui Zhang, Teli Ma, Rongyao Fang, Yongfeng Zhang, Hongsheng Li, and Yu Qiao. *CLIP-Adapter: Better vision-language models with feature adapters.* IJCV, 132(2):581–595, 2024.
19. Songwei Ge, Shlok Mishra, Simon Kornblith, Chun-Liang Li, and David Jacobs. *Hyperbolic contrastive learning for visual representations beyond objects.* CVPR, pages 6840–6849, 2023.
20. Yunye Gong, Srikrishna Karanam, Ziyan Wu, Kuan-Chuan Peng, Jan Ernst, and Peter C. Doerschuk. *Learning compositional visual concepts with mutual consistency.* CVPR, 2018.
21. Alon Hafri, E. J. Green, and Chaz Firestone. *Compositionality in visual perception.* Behavioral and Brain Sciences, 46:e277, 2023.
22. Irina Higgins, Nicolas Sonnerat, Loic Matthey, Arka Pal, Christopher P. Burgess, Matko Bošnjak, Murray Shanahan, Matthew Botvinick, Demis Hassabis, and Alexander Lerchner. *SCAN: Learning hierarchical compositional visual concepts.* ICLR, 2018.
23. Zhi Hou, Xiaojiang Peng, Yu Qiao, and Dacheng Tao. *Visual compositional learning for human-object interaction detection.* ECCV, 2020.
24. Zhi Hou, Baosheng Yu, and Dacheng Tao. *Discovering human-object interaction concepts via self-compositional learning.* ECCV, 2022.
25. Cheng-Yu Hsieh, Jieyu Zhang, Zixian Ma, Aniruddha Kembhavi, and Ranjay Krishna. *SugarCrepe: Fixing hackable benchmarks for vision-language compositionality.* NeurIPS, 2023.
26. Drew A. Hudson and Christopher D. Manning. *Compositional attention networks for machine reasoning.* ICLR, 2018.
27. Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. *OpenCLIP*, 2021.
28. Phillip Isola, Joseph J. Lim, and Edward H. Adelson. *Discovering states and transformations in image collections.* CVPR, 2015.
29. Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. *Scaling up visual and vision-language representation learning with noisy text supervision.* ICML, pages 4904–4916, 2021.
30. Shyamgopal Karthik, Karsten Roth, Massimiliano Mancini, and Zeynep Akata. *Vision-by-language for training-free compositional image retrieval.* ICLR, 2024.
31. Keizo Kato, Yin Li, and Abhinav Gupta. *Compositional learning for human object interaction.* ECCV, 2018.
32. Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. *Imagic: Text-based real image editing with diffusion models.* CVPR, 2023.
33. Polina Kirichenko, Pavel Izmailov, and Andrew Gordon Wilson. *Last layer re-training is sufficient for robustness to spurious correlations.* ICLR, 2023.
34. Jayanth Koushik, Hiroaki Hayashi, and Devendra Singh Sachan. *Compositional reasoning for visual question answering.* ICML, 2017.
35. Ananya Kumar, Aditi Raghunathan, Robbie Matthew Jones, Tengyu Ma, and Percy Liang. *Fine-tuning can distort pretrained features and underperform out-of-distribution.* ICLR, 2022.
36. Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham Kakade, Prateek Jain, et al. *Matryoshka representation learning.* NeurIPS, 35:30233–30249, 2022.
37. Kevin J. Lande. *Compositionality in perception: A framework.* WIREs Cognitive Science, 15(6):e1691, 2024.
38. Martha Lewis, Nihal Nayak, Peilin Yu, Jack Merullo, Qinan Yu, Stephen Bach, and Ellie Pavlick. *Does CLIP bind concepts? Probing compositionality in large image models.* Findings of the ACL: EACL 2024, pages 1487–1500, 2024.
39. Xilai Li, Xi Song, and Tianfu Wu. *AOGNets: Compositional grammatical architectures for deep learning.* CVPR, 2019.
40. Yong-Lu Li, Yue Xu, Xiaohan Mao, and Cewu Lu. *Symmetry and group in attribute-object compositions.* CVPR, 2020.
41. Victor Weixin Liang, Yuhui Zhang, Yongchan Kwon, Serena Yeung, and James Y. Zou. *Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning.* NeurIPS, 35:17612–17625, 2022.
42. Renjie Liao, Alex Schwing, Richard Zemel, and Raquel Urtasun. *Learning deep parsimonious representations.* NeurIPS, 2016.
43. Giorgio Longari, Lorenzo Olearo, Simone Melzi, Rafael Peñaloza, and Alessandro Raganato. *How to blend concepts in diffusion models.* arXiv:2407.14280, 2024.
44. Xiaocheng Lu, Song Guo, Ziming Liu, and Jingcai Guo. *Decomposed soft prompt guided fusion enhancing for compositional zero-shot learning.* CVPR, 2023.
45. Massimiliano Mancini, Muhammad Ferjad Naeem, Yongqin Xian, and Zeynep Akata. *Open world compositional zero-shot learning.* CVPR, pages 5222–5230, 2021.
46. Massimiliano Mancini, Muhammad Ferjad Naeem, Yongqin Xian, and Zeynep Akata. *Learning graph embeddings for open world compositional zero-shot learning.* IEEE TPAMI, 46(3):1545–1560, 2022.
47. Ishan Misra, Abhinav Gupta, and Martial Hebert. *From red wine to red tomato: Composition with context.* CVPR, 2017.
48. Tushar Nagarajan and Kristen Grauman. *Attributes as operators: Factorizing unseen attribute-object compositions.* ECCV, 2018.
49. Nihal V. Nayak, Peilin Yu, and Stephen Bach. *Learning to compose soft prompts for compositional zero-shot learning.* ICLR, 2023.
50. James Oldfield, Christos Tzelepis, Yannis Panagakis, Mihalis Nicolaou, and Ioannis Patras. *Parts of speech–grounded subspaces in vision-language models.* NeurIPS, 36:2700–2724, 2023.
51. Bjorn Ommer and Joachim Buhmann. *Learning the compositional nature of visual object categories for recognition.* IEEE TPAMI, 32(3):501–516, 2009.
52. Bjorn Ommer and Joachim M. Buhmann. *Learning the compositional nature of visual objects.* CVPR, 2007.
53. Dim P. Papadopoulos, Youssef Tamaazousti, Ferda Ofli, Ingmar Weber, and Antonio Torralba. *How to make a pizza: Learning a compositional layer-based GAN model.* CVPR, 2019.
54. Barbara Partee et al. *Lexical semantics and compositionality.* An Invitation to Cognitive Science: Language, 1:311–360, 1995.
55. Barbara H. Partee. *Compositionality in formal semantics: Selected papers.* John Wiley & Sons, 2008.
56. Xavier Pennec. *Probabilities and statistics on Riemannian manifolds: Basic tools for geometric measurements.* NSIP, pages 194–198, 1999.
57. Pramuditha Perera, Matthew Trager, Luca Zancato, Alessandro Achille, and Stefano Soatto. *Prompt algebra for task composition.* arXiv:2306.00310, 2023.
58. Senthil Purushwalkam, Maximilian Nickel, Abhinav Gupta, and Marc'Aurelio Ranzato. *Task-driven modular networks for zero-shot compositional learning.* ICCV, 2019.
59. Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. *Learning transferable visual models from natural language supervision.* ICML, pages 8748–8763, PMLR, 2021.
60. Sameera Ramasinghe, Violetta Shevchenko, Gil Avraham, and Ajanthan Thalaiyasingam. *Accept the modality gap: An exploration in the hyperbolic space.* CVPR, pages 27263–27272, 2024.
61. Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. *Hierarchical text-conditional image generation with CLIP latents.* arXiv:2204.06125, 1(2):3, 2022.
62. Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. *High-resolution image synthesis with latent diffusion models.* CVPR, pages 10684–10695, 2022.
63. Shiori Sagawa, Pang Wei Koh, Tatsunori B. Hashimoto, and Percy Liang. *Distributionally robust neural networks.* ICLR, 2020.
64. Kuniaki Saito, Kihyuk Sohn, Xiang Zhang, Chun-Liang Li, Chen-Yu Lee, Kate Saenko, and Tomas Pfister. *Pic2Word: Mapping pictures to words for zero-shot composed image retrieval.* CVPR, 2023.
65. Sascha Saralajew, Lars Holdijk, Maike Rees, Ebubekir Asan, and Thomas Villmann. *Classification-by-components: Probabilistic modeling of reasoning over a set of components.* NeurIPS, 2019.
66. Jürgen Schmidhuber. *Towards compositional learning in dynamic networks.* Technical University of Munich (Technical Report FKI-129-90), 1990.
67. Zhangzhang Si and Song-Chun Zhu. *Learning AND-OR templates for object recognition and detection.* IEEE TPAMI, 35(9):2189–2205, 2013.
68. Austin Stone, Huayan Wang, Michael Stark, Yi Liu, D. Scott Phoenix, and Dileep George. *Teaching compositionality to CNNs.* CVPR, 2017.
69. Fuwen Tan, Song Feng, and Vicente Ordonez. *Text2Scene: Generating compositional scenes from textual descriptions.* CVPR, 2019.
70. Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet Singh, Adina Williams, Douwe Kiela, and Candace Ross. *Winoground: Probing vision and language models for visio-linguistic compositionality.* CVPR, 2022.
71. Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. *Eyes wide shut? Exploring the visual shortcomings of multimodal LLMs.* CVPR, 2024.
72. Matthew Trager, Pramuditha Perera, Luca Zancato, Alessandro Achille, Parminder Bhatia, and Stefano Soatto. *Linear spaces of meanings: Compositional structures in vision-language models.* ICCV, pages 15395–15404, 2023.
73. Jianyu Wang and Alan L. Yuille. *Semantic part segmentation using compositional model combining shape and appearance.* CVPR, 2015.
74. Aron Yu and Kristen Grauman. *Fine-grained visual comparisons with local learning.* CVPR, pages 192–199, 2014.
75. Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, and James Zou. *When and why vision-language models behave like bags-of-words, and what to do about it?* ICLR, 2023.
76. Tian Yun, Usha Bhalla, Ellie Pavlick, and Chen Sun. *Do vision-language pretrained models learn composable primitive concepts?* TMLR, 2023.
77. Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. *Sigmoid loss for language image pre-training.* ICCV, 2023.
78. Michael Zhang and Christopher Ré. *Contrastive adapters for foundation model group robustness.* NeurIPS, 35:21682–21697, 2022.
79. Yizhen Zhang, Minkyu Choi, Kuan Han, and Zhongming Liu. *Explainable semantic space by grounding language to vision with cross-modal contrastive learning.* NeurIPS, 34:18513–18526, 2021.
80. Zhixing Zhang, Ligong Han, Arnab Ghosh, Dimitris N. Metaxas, and Jian Ren. *SINE: Single image editing with text-to-image diffusion models.* CVPR, 2023.
81. Bo Zhao, Bo Chang, Zequn Jie, and Leonid Sigal. *Modular generative adversarial networks.* ECCV, 2018.
