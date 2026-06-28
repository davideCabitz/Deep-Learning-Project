# **Linear Spaces of Meanings: Compositional Structures in Vision-Language Models** 

Matthew Trager 

Pramuditha Perera Luca Zancato Alessandro Achille 

Parminder Bhatia Stefano Soatto 

## AWS AI Labs 

_{_ mttrager,pramudi,aachille,parmib,soattos _}_ @amazon.com zancato@amazon.it 

## **Abstract** 

_We investigate compositional structures in data embeddings from pre-trained vision-language models (VLMs). Traditionally, compositionality has been associated with algebraic operations on embeddings of words from a preexisting vocabulary. In contrast, we seek to approximate representations from an encoder as combinations of a smaller set of vectors in the embedding space. These vectors can be seen as “ideal words” for generating concepts directly within embedding space of the model. We first present a framework for understanding compositional structures from a geometric perspective. We then explain what these compositional structures entail probabilistically in the case of VLM embeddings, providing intuitions for why they arise in practice. Finally, we empirically explore these structures in CLIP’s embeddings and we evaluate their usefulness for solving different vision-language tasks such as classification, debiasing, and retrieval. Our results show that simple linear algebraic operations on embedding vectors can be used as compositional and interpretable methods for regulating the behavior of VLMs._ 

## **1. Introduction** 

In natural language, few primitive concepts or words can be used compositionally to generate a large number of complex meanings. For example, many composite concepts can be obtained by combining attributes and nouns. The hidden representations provided by a neural model, on the other hand, a priori _do not_ have a similar compositional structure. In contextual text embeddings, in particular, the representation of a string of text is jointly affected by all of its tokens simultaneously, which means that there may not be a simple 

**==> picture [226 x 138] intentionally omitted <==**

**----- Start of picture text -----**<br>
Embedding Space<br>a green car<br>a green bike<br>a cold a cold<br>a warm rainy daya warm rainy night<br>rainy day rainy night<br>a red car a blue car<br>a red bike a blue bike a cold a cold<br>a warm sunny daya warm sunny night<br>sunny day sunny night<br>u (“a [col] [obj]”)<br>≈ u 0 +  u col +  u obj<br>**----- End of picture text -----**<br>


Figure 1: Compositional structures in contextual embeddings. We show that the embeddings of composite concepts are often approximately decomposable as a sum of vectors corresponding to each factor. These vectors are not embeddings of actual words, but they can be viewed as “ideal words” and used for interpretable manipulations of the representations. 

relationship between the representations of the entire text and the words that appear in it. 

In this paper, we investigate the existence of latent compositional structures in the embedding space. That is, we aim to decompose composite concepts as linear combinations of embedding vectors associated with different factors, as illustrated in Figure 1. If such vectors exist, they can be treated as _ideal words_ for composing new concepts directly within the representation space of the model. The first application that we envision is for vision-language models ( _e.g_ ., CLIP [41]) where embeddings of text labels are often used for image classification or retrieval. In this setting, linear compositionality would imply that we could classify an image with _n_ 1 _. . . nk_ composite labels—where _ni_ indicates the number of options for each factor—by com- 

paring each image with only _n_ 1 + _. . ._ + _nk_ ideal words, since by linearity the inner product of an image with a composed label is the sum of the product with the corresponding ideal words. Moreover, linear decompositions can be used for “post-hoc” manipulations of pre-trained data representations ( _e.g_ ., amplifying or reducing the importance of certain factors), which can be helpful to control the behavior of neural models. 

In general, the meaning of words in language is always _contextual_ , in the sense that their interpretation depends on any text that surrounds them. However, language would be completely impractical if words did not also have some stability in their meaning. The main benefit of the usage of words is, in fact, that meaning can be mostly inferred compositionally by combining meanings of words or phrases. There is, therefore, _a natural tension between compositionality and contextuality_ : the former requires some amount of independence from context, while the latter allows for general dependencies. In a sense, our goal in this work is to consider representations of meanings that were originally learned as contextual, and to later approximate them as needed with compositional ones based on ideal words. This combines the flexibility and expressiveness of contextuality with the structural efficiency of compositionality. Our main contributions can be summarized as follows: 

- We describe compositional linear structures from a geometric perspective and explain how these structures can be approximately recovered from arbitrary collections of vectors associated with a product of “factors.” We also relate these structures with previous definitions of disentangled representations that were based on mathematical representation theory [26] (Section 3). 

- We consider embeddings arising from visual-language models (VLMs) and show that the existence of decomposable embeddings is equivalent to the conditional independence of the factors for the probability defined by the model. We also discuss some relaxations of this result that illustrate how linear structures may emerge even when if true data distribution satisfies weaker “disentanglement” conditions (Section 4). 

- We empirically show that embeddings of composite concepts can often be well-approximated as linear compositional structures, and that this leads to simple but effective strategies for solving classification and retrieval problems in a compositional setting. We also visualize manipulations of decomposable embeddings using a CLIP-guided diffusion model (Stable Diffusion [42]). 

## **2. Related Work** 

Compositionality has long been recognized to be a fundamental principle in cognition [20]. It has been a central in theme in Gestalt psychology [16], cognitive sciences [19], and pattern theory [24]. The main benefit of compositional representations is that they avoid the combinatorial explosion that occurs if all composed concepts are considered to be completely distinct. This property is of course a characteristic feature of natural languages, which use a fixed vocabulary for all representions, making “infinite use of finite means” (von Humboldt) [10]. However, while there is large body of work in NLP devoted to learning compositional representations of language ( _e.g_ .,[37, 12, 5, 22, 13]), modern text representations based on transformer architectures [47] are a priori _not_ compositional in any way. Some works have studied whether compositionality is implicitly present in neural networks, for example by evaluating the ability of these models to generalize beyond the training data [27]. More relevant to our purposes, [3] proposed a framework for evaluating the compositionality of a network’s internal representations, by searching for representational primitives; however, finding such compositional primitives requires solving an optimization problem. In a broad sense, compositionality can be seen as a particular way of exploiting or imposing _structure_ in the inner representations of a network. It has also been argued that data representations should be concentrated in low-dimensional linear spaces [34, 9], or even be “disentangled” with respect to factors of variation in the data [26, 8, 1]. Our perspective on compositional representations is closely related to the definition of disentanglement given in [26]. As argued above, compositionality of text representations is naturally in tension with _contextuality_ . Since their introduction in NLP around 2018 [40, 15], contextual text embeddings have been extremely successful, and are part of modern transformer-based architectures. The amount of contextuality in these word embeddings has been quantified using different metrics in [17]. 

Linear compositionality for embeddings is often associated with popular “vector analogies” that are known to roughly hold for (non-contextual) word embeddings such as word2vec [36] and GloVe [39]. Several works have proposed theoretical justifications for this property [29, 4, 25, 2, 18, 45]. To our knowledge, however, similar properties for contextual embeddings of language models have not been considered, although [46] has evaluated the performance of transformer-based models on analogy tasks. Various limitations of linear analogies have also been pointed out [31, 7]. 

In the context of image generation, compositional approaches for controlling the output of diffusion models have been recently proposed in [32, 48]. In particular, [48] introduced a “concept agebra” that is formally similar to our decomposable representations; however, their notion 

of “concept” is based on score representations (gradient of log-probabilities), rather than on embedding vectors, which leads to a different probabilistic characterization of compositionality. Finally, [11] introduced a method for removing biases and spurious correlations from pre-trained VLM embeddings for both discriminative and generative tasks; since their proposed approach consists in applying certain linear projections to textual embeddings (with some calibration adjustments), it can be seen as conceptually similar to an application of our decompositions. 

## **3. Decomposable Embeddings** 

We begin by discussing from a purely geometric perspective what we mean by “linear compositionality.” We consider a finite set _Z_ = _Z_ 1 _× . . . × Zk_ that we view as representing a factored set of “concepts.” For example, the set _Z_ may be a collection of strings of text organized in a structured way, _e.g_ ., according to attribute-object-context. We often write elements of _Z_ as _z_ = ( _z_ 1 _, . . . , zk_ ) with _zi ∈Zi_ and refer to _zi_ as the components of _z_ . We now consider an arbitrary embedding map _r_ : _Z → V_ of _Z_ into a vector space _V_ . 

**Definition 1** (Decomposable embeddings) **.** A collection of vectors _r_ ( _Z_ ) = _{_ _**u** z_ : _z ∈Z} ⊂ V_ parameterized by _Z_ = _Z_ 1 _×. . .×Zk_ is _decomposable_ if there exist vectors _**u** zi ∈ V_ for all _zi ∈Zi_ ( _i_ = 1 _, . . . , k_ ) such that 

**==> picture [166 x 11] intentionally omitted <==**

for all _z_ = ( _z_ 1 _, . . . , zk_ ). 

This notion is very intuitive and can be seen as a generalization of the additive compositionality that has been considered for (pairwise) analogies and word embeddings [36]. 

**Lemma 2.** _1) A collection of vectors r_ ( _Z_ ) _is decomposable if and only if the vector difference_ _**u** z −_ _**u** z[′] does not depend on the components that z, z[′] ∈Z share in common. 2) If |Zi|_ = _ni, then the dimension of Span_ ( _r_ ( _Z_ )) _is at most_ 1 +[�] _[k] i_ =1[(] _[n][i][ −]_[1)] _[.]_ 

It is easy to realize that if a collection of vectors _r_ ( _Z_ ) is decomposable, then the vectors appearing on the right of equation 1 are _never_ uniquely determined. In particular, even though each _**u** zi_ is associated with a value of a factor _zi ∈Zi_ , that vector cannot carry any “semantic” content. However, we can recover uniqueness in the components by simply turning to a “centered” decomposition. 

**Lemma 3** (Centered decomposition) **.** _If a collection of vectors r_ ( _Z_ ) _is decomposable, then there exist unique vectors_ _**u**_ 0 _∈ V and_ _**u** zi ∈ V for all zi ∈Zi (i_ = 1 _, . . . , k) such that_[�] _zi∈Zi_ _**[u]**[z] i_[= 0] _[ for all][ i][ and]_ 

**==> picture [178 x 10] intentionally omitted <==**

## _for all z_ = ( _z_ 1 _, . . . , zk_ ) _._ 

In the previous decomposition, the vectors _**u** zi_ are now uniquely associated with the value of a factor _zi ∈Zi_ , but are _relative_ to the other values in _Zi_ (since they sum to zero). Similarly, the vector spaces _VZi_ := _Span_ ( _**u** zi_ : _zi ∈ Zi_ ) are uniquely associated with each factor _Zi_ . In our applications, we will refer to _**u** i_ as the _ideal words_ of the linear factorization and to each _VZi_ as the _semantic space_ associated with _Zi_ . Despite its simplicity, we believe that the decomposition in Lemma 3 paints an interesting intuitive picture of linear models of “meaning.” In this setting, the origin is not a universally meaningful point; for example, the origin of text embeddings does not correspond to the null string. Thus, meanings might be best viewed as an _affine space_ , where the origin is only chosen as a particular reference that may depend on context. Ideal words, on the other hand, provide _relative meanings_ with respect to the context. 

From Lemma 2, it follows that decomposable representations must be very low-dimensional and, in particular, “generic” embeddings will _not_ be decomposable. However, it is very easy to recover the nearest decomposable approximation for any given set of vectors _**u** z, z ∈Z_ . 

**Proposition 4.** _Let αzi zi ∈Zi be arbitrary positive weights such that_[�] _zi∈Zi[α][z] i_[=][1] _[,][and][define][β][z]_[:=] � _i[α][z] i[for][all][z]_[=][(] _[z]_[1] _[, . . . , z][k]_[)] _[.][Then,][for][any][norm][∥· ∥] induced by an inner product on V , we have that_ 

**==> picture [183 x 39] intentionally omitted <==**

˜ _is given by_ _**u** z_ = _**u**_ 0 + _**u** z_ 1 + _. . ._ + _**u** zk where_ 

**==> picture [229 x 48] intentionally omitted <==**

This fact shows that computing decomposable approximations amounts to performing simple weighted averages of the original vectors. In many cases, we will consider _αzi_ = _n_ 1 _i_[and] _[ β][z]_[=][ �] _n_[1] _i_[, however it can be useful to allow] for additional “knobs,” as the following example illustrates. 

**Example 5.** One of our main motivations to consider decomposable structures is to approximate (pre-trained) contextual text embeddings to obtain representations that are _interpretable_ and _compositional_ . More concretely, assume that each factor _Zi_ represents a finite collection of strings and that the representation _r_ : _Z_ 1 _× . . . × Zk → V_ is defined by concatenating strings and then embedding the result using a contextual language encoder. For a very simple 

example, consider 

## _Z_ = _{_ a blue, a red, a green _} × {_ bike, house _},_ 

which leads to six possible strings and six distinct embedding vectors. Using Proposition 4, we can easily find a decomposable approximation _**u**_ ( _col,obj_ ) _≈_ _**u**_ 0 + _**u** col_ + _**u** obj_ , where _**u** col_ and _**u** obj_ are the ideal words representing a particular object and color from _Z_ . As we will see, these vectors can be used for semantic manipulations of embeddings. Note that ideal words are not the same as the encodings of the original words or substrings. In fact, quite intuitively, the meaning of ideal word vectors is determined entirely by the way in which the corresponding string interacts with other factors. For example, we have _**u**_ green = _αcar_ _**u**_ (green car) + _αhouse_ _**u**_ (green house) _−_ _**u**_ 0 where _**u**_ 0 is the mean of all six embeddings. In this particular example, “green house” has distinct contextual meaning, but this can be controlled by using appropriate weights, if desired. See Section 5 and Figure 3 for more discussions on similar examples. 

We conclude this section by pointing out a connection between decomposable embeddings and a notion of “disentangled representations” proposed in [26]. We refer to the Appendix for a short summary of the relevant mathematical background and for additional discussions. In a broad sense, we can say that an embedding map _r_ : _Z → V_ into a vector space _V_ is “linearly compositional” with respect to some group of transformations _G_ if 1) _G_ acts on the set _Z_ 2) _G_ acts on _V_ as invertible linear transformations, and 3) _r_ is a _G_ -morphism, that is, if _r_ ( _g · z_ ) = _g · r_ ( _z_ ). In our case of interest, the set _Z_ = _Z_ 1 _× . . . ×Zk_ is a finite set of composite concepts ( _e.g_ ., _{_ rainy, sunny _} × {_ morning, evening _}_ ) and _G_ = S _n_ 1 _×. . .×_ S _nk_ is a product of symmetric groups that acts on _Z_ by varying each component separately ( _e.g_ ., swapping “rainy” _↔_ “sunny” and “morning” _↔_ “evening,” independently). Following [26], we say that the action of _G_ on _V_ is “linearly disentangled” if there exists a decomposition _V_ = _V_ 1 _⊕ . . . ⊕ Vk_ such that _g_ = ( _g_ 1 _v_ 1 _, . . . , gkvk_ ) for all _v_ = ( _v_ 1 _, . . . , vk_ ) _∈ V_ and _g_ = ( _g_ 1 _, . . . , gk_ ) _∈ G_ . Intuitively, this means that we can permute the different factors independently by acting with linear transformations on the embedding space. With these definitions in place we have that linear factorizations of embeddings are intimately related to disentangled compositional representations. 

**Proposition 6.** _Let r_ ( _Z_ ) _be a set of decomposable vectors of maximal dimension. Then r is compositional for some disentangled action of G_ = S _n_ 1 _× . . . ×_ S _nk on V . Conversely, if r is compositional for a disentangled action of G, then the vectors r_ ( _Z_ ) _are decomposable._ 

## **4. Decomposable Embeddings in VisionLanguage Models** 

In this section, we discuss linear factorizations from a probabilistic viewpoint in the context of vision-language models (VLMs). A priori, it may not be clear why the geometric notion of decomposable embeddings should be relevant in practice—for example, in the case of CLIP’s normalized embeddings, it may seem that non-linear spherical geometry should come into play. In this section, however, we argue that vector factorizations have simple probabilistic intepretations, and in particular, we should expect these structures to be present in real data embeddings. 

In the following, we write _X_ for a set of texts and _Y_ for a set of images (for simplicity, we consider a finite set of text and images, which will always be the case in practice). We consider a VLM that uses parametric encoders of texts _x �→_ _**u** x_ and of images _y �→_ _**v** y_ into _V_ = R _[d]_ to model the conditional log-probabilities of _x_ given _y_ and _y_ given _x_ in a bilinear fashion: 

**==> picture [236 x 37] intentionally omitted <==**

For example, CLIP [41] uses both expressions in equation 5 to optimize a symmetric cross-entropy. This setup is similar to the one used in NLP for context-based embeddings [36] and also in transformer-based language modeling [47], the main difference being that in those cases only one of the two expressions in equation 5 is used (to model words based on context). Much of the discussion that follows can be applied to these cases as well, but we focus on VLMs for clarity. 

For any given pair of embeddings _**u** x,_ _**u** y_ there exists a unique probability _p_ ( _x, y_ ) on _X × Y_ compatible with these embeddings which satisfies 

**==> picture [188 x 13] intentionally omitted <==**

In the following, we consider the distribution on _X × Y_ expressed by a model and defined by equation 6. After the learning stage, this distribution should reflect a “true” distribution on the same space. We remark, however, that the embedding dimension _d_ is in practice much smaller than the number of images or texts used in training, which means that we are actually imposing a _low-rank constraint_ on the joint probability distribution. In NLP, this effect has been referred to as the “softmax bottleneck” [49]. 

We now consider a set of factors _Z_ = _Z_ 1 _× . . . ×Zk_ and assume that each _z ∈Z_ is represented by a string _x_ ( _z_ ) _∈X_ . Note that formally we could have associated factors with images rather than texts, however it is more natural to express discrete concepts as text. The factors can correspond to combinations of particular tokens ( _e.g_ ., attributes and objects) but the association with strings could potentially be 

more complex ( _e.g_ ., (“royal”, “man”) _�→_ “king”). The VLM model now provides an embedding of _Z_ via _z �→_ _**u** x_ ( _z_ ). 

**Proposition 7.** _In the setting described above, and assuming that Span_ ( _**v** y, y ∈Y_ ) = R _[d] , the embedding z �→_ _**u** x_ ( _z_ ) _of Z is decomposable in the sense of Definition 1 if and only if there exists functions q_ 0 _, . . . , qk such that_ 

**==> picture [202 x 11] intentionally omitted <==**

**==> picture [162 x 11] intentionally omitted <==**

**Corollary 8.** _Under the assumptions of Proposition 7, an embedding z �→_ _**u** x_ ( _z_ ) _of Z is decomposable if only if the factors zi are conditionally independent given any image y._ 

It is perhaps not surprising that the log-linear form of the model translates multiplicative decompositions into additive ones. It may be counterintuitive, however, that the conditional probabilities _p_ ( _zi|y_ ) as _y_ varies actually depend on _all_ of the ideal word vectors _**u** zi_ , since normalizing constants can change with _y_ . Indeed we have that 

**==> picture [194 x 14] intentionally omitted <==**

where _h_ ( _Zj_ = _i, y_ ) is a function that depends on _y_ and all vectors corresponding to _Zj_ with _j_ = _i_ . In this sense, the geometric perspective of factorization is simpler since it disregards this dependence as _y_ varies. 

The conditional independence from Proposition 7 may seem like a strict requirement and may not be obviously true in the real world. For this reason, we discuss some relaxed conditions and explain what they imply in terms of decomposable structures. First, given an image _y ∈Y_ , we say that the probability _p_ ( _x_ ( _z_ ) _, y_ ) is _mode-disentangled_ (for the factor _Zi_ ) if 

**==> picture [227 x 29] intentionally omitted <==**

for all _z−i_ := ( _z_ 1 _, . . . , zi−_ 1 _, zi_ +1 _, . . . , zk_ ) and _z−[′] i_[:=] ( _z_ 1 _[′][, . . . , z] i[′] −_ 1 _[, z] i[′]_ +1 _[, . . . , z] k[′]_[)][.][Intuitively, this simply means] means that it is possible to determine the most likely value of the factor _Zi_ by disregarding all of the remaining factors. Similarly, we say that _p_ ( _x_ ( _z_ ) _, y_ ) is _order-disentangled_ (for the factor _Zi_ ) if 

**==> picture [215 x 27] intentionally omitted <==**

for all _z−i_ and _z−[′] i_[.] This now means that it is possible to _rank_ the values of the factor _Zi_ by disregarding all of the remaining factors. It is easy to see that conditional independence implies order-disentanglement which in turn implies mode-disentanglement. If _|Zi| ≤_ 2, then modedisentanglement and order-disentanglement are equivalent. 

**Proposition 9** (Relaxed feasibility of linear factorizations) **.** _1) If y ∈Y is such that p_ ( _x_ ( _z_ ) _, y_ ) _is mode-disentangled, theirthen onedecomposablecan replaceapproximationsthe embedding_ _**u**_ ˜ _xvectors_ ( _z_ ) _from_ _**u** xProposi-_ ( _z_ ) _with tion 4 (for any choice of weights) and obtain the same prediction for z given y; 2) If p_ ( _x_ ( _z_ ) _, y_ ) _is order-disentangled for all images y sampled from a distribution with full support over the unit sphere, then the vectors_ _**u** x_ ( _z_ ) _are necessarily decomposable._ 

The second part of this statement means that, roughly speaking, we should espect that imposing orderdisentanglement for an increasing number of images would gradually lead to decomposable embeddings. 

**Example 10.** Let _Z_ be of the form _{o_ 1 _, o_ 2 _} × {c_ 1 _, c_ 2 _}_ (objects, contexts) and let _x_ ( _z_ ) be the corresponding collection of strings ( _e.g_ ., _x_ ( _oi, cj_ ) =“a photo of a [ _oi_ ] in [ _cj_ ]”). Then mode and order disentanglement are equivalent and mean that 

**==> picture [202 x 70] intentionally omitted <==**

These are reasonable conditions on the probability _p_ ( _x_ ( _z_ ) _, y_ ) since it is normally possible to discriminate object and context in an image independently. If _p_ ( _x_ ( _z_ ) _, y_ ) and _y_ satisfy equation 11, then the first part of Proposition 9 means that we can use two (approximate) “ideal word” vectors _**u** o_ 1 = _−_ _**u** o_ 2 and _**u** c_ 1 = _−_ _**u** c_ 2 instead of the four original vectors _**u** x_ ( _oi,cj_ ) to assign the correct label to _y_ . The second part of Proposition 9 means that if equation 11 holds for “all” images _y_ ( _i.e_ ., vectors covering the unit sphere), then the original vectors _**u** x_ ( _oi,cj_ ) are actually decomposable. 

## **5. Experiments** 

We now empirically investigate the presence and usefulness of decomposable structures in real VLM embeddings. In all of our experiments, we use a pre-trained CLIP encoder [41][1] . Unless stated otherwise, we compute decomposable approximations of embeddings using Proposition 4 with _αzi_ = _n_ 1 _i_[and] _[β][z]_[=][�] _n_[1] _i_[.][We][use][different] datasets that have a compositional nature: MIT-states [28] and UTZappos [50], that are image classification datasets where labels are pairs attribute–object; CelebA [33] and Waterbirds [44] in which images have a label and a spurious attribute; and DeepFashion2 [23] with PerVL anno- 

> 1We use the HuggingFace implementation of CLIP with the publicly available checkpoint based on a ViT-L/14 vision transformer. See https: //huggingface.co/openai/clip-vit-large-patch14 

**==> picture [233 x 134] intentionally omitted <==**

**----- Start of picture text -----**<br>
0.2<br>0.2 0.00.2<br>0.00.2 0.20.0 0.2 0.20.0 0.2 0.20.00.2 0.2 0.0 0.2 0.20.00.2 0.20.00.2 0.2 0.0 0.2 0.20.00.2 0.20.00.2 0.2 0.0 0.2<br>MIT-States MIT-States Zappos Zappos<br>0.2 0.2 0.2<br>0.0 0.2 0.0 0.0<br>0.2 0.0 0.2 0.2<br>0.2<br>0.20.00.2 0.20.0 0.2 0.20.0 0.2 0.20.00.2 0.20.00.2 0.2 0.0 0.2 0.20.00.2 0.20. 0 0.2<br>**----- End of picture text -----**<br>


Figure 2: **Visualization of embeddings.** _Top_ : projected embeddings of manually constructed strings associated with decomposable concepts. _Bottom:_ projected embeddings for strings of the type “an image of a [a] [o]” for randomly chosen attributes and objects from MIT-states [28] and UTZappos [50]. Symmetric structures indicate that embeddings are approximately decomposable. See text for details. 

tations from [14], where the goal is to retrieve object instances from different contexts. We also include a visualization of ideal words using a CLIP-guided diffusion model (Stable Diffusion 2.1[2] ) [43]. We emphasize that our goal is not to achieve state-of-the-art results, although we will see that linear manipulations can be surprisingly effective and sometimes outperform significantly more complex methods. Rather, we aim to show that linear decomposable structures in embedding spaces provide a useful conceptual and practical framework for _understanding_ and _controlling_ the behavior of pre-trained VLMs. 

**Visualization of embeddings.** Figure 2 shows some examples of embeddings of composite strings, visualized in 3D using PCA. In the top row, we show examples of manually constructed strings. In order: “a photo of a _{_ red, blue, pink _} × {_ car, house _}_ ”; “a photo of a _{_ big, small _} × {_ cat, dog _} × {_ eating, drinking _}_ ”; “ _{_ a photo of a, a picture of a _} × {_ place, object, person _}_ ”; “king, queen, man, woman, boy, girl” (where one factor would correspond to male-female and the other to a generic context). In the bottom row, we present strings of the type “an image of a [a] [o]” for randomly chosen attributes and objects from MITstates [28] and UTZappos [50] (first using two attributes and three objects, and then using three attributes and two objects). Here we always use either 2 _×_ 3 or 2 _×_ 2 _×_ 2 concepts since these decomposable structures have expected affine dimension 4, or linear dimension 3. The presence of roughly parallel edges and faces in these figures indicate that embeddings are approximately decomposable. We note that in many of these examples the factorization of the concepts is already reflected in the _syntax_ of the strings, _i.e_ ., in 

the presence of repeated substrings in prompts with similar meaning. However, factorized vectors also encode semantic aspects, as can be seen in the last two examples from the first row. In the fourth example, the encoded strings have no repeated substrings, so the structure is “emergent”; in the third example, the factor corresponding to _{_ a photo of a, a picture of a _}_ results in an ideal word vector with a smaller norm compared to the to other directions (resulting in a “squashed” triangular prism), as one might expect since this factor is not semantically significant. We refer to the Appendix for a more in-depth discussion. 

**Compositional classification.** We evaluate the usefulness of linear decomposable approximations for object-attribute labels of the MIT-states [28] and UTZappos [50] datasets. The default strategy for applying CLIP in a zero-shot fashion on these datasets is to use text captions such as _x_ ( _a, o_ )=“an image of a [ _a_ ] [ _o_ ].” This results in _nobj × nattr_ captions that each image must be compared with. We want to explore whether the embedding vectors˜ _**u** x_ ( _a,o_ ) can be approximated with a decomposable set _**u** x_ ( _a,o_ ) = _**u**_ 0 + _**u** a_ + _**u** o_ , so that inference can be performed using only _nobj_ + _nattr_ embedding vectors. The intuitive choice for such vectors would be to use the representations of captions such as “image of a [ _a_ ] object” and “image of a [ _o_ ].” We compare this choice with using the “ideal words” associated with the original captions, where the representation of 1 an object _o_ is simply given by _**u** o_ := _nattr_ � _a_ _**[u]**[x]_[(] _[a,o]_[)][, and] similarly for attributes, as in Proposition 4 (in this setting, there is no need to remove the mean vector _**u**_ 0 since it is multiplied with every image vector). The resulting disjoint representations for objects and attributes ( _**u** o_ and _**u** a_ ) are “contextualized,” in the sense that they optimally approximate the original pairwise embeddings. In Table 1, “pair” refers to using the original pairwise labels, “real words” uses the embeddings of words corresponding to objects and attributes using “image of a [ _a_ ] object” and “image of a [ _o_ ].”, while “ideal words” computes the vector ideal words for the factorization. We see that ideal words clearly outperform the _real words_ baseline, and often even surpass the accuracy of _pair_ . For MIT-States, using decomposable labels translates into using 360 vs. 28175 class vectors. 

**Debiasing.** We can apply the decomposition into ideal words as a baseline strategy to remove contexts or biases from embeddings. The debiasing task can be formalized using the group robustness framework proposed in [44]. In this setting, we are given a collection of labels _Y_ and spurious attributes _A_ , and we define a “group” as a pair _g ∈Y × A_ . Assuming that each group corresponds to a probability _Pg_ on an input space _X_ , the goal is to find a classifier _f_ : _X →Y_ that leads to a small gap between worst-group error and average error: 

**==> picture [214 x 16] intentionally omitted <==**

2https://huggingface.co/stabilityai/ stable-diffusion-2-1 

||**Method**<br>**Pair Acc**<br>**Attr Acc**<br>**Obj Acc**|
|---|---|
|MIT-states [28]|pair<br>7.7%<br>16.2%<br>47.8%|
||real words<br>10.0%<br>19.3%<br>49.3%<br>ideal words<br>**11.5%**<br>**21.4%**<br>**50.8%**|
|UT Zappos [50]|pair<br>**12.4%**<br>17.1%<br>**55.7%**|
||real words<br>8.4%<br>10.3%<br>51.0%<br>ideal words<br>10.8%<br>**19.2**%<br>55.3%|



Table 1: **Zero-shot image classification results on compositional datasets.** Here “pair” refers to using all attributeobject pairs as candidate labels; “real words” refers to using labels corresponding to real words ( _i.e_ ., separate attribute and object labels); “ideal words” refers to using compositional labels based on ideal words. Ideal words always lead to better accuracy than real words and often even outperform pairwise labels. 

In a zero-shot setting with CLIP, classifiers are prompts that inherit biases from the dataset used in pre-training, so group robustness is not guaranteed. To address this problem, the authors of [11] propose a method for debiasing prompts that finds a projection map that makes spurious prompts irrelevant (following [6]) and then additionally regularizes the projection map to ensure that certain prompts are mapped near each other in embedding space. Here we note that a much simpler baseline would be to use ideal words to leverage the joint label-attribute representation provided by the pre-trained VL model and “average out” spurious attributes. More precisely, starting from a set of embeddings _**u**_ ( _y,a_ ) corresponding to prompts representing each group _g_ = ( _y, a_ ), ideal words suggest to define the encoding of 1 each label _y_ to be _**u** y_ := _|A|_ � _a∈A_ _**[u]**_[(] _[y,a]_[)] _[.]_[Once][again,] this is the same as the (shifted) ideal word corresponding to _y_ , obtained by approximating pairwise embeddings of labels and attributes in a decomposable way. Following [11], we evaluate group robustness of unbiased prompts on the Waterbird [44] and CelebA [33] datasets. For the Waterbird dataset, the labels are “landbird” and “waterbird,” and the confounding factor is water/land background. For the CelebA dataset, the labels are “blond” and “dark” hair and the confounding factor is the binary gender. For our simple unbiasing method, we prepend prompts associated with labels with prompts associated with spurious attributes, and then average over all the spurious prompts. In both datasets, we consider exactly the same prompts for spurious attributes and labels used in [11] (see the Appendix for a description). Our results are shown in Table 2. On the CelebA dataset, our simple averaging strategy achieves a much smaller gap between average and worst group accuracy than the method proposed in [11] (1.6 vs 10.1). For Waterbird datsets, the gap is larger but comparable, and average accuracy is higher. 

|**Waterbird**[44]<br>**CelebA**[33]<br>WG<br>Avg<br>Gap<br>WG<br>Avg<br>Gap|**Waterbird**[44]<br>**CelebA**[33]<br>WG<br>Avg<br>Gap<br>WG<br>Avg<br>Gap|**Waterbird**[44]<br>**CelebA**[33]<br>WG<br>Avg<br>Gap<br>WG<br>Avg<br>Gap|
|---|---|---|
|Zero-shot<br>Orth-Proj [11]<br>Orth-Cali [11]<br>Ideal Words|45.3<br>84.4<br>39.1<br>61.4<br>86.4<br>25.0<br>**68.8**<br>84.5<br>**15.7**<br>64.6<br>**88.0**<br>23.3|72.8<br>**87.6**<br>14.9<br>71.1<br>87.0<br>15.9<br>76.1<br>86.2<br>10.1<br>**83.9**<br>85.5<br>**1.6**|



Table 2: **Group robustness results.** Ideal words can be used as a simple yet performant baseline for debiasing applications. 

|plications.||
|---|---|
||Text Only<br>AvgImg+Text<br>PALAVRA [14]<br>IW|
|DeepFashion2 [23]|17.6_±_0.0<br>21.7_±_2.4<br>28.4_±_0.7_∗_<br>**37.0**_±_1.1|
||IW w.o. mean removal<br>IW with Norm on mean<br>IW|
|DeepFashion2 [23]|22.1_±_2.4<br>36.5_±_1.4<br>**37.0**_±_1.1|



Table 3: **Concept retrieval results.** Mean Reciprocal Rank retrieval metric on the DeepFashion2 [23] with annotations from PerVL [14]. Numbers with _[∗]_ are taken from [14]. 

**Composing concepts and contexts.** We perform experiments using the DeepFashion2 dataset [23] with the captions provided in PerVL [14]. This dataset contains images of 100 unique fashion items (“concepts”) with textual descriptions. The task is to retrieve an image given a text query that includes a personalized concept that is specified using a small number of examples (5 samples). An example of a text query is “The [CONCEPT] is facing a glass store display.” In [14], the authors propose a method called PALAVRA that trains new CLIP tokens to be associated with the custom concept; the learned tokens can then be used within natural language for retrieving images. The authors compare their method with a baseline approach dubbed “AvgIm+Text” which consists in averaging the CLIP embedding of the concept support images and of the embedded text query. This strategy is presented as the second best approach after PALAVRA. Inspired by our linear factorization of concepts and contexts, we propose to use a modification of AvgIm+Text where instead of averaging text and image embeddings, we add to the text embedding the _difference_ between mean image embeddings of the specialized concept (“my shirt”) and the mean embeddings of the general (coarse-grained) concept images (all images of shirts in the dataset). For a concrete example, if [CONCEPT] is a particular instance of a shirt, then the AvgIm+Text approach would be as follows: 

## **AvgIm+Text** : 

## _**u**_ (“A person wearing [CONCEPT] sitting on a couch) _≈_ _**u**_ (“A person wearing a shirt stting on a couch) + Norm(Mean _{_ _**v**_ (CONCEPT) _}_ ) _,_ 

where _**u**_ is the text embedding and _**v**_ is the image embedding, Mean means the mean over supporting samples, and Norm means normalization. In contrast, we propose to use 

the following strategy: 

## **Ideal Words** : 

_**u**_ (“A person wearing [CONCEPT] sitting on a couch) 

_≈_ _**u**_ (“A person wearing a shirt stting on a couch) _−_ Mean _{_ _**v**_ (shirt) _}_ + Mean _{_ _**v**_ (CONCEPT) _}._ 

Our results are shown in Table 3. Remarkably, this simple strategy that uses CLIP embeddings and _does not require any training_ outperforms PALAVRA by a large margin (in our experiments, we used the implementation and evaluation code provided in [14] with only minimal changes). This modified approach can be interpreted from the perspective of decomposable embeddings, since we are assuming that _**u**_ (context _,_ CONCEPT) _−_ _**u**_ (context _,_ shirt) does not significantly depend on the context and can be approximated as the difference mean vectors representing the specific CONCEPT and the generic shirt. Table 3 also includes ablations for the two modifications we made w.r.t. to AvgIm+Text proposed in [14] ( _i.e_ . skipping the normalization step and removing the mean of the coarse-grained concept). 

**Visualizing ideal words.** We propose to visualize the effect of linear-algebraic operations with ideal words using a CLIP-guided diffusion model (Stable Diffusion 2.1). In this setting, we compute ideal words of decomposable strings in the same way as before (as in Proposition 4 and Example 5), with the only difference that we now consider the encoded representation of the entire string before the final projection layer of the text encoder (treating the concatenated token representations as a long vector), since this is required for conditioning the diffusion model. An illustrative example is shown Figure 3. We mention that [48, 32] have also proposed algebraic manipulations to control visual generation in a compositional way; however both of those works perform operations on score functions rather than on embedding vectors, which means that their approach requires modifying the diffusion process. In contrast, similar to the prompt debiasing method from [11], we simply modify the prompt embeddings that condition the generation. In this paper, we use generative models as a qualitative proof of the validity of ideal words as approximations for embeddings; we leave a detailed exploration of applying these decompositions for controlling image generation to future work. 

## **6. Conclusion** 

We have investigated compositional structures in VLM embeddings and argued that contextual text embeddings are often well-approximated by linear combinations of smaller sets of vectors. Optimal choices for these vectors are not embeddings of actual words, but rather “ideal words” that can be easily obtained as weighted averages of embeddings of longer strings of text. We showed that this simple idea can be used to design effective baseline methods 

**==> picture [7 x 185] intentionally omitted <==**

**----- Start of picture text -----**<br>
original<br>IW<br>IW<br>IW<br>**----- End of picture text -----**<br>


Figure 3: **Visualization of ideal words.** _First row:_ images generated by Stable Diffusion with the prompt “a photo of a green house.” Because of the contextual encoder, “house” influences the meaning “green.” _Following rows:_ we compute ideal words approximations for strings of the form “a photo of a [color] _×_ [object],” using five colors and four objects. In the second row, we generate images using the vector _**u**_ 0 + _**u**_ green + _**u**_ house. Now _**u**_ green means greencolored because of how the string “green” composes with most objects. In the third row, we generate images using _**u**_ 0 + _**u**_ [color] + _**u**_ house for different colors; in the fourth row, we use _**u**_ 0 + _**u**_ [color] + _**u**_ bike. The images were not cherrypicked or manipulated in any way. This example shows that we can generate embeddings of composite concepts by simply adding vectors in the representation space. 

for different visual language tasks (compositional classification/retrieval, debiasing, and image generation) and to control the behavior of VLMs. 

In the future, we will focus on practical applications of ideal word decompositions such as compositional image generation. Furthermore, we would like to find ways of customizing ideal words using training data, for example by incorporating linear factorizations in fine-tuning strategies, or by introducing kernelized versions of these decompositions that have learnable parameters. 

Finally, we remark that our discussion in Section 4 was mainly focused on embedding vectors from a single modality (text), however the strategy we used for concept retrieval in Section 5 suggests that it is possible to perform linear algebraic operations using vectors from _both_ modalities (text/vision). Although it is generally known that visual and text embeddings in CLIP are not well-aligned [30], 

our linear manipulations actually only require for the _differences_ between embedding vectors of the same modality to be aligned. Interestingly, this sort of weak alignment implies that vector representations of a concept _c_ in any modality can be (approximately) written as 

**==> picture [176 x 11] intentionally omitted <==**

where _**w**_ modality may be seen as the ideal word vector corresponding to the modality factor for vision/text. 

## **References** 

- [1] Alessandro Achille and Stefano Soatto. Emergence of Invariance and Disentanglement in Deep Representations. _arXiv:1706.01350 [cs, stat]_ , June 2018. arXiv: 1706.01350. 2 

- [2] Carl Allen and Timothy Hospedales. Analogies Explained: Towards Understanding Word Embeddings. page 9. 2 

- [3] Jacob Andreas. Measuring Compositionality in Representation Learning, Apr. 2019. 2 

- [4] Sanjeev Arora, Yuanzhi Li, Yingyu Liang, Tengyu Ma, and Andrej Risteski. A latent variable model approach to pmibased word embeddings. _Transactions of the Association for Computational Linguistics_ , 4:385–399, 2016. 2 

- [5] Marco Baroni and Roberto Zamparelli. Nouns are Vectors, Adjectives are Matrices: Representing Adjective-Noun Constructions in Semantic Space. page 11. 2 

- [6] Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai. Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings, July 2016. arXiv:1607.06520 [cs, stat]. 7 

- [7] Zied Bouraoui, Shoaib Jameel, and Steven Schockaert. Relation Induction in Word Embeddings Revisited. page 11. 2 

- [8] Christopher P. Burgess, Irina Higgins, Arka Pal, Loic Matthey, Nick Watters, Guillaume Desjardins, and Alexander Lerchner. Understanding disentangling in _\_ beta-VAE. _arXiv:1804.03599 [cs, stat]_ , Apr. 2018. 2 

- [9] Kwan Ho Ryan Chan, Yaodong Yu, Chong You, Haozhi Qi, John Wright, and Yi Ma. ReduNet: A White-box Deep Network from the Principle of Maximizing Rate Reduction, Nov. 2021. 2 

- [10] Noam Chomsky. Syntactic structures. In _Syntactic Structures_ . De Gruyter Mouton, 2009. 2 

- [11] Ching-Yao Chuang, Varun Jampani, Yuanzhen Li, Antonio Torralba, and Stefanie Jegelka. Debiasing Vision-Language Models via Biased Prompts, Jan. 2023. arXiv:2302.00070 [cs]. 3, 7, 8, 5, 6 

- [12] Stephen Clark. Vector Space Models of Lexical Meaning. In Shalom Lappin and Chris Fox, editors, _The Handbook of Contemporary Semantic Theory_ , pages 493–522. John Wiley & Sons, Ltd, Chichester, UK, Aug. 2015. 2 

- [13] Bob Coecke, Mehrnoosh Sadrzadeh, and Stephen Clark. Mathematical Foundations for a Compositional Distributional Model of Meaning. page 34. 2 

- [14] Niv Cohen, Rinon Gal, Eli A. Meirom, Gal Chechik, and Yuval Atzmon. “This Is My Unicorn, Fluffy”: Personalizing Frozen Vision-Language Representations. In Shai Avidan, Gabriel Brostow, Moustapha Ciss´e, Giovanni Maria Farinella, and Tal Hassner, editors, _Computer Vision – ECCV 2022_ , volume 13680, pages 558–577. Springer Nature Switzerland, Cham, 2022. Series Title: Lecture Notes in Computer Science. 6, 7, 8, 5 

- [15] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. _arXiv:1810.04805 [cs]_ , May 2019. arXiv: 1810.04805. 2 

- [16] Willis D Ellis. _A source book of Gestalt psychology_ . Routledge, 2013. 2 

- [17] Kawin Ethayarajh. How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings, Sept. 2019. 2 

- [18] Kawin Ethayarajh, David Duvenaud, and Graeme Hirst. Towards Understanding Linear Word Analogies, Aug. 2019. arXiv:1810.04882 [cs]. 2 

- [19] Jacob Feldman. Regularity-based perceptual grouping. _Computational Intelligence_ , 13(4):582–623, 1997. 2 

- [20] Jerry A Fodor and Ernest Lepore. _The compositionality papers_ . Oxford University Press, 2002. 2 

- [21] William Fulton and Joe Harris. _Representation Theory_ , volume 129 of _Graduate Texts in Mathematics_ . Springer New York, New York, NY, 2004. 4 

- [22] Alona Fyshe, Leila Wehbe, Partha P. Talukdar, Brian Murphy, and Tom M. Mitchell. A Compositional and Interpretable Semantic Space. In _Proceedings of the 2015 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 32–41, Denver, Colorado, 2015. Association for Computational Linguistics. 2 

- [23] Yuying Ge, Ruimao Zhang, Lingyun Wu, Xiaogang Wang, Xiaoou Tang, and Ping Luo. A versatile benchmark for detection, pose estimation, segmentation and re-identification of clothing images. _CVPR_ , 2019. 5, 7 

- [24] Stuart Geman, Daniel F Potter, and Zhiyi Chi. Composition systems. _Quarterly of Applied Mathematics_ , 60(4):707–736, 2002. 2 

- [25] Alex Gittens, Dimitris Achlioptas, and Michael W. Mahoney. Skip-Gram - Zipf + Uniform = Vector Additivity. In _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 69–76, Vancouver, Canada, 2017. Association for Computational Linguistics. 2 

- [26] Irina Higgins, David Amos, David Pfau, Sebastien Racaniere, Loic Matthey, Danilo Rezende, and Alexander Lerchner. Towards a Definition of Disentangled Representations, Dec. 2018. 2, 4 

- [27] Dieuwke Hupkes, Verna Dankers, Mathijs Mul, and Elia Bruni. Compositionality decomposed: How do neural networks generalise?, Feb. 2020. 2 

- [28] Phillip Isola, Joseph J. Lim, and Edward H. Adelson. Discovering states and transformations in image collections. In _2015 IEEE Conference on Computer Vision and Pattern_ 

_Recognition (CVPR)_ , pages 1383–1391, Boston, MA, USA, June 2015. IEEE. 5, 6, 7 

- [29] Omer Levy and Yoav Goldberg. Neural word embedding as implicit matrix factorization. In _Advances in Neural Information Processing Systems_ , pages 2177–2185, 2014. 2 

- [30] Weixin Liang, Yuhui Zhang, Yongchan Kwon, Serena Yeung, and James Zou. Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning, Oct. 2022. arXiv:2203.02053 [cs]. 8 

- [31] Tal Linzen. Issues in evaluating semantic spaces using word analogies. In _Proceedings of the 1st Workshop on Evaluating Vector-Space Representations for NLP_ , pages 13–18, Berlin, Germany, 2016. Association for Computational Linguistics. 2 

- [32] Nan Liu, Shuang Li, Yilun Du, Antonio Torralba, and Joshua B. Tenenbaum. Compositional Visual Generation with Composable Diffusion Models, Jan. 2023. arXiv:2206.01714 [cs]. 2, 8 

- [33] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep Learning Face Attributes in the Wild, Sept. 2015. arXiv:1411.7766 [cs]. 5, 7, 6 

- [34] Yi Ma, Doris Tsao, and Heung-Yeung Shum. On the Principles of Parsimony and Self-Consistency for the Emergence of Intelligence, July 2022. 2 

- [35] Massimiliano Mancini, Muhammad Ferjad Naeem, Yongqin Xian, and Zeynep Akata. Learning Graph Embeddings for Open World Compositional Zero-Shot Learning, Apr. 2022. 5 

- [36] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. _arXiv preprint arXiv:1301.3781_ , 2013. 2, 3, 4 

- [37] Jeff Mitchell and Mirella Lapata. Vector-based Models of Semantic Composition. page 9. 2 

- [38] Nihal V. Nayak, Peilin Yu, and Stephen H. Bach. Learning to Compose Soft Prompts for Compositional Zero-Shot Learning, Apr. 2022. 5 

- [39] Jeffrey Pennington, Richard Socher, and Christopher Manning. Glove: Global vectors for word representation. In _Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 1532–1543, 2014. 2 

synthesis with latent diffusion models. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pages 10684–10695, June 2022. 2 

   - [43] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-Resolution Image Synthesis with Latent Diffusion Models, Apr. 2022. arXiv:2112.10752 [cs]. 6 

   - [44] Shiori Sagawa, Pang Wei Koh, Tatsunori B. Hashimoto, and Percy Liang. Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization, Apr. 2020. arXiv:1911.08731 [cs, stat]. 5, 6, 7 

   - [45] Yeon Seonwoo, Sungjoon Park, Dongkwan Kim, and Alice Oh. Additive Compositionality of Word Vectors. In _Proceedings of the 5th Workshop on Noisy User-generated Text (W-NUT 2019)_ , pages 387–396, Hong Kong, China, 2019. Association for Computational Linguistics. 2 

   - [46] Asahi Ushio, Luis Espinosa Anke, Steven Schockaert, and Jose Camacho-Collados. BERT is to NLP what AlexNet is to CV: Can Pre-Trained Language Models Identify Analogies? In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)_ , pages 3609–3624, Online, 2021. Association for Computational Linguistics. 2 

   - [47] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention Is All You Need. _arXiv:1706.03762 [cs]_ , Dec. 2017. arXiv: 1706.03762. 2, 4 

   - [48] Zihao Wang, Lin Gui, Jeffrey Negrea, and Victor Veitch. Concept Algebra for Text-Controlled Vision Models, Feb. 2023. arXiv:2302.03693 [cs, stat]. 2, 8, 6 

   - [49] Zhilin Yang, Zihang Dai, Ruslan Salakhutdinov, and William W. Cohen. Breaking the Softmax Bottleneck: A High-Rank RNN Language Model, Mar. 2018. 4 

   - [50] Aron Yu and Kristen Grauman. Fine-Grained Visual Comparisons with Local Learning. In _2014 IEEE Conference on Computer Vision and Pattern Recognition_ , pages 192–199, Columbus, OH, USA, June 2014. IEEE. 5, 6, 7 

- [40] Matthew Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. Deep Contextualized Word Representations. In _Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)_ , pages 2227– 2237, New Orleans, Louisiana, 2018. Association for Computational Linguistics. 2 

- [41] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning Transferable Visual Models From Natural Language Supervision. _arXiv:2103.00020 [cs]_ , Feb. 2021. 1, 4, 5 

- [42] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image 

## **Supplementary Material** 

This supplementary material is organized as follows: in Section A we provide proofs for all the statements of the paper and we discuss some connections with mathematical representation theory; in Section B we give details on the datasets and prompts used for our experiments; in Section C we present some additional experimental results and qualitative examples. 

## **A. Proofs** 

**Lemma 2.** _1) A collection of vectors r_ ( _Z_ ) _is decomposable if and only if the vector difference_ _**u** z −_ _**u** z′ does not depend on the components that z, z[′] ∈Z share in common. 2) If |Zi|_ = _ni, then the dimension of Span_ ( _r_ ( _Z_ )) _is at most_ 1 +[�] _[k] i_ =1[(] _[n][i][ −]_[1)] _[.]_ 

_Proof._ (1) If the vectors are decomposable, then clearly the vector differences _**u** z −_ _**u** z′_ do not depend on the components that _z, z[′]_ share in common since the corresponding vectors cancel out. For the converse, fix _z_ = ( _z_ 1 _, . . . , zk_ ) _∈ Z_ arbitrarily and choose any _k_ vectors _**u** z_ 1 _, . . . ,_ _**u** zk_ such that _**u** z_ = _**u** z_ 1 + _. . ._ + _**u** zk_ . Now for any _zi[′][∈Z][i]_[and any] _i_ = 1 _, . . . , k_ , define 

**==> picture [150 x 25] intentionally omitted <==**

**==> picture [153 x 12] intentionally omitted <==**

**==> picture [194 x 76] intentionally omitted <==**

(2) We have that 

**==> picture [229 x 78] intentionally omitted <==**

**==> picture [18 x 10] intentionally omitted <==**

¯ 1 where _**u** Z_ 1 := _ni_ � _zi∈Zi_ _**[u]**[z] i_[and] _**[u]**_[˜] _[z] i_[:=] _**[ u]**[z] i[−]_ _**[u]**[Z] i_[. Since] � _zi∈Zi_ _**[u]**_[˜] _[z] i_[= 0][, equation][ 14][ shows that any linear combi-] nation of the vectors _**u** z, z ∈Z_ can be written as a linear combination of 1 +[�] _[k] i_ =1[(] _[n][i][ −]_[1)][ vectors.] 

**Lemma 3** (Centered decomposition) **.** _If a collection of vectors r_ ( _Z_ ) _is decomposable, then there exist unique vectors_ 

_**u**_ 0 _∈ V and_ _**u** zi ∈ V for all zi ∈Zi (i_ = 1 _, . . . , k) such that_[�] _zi∈Zi_ _**[u]**[z] i_[= 0] _[ for all][ i][ and]_ 

**==> picture [178 x 11] intentionally omitted <==**

_for all z_ = ( _z_ 1 _, . . . , zk_ ) _._ 

_Proof._ Following the proof of part 2 of the previous Lemma, it is enough to let _**u**_ 0 := _**u**_ ¯ _Z_ 1 + _. . ._ + _**u**_ ¯ _Zk_ where _**u**_ ¯ _Z_ 1 := _n_ 1 _i_ � _zi∈Zi_ _**[u]**[z] i_[,][and][then][re-center][the][remaining] vectors accordingly. For the uniqueness, we note that equation 2 implies that the vectors _**u**_ 0 _,_ _**u** zi_ , _zi ∈Zi_ satisfy 

**==> picture [233 x 38] intentionally omitted <==**

where _N_ = _n_ 1 _. . . nk_ . In particular, equation 15 shows that _**u**_ 0 _,_ _**u** zi_ , _zi ∈Zi_ are uniquely determined by the original vectors _**u** z_ . 

In the previous proof, we considered a map associating each _**u** z, z ∈Z_ with the vectors given by 

**==> picture [231 x 38] intentionally omitted <==**

˜ It is easy to see that if we define _**u** z_ = _**u**_ 0 + _**u** z_ 1 + _. . ._ + _**u** zk_ then applying equation 16 with the new vectors _**u**_ ˜ _z_ instead of _**u** z_ yields the same components _**u** zi_ . Thus, this map can be seen as a projection onto a decomposable set of vectors. Note that the component vectors satisfy[�] _zi∈Zi_ _**[u]**[z] i_[=][0][.] The following result considers a slightly more general setting in which these components vectors satisfy[�] _αzivzi_ = 0 for some weights _αi_ that sum to 1. 

**Proposition 4.** _Let αzi zi ∈Zi be arbitrary positive weights such that_[�] _zi∈Zi[α][z] i_[=][1] _[,][and][define][β][z]_[:=] � _i[α][z] i[for][all][z]_[=][(] _[z]_[1] _[, . . . , z][k]_[)] _[.][Then,][for][any][norm][∥· ∥] induced by an inner product on V , we have that_ 

**==> picture [183 x 38] intentionally omitted <==**

˜ _is given by_ _**u** z_ = _**u**_ 0 + _**u** z_ 1 + _. . ._ + _**u** zk where_ 

**==> picture [229 x 48] intentionally omitted <==**

_Proof._ Without loss of generality, we may assume that � _z[β][z]_ _**[u]**[z] i_[=][�] _[α][z] i_ _**[u]**[z] i_[=][0][.][Imposing that the derivative] 

of equation 3 with respect to _**u**_ 0 is zero leads to 

**==> picture [197 x 50] intentionally omitted <==**

which implies _**u**_ 0 =[�] _z[β][z]_ _**[u]**[z][.]_[Similarly,][differentiating] with respect to _**u** zi_ we have 

**==> picture [223 x 73] intentionally omitted <==**

which implies that 

**==> picture [192 x 34] intentionally omitted <==**

so _**u** zi_ is as in equation 4. 

**Proposition 6.** _Let r_ ( _Z_ ) _be a set of decomposable vectors of maximal dimension. Then r is compositional for some disentangled action of G_ = S _n_ 1 _× . . . ×_ S _nk on V . Conversely, if r is compositional for a disentangled action of G, then the vectors r_ ( _Z_ ) _are decomposable._ 

_Proof._ Let _r_ ( _Z_ ) be a set of decomposable vectors of maximal dimension. If _W_ := _Span_ ( _**u** z, z ∈Z_ ), then we write _V_ = _W ⊕ W[′]_ , and define a linear action of _G_ on R _[d]_ by associating each group element _g_ = ( _g_ 1 _, . . . , gk_ ) with an invertible linear transformation so that each _gi_ determines a permutation of the vectors _**u** zi_ , while fixing other terms and _W[′]_ . This describes a disentangled action of _G_ , where _V_ = _W[′] ⊕⟨_ _**u**_ 0 _⟩⊕ VZ_ 1 _⊕ . . . ⊕ VZk_ (to be consistent with the original definition, we can set _V_ 1 = _W[′] ⊕⟨_ _**u**_ 0 _⟩⊕ VZ_ 1 and _Vi_ = _VZi_ for _i ≥_ 2). 

For the converse, let _ρ_ : _G → GL_ ( _V_ ) be any linear action of _G_ on _V_ (a group representation). Writing _G_ ˆ _i_ = S1 _× . . . × {e} × . . . ×_ S _k_ (with the identity at the _i_ -th component), we define 

**==> picture [196 x 28] intentionally omitted <==**

Since _G_ acts linearly, these are vector spaces. We also define the linear maps 

**==> picture [171 x 60] intentionally omitted <==**

These are linear projections onto _V_ 0 and _V_[˜] _i_ , respectively, since they map onto these spaces and they fix them. We now define _πi_ := _π_ ˜ _i − π_ 0 and _Vi_ := _Im_ ( _πi_ ). Since _V_[˜] _i ∩ V_[˜] _j_ = _V_ 0 for _i ̸_ = _j_ , we have that _Vi ∩ Vj_ = _{_ 0 _}_ for _i ̸_ = _j_ . In general, we now have that _V_ 0 _⊕ V_ 1 _⊕ . . . ⊕ Vk ⊂ V_ ; if the action _ρ_ is disentangled, however, then 

**==> picture [171 x 10] intentionally omitted <==**

Thus, for any _v ∈ V_ , we have _v_ = _π_ 0( _v_ ) + _π_ 1( _v_ ) + _. . ._ + _πk_ ( _v_ ). Now assume that _r_ : _Z → V_ is a compositional embedding, so _g · r_ ( _z_ ) = _r_ ( _g · z_ ). We observe that _**u** zi_ = _πi_ ( _**u** z_ ) is fixed by S _j_ for _j_ = _i_ , and thus depends only on _zi_ . In fact, the expressions for _π_ 0 _, πi_ applied to _**u** z_ are exactly the projection maps from equation 16. Thus, we can write _**u** z_ = _**u**_ 0 + _**u** z_ 1 + _. . ._ + _**u** zk_ , which means that _r_ ( _Z_ ) are decomposable. 

**Proposition 7.** _In the setting described above, and assuming that Span_ ( _**v** y, y ∈Y_ ) = R _[d] , the embedding z �→_ _**u** x_ ( _z_ ) _of Z is decomposable in the sense of Definition 1 if and only if there exists functions q_ 0 _, . . . , qk such that_ 

**==> picture [202 x 11] intentionally omitted <==**

**==> picture [163 x 11] intentionally omitted <==**

_Proof._ Assume that equation 7 holds, and let _g_ 0( _y_ ) := log( _q_ 0( _y_ )) and _gi_ ( _zi, y_ ) := log( _qi_ ( _z, y_ )). For all _z ∈Z_ , we can write 

**==> picture [227 x 65] intentionally omitted <==**

¯ 1 where _g_ 0( _y_ ) := _g_ 0( _y_ ) +[�] _[k] j_ =1 _nj_ � _zj ∈Zj[g][j]_[(] _[z][j][, y]_[)][and] _g_ ¯ _i_ ( _zi, y_ ) := _g_ ( _zi, y_ ) _− n_ 1 _i_ � _zi[′][∈Z][i][ g][i]_[(] _[z] i[′][, y]_[)][.][It][is][easy][to] verify the following identities for _i_ = 1 _, . . . , k_ : 

**==> picture [219 x 43] intentionally omitted <==**

**==> picture [235 x 86] intentionally omitted <==**

where we used the expression for log _p_ ( _x, y_ ) from equation 6 and the definition of the terms _**u**_ 0 _,_ _**u** zi_ from equation 16. If we now define _**u**_ ˜ _x_ ( _z_ ) := _**u**_ 0 + _**u** z_ 1 + _. . ._ + _**u** zk_ , 

then it follows from equation 24 that _**u**_ ˜ _[⊤] x_ ( _z_ ) _**[v]**[y]_[=] _**[ u]**[⊤] x_ ( _z_ ) _**[v]**[y]_[(=] log _p_ ( _x_ ( _z_ ) _, y_ )) _− c_ 0) for all _z ∈Z_ , _y ∈Y_ . Since by hypothesis _Span_ ( _**v** y, y ∈Y_ ) = R _[d]_ , we conclude that ˜ _**u** x_ ( _z_ ) = _**u** x_ ( _z_ ). Conversely, it is clear that if all _**u** x_ ( _z_ ) decompose as in equation 2, then _p_ ( _x_ ( _z_ ) _, y_ ) has a factored form as in equation 7 for all _y ∈Y_ . 

**Corollary 8.** _Under the assumptions of Proposition 7, an embedding z �→_ _**u** x_ ( _z_ ) _of Z is decomposable if only if the factors zi are conditionally independent given any image y._ 

_Proof._ This follows immediately from the factored form of equation 7. More precisely, the statement means that 

**==> picture [183 x 11] intentionally omitted <==**

˜ 1 ˜ where _p_ ( _z | y_ ) := _Zy[p]_[(] _[x]_[(] _[z]_[)] _[ |][ y]_[)][,] _p_ ( _zi | y_ ) := _Z_ 1 _y_ � _zk_ = _i[p]_[(] _[x]_[(] _[z]_[)] _[ |][ y]_[)][and] _[Z][y]_ := � _z[p]_[(] _[x]_[(] _[z]_[)] _[ |][ y]_[)][.] We observe that equation 25 implies equation 7, since we can write 

**==> picture [200 x 11] intentionally omitted <==**

which has the desired factored form. Conversely, equation 7 means that 

**==> picture [225 x 25] intentionally omitted <==**

where _Zi_ =[�] _zi∈Zi_ ) _[q][i]_[(] _[z][i][, y]_[)][and] _[q]_[˜] _[i]_[(] _[z][i][, y]_[)][=] _Z_ 1 _i[q]_[(] _[z][i][, y]_[)][.] Since[�] _z∈Z[p]_[˜][(] _[z][ |][ y]_[)][=][1][,][we deduce that the] _[ y]_[-dependent] constant on the right of equation 27 is equal to 1, and _q_ ˜ _i_ ( _z, y_ ) = _p_ ˜( _zi|y_ ). 

**Proposition 9** (Relaxed feasibility of linear factorizations) **.** _1) If y ∈Y is such that p_ ( _x_ ( _z_ ) _, y_ ) _is mode-disentangled, thentheir onedecomposablecan replaceapproximationsthe embedding_ _**u**_ ˜ _xvectors_ ( _z_ ) _from_ _**u** xProposi-_ ( _z_ ) _with tion 4 (for any choice of weights) and obtain the same prediction for z given y; 2) If p_ ( _x_ ( _z_ ) _, y_ ) _is order-disentangled for all images y sampled from a distribution with full support over the unit sphere, then the vectors_ _**u** x_ ( _z_ ) _are necessarily decomposable._ 

_Proof._ (1) Assume that _p_ ( _x_ ( _z_ ) _, y_ ) is mode-disentangled. Then we have that 

**==> picture [179 x 104] intentionally omitted <==**

where _**u** zi_ is as in equation 16, or as in the weighted version from equation 4. This implies that we can perform inference using the decomposable approximations _**u**_ ˜ _x_ ( _z_ ) instead of the original vectors. 

2) We will use the notation _z_ = ( _zi, zj, z−{i,j}_ ) where _z−{i,j}_ := ( _z_ 1 _, . . . , zi−_ 1 _, zi_ +1 _, . . . zj−_ 1 _, zj_ +1 _, . . . , zk_ ). If _p_ ( _x_ ( _z_ ) _, y_ ) is order-disentangled for _y_ , then for any _zi, zi[′][∈] Zi_ and _zj, zj[′][∈Z][j]_ 

**==> picture [223 x 32] intentionally omitted <==**

and similarly 

**==> picture [223 x 33] intentionally omitted <==**

If these relations hold for any vector _**u** y_ , then it means that 

**==> picture [215 x 62] intentionally omitted <==**

for some positive scalars _λ, µ ∈_ R. It follows from Lemma 11 below that either all four points in equation 31 are aligned, or _λ_ = _µ_ = 1. However, we can exclude that all four points are aligned for otherwise the largest between _p_ ( _x_ ( _zi, zj, z−{i,j}_ ) _, y_ ) and _p_ ( _x_ ( _zi[′][, z][j][, z][−{][i,j][}]_[)] _[, y]_[)] would determine the largest among _p_ ( _x_ ( _zi, zj, z−{i,j}_ ) _, y_ ) and _p_ ( _x_ ( _zi, zj[′][, z][−{][i,j][}]_[)] _[, y]_[)][,] _[i.e]_[.,][the][factors] _[Z][i][,][ Z][j]_[would] not be distinct. (Technically, we can assume in our definition of “factors” that all possible rankings of values of _Zi_ are possible for any choice of _z−i_ ). Thus, _λ_ = _µ_ = 1 in equation 31 for all _zi, zi[′][, z][j][, z] j[′]_[.] This implies that _**u**_ ( _zi,z−i_ ) _−_ _**u**_ ( _zi[′][,z][−][i]_[)][ does not depend on] _[ z][−][i]_[, which in turn] means that the vectors _**u** z_ are decomposable, since _**u** z −_ _**u** z′_ does not depend on components that _z, z[′]_ have in common. 

**==> picture [176 x 12] intentionally omitted <==**

**==> picture [202 x 11] intentionally omitted <==**

_for some scalars λ, µ ∈_ R _, then either_ _**p** ,_ _**q** ,_ _**r** ,_ _**s** lie on the same affine line (_ i.e _., all pairwise differences are scalar multiples of each other) or λ_ = _µ_ = 1 _._ 

_Proof._ Substituting _**p**_ = _**q**_ + _λ_ ( _**r** −_ _**s**_ ) in the second equality in equation 32 yields 

**==> picture [197 x 11] intentionally omitted <==**

If _µ_ = 1 or _ν_ = 1, then this shows that _**p** ,_ _**r** ,_ _**s**_ are aligned (note that coefficients sum to 1). Using the relation for _**p**_ , we conclude that either _µ_ = _ν_ = 1 or all four points are aligned. 

We conclude this section by elaborating on the connection with mathematical representation theory. This discussion is not necessary for understanding the paper, but we believe that the symmetry-based viewpoint introduced in [26] is a useful framework for studying disentanglement and compositionality in machine learning. For convenience to the reader, we include here a minimal set of definitions and basic results from representation theory, focusing on the representation of finite groups. More details can be found, for example, in [21]. 

A _representation_ of a group _G_ is a homomorphism _ρ_ : _G → GL_ ( _V_ ), where _V_ is a finite-dimensional vector space (typically over the complex numbers, but we can focus on the the real setting here). Often the map _ρ_ is omitted and the representation is identified with _V_ . It also common to say that _V_ is a “ _G_ -module” or a “ _G_ -representation.” Given two _G_ -representations _V, W_ , a _homomorphism of representations_ is a linear map _φ_ : _V → W_ that is _G_ -equivariant: 

**==> picture [196 x 11] intentionally omitted <==**

A _subrepresentation_ (or _submodule_ ) of a _G_ -representation _V_ is a vector subspace _H ⊂ V_ such that is _G_ -invariant: 

**==> picture [184 x 11] intentionally omitted <==**

If _φ_ : _V → W_ is a homorphism of representations, then the kernel and image of _φ_ are subrepresentations of _V_ and _W_ , respectively. A _G_ -representation of _V_ is _irreducible_ if it has no proper subrepresentations, _i.e_ ., if its only subrepresentations are _{_ 0 _}_ and itself. 

**Example 12** (Trivial representation) **.** Let _G_ be any group and let _V_ = R be a one-dimensional vector space. Then the map _ρ_ : _G → GL_ ( _V_ ) that every element of _G_ with the identity on _V_ is an irreducible representation, called the _trivial representation_ . 

**Example 13** (Permutation representation) **.** Let _V_ = R _[n]_ and consider the representation _ρ_ : S _n → GL_ ( _V_ ) that permutes coordinates. This is not an irreducible representation since the one-dimensional subspace _V_ 0 = _⟨_ (1 _, . . . ,_ 1) _⟩_ is a subrepresentation (a “copy” of the trivial representation). In fact, we have that _V_ = _V_ 0 _⊕ V_ 1 where _V_ 1 = _{v_ : _v_ 1 + _. . ._ + _vn_ = 0 _}_ . One can show that _V_ 1 is irreducible, and it is called the _standard representation_ of S _n_ . 

The next statements imply that, for finite groups, irreducible representations can always be used as “building blocks” for describing arbitrary representations. The irreducible components of a representation are (nearly) uniquely determined; moreover, there are only finitely many irreducible representations of a group up to isomorphism. 

**Proposition 14** (Corollary 1.6, [21]) **.** _If G is a finite group, any G-representation can be decomposed as a direct sum of irreducible representations._ 

**Proposition 15** (Proposition 1.8, [21]) **.** _Let V be a G- representation, and consider its decomposition into irreducible representations:_ 

**==> picture [173 x 13] intentionally omitted <==**

_Then the spaces Vi[⊕][a][i] are uniquely determined. The irreducible representations Vi are determined up to isomoprhism._ 

**Proposition 16** (Corollary 2.18, [21]) **.** _Every finite group only has a finite set of irreducible representations, up to isomorphism._ 

For example, the irreducible representations of a symmetric group S _n_ are in one-to-one correspondence with the (unordered) partitions of _n_ elements. See [21, Chapter 4] for an explicit description. 

We now return to our factored set _Z_ = _Z_ 1 _× . . . × Zk_ . We consider the vector space _⟨Z⟩_ = _Span_ ( _**e** z_ : _z ∈Z_ ), spanned by independent basis vectors associated with elements of _Z_ . We can identify _⟨Z⟩_ with the space R _[n]_[1] _⊗. . .⊗_ R _[n][k]_ . As S _i_ -modules, R _[n][i][∼]_ = _V_ 0 _,ni ⊕ V_ 1 _,ni_ where _V_ 0 _,ni_ is a trivial representation and _V_ 1 _,ni_ is the standard representation for S _i_ . We thus have that 

**==> picture [193 x 90] intentionally omitted <==**

with _Vϵ_ := _Vϵ_ 1 _,n_ 1 _⊗ . . . ⊗ Vϵk,nk_ . This is a decomposition of _⟨Z⟩_ into irreducible _G_ -representations (see [21, Exercise 2.36]). We can describe the projection _πϵ_ onto _Vϵ_ explicitly 

**==> picture [175 x 11] intentionally omitted <==**

**==> picture [155 x 10] intentionally omitted <==**

**==> picture [177 x 44] intentionally omitted <==**

A data embedding _r_ : _Z →_ R _[d]_ can be uniquely associated with a linear map _⟨r⟩_ : _⟨Z⟩→_ R _[d]_ or can equivalently be viewed as a tensor in [ _r_ ] _∈_ R _[n]_[1] _⊗ . . . ⊗_ R _[n][k] ⊗_ R _[d]_ . The image of _⟨r⟩_ is a _G_ -module in R _[d]_ and its decomposition will contain a subset of the irreducible components in equation 37. The notion of disentangled representation given in [26] means that the only irreducible components that contribute to the image of _r_ are the representations _Vϵ_ such that _ϵi_ = 1 for at most one index _i_ . Equivalently, we 

require that the projection of the image of _r_ onto the “entangled components” is zero, _i.e_ ., _πϵ_ ( _**u** z_ ) = 0 whenever _|{i_ : _ϵi_ = 1 _}| >_ 1. An intuitive way to understand this notion is in terms of the tensor [ _r_ ] _∈_ R _[n]_[1] _⊗. . .⊗_ R _[n][k] ⊗_ R _[d]_ : we require that each of the _d_ “slices” R _[n]_[1] _⊗ . . . ⊗_ R _[n][k]_ can be obtained by summing “one-dimensional slices” of the form **1** _⊗. . .⊗_ _**u** i ⊗. . .⊗_ **1** (similar to summing vectors into a tensor by “array broadcasting”). In fact, this observation leads to the following characterization of linear factorization in terms of tensor-rank. 

**Proposition 17.** _A tensor_ [ _r_ ] _∈_ R _[n]_[1] _⊗ . . . ⊗_ R _[n][k] ⊗_ R _[d] corresponds to a decomposable representation if and only if all_ (R _[n]_[1] _⊗ . . . ⊗_ R _[n][k]_ ) _-slices of_ exp([ _r_ ]) _have tensor-rank one, where_ exp([ _r_ ]) _is obtained from_ [ _r_ ] _by exponentiating element-wise. This is true if and only if for all φ ∈_ (R _[d]_ ) _[∗]_ exp( _φ_ ([ _r_ ])) _has tensor-rank one._ 

_Proof sketch._ The first claim follows from the previous discussion and the fact that exp ([�] _i_ **[1]** _[ ⊗][. . .][ ⊗]_ _**[u]**[i][ ⊗][. . .][ ⊗]_ **[1]**[)] = exp( _**u**_ 1) _⊗ . . . ⊗_ exp( _**u** k_ ). For the second statement, we note exp( _t_ ) having rank-one is a linear condition on a tensor _t_ . 

For categorical distributions of multiple variables, the distribution tensor having rank equal to one corresponds to statistical independence of variables, so the result above can be seen as an algebraic reformulation of Proposition 7 in the main body of the paper. We also note that that other probabilistic conditions could be considered by allowing for more irreducible components in from equation 37 to appear in the image of _r_ . This is similar to the log-linear representations of multivariate data. In fact, it is possible to express any conditional independence assumption on _p_ ( _Z|Y_ ) in terms of linear-algebraic conditions on the data representation _r_ . 

## **B. Experimental Details** 

**Datasets.** The MIT-states dataset [28] contains images of 245 objects modified by 115 adjectives, for a total of 28175 classes. The test set has size 12995. The UTZappos dataset [50] contains images of 12 shoe types with 16 fine-grained states. The test set has size 2914. Note that in both of these datasets only a small portion of all possible attribute-object pairs actually occurs in the test set. However, in our experiments we assume that we do not have access to this information. We also mention that prior works that have used these datasets such as [35, 38] have differentiatied between the performance on label pairs that were seen in training and those that were not. Since this distinction is not relevant in a zero-shot setting, we simply report accuracy on objects, attributes, and attributeobject pairs. In the Waterbird dataset [44] labels are “waterbird/landbird” and spurious attributes are “water background/land background.” There are 5794 test samples di- 

|**Class Prompts**|**Class Prompts**|
|---|---|
|This is a picture of a landbird.||
|This is a picture|of a waterbird.|
|**Spurious**|**Prompts**|
|This is a land background.|This is a picture of a forest.|
|This is a picture of a moutain.|This is a picture of a wood.|
|This is a water background.|This is a picture of an ocean.|
|This is a picture of a beach.|This is a picture of a port.|



Table 4: Prompts for Waterbird dataset [44] from [11]. 

vided in four unbalanced groups. On the CelebA [33], labels are “not blond/blond” and spurious attributes are “male/female.” There are a total 19962 test samples with unbalanced groups. The DeepFashion2 dataset [23] with the captions provided in PerVL [14] contains 1700 images from 100 unique fashion items. Following [14] val/test splitting, we retrieve 50 of these concepts selected for testing. We use 5 randomly chosen images per fashion item as per-concept supporting images, and use a test set with 221 images containing all 50 concepts and their captions (see [14] for more details). Final results are obtained by averaging the Mean Reciprocal Rank metric over 5 random seeds. 

**Prompts.** For MIT-States and UTZappos, we use the prompt “image of a [ _a_ ][ _o_ ],” “image of a [ _a_ ] object,” and “image of a [ _o_ ],” as explained in the main body of the paper. Here [ _a_ ] and [ _o_ ] are the lower-case original class labels.[3] For our experiments on debiasing on the Waterbirds and CelebA datasets we use the same prompts and spurious attributes used in [11]. These are shown in Tables 4 and 5. To compute debiased prompts we simply prepend all spurious prompts to each class prompts and then average the spurious prompts to obtain debiased class prompts (note that spurious prompts are “balanced” in their bias); this simpler but conceptually similar to the “Orth-Proj” approach used in in [11] that computes an orthogonal projection in the orthogonal complement of the linear space spanned by the spurious prompts. We do not make use of the “positive pairs” of prompts that are used in that work for regularization of the projection map. 

## **C. Additional Results and Discussions** 

**Quantifying compositionality.** Given a set of vectors _**u** z, z ∈Z_ in R _[d]_ , we can measure how close the vectors 

> 3In the case of objects for UTZappos, we perform a simple split ‘Boots.Mid-Calf’ _→_ “boots mid-calf” 

|**Class Prompts**|
|---|
|A photo of a celebrity with dark hair.|
|A photo of a celebrity with blond hair.|
|**Spurious Prompts**|
|A photo of a male.<br>A photo of a male celebrity.|
|A photo of a man.<br>A photo of a female.|
|A photo of a female celebrity.<br>A photo of a woman.|



Table 5: Prompts for CelebA dataset [33] from [11]. 

||**IW**|**RW**|**Avg**|
|---|---|---|---|
|MIT-States [28]|0.23_±_0.05|0.43_±_0.06|0.78_±_0.13|
|UT Zappos [50]|0.16_±_0.04|0.51_±_0.05|0.58_±_0.18|



Table 6: Quantifying compositionality using a trained encoder. 

.5cm 

||**IW**|**RW**|**Avg**|
|---|---|---|---|
|MIT-States [28]|0.04_±_0.02|0.16_±_0.02|0.10_±_0.03|
|UT Zappos [50]|0.10_±_0.02|0.22_±_0.04|0.14_±_0.05|



Table 7: Quantifying compositionality using a randomly initialized encoder. 

are to being decomposable by using 

**==> picture [217 x 43] intentionally omitted <==**

The optimal vectors _**u**_ ˜ _z_ here are the ideal word approximations given by Proposition 4. In Table 6, we report this quantity for embeddings of objects-attributes in the datasets MIT-States [28] and UT Zappos [50] (IW column). For comparison, we also include the average squared distance between the original embeddings and the average of the individual object and attribute embddings based on “real words” (RW column), and the average squared distance between pairs of the original embedding vectors (Avg). In Table 7, we report the same quantities but using embeddings obtained from a _randomly initialized_ encoder. These results suggest that embeddings at initialization are already compositional. We discuss this point further in the next paragraph. 

**Visualized embeddings.** We present more examples of projected embeddings of composite strings. In Figure 4, we consider again the four manually constructed examples from Figure 2 in the main body of the paper: “a photo of a _{_ red, blue, pink _} × {_ car, house _}_ ”; “a photo of a _{_ big, 

small _} × {_ cat, dog _} × {_ eating, drinking _}_ ”; “ _{_ a photo of a, a picture of a _} × {_ place, object, person _}_ ”; “king, queen, man, woman, boy, girl.” The top row of Figure 4 is the same as the top row from Figure 2. In the bottom row of 4, we visualize the embeddings of the same strings using a randomly initialized text encoder. In the first three examples, the factored structure is also _syntactic_ , _i.e_ ., it is based on the string structure. In these cases, the embeddings remain roughly decomposable even with random encoder. In the last case, however, decomposable structures are not visible anymore, since the strings in this example contain no repeated substrings. Note also that in third case, the factor corresponding to _{_ a photo of a, a picture of a _}_ is no longer “squashed” since these two strings not considered similar by the randomly initialized encoder. 

We show other examples of this effect in Figure 5. Here each pair of plots shows projections of the same strings using a trained encoder (left figure) and a randomly initialized encoder (right figure). As one might expect, for strings corresponding to capital-country relation (first row), the approximate symmetries that can be seen in the embbedings from the trained encoder are no longer present when using the random encoder. The strings in the second row, however, have a synctatic factored structure. In this case, we visually observe strong symmetries in the embeddings from the trained encoder as well as from the random encoder. 

In Figure 6, we consider 2D projections of embeddings of factored strings that include idioms such as “cold shoulder,” “big apple”, “black friday,” “hot pepper.” We compare these embeddings with those of similar factored strings in which meanings of words are more conventional and uniform. In both cases, we quantify the amount of linear compositionality both visually and using the squared residual as in equation 40. The results confirm the natural intuition that linear compositionality is measurably weaker when strong contextual effects between words are present. 

**Other notions of probabilistic disentanglement.** Proposition 7 shows that linear factorization of embeddings corresponds to conditional independence of factors _zi_ given the image _y_ . One might also consider a different sort of probabilistic disentanglement in which conditionals are reversed: 

**==> picture [228 x 11] intentionally omitted <==**

This can be viewed as a sort of “causal disentanglement” (similar to the notion used in [48]). It follows from Corollary 8 that decomposable embeddings mean that 

**==> picture [225 x 34] intentionally omitted <==**

**==> picture [233 x 130] intentionally omitted <==**

**----- Start of picture text -----**<br>
0.2<br>0.2 0.00.2<br>0.00.2 0.20.0 0.2 0.20.0 0.2 0.20.00.2 0.2 0.0 0.2 0.20.00.2 0.20.00.2 0.2 0.0 0.2 0.20.00.2 0.20.00.2 0.2 0.0 0.2<br>0.10 0.2<br>0.050.00 0.050.00 0.05 0.10.0<br>0.05 0.05 0.00 0.1<br>0.100.050.000.050.10 0.100.050.000.050.10 0.100.050.000.050.10 0.100.050.000.050.10 0.05 0.050.000.050.10 0.10.00.1 0.20.10.00.10.2 0.20.10.0 0.1<br>**----- End of picture text -----**<br>


Figure 4: Projected embeddings of manually constructed strings associated with factored concepts, as described in Section 5 in the main body of the paper. _Top:_ trained encoder (same as in Figure 2). _Bottom:_ visualization of the embeddings for the same strings using a randomly initialized encoder. Even without semantic information, the embeddings in the first three examples are still roughly decomposable. 

**==> picture [235 x 131] intentionally omitted <==**

**----- Start of picture text -----**<br>
rome, paris, berlin; italy, france, germany nairobi, baku, ankara; kenya, azerbaijan, turkey<br>0.20.10.00.10.2 0.150.100.050.000.050.100.15 0.40.30.20.10.00.10.2 0.200.150.100.050.000.050.10<br>0.30.20.10.00.10.20.3 0.40.30.20.10.0 0.10.20.3 0.150.100.050.000.050.100.150.20 0.2 0.1 0.0 0.1 0.4 0.20.0 0.2 0.4 0.4 0.2 0.0 0.2 0.100.050.000.050.100.150.200.25 0.2 0.1 0.0 0.1 0.2<br>rome under the rain, paris under the rain, berlin under the rain; a person in rome, a person in paris, a person in berlin;<br>rome under the moon, paris under the moon, berlin under the moon a building in rome, a building in paris, a building in berlin<br>0.30.20.10.00.1 0.100.050.000.050.10 0.20.10.00.1 0.100.050.000.05<br>0.30.20.10.00.10.20.3 0.4 0.2 0.0 0.2 0.4 0.1 0.0 0.1 0.2 0.2 0.1 0.0 0.1 0.2 0.2 0.10.0 0.1 0.2 0.3 0.30.20.10.0 0.10.20.3 0.100.050.000.050.10 0.150.100.050.000.050.100.15<br>**----- End of picture text -----**<br>


Figure 5: Comparison between projected embeddings using a trained encoder (left figure in each pair) and using a randomly encoder (right figure in each pair). Both encoders lead to symmetric structures when the strings have a factored syntax (bottom row), while only the trained encoder shows these approximate structures when the factorization is semantic (top row). 

Thus, conditional independence has the same form as equation 41 up to the factor _[p] p_[(] ( _[z] z_[1] 1[)] _,...,z[...][p]_[(] _[z] k[k]_ )[)][(pointwise mutual infor-] mation) that does not depend on _y_ . If factors are globally independent, then equation 42 and equation 41 are equivalent. It is also worth noting that equation 41 does not determine the marginal distribution _p_ ( _z_ = ( _z_ 1 _, . . . , zk_ )). In general, linear factorization of the embeddings can be seen as a relaxed version of causal disentanglement. 

**Normalization.** Embedding vectors for CLIP are typically normalized, however ideal word vectors are _never_ normalized. While this may appear strange, we note that 

**==> picture [236 x 123] intentionally omitted <==**

**----- Start of picture text -----**<br>
[['warm'0.40.30.20.10.00.10.20.3 , 'cold'0.2 D=0.169],0.0['shoulder'0.2 0.4, 'tea']] [['warm'0.30.20.10.00.10.20.3 , 'cold'0.2 D=0.086],0.0['shower'0.2 , 'tea']] [['bi0.40.30.20.10.00.10.20.30.4 g', 'small'0.2 D=0.158], 0.0['apple'0.2 , 'oran0.4 ge']] [['big'0.30.20.10.00.10.20.3 , 'small'0.2 D=0.102], ['banana'0.0 0.2, 'orange']]<br>[['black'0.50.40.30.2 , 'green'D=0.225], ['friday', 'day']] [['black'0.30.20.1 , 'green'D=0.053], ['car', 'bike']] [['hot'0.30.20.1 , 'cold'D=0.107], ['pepper', 'pizza']] [['hot'0.20.1 , 'cold'D=0.058], ['pasta', 'pizza']]<br>0.1 0.0 0.0 0.0<br>0.00.10.20.3 0.10.20.3 0.10.20.3 0.10.2<br>0.2 0.0 0.2 0.4 0.2 0.0 0.2 0.3 0.2 0.1 0.0 0.1 0.2 0.3 0.2 0.1 0.0 0.1 0.2<br>**----- End of picture text -----**<br>


Figure 6: Comparison between projected embeddings for factored strings with and without idioms that have noncompositional meaning (left and right in the subfigures, respectively). We can qualitatively and quantitatively see that idioms lead to weaker compositionality. 

the norm of the embeddings does not carry a probabilistic meaning: we can replace the embeddings _**u** ,_ _**v**_ from the two modalities with _**T u**_ and _**T**[−]_[1] _**v**_ for any invertible linear transformation _**T**_ of R _[d]_ without changing the probability model on _X × Y_ . In general, ideal word manipulations require starting from normalized embeddings for consistency between modalities, but then normalization is never applied again (in fact, the inner product structure on the embedding space is not used). This explains our modification to the AvgIm+Text approach in Section 5 in the paper. 

**Visualizations using SD.** We present a few additional visualizations of ideal words using Stable Diffusion. In Figure 7, we consider the same ideal word approximation as in Figure 3 in the main body of the paper and observe the effect of scaling the ideal word corresponding to “green.” That is, we consider _**u**_ 0 + _**u**_ house + _γ ·_ _**u**_ green for different _γ_ . In the top row, we compute _**u**_ green using the standard “balanced” computation for ideal words (uniform _αi_ in Proposition 4). In the bottom row, we use weights _α_ house = 1 and _α_ obj = 0 otherwise. This implies that the IW corresponding to _**u**_ green is determined by how “green” composes with “house.” Amplifying _**u**_ green now increases the “greenhouse-ness” of the generated image. 

In Figure 8, we consider the problem of _transferring_ ideal words. That is, we consider a different ( _i.e_ ., totally disjoint) set of objects and colors compared to the ones used for Figure 3 in the paper and compute the corresponding ideal words, that we write as _**u**_ color _′_ object _′ ≈_ _**u**[′]_ 0[+] _**[u]**[′]_ color _[′]_[+] _**[u]**[′]_ obj _[′]_[. We then investigate whether families of] ideal words computed independently can be “mixed,” combining ideal words for colors from the first collection and ideal words for objects from the second one, and vice-versa. Figure 8 shows that this is possible, at least in our restricted setting. In the first row, we show examples of four new ob- 

**==> picture [6 x 103] intentionally omitted <==**

**----- Start of picture text -----**<br>
IW green scale<br>IW green scale<br>**----- End of picture text -----**<br>


Figure 7: Scaling the ideal word _**u**_ green a by factor _γ_ = _._ 5 _,_ 1 _,_ 1 _._ 5 _,_ 2, respectively. _Top:_ _**u**_ green is computed using all objects as contexts. _Bottom:_ _**u**_ green is computed only “house” as context. 

jects with different colors computed by adding associated ideal words ( _{_ white, pink, orange, black _} × {_ chair, wallet, shirt, pen _}_ ). In the next two rows, we use the ideal words for objects with the ideal words for colors obtained previously; in the last two rows, we use the ideal words for the new colors together with the ideal words for the objects obtained previously. To obtain all of these images, we simply used _**u**_ color _′_ object _≈_ ( _**u**_ 0 + _**u**[′]_ 0[)] _[/]_[2 +] _**[ u]**[′]_ color _[′]_[+ 2] _[ ·]_ _**[ u]**_[obj][(we] found that amplifying the ideal words for objects helps ensure that objects are more centered). Analyzing the limits of this sort of transferability is left for future work. 

Finally, in Figure 9 we generate images with ideal words while also using a third “context” factor, in addition to the ones corresponding to color and object (for those we use the same colors and objects as in Figure 3). Here we see that linear compositionality is effective using simple contexts such as _{_ on the beach, on a street _}_ (first two rows), however using more complex contexts such as _{_ underwater, in a volcano _}_ (third and fourth row) it fails to produce good results. 

**==> picture [5 x 230] intentionally omitted <==**

**----- Start of picture text -----**<br>
IW<br>IW transfer<br>IW transfer<br>IW transfer<br>IW transfer<br>**----- End of picture text -----**<br>


Figure 8: Transferring ideal words. _Top row:_ Images generated ideal words for a different set of colors and objects compared to the ones used Figure 3. _Second and third rows:_ images generated by adding new ideal words for objects with the previous ideal words for colors; _Fourth and fifth rows:_ images generated by adding new ideal words for colors with the previous ideal words for objects. 

**==> picture [5 x 167] intentionally omitted <==**

**----- Start of picture text -----**<br>
IW<br>IW<br>IW<br>IW<br>**----- End of picture text -----**<br>


Figure 9: Images generated using ideal words with t8ree factors: color, object, context. _First two rows:_ using context factor _{_ on the beach, on a street _}_ ; _Second two rows:_ using context factor _{_ underwater, in a volcano _}_ . 

