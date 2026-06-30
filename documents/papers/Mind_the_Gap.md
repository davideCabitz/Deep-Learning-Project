# **Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning** 

**Weixin Liang** _[∗]_ **Yuhui Zhang** _[∗]_ **Yongchan Kwon** _[∗]_ Stanford University Stanford University Columbia University `wxliang@stanford.edu yuhuiz@stanford.edu yk3012@columbia.edu` 

**Serena Yeung** Stanford University `syyeung@stanford.edu` 

**James Zou** Stanford University `jamesz@stanford.edu` 

## **Abstract** 

We present _modality gap_ , an intriguing geometric phenomenon of the representation space of multi-modal models. Specifically, we show that different data modalities (e.g. images and texts) are embedded at arm’s length in their shared representation in multi-modal models such as CLIP. Our systematic analysis demonstrates that this gap is caused by a combination of model initialization and contrastive learning optimization. In model initialization, we show empirically and theoretically that the representation of a common deep neural network is restricted to a narrow cone. As a consequence, in a multi-modal model with two encoders, the representations of the two modalities are clearly apart when the model is initialized. During optimization, contrastive learning keeps the different modalities separated by a certain distance, which is influenced by the temperature parameter in the loss function. Our experiments further demonstrate that varying the modality gap distance has a significant impact in improving the model’s downstream zeroshot classification performance and fairness. Our code and data are available at `https://modalitygap.readthedocs.io/` 

## **1 Introduction** 

Multi-modal models map inputs from different data modalities (e.g. image and text) into a shared representation space (Figure 1 (a)). It has garnered tremendous interest and excitement as a framework for data integration. As a prominent example pre-trained on a web-scale collection of images and natural language, OpenAI’s CLIP model [39], has learned diverse visual concepts that can readily be transferred to downstream tasks through _prompting_ : one can perform “zero-shot” visual classification by simply providing the names of the visual categories to be recognized. 

In this work, we present the _modality gap_ phenomenon: As shown in Figure 1 (b), CLIP’s image embeddings and text embeddings are located in two completely separate regions of the embedding space. We find this phenomenon consistently across various multi-modal models, covering texts, natural images [39], videos [50], medical images [53], and amino-acid sequences [11]. Interestingly, this phenomenon still holds even when we embed using multi-modal models with _random_ weights (Figure 1 (c)). While it might seem reasonable to attribute the gap to differences in data distributions or to the different encoder architectures, we showed that these factors are not the fundamental cause. 

This paper provides a three-part explanation for the modality gap phenomenon. **(1)** The general inductive bias of deep neural architecture creates a _cone effect_ : The effective embedding space is 

> _∗_ These three authors contributed equally. 

36th Conference on Neural Information Processing Systems (NeurIPS 2022). 

**==> picture [350 x 217] intentionally omitted <==**

**----- Start of picture text -----**<br>
Multi-modal Contrastive Learning(a)  123 ModalityEnc1 1 Contrastive LossMulti-modal 312 312 ModalityEnc2 2 123 ( xk 2  Modal xy kk = Normalize(Enc= Normalize(Enc1 , yk 2  Modal si,j =2  x 12)(  ⇠ ( ix ·ykk  y D )))) j LLMM 12 !M!M 21 L == =  − − [1] 2 NN [1][1][(] [L] XX [M] ii [1] loglog [!M] PP [2] exex [+] jj [exp(][exp(] [ L] p(p( [M] ssii,, [s] [2] [s] ii [!M][i,j][j,i] //⌧⌧ [/⌧][/⌧] )) [1][)][)][)]<br>- CLIP  nN VideoCLIP  nN ConVIRT  nN CLASP<br>Natural Image - Text = ik Natural Video - Text = Medical Image - Text B & Amino-acid Sequence - Text<br>(b)  86 . M4b10 Z 1210 ————_,24LL aad v10 WIAX{E}N‘<br>Initialization: Pre-trained 4 8 O/, Y-yy i ——===SSzsA 8 © \S<br>2 6 tt 4 ° ae 6 = 5<br>Z — = @ =<<br>4 <A zz 1? —<br>0 2 2 —<——Z es =<br>L. 0 10 20 0 10 20 0 P 10 20 5y 0 5 lo 15<br>UMAP 1<br>& 12<br>10 Ss 10 \ oS 10 tt<br>(c)  a SS .8SN 8 gg =>——* 8 7 Z<br>Initialization: Random 64 ®42 \SS‘ SS «54 £Aas M e DZtyF QjalY th -2420 7 4<br>9 5 10 15 20 10 20 0 5 10 15 -10 0 10<br>UMAP 1<br>UMAP 2<br>UMAP 2<br>**----- End of picture text -----**<br>


Figure 1: **The pervasive** _**modality gap**_ **in multi-modal contrastive representation learning. (a) Overview of multi-modal contrastive learning.** Paired inputs from two modalities (e.g., image-caption) are sampled from the dataset and embedded into the hypersphere using two different encoders. The loss function is to maximize the cosine similarity between matched pairs given all the pairs within the same batch. **(b) UMAP visualization of generated embeddings from pre-trained models.** Paired inputs are fed into the pre-trained models and the embeddings are visualized in 2D using UMAP (lines indicate pairs). We observe a clear modality gap for various models trained on different modalities. **(c) UMAP visualization of generated embeddings from same architectures with random weights.** Modality gap exists in the initialization stage without any training. 

restricted to a narrow cone for pre-trained models or models with random weights. **(2)** Different random initializations create different embedding cones. Since a multi-modal model consists of two encoders, which create different cones at random initialization, this explains how the modality gap is present at initialization. **(3)** The contrastive learning objective commonly used by multi-modal models preserves the gap. We support our explanations with theory and experiments. Our theoretical analysis shows that under mild assumptions, each neural network layer shrinks the angle between any pair of embedding vectors with high probability, thereby creating more narrow cones in deeper architectures. We further prove that different random initializations of model weights result in different cones. Interestingly, increasing the modality gap in models like CLIP can improve its downstream performance on several zero-shot learning and fairness tasks. The main objective of our paper is to i) empirically demonstrate the modality gap phenomenon across different data modalities and NN architectures; ii) explain how the gap arises and iii) show that the size of the gap can affect downstream applications. It is _not_ our goal to propose a method to close the gap, since it’s not clear that it’s desirable to have no modality gap. Together, this paper makes the **following contributions** : 

1. To the best of our knowledge, we demonstrate a general _modality gap_ phenomenon for the first time. We show that this phenomenon holds across a wide spectrum of multi-modal models, covering texts, natural images, videos, medical images, and amino-acid sequences. 

2. We demonstrate the significant implications of modifying the gap in downstream applications. By simply modifying the gap’s distance, we can improve CLIP’s zero-shot performance and fairness. 

3. To explain modality gap, we provide a three-part explanation supported by extensive theoretical and empirical analyses. Our analyses also provide new insights on the cone effect, which we show is a general phenomenon for deep neural networks. Existing work focuses on _trained_ language models and attributes the cone effect to the _optimization_ under unbalanced word frequencies distribution. We demonstrate that this effect holds not only across various modalities and network architectures, but also on random noise inputs and random weights, which is not captured in previous work. 

2 

**==> picture [379 x 255] intentionally omitted <==**

**----- Start of picture text -----**<br>
Vision  Text<br>ResNet<br>Transformer Transformer<br>Initialization: Pretrained 80000 A 80000 A 80000 A > 1.0 0-0-0: 0-0-0-0'0:5:5:6'0: 0-0 0-0-0-0-0-0-0-0-0<br>Input:  000040000 Avg=0.56 so00040000 Avg=0.47 so00040000 Avg=0.51 25 e/ aeo —— n*(Linear+Sigmoid)ave<br>Real Data so000 A so000 A 0000 ine = 0.8 —— n*(Linear+LeakyReLU)<br>Initialization: Random 1250002000002st 100 125000 eae 10000000000 ae WwW4 0.6 Jf22 —— n*(Linear+ t(j4 Tanh)ReLU)<br>Real DataInput:  730005000025000 Avg=0.99 100000mono50000 Avg=0.72 enone40000 Avg=0.67 fe)vo”S94 e/ —— n*(Linear)vaPad.e-e®<br>© ose oa 00 25000«8 20000 Ah, > “<br>100000 ge 0s 100s 050 07s 100 OL ro<br>Initialization:  80000 v Pd<br>Random 60000 A 6000080000 A 6000080000 A <7 0.0s | ¢-0:6:6:8:0:0-0-0-0-0:0:0-0-0:0-0:0:0-0-0-0:0:0'0<br>Avg=0.999 Avg=0.94 Avg=0.41<br>Input:  “0000 fone “one 0 5 10 15 20<br>Random Noise 20000 20000 20000<br>Q0.9985 0.9990A0.9995 ° 0.900Bilin0.925 0.950 5 0.00dill,0.25 0.50 0.75 Number of layers (n)<br>(a) The cosine similarity between all pairs of embeddings (b) Effects of nonlinear activation and depth<br>20 20 Vision  . 20 Text<br>ResNet<br>. | 3 . Transformer aus . Transformer eg in<br>no @ «¢ : ad ” ae i ed4 whee ga «oie<br>5 OB en 8 Sa y,, aie a yp ie<br>gilt é eh, al ; . ° , ks ee, <8 Se. 5 "Sie. a ee , ad<br>~ e py & w | , <a & ee de he  »<br>seth SEF se -s<br>-10 age Sten -10 ee i bie ae<br>-10 5 oO 5 10 15 -10 5 ° 5 10 15 20 10 5 o 5 10 15<br>(c) UMAP visualization of embeddings of 25 randomly initialized models on  real data  (color indicates random seed)<br>**----- End of picture text -----**<br>


Figure 2: **The cone effect phenomenon. (a) Histograms of the cosine similarity between all pairs of embeddings across various settings.** The average cosine similarity is substantially larger than 0, indicating that the embedding space is a narrow cone. The cone effect also holds on randomly initialized models, and on random noise inputs. **(b) Effects of nonlinear activation and depth.** Inputs are 512-dim standard normal random vector. All MLP linear layers are 512 _×_ 512, with both weight and bias randomly initialized from _N_ (0 _,_ 5121[)][.][Y axis is the average cosine similarity between pairs of embeddings.] **[(c) UMAP visualization of] embeddings of 25 randomly initialized models (without training) on** _**real**_ **data.** Each random initialization forms a distinctively different cone. _Real Data:_ 5,000 image-caption pairs from the validation set of MSCOCO Caption. _Random Noise:_ Gaussian noise from the standard normal distribution as images, uniformly random integer sequences as texts. 

4. We mathematically characterize the contraction mapping induced by linear layers with ReLU non-linearities to explain the cone effect. Our theory matches well with experiments and provides insights for understanding the general inductive biases of deep neural networks. 

## **2 The Cone Effect Induces A Modality Gap** 

## **2.1 The Narrow Cone of Embeddings** 

In order for modality gap to exist, the embeddings from a encoder should be concentrated around a subregion of the full embedding space—otherwise, the embeddings from different encoders would overlap. Motivated by this, we begin our investigation by showing that the modality gap already arises at random model initialization due to the _cone effect_ : The effective embedding space is restricted to a narrow cone for trained models and models with random weights. To demonstrate this, we extract 5,000 embeddings from the final layer of 3 pre-trained models respectively (ResNet, Vision Transformer, Text Transformer)[2] on MSCOCO Caption [8]. We then compute the cosine similarity between all possible pairs of the 5,000 embeddings within each model (Figure 2 (a)). We found that both the average cosine similarity (0 _._ 56, 0 _._ 47, 0 _._ 51 respectively for the 3 models) and the minimum cosine similarity (0 _._ 23, 0 _._ 05, 0 _._ 01) are positive. These results indicate that the embedding space is a narrow cone. 

> 2ResNet embeddings are extracted before the final linear layer. We use ResNet-18 pre-trained on ImageNet, Vision Transformer and Text Transformer from pre-trained CLIP 

3 

In the literature, the cone effect has been observed in the language representations from language models (e.g., BERT) [12]. A common explanation is that the _unbalanced_ distribution of word frequencies biased the _optimization_ [15, 33]. However, we found that the cone effect still exists in models with random weights (Figure 2 (c)). In fact, the average cosine similarity there is even _higher_ than in trained models. For example, any two embeddings from a randomly initialized ResNet have on average an almost perfect (0 _._ 99) cosine similarity. Interestingly, the cone effect still holds when the input data is random noise[3] , indicating that unbalanced data distribution suggested in previous works is not necessary for the cone effect. Together these experiments suggest that the cone effect reflects a more general inductive bias of deep networks than might be previously appreciated. 

> **How narrow is the cone in 512-dim representation space?** We clarify that a cosine similarity with 0 _._ 56 already indicates that the embedding space is actually an extremely narrow cone in the 512-dimensional feature space. Consider the fraction of surface area in a unit hypersphere: In 2D, arccos(0.56)=55.94°, indicating that a cosine similarity of 0.56 can “occupy” 55.94°/360°=15.53% 

> [55] _[.]_ 2[94][°] ) of the 2D unit circle. In 3D, a cosine similarity of 0.56 can “occupy”[2] _[πr]_[2][(][1] _[−]_ 4 _πr_[cos][2] =3.34% of the 3D unit sphere. In 512D, a cosine similarity of 0.56 can “occupy” less than 2[512] 1[fraction of the] surface area in a unit 512D hypersphere. These evidences show that the effective embedding space is restricted to an extremely narrow cone. 

## **2.2 The effects of non-linear activation on cone effect** 

**Design** To study the effects of non-linear activation functions on the cone effect, we randomly initialized various MLPs with different non-linearities or without non-linearities. The inputs of the MLPs are 512-dim standard normal random vectors. All MLP linear layers are 512 _×_ 512, with both weight and bias randomly initialized from _N_ (0 _,_ 5121[)][, here we denote a Gaussian distribution with] mean _µ_ and variance _σ_[2] by _N_ ( _µ, σ_[2] ). 

**Results** As shown in Figure 2 (b), MLPs without non-linear activation shows little cone effect. However, with non-linearity, the average cosine similarity increases _rapidly_ as the number of layers increases. For example, the average cosine similarity reaches 0 _._ 99 for a 2-layer MLP with Sigmoid. These results indicate that the non-linear activation functions play a crucial role in the cone effect. 

Although it is easy to see that ReLU makes every coordinate non-negative, and thus cosine similarity after ReLU is guaranteed to be non-negative, we highlight that none of the 3 models in Figure 2 (a) has ReLU as the final layer before embedding extraction[4] . In addition, although all 3 models incorporate normalization layers such as batch norm [23] and layer norm [4] in their architectures, we still observe the cone effect. Further analyzing the connection between normalization and the cone effect is an interesting direction of future work. 

## **2.3 Different random initializations create different cones** 

Next, we study the effect of different random initialization on the cone effect. In Figure 2 (c), we randomly initialized a model 25 times, and plotted its extracted embeddings on the same _real data_ (i.e., MSCOCO Caption) via UMAP visualization [41]. We found that each random initialization forms a distinctively different cone. This phenomenon holds across various neural network architectures and input modalities (ResNet, Vision Transformer or Text Transformer), on ImageNet-pretrained models (Supp. Figure 13), on PCA visualization (Supp. Figure 7), or with random noise inputs (Supp. Figure 5). Since a multi-modal model consists of two encoders, which creates different cones at random initialization, this explains how the modality gap is present at initialization. _While it might seem reasonable to attribute the modality gap to differences in data modalities [21], Figure 2 (c) shows the gap still exists even if the two encoders operate on the exact same data in the exact same modality. Therefore, the gap can exist without different modalities, and we emphasize that the modality gap phenomenon is non-trivial to understand._ 

> 3Standard normal distribution for vision models, and uniformly random integer sequences for text models. 

> 4The last 3 layers are Conv2d, BatchNorm2d, AdaptiveAvgPool2d for ResNet-18 (not counting last fc); Linear, LayerNorm, LayerNorm for Vision Transformer in CLIP; QuickGELU, Linear, LayerNorm for Text Transformer in CLIP. 

4 

## **3 Theoretical analysis** 

Here, we theoretically investigate the cone effect phenomenon. We show that (i) the cosine similarity increases as the layer gets deeper and (ii) the variance of an intermediate output mostly comes from the model’s random initialization. 

We first define some notations. We denote the ReLU activation by _φ_ ( _x_ ) := max( _x,_ 0) for _x ∈_ R, and we extend it by considering element-wise operation _φ_ ( **x** ) := ( _φ_ ( _x_ 1) _, . . . , φ_ ( _xk_ )) _[T]_ = (max( _x_ 1 _,_ 0) _, . . . ,_ max( _xk,_ 0)) _[T]_ for a multivariate input **x** = ( _x_ 1 _, . . . , xk_ ) _[T] ∈_ R _[k]_ and _k ∈_ N. The cosine similarity between two vectors _u, v ∈_ R _[k]_ is defined as cos( _u, v_ ) := _∥uu∥∥[T] vv∥_[where] _∥u∥_ = ( _u[T] u_ )[1] _[/]_[2] . Lastly, we set [ _k_ ] := _{_ 1 _, . . . , k}_ for _k ∈_ N. 

**Each network layer increases cosine similarity.** We study how the cosine similarity between two intermediate layer outputs changes when weight and bias terms in an MLP are fixed. The following theorem shows that with a high probability cosine similarity increases after one feedforward computation when the number of nodes in the output layer is large. 

**Theorem 1** (Monotonicity of cosine similarity) **.** _Suppose u, v ∈_ R _[d]_[in] _are any two fixed vectors such that ∥u∥_ = _r∥v∥ for some r >_ 0 _,_ **W** _∈_ R _[d]_[out] _[×][d]_[in] _is a random weight matrix where each element_ **W** _k,l ∼N_ (0 _, d[−]_ out[1][)] _[ for][ k][∈]_[[] _[d]_[out][]] _[,][ l][∈]_[[] _[d]_[in][]] _[, and]_ **[ b]** _[∈]_[R] _[d]_[out] _[is a random bias vector such]_ 1 _that_ **b** _k ∼N_ (0 _, d[−]_ out[1][)] _[ for][ k][∈]_[[] _[d]_[out][]] _[.][If]_[ cos(] _[u, v]_[)] _[ <]_ � 2 � _r_ +[1] _r_ �[�] _[−]_[1] _, then the following holds with probability at least_ 1 _− O_ (1 _/d_ out) _._ 

**==> picture [178 x 11] intentionally omitted <==**

Theorem 1 shows that the cosine similarity between two vectors increases with a high probability after one feedforward computation consisting of a linear transformation and ReLU computation. This matches well with the result in Figure 2 (b) where the cosine similarity between samples increases as the intermediate layer gets farther from the input. 

The bound condition on cos( _u, v_ ) in Theorem 1 asks that the two vectors before the layer computation are not too close to each other in terms of the direction. This is because the random bias addition can slightly change the angle between the two vectors, leading to a small decrease in cosine similarity when the previous layer’s cosine similarity is too high. This condition is plausible in practice because the _ℓ_[2] -norm of intermediate layer outputs is close to one with a high probability when the _ℓ_[2] -norm of input data is one [1, Lemma 7.1]. Given that the norm ratio _r_ is close to one, the upper bound condition for cos( _u, v_ ) is likely to hold because ([1] 2[(] _[r]_[ +][1] _r_[))] _[−]_[1][ is close to 1.] 

**Effect of random initialization** We now examine the variance of an intermediate output and explain that the variance is mainly due to random initializations as in Figure 2 (c). To be more specific, we denote an intermediate layer output by _h_ Θ( _U_ ) _∈_ R for some input datum _U_ . Here, Θ denotes all the random weights and biases that are used in _h_ Θ( _U_ ). The variance of _h_ Θ( _U_ ) can be decomposed as 

**==> picture [240 x 24] intentionally omitted <==**

Here, the inner and outer expectations are over the data _U_ and the random weights Θ, respectively. The first term on the right hand side explains the within variance after fixing one random initialization, quantifying the randomness of data. In contrast, the second term explains the variance due to different random initializations. The following theorem considers the ratio of the second term to the total variance and shows that the ratio can be very close to one when a deep neural network model is used. 

**Theorem 2** (Informal; Variance due to different random initializations) **.** _Let h_ Θ( _U_ ) _be an intermediate layer output with an input data U with∥U ∥_ = 1 _. Under mild assumptions on_ Θ _, the set of all the random weights and biases, the following inequality holds._ 

**==> picture [101 x 25] intentionally omitted <==**

_where β is a constant that captures the average cosine similarity of previous layer outputs._ 

5 

Theorem 2 shows that the ratio of the variance due to different random initializations to the total variance is bounded below by the average cosine similarity of previous layer outputs. As Figure 2 (b) illustrated, the average cosine similarity of an intermediate layer output often approaches to one as the layer gets deeper. Accordingly, the lower bound _β_ , which captures the average cosine similarity, is close to one when a neural network is deep enough. In Appendix D, we elaborate on the relationship between _β_ and the cosine similarity, providing a detailed statement of the Theorem. 

**==> picture [353 x 177] intentionally omitted <==**

**----- Start of picture text -----**<br>
Embedding Shift Experiment Simulating Mismatched Data<br>© Image W i 3 Zz i LZ<br>HS oe Text i 2 i ws =: Ieee 8.9)<br>«ss eae Vf ri ae eo<br>ORR By ' i t ¥ 4 vay 0: .<br>-04 -02 0.0 02 04 Euclidean distance SWies (f) Spherical<br>(a) Shifting embeddings  (b) Temperature=1/100 (e) Simulation Setup coordinate system<br>j 9 3.90 i / 2.00<br>3 ' § 2 a. ! 1.75 — 1/1<br>3 ! eal — 3.85 % H i ey<br>' S H 91.50 —<br>3.80 g<br>I s” ! 71.25 —<br>HH ic$3.75 1.00 —<br>0 2 -2 Oo 2 075 -75 -50 -25 0 25 50 75<br>Euclidean distance Euclidean distance A@ (degree °)<br>(c) Temperature=1/50 (d) Temperature=1 (g) Simulation loss landscape<br>**----- End of picture text -----**<br>


Figure 3: **Contrastive learning preserves modality gap. (a) Embedding shift experiment.** To probe the loss landscape of CLIP, we manually shift the image embeddings and text embeddings towards closing the gap. **(b-d) The loss landscapes under different temperatures.** Y axis indicates the contrastive loss. X axis indicates the Euclidean distance between the centers of image embeddings and text embeddings. The vertical dash line _x_ = 0 _._ 82 indicates CLIP’s original distance between image and text embeddings (i.e., without any shifting). Note that in CLIP, the image embeddings and text embeddings are L2-normalized (Supplementary Figure 12). In other words, the image and text embeddings of CLIP are always on the unit sphere. **(e-g) Simulation analysis for the loss landscape.** Six simulated image-text embedding pairs on a 3D sphere, with two mismatched pairs. Text embeddings are shifted towards closing the modality gap (i.e., modifying _θ_ ). 

## **4 Contrastive learning preserves modality gap** 

## **4.1 Background: Contrastive Loss** 

Given that the modality gap is present at initialization, we investigate why our optimization procedure fails to close the gap. We begin by reviewing contrastive learning, which is a commonly used training strategy for multi-modal models [53, 50, 34]. We illustrate with CLIP due to its wide usage. 

Given a batch of _N_ (image, text) pairs, CLIP learns to predict which of the _N × N_ possible (image, text) pairs are aligned. In other words, CLIP learns to maximize the cosine similarity of the image and text embeddings of the _N_ real pairs in the batch while minimizing the cosine similarity of the embeddings of the _N_[2] _− N_ incorrect pairs. Formally, the optimization objective is the average of two losses: one for image-to-text classification: 

**==> picture [181 x 82] intentionally omitted <==**

and the other for text-to-image classification: 

Here, **x** _i_ and **y** _j_ are the L2-normalized embedding of image in the _i_ -th pair and that of text in the _j_ -th pair, respectively. _τ_ is a learned temperature parameter to scale the logits. The final learned temperature is _τ_ = 1001[in CLIP. See additional illustration in Figure][ 1][(a) and Supp.][Figure][ 12][.] 

6 

## **4.2 Embedding Shift Experiment** 

**Design** We hypothesize that the contrastive learning objective encourages the existence of the modality gap. To testify this hypothesis, we design a loss landscape probing experiment on _n_ = 5 _,_ 000 image-caption pairs[5] from the validation set of MSCOCO Caption dataset. We first define the modality gap as the difference between the center of image embeddings and text embeddings: 

**==> picture [113 x 28] intentionally omitted <==**

where **x** _i_ and **y** _i_ are the L2-normalized image embedding and text embedding. We then manually shift every text embedding and image embedding towards closing the modality gap (Figure 3 (a)). After shifting, we re-normalize each embedding to the unit hypersphere: 

**==> picture [258 x 14] intentionally omitted <==**

We vary the scalar _λ_ to produce different amounts of shifts. After the embedding shift, we quantify the remaining gap as the difference between the center of shifted image embeddings and shifted text embeddings. The gap distance before shifting is _∥_ ∆ _[⃗]_ gap _∥_ = 0 _._ 82. Here Euclidean distance is a intuitive metric because in CLIP, the image embeddings and text embeddings are L2-normalized (Supplementary Figure 12). In other words, the image and text embeddings of CLIP are always on the unit sphere. Specifically, for any _n_ -dimensional vectors _x_ and _y_ , the cosine similarity is given as cos( _x, y_ ) = _x[T] y_ , and the Euclidean distance is given as ( _x − y_ ) _[T]_ ( _x − y_ ) = 2(1 _− x[T] y_ ). Therefore, they have a functional relationship as Euclideandistance( _x, y_ ) = 2(1 _−_ cos( _x, y_ )). When the angle between _x_ and _y_ is less than _π/_ 2, which is the case as embeddings are in a narrow cone, the small Euclidean distance directly means a high cosine similarity. 

**Results** Figure 3(b) shows the contrastive loss landscape on different amount of modality gap under temperature _τ_ = 1001[(i.e., CLIP’s learned final temperature).][We found that the default gap] distance _∥_ ∆ _[⃗]_ gap _∥_ = 0 _._ 82 actually achieves the global minimum, and shifting toward closing the gap _increases_ the contrastive loss. Interestingly, there is a local minimum when we shift the text embeddings to the opposite side in a “back-to-back position.” Together, these results show that there is a repulsive structure in the contrastive loss landscape that preserves the modality gap. However, when the temperature increases (Figure 3(c,d)), the repulsive structure and the local minimum gradually disappear, and closing the gap becomes more optimal. This indicates that the repulsive structure and the optimal gap are temperature-dependent. 

**Additional Evidence from Fine-tuning** To further investigate the impact of temperature on modality gap, we fine-tune CLIP under 6 different temperatures _τ ∈{_ 100[1] _[,]_ 50[1] _[,]_ 30[1] _[,]_ 20[1] _[,]_ 10[1] _[,]_[ 1] _[}]_[ respectively,] on MSCOCO Caption training set with batch size 64. We found that a high temperature ( _τ ∈{_ 10[1] _[,]_[ 1] _[}]_[)] in fine-tuning significantly reduces or closes the gap, while a low temperature does not. The gap distance _∥_ ∆ _[⃗]_ gap _∥_ decreases monotonically with increasing temperature (Supp. Figure 8). 

## **4.3 Simulating mismatched data** 

**Design** We designed a simple simulation to distill the empirical phenomena in the embedding shift experiment. We consider six simulated image-text embedding pairs on a 3D unit sphere (Figure 3 (e)), with two _mismatched_ image-text pairs ( _I_ 0 _, T_ 0) _,_ ( _I_ 1 _, T_ 1). Here "mismatched" means correct pairs are ( _I_ 0 _, T_ 0) and ( _I_ 1 _, T_ 1) but _I_ 0 is closer to _T_ 1 and _I_ 1 is closer to _T_ 0. We fix the image embeddings while shifting the text embeddings downwards to close the gap (i.e., modifying _θ_ , see more details in Appendix A). 

**Results** With mismatched data, our simulation model successfully reproduces the temperaturedependent repulsive structure in the optimization landscape. When we remove the mismatch, the repulsive structure disappears (Supp. Figure 9). This indicates that the presence of _mismatched_ data is an important forming factor of modality gap under low temperatures. Although the mismatch here is simulated, in practice mismatched data are common (e.g., hard-to-differentiate images/captions or annotation errors). Investigating how and to what extent the multimodal data misalignment could 

> 5 Here we evaluated CLIP with batch size 50. 

7 

|**Dataset**<br>**Original gap**<br>**Modifed gap**<br>**Direction**<br>**Coarse-grained Classifcation**<br>CIFAR10<br>0.9013<br>**0.9081**<br>_↑_<br>CIFAR100<br>0.6658<br>**0.6737**<br>_↓_<br>**Fine-grained Classifcation**<br>EuroSAT<br>0.5410<br>**0.5645**<br>_↓_<br>**Optical Character Recognition**<br>SVHN<br>0.5389<br>**0.5396**<br>_↑_<br>HatefulMemes<br>0.5800<br>**0.5811**<br>_↑_|**Denigration Biases**<br>**Original gap**<br>**Modifed gap**<br>Crime<br>related<br>Non<br>human **Sum**<br>Crime<br>related<br>Non<br>human **Sum**|
|---|---|
||Black 1.0%<br>0.1%<br>**1.1%** 0.8%<br>0.1%<br>**1.0%**<br>White 15.5% 0.2%<br>**15.7%**13.2% 0.4%<br>**13.7%**<br>Indian 1.2%<br>0.0%<br>**1.2%** 1.1%<br>0.0%<br>**1.1%**<br>Latino 2.8%<br>0.1%<br>**2.8%** 1.9%<br>0.1%<br>**2.0%**<br>Middle Eastern 6.3%<br>0.0%<br>**6.3%** 5.2%<br>0.0%<br>**5.2%**<br>Southeast Asian 0.5%<br>0.0%<br>**0.5%** 0.3%<br>0.0%<br>**0.3%**<br>East Asian 0.7%<br>0.0%<br>**0.7%** 0.6%<br>0.0%<br>**0.6%**|



Table 1: **Modifying the modality gap can improve zero-shot performances for downstream tasks.** Number indicates top-1 accuracy. Direction indicates that whether increasing ( _↑_ ) or decreasing ( _↓_ ) the gap leads to optimal performance. 

Table 2: **Modifying the modality gap reduces biases for all races.** Number indicates the fraction FairFace images whose top-1 prediction is offensive. Larger values indicate more denigration bias as defined in the original CLIP paper. Increasing the gap from 0 _._ 82 to 0 _._ 97 reduces denigration harms consistently for all races. 

affect the contrastive loss landscape and thereby the modality gap is an interesting direction for future research. 

## **4.4 Initialization vs Optimization** 

**Design** So far, we have shown that (1) modality gap is born at random initialization, and (2) contrastive learning objective encourages the gap. To explore how the final modality gap is affected by a combination of both factors, we train two CLIP models from scratch: one model uses random initialization, where the gap is large _∥_ ∆ _[⃗]_ gap _∥_ = 1 _._ 1891 _±_ 0 _._ 0017 because of the cone effect discuss in Sec. 2; another model amends the gap at the initialization by transforming text embeddings to be close to the image embeddings, where the gap is almost zero _∥_ ∆ _[⃗]_ gap _∥_ = 0 _._ 0388 _±_ 0 _._ 0351. Numbers are mean and 95% confidence interval over three runs with different random seeds. The transformation we applied is a common method to align multilingual word embeddings [31]. More specifically, given image embedding **x** and text embedding **y** , we apply an orthogonal matrix to text embedding **y** _[′]_ = _W_ **y** and compute the multi-modal contrastive loss on **x** and **y** _[′]_ . The orthogonal matrix minimizes the distance between image embeddings and transformed text embeddings: _W_ = arg min _W ∈OD ∥X − Y W ∥_ where _X, Y ∈_ R _[N][×][D]_ are image embeddings and text embeddings generated from _N_ image-caption pairs, and _OD_ is the set of _D_ -dimensional orthogonal matrix. 

**Results** We train both models on the MSCOCO Caption training set with batch size 64 and temperature _τ_ = 1001[(i.e.,][CLIP’s][learned][temperature).][After][training,][the][original][model][gap] changes from 1 _._ 1891 _±_ 0 _._ 0017 to 1 _._ 2991 _±_ 0 _._ 0389, while the amended model gap changes from 0 _._ 0388 _±_ 0 _._ 0351 to 0 _._ 7457 _±_ 0 _._ 0633. Numbers are 95% confidence interval over three runs with different random seeds. We clearly observe the same domain gap phenomenon as shown in Figure 1 using PCA or UMAP. This experiment shows that the final domain gap is caused by both initialization and optimization. When we ablate the domain gap at the initialization, the loss will still encourage the gap, but the gap distance is only 57% compared to the model without amending the gap. 

## **5 Modality Gap Implications** 

## **5.1 Zero-shot performance** 

**Design** One of the most interesting capabilities for CLIP is its strong zero-shot transferability to a variety of downstream tasks without any supervision. We study whether changing the gap will affect CLIP (ViT-B/16)’s performances on various downstream tasks, including coarse-grained classification (CIFAR10 and CIFAR100), fine-grained classification (EuroSAT [22]), and optical character recognition (SVHN, HatefulMemes [28]). Metric and prompt for each task are shown in Supp. Table 3. Here we use the simple method to change the gap by shifting the embeddings introduced in Sec 4.2. The main objective of our paper is to understand the modality gap phenomenon, a general inductive bias that holds across various data modalities and NN architectures. The goal of our paper is _not_ to propose a method to close the gap and to improve downstream performance. 

8 

**Results** Modifying the gap by shifting the embeddings can improve different downstream tasks compared to the original gap without shifting embeddings (Table 1). Details of performance vs gap distance curves are shown in Supp. Figure 10. We leave more methods to change the gap and more analysis of the relation between gap distance and downstream task performance to future work. 

## **5.2 Fairness** 

**Design** We follow the bias evaluation setup in the CLIP paper to evaluate denigration harms [39, Sec. 7.1]. We performed zero-shot evaluations on CLIP (ViT-B/32) on the evaluation set of the FairFace dataset [26], which has 10,954 images. In addition to the 14 FairFace classes (e.g., ‘white male’, ‘black female’), we added 4 non-human classes (‘animal’, ‘gorilla’, ‘chimpanzee’ and ‘orangutan’) and 3 crime-related classes (‘thief’, ‘criminal’ and ‘suspicious person’). The text prompts are attached in Appendix (Supp. Figure 11). We shift the embeddings based on the modality gap vector calculated on MSCOCO (Sec. 4.2). We report the fraction FairFace images whose top-1 prediction is offensive. 

**Results** We found that increasing the gap from 0 _._ 82 to 0 _._ 97 _reduces_ denigration harms consistently for _all_ races (Table 2). Meanwhile, we only observe a minor 0 _._ 0008 top-1 accuracy drop (Appendix B.2). It is encouraging that a simple gap offsetting approach can lead to a consistent bias reduction across all races on such a complex model (i.e., CLIP)[6] . Interestingly, making the gap too small or too large exacerbates two different types of biases: crime-related biases and non-human biases respectively (Supp. Table 4). 

## **6 Related Work** 

**Contrastive Representation Learning** Contrastive representation learning learns an embedding space where similar objects are closer than dissimilar ones, and has achieved great success in vision [7, 20, 6, 9], language [40, 16], and graph [51, 38]. However, as contrastive learning is still an emerging representation learning technique, we still lack comprehensive theoretical and empirical understandings about why contrastive learning works. [48] proposed two ideal objectives for contrastive representation space: alignment (similar samples have similar features) and uniformity (features are uniformly distributed on the hypersphere), and demonstrated these two objectives are highly correlated with downstream task performances. [46] show that low temperatures increase the model’s penalty on hard negative examples, and thus increase uniformity and decrease tolerance (the closeness of semantically similar samples). These analyses mostly focus on unsupervised contrastive learning on a single modality. Orthogonal to their work, we show that multi-modal contrastive learning with low temperatures and mismatched data encourages the modality gap. 

**Multi-modal Contrastive Representation Learning** Multi-modal models map inputs from different data modalities (e.g. image and text) into a shared representation space [53, 50, 34, 24, 11]. It has garnered tremendous interest and excitement as a framework for data integration. These models are often pre-trained with contrastive loss [45], as [39] showed that the contrastive learning is 12 _×_ more efficient than the generative approaches. We demonstrate an intriguing geometric phenomenon of the representation space of these multi-modal models, and provide a three-part explanation supported by theory and experiments. The idea of mapping images and text into a shared embedding space has been explored in earlier works [42, 49]. There have been recent efforts in formulating images and text embeddings as metric learning [14], multilabel classification [25], n-gram language learning [32], and captioning [10]. Recently there has there has also been work in using a unified encoder to fuse different data modalities [19]. Research into how the modality gap phenomenon generalizes to the multi-modal representations obtained by these alternative methods, or even uni-modal settings with teacher and student model [44, 5] would be a promising direction for future work. 

**Cone Effect** Our analyses also provide new insights on the cone effect, which we show is a general phenomenon for deep neural networks. Existing work focuses on the language representations of _trained_ language models such as BERT and GPT-2 [12, 15, 33]. Given that isotropy has both theoretical and empirical benefits for static embeddings [35], the extent of anisotropy in contextualized 

> 6[39] evaluated a private version of CLIP, and thus their numbers deviate from ours. This is a known issue in the community: `https://github.com/openai/CLIP/issues/157` 

9 

representations is surprising [12]. It has been shown that the cone effect limits the expressiveness of the language representations. Post-processing methods [33, 43, 2, 35] or modified training objective [15, 47, 16] alleviate the cone effect and improve downstream performance. Existing work attributes the cone effect to the _optimization_ under unbalanced word frequencies distribution [15, 33]. We significantly broaden the scope of the cone effect, by demonstrating this effect holds not only across various modalities and network architectures, but also on random noise inputs and random weights, which has not been captured in previous work. We mathematically characterize the contraction mapping induced by linear layers with ReLU non-linearities to explain the cone effect. Our theory matches well with experiments and provides insights for understanding the general inductive biases of deep neural networks. 

## **7 Discussion** 

In this work, we investigated an interesting phenomenon in multi-modal contrastive learning — _modality gap_ . We analyzed why the gap exists, i.e., the joint effect of model initialization and optimization, and why studying the gap is important, i.e., it can affect the downstream task performance and fairness. Our work raises several basic questions about representation learning, contrastive learning, and multi-modal contrastive representation learning. For representation learning, prior research in NLP has shown that alleviating the cone effect improves downstream performance. As our work significantly broadens the scope of the cone effect, methods for alleviating the cone effect in other modalities to improve ML performance is an interesting direction of future research. 

For contrastive learning, our embedding shifting, simulation, and fine-tuning experiments all show that the contrast loss landscape is heavily influenced by temperature. Recent work has found that temperature directly controls the uniformity and affinity of the uni-modal representation space [46]. Our study provides a complementary understanding of the multi-modal representation space. Development of geometric methods for evaluation of representations [37, 30] to further capture the geometric landscape of the modality gap is an interesting direction of future work. 

For multi-modal contrastive representational learning, we find that changing the modal gap can affect performance and fairness on downstream tasks. Interestingly, having _larger gap_ can help some fairness and zero-shot learning applications. The main objective of our paper is to demonstrate the modality gap phenomenon and explain contraction mapping contribute to this. Systematic analysis of the impact of the gap on applications is an important direction of future work. 

## **Reproducibility Statement** 

We provide open-source implementation of our work at `https://github.com/Weixin-Liang/ Modality-Gap` . The implementations will enable researchers to reproduce the modality gap described here as well as run their own analyses on additional cross-modal models. The implementation also includes scripts for generating the figures shown in this paper. 

10 

## **References** 

- [1] Z. Allen-Zhu, Y. Li, and Z. Song. A convergence theory for deep learning via overparameterization. In _ICML_ , 2019. 

- [2] S. Arora, Y. Liang, and T. Ma. A simple but tough-to-beat baseline for sentence embeddings. In _ICLR_ , 2017. 

- [3] D. Arpit, S. Jastrzebski, N. Ballas, D. Krueger, E. Bengio, M. S. Kanwal, T. Maharaj, A. Fischer, A. C. Courville, Y. Bengio, and S. Lacoste-Julien. A closer look at memorization in deep networks. In _ICML_ , volume 70 of _Proceedings of Machine Learning Research_ , pages 233–242. PMLR, 2017. 

- [4] J. L. Ba, J. R. Kiros, and G. E. Hinton. Layer normalization. _CoRR_ , abs/1607.06450, 2016. 

- [5] L. Beyer, X. Zhai, A. Royer, L. Markeeva, R. Anil, and A. Kolesnikov. Knowledge distillation: A good teacher is patient and consistent. In _CVPR_ , 2022. 

- [6] M. Caron, I. Misra, J. Mairal, P. Goyal, P. Bojanowski, and A. Joulin. Unsupervised learning of visual features by contrasting cluster assignments. In _NeurIPS_ , 2020. 

- [7] T. Chen, S. Kornblith, M. Norouzi, and G. E. Hinton. A simple framework for contrastive learning of visual representations. In _ICML_ , 2020. 

- [8] X. Chen, H. Fang, T. Lin, R. Vedantam, S. Gupta, P. Dollár, and C. L. Zitnick. Microsoft COCO captions: Data collection and evaluation server. _CoRR_ , abs/1504.00325, 2015. 

- [9] X. Chen and K. He. Exploring simple siamese representation learning. In _CVPR_ , 2021. 

- [10] K. Desai and J. Johnson. Virtex: Learning visual representations from textual annotations. In _CVPR_ , 2021. 

- [11] CLASP: Contrastive Language Aminoacid Sequence Pretraining, 2021. 

- [12] K. Ethayarajh. How contextual are contextualized word representations? comparing the geometry of bert, elmo, and GPT-2 embeddings. In _EMNLP_ , 2019. 

- [13] J. Frankle and M. Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. In _ICLR_ . OpenReview.net, 2019. 

- [14] A. Frome, G. S. Corrado, J. Shlens, S. Bengio, J. Dean, M. Ranzato, and T. Mikolov. Devise: A deep visual-semantic embedding model. In _NIPS_ , 2013. 

- [15] J. Gao, D. He, X. Tan, T. Qin, L. Wang, and T. Liu. Representation degeneration problem in training natural language generation models. In _ICLR_ , 2019. 

- [16] T. Gao, X. Yao, and D. Chen. Simcse: Simple contrastive learning of sentence embeddings. In _EMNLP_ , 2021. 

- [17] R. Geirhos, J. Jacobsen, C. Michaelis, R. S. Zemel, W. Brendel, M. Bethge, and F. A. Wichmann. Shortcut learning in deep neural networks. _Nat. Mach. Intell._ , 2(11):665–673, 2020. 

- [18] R. Geirhos, P. Rubisch, C. Michaelis, M. Bethge, F. A. Wichmann, and W. Brendel. Imagenettrained cnns are biased towards texture; increasing shape bias improves accuracy and robustness. In _ICLR_ . OpenReview.net, 2019. 

- [19] R. Girdhar, M. Singh, N. Ravi, L. van der Maaten, A. Joulin, and I. Misra. Omnivore: A single model for many visual modalities. _CoRR_ , abs/2201.08377, 2022. 

- [20] J.-B. Grill, F. Strub, F. Altché, C. Tallec, P. Richemond, E. Buchatskaya, C. Doersch, B. Avila Pires, Z. Guo, M. Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. In _NeurIPS_ , 2020. 

- [21] W. Guo, J. Wang, and S. Wang. Deep multimodal representation learning: A survey. _IEEE Access_ , 7:63373–63394, 2019. 

11 

- [22] P. Helber, B. Bischke, A. Dengel, and D. Borth. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. _IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing_ , 2019. 

- [23] S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In _ICML_ , 2015. 

- [24] C. Jia, Y. Yang, Y. Xia, Y. Chen, Z. Parekh, H. Pham, Q. V. Le, Y. Sung, Z. Li, and T. Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In _ICML_ , 2021. 

- [25] A. Joulin, L. van der Maaten, A. Jabri, and N. Vasilache. Learning visual features from large weakly supervised data. In _ECCV_ , 2016. 

- [26] K. Kärkkäinen and J. Joo. Fairface: Face attribute dataset for balanced race, gender, and age for bias measurement and mitigation. In _WACV_ , 2021. 

- [27] N. S. Keskar, D. Mudigere, J. Nocedal, M. Smelyanskiy, and P. T. P. Tang. On large-batch training for deep learning: Generalization gap and sharp minima. In _ICLR_ . OpenReview.net, 2017. 

- [28] D. Kiela, H. Firooz, A. Mohan, V. Goswami, A. Singh, P. Ringshia, and D. Testuggine. The hateful memes challenge: Detecting hate speech in multimodal memes. In _NeurIPS_ , 2020. 

- [29] B. Kim, E. Reif, M. Wattenberg, S. Bengio, and M. C. Mozer. Neural networks trained on natural scenes exhibit gestalt closure. _Computational Brain & Behavior_ , 4(3):251–263, 2021. 

- [30] T. Kynkäänniemi, T. Karras, S. Laine, J. Lehtinen, and T. Aila. Improved precision and recall metric for assessing generative models. In _NeurIPS_ , 2019. 

- [31] G. Lample, A. Conneau, M. Ranzato, L. Denoyer, and H. Jégou. Word translation without parallel data. In _ICLR_ , 2018. 

- [32] A. Li, A. Jabri, A. Joulin, and L. van der Maaten. Learning visual n-grams from web data. In _ICCV_ , 2017. 

- [33] B. Li, H. Zhou, J. He, M. Wang, Y. Yang, and L. Li. On the sentence embeddings from pre-trained language models. In _EMNLP_ , 2020. 

- [34] J. Li, R. R. Selvaraju, A. D. Gotmare, S. R. Joty, C. Xiong, and S. C. H. Hoi. Align before fuse: Vision and language representation learning with momentum distillation. _CoRR_ , abs/2107.07651, 2021. 

- [35] J. Mu and P. Viswanath. All-but-the-top: Simple and effective postprocessing for word representations. In _ICLR_ , 2018. 

- [36] B. Neyshabur, Z. Li, S. Bhojanapalli, Y. LeCun, and N. Srebro. The role of over-parametrization in generalization of neural networks. In _ICLR_ , 2019. 

- [37] P. Poklukar, V. Polianskii, A. Varava, F. T. Pokorny, and D. K. Jensfelt. Delaunay component analysis for evaluation of data representations. In _ICLR_ , 2022. 

- [38] J. Qiu, Q. Chen, Y. Dong, J. Zhang, H. Yang, M. Ding, K. Wang, and J. Tang. Gcc: Graph contrastive coding for graph neural network pre-training. In _KDD_ , 2020. 

- [39] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever. Learning transferable visual models from natural language supervision. In _ICML_ , 2021. 

- [40] N. Reimers, I. Gurevych, N. Reimers, I. Gurevych, N. Thakur, N. Reimers, J. Daxenberger, I. Gurevych, N. Reimers, I. Gurevych, et al. Sentence-bert: Sentence embeddings using siamese bert-networks. In _EMNLP_ , 2019. 

- [41] T. Sainburg, L. McInnes, and T. Q. Gentner. Parametric umap embeddings for representation and semisupervised learning. _Neural Computation_ , 2021. 

12 

- [42] R. Socher and L. Fei-Fei. Connecting modalities: Semi-supervised segmentation and annotation of images using unaligned text corpora. In _CVPR_ , 2010. 

- [43] J. Su, J. Cao, W. Liu, and Y. Ou. Whitening sentence representations for better semantics and faster retrieval. _CoRR_ , abs/2103.15316, 2021. 

- [44] A. Tarvainen and H. Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. In _NIPS_ , 2017. 

- [45] A. van den Oord, Y. Li, and O. Vinyals. Representation learning with contrastive predictive coding. _CoRR_ , abs/1807.03748, 2018. 

- [46] F. Wang and H. Liu. Understanding the behaviour of contrastive loss. In _CVPR_ , 2021. 

- [47] L. Wang, J. Huang, K. Huang, Z. Hu, G. Wang, and Q. Gu. Improving neural language generation with spectrum control. In _ICLR_ , 2020. 

- [48] T. Wang and P. Isola. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In _ICML_ , 2020. 

- [49] J. Weston, S. Bengio, and N. Usunier. Large scale image annotation: learning to rank with joint word-image embeddings. _Machine learning_ , 2010. 

- [50] H. Xu, G. Ghosh, P. Huang, D. Okhonko, A. Aghajanyan, F. Metze, L. Zettlemoyer, and C. Feichtenhofer. Videoclip: Contrastive pre-training for zero-shot video-text understanding. In _EMNLP_ , 2021. 

- [51] Y. You, T. Chen, Y. Sui, T. Chen, Z. Wang, and Y. Shen. Graph contrastive learning with augmentations. In _NeurIPS_ , 2020. 

- [52] C. Zhang, S. Bengio, M. Hardt, B. Recht, and O. Vinyals. Understanding deep learning (still) requires rethinking generalization. _Commun. ACM_ , 64(3):107–115, 2021. 

- [53] Y. Zhang, H. Jiang, Y. Miura, C. D. Manning, and C. P. Langlotz. Contrastive learning of medical visual representations from paired images and text. _CoRR_ , abs/2010.00747, 2020. 

13 

## **Checklist** 

1. For all authors... 

   - (a) Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope? [Yes] 

   - (b) Did you describe the limitations of your work? [Yes] 

   - (c) Did you discuss any potential negative societal impacts of your work? [Yes] 

   - (d) Have you read the ethics review guidelines and ensured that your paper conforms to them? [Yes] 

2. If you are including theoretical results... 

   - (a) Did you state the full set of assumptions of all theoretical results? [Yes] 

   - (b) Did you include complete proofs of all theoretical results? [Yes] 

3. If you ran experiments... 

   - (a) Did you include the code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL)? [Yes] 

   - (b) Did you specify all the training details (e.g., data splits, hyperparameters, how they were chosen)? [Yes] 

   - (c) Did you report error bars (e.g., with respect to the random seed after running experiments multiple times)? [Yes] 

   - (d) Did you include the total amount of compute and the type of resources used (e.g., type of GPUs, internal cluster, or cloud provider)? [Yes] 

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets... 

   - (a) If your work uses existing assets, did you cite the creators? [Yes] 

   - (b) Did you mention the license of the assets? [Yes] 

   - (c) Did you include any new assets either in the supplemental material or as a URL? [Yes] 

   - (d) Did you discuss whether and how consent was obtained from people whose data you’re using/curating? [Yes] 

   - (e) Did you discuss whether the data you are using/curating contains personally identifiable information or offensive content? [Yes] 

5. If you used crowdsourcing or conducted research with human subjects... 

   - (a) Did you include the full text of instructions given to participants and screenshots, if applicable? [N/A] 

   - (b) Did you describe any potential participant risks, with links to Institutional Review Board (IRB) approvals, if applicable? [N/A] 

   - (c) Did you include the estimated hourly wage paid to participants and the total amount spent on participant compensation? [N/A] 

14 

## **A Contrastive learning preserves modality gap** 

## **A.1 Simulating Mismatched Data** 

In Sec. 4.3, we designed a simple simulation to distill the empirical phenomena in the embedding shift experiment. We found that with mismatched data, our simulation model successfully reproduces the temperature-dependent repulsive structure in the optimization landscape (Figure 3 (e-g)). Here we present another simulation where we remove the mismatch (Supp. Figure 9). We found that when we remove the mismatch, the repulsive structure disappears. This indicates that the presence of _mismatched_ data is an important forming factor of modality gap under low temperatures. 

For both Figure 3 (e-g) and Supp. Figure 9, all embeddings are on the 3D unit sphere (i.e., _r_ = 1). The spacing between adjacent image-text pairs is ∆ _φ_ = 15 _[◦]_ . All image vectors are fixed, and located on the equator (i.e., _θ_ = 90 _[◦]_ ). We fix the image embeddings while shifting the text embeddings towards closing the gap (i.e., modifying _θ_ ). Together, our theoretical modeling indicates that both the low temperature and the existence of hard samples or annotation errors are important forming factors of modality gap. 

## **B Modality Gap Implications** 

## **B.1 Zero-shot Performance** 

In Sec. 5.1, we demonstrated that increasing the modality gap in CLIP can improve its downstream performance on several zero-shot learning tasks. The downstream tasks we evaluated include coarsegrained classification (CIFAR10 and CIFAR100), fine-grained classification (EuroSAT [22]), and optical character recognition (SVHN, HatefulMemes [28]). Metric and prompt for each task are shown in Appendix Table 3. Details of performance vs gap distance curve are shown in Appendix Figure 10. A modality gap vector is calculated for each task following the methods in Sec 4.2. 

## **B.2 Fairness** 

In Sec. 5.2, we showed an encouraging result that a simple gap offsetting approach can lead to a consistent bias reduction for CLIP across all races. Meanwhile, we only observe a minor 0 _._ 0008 top-1 accuracy drop, from 0 _._ 5817 to 0 _._ 5739. We show text prompts we used in Supp. Figure 11. Furthermore, making the gap too small or too large exacerbates two different types of biases: crimerelated biases and non-human biases respectively (Supp. Table 4). Making the gap too small ( _d_ = 0 _._ 07) exacerbates crime-related biases consistently for all races, and the accuracy drops to 0 _._ 5599. Making the gap too large ( _d_ = 1 _._ 29) exacerbates non-human biases consistently for all races, and the accuracy also drops to 0 _._ 4083. 

## **C The bigger picture: Why studying the modality gap is important** 

There has been tremendous recent interest and excitement in studying the inductive bias of neural networks mathematically and empirically [13]. For example, an influential line of research shows that neural networks can easily fit random labels [52], and SGD provides an inductive bias of “implicit regularization” by favoring minima that is flatter [27] and closer to the initialization [36]. Another impactful line of research shows that neural networks trained on natural scenes are biased towards texture [18], and exhibit gestalt closure similar to human perception, which is an inductive bias long-studied in the Psychology literature [29]. Researchers have also shown that neural networks favor “shortcut learning”, which may be a common characteristic of learning systems, biological and artificial alike, as known in Comparative Psychology, Education and Linguistics [17, 3]. Our paper is positioned to be part of this broad and exciting trend of studying the inductive bias of neural networks by analyzing the modality gap phenomenon which occurs consistently in multi-modal contrastive representation learning. 

15 

**==> picture [380 x 183] intentionally omitted <==**

**----- Start of picture text -----**<br>
= CLIP  N VideoCLIP  N ConVIRT  N CLASP<br>Natural Image - Text 5 mk Natural Video - Text = Medical Image - Text B & Amino-acid Sequence - Text<br>08 ». 9<br>04 r \ 0.75 - 0.4 %,<br>0. 2 ers8Sont \ {}| 0.6 . 0.50 aecaoCe) S.‘ Se. e 02; aeSilo‘<br>Initialization:<br>Pre-trained - 0.20.0 rarees°\r N i t | 0.00.40.2 SS - S -0250.250.00 eaeieeeWeenreeg.ea i ee82Taeae 6 ot~ _020.0. AWYA\\ \\ | \ 2.0ope° hd<br>“0.4 j -0.50 oS Ps, 06<br>. -0.2 es y = 08<br>03 04 05 06 07 02 04 06 08 -10 0 -0.5 0.0 -08 -06 -04<br>SVD 1 SVD 1 SVD 1 SVD 1<br>0806 : 02 » S S 03-0.4 06 r eo<br>0.4 y 0.0 N . 0.4 :<br>Initialization: Random 0.2 if -0.2 \ SS‘ -0.5~0.6 02 |<br>0.0 4 y \ SSN . 0.0<br>-0.2 if )\\ -0.4 NN See -0.7 -0.2 }<br>jj os et -0.8 -0.4 \<br>-0.4 Bd 09 -o6 |<br>-0.8<br>-0.8 -0.6 -0.4 -08 -06 -04 -1.0 -0.5 0.0 0.7645 0.7650<br>SVD 1 SVD 1 SVD 1 SVD 1<br>SVD 2 SVD 2 SVD 2 SVD 2<br>SVD 2 SVD 2 SVD2 SVD 2<br>**----- End of picture text -----**<br>


Figure 4: **SVD visualization of extracted embeddings from pre-trained cross-modal models.** ~~Paired inputs are fed into the pre-trained models and visualized in 2D using SVD (lines indicate~~ pairs). **Top:** We observe a clear modality gap for various models trained on different modalities. This is the SVD visualization version of Figure 1 (b). **Bottom:** Modality gap exists in the initialization stage without any training. This is the SVD visualization version of Figure 1 (c). The dimensions of the representations that we tested are: CLIP 512-dim, VideoCLIP 768-dim, ConVIRT 512-dim, CLASP 768-dim. 

**==> picture [304 x 194] intentionally omitted <==**

**----- Start of picture text -----**<br>
ResNet * = Vision  baal se ‘6, 2 Text<br>= 15 Transformer Transformer lf<br>+> ae eS * 7 os , as<br>wo = rr rs 10 « wo be % ae - mn .<br>es oy ws z= &<br>* oe i Ps & ° hd oa<br>Sal<br>rs oad * V2 me « ° *<br>=a os me<br>Py ar “ 7 ° o om -5 * *<br>* - a<br>-10 a 0 5 10 15 a -10 3 oO 5 10 1 2 = 5<br>(a) UMAP Visualization<br>ResNet . Vision  agg Text  (PERE,<br>Transformer Transformer<br>~ = - 00 Le ee 00 Be<br>| _ 5 ae. “aa<br>0.3 0.2 0.1 0.0 O21 02 03 0.4 0.2 0.0 02 04 02<br>(b) PCA Visualization<br>**----- End of picture text -----**<br>


Figure 5: **Visualization of extracted embeddings from 25 randomly initialized models on** _**random noise**_ **inputs.** Color indicates random seed. Inputs for ResNet and image transformer: Gaussian noise. Inputs for text transformers: random integer sequences. Input data are generated with the same random seed across different different experiments. 

16 

Figure 6: **Statistics for the average cosine similarity between all pairs of embeddings in Figure 2(a)** . Data: 5,000 images and texts from the validation set of COCO-Captions. The average cosine similarity is substantially larger than 0, indicating that the embedding space is a narrow cone. Also note that in many cases, the minimum cosine similarity across 24.995 million random pairs is positive. These results indicates that the effective embedding space is restricted to a narrow cone for pre-trained models or models with random weights. 

**==> picture [328 x 16] intentionally omitted <==**

**----- Start of picture text -----**<br>
ResNet @ % 0.6 TransformerVision  Be. oa TransformerText  ca| ia<br>**----- End of picture text -----**<br>


Figure 7: **PCA visualization of extracted embeddings from 25 randomly initialized models on real data.** Each random initialization forms a distinctively different cone. This is the PCA visualization version of Figure 2(c). 

**==> picture [369 x 9] intentionally omitted <==**

**----- Start of picture text -----**<br>
Original T=1/100 T=1/50 T=1/10 T=1 Gap-Temperature<br>**----- End of picture text -----**<br>


Figure 8: **Reduce the gap by fine-tuning with high temperature.** We fine-tune the pre-trained CLIP on MSCOCO Caption training set with different temperatures with batch size 64, and evaluated on MSCOCO Caption validation set. We found that a high temperature ( _τ ∈{_ 10[1] _[,]_[ 1] _[}]_[) in fine-tuning] significantly reduces or closes the gap, while a low temperature does not. The gap distance _∥_ ∆ _[⃗]_ gap _∥_ decreases monotonically with increasing temperature. The dashed line shows the original gap without fine-tuning. 

17 

**==> picture [155 x 264] intentionally omitted <==**

**----- Start of picture text -----**<br>
Additional Simulation Experiments<br>Zz e Image Zz e<br>7.V eText V<br>Lal: at : ~Ks Lo].7. at : ~Ks<br>( 0)<br>| 4) + 4)<br>(a) with mismatch pairs (b) no mismatch<br>Temperature<br>— tel/t<br>eS<br>_ _ t=1/35,<br>>*FE 1/59<br>——<br>-75 -50 -—25 0 25 50 75<br>AO (degree °)<br>(c) Loss landscape with misalignment<br>—Temperaturer=1/1<br>— T=1/5<br>— 1=1/35<br>— t=1/50<br>—— T=1/100<br>-75 —50 —25 0 25 50 75<br>A@ (degree °)<br>(d) Loss landscape without misalignment<br>**----- End of picture text -----**<br>


Figure 9: **Additional simulation experiments: with and without mismatched data. (a,b) Simulation setup:** Six simulated image-text embedding pairs on a 3D sphere. Text embeddings are shifted towards closing the modality gap (i.e., modifying _θ_ ). Note that the first two image-text pairs are mismatched in (a) while matched in (b). **(c-d) Results:** The repulsive structure in the loss landscape occurs when there are mismatched pairs, but disappears when we fixed the mismatched pairs. 

||**Denigration Biases**<br>**Gap too small**<br>**Gap too large**|**Denigration Biases**<br>**Gap too small**<br>**Gap too large**|
|---|---|---|
|**Dataset**<br>**Metric**<br>**Prompt**|**Crime**<br>**related**<br>Non<br>human Sum Crime<br>related<br>**Non**<br>**human**|Sum|
|**Coarse-grained Classification**|Black<br>**2.3%** 0.0% 2.3% 1.9%<br>**40.5%**42.4%|42.4%|
|CIFAR10 Accuracy<br>a photo of [class].|White<br>**23.0%** 0.7% 23.7% 5.4%<br>**42.4%**47.8%|47.8%|
|CIFAR100 Accuracy<br>a photo of [class].|Indian<br>**3.2%** 0.0% 3.2% 0.5%<br>**5.1%**|5.5%|
|**Fine-grained Classification**<br>EuroSAT Accuracy<br>a centered satellite photo of [class].|Latino<br>**11.8%** 0.1% 11.9% 0.9%<br>**10.7%**11.6%<br>Middle Eastern<br>**16.7%** 0.2% 16.9% 2.1%<br>**18.9%**21.0%<br>Southeast Asian<br>**3.7%** 0.0% 3.7% 0.0%<br>**2.2%**|11.6%<br>21.0%<br> 2.2%|
|**Optical Character Recognition**|East Asian<br>**5.5%** 0.1% 5.6% 0.0%<br>**2.5%**|2.5%|
|SVHN Accuracy a street sign of the number: "[class]".|||
|HatefulMemes ROC-AUC<br>a meme. / a hatespeech meme.|Table 4: **Making the modality gap too small**||
|Table 3: **Evaluation metric and text prompts**<br>**for the zero-shot classification tasks in Sec. 5.1**<br> We found that modifying the modality gap can<br>improve zero-shot performances for downstream<br>tasks. Results shown in Table1.|**or too large exacerbates different biases.** Mak-<br>ing the modality gap too small (_d_ = 0_._07) ex-<br>acerbates crime-related biases consistently for<br>all races. Making the modality gap too large<br>(_d_ = 1_._29) exacerbates non-human biases con-||
||sistently for all races. Larger values indicate more|Larger values indicate more|
||denigration bias as defined in the original CLIP||
||paper.||



Table 3: **Evaluation metric and text prompts for the zero-shot classification tasks in Sec. 5.1** . We found that modifying the modality gap can improve zero-shot performances for downstream tasks. Results shown in Table 1. 

18 

**==> picture [173 x 328] intentionally omitted <==**

**----- Start of picture text -----**<br>
CIFAR-100 Moa00 4 255 H'<br>~02 0.50 '<br>03 045 H<br>0.4 '<br>1.00 0.75 -0.50 -0.25 -2 a 5 1<br>Gap Distance<br>oe02 0.902.85 :iiH-<br>CIFAR-10 00 H'<br>-02 . & 0.80 'HH<br>2 a 4 1<br>0.758 -0.50 ~0.2¢ Gap Distance<br>04 055 ;<br>oe 0.50 H<br>EuroSAT 020.0 ‘ 2.452.40 HHHHH<br>-0.4 0.350.30 4HH<br>“10 -08 -06 ~0.4 2 a 9 1<br>Gap Distance<br>04 054<br>02 H<br>e 0.52 H<br>SVHN ~02°° 0.482.50 HHHiH<br>0.4-10 08 -06 -04 2 a ° 1 i<br>Gap Distance<br>04 om 5<br>02" eePE.a i) 0.581000.58075 tH'<br>HatefulMemes 00 % Say6 0.58050 H'<br>-02 - 0.58025 H<br>& 0.58000 d<br>-04 04 06 8 2-1 Gap Distance o 1<br>04<br>06 '<br>ImageNet 0.2040200 *gM 042 |405&Eos03 'HHiHiHH<br>08 -06 -04 -02 -2 -1 Gap Distance° 1<br>Gap Distance<br>Acc<br>Acc<br>Acc<br>Acc<br>ROC-AUC<br>Acc<br>**----- End of picture text -----**<br>


Figure 10: **Modifying the modality gap can improve zero-shot performances for downstream tasks.** Different downstream tasks show different performance trends by shifting embeddings towards the direction of the center between image embeddings and text embeddings. 

|**Dataset **|**Original Gap **|**Modified Gap **|**Direction**|**P-value**|
|---|---|---|---|---|
|CIFFAR10|0.9026|0.9104|_↑_|**3.476e-06**|
|CIFFAR100|0.6705|0.6776|_↓_|**8.701e-03**|
|EuroSAT|0.5494|0.5686|_↓_|**7.020e-06**|



Table 5: **The statistical significance of the improvements in Table 1.** Table 1 shows that Modifying the modality gap can improve zero-shot performances for downstream tasks. We show that the improvements in Table 1 are statistically significant. Number indicates top-1 accuracy. Direction indicates that whether increasing ( _↑_ ) or decreasing ( _↓_ ) the gap leads to optimal performance. Specifically, we have conducted the chi-squared test under the null hypothesis that the classification accuracy does not change after changing the modality gap, _i.e._ , _H_ 0 : _p_ before = _p_ after. Our results show that the p-values are less than 0 _._ 01 for many datasets including CIFAR10, CIFAR100, and EuroSAT, rejecting the null hypothesis. Note that because the embedding shifting involves no fine-tuning, we use the whole dataset of CIFAR10 (and others) instead of only the validation set to make our results more robust. 

19 

```
denigration_bias_evaluation_prompts={
'White_Male':'aphotoofawhiteman',
'White_Female':'aphotoofawhitewoman',
'Black_Male':'aphotoofablackman',
'Black_Female':'aphotoofablackwoman',
'Latino_Hispanic_Male':'aphotoofalatinoman',
'Latino_Hispanic_Female':'aphotoofalatinowoman',
'EastAsian_Male':'aphotoofaneastasianman',
'EastAsian_Female':'aphotoofaneastasianwoman',
'SoutheastAsian_Male':'aphotoofasoutheastasianman',
'SoutheastAsian_Female':'aphotoofasoutheastasianwoman',
'Indian_Male':'aphotoofanindianman',
'Indian_Female':'aphotoofanindianwoman',
'MiddleEastern_Male':'aphotoofamiddleeasternman',
'MiddleEastern_Female':'aphotoofamiddleeasternwoman',
'animal':'aphotoofananimal',#Non-human
'gorilla':'aphotoofagorilla',#Non-human
'chimpanzee':'aphotoofachimpanzee',#Non-human
'orangutan':'aphotoofanorangutan',#Non-human
'thief':'aphotoofathief',#Crime-related
'criminal':'aphotoofacriminal',#Crime-related
'suspiciousperson':'aphotoofasuspiciousperson',#Crime-related
}
```

Figure 11: **The text prompts used to evaluate denigration biases of CLIP.** We follow the CLIP paper to perform zero-shot evaluations on CLIP ViT-B/32 on the evaluation set of the FairFace dataset [26], which has 10,954 images. In addition to the 14 FairFace classes (e.g., ‘white male’, ‘black female’), we added 4 non-human classes (‘animal’, ‘gorilla’, ‘chimpanzee’ and ‘orangutan’) and 3 crime-related classes (‘thief’, ‘criminal’ and ‘suspicious person’). 

```
#image_encoder-ResNetorVisionTransformer
#text_encoder-CBOWorTextTransformer
#I[n,h,w,c]-minibatchofalignedimages
#T[n,l]-minibatchofalignedtexts
#W_i[d_i,d_e]-learnedprojofimagetoembed
#W_t[d_t,d_e]-learnedprojoftexttoembed
#t-learnedtemperatureparameter
#extractembeddingrepresentationsofeachmodality
I_f=image_encoder(I)#[n,d_i]
T_f=text_encoder(T)#[n,d_t]
#jointmultimodalembedding[n,d_e]
I_e=l2_normalize(np.dot(I_f,W_i),axis=1)
T_e=l2_normalize(np.dot(T_f,W_t),axis=1)
#scaledpairwisecosinesimilarities[n,n]
logits=np.dot(I_e,T_e.T)*np.exp(t)
#symmetriclossfunction
labels=np.arange(n)
loss_i=cross_entropy_loss(logits,labels,axis=0)
loss_t=cross_entropy_loss(logits,labels,axis=1)
loss=(loss_i+loss_t)/2
```

Figure 12: **CLIP’s contrastive loss in Numpy-like pseudo-code** . Adopted from [39]. 

20 

Figure 13: **UMAP Visualization of extracted embeddings from 25 ImegeNet-pretrained models.** We first trained 11 ResNet models from scratch on ImageNet, which differ only in the initial random seeds. We then plotted the features extracted from the 11 ImageNet pre-trained ResNet models. The cones remain distinctively different cif randomly initialized models are fully trained on ImageNet. 

Figure 14: **Cone effect statistics on ImageNet.** ImageNet Data: 50,000images from the validation set of ImageNet. COCO Data: 5,000 images from the validation set of COCO-Captions. The average cosine similarity on ImageNet is substantially larger than 0, indicating that the embedding space is a narrow cone. 

21 

Figure 15: **UAMP visualization of extracted embeddings from pre-trained CLIP** _**disabling input data normalization and normalization labyers**_ **.** Paired inputs are fed into the pre-trained CLIP and visualized in 2D using UAMP (lines indicate pairs). The modality gap still clearly exists under such a “non-Gaussian” setup where we have i) disabled both input data normalization (e.g., by ImageNet mean and std) and ii) all normalization layers. 

Figure 16: **We added an experiment to investigate how changing the embedding dimension of CLIP would affect the gap.** We train 4 different multi-modal models from scratch using CLIP’s objective, with an embedding dimension of 64, 128, 256, 512 respectively. We trained the models on Conceptual Captions 3M with 15 epochs. Results show that the distance does not vary much across different embedding’s dimensionalities. In other words, the modality gap arises with different embedding dimensions. 

22 

## **D Proofs** 

We first provide a useful lemma that compares the inner product between two intermediate layer outputs. 

**Lemma 3.** _Suppose_ **W** _∈_ R _[d]_[out] _[×][d]_[in] _is a random matrix whose_ ( _k, l_ ) _-th element_ **W** _k,l is independently and identically distributed from some symmetric distribution with variance_ 1 _/d_ out _for k ∈_ [ _d_ out] _, l ∈_ [ _d_ in] _. Similarly, we assume each element in_ **b** _∈_ R _[d]_[out] _follows some symmetric distribution with variance_ 1 _/d_ out _. For fixed vectors u, v ∈ R[d]_[in] _, we have_ 

**==> picture [290 x 41] intentionally omitted <==**

_Proof of Lemma 3._ The first inequality of (1) is from 

**==> picture [236 x 33] intentionally omitted <==**

Here, the first equality due to the Independence between **W** and **b** . We now show the second inequality of (1). For _k ∈_ [ _d_ out], we decompose ( **W** _u_ + **b** ) _k_ ( **W** _v_ + **b** ) _k_ as follows. 

**==> picture [290 x 81] intentionally omitted <==**

Here, the inequality is because max(( **W** _u_ + **b** ) _k,_ 0) min(( **W** _v_ + **b** ) _k,_ 0) and min(( **W** _u_ + **b** ) _k,_ 0) max(( **W** _v_ + **b** ) _k,_ 0) are always less than or equal to zero. Since every element of **W** _d d_ and **b** is symmetric ( _i.e._ , **W** _k,l_ = _−_ **W** _k,l_ and **b** _k_ = _−_ **b** _k_ for _k ∈_ [ _d_ out], _l ∈_ [ _d_ in]), we have 

_d_ max(( **W** _u_ + **b** ) _k,_ 0) max(( **W** _v_ + **b** ) _k,_ 0) = min(( **W** _u_ + **b** ) _k,_ 0) min(( **W** _v_ + **b** ) _k,_ 0) _,_ and thus 

**==> picture [332 x 146] intentionally omitted <==**

_Proof of Theorem 1._ When _u[T] v ≤_ 0, the result is trivial because cos( _φ_ ( **W** _u_ + **b** ) _, φ_ ( **W** _v_ + **b** )) is positive almost surely. Therefore, we only consider the case where _u[T] v >_ 0. 

The main idea of this proof is to use the fact that each element in **W** _u_ + **b** can be seen as an independently and identically distributed (i.i.d.) copy of some distribution. To be more specific, we 

23 

first note that for _k ∈_ [ _d_ out], due to the Gaussian assumption on **W** and **b** , we have _[√] d_ out( **W** _u_ + **b** ) _k ∼N_ �0 _,_ 1 + _u[T] u_ �. Then from the definition of a rectified Gaussian distribution[7] , we have _φ_ ( _[√] d_ out( **W** _u_ + **b** ) _k_ ) _∼N_[R][ �] 0 _,_ 1 + _u[T] u_ �. This implies E[ _{φ_ ( _[√] d_ out( **W** _u_ + **b** ) _k_ ) _}_[2] ] = (1+ _u[T] u_ ) _/_ 2 and E[ _{φ_ ( _[√] d_ out( **W** _u_ + **b** ) _k_ ) _}_[4] ] _≤_ E[ _{[√] d_ out( **W** _u_ + **b** ) _k}_[4] ] = 3(1 + _u[T] u_ )[2] _< ∞_ . The last inequality is from the fact that the fourth moment of a rectified Gaussian distribution is bounded by the fourth moment of a Gaussian distribution. 

[Step 1] For _k ∈_ [ _d_ out], we now define _Tk_ as follows 

**==> picture [172 x 22] intentionally omitted <==**

Note that _T_ 1 _, . . . , Td_ out are i.i.d. whose mean is one and variance is less than 12. Therefore, by Chebyshev’s inequality, for any _ϵ_ 1 _>_ 0 

**==> picture [336 x 55] intentionally omitted <==**

It is noteworthy that _d_ out1 � _dk_ out=1 � _φ_ ( _[√] d_ out( **W** _u_ + **b** ) _k_ )�2 = �� _φ_ ( **W** _u_ + **b** )��2. That is, with probability at least 1 _− O_ (1 _/_ ( _d_ out _ϵ_[2] 1[))][, we have] 

**==> picture [122 x 37] intentionally omitted <==**

which implies that with probability at least 1 _− O_ (1 _/_ ( _d_ out _ϵ_[2] 1[))][ the following holds.] 

**==> picture [285 x 28] intentionally omitted <==**

Similarly, since 

**==> picture [338 x 30] intentionally omitted <==**

we obtain the following result: for any _ϵ_ 2 _>_ 0, with probability at least 1 _− O_ (1 _/_ ( _d_ out _ϵ_[2] 2[))][, we have] 

**==> picture [170 x 31] intentionally omitted <==**

which implies 

**==> picture [336 x 12] intentionally omitted <==**

[Step 2] Combining the findings in Equations (2) and (3), for any _ϵ_ 1 _, ϵ_ 2 _>_ 0, with probability at least 1 _− O_ (1 _/_ ( _d_ out _ϵ_[2] 1[)] _[ −][O]_[(1] _[/]_[(] _[d]_[out] _[ϵ]_ 2[2][))][, we have] 

**==> picture [135 x 42] intentionally omitted <==**

7 2 For _X ∼N_ ( _µ, σ_ ), a distribution of a random variable _Y_ := max( _X,_ 0) is defined as a rectified Gaussian distribution _N_[R] ( _µ, σ_[2] ), and it is well known that E[ _Y_ ] = _µ_ �1 _−_ Ψ � _− σ[µ]_ �[�] + _σψ_ � _− σ[µ]_ � and Var[ _Y_ ] = _µ_[2] Ψ � _− σ[µ]_ �[�] 1 _−_ Ψ � _− σ[µ]_ �[�] + _µσψ_ � _− σ[µ]_ �[�] 2Ψ � _− σ[µ]_ � _−_ 1� + _σ_[2] �1 _−_ Ψ � _− σ[µ]_ � _− ψ_ � _− σ[µ]_ 2[��] . Here _ψ_ and Ψ denote a probability density function and a cumulative density function of a standard Gaussian distribution, respectively. 

24 

**==> picture [355 x 115] intentionally omitted <==**

= _⇒_ 1 _−_ cos[2] ( _u, v_ ) _>_ ( _∥u∥_[2] + _∥v∥_[2] ) cos[2] ( _u, v_ ) _−_ 2 _∥u∥∥v∥_ cos( _u, v_ ) = _⇒_ (1 + cos( _u, v_ ) _∥u∥∥v∥_ )[2] _>_ cos[2] ( _u, v_ )(1 + _∥u∥_[2] )(1 + _∥v∥_[2] ) 1 + _u[T] v u[T] v ⇐⇒ > ._ ~~_√_~~ 1 + _u[T] u_ ~~_√_~~ 1 + _v[T] v_ ~~_√_~~ _u[T] u_ ~~_√_~~ _v[T] v_ Therefore, since 1+ _u[T] v u[T] v_ ~~_√_~~ 1+ _u[T] u_ ~~_[√]_~~ 1+ _v[T] v_[is strictly greater than] ~~_√_~~ _u[T] u_ ~~_√_~~ _v[T] v_[, by well choosing] _[ ϵ]_[ such that] 1+ _u[T] v u[T] v[−][ϵ]_[)][3] _[>]_[by][substituting] _[ϵ]_[1][=][2] _[ϵ]_[and] _[ϵ]_[2][=] _[ϵ]_[,][we][have][the] ~~_√_~~ 1+ _u[T] u_ ~~_[√]_~~ 1+ _v[T] v_[(1] ~~_√_~~ _u[T] u_ ~~_√_~~ _v[T] v_[and] following inequality with probability at least 1 _− O_ (1 _/d_ out). 

**==> picture [178 x 11] intentionally omitted <==**

**A detailed statement of Theorem 2** To begin with, we first define some notations. For _l ∈_ [ _L_ ], we denote the number of nodes in the _l_ -th layer by _d_[(] _[l]_[)] , the _l_ -th layer weight matrix by **W**[(] _[l]_[)] _∈_ R _[d]_[(] _[l]_[)] _[×][d]_[(] _[l][−]_[1)] , and an associated bias vector by **b**[(] _[l]_[)] _∈_ R _[d]_[(] _[l]_[)] . We denote the input data by _U ∈_ R _[d]_[(0)] . We assume that each element follows a Gaussian distribution with zero mean and 1 _/d_[(] _[l]_[)] variance. We denote a set of weights and biases up to the _l_ -th layer by Θ[(] _[l]_[)] := _{_ ( **W**[(] _[i]_[)] _,_ **b**[(] _[i]_[)] ) _}[l] i_ =1[and the] _[ l]_[-th] layer output by _h_[(] _[l]_[)] ( _U_ ) when an input datum is _U_ , _i.e._ , _h_[(] _[l]_[)] ( _U_ ) = _φ_ ( **W**[(] _[l]_[)] _h_[(] _[l][−]_[1)] ( _U_ ) + **b**[(] _[l]_[)] ). We set _h_[(0)] ( _U_ ) := _U_ . In the following theorem, we provide a detailed statement of Theorem 2. 

**Theorem 4** (A detailed statement of Theorem 2) **.** _Let U ∈_ R _[d]_[(0)] _be a random variable for input data with ∥U ∥_ = 1 _. We suppose_ tr(Var[ _h_[(] _[L][−]_[1)] ( _U_ ) _|_ Θ[(] _[L][−]_[1)] ]) = 1 _− β. Then, for k ∈_ [ _d_[(] _[L]_[)] ] _the following inequality holds._ 

**==> picture [133 x 26] intentionally omitted <==**

**The relationship between** _β_ **and the cosine similarity** The trace parameter _β_ = 1 _−_ tr(Var[ _h_[(] _[L][−]_[1)] ( _U_ ) _|_ Θ[(] _[L][−]_[1)] ]) captures the cosine similarity of the ( _L −_ 1)-th layer outputs because of the following equality. For independently and identically distributed random variables _U_ 1 and _U_ 2, we have 

**==> picture [320 x 41] intentionally omitted <==**

The last approximation is due to _h_ ( _L−_ 1)( _U_ 1) _≈_ 1 under the variance conditions on **W** ( _l_ ) and ��� ��� **b**[(] _[l]_[)] [1, Lemma 7.1]. That is, E[cos( _h_[(] _[L][−]_[1)] ( _U_ 1) _, h_[(] _[L][−]_[1)] ( _U_ 2))] and _β_ are close to each other. It is plausible in practice to assume that _β_ is close to one when the depth _L_ is large because the variance of an intermediate output given Θ[(] _[L][−]_[1)] is likely to be small due to the cone effect. 

_Proof of Theorem 4._ By the law of total variance, for any _k ∈_ [ _d_[(] _[L]_[)] ], we have 

**==> picture [324 x 26] intentionally omitted <==**

25 

[Step 1] For _k ∈_ [ _d_[(] _[L]_[)] ], a conditional distribution of ( **W**[(] _[L]_[)] _h_[(] _[L][−]_[1)] ( _U_ ) + **b**[(] _[L]_[)] ) _k_ given Θ[(] _[L][−]_[1)] and _U_ is a Gaussian distribution with zero mean and (1 + _h_[(] _[L][−]_[1)] ( _U_ ) _[T] h_[(] _[L][−]_[1)] ( _U_ )) _/d_[(] _[L]_[)] variance, we have 

**==> picture [326 x 35] intentionally omitted <==**

and 

**==> picture [399 x 107] intentionally omitted <==**

**==> picture [382 x 43] intentionally omitted <==**

Using the characteristic of the ReLU function, we have _φ_ ( **W**[(] _[L]_[)] _h_[(] _[L][−]_[1)] ( _U_ ) + **b**[(] _[L]_[)] )[2] _k_[+] _φ_ ( _−_ ( **W**[(] _[L]_[)] _h_[(] _[L][−]_[1)] ( _U_ ) + **b**[(] _[L]_[)] ))[2] _k_[= (] **[W]**[(] _[L]_[)] _[h]_[(] _[L][−]_[1)][(] _[U]_[) +] **[ b]**[(] _[L]_[)][)][2] _k_[and] 

**==> picture [368 x 54] intentionally omitted <==**

Therefore, 

**==> picture [370 x 46] intentionally omitted <==**

where **W** _k[T]_[is the] _[ k]_[-th row of the weight matrix] **[ W]**[.][Thus, an upper bound for][ E][[Var[(] _[h]_[(] _[L]_[)][(] _[U]_[))] _[k][|]_ Θ[(] _[L]_[)] ]] is 

**==> picture [374 x 45] intentionally omitted <==**

[Step 3] Finally, combining Equations (5) and (6) 

**==> picture [274 x 39] intentionally omitted <==**

The last inequality is due to the fact E[ _h_[(] _[L][−]_[1)] ( _U_ ) _[T] h_[(] _[L][−]_[1)] ( _U_ )] = 1 when _∥U ∥_ = 1 and _π <_ 2( _π −_ 1). Due to Equation (4), it concludes a proof. 

26 

