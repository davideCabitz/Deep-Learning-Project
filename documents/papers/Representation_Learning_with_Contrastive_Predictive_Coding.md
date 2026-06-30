# **Representation Learning with Contrastive Predictive Coding** 

**Aaron van den Oord** DeepMind `avdnoord@google.com` 

**Yazhe Li Oriol Vinyals** DeepMind DeepMind `yazhe@google.com vinyals@google.com` 

## **Abstract** 

While supervised learning has enabled great progress in many applications, unsupervised learning has not seen such widespread adoption, and remains an important and challenging endeavor for artificial intelligence. In this work, we propose a universal unsupervised learning approach to extract useful representations from high-dimensional data, which we call Contrastive Predictive Coding. The key insight of our model is to learn such representations by predicting the future in _latent_ space by using powerful autoregressive models. We use a probabilistic contrastive loss which induces the latent space to capture information that is maximally useful to predict future samples. It also makes the model tractable by using negative sampling. While most prior work has focused on evaluating representations for a particular modality, we demonstrate that our approach is able to learn useful representations achieving strong performance on four distinct domains: speech, images, text and reinforcement learning in 3D environments. 

## **1 Introduction** 

Learning high-level representations from labeled data with layered differentiable models in an endto-end fashion is one of the biggest successes in artificial intelligence so far. These techniques made manually specified features largely redundant and have greatly improved state-of-the-art in several real-world applications [1, 2, 3]. However, many challenges remain, such as data efficiency, robustness or generalization. 

Improving representation learning requires features that are less specialized towards solving a single supervised task. For example, when pre-training a model to do image classification, the induced features transfer reasonably well to other image classification domains, but also lack certain information such as color or the ability to count that are irrelevant for classification but relevant for e.g. image captioning [4]. Similarly, features that are useful to transcribe human speech may be less suited for speaker identification, or music genre prediction. Thus, unsupervised learning is an important stepping stone towards robust and generic representation learning. 

Despite its importance, unsupervised learning is yet to see a breakthrough similar to supervised learning: modeling high-level representations from raw observations remains elusive. Further, it is not always clear what the ideal representation is and if it is possible that one can learn such a representation without additional supervision or specialization to a particular data modality. 

One of the most common strategies for unsupervised learning has been to predict future, missing or contextual information. This idea of predictive coding [5, 6] is one of the oldest techniques in signal processing for data compression. In neuroscience, predictive coding theories suggest that the brain predicts observations at various levels of abstraction [7, 8]. Recent work in unsupervised learning has successfully used these ideas to learn word representations by predicting neighboring words [9]. For images, predicting color from grey-scale or the relative position of image patches has also been 

Preprint. Work in progress. 

**==> picture [343 x 131] intentionally omitted <==**

**----- Start of picture text -----**<br>
ct Predictions<br>~, ™, ae ~N<br>g ar g ar g ar g ar _\ _\ nN‘. an \,<br>\ \. ‘ \<br>zt zt +1 zt +2 zt +3 zt +4<br>g enc g enc g enc g enc g enc g enc g enc g enc<br>xt− 3 | xt− 2 | xt− 1 | xt | xt +1 | xt +2 | xt +3 | xt +4<br>**----- End of picture text -----**<br>


Figure 1: Overview of Contrastive Predictive Coding, the proposed representation learning approach. Although this figure shows audio as input, we use the same setup for images, text and reinforcement learning. 

shown useful [10, 11]. We hypothesize that these approaches are fruitful partly because the context from which we predict related values are often conditionally dependent on the same shared high-level latent information. And by casting this as a prediction problem, we automatically infer these features of interest to representation learning. 

In this paper we propose the following: first, we compress high-dimensional data into a much more compact latent embedding space in which conditional predictions are easier to model. Secondly, we use powerful autoregressive models in this latent space to make predictions many steps in the future. Finally, we rely on Noise-Contrastive Estimation [12] for the loss function in similar ways that have been used for learning word embeddings in natural language models, allowing for the whole model to be trained end-to-end. We apply the resulting model, Contrastive Predictive Coding (CPC) to widely different data modalities, images, speech, natural language and reinforcement learning, and show that the same mechanism learns interesting high-level information on each of these domains, outperforming other approaches. 

## **2 Contrastive Predicting Coding** 

We start this section by motivating and giving intuitions behind our approach. Next, we introduce the architecture of Contrastive Predictive Coding (CPC). After that we explain the loss function that is based on Noise-Contrastive Estimation. Lastly, we discuss related work to CPC. 

## **2.1 Motivation and Intuitions** 

The main intuition behind our model is to learn the representations that encode the underlying shared information between different parts of the (high-dimensional) signal. At the same time it discards low-level information and noise that is more local. In time series and high-dimensional modeling, approaches that use next step prediction exploit the local smoothness of the signal. When predicting further in the future, the amount of shared information becomes much lower, and the model needs to infer more global structure. These ’slow features’ [13] that span many time steps are often more interesting (e.g., phonemes and intonation in speech, objects in images, or the story line in books.). 

One of the challenges of predicting high-dimensional data is that unimodal losses such as meansquared error and cross-entropy are not very useful, and powerful conditional generative models which need to reconstruct every detail in the data are usually required. But these models are computationally intense, and waste capacity at modeling the complex relationships in the data _x_ , often ignoring the context _c_ . For example, images may contain thousands of bits of information while the high-level latent variables such as the class label contain much less information (10 bits for 1,024 categories). This suggests that modeling _p_ ( _x|c_ ) directly may not be optimal for the purpose of extracting shared information between _x_ and _c_ . When predicting future information we instead encode the target _x_ (future) and context _c_ (present) into a compact distributed vector representations (via non-linear 

2 

learned mappings) in a way that maximally preserves the mutual information of the original signals _x_ and _c_ 

**==> picture [265 x 28] intentionally omitted <==**

By maximizing the mutual information between the encoded representations (which is bounded by the MI between the input signals), we extract the underlying latent variables the inputs have in commmon. 

## **2.2 Contrastive Predictive Coding** 

Figure 1 shows the architecture of Contrastive Predictive Coding models. First, a non-linear encoder _g_ enc maps the input sequence of observations _xt_ to a sequence of latent representations _zt_ = _g_ enc( _xt_ ), potentially with a lower temporal resolution. Next, an autoregressive model _g_ ar summarizes all _z≤t_ in the latent space and produces a context latent representation _ct_ = _g_ ar( _z≤t_ ). 

As argued in the previous section we do not predict future observations _xt_ + _k_ directly with a generative model _pk_ ( _xt_ + _k|ct_ ). Instead we model a density ratio which preserves the mutual information between _xt_ + _k_ and _ct_ (Equation 1) as follows (see next sub-section for further details): 

**==> picture [254 x 25] intentionally omitted <==**

where _∝_ stands for ’proportional to’ (i.e. up to a multiplicative constant). Note that the density ratio _f_ can be unnormalized (does not have to integrate to 1). Although any positive real score can be used here, we use a simple log-bilinear model: 

**==> picture [267 x 19] intentionally omitted <==**

In our experiments a linear transformation _Wk[T][c][t]_[ is used for the prediction with a different] _[ W][k]_[for] every step _k_ . Alternatively, non-linear networks or recurrent neural networks could be used. 

By using a density ratio _f_ ( _xt_ + _k, ct_ ) and inferring _zt_ + _k_ with an encoder, we relieve the model from modeling the high dimensional distribution _xtk_ . Although we cannot evaluate _p_ ( _x_ ) or _p_ ( _x|c_ ) directly, we can use samples from these distributions, allowing us to use techniques such as Noise-Contrastive Estimation [12, 14, 15] and Importance Sampling [16] that are based on comparing the target value with randomly sampled negative values. 

In the proposed model, either of _zt_ and _ct_ could be used as representation for downstream tasks. The autoregressive model output _ct_ can be used if extra context from the past is useful. One such example is speech recognition, where the receptive field of _zt_ might not contain enough information to capture phonetic content. In other cases, where no additional context is required, _zt_ might instead be better. If the downstream task requires one representation for the whole sequence, as in e.g. image classification, one can pool the representations from either _zt_ or _ct_ over all locations. 

Finally, note that any type of encoder and autoregressive model can be used in the proposed framework. For simplicity we opted for standard architectures such as strided convolutional layers with resnet blocks for the encoder, and GRUs [17] for the autoregresssive model. More recent advancements in autoregressive modeling such as masked convolutional architectures [18, 19] or self-attention networks [20] could help improve results further. 

## **2.3 InfoNCE Loss and Mutual Information Estimation** 

Both the encoder and autoregressive model are trained to jointly optimize a loss based on NCE, which we will call InfoNCE. Given a set _X_ = _{x_ 1 _, . . . xN }_ of _N_ random samples containing one positive sample from _p_ ( _xt_ + _k|ct_ ) and _N −_ 1 negative samples from the ’proposal’ distribution _p_ ( _xt_ + _k_ ), we optimize: 

**==> picture [272 x 30] intentionally omitted <==**

3 

Optimizing this loss will result in _fk_ ( _xt_ + _k, ct_ ) estimating the density ratio in equation 2. This can be shown as follows. 

The loss in Equation 4 is the categorical cross-entropy of classifying the positive sample correctly, with � _fXk[f][k]_[being][the][prediction][of][the][model.][Let][us][write][the][optimal][probability][for][this][loss] as _p_ ( _d_ = _i|X, ct_ ) with [ _d_ = _i_ ] being the indicator that sample _xi_ is the ’positive’ sample. The probability that sample _xi_ was drawn from the conditional distribution _p_ ( _xt_ + _k|ct_ ) rather than the proposal distribution _p_ ( _xt_ + _k_ ) can be derived as follows: 

**==> picture [290 x 68] intentionally omitted <==**

As we can see, the optimal value for _f_ ( _xt_ + _k, ct_ ) in Equation 4 is proportional to _[p]_[(] _p[x]_ ( _[t] x_[+] _t_ + _[k][|] k[c]_ ) _[t]_[)] and this is independent of the the choice of the number of negative samples _N −_ 1. 

Though not required for training, we can evaluate the mutual information between the variables _ct_ and _xt_ + _k_ as follows: 

**==> picture [116 x 11] intentionally omitted <==**

which becomes tighter as N becomes larger. Also observe that minimizing the InfoNCE loss _L_ N maximizes a lower bound on mutual information. For more details see Appendix. 

## **2.4 Related Work** 

CPC is a new method that combines predicting future observations (predictive coding) with a probabilistic contrastive loss (Equation 4). This allows us to extract slow features, which maximize the mutual information of observations over long time horizons. Contrastive losses and predictive coding have individually been used in different ways before, which we will now discuss. 

Contrastive loss functions have been used by many authors in the past. For example, the techniques proposed by [21, 22, 23] were based on triplet losses using a max-margin approach to separate positive from negative examples. More recent work includes Time Contrastive Networks [24] which proposes to minimize distances between embeddings from multiple viewpoints of the same scene and whilst maximizing distances between embeddings extracted from different timesteps. In Time Contrastive Learning [25] a contrastive loss is used to predict the segment-ID of multivariate time-series as a way to extract features and perform nonlinear ICA. 

There has also been work and progress on defining prediction tasks from related observations as a way to extract useful representations, and many of these have been applied to language. In Word2Vec [9] neighbouring words are predicted using a contrastive loss. Skip-thought vectors [26] and Byte mLSTM [27] are alternatives which go beyond word prediction with a Recurrent Neural Network, and use maximum likelihood over sequences of observations. In Computer Vision [28] use a triplet loss on tracked video patches so that patches from the same object at different timesteps are more similar to each other than to random patches. [11, 29] propose to predict the relative postion of patches in an image and in [10] color values are predicted from a greyscale images. 

## **3 Experiments** 

We present benchmarks on four different application domains: speech, images, natural language and reinforcement learning. For every domain we train CPC models and probe what the representations contain with either a linear classification task or qualitative evaluations, and in reinforcement learning we measure how the auxiliary CPC loss speeds up learning of the agent. 

## **3.1 Audio** 

For audio, we use a 100-hour subset of the publicly available LibriSpeech dataset [30]. Although the dataset does not provide labels other than the raw text, we obtained force-aligned phone sequences 

4 

Figure 2: t-SNE visualization of audio (speech) representations for a subset of 10 speakers (out of 251). Every color represents a different speaker. 

Figure 3: Average accuracy of predicting the positive sample in the contrastive loss for 1 to 20 latent steps in the future of a speech waveform. The model predicts up to 200ms in the future as every step consists of 10ms of audio. 

|**Method**|**ACC**|
|---|---|
|**Phone classification**||
|Random initialization|27.6|
|MFCC features|39.7|
|CPC|64.6|
|Supervised|74.6|
|**Speaker classification**||
|Random initialization|1.87|
|MFCC features|17.6|
|CPC|97.4|
|Supervised|98.5|



Table 1: LibriSpeech phone and speaker classification results. For phone classification there are 41 possible classes and for speaker classification 251. All models used the same architecture and the same audio input sizes. 

|**#steps predicted**<br>2 steps<br>4 steps<br>8 steps<br>12 steps<br>16 steps<br>**Negative samples from**<br>Mixed speaker<br>Same speaker<br>Mixed speaker (excl.)<br>Same speaker (excl.)<br>Current sequence only|28.5<br>57.6<br>63.6<br>64.6<br>63.8<br>64.6<br>65.5<br>57.3<br>64.6<br>65.2|
|---|---|



Table 2: LibriSpeech phone classification ablation experiments. More details can be found in Section 3.1. 

with the Kaldi toolkit [31] and pre-trained models on Librispeech[1] . We have made the aligned phone labels and our train/test split available for download on Google Drive[2] . The dataset contains speech from 251 different speakers. 

The encoder architecture _genc_ used in our experiments consists of a strided convolutional neural network that runs directly on the 16KHz PCM audio waveform. We use five convolutional layers with strides [5, 4, 2, 2, 2], filter-sizes [10, 8, 4, 4, 4] and 512 hidden units with ReLU activations. The total downsampling factor of the network is 160 so that there is a feature vector for every 10ms of speech, which is also the rate of the phoneme sequence labels obtained with Kaldi. We then use a GRU RNN [17] for the autoregressive part of the model, _gar_ with 256 dimensional hidden state. The output of the GRU at every timestep is used as the context _c_ from which we predict 12 timesteps in the future using the contrastive loss. We train on sampled audio windows of length 20480. We use the Adam optimizer [32] with a learning rate of 2e-4, and use 8 GPUs each with a minibatch of 8 examples from which the negative samples in the contrastive loss are drawn. The model is trained until convergence, which happens roughly at 300,000 updates. 

Figure 3 shows the accuracy of the model to predict latents in the future, from 1 to 20 timesteps. We report the average number of times the logit for the positive sample is higher than for the negative samples in the probabilistic contrastive loss. This figure also shows that the objective is neither trivial nor impossible, and as expected the prediction task becomes harder as the target is further away. 

> 1www.kaldi-asr.org/downloads/build/6/trunk/egs/librispeech/ 

> 2https://drive.google.com/drive/folders/1BhJ2umKH3whguxMwifaKtSra0TgAbtfb 

5 

**==> picture [357 x 171] intentionally omitted <==**

**----- Start of picture text -----**<br>
g ar - output<br>g enc - output<br>c<br>= -- Deoc ten-- - | Pre t<br>— a7 —< | | ft ft Horm Le<br>64 px u--| agers-- 7 -27ardacees7 Z 2° ¢ _ -+-F-{T|¥ _ ---- eee,“TyZoey7/i<br>4 7 4 7 , zt +2 - _ Yoo:<br>Ded - 2Pa a 7 Y 7 zt +3 _— - . y, . Predictions<br>—Ey- zt +4 _—<br>50% overlap<br>256 px<br>input image<br>**----- End of picture text -----**<br>


Figure 4: Visualization of Contrastive Predictive Coding for images (2D adaptation of Figure 1). 

To understand the representations extracted by CPC, we measure the phone prediction performance with a linear classifier trained on top of these features, which shows how linearly separable the relevant classes are under these features. We extract the outputs of the GRU (256 dimensional), i.e. _ct_ , for the whole dataset after model convergence and train a multi-class linear logistic regression classifier. The results are shown in Table 1 (top). We compare the accuracy with three baselines: representations from a random initialized model (i.e., _genc_ and _gar_ are untrained), MFCC features, and a model that is trained end-to-end supervised with the labeled data. These two models have the same architecture as the one used to extract the CPC representations. The fully supervised model serves as an indication for what is achievable with this architecture. We also found that not all the information encoded is linearly accessible. When we used a single hidden layer instead the accuracy increases from **64.6** to **72.5** , which is closer to the accuracy of the fully supervised model. 

Table 2 gives an overview of two ablation studies of CPC for phone classification. In the first set we vary the number of steps the model predicts showing that predicting multiple steps is important for learning useful features. In the second set we compare different strategies for drawing negative sample, all predicting 12 steps (which gave the best result in the first ablation). In the mixed speaker experiment the negative samples contain examples of different speakers (first row), in contrast to same speaker experiment (second row). In the third and fourth experiment we exclude the current sequence to draw negative samples from (so only other examples in the minibatch are present in _X_ ) and in the last experiment we only draw negative samples within the sequence (thus all samples are from the same speaker). 

Beyond phone classification, Table 1 (bottom) shows the accuracy of performing speaker identity (out of 251) with a linear classifier from the same representation (we do not average utterances over time). Interestingly, CPCs capture both speaker identity and speech contents, as demonstrated by the good accuracies attained with a simple linear classifier, which also gets close to the oracle, fully supervised networks. 

Additionally, Figure 2 shows a t-SNE visualization [33] of how discriminative the embeddings are for speaker voice-characteristics. It is important to note that the window size (maximum context size for the GRU) has a big impact on the performance, and longer segments would give better results. Our model had a maximum of 20480 timesteps to process, which is slightly longer than a second. 

## **3.2 Vision** 

In our visual representation experiments we use the ILSVRC ImageNet competition dataset [34]. The ImageNet dataset has been used to evaluate unsupervised vision models by many authors [28, 11, 35, 10, 29, 36]. We follow the same setup as [36] and use a ResNet v2 101 architecture [37] as the image encoder _genc_ to extract CPC representations (note that this encoder is not pretrained). We did not use Batch-Norm [38]. After unsupervised training, a linear layer is trained to measure classification accuracy on ImageNet labels. 

6 

Figure 5: Every row shows image patches that activate a certain neuron in the CPC architecture. 

The training procedure is as follows: from a 256x256 image we extract a 7x7 grid of 64x64 crops with 32 pixels overlap. Simple data augmentation proved helpful on both the 256x256 images and the 64x64 crops. The 256x256 images are randomly cropped from a 300x300 image, horizontally flipped with a probability of 50% and converted to greyscale. For each of the 64x64 crops we randomly take a 60x60 subcrop and pad them back to a 64x64 image. 

Each crop is then encoded by the ResNet-v2-101 encoder. We use the outputs from the third residual block, and spatially mean-pool to get a single 1024-d vector per 64x64 patch. This results in a 7x7x1024 tensor. Next, we use a PixelCNN-style autoregressive model [19] (a convolutional rowGRU PixelRNN [39] gave similar results) to make predictions about the latent activations in following rows top-to-bottom, visualized in Figure 4. We predict up to five rows from the 7x7 grid, and we apply the contrastive loss for each patch in the row. We used Adam optimizer with a learning rate of 2e-4 and trained on 32 GPUs each with a batch size of 16. 

For the linear classifier trained on top of the CPC features we use SGD with a momentum of 0.9, a learning rate schedule of 0.1, 0.01 and 0.001 for 50k, 25k and 10k updates and batch size of 2048 on a single GPU. Note that when training the linear classifier we first spatially mean-pool the 7x7x1024 representation to a single 1024 dimensional vector. This is slightly different from [36] which uses a 3x3x1024 representation without pooling, and thus has more parameters in the supervised linear mapping (which could be advantageous). 

Tables 3 and 4 show the top-1 and top-5 classification accuracies compared with the state-of-the-art. Despite being relatively domain agnostic, CPCs improve upon state-of-the-art by 9% absolute in top-1 accuracy, and 4% absolute in top-5 accuracy. 

## **3.3 Natural Language** 

Our natural language experiments follow closely the procedure from [26] which was used for the skip-thought vectors model. We first learn our unsupervised model on the BookCorpus dataset [42], and evaluate the capability of our model as a generic feature extractor by using CPC representations for a set of classification tasks. To cope with words that are not seen during training, we employ vocabulary expansion the same way as [26], where a linear mapping is constructed between word2vec and the word embeddings learned by the model. 

For the classification tasks we used the following datasets: movie review sentiment (MR) [43], customer product reviews (CR) [44], subjectivity/objectivity [45], opinion polarity (MPQA) [46] and question-type classification (TREC) [47]. As in [26] we train a logistic regression classifier and 

7 

|**Method**|**Top-1 ACC**|||
|---|---|---|---|
|**Using AlexNet conv5**||||
|Video [28]|29.8|||
|Relative Position [11]|30.4|||
|BiGan [35]<br>Colorization [10]|34.8<br>35.2|**Method**|**Top-5 ACC**|
|Jigsaw [29] *|38.1|Motion Segmentation (MS)|48.3|
|**Using ResNet-V2**||Exemplar (Ex)|53.1|
|Motion Segmentation [36]|27.6|Relative Position (RP)|59.2|
|Exemplar [36]|31.5|Colorization (Col)|62.5|
|Relative Position [36]|36.2|Combination of||
|Colorization [36]|39.6|MS + Ex + RP + Col|69.3|
|**CPC**|**48.7**|**CPC**|**73.6**|



Table 3: ImageNet top-1 unsupervised classification results. *Jigsaw is not directly comparable to the other AlexNet results because of architectural differences. 

Table 4: ImageNet top-5 unsupervised classification results. Previous results with MS, Ex, RP and Col were taken from [36] and are the best reported results on this task. 

|**Method**<br>**MR**<br>**CR**<br>**Subj**<br>**MPQA**<br>**TREC**|**Method**<br>**MR**<br>**CR**<br>**Subj**<br>**MPQA**<br>**TREC**|**Method**<br>**MR**<br>**CR**<br>**Subj**<br>**MPQA**<br>**TREC**|**Method**<br>**MR**<br>**CR**<br>**Subj**<br>**MPQA**<br>**TREC**|**Method**<br>**MR**<br>**CR**<br>**Subj**<br>**MPQA**<br>**TREC**|**Method**<br>**MR**<br>**CR**<br>**Subj**<br>**MPQA**<br>**TREC**|
|---|---|---|---|---|---|
|||||||
|Paragraph-vector [40]<br>Skip-thought vector [26]<br>Skip-thought + LN [41]|74.8<br>75.5<br>79.5|78.1<br>79.3<br>82.6|90.5<br>92.1<br>93.4|74.2<br>86.9<br>89.0|91.8<br>91.4<br>-|
|CPC<br>76.9<br>80.1<br>91.2<br>87.7<br>96.8||||||



Table 5: Classification accuracy on five common NLP benchmarks. We follow the same transfer learning setup from Skip-thought vectors [26] and use the BookCorpus dataset as source. [40] is an unsupervised approach to learning sentence-level representations. [26] is an alternative unsupervised learning approach. [41] is the same skip-thought model with layer normalization trained for 1M iterations. 

evaluate with 10-fold cross-validation for MR, CR, Subj, MPQA and use the train/test split for TREC. A L2 regularization weight was chosen via cross-validation (therefore nested cross-validation for the first 4 datasets). 

Our model consists of a simple sentence encoder _genc_ (a 1D-convolution + ReLU + mean-pooling) that embeds a whole sentence into a 2400-dimension vector _z_ , followed by a GRU (2400 hidden units) which predicts up to 3 future sentence embeddings with the contrastive loss to form _c_ . We used Adam optimizer with a learning rate of 2e-4 trained on 8 GPUs, each with a batch size of 64. We found that more advanced sentence encoders did not significantly improve the results, which may be due to the simplicity of the transfer tasks (e.g., in MPQA most datapoints consists of one or a few words), and the fact that bag-of-words models usually perform well on many NLP tasks [48]. 

Results on evaluation tasks are shown in Table 5 where we compare our model against other models that have been used using the same datasets. The performance of our method is very similar to the skip-thought vector model, with the advantage that it does not require a powerful LSTM as word-level decoder, therefore much faster to train. Although this is a standard transfer learning benchmark, we found that models that learn better relationships in the childeren books did not necessarily perform better on the target tasks (which are very different: movie reviews etc). We note that better [49, 27] results have been published on these target datasets, by transfer learning from a different source task. 

8 

Figure 6: Reinforcement Learning results for 5 DeepMind Lab tasks used in [50]. Black: batched A2C baseline, Red: with auxiliary contrastive loss. 

## **3.4 Reinforcement Learning** 

Finally, we evaluate the proposed unsupervised learning approach on five reinforcement learning in 3D environments of DeepMind Lab [51]: rooms_watermaze, explore_goal_locations_small, seekavoid_arena_01, lasertag_three_opponents_small and rooms_keys_doors_puzzle. 

This setup differs from the previous three. Here, we take the standard batched A2C [52] agent as base model and add CPC as an auxiliary loss. We do not use a replay buffer, so the predictions have to adapt to the changing behavior of the policy. The learned representation encodes a distribution over its future observations. 

Following the same approach as [50], we perform a random search over the entropy regularization weight, the learning-rate and epsilon hyperparameters for RMSProp [53]. The unroll length for the A2C is 100 steps and we predict up to 30 steps in the future to derive the contrastive loss. The baseline agent consists of a convolutional encoder which maps every input frame into a single vector followed by a temporal LSTM. We use the same encoder as in the baseline agent and only add the linear prediction mappings for the contrastive loss, resulting in minimal overhead which also showcases the simplicity of implementing our method on top of an existing architecture that has been designed and tuned for a particular task. We refer to [50] for all other hyperparameter and implementation details. 

Figure 6 shows that for 4 out of the 5 games performance of the agent improves significantly with the contrastive loss after training on 1 billion frames. For lasertag_three_opponents_small, contrastive loss does not help nor hurt. We suspect that this is due to the task design, which does not require memory and thus yields a purely reactive policy. 

## **4 Conclusion** 

In this paper we presented Contrastive Predictive Coding (CPC), a framework for extracting compact latent representations to encode predictions over future observations. CPC combines autoregressive modeling and noise-contrastive estimation with intuitions from predictive coding to learn abstract representations in an unsupervised fashion. We tested these representations in a wide variety of domains: audio, images, natural language and reinforcement learning and achieve strong or stateof-the-art performance when used as stand-alone features. The simplicity and low computational requirements to train the model, together with the encouraging results in challenging reinforcement learning domains when used in conjunction with the main loss are exciting developments towards useful unsupervised learning that applies universally to many more data modalities. 

## **5 Acknowledgements** 

We would like to thank Andriy Mnih, Andrew Zisserman, Alex Graves and Carl Doersch for their helpful comments on the paper and Lasse Espeholt for making the A2C baseline available. 

## **References** 

- [1] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In _Advances in neural information processing systems_ , pages 1097–1105, 2012. 

9 

- [2] Geoffrey Hinton, Li Deng, Dong Yu, George E Dahl, Abdel-rahman Mohamed, Navdeep Jaitly, Andrew Senior, Vincent Vanhoucke, Patrick Nguyen, Tara N Sainath, et al. Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups. _IEEE Signal Processing Magazine_ , 29(6):82–97, 2012. 

- [3] Ilya Sutskever, Oriol Vinyals, and Quoc V Le. Sequence to sequence learning with neural networks. In _Advances in neural information processing systems_ , pages 3104–3112, 2014. 

- [4] Oriol Vinyals, Alexander Toshev, Samy Bengio, and Dumitru Erhan. Show and tell: A neural image caption generator. In _Computer Vision and Pattern Recognition (CVPR), 2015 IEEE Conference on_ , pages 3156–3164. IEEE, 2015. 

- [5] Peter Elias. Predictive coding–i. _IRE Transactions on Information Theory_ , 1(1):16–24, 1955. 

- [6] Bishnu S Atal and Manfred R Schroeder. Adaptive predictive coding of speech signals. _The Bell System Technical Journal_ , 49(8):1973–1986, 1970. 

- [7] Rajesh PN Rao and Dana H Ballard. Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. _Nature neuroscience_ , 2(1):79, 1999. 

- [8] Karl Friston. A theory of cortical responses. _Philosophical transactions of the Royal Society B: Biological sciences_ , 360(1456):815–836, 2005. 

- [9] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. _arXiv preprint arXiv:1301.3781_ , 2013. 

- [10] Richard Zhang, Phillip Isola, and Alexei A Efros. Colorful image colorization. In _European Conference on Computer Vision_ , pages 649–666. Springer, 2016. 

- [11] Carl Doersch, Abhinav Gupta, and Alexei A. Efros. Unsupervised visual representation learning by context prediction. In _The IEEE International Conference on Computer Vision (ICCV)_ , December 2015. 

- [12] Michael Gutmann and Aapo Hyvärinen. Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. In _Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics_ , pages 297–304, 2010. 

- [13] Laurenz Wiskott and Terrence J Sejnowski. Slow feature analysis: Unsupervised learning of invariances. _Neural computation_ , 14(4):715–770, 2002. 

- [14] Andriy Mnih and Yee Whye Teh. A fast and simple algorithm for training neural probabilistic language models. _arXiv preprint arXiv:1206.6426_ , 2012. 

- [15] Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu. Exploring the limits of language modeling. _arXiv preprint arXiv:1602.02410_ , 2016. 

- [16] Yoshua Bengio and Jean-Sebastien Senecal. Adaptive importance sampling to accelerate training of a neural probabilistic language model. _IEEE Trans. Neural Networks_ , 19, 2008. 

- [17] Kyunghyun Cho, Bart Van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoderdecoder for statistical machine translation. _arXiv preprint arXiv:1406.1078_ , 2014. 

- [18] Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, and Koray Kavukcuoglu. Wavenet: A generative model for raw audio. _arXiv preprint arXiv:1609.03499_ , 2016. 

- [19] Aaron van den Oord, Nal Kalchbrenner, Oriol Vinyals, Lasse Espeholt, Alex Graves, and Koray Kavukcuoglu. Conditional image generation with pixelcnn decoders. _CoRR_ , abs/1606.05328, 2016. 

- [20] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, _Advances in Neural Information Processing Systems 30_ , pages 5998–6008. Curran Associates, Inc., 2017. 

- [21] Sumit Chopra, Raia Hadsell, and Yann LeCun. Learning a similarity metric discriminatively, with application to face verification. In _Computer Vision and Pattern Recognition, 2005. CVPR 2005. IEEE Computer Society Conference on_ , volume 1, pages 539–546. IEEE, 2005. 

- [22] Kilian Q Weinberger and Lawrence K Saul. Distance metric learning for large margin nearest neighbor classification. _Journal of Machine Learning Research_ , 10(Feb):207–244, 2009. 

10 

- [23] Florian Schroff, Dmitry Kalenichenko, and James Philbin. Facenet: A unified embedding for face recognition and clustering. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 815–823, 2015. 

- [24] Pierre Sermanet, Corey Lynch, Jasmine Hsu, and Sergey Levine. Time-contrastive networks: Self-supervised learning from multi-view observation. _arXiv preprint arXiv:1704.06888_ , 2017. 

- [25] Aapo Hyvarinen and Hiroshi Morioka. Unsupervised feature extraction by time-contrastive learning and nonlinear ica. In D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon, and R. Garnett, editors, _Advances in Neural Information Processing Systems 29_ , pages 3765–3773. Curran Associates, Inc., 2016. 

- [26] Ryan Kiros, Yukun Zhu, Ruslan R Salakhutdinov, Richard Zemel, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. Skip-thought vectors. In _Advances in neural information processing systems_ , pages 3294–3302, 2015. 

- [27] Alec Radford, Rafal Jozefowicz, and Ilya Sutskever. Learning to generate reviews and discovering sentiment. _arXiv preprint arXiv:1704.01444_ , 2017. 

- [28] Xiaolong Wang and Abhinav Gupta. Unsupervised learning of visual representations using videos. _arXiv preprint arXiv:1505.00687_ , 2015. 

- [29] Mehdi Noroozi and Paolo Favaro. Unsupervised learning of visual representations by solving jigsaw puzzles. In _European Conference on Computer Vision_ , pages 69–84. Springer, 2016. 

- [30] Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Librispeech: an asr corpus based on public domain audio books. In _Acoustics, Speech and Signal Processing (ICASSP), 2015 IEEE International Conference on_ , pages 5206–5210. IEEE, 2015. 

- [31] Daniel Povey, Arnab Ghoshal, Gilles Boulianne, Lukas Burget, Ondrej Glembek, Nagendra Goel, Mirko Hannemann, Petr Motlicek, Yanmin Qian, Petr Schwarz, et al. The kaldi speech recognition toolkit. In _IEEE 2011 workshop on automatic speech recognition and understanding_ . IEEE Signal Processing Society, 2011. 

- [32] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_ , 2014. 

- [33] Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. _Journal of machine learning research_ , 9(Nov):2579–2605, 2008. 

- [34] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. _International Journal of Computer Vision (IJCV)_ , 115(3):211–252, 2015. 

- [35] Jeff Donahue, Philipp Krähenbühl, and Trevor Darrell. Adversarial feature learning. _arXiv preprint arXiv:1605.09782_ , 2016. 

- [36] Carl Doersch and Andrew Zisserman. Multi-task self-supervised visual learning. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 2051–2060, 2017. 

- [37] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Identity mappings in deep residual networks. In _European Conference on Computer Vision_ , pages 630–645. Springer, 2016. 

- [38] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In _International conference on machine learning_ , pages 448–456, 2015. 

- [39] Aaron van den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel recurrent neural networks. _CoRR_ , abs/1601.06759, 2016. 

- [40] Quoc Le and Tomas Mikolov. Distributed representations of sentences and documents. In _International Conference on Machine Learning_ , pages 1188–1196, 2014. 

- [41] Lei Jimmy Ba, Ryan Kiros, and Geoffrey E Hinton. Layer normalization. _CoRR_ , abs/1607.06450, 2016. 

- [42] Yukun Zhu, Ryan Kiros, Rich Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. In _Proceedings of the IEEE international conference on computer vision_ , pages 19–27, 2015. 

11 

- [43] Bo Pang and Lillian Lee. Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales. In _Proceedings of the 43rd annual meeting on association for computational linguistics_ , pages 115–124. Association for Computational Linguistics, 2005. 

- [44] Minqing Hu and Bing Liu. Mining and summarizing customer reviews. In _Proceedings of the tenth ACM SIGKDD international conference on Knowledge discovery and data mining_ , pages 168–177. ACM, 2004. 

- [45] Bo Pang and Lillian Lee. A sentimental education: Sentiment analysis using subjectivity summarization based on minimum cuts. In _Proceedings of the 42nd annual meeting on Association for Computational Linguistics_ , page 271. Association for Computational Linguistics, 2004. 

- [46] Janyce Wiebe, Theresa Wilson, and Claire Cardie. Annotating expressions of opinions and emotions in language. _Language resources and evaluation_ , 39(2-3):165–210, 2005. 

- [47] Xin Li and Dan Roth. Learning question classifiers. In _Proceedings of the 19th international conference on Computational linguistics-Volume 1_ , pages 1–7. Association for Computational Linguistics, 2002. 

- [48] Sida Wang and Christopher D. Manning. Baselines and bigrams: Simple, good sentiment and topic classification. In _Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics: Short Papers - Volume 2_ , ACL ’12, pages 90–94, 2012. 

- [49] Han Zhao, Zhengdong Lu, and Pascal Poupart. Self-adaptive hierarchical sentence model. In _IJCAI_ , pages 4069–4076, 2015. 

- [50] Lasse Espeholt, Hubert Soyer, Remi Munos, Karen Simonyan, Volodymir Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, Shane Legg, and Koray Kavukcuoglu. Impala: Scalable distributed deep-rl with importance weighted actor-learner architectures. _arXiv preprint arXiv:1802.01561_ , 2018. 

- [51] Charles Beattie, Joel Z Leibo, Denis Teplyashin, Tom Ward, Marcus Wainwright, Heinrich Küttler, Andrew Lefrancq, Simon Green, Víctor Valdés, Amir Sadik, et al. Deepmind lab. _arXiv preprint arXiv:1612.03801_ , 2016. 

- [52] Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In _International Conference on Machine Learning_ , pages 1928–1937, 2016. 

- [53] Geoffrey Hinton, Nitish Srivastava, and Kevin Swersky. Neural networks for machine learninglecture 6a-overview of mini-batch gradient descent. 

- [54] Ishmael Belghazi, Sai Rajeswar, Aristide Baratin, R Devon Hjelm, and Aaron Courville. Mine: Mutual information neural estimation. 2018. 

12 

## **A Appendix** 

## **A.1 Estimating the Mutual Information with InfoNCE** 

By optimizing InfoNCE, the CPC loss we defined in Equation 4, we are maximizing the mutual information between _ct_ and _zt_ + _k_ (which is bounded by the MI between _ct_ and _xt_ + _k_ ). This can be shown as follows. 

As already shown in Section 2.3, the optimal value for _f_ ( _xt_ + _k, ct_ ) is given by _[p]_[(] _p[x]_ ( _[t] x_[+] _t_ + _[k][|] k[c]_ ) _[t]_[)][.][Inserting] this back in to Equation 4 and splitting _X_ into the positive example and the negative examples _X_ neg results in: 

**==> picture [303 x 160] intentionally omitted <==**

**==> picture [283 x 11] intentionally omitted <==**

Therefore, _I_ ( _xt_ + _k, ct_ ) _≥_ log( _N_ ) _−L_[opt] N[.][This trivially also holds for other] _[ f]_[that obtain a worse] (higher) _L_ N. Equation 8 quickly becomes more accurate as _N_ increases. At the same time log( _N_ ) _− L_ N also increases, so it’s useful to use large values of _N_ . 

InfoNCE is also related to MINE [54]. Without loss of generality let’s write _f_ ( _x, c_ ) = _e[F]_[ (] _[x,c]_[)] , then 

**==> picture [384 x 137] intentionally omitted <==**

is equivalent to the MINE estimator (up to a constant). So we maximize a lower bound on this estimator. We found that using MINE directly gave identical performance when the task was nontrivial, but became very unstable if the target was easy to predict from the context (e.g., when predicting a single step in the future and the target overlaps with the context). 

13 

