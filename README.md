# Chinese To Nepali Speech Translator

We present a simple application which can translate Chinese Speech to Nepali Speech offline. For this we are using Machine Learning Approach.

# Basic WorkFlow

### 1. Recognition of Chinese Speech
At first few seconds sample of Chinese speech is taken . For this propose [CMU Sphinx](http://cmusphinx.sourceforge.net) Open Source Toolkit For Speech Recognition is used.

> python cn_asr.py

### 2. Chinese Text to Nepali Text Translation

After Chinese text is generated from the speech it proceeds to the translation to Nepalese Text. For the translation we train our Ch-Np Dataset 
 - Bible corpus dataset is used for training



 To train the model and predict
 > python scratch_transformer.py
 
### Transformer Architecture

The  [transformer model](https://arxiv.org/pdf/1706.03762.pdf)  introduces an architecture that is solely based on attention mechanism and does not use any Recurrent Networks but yet produces results superior in quality to Seq2Seq models. It addresses the long term dependency problem of the Seq2Seq model. The transformer architecture is also parallelizable and the training process is considerably faster.
![enter image description here](https://lh5.googleusercontent.com/2CPb5BSXNmw3Kyy8ge9JcJZJP0rm2udOX9yjqJcJZmqcAZhn6MV217jL0Vk3oIqzUE4bXGNY14hLk0jMNT5ICdEEZu4bXXWOSXAE2o-05cAjHYzfOP6xE4Af20hm_Szp7mHhbbFc)
**Encoder**: The encoder has 6 identical layers in which each layer consists of a multi-head self-attention mechanism and a fully connected feed-forward network. The multi-head attention system and feed-forward network both have a residual connection and a normalization layer.

**Decoder**: The decoder also consists of 6 identical layers with an additional sublayer in each of the 6 layers. The additional sublayer performs multi-head attention over the output of the encoder stack.

**Attention Mechanism**:

Attention is the mapping of a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The attention mechanism allows the model to understand the context of a text.

-   Scaled Dot-Product Attention:

![](https://lh5.googleusercontent.com/C_WBeWIb7ZHZ9Efm-ee3u4PZ3-Eykowt2S0fiTom1GLKUsVvSTTSaYdMiRXvg5BQ_Jy7Rnv0zMXh5bKIZ34v6RxeTxOB8Kd9M21ReaPkgAQeg8qsOYZKyTNoSjGsMMWPvj8X8ZtB)

-   Multi-Head Attention:

![](https://lh6.googleusercontent.com/K_T6OFaDyWNN5Cq6Q2M_Um3VJ9B0W2QuaUTvcn8jj220t3qtg8IFTh1RidblbGtleGxQEwb9NWVJ4jyW5iTIS79mFNqhN4WYDufZ6NY6MrDbNsZ16OWqY2uya7CLC4YY2gkq5dCM)


### 3. Nepali Text to Speech

Nepali text generated from transformer model is converted to speech.
Tacotron 2 model is used for making TTS system.
Tacotron 2, a neural network architecture for speech synthesis directly from text. The system is composed of a recurrent sequence-to-sequence feature prediction network that maps character embeddings to mel-scale spectrograms, followed by a modified WaveNet model acting as a vocoder to synthesize timedomain waveforms from those spectrograms. This model achieves a mean opinion score (MOS) of 4.53 comparable to a MOS of 4.58 for professionally recorded speech. To validate our design choices, we present ablation studies of key components of our system and evaluate the impact of using mel spectrograms as the input to WaveNet instead of linguistic, duration, and F0 features. We further demonstrate that using a compact acoustic intermediate representation enables significant simplification of the WaveNet architecture.

![](https://3.bp.blogspot.com/-bjFYjr2Po2U/WjlNgrInWZI/AAAAAAAACSQ/tfdMAidI8O8EULlJgYoqRWWE9UGIENAkgCLcBGAs/s640/image1.png)

