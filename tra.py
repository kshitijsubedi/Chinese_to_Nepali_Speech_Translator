# -*- coding: utf-8 -*-

import tensorflow as tf
#tf.enable_eager_execution()

import pandas as pd

import numpy as np
import unicodedata
import re
import os
import requests
from zipfile import ZipFile
import time


dat=np.load('ch_np.npz')
data = dat['arr_0']


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def normalize_string(s):
    s = unicode_to_ascii(s)
    s = re.sub(r'([!.?])', r' \1', s)
    s = re.sub(r'\s+', r' ', s)
    return s

"""## Preprocessing"""
raw_data_en=data[:,0]
raw_data_fr=data[:,1]


#raw_data_en, raw_data_fr = list(zip(*raw_data))
raw_data_en = [normalize_string(data) for data in raw_data_en]
raw_data_fr_in = ['<start> ' + normalize_string(data) for data in raw_data_fr]
raw_data_fr_out = [normalize_string(data) + ' <end>' for data in raw_data_fr]

"""## Tokenization"""

en_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
en_tokenizer.fit_on_texts(raw_data_en)
data_en = en_tokenizer.texts_to_sequences(raw_data_en)
data_en = tf.keras.preprocessing.sequence.pad_sequences(data_en,
                                                        padding='post')

fr_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
fr_tokenizer.fit_on_texts(raw_data_fr_in)
fr_tokenizer.fit_on_texts(raw_data_fr_out)
data_fr_in = fr_tokenizer.texts_to_sequences(raw_data_fr_in)
data_fr_in = tf.keras.preprocessing.sequence.pad_sequences(data_fr_in,
                                                           padding='post')

data_fr_out = fr_tokenizer.texts_to_sequences(raw_data_fr_out)
data_fr_out = tf.keras.preprocessing.sequence.pad_sequences(data_fr_out,
                                                            padding='post')

"""## Create tf.data.Dataset object"""

BATCH_SIZE = 64
dataset = tf.data.Dataset.from_tensor_slices(
    (data_en, data_fr_in, data_fr_out))
dataset = dataset.shuffle(len(data_en)).batch(BATCH_SIZE)

def positional_encoding(pos, model_size):

    PE = np.zeros((1, model_size))
    for i in range(model_size):
        if i % 2 == 0:
            PE[:, i] = np.sin(pos / 10000 ** (i / model_size))
        else:
            PE[:, i] = np.cos(pos / 10000 ** ((i - 1) / model_size))
    return PE
MODEL_SIZE = 128
max_length=128
pes = []
for i in range(max_length):
    pes.append(positional_encoding(i, MODEL_SIZE))

pes = np.concatenate(pes, axis=0)
pes = tf.constant(pes, dtype=tf.float32)

"""### Multi-Head Attention layer"""

"""## Create the Multihead Attention layer"""


class MultiHeadAttention(tf.keras.Model):

    def __init__(self, model_size, h):
        super(MultiHeadAttention, self).__init__()
        self.key_size = model_size // h
        self.h = h
        self.wq = tf.keras.layers.Dense(model_size) #[tf.keras.layers.Dense(key_size) for _ in range(h)]
        self.wk = tf.keras.layers.Dense(model_size) #[tf.keras.layers.Dense(key_size) for _ in range(h)]
        self.wv = tf.keras.layers.Dense(model_size) #[tf.keras.layers.Dense(value_size) for _ in range(h)]
        self.wo = tf.keras.layers.Dense(model_size)

    def call(self, query, value, mask=None):
    
        query = self.wq(query)
        key = self.wk(value)
        value = self.wv(value)
        
        # Split matrices for multi-heads attention
        batch_size = query.shape[0]
        
        # Originally, query has shape (batch, query_len, model_size)
        # We need to reshape to (batch, query_len, h, key_size)
        query = tf.reshape(query, [batch_size, -1, self.h, self.key_size])
        # In order to compute matmul, the dimensions must be transposed to (batch, h, query_len, key_size)
        query = tf.transpose(query, [0, 2, 1, 3])
        
        # Do the same for key and value
        key = tf.reshape(key, [batch_size, -1, self.h, self.key_size])
        key = tf.transpose(key, [0, 2, 1, 3])
        value = tf.reshape(value, [batch_size, -1, self.h, self.key_size])
        value = tf.transpose(value, [0, 2, 1, 3])
        
        
        score = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.dtypes.cast(self.key_size, dtype=tf.float32))
        
        if mask is not None:
            score *= mask
            score = tf.where(tf.equal(score, 0), tf.ones_like(score) * -1e9, score)
        
        # Alignment vector: (batch, h, query_len, value_len)
        alignment = tf.nn.softmax(score, axis=-1)
        
        # Context vector: (batch, h, query_len, key_size)
        context = tf.matmul(alignment, value)
        
        # Finally, do the opposite to have a tensor of shape (batch, query_len, model_size)
        context = tf.transpose(context, [0, 2, 1, 3])
        context = tf.reshape(context, [batch_size, -1, self.key_size * self.h])
        
        # Apply one last full connected layer (WO)
        heads = self.wo(context)
        
        return heads, alignment

"""### The Encoder"""

"""## Create the Encoder"""


class Encoder(tf.keras.Model):
    
    def __init__(self, vocab_size, model_size, num_layers, h):
        super(Encoder, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_size)
        self.embedding_dropout = tf.keras.layers.Dropout(0.1)
        self.attention = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_dropout = [tf.keras.layers.Dropout(0.1) for _ in range(num_layers)]

        self.attention_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]

        self.dense_1 = [tf.keras.layers.Dense(
            MODEL_SIZE * 4, activation='relu') for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(
            model_size) for _ in range(num_layers)]
        self.ffn_dropout = [tf.keras.layers.Dropout(0.1) for _ in range(num_layers)]
        self.ffn_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]

    def call(self, sequence, training=True, encoder_mask=None):
        
        embed_out = self.embedding(sequence)

        embed_out *= tf.math.sqrt(tf.cast(self.model_size, tf.float32))
        embed_out += pes[:sequence.shape[1], :]
        embed_out = self.embedding_dropout(embed_out)

        sub_in = embed_out
        alignments = []

        for i in range(self.num_layers):
            sub_out, alignment = self.attention[i](sub_in, sub_in, encoder_mask)
            sub_out = self.attention_dropout[i](sub_out, training=training)
            sub_out = sub_in + sub_out
            sub_out = self.attention_norm[i](sub_out)
            
            alignments.append(alignment)
            ffn_in = sub_out

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            ffn_out = self.ffn_dropout[i](ffn_out, training=training)
            ffn_out = ffn_in + ffn_out
            ffn_out = self.ffn_norm[i](ffn_out)

            sub_in = ffn_out

        return ffn_out, alignments


"""### The Decoder"""

class Decoder(tf.keras.Model):

    def __init__(self, vocab_size, model_size, num_layers, h):
        super(Decoder, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_size)
        self.embedding_dropout = tf.keras.layers.Dropout(0.1)
        self.attention_bot = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_bot_dropout = [tf.keras.layers.Dropout(0.1) for _ in range(num_layers)]
        self.attention_bot_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]
        self.attention_mid = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_mid_dropout = [tf.keras.layers.Dropout(0.1) for _ in range(num_layers)]
        self.attention_mid_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]

        self.dense_1 = [tf.keras.layers.Dense(
            MODEL_SIZE * 4, activation='relu') for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(
            model_size) for _ in range(num_layers)]
        self.ffn_dropout = [tf.keras.layers.Dropout(0.1) for _ in range(num_layers)]
        self.ffn_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]

        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, sequence, encoder_output, training=True, encoder_mask=None):

        # EMBEDDING AND POSITIONAL EMBEDDING
        embed_out = self.embedding(sequence)

        embed_out *= tf.math.sqrt(tf.cast(self.model_size, tf.float32))
        embed_out += pes[:sequence.shape[1], :]
        embed_out = self.embedding_dropout(embed_out)

        bot_sub_in = embed_out
        bot_alignments = []
        mid_alignments = []

        for i in range(self.num_layers):
            # BOTTOM MULTIHEAD SUB LAYER
            seq_len = bot_sub_in.shape[1]

            if training:
                mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
            else:
                mask = None
            bot_sub_out, bot_alignment = self.attention_bot[i](bot_sub_in, bot_sub_in, mask)
            bot_sub_out = self.attention_bot_dropout[i](bot_sub_out, training=training)
            bot_sub_out = bot_sub_in + bot_sub_out
            bot_sub_out = self.attention_bot_norm[i](bot_sub_out)
            
            bot_alignments.append(bot_alignment)

            # MIDDLE MULTIHEAD SUB LAYER
            mid_sub_in = bot_sub_out

            mid_sub_out, mid_alignment = self.attention_mid[i](
                mid_sub_in, encoder_output, encoder_mask)
            mid_sub_out = self.attention_mid_dropout[i](mid_sub_out, training=training)
            mid_sub_out = mid_sub_out + mid_sub_in
            mid_sub_out = self.attention_mid_norm[i](mid_sub_out)
            
            mid_alignments.append(mid_alignment)

            # FFN
            ffn_in = mid_sub_out

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            ffn_out = self.ffn_dropout[i](ffn_out, training=training)
            ffn_out = ffn_out + ffn_in
            ffn_out = self.ffn_norm[i](ffn_out)

            bot_sub_in = ffn_out

        logits = self.dense(ffn_out)

        return logits, bot_alignments, mid_alignments

"""### Loss function"""

crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True)


def loss_func(targets, logits):
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    mask = tf.cast(mask, dtype=tf.int64)
    loss = crossentropy(targets, logits, sample_weight=mask)

    return loss

"""### Learning rate scheduling and optimization"""

class WarmupThenDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    
    def __init__(self, model_size, warmup_steps=4000):
        super(WarmupThenDecaySchedule, self).__init__()

        self.model_size = model_size
        self.model_size = tf.cast(self.model_size, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step_term = tf.math.rsqrt(step)
        warmup_term = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.model_size) * tf.math.minimum(step_term, warmup_term)
H = 8
NUM_LAYERS = 4
MODEL_SIZE=128

encoder = Encoder(3179, MODEL_SIZE, NUM_LAYERS, H)
sequence_in = tf.constant([[1, 2, 3, 0, 0]])
encoder_output, _ = encoder(sequence_in)
print(encoder_output.shape)
decoder = Decoder(44675, MODEL_SIZE, NUM_LAYERS, H)
sequence_in = tf.constant([[14, 24, 36, 0, 0]])
decoder_output, _, _ = decoder(sequence_in, encoder_output)
print(decoder_output.shape)
"""### The predict function"""

def predict(test_source_text=None):
    
    if test_source_text is None:
        test_source_text = raw_data_en[np.random.choice(len(raw_data_en))]
    print(test_source_text)
    test_source_seq = en_tokenizer.texts_to_sequences([test_source_text])
    print(test_source_seq)

    en_output, en_alignments = encoder(tf.constant(test_source_seq), training=False)

    de_input = tf.constant(
        [[fr_tokenizer.word_index['<start>']]], dtype=tf.int64)

    out_words = []

    while True:
        de_output, de_bot_alignments, de_mid_alignments = decoder(de_input, en_output, training=False)
        new_word = tf.expand_dims(tf.argmax(de_output, -1)[:, -1], axis=1)
        out_words.append(fr_tokenizer.index_word[new_word.numpy()[0][0]])

       
        de_input = tf.concat((de_input, new_word), axis=-1)

        
        if out_words[-1] == '<end>' :#or len(out_words) >= 14:
            break

    print(' '.join(out_words))
    return en_alignments, de_bot_alignments, de_mid_alignments, test_source_text.split(' '), out_words

"""### The train_step function"""


epochnumber=301  #change this

encoder.load_weights("cp/encoder_epoch_{}.h5".format(epochnumber))
decoder.load_weights("cp/decoder_epoch_{}.h5".format(epochnumber))
epoch_start = epochnumber
print(epoch_start)


#test_sents = (
    #"那 時  耶 穌 對 他 們 說  今 夜 你 們 為 我 的 緣 故  都 要 跌 倒  因 為 經 上 記 著 說  『 我 要 擊 打 牧 人  羊 就 分 散 了", 
#"但 你 們 不 可 這 樣  你 們 裡 頭 為 大 的  倒 要 像 年 幼 的  為 首 領 的  倒 要 像 服 事 人 的", 
#"與 我 說 話 的 天 使 去 的 時 候  又 有 一 位 天 使 迎 著 他 來",
#"以 色 列 王 出 城 攻 打 車 馬  大 大 擊 殺 亞 蘭 人 ",
#"故 殺 人 犯 死 罪 的  你 們 不 可 收 贖 價 代 替 他 的 命 他 必 被 治 死", 
#"亞 伯 蘭 俯 伏 在 地   神 又 對 他 說 ",
#"向 來 你 們 沒 有 奉 我 的 名 求 甚 麼  如 今 你 們 求 就 必 得 著  叫 你 們 的 喜 樂 可 以 滿 足 ",
#"按 作 香 之 法  調 和 作 成 聖 膏 油 ",
#"我 們 要 起 來  上 伯 特 利 去  在 那 裡 我 要 築 一 座 壇 給  神  就 是 在 我 遭 難 的 日 子  應 允 我 的 禱 告  在 我 行 的 路 上 保 佑 我 的 那 位 ",
#)

#for i, test_sent in enumerate(test_sents):
    #test_sequence = normalize_string(test_sent)
    #predict(test_sequence)

import speech_recognition as sr


def chinese_speech2text(duration = 3, language = 'zh-CN'):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.record(source, duration = duration)
        audio = r.listen(source)
    
    return r.recognize_sphinx(audio, language = language)

xx=chinese_speech2text()
## custom text halda ni hunxa microphone use nagarni vaye
#xx ="我 們 要 起 來"
hello=normalize_string(xx)
final_np=predict(hello)
##aba final lai tacotron ma input dini
