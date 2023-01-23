import tensorflow as tf

from config import Config
from core import Module

# region Config

config = Config.config()

SOS = config["sos"]
EOS = config["eos"]
UNK = config["unk"]

# endregion


class Encoder(tf.keras.layers.Layer):
    def __init__(self, text_processor, emb_dim, units):
        super(Encoder, self).__init__()
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()

        self.embedding = tf.keras.layers.Embedding(
            self.vocab_size, emb_dim, mask_zero=True
        )

        self.rnn = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                units,
                return_sequences=True,
                kernel_initializer="orthogonal",
                dropout=0.2,
            ),
            "sum",
        )

    def call(self, x):
        shape_checker = Module.ShapeChecker()
        shape_checker(x, "batch s")

        x = self.embedding(x)
        shape_checker(x, "batch s units")

        x = self.rnn(x)
        # shape_checker(x, "batch s units")

        return x

    def convert_input(self, texts):
        texts = tf.convert_to_tensor(texts)
        if len(texts.shape) == 0:
            texts = tf.convert_to_tensor(texts)[tf.newaxis]
        context = self.text_processor(texts)
        context = self(context)
        return context


class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(4, units, **kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, x, context):
        shape_checker = Module.ShapeChecker()

        shape_checker(x, "batch t units")
        shape_checker(context, "batch s units")

        attn_output, attn_scores = self.mha(x, context, return_attention_scores=True)

        shape_checker(x, "batch t units")
        shape_checker(attn_scores, "batch heads t s")

        attn_scores = tf.reduce_mean(attn_scores, axis=1)
        shape_checker(attn_scores, "batch t s")
        self.last_attention_weights = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


class Decoder(tf.keras.layers.Layer):
    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun

    def __init__(self, text_processor, emb_dim, units):
        super(Decoder, self).__init__()
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()
        self.word_to_id = tf.keras.layers.StringLookup(
            vocabulary=text_processor.get_vocabulary(), mask_token="", oov_token=UNK
        )
        self.id_to_word = tf.keras.layers.StringLookup(
            vocabulary=text_processor.get_vocabulary(),
            mask_token="",
            oov_token=UNK,
            invert=True,
        )
        self.start_token = self.word_to_id(SOS)
        self.end_token = self.word_to_id(EOS)

        self.embedding = tf.keras.layers.Embedding(
            self.vocab_size, emb_dim, mask_zero=True
        )

        self.rnn = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            kernel_initializer="orthogonal",
            dropout=0.2,
        )

        self.attention = CrossAttention(units, dropout=0.2)

        self.output_layer = tf.keras.layers.Dense(self.vocab_size)

    def call(self, context, x, state=None, return_state=False):
        shape_checker = Module.ShapeChecker()
        shape_checker(x, "batch t")
        # shape_checker(context, "batch s units")

        x = self.embedding(x)
        shape_checker(x, "batch t units")

        x, state = self.rnn(x, initial_state=state)
        # shape_checker(x, "batch t units")

        x = self.attention(x, context)
        self.last_attention_weights = self.attention.last_attention_weights
        # shape_checker(x, "batch t units")
        shape_checker(self.last_attention_weights, "batch t s")

        logits = self.output_layer(x)
        shape_checker(logits, "batch t target_vocab_size")

        if return_state:
            return logits, state
        else:
            return logits
