# coding=utf-8
import tensorflow as tf

from config import getConfig

# region Config

gConfig = {}
gConfig = getConfig.get_config()

SOS = gConfig["sos"]
EOS = gConfig["eos"]

MAX_LENGTH = gConfig["max_length"]

# endregion


class Translator(tf.Module):
    def __init__(self, qn_tokenizer, ans_tokenizer, transformer):
        self.qn_tokenizer = qn_tokenizer
        self.ans_tokenizer = ans_tokenizer
        self.transformer = transformer

    def __call__(self, sentence, max_length=MAX_LENGTH):
        assert isinstance(sentence, tf.Tensor), "Input must be a tensor. (tf.constant)"
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]

        sentence = self.qn_tokenizer(sentence).to_tensor()

        encoder_input = sentence

        start_end = self.ans_tokenizer.get_vocabulary()
        start = tf.constant(start_end.index(SOS), dtype=tf.int64)[tf.newaxis]
        end = tf.constant(start_end.index(EOS), dtype=tf.int64)[tf.newaxis]

        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions = self.transformer([encoder_input, output], training=False)

            predictions = predictions[:, -1:, :]

            predicted_id = tf.argmax(predictions, axis=-1)

            output_array = output_array.write(i + 1, predicted_id[0])

            if predicted_id == end:
                break

        output = tf.transpose(output_array.stack())

        v = self.ans_tokenizer.get_vocabulary()

        detokenize = dict(zip(range(len(v)), v))
        lookup = {i: v for v, i in detokenize.items()}

        text = " ".join([detokenize[i] for i in output.numpy().tolist()[0][1:-1]])
        tokens = [lookup[i] for i in output.numpy().tolist()[0]]

        self.transformer([encoder_input, output[:, :-1]], training=False)
        attention_weights = self.transformer.decoder.last_attn_scores

        return text, tokens, attention_weights


class ExportTranslator(tf.Module):
    def __init__(self, translator):
        self.translator = translator

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, sentence):
        (result, tokens, attention_weights) = self.translator(
            sentence, max_length=MAX_LENGTH
        )

        return result
