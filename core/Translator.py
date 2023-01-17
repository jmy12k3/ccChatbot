# coding=utf-8
import tensorflow as tf

from config import getConfig

gConfig = {}
gConfig = getConfig.get_config()

MAX_LENGTH = gConfig["max_length"]

SOS = gConfig["sos"]
EOS = gConfig["eos"]


class Translator(tf.Module):
    def __init__(self, inputs_tokenizer, targets_tokenizer, transformer):
        self.inputs_tokenizer = inputs_tokenizer
        self.targets_tokenizer = targets_tokenizer
        self.transformer = transformer

    def __call__(self, sentence, max_length=MAX_LENGTH):
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]

        sentence = self.inputs_tokenizer(sentence).to_tensor()

        encoder_input = sentence

        start_end = self.targets_tokenizer.get_vocabulary()
        start = tf.constant(start_end.index(SOS), dtype=tf.int64)[tf.newaxis]
        end = tf.constant(start_end.index(EOS), dtype=tf.int64)[tf.newaxis]

        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions = self.transformer([encoder_input, output], training=False)

            predictions = predictions[:, -1:, :]

            predicted_id = tf.argmax(predictions, axis=-1)

            if predicted_id == end:
                break

            output_array = output_array.write(i + 1, predicted_id[0])

        output = tf.transpose(output_array.stack())[:, 1:]

        # text = tokenizers.en.detokenize(output)[0]

        # tokens = tokenizers.en.lookup(output)[0]

        self.transformer([encoder_input, output[:, :-1]], training=False)
        attention_weights = self.transformer.decoder.last_attn_scores

        return output, attention_weights
