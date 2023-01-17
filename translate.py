import hanlp
import tensorflow as tf

from core.Translator import Translator

tok = hanlp.load(hanlp.pretrained.tok.LARGE_ALBERT_BASE)
tok.dict_force = {}
tok.dict_combine = {}

inputs_tokenizer = tf.keras.layers.TextVectorization(standardize=None, ragged=True)
targets_tokenizer = tf.keras.layers.TextVectorization(standardize=None, ragged=True)


"""
def translate(sentence):
    sentence = " ".join(tok(sentence))
    sentence = tf.constant(sentence)

    translator = Translator(
        inputs_tokenizer, targets_tokenizer, transformer
    )

    result, _ = translator(sentence)

    return result

if __name__ == "__main__":
    print('Please run \"python ./web/app.py\" instead.')
    exit()
"""
