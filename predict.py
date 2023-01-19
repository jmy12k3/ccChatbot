# coding=utf-8
import glob
import sys

import hanlp
import tensorflow as tf

from config import getConfig
from core.Model import Transformer
from core.Translator import ExportTranslator, Translator

# region Config

gConfig = {}
gConfig = getConfig.get_config()

TSV_PATH = gConfig["tsv_path"]

NUM_LAYERS = gConfig["num_layers"]
D_MODEL = gConfig["d_model"]
NUM_HEADS = gConfig["num_heads"]
DFF = gConfig["dff"]
MAX_LENGTH = gConfig["max_length"]
DROPOUT_RATE = gConfig["dropout_rate"]

MODEL_DIR = gConfig["model_dir"]

# endregion

qn_tokenizer = tf.keras.layers.TextVectorization(standardize=None, ragged=True)
ans_tokenizer = tf.keras.layers.TextVectorization(standardize=None, ragged=True)

tok = hanlp.load(hanlp.pretrained.tok.CTB9_TOK_ELECTRA_BASE)

print("Building model...")


def prepare_dataset(tsv, qn_tokenizer, ans_tokenizer):
    # load sample batches and adapt tokenizer
    # ...

    return sample_batches


sample_batches = prepare_dataset(TSV_PATH, qn_tokenizer, ans_tokenizer)

sys.stdout.write("\x1b[2K")

transformer = Transformer(
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dff=DFF,
    input_vocab_size=qn_tokenizer.vocabulary_size(),
    target_vocab_size=ans_tokenizer.vocabulary_size(),
    dropout_rate=DROPOUT_RATE,
)

for (inputs, targets_inputs), _ in sample_batches.take(1):
    transformer((inputs, targets_inputs))

weights = sorted(glob.glob(f"{MODEL_DIR}/*.h5"))

if not weights:
    exit()

transformer.load_weights(f"{weights[-1]}")


def translate(sentence):
    sentence = " ".join(tok(sentence))
    sentence = tf.constant(sentence)

    translator = Translator(qn_tokenizer, ans_tokenizer, transformer)

    result, _, _ = translator(sentence)

    return result


def translate_test(sentence):
    sentence = " ".join(tok(sentence))

    translator = Translator(qn_tokenizer, ans_tokenizer, transformer)
    translator = ExportTranslator(translator)

    result = translator(sentence).numpy()

    return result


if __name__ == "__main__":
    exit()
