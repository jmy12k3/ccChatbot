# coding=utf-8
import glob

import hanlp
import pandas as pd
import tensorflow as tf

from config import getConfig
from core.Model import Transformer
from core.Translator import ExportTranslator, Translator

# region Config

gConfig = {}
gConfig = getConfig.get_config()

TSV_PATH = gConfig["tsv_path"]

UNK = gConfig["unk"]

NUM_LAYERS = gConfig["num_layers"]
D_MODEL = gConfig["d_model"]
NUM_HEADS = gConfig["num_heads"]
DFF = gConfig["dff"]
MAX_LENGTH = gConfig["max_length"]
DROPOUT_RATE = gConfig["dropout_rate"]

MODEL_DIR = gConfig["model_dir"]

BATCH_SIZE = gConfig["batch_size"]

# endregion

inputs_tokenizer = tf.keras.layers.TextVectorization(standardize=None, ragged=True)
targets_tokenizer = tf.keras.layers.TextVectorization(standardize=None, ragged=True)

tok = hanlp.load(hanlp.pretrained.tok.CTB9_TOK_ELECTRA_BASE)

print("Building model...")


def prepare_dataset(inputs_tokenizer, targets_tokenizer, tsv):
    """A minimal version of prepare_dataset() from train.py"""

    def apply_unk(x):
        return UNK if not x else x

    def prepare_batch(inputs, targets):
        inputs = inputs_tokenizer(inputs)
        inputs = inputs[:, :MAX_LENGTH]
        inputs = inputs.to_tensor()

        targets = targets_tokenizer(targets)
        targets = targets[:, : (MAX_LENGTH + 2)]
        targets_inputs = targets[:, :-1].to_tensor()
        targets_labels = targets[:, 1:].to_tensor()

        return (inputs, targets_inputs), targets_labels

    def make_batches(ds):
        return (
            ds.cache()
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .map(prepare_batch, tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )

    df = pd.read_csv(tsv, sep="\t", nrows=0).columns
    df = pd.read_csv(tsv, sep="\t", converters={column: apply_unk for column in df})

    ds = df.values.tolist()

    inputs, targets = zip(*ds)

    inputs_tokenizer.adapt(list(inputs))
    targets_tokenizer.adapt(list(targets))

    ds = tf.data.Dataset.from_tensor_slices(
        (list(inputs[:BATCH_SIZE]), list(targets[:BATCH_SIZE]))
    )

    BUFFER_SIZE = len(inputs)

    batches = make_batches(ds)

    return batches


batches = prepare_dataset(inputs_tokenizer, targets_tokenizer, TSV_PATH)

transformer = Transformer(
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dff=DFF,
    input_vocab_size=inputs_tokenizer.vocabulary_size(),
    target_vocab_size=targets_tokenizer.vocabulary_size(),
    dropout_rate=DROPOUT_RATE,
)

for (inputs, targets_inputs), _ in batches.take(1):
    transformer((inputs, targets_inputs))

weights = sorted(glob.glob(f"{MODEL_DIR}/*.h5"))
if weights:
    transformer.load_weights(f"{weights[-1]}")
    print("Latest weights restored!")


def predict(sentence):
    sentence = " ".join(tok(sentence))
    sentence = tf.constant(sentence)

    translator = Translator(inputs_tokenizer, targets_tokenizer, transformer)

    result, _, _ = translator(sentence)

    return result


def predict_test(sentence):
    sentence = " ".join(tok(sentence))

    translator = Translator(inputs_tokenizer, targets_tokenizer, transformer)
    translator = ExportTranslator(translator)

    result = translator(sentence).numpy()

    return result


if __name__ == "__main__":
    print("This is a module. Please run ./web/app.py instead.")
    exit()
