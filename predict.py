# coding=utf-8
import glob

import hanlp
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import getConfig
from core.Model import Transformer
from core.Translator import ExportTranslator, Translator

# region Config

gConfig = {}
gConfig = getConfig.get_config()

TSV_PATH = gConfig["tsv_path"]

SOS = gConfig["sos"]
EOS = gConfig["eos"]
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

    df = pd.read_csv(TSV_PATH, sep="\t", nrows=0).columns
    df = pd.read_csv(
        TSV_PATH,
        sep="\t",
        converters={column: (lambda x: UNK if not x else x) for column in df},
    )

    df.iloc[:, -1:] = df.iloc[:, -1:].applymap(
        lambda x: (f"{SOS} {str(x).rstrip()} {EOS}") if x != UNK else UNK
    )

    ds = df.values.tolist()
    inputs, targets = zip(*ds)

    inputs_tokenizer.adapt(list(inputs))
    targets_tokenizer.adapt(list(targets))

    _ds, sample_ds = train_test_split(df, test_size=0.1, random_state=42)

    sample_ds = sample_ds.values.tolist()
    sample_ds_inputs, sample_ds_targets = zip(*sample_ds)

    sample_ds = tf.data.Dataset.from_tensor_slices(
        (sample_ds_inputs, sample_ds_targets)
    )

    BUFFER_SIZE = len(sample_ds_inputs)

    sample_batches = make_batches(sample_ds)

    return sample_batches


sample_batches = prepare_dataset(inputs_tokenizer, targets_tokenizer, TSV_PATH)

transformer = Transformer(
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dff=DFF,
    input_vocab_size=inputs_tokenizer.vocabulary_size(),
    target_vocab_size=targets_tokenizer.vocabulary_size(),
    dropout_rate=DROPOUT_RATE,
)

for (inputs, targets_inputs), _ in sample_batches.take(1):
    transformer((inputs, targets_inputs))

weights = sorted(glob.glob(f"{MODEL_DIR}/*.h5"))
if weights:
    transformer.load_weights(f"{weights[-1]}")
    print("\nLatest weights restored!\n")


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
    exit()
