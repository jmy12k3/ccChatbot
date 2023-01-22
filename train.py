import pickle

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import getConfig
from core import Checkpoint
from core.Model import Translator

# region Config

gConfig = {}
gConfig = getConfig.get_config()

TSV_PATH = gConfig["tsv_path"]
CTX_PATH = gConfig["ctx_path"]
TGT_PATH = gConfig["tgt_path"]

SOS = gConfig["sos"]
EOS = gConfig["eos"]
UNK = gConfig["unk"]

LOG_DIR = gConfig["log_dir"]
MODEL_DIR = gConfig["model_dir"]

MAX_LENGTH = gConfig["max_length"]

UNITS = gConfig["units"]

BATCH_SIZE = gConfig["batch_size"]
EPOCHS = gConfig["epochs"]

# endregion


def get_batches(tsv_path, context_text_processor, target_text_processor):
    def apply_unk(x):
        return UNK if not x else x

    def apply_sos_eos(x):
        return f"{SOS} {str(x).rstrip()} {EOS}" if x != UNK else UNK

    def prepare_batch(context, target):
        context = context_text_processor(context)
        context = context[:, :MAX_LENGTH]
        context = context.to_tensor()

        target = target_text_processor(target)
        target = target[:, : (MAX_LENGTH + 2)]
        target_inputs = target[:, :-1].to_tensor()
        target_labels = target[:, 1:].to_tensor()

        return (context, target_inputs), target_labels

    def make_batches(ds):
        return (
            ds.cache()
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .map(prepare_batch, tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )

    df = pd.read_csv(tsv_path, sep="\t", nrows=0).columns

    df = pd.read_csv(tsv_path, sep="\t", converters={col: apply_unk for col in df})

    # The TSV file must be in 2 columns only for this operation
    df.iloc[:, -1:] = df.iloc[:, -1:].applymap(apply_sos_eos)

    train_ds, val_ds = train_test_split(df, test_size=0.2, random_state=42)

    ds = df.values.tolist()
    train_ds = train_ds.values.tolist()
    val_ds = val_ds.values.tolist()

    context, target = zip(*ds)
    train_context, train_target = zip(*train_ds)
    val_context, val_target = zip(*val_ds)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (list(train_context), list(train_target))
    )
    val_ds = tf.data.Dataset.from_tensor_slices((list(val_context), list(val_target)))

    BUFFER_SIZE = train_ds.cardinality().numpy()

    try:
        print("Loading cache...")

        ctx_cache = pickle.load(open(CTX_PATH, "rb"))
        tgt_cache = pickle.load(open(TGT_PATH, "rb"))

        context_text_processor.set_weights(ctx_cache["weights"])
        target_text_processor.set_weights(tgt_cache["weights"])

    except OSError:
        print("Cache not found. Caching...")

        context_text_processor.adapt(list(context))
        target_text_processor.adapt(list(target))

        pickle.dump(
            {
                "config": context_text_processor.get_config(),
                "weights": context_text_processor.get_weights(),
            },
            open(CTX_PATH, "wb"),
        )
        pickle.dump(
            {
                "config": target_text_processor.get_config(),
                "weights": target_text_processor.get_weights(),
            },
            open(TGT_PATH, "wb"),
        )

    train_batches = make_batches(train_ds)
    val_batches = make_batches(val_ds)

    return train_batches, val_batches


def train(train_batches, val_batches, model):
    optimizer = tf.optimizers.Adam()

    model.compile(optimizer=optimizer)

    ckpt, ckpt_manager = Checkpoint.get_ckpt(model, optimizer, MODEL_DIR, patience=4)

    checkpoint_callback = Checkpoint.Checkpoint(ckpt, ckpt_manager)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=3)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(LOG_DIR)

    callbacks = [
        checkpoint_callback,
        early_stopping_callback,
        tensorboard_callback,
    ]

    model.fit(
        train_batches,
        epochs=EPOCHS,
        validation_data=val_batches,
        callbacks=callbacks,
    )


def main():
    context_text_processor = tf.keras.layers.TextVectorization(
        standardize=None, ragged=True
    )
    target_text_processor = tf.keras.layers.TextVectorization(
        standardize=None, ragged=True
    )

    train_batches, val_batches = get_batches(
        TSV_PATH, context_text_processor, target_text_processor
    )

    writer = tf.summary.create_file_writer(LOG_DIR)

    model = Translator(UNITS, context_text_processor, target_text_processor, writer)

    for (context, target_inputs), _ in train_batches.take(1):
        model((context, target_inputs))

    model.summary()

    train(train_batches, val_batches, model)


if __name__ == "__main__":
    main()
