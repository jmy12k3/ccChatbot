import pickle

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import Config
from core import Checkpoint, Model

# region Config

config = Config.config()

TSV_PATH = config["tsv_path"]
CTX_PATH = config["ctx_path"]
TGT_PATH = config["tgt_path"]

SOS = config["sos"]
EOS = config["eos"]
UNK = config["unk"]

LOG_DIR = config["log_dir"]
MODEL_DIR = config["model_dir"]

MAX_LENGTH = config["max_length"]

EMB_DIM = config["emb_dim"]
UNITS = config["units"]

BATCH_SIZE = config["batch_size"]
EPOCHS = config["epochs"]

# endregion


def prepare_batches(tsv_path, context_text_processor, target_text_processor):
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

    ckpt, ckpt_manager = Checkpoint.checkpoint(
        MODEL_DIR, 6, optimizer=optimizer, model=model
    )
    checkpoint_callback = Checkpoint.Checkpoint(ckpt, ckpt_manager)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        "val_pp", patience=5, baseline=2
    )

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

    train_batches, val_batches = prepare_batches(
        TSV_PATH, context_text_processor, target_text_processor
    )

    model = Model.Translator(
        EMB_DIM, UNITS, context_text_processor, target_text_processor
    )

    for (context, target_inputs), _ in train_batches.take(1):
        model((context, target_inputs))

    model.summary()

    train(train_batches, val_batches, model)


if __name__ == "__main__":
    main()
