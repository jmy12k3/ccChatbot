# coding=utf-8
import glob

import numpy as np
import tensorflow as tf

import data_util
import model
from config import getConfig

# region CONSTANTS
gConfig = {}
gConfig = getConfig.get_config()

# Preprocessing related
SEQ_PATH = gConfig["seq_path"]

# Training related
BATCH_SIZE = gConfig["batch_size"]
EPOCHS = gConfig["epoch"]

# Callbacks related
LOG_DIR = gConfig["log_dir"]
MODEL_DIR = gConfig["model_dir"]

# endregion


# shuffle=False if model will be loaded from checkpoint to train
def train_val_split(ds, val_size=0.2, shuffle=False):
    if shuffle:
        np.random.shuffle(ds)
    idx = int(len(ds) * (1 - val_size))
    train_ds = ds[:idx]
    val_ds = ds[idx:]
    return train_ds, val_ds


def read_data(path):
    lines = open(path, encoding="utf-8").readlines()
    pairs = [
        [
            w if i % 2 == 0 else data_util.preprocess_sentence(w)
            for i, w in enumerate(l.split("\t"))
        ]
        for l in lines
    ]
    train_pairs, val_pairs = train_val_split(pairs)

    inputs, targets = zip(*pairs)
    train_inputs, train_targets = zip(*train_pairs)
    val_inputs, val_targets = zip(*val_pairs)

    model.inputs_tokenizer.adapt(list(inputs))
    model.targets_tokenizer.adapt(list(targets))

    return (
        train_inputs,
        train_targets,
        val_inputs,
        val_targets,
    )


def prepare_batch(inputs, targets):
    inputs = model.inputs_tokenizer(inputs)
    inputs = inputs[:, : model.MAX_LENGTH]
    inputs = inputs.to_tensor()

    targets = model.targets_tokenizer(targets)
    targets = targets[:, : (model.MAX_LENGTH + 2)]
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


def train():
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="%s/model.{epoch:02d}-{val_loss:.4f}.h5" % MODEL_DIR,
        save_best_only=True,
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)
    callbacks = [model_checkpoint_callback, tensorboard_callback]

    transformer.fit(
        train_batches,
        epochs=EPOCHS,
        validation_data=val_batches,
        callbacks=callbacks,
    )


# Test version: not for serve purpose!
# Internal use only in-file
def predict(sentence):
    sentence = " ".join(data_util.tok(sentence))
    sentence = tf.constant(sentence)

    translator = model.Translator(transformer)

    result = translator(sentence)

    return result


if __name__ == "__main__":
    train_inputs, train_targets, val_inputs, val_targets = read_data(SEQ_PATH)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (list(train_inputs), list(train_targets))
    )
    val_ds = tf.data.Dataset.from_tensor_slices((list(val_inputs), list(val_targets)))

    BUFFER_SIZE = len(train_inputs)

    train_batches = make_batches(train_ds)
    val_batches = make_batches(val_ds)
    
    transformer = model.instantiate()
    for (inputs, targets_inputs), _ in train_batches.take(1):
        transformer((inputs, targets_inputs))
    transformer.summary()

    weights = glob.glob(f"{MODEL_DIR}/*.h5")[-1]
    if weights:
        transformer.load_weights(f"{MODEL_DIR}/" + weights)

    train()
