# coding=utf-8
import io

import numpy as np
import tensorflow as tf

import data_util
import model

from config import getConfig

# I just place the constant here randomly
gConfig = {}
gConfig = getConfig.get_config()

# I just place the constant here randomly
BATCH_SIZE = gConfig["batch_size"]
EPOCH = gConfig["epoch"]


# Simplified implementation of sklearn.model_selection.train_test_split
# Shuffle should only be True if:
# - The dataset is not continuous
# - The training will NOT be continued from a checkpoint in the future
def train_val_split(ds, val_size=0.2, shuffle=False):
    if shuffle:
        np.random.shuffle(ds)
    idx = int(len(ds) * (1 - val_size))
    train_ds = ds[:idx]
    val_ds = ds[idx:]
    return train_ds, val_ds


def read_data(path):
    lines = io.open(path, encoding="utf-8").readlines()
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


# Reference: https://www.tensorflow.org/text/tutorials/transformer#set_up_a_data_pipeline_with_tfdata
def prepare_batch(inputs, targets):
    inputs = model.inputs_tokenizer(inputs)
    inputs = inputs[:, : model.MAX_LENGTH]

    targets = model.targets_tokenizer(targets)
    targets = targets[:, : (model.MAX_LENGTH + 1)]
    targets_inputs = targets[:, :-1]
    targets_labels = targets[:, 1:]

    return (inputs, targets_inputs), targets_labels


# Reference: https://www.tensorflow.org/text/tutorials/transformer#set_up_a_data_pipeline_with_tfdata
def make_batches(ds):
    return (
        ds.cache()
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .map(prepare_batch, tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )


# Reference: https://www.tensorflow.org/text/tutorials/transformer#train_the_model
def train():
    transformer, ckpt, ckpt_manager = model.delayed_initialize()

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!")

    for (inputs, targets_inputs), _ in train_batches.take(1):
        transformer((inputs, targets_inputs))
    transformer.summary()

    transformer.compile(
        loss=model.masked_loss,
        optimizer=model.optimizer,
        metrics=[model.masked_accuracy],
    )

    transformer.fit(
        train_batches,
        epochs=EPOCH,
        validation_data=val_batches,
        callbacks=[model.tensorboard_callback],
    )


if __name__ == "__main__":
    # I just place the constant here randomly
    SEQ_DATA = gConfig["seq_data"]

    (
        train_inputs,
        train_targets,
        val_inputs,
        val_targets,
    ) = read_data(SEQ_DATA)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (list(train_inputs), list(train_targets))
    )
    val_ds = tf.data.Dataset.from_tensor_slices((list(val_inputs), list(val_targets)))

    BUFFER_SIZE = len(train_inputs)

    train_batches = make_batches(train_ds)
    val_batches = make_batches(val_ds)

    train()
