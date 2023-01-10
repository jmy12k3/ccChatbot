# coding=utf-8
import io
import json
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import data_util
import model
from config import getConfig


gConfig = {}
gConfig = getConfig.get_config()

MAX_LENGTH = gConfig["max_length"]

INPUT_VOCAB_PATH = gConfig["input_vocab_path"]
TARGET_VOCAB_PATH = gConfig["target_vocab_path"]


def train_test_split(dataset, test_size=0.2):
    np.random.shuffle(dataset)
    idx = int(len(dataset) * (1 - test_size))
    train_dataset = dataset[:idx]
    test_dataset = dataset[idx:]
    return train_dataset, test_dataset


def tokenize(vocab_file):
    with open(vocab_file, "r", encoding="utf-8") as f:
        tokenize_config = json.dumps(json.load(f), ensure_ascii=False)
        lang_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(
            tokenize_config
        )
    return lang_tokenizer


def read_data(path):
    # Dataset Map (Substitute to tf.data.dataset.map())
    lines = io.open(path, encoding="utf-8").readlines()
    word_pairs = [
        [data_util.preprocess_sentence(w) for w in l.split("\t")] for l in lines
    ]

    # Split
    train_word_pairs, val_word_pairs = train_test_split(word_pairs)
    train_input_lang, train_target_lang = zip(*train_word_pairs)
    val_input_lang, val_target_lang = zip(*val_word_pairs)

    # Encode
    input_tokenizer = tokenize(INPUT_VOCAB_PATH)
    target_tokenizer = tokenize(TARGET_VOCAB_PATH)

    _train_input_tensor = input_tokenizer.texts_to_sequences(train_input_lang)
    _train_target_tensor = target_tokenizer.texts_to_sequences(train_target_lang)
    _val_input_tensor = input_tokenizer.texts_to_sequences(val_input_lang)
    _val_target_tensor = target_tokenizer.texts_to_sequences(val_target_lang)

    # Dataset Filter (Substitute to tf.data.dataset.filter())
    print("Filtering dataset...")

    train_input_tensor = []
    train_target_tensor = []

    assert len(_train_input_tensor) == len(_train_target_tensor)

    for (i, (x, y)) in tqdm(
        enumerate(zip(_train_input_tensor, _train_target_tensor)),
        total=len(_train_input_tensor),
        ascii=" >=",
        desc="Train",
    ):
        if tf.logical_and(tf.size(x) <= MAX_LENGTH, tf.size(y) <= MAX_LENGTH):
            train_input_tensor.append(_train_input_tensor[i])
            train_target_tensor.append(_train_target_tensor[i])

    val_input_tensor = []
    val_target_tensor = []

    assert len(_val_input_tensor) == len(_val_target_tensor)

    for (i, (x, y)) in tqdm(
        enumerate(zip(_val_input_tensor, _val_target_tensor)),
        total=len(_val_input_tensor),
        ascii=" >=",
        desc="Validation",
    ):
        if tf.logical_and(tf.size(x) <= MAX_LENGTH, tf.size(y) <= MAX_LENGTH):
            val_input_tensor.append(_val_input_tensor[i])
            val_target_tensor.append(_val_target_tensor[i])

    # Pad
    train_input_tensor = tf.keras.preprocessing.sequence.pad_sequences(
        train_input_tensor
    )
    train_target_tensor = tf.keras.preprocessing.sequence.pad_sequences(
        train_target_tensor
    )
    val_input_tensor = tf.keras.preprocessing.sequence.pad_sequences(val_input_tensor)
    val_target_tensor = tf.keras.preprocessing.sequence.pad_sequences(val_target_tensor)

    return (
        train_input_tensor,
        train_target_tensor,
        val_input_tensor,
        val_target_tensor,
        input_tokenizer,
        target_tokenizer,
    )


def train():
    if model.ckpt_manager.latest_checkpoint:
        model.ckpt.restore(model.ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!")

    for epoch in range(EPOCHS):
        model.train_loss.reset_state()
        model.accuracy.reset_state()

        for (batch, (inp, tar)) in tqdm(
            enumerate(train_dataset),
            total=STEPS_PER_EPOCH,
            ascii=" >=",
            desc=f"Epoch {epoch + 1}",
        ):
            model.train_step(inp, tar)

            with writer.as_default():
                tf.summary.scalar(
                    "train loss",
                    model.train_loss.result(),
                    step=model.optimizer.iterations,
                )
                tf.summary.scalar(
                    "train accuracy",
                    model.accuracy.result(),
                    step=model.optimizer.iterations,
                )

        model.ckpt_manager.save()

    model.accuracy.reset_state()

    for (batch, (inp, tar)) in tqdm(
        enumerate(val_dataset),
        total=(len(val_input_tensor) // BATCH_SIZE),
        ascii=" >=",
        desc="Validation",
    ):
        model.test_step(inp, tar)

        with writer.as_default():
            tf.summary.scalar("test accuracy", model.accuracy.result(), step=batch)


def predict(inp_sentence):
    input_tokenizer = tokenize(os.path.dirname(os.getcwd()) + "/" + INPUT_VOCAB_PATH)
    target_tokenizer = tokenize(os.path.dirname(os.getcwd() + "/" + TARGET_VOCAB_PATH))

    model.ckpt.restore(model.ckpt_manager.latest_checkpoint)

    inp_sentence = input_tokenizer.texts_to_sequences(
        data_util.preprocess_sentence(inp_sentence)
    )
    encoder_input = tf.expand_dims(inp_sentence, 0)

    decoder_input = [target_tokenizer.word_index[data_util.SOS]]
    output = tf.expand_dims(decoder_input, 0)

    result = ""

    for _ in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = model.create_masks(
            encoder_input, output
        )

        predictions, _ = model.transformer(
            encoder_input,
            output,
            False,
            enc_padding_mask,
            combined_mask,
            dec_padding_mask,
        )

        predictions = predictions[:, -1:, :]

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if target_tokenizer.index_word[predicted_id] == data_util.EOS:
            break

        result += str(target_tokenizer.index_word[predicted_id]) + " "

        output = tf.concat([output, predicted_id], axis=-1)

    return result


if __name__ == "__main__":
    EPOCHS = gConfig["epochs"]
    BATCH_SIZE = gConfig["batch_size"]

    writer = tf.summary.create_file_writer(gConfig["log_dir"])

    (
        train_input_tensor,
        train_target_tensor,
        val_input_tensor,
        val_target_tensor,
        _,
        target_tokenizer,
    ) = read_data(gConfig["seq_data"])

    BUFFER_SIZE = len(train_input_tensor)
    STEPS_PER_EPOCH = BUFFER_SIZE // BATCH_SIZE

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_input_tensor, train_target_tensor)
    )
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(
        BATCH_SIZE, drop_remainder=True
    )
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices(
        (val_input_tensor, val_target_tensor)
    )
    val_dataset = val_dataset.padded_batch(BATCH_SIZE, drop_remainder=True)

    train()
