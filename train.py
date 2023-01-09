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
CHECKPOINT_DIR = gConfig["model_data"]

input_vocab_size = gConfig["vocab_inp_size"]
target_vocab_size = gConfig["vocab_tar_size"]
vocab_inp_path = gConfig["vocab_inp_path"]
vocab_tar_path = gConfig["vocab_tar_path"]


def train_test_split(word_pairs, test_size=0.2):
    np.random.shuffle(word_pairs)
    split_idx = int(len(word_pairs) * (1 - test_size))
    train_word_pairs = word_pairs[:split_idx]
    test_word_pairs = word_pairs[split_idx:]
    return train_word_pairs, test_word_pairs


def tokenize(vocab_file):
    with open(vocab_file, "r", encoding="utf-8") as f:
        tokenize_config = json.dumps(json.load(f), ensure_ascii=False)
        lang_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(
            tokenize_config
        )
    return lang_tokenizer


def read_data(path):
    path = os.getcwd() + "/" + path
    lines = io.open(gConfig["seq_data"], encoding="utf-8").readlines()
    word_pairs = [
        [data_util.preprocess_sentence(w) for w in l.split("\t")] for l in lines
    ]

    train_word_pairs, test_word_pairs = train_test_split(word_pairs)

    train_input_lang, train_target_lang = zip(*train_word_pairs)
    test_input_lang, test_target_lang = zip(*test_word_pairs)

    input_tokenizer = tokenize(vocab_inp_path)
    target_tokenizer = tokenize(vocab_tar_path)

    train_input_tensor = input_tokenizer.texts_to_sequences(train_input_lang)
    train_target_tensor = target_tokenizer.texts_to_sequences(train_target_lang)
    test_input_tensor = input_tokenizer.texts_to_sequences(test_input_lang)
    test_target_tensor = target_tokenizer.texts_to_sequences(test_target_lang)

    train_input_tensor = tf.keras.preprocessing.sequence.pad_sequences(
        train_input_tensor, padding="post"
    )
    train_target_tensor = tf.keras.preprocessing.sequence.pad_sequences(
        train_target_tensor, padding="post"
    )
    test_input_tensor = tf.keras.preprocessing.sequence.pad_sequences(
        test_input_tensor, padding="post"
    )
    test_target_tensor = tf.keras.preprocessing.sequence.pad_sequences(
        test_target_tensor, padding="post"
    )

    return (
        train_input_tensor,
        test_input_tensor,
        input_tokenizer,
        train_target_tensor,
        test_target_tensor,
        target_tokenizer,
    )


def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)


def train():
    if model.ckpt_manager.latest_checkpoint:
        model.ckpt.restore(model.ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!")

    for epoch in range(EPOCHS):
        model.train_loss.reset_state()
        model.train_accuracy.reset_state()

        for (batch, (inp, tar)) in tqdm(
            enumerate(train_dataset.take(STEPS_PER_EPOCH)),
            total=STEPS_PER_EPOCH,
            desc=f"epoch {epoch + 1}",
        ):
            model.train_step(inp, tar)

            with writer.as_default():
                tf.summary.scalar(
                    "train loss",
                    model.train_loss.result(),
                    step=batch,
                )
                tf.summary.scalar(
                    "train accuracy",
                    model.train_accuracy.result(),
                    step=batch,
                )

        model.ckpt_manager.save()

    # test loop
    # ...


if __name__ == "__main__":
    EPOCHS = gConfig["epochs"]
    BATCH_SIZE = gConfig["batch_size"]

    writer = tf.summary.create_file_writer(gConfig["log_dir"])

    (
        train_input_tensor,
        test_input_tensor,
        input_tokenizer,
        train_target_tensor,
        test_target_tensor,
        target_tokenzier,
    ) = read_data(gConfig["seq_data"])

    BUFFER_SIZE = len(train_input_tensor)
    STEPS_PER_EPOCH = BUFFER_SIZE // BATCH_SIZE

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_input_tensor, train_target_tensor)
    )
    train_dataset = train_dataset.filter(filter_max_length)
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(
        BATCH_SIZE, drop_remainder=True
    )
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices(
        (test_input_tensor, test_target_tensor)
    )
    val_dataset = val_dataset.filter(filter_max_length).padded_batch(
        BATCH_SIZE, drop_remainder=True
    )

    train()
