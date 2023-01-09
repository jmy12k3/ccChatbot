# coding=utf-8
import io
import json
import os

import tensorflow as tf
from tqdm import tqdm

import data_util
import model
from config import getConfig


gConfig = {}
gConfig = getConfig.get_config()

MAX_LENGTH = gConfig["max_length"]

CHECKPOINT_DIR = gConfig["model_data"]
INPUT_VOCAB_PATH = gConfig["input_vocab_path"]
TARGET_VOCAB_PATH = gConfig["target_vocab_path"]


def tokenize(vocab_file):
    with open(vocab_file, "r", encoding="utf-8") as f:
        tokenize_config = json.dumps(json.load(f), ensure_ascii=False)
        lang_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(
            tokenize_config
        )
    return lang_tokenizer


def read_data(path):
    path = os.getcwd() + "/" + path

    # Dataset Map
    lines = io.open(gConfig["seq_data"], encoding="utf-8").readlines()
    word_pairs = [
        [data_util.preprocess_sentence(w) for w in l.split("\t")] for l in lines
    ]
    input_lang, target_lang = zip(*word_pairs)

    # Dataset Encode
    input_tokenizer = tokenize(INPUT_VOCAB_PATH)
    target_tokenizer = tokenize(TARGET_VOCAB_PATH)
    input_tensor = input_tokenizer.texts_to_sequences(input_lang)
    target_tensor = target_tokenizer.texts_to_sequences(target_lang)

    # Dataset Filter
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(
        input_tensor, maxlen=MAX_LENGTH, padding="post"
    )
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(
        target_tensor, maxlen=MAX_LENGTH, padding="post"
    )

    return input_tensor, input_tokenizer, target_tensor, target_tokenizer


def train():
    if model.ckpt_manager.latest_checkpoint:
        model.ckpt.restore(model.ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!")

    for epoch in range(EPOCHS):
        model.train_loss.reset_state()
        model.train_accuracy.reset_state()

        for (batch, (inp, tar)) in tqdm(
            enumerate(dataset.take(STEPS_PER_EPOCH)),
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

    input_tensor, input_token, target_tensor, target_token = read_data(
        gConfig["seq_data"]
    )

    BUFFER_SIZE = len(input_tensor)
    STEPS_PER_EPOCH = BUFFER_SIZE // BATCH_SIZE

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor))
    dataset = dataset.cache()
    dataset = dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    train()
