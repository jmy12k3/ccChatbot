# coding=utf-8
import io
import json

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tqdm.contrib import tzip

import data_util
import model
from config import getConfig

gConfig = {}
gConfig = getConfig.get_config()

MAX_LENGTH = gConfig["max_length"]

INPUT_VOCAB_PATH = gConfig["input_vocab_path"]
TARGET_VOCAB_PATH = gConfig["target_vocab_path"]


# Simplified version of sklearn.model_selection.train_test_split()
# Shuffle should not be used under 2 cases:
# 1. The data is continuous
# 2. The model will be trained from checkpoint in the future.
#    As we do not know the previous shuffled distirbution, this may lead to the model learning from the test set.
def train_test_split(dataset, test_size=0.2, shuffle=False):
    if shuffle:
        np.random.shuffle(dataset)
    idx = int(len(dataset) * (1 - test_size))
    train_dataset = dataset[:idx]
    test_dataset = dataset[idx:]
    return train_dataset, test_dataset


def tokenize(vocab_file):
    with open(vocab_file, "r", encoding="utf-8") as f:
        tokenize_config = json.dumps(json.load(f), ensure_ascii=False)
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenize_config)
    return tokenizer


def read_data(path):
    # Dataset Map (Substitute to tf.data.dataset.map() as the restriction of tensorflow-metal)
    lines = io.open(path, encoding="utf-8").readlines()
    pairs = [[data_util.preprocess_sentence(w) for w in l.split("\t")] for l in lines]

    # Split
    train_pairs, test_pairs = train_test_split(pairs)
    train_input, train_target = zip(*train_pairs)
    test_input, test_target = zip(*test_pairs)

    # Dataset Encode (Substitute to tf.keras.layers.TextVectorization() as the restriction of tensorflow-metal)
    input_tokenizer = tokenize(INPUT_VOCAB_PATH)
    target_tokenizer = tokenize(TARGET_VOCAB_PATH)

    train_input_tensor = input_tokenizer.texts_to_sequences(train_input)
    train_target_tensor = target_tokenizer.texts_to_sequences(train_target)
    test_input_tensor = input_tokenizer.texts_to_sequences(test_input)
    test_target_tensor = target_tokenizer.texts_to_sequences(test_target)

    # Dataset Filter (Substitute to tf.data.dataset.filter() as the restriction of tensorflow-metal)
    print("Filtering dataset...")

    assert len(train_input_tensor) == len(train_target_tensor)
    assert len(test_input_tensor) == len(test_target_tensor)

    for (i, (x, y)) in enumerate(
        tzip(train_input_tensor, train_target_tensor, ascii=" >=", desc="Train")
    ):
        if tf.logical_or(tf.size(x) > MAX_LENGTH, tf.size(y) > MAX_LENGTH):
            train_input_tensor.remove(train_input_tensor[i])
            train_target_tensor.remove(train_target_tensor[i])

    for (i, (x, y)) in enumerate(
        tzip(test_input_tensor, test_target_tensor, ascii=" >=", desc="Test")
    ):
        if tf.logical_or(tf.size(x) > MAX_LENGTH, tf.size(y) > MAX_LENGTH):
            test_input_tensor.remove(test_input_tensor[i])
            test_target_tensor.remove(test_target_tensor[i])

    # Padding
    train_input_tensor = tf.keras.preprocessing.sequence.pad_sequences(
        train_input_tensor, maxlen=MAX_LENGTH
    )
    train_target_tensor = tf.keras.preprocessing.sequence.pad_sequences(
        train_target_tensor, maxlen=MAX_LENGTH
    )
    test_input_tensor = tf.keras.preprocessing.sequence.pad_sequences(
        test_input_tensor, maxlen=MAX_LENGTH
    )
    test_target_tensor = tf.keras.preprocessing.sequence.pad_sequences(
        test_target_tensor, maxlen=MAX_LENGTH
    )

    return (
        train_input_tensor,
        train_target_tensor,
        test_input_tensor,
        test_target_tensor,
        input_tokenizer,
        target_tokenizer,
    )


def train():
    if model.ckpt_manager.latest_checkpoint:
        model.ckpt.restore(model.ckpt_manager.latest_checkpoint)
        print("\nLatest checkpoint restored!")

    for epoch in range(EPOCH):
        model.train_loss.reset_state()
        model.accuracy.reset_state()

        for (batch, (inp, tar)) in enumerate(
            tqdm(train_dataset, ascii=" >=", desc=f"Epoch {epoch + 1}")
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

    for (batch, (inp, tar)) in enumerate(tqdm(test_dataset, ascii=" >=", desc="Test")):
        model.test_step(inp, tar)

        with writer.as_default():
            tf.summary.scalar("test accuracy", model.accuracy.result(), step=batch)


def predict(sentence):
    model.ckpt.restore(model.ckpt_manager.latest_checkpoint)

    sentence = " ".join(data_util.tok(sentence))

    input_tokenizer = tokenize(INPUT_VOCAB_PATH)
    target_tokenizer = tokenize(TARGET_VOCAB_PATH)

    encoder_input = input_tokenizer.texts_to_sequences(
        [data_util.preprocess_sentence(sentence)]
    )
    encoder_input = tf.keras.preprocessing.sequence.pad_sequences(
        encoder_input, maxlen=MAX_LENGTH
    )
    encoder_input = tf.convert_to_tensor(encoder_input)

    decoder_input = [target_tokenizer.word_index[data_util.SOS]]
    output = tf.expand_dims(decoder_input, 0)

    result = ""
    index_word = {v: k for k, v in target_tokenizer.word_index.items()}

    for _ in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = model.create_masks(
            encoder_input, output
        )

        # attention_weights
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

        if predicted_id == target_tokenizer.word_index[data_util.EOS]:
            break

        result += str(index_word[int(predicted_id)]) + " "

        output = tf.concat([output, predicted_id], axis=-1)

    return result


if __name__ == "__main__":
    BATCH_SIZE = gConfig["batch_size"]
    EPOCH = gConfig["epoch"]

    LOG_DIR = gConfig["log"]
    writer = tf.summary.create_file_writer(LOG_DIR)

    # input_tokenizer, target_tokenizer
    SEQ_DATA = gConfig["seq_data"]
    (
        train_input_tensor,
        train_target_tensor,
        test_input_tensor,
        test_target_tensor,
        _,
        _,
    ) = read_data(SEQ_DATA)

    BUFFER_SIZE = len(train_input_tensor)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_input_tensor, train_target_tensor)
    )
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(
        BATCH_SIZE, drop_remainder=True
    )
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_input_tensor, test_target_tensor)
    )
    test_dataset = test_dataset.padded_batch(BATCH_SIZE, drop_remainder=True)

    train()
