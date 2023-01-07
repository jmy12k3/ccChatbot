# coding=utf-8
import io
import json
import os
import sys

import tensorflow as tf
from tqdm import tqdm

import data_util
import model
import callbacks
from config import getConfig

gConfig = {}
gConfig = getConfig.get_config()

CHECKPOINT_DIR = gConfig["model_data"]
VOCAB_INP_PATH = gConfig["vocab_inp_path"]
VOCAB_TAR_PATH = gConfig["vocab_tar_path"]
MAX_LENGTH_INP = gConfig["max_length"]
MAX_LENGTH_TAR = gConfig["max_length"]


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
    input_lang, target_lang = zip(*word_pairs)
    input_tokenizer = tokenize(VOCAB_INP_PATH)
    target_tokenizer = tokenize(VOCAB_TAR_PATH)
    input_tensor = input_tokenizer.texts_to_sequences(input_lang)
    target_tensor = target_tokenizer.texts_to_sequences(target_lang)
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(
        input_tensor, maxlen=MAX_LENGTH_INP, padding="post"
    )
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(
        target_tensor, maxlen=MAX_LENGTH_TAR, padding="post"
    )
    return input_tensor, input_tokenizer, target_tensor, target_tokenizer


def train():
    ckpt = tf.io.gfile.listdir(CHECKPOINT_DIR)
    if ckpt:
        print("Reloaded pretrained model!")
        model.checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR))

    epoch = 1

    while epoch <= EPOCHS:
        # region on_epoch_begin
        enc_callback.on_epoch_begin(epoch, logs=logs)
        dec_callback.on_epoch_begin(epoch, logs=logs)
        # endregion

        for batch, (inp, targ) in tqdm(
            enumerate(dataset.take(steps_per_epoch)),
            total=steps_per_epoch,
            desc=f"epoch {epoch}",
            ascii="░▒█",
        ):
            # region on_train_batch_begin, on_batch_begin
            enc_callback.on_batch_begin(batch, logs=logs)
            dec_callback.on_batch_begin(batch, logs=logs)
            enc_callback.on_train_batch_begin(batch, logs=logs)
            dec_callback.on_train_batch_begin(batch, logs=logs)
            # endregion

            batch_loss, train_acc = model.training_step(
                inp, targ, target_token, enc_hidden
            )

            # region on_train_batch_end, on_batch_end
            enc_callback.on_train_batch_end(batch, logs=logs)
            dec_callback.on_train_batch_end(batch, logs=logs)
            enc_callback.on_batch_end(batch, logs=logs)
            dec_callback.on_batch_end(batch, logs=logs)
            # endregion

            with writer.as_default():
                tf.summary.scalar(
                    "batch loss", batch_loss, step=model.optimizer.iterations
                )
                tf.summary.scalar(
                    "train accuracy", train_acc, step=model.optimizer.iterations
                )

        model.manager.save()
        model.acc_metric.reset_state()

        # region on_epoch_end
        enc_callback.on_epoch_end(epoch, logs=logs)
        dec_callback.on_epoch_end(epoch, logs=logs)
        # endregion

        epoch += 1

    # region on_train_end
    enc_callback.on_train_end(logs=logs)
    dec_callback.on_train_end(logs=logs)
    # endregion

    sys.stdout.flush()


def predict(sentence, path=os.path.dirname(os.getcwd())):
    input_tokenizer = tokenize(path + "/" + VOCAB_INP_PATH)
    target_tokenizer = tokenize(path + "/" + VOCAB_TAR_PATH)

    model.checkpoint.restore(tf.train.latest_checkpoint(path + "/" + CHECKPOINT_DIR))

    sentence = (data_util.preprocess_sentence(sentence),)
    inputs = input_tokenizer.texts_to_sequences(sentence)
    inputs = tf.keras.preprocessing.sequence.pad_sequences(
        inputs, maxlen=MAX_LENGTH_INP, padding="post"
    )
    inputs = tf.convert_to_tensor(inputs)

    result = ""

    hidden = [tf.zeros((1, gConfig["layer_size"]))]
    enc_out, enc_hidden = model.encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([target_tokenizer.word_index[data_util.SOS]], 0)

    for _ in range(MAX_LENGTH_TAR):
        predictions, dec_hidden, _ = model.decoder(dec_input, dec_hidden, enc_out)
        predicted_id = tf.argmax(predictions[0]).numpy()
        if target_tokenizer.index_word[predicted_id] == data_util.EOS:
            break
        result += str(target_tokenizer.index_word[predicted_id]) + " "
        dec_input = tf.expand_dims([predicted_id], 0)

    return result


if __name__ == "__main__":
    EPOCHS = gConfig["epochs"]
    BATCH_SIZE = gConfig["batch_size"]

    writer = tf.summary.create_file_writer(gConfig["log_dir"])

    input_tensor, _, target_tensor, target_token = read_data(gConfig["seq_data"])
    BUFFER_SIZE = len(input_tensor)
    steps_per_epoch = BUFFER_SIZE // BATCH_SIZE
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(
        BUFFER_SIZE
    )
    dataset = (
        dataset.batch(BATCH_SIZE, drop_remainder=True)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    enc_hidden = model.encoder.initialize_hidden_state()

    LRTracker = callbacks.AdamLearningRateTracker(writer=writer)
    _callbacks = [LRTracker]
    enc_callback = tf.keras.callbacks.CallbackList(
        _callbacks, add_history=True, model=model.Encoder
    )
    dec_callback = tf.keras.callbacks.CallbackList(
        _callbacks, add_history=True, model=model.Decoder
    )
    logs = {}

    # region on_train_begin
    enc_callback.on_train_begin(logs=logs)
    dec_callback.on_train_begin(logs=logs)
    # endregion

    train()

    enc_hist = dec_hist = None
    for enc_cb, dec_cb in zip(enc_callback, dec_callback):
        if isinstance(enc_cb, tf.keras.callbacks.History):
            enc_hist = enc_cb
        if isinstance(dec_cb, tf.keras.callbacks.History):
            dec_hist = dec_cb

    assert enc_hist and dec_hist is not None
