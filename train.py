# coding=utf-8
import glob

import pandas
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import getConfig
from core.Model import Transformer
from core.Optimizer import ScheduledLearningRate

# region Config
gConfig = {}
gConfig = getConfig.get_config()

# Preprocessing - Data
TSV_PATH = gConfig["tsv_path"]

# Preprocessing - Tokens
SOS = gConfig["sos"]
EOS = gConfig["eos"]

# Hyperparameters - Model
NUM_LAYERS = gConfig["num_layers"]
D_MODEL = gConfig["d_model"]
NUM_HEADS = gConfig["num_heads"]
DFF = gConfig["dff"]
MAX_LENGTH = gConfig["max_length"]
DROPOUT_RATE = gConfig["dropout_rate"]

# Callbacks
LOGS_DIR = gConfig["logs_dir"]
MODELS_DIR = gConfig["models_dir"]

# Hyperparameters - Training
BATCH_SIZE = gConfig["batch_size"]
EPOCHS = gConfig["epoch"]
# endregion


def prepare_dataset(tsv):
    def prepare_batch(inputs, targets):
        inputs = inputs_tokenizer(inputs)
        inputs = inputs[:, :MAX_LENGTH]
        inputs = inputs.to_tensor()

        targets = targets_tokenizer(targets)
        targets = targets[:, : (MAX_LENGTH + 2)]
        targets_inputs = targets[:, :-1].to_tensor()
        targets_labels = targets[:, 1:].to_tensor()

        return (inputs, targets_inputs), targets_labels

    def make_batches(ds, BUFFER_SIZE):
        return (
            ds.cache()
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .map(prepare_batch, tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )

    df = pandas.read_csv(tsv, sep="\t", header=None, on_bad_lines="warn")
    df.iloc[:, -1:] = df.iloc[:, -1:].applymap(
        (lambda x: f"{SOS} {str(x).rstrip()} {EOS}")
    )
    train_ds, val_ds = train_test_split(df, test_size=0.2, shuffle=False)

    ds = df.values.tolist()
    train_ds = train_ds.values.tolist()
    val_ds = val_ds.values.tolist()

    inputs, targets = zip(*ds)
    train_inputs, train_targets = zip(*train_ds)
    val_inputs, val_targets = zip(*val_ds)

    inputs_tokenizer.adapt(list(inputs))
    targets_tokenizer.adapt(list(targets))

    train_ds = tf.data.Dataset.from_tensor_slices(
        (list(train_inputs), list(train_targets))
    )
    val_ds = tf.data.Dataset.from_tensor_slices((list(val_inputs), list(val_targets)))

    BUFFER_SIZE = len(train_inputs)

    train_batches = make_batches(train_ds, BUFFER_SIZE)
    val_batches = make_batches(val_ds, BUFFER_SIZE)

    return train_batches, val_batches


def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match) / tf.reduce_sum(mask)


def train(train_batches, val_batches, transformer, optimizer):
    transformer.compile(
        loss=masked_loss,
        optimizer=optimizer,
        metrics=[masked_accuracy],
    )

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="%s/model.{epoch:02d}-{val_loss:.4f}.h5" % MODELS_DIR,
        save_best_only=True,
        save_weights_only=True,
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOGS_DIR)
    callbacks = [model_checkpoint_callback, tensorboard_callback]

    transformer.fit(
        train_batches,
        epochs=EPOCHS,
        validation_data=val_batches,
        callbacks=callbacks,
    )


def main():
    # Global variables are generally considered as a bad practice
    global inputs_tokenizer, targets_tokenizer

    inputs_tokenizer = tf.keras.layers.TextVectorization(standardize=None, ragged=True)
    targets_tokenizer = tf.keras.layers.TextVectorization(standardize=None, ragged=True)

    train_batches, val_batches = prepare_dataset(TSV_PATH)

    transformer = Transformer(
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dff=DFF,
        input_vocab_size=inputs_tokenizer.vocabulary_size(),
        target_vocab_size=targets_tokenizer.vocabulary_size(),
        dropout_rate=DROPOUT_RATE,
    )

    for (inputs, targets_inputs), _ in train_batches.take(1):
        transformer((inputs, targets_inputs))
    transformer.summary()

    weights = sorted(glob.glob(f"{MODELS_DIR}/*.h5"))
    if weights:
        transformer.load_weights(f"{weights[-1]}")
        print(f"Latest weights {weights[-1]} restored!")

    learning_rate = ScheduledLearningRate(D_MODEL)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    train(train_batches, val_batches, transformer, optimizer)


if __name__ == "__main__":
    main()
