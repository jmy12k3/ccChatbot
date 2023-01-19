# coding=utf-8
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import getConfig
from core.Model import Transformer
from core.Optimizer import ScheduledLearningRate

# region Config

gConfig = {}
gConfig = getConfig.get_config()

TSV_PATH = gConfig["tsv_path"]

SOS = gConfig["sos"]
EOS = gConfig["eos"]
UNK = gConfig["unk"]

NUM_LAYERS = gConfig["num_layers"]
D_MODEL = gConfig["d_model"]
NUM_HEADS = gConfig["num_heads"]
DFF = gConfig["dff"]
MAX_LENGTH = gConfig["max_length"]
DROPOUT_RATE = gConfig["dropout_rate"]

LOG_DIR = gConfig["log_dir"]
MODEL_DIR = gConfig["model_dir"]

BATCH_SIZE = gConfig["batch_size"]
EPOCHS = gConfig["epoch"]

# endregion


def prepare_dataset(tsv, qn_tokenizer, ans_tokenizer):
    # Tokenize, filter and pad sentences.
    def prepare_batch(qn, ans):
        qn = qn_tokenizer(qn)
        qn = qn[:, :MAX_LENGTH]
        qn = qn.to_tensor()

        ans = ans_tokenizer(ans)
        ans = ans[:, : (MAX_LENGTH + 2)]
        ans_inputs = ans[:, :-1].to_tensor()
        ans_labels = ans[:, 1:].to_tensor()

        return (qn, ans_inputs), ans_labels

    # Create batches and cache the dataset to memory to get a speedup while reading from it. # noqa E501
    def make_batches(ds):
        return (
            ds.cache()
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .map(prepare_batch, tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )

    # Read the dataset.
    df = pd.read_csv(tsv, sep="\t", nrows=0).columns
    df = pd.read_csv(
        tsv, sep="\t", converters={col: (lambda x: UNK if not x else x) for col in df}
    )

    # Add SOS and EOS to the answers.
    df.iloc[:, -1:] = df.iloc[:, -1:].applymap(
        lambda x: (f"{SOS} {str(x).rstrip()} {EOS}") if x != UNK else UNK
    )

    # Split the dataset into train and validation sets.
    train_ds, val_ds = train_test_split(df, test_size=0.2, random_state=42)

    # Convert the dataset into lists.
    ds = df.values.tolist()
    train_ds = train_ds.values.tolist()
    val_ds = val_ds.values.tolist()

    # Split the questions and answers.
    qn, ans = zip(*ds)
    train_qn, train_ans = zip(*train_ds)
    val_qn, val_ans = zip(*val_ds)

    # Create tf.data.Dataset objects.
    train_ds = tf.data.Dataset.from_tensor_slices((list(train_qn), list(train_ans)))
    val_ds = tf.data.Dataset.from_tensor_slices((list(val_qn), list(val_ans)))

    # Calculate the number of batches.
    BUFFER_SIZE = len(train_qn)

    # Create the vocabulary.
    qn_tokenizer.adapt(list(qn))
    ans_tokenizer.adapt(list(ans))

    # Create batches.
    train_batches = make_batches(train_ds)
    val_batches = make_batches(val_ds)

    return train_batches, val_batches


def masked_loss(label, pred):
    """Calculate the loss for the masked labels."""

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
    """Calculate the accuracy for the masked labels."""

    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match) / tf.reduce_sum(mask)


def train(train_batches, val_batches, transformer):
    # Create a BackupAndRestore callback.
    checkpoint_callback = tf.keras.callbacks.BackupAndRestore(backup_dir=MODEL_DIR)

    # Create a ModelCheckpoint callback.
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="%s/weights.{epoch:02d}-{val_loss:.4f}.h5" % MODEL_DIR,
        save_weights_only=True,
        save_best_only=True,
    )

    # Create a TensorBoard callback.
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)

    # Create the callbacks.
    callbacks = [checkpoint_callback, model_checkpoint_callback, tensorboard_callback]

    # Fit the model.
    transformer.fit(
        train_batches,
        epochs=EPOCHS,
        validation_data=val_batches,
        callbacks=callbacks,
    )


def main():
    # Create the tokenizer.
    qn_tokenizer = tf.keras.layers.TextVectorization(standardize=None, ragged=True)
    ans_tokenizer = tf.keras.layers.TextVectorization(standardize=None, ragged=True)

    # Prepare the dataset.
    train_batches, val_batches = prepare_dataset(TSV_PATH, qn_tokenizer, ans_tokenizer)

    # Create the model.
    transformer = Transformer(
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dff=DFF,
        input_vocab_size=qn_tokenizer.vocabulary_size(),
        target_vocab_size=ans_tokenizer.vocabulary_size(),
        dropout_rate=DROPOUT_RATE,
    )

    # Print the model summary.
    for (q, a_inputs), _ in train_batches.take(1):
        transformer((q, a_inputs))
    transformer.summary()

    # Create the learning rate scheduler.
    learning_rate = ScheduledLearningRate(D_MODEL)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    # Compile the model.
    transformer.compile(
        loss=masked_loss,
        optimizer=optimizer,
        metrics=[masked_accuracy],
    )

    train(train_batches, val_batches, transformer)


if __name__ == "__main__":
    main()
