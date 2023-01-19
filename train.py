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
    # Avoiding the use of lambda functions.
    def apply_unk(x):
        return UNK if not x else x

    # Avoiding the use of lambda functions.
    def apply_sos_eos(x):
        return (f"{SOS} {str(x).rstrip()} {EOS}") if x != UNK else UNK

    # Tokenize, clamp, and pad the questions and answers.
    def prepare_batch(qn, ans):
        qn = qn_tokenizer(qn)
        qn = qn[:, :MAX_LENGTH]
        qn = qn.to_tensor()

        ans = ans_tokenizer(ans)
        ans = ans[:, : (MAX_LENGTH + 2)]
        ans_inputs = ans[:, :-1].to_tensor()
        ans_labels = ans[:, 1:].to_tensor()

        return (qn, ans_inputs), ans_labels

    # Cache, shuffle, batch, map and prefetch the dataset.
    def make_batches(ds):
        return (
            ds.cache()
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .map(prepare_batch, tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )

    # Read the tsv file header.
    df = pd.read_csv(tsv, sep="\t", nrows=0).columns

    # Read the tsv file and apply the UNK token to the empty cells.
    df = pd.read_csv(tsv, sep="\t", converters={col: apply_unk for col in df})

    # Apply the SOS and EOS tokens to the answers.
    df.iloc[:, -1:] = df.iloc[:, -1:].applymap(apply_sos_eos)

    # Random_state is set to 42 for reproducibility.
    train_ds, val_ds = train_test_split(df, test_size=0.2, random_state=42)

    ds = df.values.tolist()
    train_ds = train_ds.values.tolist()
    val_ds = val_ds.values.tolist()

    qn, ans = zip(*ds)
    train_qn, train_ans = zip(*train_ds)
    val_qn, val_ans = zip(*val_ds)

    train_ds = tf.data.Dataset.from_tensor_slices((list(train_qn), list(train_ans)))
    val_ds = tf.data.Dataset.from_tensor_slices((list(val_qn), list(val_ans)))

    # Avoiding the use of len(train_qn).
    BUFFER_SIZE = train_ds.cardinality().numpy()

    qn_tokenizer.adapt(list(qn))
    ans_tokenizer.adapt(list(ans))

    train_batches = make_batches(train_ds)
    val_batches = make_batches(val_ds)

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


def train(train_batches, val_batches, transformer):
    # Save the model every epoch in case of interruption.
    checkpoint_callback = tf.keras.callbacks.BackupAndRestore(backup_dir=MODEL_DIR)

    # Save the weights with the lowest validation loss for inference.
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="%s/weights.{epoch:02d}-{val_loss:.4f}.h5" % MODEL_DIR,
        save_weights_only=True,
        save_best_only=True,
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)

    callbacks = [checkpoint_callback, model_checkpoint_callback, tensorboard_callback]

    transformer.fit(
        train_batches,
        epochs=EPOCHS,
        validation_data=val_batches,
        callbacks=callbacks,
    )


def main():
    # Avoiding the use of tf.keras.preprocessing.text.Tokenizer.
    qn_tokenizer = tf.keras.layers.TextVectorization(standardize=None, ragged=True)
    ans_tokenizer = tf.keras.layers.TextVectorization(standardize=None, ragged=True)

    train_batches, val_batches = prepare_dataset(TSV_PATH, qn_tokenizer, ans_tokenizer)

    transformer = Transformer(
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dff=DFF,
        input_vocab_size=qn_tokenizer.vocabulary_size(),
        target_vocab_size=ans_tokenizer.vocabulary_size(),
        dropout_rate=DROPOUT_RATE,
    )

    # Run a test batch to initialize the model.
    for (qn, ans_inputs), _ in train_batches.take(1):
        transformer((qn, ans_inputs))

    transformer.summary()

    # Learning rate scheduler as per the paper.
    learning_rate = ScheduledLearningRate(D_MODEL)

    # Adam optimizer beta_1, beta_2 and epsilon values as per the paper.
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    # Masked loss and accuracy as per the paper to ignore padding tokens.
    transformer.compile(
        loss=masked_loss,
        optimizer=optimizer,
        metrics=[masked_accuracy],
    )

    train(train_batches, val_batches, transformer)


if __name__ == "__main__":
    main()
