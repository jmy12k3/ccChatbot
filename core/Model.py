import tensorflow as tf

from config import Config
from core import Layers

# region Config

config = Config.config()

LOG_DIR = config["log_dir"]

# endregion

writer = tf.summary.create_file_writer(LOG_DIR)


def masked_loss(y_true, y_pred):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(True, reduction="none")
    loss = loss_fn(y_true, y_pred)

    mask = tf.cast(y_true != 0, loss.dtype)
    loss *= mask

    return tf.math.reduce_sum(loss) / tf.math.reduce_sum(mask)


def masked_acc(y_true, y_pred):
    y_pred = tf.math.argmax(y_pred, -1)
    y_pred = tf.cast(y_pred, y_true.dtype)

    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)

    return tf.math.reduce_sum(match) / tf.math.reduce_sum(mask)


def perplexity(loss):
    return tf.math.exp(loss)


class Translator(tf.keras.Model):
    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun

    def __init__(self, emb_dim, units, context_text_processor, target_text_processor):
        super().__init__()
        encoder = Layers.Encoder(context_text_processor, emb_dim, units)
        decoder = Layers.Decoder(target_text_processor, emb_dim, units)

        self.encoder = encoder
        self.decoder = decoder

        self.loss_tracker = tf.keras.metrics.Mean("loss")
        self.acc_metric = tf.keras.metrics.Mean("acc")
        self.pp_metric = tf.keras.metrics.Mean("pp")

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = masked_loss(y, y_pred)

        acc = masked_acc(y, y_pred)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_tracker(loss)
        self.acc_metric(acc)

        with writer.as_default(self._train_counter):
            tf.summary.scalar("loss", self.loss_tracker.result())
            tf.summary.scalar("acc", self.acc_metric.result())

        return {
            "loss": self.loss_tracker.result(),
            "acc": self.acc_metric.result(),
        }

    def test_step(self, data):
        x, y = data

        y_pred = self(x, training=False)

        loss = masked_loss(y, y_pred)
        acc = masked_acc(y, y_pred)
        pp = perplexity(loss)

        self.loss_tracker(loss)
        self.acc_metric(acc)
        self.pp_metric(pp)

        return {
            "loss": self.loss_tracker.result(),
            "acc": self.acc_metric.result(),
            "pp": self.pp_metric.result(),
        }

    def call(self, inputs):
        context, x = inputs
        context = self.encoder(context)
        logits = self.decoder(context, x)

        try:
            del logits._keras_mask
        except AttributeError:
            pass

        return logits

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_metric, self.pp_metric]
