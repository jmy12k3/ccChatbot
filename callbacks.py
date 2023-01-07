import tensorflow as tf

import model


class AdamLearningRateTracker(tf.keras.callbacks.Callback):
    def __init__(self, writer):
        self.writer = writer
        self.iteration = model.optimizer.iterations

    def on_batch_begin(self, batch, logs=None):
        logs = logs or {}
        lr = model.optimizer.lr
        wd = model.optimizer.weight_decay
        beta_1 = model.optimizer.beta_1
        beta_2 = model.optimizer.beta_2
        t = tf.cast(model.optimizer.iterations, tf.float32) + 1
        if wd > 0:
            lr = tf.keras.backend.eval(
                lr
                * (1.0 / (1.0 + wd * tf.cast(model.optimizer.iterations, tf.float32)))
            )
        lr_t = lr * (
            tf.math.sqrt(1.0 - tf.math.pow(beta_2, t)) / (1.0 - tf.math.pow(beta_1, t))
        )
        with self.writer.as_default():
            tf.summary.scalar("learning rate", lr_t, step=self.iteration)
