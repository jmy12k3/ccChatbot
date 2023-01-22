import tensorflow as tf


def get_ckpt(model, optimizer, checkpoint_dir, patience=None):
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, checkpoint_dir, max_to_keep=patience
    )

    return ckpt, ckpt_manager


class Checkpoint(tf.keras.callbacks.Callback):
    def __init__(self, ckpt, ckpt_manager):
        self.ckpt = ckpt
        self.ckpt_manager = ckpt_manager

    def on_train_begin(self, logs=None):
        ckpt = self.ckpt_manager.latest_checkpoint

        if ckpt:
            self.ckpt.restore(ckpt)
            print(f"Restored from {ckpt}")
        else:
            print("Initializing from scratch.")

    def on_epoch_end(self, epoch, logs=None):
        self.ckpt_manager.save()
