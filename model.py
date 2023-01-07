import tensorflow as tf

from config import getConfig

tf.config.run_functions_eagerly(True)

gConfig = {}
gConfig = getConfig.get_config()

CHECKPOINT_DIR = gConfig["model_data"]
VOCAB_INP_SIZE = gConfig["vocab_inp_size"]
VOCAB_TAR_SIZE = gConfig["vocab_tar_size"]

EMBEDDING_DIM = gConfig["embedding_dim"]
UNITS = gConfig["layer_size"]
BATCH_SIZE = gConfig["batch_size"]
LEARNING_RATE = gConfig["learning_rate"]
WEIGHT_DECAY = gConfig["weight_decay"]


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.batch_size = batch_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.enc_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )

    def call(self, x, hidden):
        x_emb = self.embedding(x)
        output, state = self.gru(x_emb, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_units))


class BahdanauAttention(tf.keras.Model):
    def __init__(self, UNITS):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(UNITS)
        self.W2 = tf.keras.layers.Dense(UNITS)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.dec_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, y, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        y = self.embedding(y)
        y = tf.concat([tf.expand_dims(context_vector, 1), y], axis=-1)
        output, state = self.gru(y)
        output = tf.reshape(output, (-1, output.shape[2]))
        y = self.fc(output)
        return y, state, attention_weights

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.dec_units))


def loss_fn(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
encoder = Encoder(VOCAB_INP_SIZE, EMBEDDING_DIM, UNITS, BATCH_SIZE)
decoder = Decoder(VOCAB_TAR_SIZE, EMBEDDING_DIM, UNITS, BATCH_SIZE)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
manager = tf.train.CheckpointManager(
    checkpoint, directory=CHECKPOINT_DIR, max_to_keep=3
)


@tf.function
def training_step(inp, targ, targ_lang, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang.word_index["start"]] * BATCH_SIZE, 1)
        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_fn(targ[:, t], predictions)
            acc_metric.update_state(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)
    step_loss = loss / int(targ.shape[1])
    step_acc = acc_metric.result()
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return step_loss, step_acc


# test loop
# @tf.function
# def test_step(...):
#   ...
