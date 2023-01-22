import pickle

import einops
import hanlp
import tensorflow as tf

from config import getConfig
from core import Checkpoint
from core.Layers import Decoder
from core.Model import Translator
from core.Module import ShapeChecker

# region Config

gConfig = {}
gConfig = getConfig.get_config()

CTX_PATH = gConfig["ctx_path"]
TGT_PATH = gConfig["tgt_path"]

MODEL_DIR = gConfig["model_dir"]

MAX_LENGTH = gConfig["max_length"]

UNITS = gConfig["units"]

# endregion

tok = hanlp.load(hanlp.pretrained.tok.CTB9_TOK_ELECTRA_BASE)

context_text_processor = tf.keras.layers.TextVectorization()
target_text_processor = tf.keras.layers.TextVectorization()

try:
    ctx_cache = pickle.load(open(CTX_PATH, "rb"))
    tgt_cache = pickle.load(open(TGT_PATH, "rb"))

    context_text_processor.from_config(ctx_cache["config"])
    context_text_processor.set_weights(ctx_cache["weights"])

    target_text_processor.from_config(tgt_cache["config"])
    target_text_processor.set_weights(tgt_cache["weights"])

except OSError:
    raise FileNotFoundError("Cache not found. Please run train.py first.")

model = Translator(UNITS, context_text_processor, target_text_processor)

ckpt, ckpt_manager = Checkpoint.get_ckpt(
    model, optimizer=None, checkpoint_dir=MODEL_DIR
)

if not ckpt_manager.latest_checkpoint:
    raise FileNotFoundError("Checkpoint not found. Please run train.py first.")
else:
    ckpt.restore(ckpt_manager.latest_checkpoint)


@Decoder.add_method
def get_initial_state(self, context):
    batch_size = tf.shape(context)[0]
    start_tokens = tf.fill([batch_size, 1], self.start_token)
    done = tf.zeros([batch_size, 1], dtype=tf.bool)
    embedded = self.embedding(start_tokens)
    return start_tokens, done, self.rnn.get_initial_state(embedded)[0]


@Decoder.add_method
def tokens_to_text(self, tokens):
    words = self.id_to_word(tokens)
    result = tf.strings.reduce_join(words, axis=-1, separator=" ")
    return result


@Decoder.add_method
def get_next_token(self, context, next_token, done, state, temperature=0.0):
    logits, state = self(context, next_token, state=state, return_state=True)

    if temperature == 0.0:
        next_token = tf.argmax(logits, axis=-1)
    else:
        logits = logits[:, -1, :] / temperature
        next_token = tf.random.categorical(logits, num_samples=1)

    done = done | (next_token == self.end_token)
    next_token = tf.where(done, tf.constant(0, dtype=tf.int64), next_token)

    return next_token, done, state


@Translator.add_method
def translate(self, texts, *, max_length=MAX_LENGTH, temperature=tf.constant(0.0)):
    shape_checker = ShapeChecker()
    context = self.encoder.convert_input(texts)
    shape_checker(context, "batch s units")

    next_token, done, state = self.decoder.get_initial_state(context)

    tokens = tf.TensorArray(tf.int64, size=1, dynamic_size=True)

    for t in tf.range(max_length):
        next_token, done, state = self.decoder.get_next_token(
            context, next_token, done, state, temperature
        )
        shape_checker(next_token, "batch t1")

        tokens = tokens.write(t, next_token)

        if tf.reduce_all(done):
            break

    tokens = tokens.stack()
    shape_checker(tokens, "t batch t1")
    tokens = einops.rearrange(tokens, "t batch 1 -> batch t")
    shape_checker(tokens, "batch t")

    text = self.decoder.tokens_to_text(tokens)
    shape_checker(text, "batch")

    return text


def predict(sentence):
    sentence = " ".join(tok(sentence))

    result = model.translate([sentence])

    return result.numpy()[0].decode("utf-8")


if __name__ == "__main__":
    raise RuntimeError("Please run app.py for inference.")
