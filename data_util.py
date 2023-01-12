# coding=utf-8
import io
import json
import os
import re

import hanlp
import tensorflow as tf
from zhon.hanzi import punctuation

from config import getConfig

SOS = "start"
EOS = "end"

tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
tok.dict_force = {}
tok.dict_combine = {}


# Helper function 1
def preprocess_sentence(w):
    w = f"{SOS} " + w.rstrip() + f" {EOS}"
    return w


# Helper function 2
def clean_sentence(w, pattern=False):
    w = re.sub(r"[%s]+" % punctuation, "", w)
    if pattern:
        _filterDict = {"": ""}
        filterDict = dict((re.escape(k), v) for k, v in _filterDict.items())
        pattern = re.compile("|".join(filterDict.keys()))
        w = pattern.sub(lambda x: filterDict[re.escape(x.group(0))], w)
    return w


# Reference:
# https://github.com/zhaoyingjun/chatbot/blob/9533385c5a89053192a6a2f1c05f3d3f13490368/Chatbot-tensowflow2.0/Seq2seqchatbot/data_util.py#L20
def create_sequence():
    if not os.path.exists(RESOURCE_DATA):
        print(f"Could not find Corpus. Make sure that it is located at {SEQ_DATA}")
        exit()

    seq = open(SEQ_DATA, "w")
    with open(RESOURCE_DATA, encoding="utf-8") as f:
        one_conv = ""
        i = 0
        for line in f:
            line = line.strip("\n")
            if line == "":
                continue
            if line[0] == "E":
                if one_conv:
                    seq.write(one_conv[:-1] + "\n")
                one_conv = ""
                i += 1
                if i % 1000 == 0:
                    print(f"Processed: {i}")
            elif line[0] == "M":
                one_conv = (
                    one_conv
                    + str(clean_sentence(" ".join(tok(line.split(" ")[1]))))
                    + "\t"
                )
    seq.close()


# Reference:
# https://github.com/zhaoyingjun/chatbot/blob/9533385c5a89053192a6a2f1c05f3d3f13490368/Chatbot-tensowflow2.0/Seq2seqchatbot/data_util.py#L51
def create_vocab(lang, vocab_path, vocab_size):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(lang)
    vocab = json.loads(tokenizer.to_json(ensure_ascii=False))
    vocab["index_word"] = tokenizer.index_word
    vocab["word_index"] = tokenizer.word_index
    vocab["document_count"] = tokenizer.document_count
    vocab = json.dumps(vocab, ensure_ascii=False)
    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write(vocab)
    f.close()
    print("Dictionary saved in: {}".format(vocab_path))


if __name__ == "__main__":
    gConfig = {}
    gConfig = getConfig.get_config()

    RESOURCE_DATA = gConfig["resource_data"]
    SEQ_DATA = gConfig["seq_data"]
    INPUT_VOCAB_PATH = gConfig["input_vocab_path"]
    TARGET_VOCAB_PATH = gConfig["target_vocab_path"]
    INPUT_VOCAB_SIZE = gConfig["input_vocab_size"]
    TARGET_VOCAB_SIZE = gConfig["target_vocab_size"]

    create_sequence()

    # Subsitute to tf.keras.layers.TextVectorization() as the restriction of tensorflow-metal
    lines = io.open(SEQ_DATA, encoding="utf-8").readlines()
    pairs = [[preprocess_sentence(w) for w in l.split("\t")] for l in lines]
    input, target = zip(*pairs)
    create_vocab(input, INPUT_VOCAB_PATH, INPUT_VOCAB_SIZE)
    create_vocab(target, TARGET_VOCAB_PATH, TARGET_VOCAB_SIZE)
