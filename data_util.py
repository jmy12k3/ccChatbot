# coding=utf-8
import io
import json
import os
import re

import hanlp
import tensorflow as tf
from zhon.hanzi import punctuation

from config import getConfig

gConfig = {}
gConfig = getConfig.get_config()

SOS = "[START] "
EOS = " [END]"

resource_path = gConfig["resource_data"]
seq_path = gConfig["seq_data"]
vocab_inp_path = gConfig["vocab_inp_path"]
vocab_tar_path = gConfig["vocab_tar_path"]
vocab_inp_size = gConfig["vocab_inp_size"]
vocab_tar_size = gConfig["vocab_tar_size"]

# https://hanlp.hankcs.com/docs/api/hanlp/pretrained/tok.html
tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
# https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/tok_stl.ipynb
tok.dict_force = {}
tok.dict_combine = {}


def preprocess_sentence(w):
    w = SOS + w.rstrip() + EOS
    return w


def clean_sentence(w, pattern=False):
    w = re.sub(r"[%s]+" % punctuation, "", w)
    w = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", w)
    if pattern:
        _filterDict = {"": ""}
        filterDict = dict((re.escape(k), v) for k, v in _filterDict.items())
        pattern = re.compile("|".join(filterDict.keys()))
        w = pattern.sub(lambda x: filterDict[re.escape(x.group(0))], w)
    return w


def predata_util():
    if not os.path.exists(resource_path):
        print(f"Could not find corpus. Please confirm it is located at {resource_path}")
        exit()

    seq_train = open(seq_path, "w")
    with open(resource_path, encoding="utf-8") as f:
        one_conv = ""
        i = 0
        for line in f:
            line = line.strip("\n")
            if line == "":
                continue
            if line[0] == gConfig["e"]:
                if one_conv:
                    seq_train.write(one_conv[:-1] + "\n")
                one_conv = ""
                i += 1
                if i % 1000 == 0:
                    print(f"Processed: {i}")
            elif line[0] == gConfig["m"]:
                one_conv = (
                    one_conv
                    + str(clean_sentence(" ".join(tok(line.split(" ")[1]))))
                    + "\t"
                )
    seq_train.close()


def create_vocab(lang, vocab_path, vocab_size):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token=3)
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
    predata_util()
    lines = io.open(seq_path, encoding="utf-8").readlines()
    word_pairs = [[preprocess_sentence(w) for w in l.split("\t")] for l in lines]
    input_lang, target_lang = zip(*word_pairs)
    create_vocab(input_lang, vocab_inp_path, vocab_inp_size)
    create_vocab(target_lang, vocab_tar_path, vocab_tar_size)
