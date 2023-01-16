# coding=utf-8
import os
import re

import hanlp
from zhon.hanzi import punctuation

from config import getConfig

SOS = "[START]"
EOS = "[END]"

# region CONSTANTS
gConfig = {}
gConfig = getConfig.get_config()

# Preprocessing related
RESOURCE_PATH = gConfig["resource_path"]
SEQ_PATH = gConfig["seq_path"]
# endregion

tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
tok.dict_force = {}
tok.dict_combine = {}


def preprocess_sentence(w):
    w = f"{SOS} " + w.rstrip() + f" {EOS}"
    return w


def sequencer(resource, seq_data):
    if not os.path.exists(resource):
        print(f"Could not find Corpus. Make sure that it is located at {seq_data}")
        exit()

    seq = open(seq_data, "w")
    with open(resource, encoding="utf-8") as f:
        one_conv = ""
        i = 0
        for line in f:
            line = line.strip("\n")

            """
            _filterDict = {"": ""}
            filterDict = dict((re.escape(k), v) for k, v in _filterDict.items())
            pattern = re.compile("|".join(filterDict.keys()))
            line = pattern.sub(lambda x: filterDict[re.escape(x.group(0))], line)
            """

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
                    + re.sub(
                        r"[%s]+" % punctuation,
                        "",
                        str(" ".join(tok(line.split(" ")[1]))),
                    )
                    + "\t"
                )
    seq.close()


if __name__ == "__main__":
    sequencer(RESOURCE_PATH, SEQ_PATH)
