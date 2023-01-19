# coding=utf-8
import os

import hanlp
from text_cleaner import remove
from text_cleaner.processor.chinese import CHINESE_SYMBOLS_AND_PUNCTUATION
from text_cleaner.processor.common import (
    GENERAL_PUNCTUATION,
    SYMBOLS_AND_PUNCTUATION_EXTENSION,
)

from config import getConfig

# region Config

gConfig = {}
gConfig = getConfig.get_config()

RESOURCE_PATH = gConfig["resource_path"]
TSV_PATH = gConfig["tsv_path"]

# endregion

if not os.path.exists(RESOURCE_PATH):
    raise FileNotFoundError

tok = hanlp.load(hanlp.pretrained.tok.CTB9_TOK_ELECTRA_SMALL)

tsv = open(TSV_PATH, "w")

tsv.write("question\tanswer\n")

with open(RESOURCE_PATH, encoding="utf-8") as f:
    one_conv = ""
    i = 0

    for line in f:
        line = line.strip("\n")
        line = remove(
            line,
            [
                SYMBOLS_AND_PUNCTUATION_EXTENSION,
                GENERAL_PUNCTUATION,
                CHINESE_SYMBOLS_AND_PUNCTUATION,
            ],
        )

        if line == "":
            continue

        if line[0] == "E":
            if one_conv:
                tsv.write(one_conv[:-1] + "\n")
                i += 1
                if i % 1000 == 0:
                    print(f"Processed {i} conversations")
                one_conv = ""

            elif line[0] == "M":
                one_conv = one_conv + " ".join(tok(line.split(" ")[1])) + "\t"

tsv.close()
