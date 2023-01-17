# coding=utf-8
import os
import re

import hanlp
from zhon.hanzi import punctuation

from config import getConfig

gConfig = {}
gConfig = getConfig.get_config()

RESOURCE_PATH = gConfig["resource_path"]
TSV_PATH = gConfig["tsv_path"]

tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
tok.dict_force = {}
tok.dict_combine = {}

if not os.path.exists(RESOURCE_PATH):
    print(f" {TSV_PATH}")
    exit()

tsv = open(TSV_PATH, "w")

with open(RESOURCE_PATH, encoding="utf-8") as f:
    one_conv = ""
    i = 0
    for line in f:
        line = line.strip("\n")
        
        # Filter out some special characters
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
                tsv.write(one_conv[:-1] + "\n")
            one_conv = ""
            i += 1
            if i % 1000 == 0:
                print(f"Processed: {i}")
        elif line[0] == "M":
            one_conv = (one_conv + re.sub(r"[%s]+" % punctuation, "", str(" ".join(tok(line.split(" ")[1])))) + "\t")

tsv.close()
