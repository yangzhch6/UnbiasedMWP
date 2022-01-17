import random
import json
import copy
import re
import os
import numpy as np
from copy import deepcopy
import pprint
from tqdm import tqdm

def read_json(filename):
    with open(filename,'r') as f:
        json_data = json.load(f)
    return json_data


def write_json(filename, data):
    with open(filename,'w') as f:
        json_data = json.dumps(data, indent=4, ensure_ascii=False)
        f.write(json_data)

def count_output(data):
    len_count = dict()
    for line in data:
        output_len = len(line["output_prefix"].split())
        if output_len in len_count:
            len_count[output_len] += 1
        else:
            len_count[output_len] = 1
    return len_count

if __name__ == "__main__":
    unbias_data = list()
    ori_data = list()

    unbias_data += read_json("train.json")
    unbias_data += read_json("valid.json")
    unbias_data += read_json("test.json")

    ori_data += read_json("train_ori.json")
    ori_data += read_json("valid_ori.json")
    ori_data += read_json("test_ori.json")

    unbias_count = count_output(unbias_data)
    ori_count = count_output(ori_data)

    print(unbias_count)
    print(ori_count)

