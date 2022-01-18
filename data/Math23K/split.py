from asyncore import write
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

def load_raw_data(filename):  # load the json data to list(dict()) for MATH 23K
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    js = ""
    data = []
    for i, s in enumerate(f):
        js += s
        i += 1
        if i % 7 == 0:  # every 7 line is a json
            data_d = json.loads(js)
            if "千米/小时" in data_d["equation"]:
                data_d["equation"] = data_d["equation"][:-5]
            data.append(data_d)
            js = ""

    return data

def find_data(data, id):
    find_list = list()
    for line in data:
        if line['id'] == id:
            find_list.append(line)
    # assert len(find_list) == 1
    if len(find_list) != 1:
        print(id) 
        return find_list
    return find_list[0]

train = load_raw_data('original/math23k_train.json')
write_json('train_all.json', train)
test = load_raw_data('original/math23k_test.json')
print(len(train), len(test))

train_id = read_json('original/train23k_processed.json')
valid_id = read_json('original/valid23k_processed.json')
test_id = read_json('original/test23k_processed.json')

train_id = [line['id'] for line in train_id]
valid_id = [line['id'] for line in valid_id]
test_id = [line['id'] for line in test_id]

train_split = list()
valid_split = list()
test_split = list()

for id in train_id:
    train_split.append(find_data(train, id))

for id in valid_id:
    valid_split.append(find_data(train, id))

for id in test_id:
    test_split.append(find_data(test, id))

print(len(train_split), len(valid_split), len(test_split))

write_json('train.json', train_split)
write_json('valid.json', valid_split)
write_json('test.json', test_split)

for line in train:
    find_data(train_split+valid_split, line['id'])