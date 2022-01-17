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

train_ori = read_json('/data3/yangzhicheng/Math_Word_Problem/data/math23k/train.json')
test_ori = read_json('/data3/yangzhicheng/Math_Word_Problem/data/math23k/test.json')
valid_ori = read_json('/data3/yangzhicheng/Math_Word_Problem/data/math23k/valid.json')

train = read_json('train.json')
test = read_json('test.json')
valid = read_json('valid.json')

for i in range(len(train)):
    if train[i] != train_ori[i]:
        print(train[i]['id'])