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

train = read_json('train.json')
test = read_json('test.json')
valid = read_json('valid.json')

train_all = train+valid
write_json('train_all.json', train_all)