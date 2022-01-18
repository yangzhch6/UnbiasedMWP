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

train = read_json('train.json')
train_all = read_json('train_all.json')
valid = read_json('valid.json')
test = read_json('test.json')

print(len(train_all))
print(len(train), len(valid), len(test))

ori_train_all = read_json('../train_all.json')
ori_train = read_json('../train.json')
ori_valid = read_json('../valid.json')
ori_test = read_json('../test.json')


print(len(ori_train_all))
print(len(ori_train), len(ori_valid), len(ori_test))

for line in train_all:
    find = find_data(ori_train_all, line['id'])
    if find != line:
        print(line['id'])

for line in ori_train_all:
    find = find_data(train_all, line['id'])
    if find != line:
        print(line['id'])

for line in test:
    find = find_data(ori_test, line['id'])
    if find != line:
        print(line['id'])

for line in ori_test:
    find = find_data(test, line['id'])
    if find != line:
        print(line['id'])


for line in train+valid:
    find = find_data(train_all, line['id'])
    if find != line:
        print(line['id'])