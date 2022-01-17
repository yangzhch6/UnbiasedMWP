import random
import json
import copy
import re
import numpy as np
from copy import deepcopy
import pprint

from generate_unbias_data import prefix_to_infix

def load_json(filename):
    with open(filename,'r') as f:
        json_data = json.load(f)
    return json_data

def write_json(filename, data):
    with open(filename,'w') as f:
        json_data = json.dumps(data, indent=4, ensure_ascii=False)
        f.write(json_data)

if __name__ == "__main__":
    prefix = "data_dmai"
    seg_len = 30
    data = load_json(prefix + ".json")
    for i in range(int(len(data)/seg_len)+1):
        write_json(prefix + "/" + prefix + str(i) + ".json", data[seg_len*i : seg_len*i + seg_len])