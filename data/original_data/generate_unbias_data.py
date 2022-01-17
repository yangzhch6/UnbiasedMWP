import random
import json
import copy
import re
import numpy as np
from copy import deepcopy
import pprint
from tqdm import tqdm
def load_json(filename):
    with open(filename,'r') as f:
        json_data = json.load(f)
    return json_data

def write_json(filename, data):
    with open(filename,'w') as f:
        json_data = json.dumps(data, indent=4, ensure_ascii=False)
        f.write(json_data)

def cut_sent(para):
    para = re.sub('([。！! ， ． . , ？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！! ， ． . , ？\?][”’])([^，。！! ， ． . ？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")

def unbias_equation(formula):
    op = ['+', '-', '*', '/']
    nums = list()

    # search for all nums it may use
    for t in formula.split():
        if t not in op:
            nums.append(t)

    # generate unbias equations
    unbias_equ = copy.deepcopy(nums)
    for _ in range(1):
        current_len = len(unbias_equ)
        for i in range(current_len):
            for j in range(current_len):
                if i == j:
                    continue
                
                if j > i:
                    op = ['+', '-', '*', '/']
                else:
                    op = ['-', '/']

                for op_ in op:
                    equ_i = unbias_equ[i]
                    equ_j = unbias_equ[j]
                    if i >= len(nums):
                        equ_i = '( ' + equ_i + ' )'
                    if j >= len(nums):
                        equ_j = '( ' + equ_j + ' )'   
                    expression = equ_i + ' ' + op_ + ' ' + equ_j
                    if expression not in unbias_equ:
                        unbias_equ.append(expression)
    return unbias_equ


def prefix_to_infix(prefix):
    prefix.reverse()
    op = ['+', '-', '*', '/']
    equ = list()
    nums = list()
    for i in range(len(prefix)):
        if prefix[i] in op:
            left = nums.pop()
            right = nums.pop()
            equ_node = left + ' ' + prefix[i] + ' ' + right
            equ.append(equ_node)
            if i != 0:
                equ_node = '( ' + equ_node + ' )'
            nums.append(equ_node)
        else:
            nums.append(prefix[i])

    # print(len(equ), prefix)
    original_infix = equ[-1]
    # print(original_infix)
    equ_transform = list()
    for exp_ in equ:
        exp = copy.deepcopy(exp_).split()
        equ_tmp = list()
        change_equ(exp, 0, equ_tmp)
        equ_transform += equ_tmp
    equ = equ + equ_transform 
    # print(original_infix)
    # print("--------------------------")
    return equ, original_infix


def change_equ(equ, idx, equ_list):
    if len(equ) > 8:
        return

    while idx < len(equ) and equ[idx] not in ['+', '-', '*', '/']:
        idx += 1
    
    if idx >= len(equ):
        return
    
    for op in ['+', '-', '*', '/']:
        if op != equ[idx]:
            add_line = copy.deepcopy(equ)
            add_line[idx] = op
            equ_list.append(' '.join(add_line))
        change_equ(equ, idx+1, equ_list)


def merge(a, b, nums, q_nums_len):
    ignore_N = ['N'+str(i) for i in range(len(nums.split())-q_nums_len, len(nums.split()))]
    all = copy.deepcopy(a)
    for e in b:
        if e not in all:
            append = True
            for N_str in ignore_N:
                if N_str in e:
                    append = False
            if append:
                all.append(e)
    return all


def filt_data(data):
    data_dmai = list()
    data_else = list()
    for line in data:
        try:
            int(line["id"])
            data_else.append(line)
        except:
            data_dmai.append(line)
    return data_dmai, data_else

def q_num(question_):
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    question = copy.deepcopy(question_)
    num_count = 0
    pos = re.search(pattern, question)
    while(pos):
        num_count += 1
        # nums.append(s[pos.start():pos.end()])
        question = question[:pos.start()] + ' [NUM] ' + question[pos.end():] # 将数字改写为[NUM] token
        pos = re.search(pattern, question)
    return num_count

if __name__ == "__main__":
    data_train = load_json("train.json")
    data_valid = load_json("valid.json")
    data_test = load_json("test.json")
    data = data_train + data_valid + data_test

    print("## DATA LEN:", len(data))
    data_unbias = list()

    for line in data:
        # print(line["id"])
        line_unbias = dict()
        line_unbias["id"] = line["id"]
        line_unbias["original_text"] = line["original_text"]
        line_unbias["mask_text"] = line["mask_text"]
        
        cut_sentence = cut_sent(line["original_text"])
        line_unbias["context"] = ''.join(cut_sentence[:-1])
        line_unbias["original_question"] = cut_sentence[-1]

        line_unbias["nums"] = line["nums"]

        equ_unbias = unbias_equation(line["output_prefix"])
        equ_sub, original_infix = prefix_to_infix(line["output_prefix"].split())
        equ_unbias = merge(equ_unbias, equ_sub, line["nums"], q_num(cut_sentence[-1]))
        equ_unbias = {equ:"" for equ in equ_unbias}
        line_unbias["equ_unbias"] = equ_unbias
        
        line_unbias["equ_unbias"][original_infix] = cut_sentence[-1]
        
        data_unbias.append(line_unbias)
    
    data_dmai, data_else = filt_data(data_unbias)

    write_json('data_dmai.json', data_dmai)
    write_json('data_else.json', data_else)


    # train_ids = [line["id"] for line in data_train]
    # valid_ids = [line["id"] for line in data_valid]
    # test_ids = [line["id"] for line in data_test]

    # train_unbias = list()
    # valid_unbias = list()
    # test_unbias = list()

    # for line in data_unbias:
    #     if line["id"] in train_ids:
    #         train_unbias.append(line)
    #     if line["id"] in valid_ids:
    #         valid_unbias.append(line)
    #     if line["id"] in test_ids:
    #         test_unbias.append(line)

    write_json('train_unbias.json', data_unbias[:len(data_train)])
    write_json('valid_unbias.json', data_unbias[len(data_train):len(data_train)+len(data_valid)])
    write_json('test_unbias.json', data_unbias[len(data_train)+len(data_valid):])

    # print(len(train_ids), len(valid_ids), len(test_ids))
    print(len(data_train), len(data_valid), len(data_test))
    print(len(data_unbias[:len(data_train)]), len(data_unbias[len(data_train):len(data_train)+len(data_valid)]), len(data_unbias[len(data_train)+len(data_valid):]))