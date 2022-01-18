from operator import length_hint
import random
import json
import copy
import re
import os
from this import d
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


def variable_assortment(formula):
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

def sub_and_whole(prefix): # 子表达式及其变种
    prefix = prefix.split()
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
    sub_f = equ[:-1]
    whole_f = [equ[-1]]

    sub_add = list()
    for exp_ in sub_f:
        exp = copy.deepcopy(exp_).split()
        equ_tmp = list()
        change_equ(exp, 0, equ_tmp)
        sub_add += equ_tmp
    sub_f = sub_f + sub_add 

    whole_add = list()
    for exp_ in whole_f:
        exp = copy.deepcopy(exp_).split()
        equ_tmp = list()
        change_equ(exp, 0, equ_tmp)
        whole_add += equ_tmp
    whole_f = whole_f + whole_add 
    # print("--------------------------")
    return sub_f, whole_f, original_infix


def from_infix_to_prefix(expression):
    st = list()
    res = list()
    priority = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
    expression = deepcopy(expression)
    expression.reverse()
    for e in expression:
        if e in [")", "]"]:
            st.append(e)
        elif e == "(":
            c = st.pop()
            while c != ")":
                res.append(c)
                c = st.pop()
        elif e == "[":
            c = st.pop()
            while c != "]":
                res.append(c)
                c = st.pop()
        elif e in priority:
            while len(st) > 0 and st[-1] not in [")", "]"] and priority[e] < priority[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    res.reverse()
    return res


def create_interpretation(root, output_prefix, idx):
    inter = {
        "logic": "",
        "op": "",
        "left": {},
        "right": {}
    } 
    # print(idx)
    # print(output_prefix[idx], output_prefix)
    if output_prefix[idx] in ['+', '-', '*', '/', '^']:
        inter["logic"] = ""
        inter["op"] = output_prefix[idx]
        # print("left")
        left_tree = create_interpretation(root, output_prefix, idx+1)
        inter["left"] = left_tree[0]
        # print("right")
        right_tree = create_interpretation(root, output_prefix, left_tree[1])
        inter["right"] = right_tree[0]
        # print(root)
        return (inter, right_tree[1])
    else: 
        inter["logic"] = "0"
        inter["op"] = output_prefix[idx]
        # print("LEAF")
        # print(root)
        return (inter, idx+1)

# interpretation = create_interpretation({}, out_seq, 0)[0]

def generate_num(s):
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pos = re.search(pattern, s)
    nums = list()
    while(pos):
        nums.append(s[pos.start():pos.end()])
        s = s[:pos.start()] + ' [NUM] ' + s[pos.end():] # 将数字改写为[NUM] token
        pos = re.search(pattern, s)
    return ' '.join(nums)

def extract_line(line):
    extracted = dict()
    for key in line["equ_unbias"]:
        if line["equ_unbias"][key] != "":
            extracted[key] = line["equ_unbias"][key]
    return extracted

def output_original(nums, infix):
    original = infix.split()
    nums = nums.split()
    for i in range(len(original)):
        if original[i][0] == 'N':
            original[i] = nums[int(original[i][1:])]
    return 'x=' + ''.join(original)

def process_ano_line(line):
    processed_lines_all = list()
    extracted = extract_line(line)
    for key in extracted:
        copy_line = deepcopy(line)
        del copy_line["equ_unbias"]
        del copy_line["mask_text"]
        del copy_line["original_question"]
        copy_line["original_text"] = copy_line["context"] + extracted[key]
        copy_line["question"] = extracted[key]
        copy_line["nums"] = generate_num(copy_line["original_text"])
        copy_line["output_infix"] = key
        copy_line["output_prefix"] = ' '.join(from_infix_to_prefix(key.split()))
        copy_line["output_original"] = output_original(copy_line["nums"], key)
        copy_line["interpretation"] = {}

        # control output length
        if len(copy_line["output_prefix"].split()) >= 3:
            processed_lines_all.append(copy_line)

    return processed_lines_all


def find_ori_output(equ, q):
    for output in equ:
        if equ[output] == q:
            return output
    return ""


def generate_ori(line_):
    line = deepcopy(line_)
    line["original_text"] = line["context"] + line["original_question"]
    line["question"] = line["original_question"]
    line["nums"] = generate_num(line["original_text"])
    output = find_ori_output(line["equ_unbias"], line["question"])
    assert output != ""
    line["output_infix"] = output
    line["output_prefix"] = ' '.join(from_infix_to_prefix(output.split()))
    line["output_original"] = output_original(line["nums"], output)
    line["interpretation"] = {}
    del line["equ_unbias"]
    del line["mask_text"]
    del line["original_question"]
    return line


def process_data(ano_data):
    np.random.seed(100)
    random.seed(100)

    processed_data_all = list()
    ori_data = list()

    all_cut1 = -1
    all_cut2 = -1

    idx = 0
    for line in ano_data:
        idx += 1
        
        print(idx)
        print(line["id"])
        lines_all = process_ano_line(line)
        processed_data_all += lines_all
        ori_data.append(generate_ori(line))

        if len(ori_data) == 200:
            all_cut1 = len(processed_data_all)
        if len(ori_data) == 400:
            all_cut2 = len(processed_data_all)

    return processed_data_all, ori_data, [all_cut1, all_cut2]

def extract_line_ablation(line, equ):
    extracted = dict()
    for key in line["equ_unbias"]:
        if key in equ and line["equ_unbias"][key] != "":
            extracted[key] = line["equ_unbias"][key]
    return extracted

def process_ano_line_ablation(line, equ):
    processed_lines_all = list()
    extracted = extract_line_ablation(line, equ)
    for key in extracted:
        copy_line = deepcopy(line)
        del copy_line["equ_unbias"]
        del copy_line["mask_text"]
        del copy_line["original_question"]
        copy_line["original_text"] = copy_line["context"] + extracted[key]
        copy_line["question"] = extracted[key]
        copy_line["nums"] = generate_num(copy_line["original_text"])
        copy_line["output_infix"] = key
        copy_line["output_prefix"] = ' '.join(from_infix_to_prefix(key.split()))
        copy_line["output_original"] = output_original(copy_line["nums"], key)
        copy_line["interpretation"] = {}

        # control output length
        if len(copy_line["output_prefix"].split()) >= 3:
            processed_lines_all.append(copy_line)

    return processed_lines_all

def delete_ori_equ(ori, equ):
    if ori in equ:
        del equ[equ.index(ori)]
    return equ

def ablation_data(ano_data):
    np.random.seed(100)
    random.seed(100)
    ori_data = list()
    va_data = list()
    sub_data = list()
    whole_data = list()

    va_cut = [0, 0]    
    sub_cut = [0, 0]   
    whole_cut = [0, 0]   

    idx = 0
    for line in ano_data:
        idx += 1
        
        print(idx)
        print(line["id"])
        # lines_all = process_ano_line(line)
        ori_line = generate_ori(line)
        ori_data.append(ori_line)

        va_list = variable_assortment(ori_line["output_prefix"])
        sub_f, whole_f, _ = sub_and_whole(ori_line["output_prefix"])
        va_list = delete_ori_equ(ori_line["output_infix"], va_list)
        sub_f = delete_ori_equ(ori_line["output_infix"], sub_f)
        whole_f = delete_ori_equ(ori_line["output_infix"], whole_f)
        
        va_data += process_ano_line_ablation(line, va_list)
        sub_data += process_ano_line_ablation(line, sub_f)
        whole_data += process_ano_line_ablation(line, whole_f)

        lines_all = process_ano_line(line) # 标注人员自己标的东西
        for line_add in lines_all:
            if line_add not in ori_data+va_data+sub_data+whole_data:
                whole_data.append(line_add)

        if len(ori_data) == 200:
            va_cut[0] = len(va_data)
            sub_cut[0] = len(sub_data)
            whole_cut[0] = len(whole_data)
        if len(ori_data) == 400:
            va_cut[1] = len(va_data)
            sub_cut[1] = len(sub_data)
            whole_cut[1] = len(whole_data)
    return ori_data, va_data, sub_data, whole_data, va_cut, sub_cut, whole_cut

def count_exp_len(data):
    length = {'3':0, '4':0, '5':0, '6':0 ,'7':0, '8':0, '9':0, '10':0, '11':0, 'else':0}
    for line in data:
        key = len(line['output_prefix'].split())
        if key <= 11:
            key = str(key)
        else:
            key = 'else'
        length[key] += 1
    return length

if __name__ == "__main__":
    # train = read_json('train_all.json')
    # valid = read_json('valid_all.json')
    # test = read_json('test_all.json')
    # data = train + valid + test
    # print(count_exp_len(data))

    ano_data = read_json("annotated_all.json")
    ori_data, va_data, sub_data, whole_data, va_cut, sub_cut, whole_cut = ablation_data(ano_data)
    print(len(ori_data), len(va_data), len(sub_data), len(whole_data))
    print(va_cut, sub_cut, whole_cut)

    write_json('test_va.json', va_data[:va_cut[0]] + ori_data[:200])
    write_json('valid_va.json', va_data[va_cut[0]:va_cut[1]] + ori_data[200:400])
    write_json('train_va.json', va_data[va_cut[1]:] + ori_data[400:])

    write_json('test_sub.json', sub_data[:sub_cut[0]] + ori_data[:200])
    write_json('valid_sub.json', sub_data[sub_cut[0]:sub_cut[1]] + ori_data[200:400])
    write_json('train_sub.json', sub_data[sub_cut[1]:] + ori_data[400:])

    write_json('test_whole.json', whole_data[:whole_cut[0]] + ori_data[:200])
    write_json('valid_whole.json', whole_data[whole_cut[0]:whole_cut[1]] + ori_data[200:400])
    write_json('train_whole.json', whole_data[whole_cut[1]:] + ori_data[400:])


    # write_json('test_only_va.json', va_data[:va_cut[0]])
    # write_json('valid_only_va.json', va_data[va_cut[0]:va_cut[1]])
    # write_json('train_only_va.json', va_data[va_cut[1]:])

    # write_json('test_only_sub.json', sub_data[:sub_cut[0]])
    # write_json('valid_only_sub.json', sub_data[sub_cut[0]:sub_cut[1]])
    # write_json('train_only_sub.json', sub_data[sub_cut[1]:])

    # write_json('test_only_whole.json', whole_data[:whole_cut[0]])
    # write_json('valid_only_whole.json', whole_data[whole_cut[0]:whole_cut[1]])
    # write_json('train_only_whole.json', whole_data[whole_cut[1]:])

    # all_data, ori_data, all_split = process_data(ano_data)
    # print(len(all_data))
    # write_json("train_all.json", all_data[all_split[1]:])
    # write_json("valid_all.json", all_data[all_split[0]:all_split[1]])
    # write_json("test_all.json", all_data[:all_split[0]])

    # equ_unbias = variable_assortment(line["output_prefix"]) # 长度为3的表达式
    # equ_sub, original_infix = sub_and_whole(line["output_prefix"])

