import random
import json
import copy
import re
import nltk
import jieba
from src.tree import *
import jieba.posseg as pseg
import numpy as np
from copy import deepcopy

from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertModel

PAD_token = 0

class MathWP_Dataset(Dataset):
    def __init__(self, data_pairs): 
        super().__init__()
        self.data = data_pairs
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]


def pad_output_seq(seq, max_length, PAD_token):
    # PAD_token = 0
    seq = [line+[PAD_token for _ in range(max_length-len(line))] for line in seq]
    return seq # B x L


def pad_output_seq_list(seq_list, max_length, PAD_token):
    # PAD_token = 0
    seq_list = [pad_output_seq(line_list, max_length, PAD_token) for line_list in seq_list]
    return seq_list # B x [] x L


def my_collate(batch_line):
    batch_line = deepcopy(batch_line)
    token_len = []
    token_ids = []
    token_type_ids = []
    attention_mask = []
    output = []
    output_list = []
    output_len = []
    inter = []
    nums = []
    num_size = []
    num_idx = []
    ids = []

    for line in batch_line:
        token_len.append(len(line["tokens"]))
        token_ids.append(line["token_ids"])
        token_type_ids.append(line["token_type_ids"])
        attention_mask.append(line["attention_mask"])
        output.append(line["output"])
        output_len.append(len(line["output"]))
        output_list.append(line["output_list"])
        inter.append(line["inter_prefix"])
        nums.append(line["nums"])
        num_size.append(len(line["nums"]))
        num_idx.append(line["num_idx"])
        ids.append(line["id"])

    batch = {
        "max_token_len": max(token_len),
        "token_ids": [line[:max(token_len)] for line in token_ids],
        "token_type_ids": [line[:max(token_len)] for line in token_type_ids],
        "attention_mask": [line[:max(token_len)] for line in attention_mask],
        "output": pad_output_seq(output, max(output_len), 0),
        "output_list":pad_output_seq_list(output_list, max(output_len), 0),
        "output_len":output_len,
        "inter": pad_output_seq(inter, max(output_len), -1),
        "nums": nums,
        "num_size":num_size,
        "num_idx": num_idx,
        "id":ids,   
    }
    return batch


class Lang:
    """
    class to save the vocab and two dict: the word->index and index->word
    """
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count word tokens
        self.num_start = 0 # start index of nums and generate nums

    def add_sen_to_vocab(self, sentence):  # add words of sentence to vocab
        for word in sentence:
            pattern = "N\d+|\[NUM\]|\d+"
            if re.search(pattern, word): # 跳过数字,仅包含 + - * / 
                continue
            if word not in self.index2word:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word.append(word)
                self.n_words += 1
            else:
                self.word2count[word] += 1

    def trim(self, min_count):  # trim words betlow a certain count threshold
        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.index2word), len(keep_words) / len(self.index2word)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count default tokens

        for word in keep_words:
            self.word2index[word] = self.n_words
            self.index2word.append(word)
            self.n_words += 1

    def build_input_lang(self, trim_min_count):  # build the input lang vocab and dict
        if trim_min_count > 0:
            self.trim(trim_min_count)
            self.index2word = ["[PAD]", "[NUM]", "[UNK]"] + self.index2word
        else:
            self.index2word = ["[PAD]", "[NUM]"] + self.index2word
        self.word2index = {}
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

    # def build_output_lang(self, generate_num, copy_nums):  # build the output lang vocab and dict
    #     self.index2word = ["[PAD]", "EOS"] + self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] +\
    #                       ["SOS", "[UNK]"]
    #     self.n_words = len(self.index2word)
    #     for i, j in enumerate(self.index2word):
    #         self.word2index[j] = i

    def build_output_lang_for_tree(self, generate_num, copy_nums):  # build the output lang vocab and dict
        self.num_start = len(self.index2word) + 1
        self.index2word = ["[PAD]"] + self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] + ["[UNK]"]
        self.n_words = len(self.index2word)

        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

"""
data格式：
{
    "id":"1",
    "original_text":"镇海雅乐学校二年级的小朋友到一条小路的一边植树．小朋友们每隔2米种一棵树（马路两头都种了树），最后发现一共种了11棵，这条小路长多少米．",
    "segmented_text":"镇海 雅乐 学校 二年级 的 小朋友 到 一条 小路 的 一边 植树 ． 小朋友 们 每隔 2 米 种 一棵树 （ 马路 两头 都 种 了 树 ） ， 最后 发现 一共 种 了 11 棵 ， 这 条 小路 长 多少 米 ．",
    "equation":"x=(11-1)*2",
    "ans":"20"
}
"""
def load_raw_data(filename, linenum=7):  # load the json data to list(dict()) for MATH 23K
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    js = ""
    data = []
    for i, s in enumerate(f):
        js += s
        i += 1
        if i % linenum == 0:  # every [linenum] line is a json
            data_d = json.loads(js)
            if "千米/小时" in data_d["equation"]:
                data_d["equation"] = data_d["equation"][:-5]
            data.append(data_d)
            js = ""

    return data

def read_json(filename):
    with open(filename,'r') as f:
        json_data = json.load(f)
    return json_data

def write_json(filename, data):
    with open(filename,'w') as f:
        json_data = json.dumps(data, indent=4, ensure_ascii=False)
        f.write(json_data)

# 对表达式进行前序遍历
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


def indexes_from_sentence_output(lang, sentence, tree=False):
    res = []
    idx = 0
    for word in sentence:
        if len(word) == 0:
            print("##wrong output:", sentence)
            continue
        if word in lang.word2index:
            res.append(lang.word2index[word])
            idx += 1
        else:
            res.append(lang.word2index["[UNK]"])
            print("##output got [UNK]! :", sentence)
            idx += 1
        
    if "EOS" in lang.index2word and not tree:
        res.append(lang.word2index["EOS"])
    return res
    
def word_is_ignore(words, ignore):
    for ig in ignore:
        if ig in words:
            return True
    return False
    
# 删除问题
def del_question_old(para):
    para = re.sub('([。！! ， ． . , ？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！! ， ． . , ？\?][”’])([^，。！! ， ． . ？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    seg = para.split("\n")
    return ''.join(seg[:-1])#, q_num(seg[-1])

# 删除问题
def del_question(para):
    para = re.sub('([。！! ， ． . , ？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！! ， ． . , ？\?][”’])([^，。！! ， ． . ？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    seg = para.split("\n")
    if q_num(seg[-1]) == 0:
        return ''.join(seg[:-1])
    else:
        return ''.join(seg)

# 计算问题文本中有多少数字
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

"""
面向数据:(仅考虑最基础的数据) # 中英文均可
{
    "id":"1",
    "original_text":"镇海雅乐学校二年级的小朋友到一条小路的一边植树．小朋友们每隔2米种一棵树（马路两头都种了树），最后发现一共种了11棵，这条小路长多少米．",
    "equation":"x=(11-1)*2",
    "ans":"20"
}

输出数据:
{
    'original_text':line["original_text"].strip().split(" "),
    'tokens': tokens, # 对数字改为[NUM]后的token列表
    'output': line["output_prefix"].split(" "),# 先序遍历后的表达式
    'nums': line["nums"].split(' '), # 按顺序记录出现的数字 
    'num_idx': num_idx, # 记录input列表中出现[NUM]的下标
    'id':line['id'],
    'interpretation': line["interpretation"], # 可解释性标注
    'inter_prefix':inter_prefix,# 可解释性标注的前序遍历logic
}
"""
def transfer_num(data, tokenizer, mask=False, trainset=False):
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    attr_use = ["n", "t", "nr", "ns", "nt", "PER", "LOC", "TIME"]
    ignore_ch = [",", "，", ".", "。", "?", "？", "!", "！", ":", "：", "、", ";", "；", "．", 
        "+", "-", "*", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "(", ")", 
        "N", "NUM"]

    pairs = []
    generate_nums = ['1', '2', '3.14', '4']
    generate_nums_dict = {'1':0, '2':0, '3.14':0, '4':0}
    count_num_max = 0 # 一句话里最多有多少个[NUM]

    for line in data:
        ## *** 对数据集有一个假设:
        ##      equation中数字的形式和文本一样,不然该函数无法提取equation中的数字
        
        nums = [] # 按顺序记录文本中所有出现的数字（可重复）--str
        nums_fraction = [] # 只记录文本中出现的分数（可重复）

        s = line["original_text"]
        equation = line["output_original"][2:]

        # jieba会把 单空格 自动分出一个token，利用此特性可方便的做出NUM mask
        pos = re.search(pattern, s)
        while(pos):
            nums.append(s[pos.start():pos.end()])
            s = s[:pos.start()] + ' [NUM] ' + s[pos.end():] # 将数字改写为[NUM] token
            pos = re.search(pattern, s)
        
        if mask:    
            pseg_attr = pseg.cut(s,use_paddle=True) #paddle模式
            seg = jieba.tokenize(s)
            seg = [t[0] for t in seg]
            
            words_attr = {}
            for word, flag in pseg_attr:   
                words_attr[word] = flag

            mask_list = []
            for w in seg:
                if w in words_attr and words_attr[w] in attr_use and not word_is_ignore(w, ignore_ch):
                    mask_list.append('[MASK]')
                else:
                    mask_list.append(w)
            
            mask_text = ''.join(mask_list)
            s = mask_text

        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)
        

        tokens = [] # 最终 tokens
        match_list = []
        num_token = '[num]' # 被tokenizer处理为小写了
        tokens_initial = tokenizer.tokenize(s) # 先使用tokenizer进行初步切割
        # 假设: 被tokenizer切割的'[NUM]'token, 必定以'['开始,以']'结束
        for t in tokens_initial:
            match_text = ''.join(match_list).replace('#', '')
            text_now = match_text + t.replace('#', '')
            if text_now == num_token: # 完全匹配则直接加入
                tokens.append('[NUM]')
                match_list = []
            elif num_token.startswith(text_now): # 匹配前缀
                match_list.append(t)
            else:
                tokens += match_list
                match_list = []
                if num_token.startswith(t.replace('#', '')):
                    match_list.append(t)
                else:
                    tokens.append(t)

        tokens = ['[CLS]'] + tokens + ['[SEP]']

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1: # 如果只出现一次
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) >= 1:
                    res.append("N"+str(nums.index(st_num)))
                # elif nums.count(st_num) > 1:
                #     res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equation)
        out_seq_infix = out_seq
        out_seq_prefix = from_infix_to_prefix(out_seq)

        if "output_prefix" in line: # 若字段包含output_prefix,则其一定为ground truth
            out_seq_prefix = line["output_prefix"].split(" ")
        
        try:
             prefix_list = equivalent_expression_old(out_seq_prefix)
        except:
            print('ignore id:', line['id'])
            continue

        ignore = False # 筛未检测变量的文本
        for s in out_seq_prefix:  
            if s[0].isdigit() and s not in generate_nums and trainset:
                ignore = True
            if s in generate_nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1
        if ignore:
            print('ignore id:', line['id'])
            continue

        count_num_max = max(count_num_max, len(nums))
        
        num_idx = []
        for i, j in enumerate(tokens):
            if j == "[NUM]":
                num_idx.append(i)
        assert len(nums) == len(num_idx)
        if "nums" in line and nums != line["nums"].split(' '):
            print("Different NUM!", line["id"])

        inter_prefix = list()
        pairs.append({
            'original_text':line["original_text"].strip().split(" "),
            'tokens': tokens, # 对数字改为[NUM]后的token列表
            'output': out_seq_prefix, # 先序遍历后的表达式
            'output_list': prefix_list, # 等价表达式list(前序形式)
            'nums': nums, # 按顺序记录出现的数字 
            'num_idx': num_idx, # 记录input列表中出现[NUM]的下标
            'id':line['id'],
            'interpretation': line["interpretation"] if "interpretation" in line else {}, # 可解释性标注
            'inter_prefix':inter_prefix,# 可解释性标注的前序遍历logic
        })
    return pairs, generate_nums, count_num_max#, ignore_list
    
"""
面向数据:(仅考虑最基础的数据) # 中英文均适配
{
"id":"1",
"original_text":"镇海雅乐学校二年级的小朋友到一条小路的一边植树．小朋友们每隔2米种一棵树（马路两头都种了树），最后发现一共种了11棵，这条小路长多少米．",
"equation":"x=(11-1)*2",
"ans":"20"
}

输出数据:
{
    'original_text':line["original_text"].strip().split(" "),
    'tokens': tokens, # 对数字改为[NUM]后的token列表
    'output': line["output_prefix"].split(" "),# 先序遍历后的表达式
    'nums': line["nums"].split(' '), # 按顺序记录出现的数字 
    'num_idx': num_idx, # 记录input列表中出现[NUM]的下标
    'id':line['id'],
    'interpretation': line["interpretation"], # 可解释性标注
    'inter_prefix':inter_prefix,# 可解释性标注的前序遍历logic
}
"""
def process_data_pipeline(train_data_path, valid_data_path, test_data_path, tokenizer, debug=False, mask=False):
    train_data = read_json(train_data_path)
    valid_data = read_json(valid_data_path)
    if test_data_path:
        test_data = read_json(test_data_path)
    else:
        test_data = []

    if debug:
        train_data = train_data[:100]
        valid_data = valid_data[:30]
        test_data = test_data[:30]

    train_data, generate_nums, copy_nums = transfer_num(train_data, tokenizer, mask, trainset=True)
    valid_data, _, _ = transfer_num(valid_data, tokenizer, mask)
    test_data, _, _ = transfer_num(test_data, tokenizer, mask)

    # ignore_list = [ignore_list_train, ignore_list_valid, ignore_list_test]
    return train_data, valid_data, test_data, generate_nums, copy_nums#, ignore_list


def transfer_num_delq(data, tokenizer, mask=False, trainset=False):#, ignore=False):  
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    attr_use = ["n", "t", "nr", "ns", "nt", "PER", "LOC", "TIME"]
    ignore_ch = [",", "，", ".", "。", "?", "？", "!", "！", ":", "：", "、", ";", "；", "．", 
        "+", "-", "*", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "(", ")", 
        "N", "NUM"]

    pairs = []
    generate_nums = ['1', '2', '3.14', '4']
    generate_nums_dict = {'1':0, '2':0, '3.14':0, '4':0}
    count_num_max = 0 # 一句话里最多有多少个[NUM]

    for line in data:
        ## *** 对数据集有一个假设:
        ##      equation中数字的形式和文本一样,不然该函数无法提取equation中的数字
        
        nums = [] # 按顺序记录文本中所有出现的数字（可重复）--str
        nums_fraction = [] # 只记录文本中出现的分数（可重复）

        s = line["original_text"]
        # 控制是否mask question
        if "context" in line:
            s = line["context"]
        else:
            s = del_question(s) 
        
        equation = line["output_original"][2:]

        # jieba会把 单空格 自动分出一个token，利用此特性可方便的做出NUM mask
        pos = re.search(pattern, s)
        while(pos):
            nums.append(s[pos.start():pos.end()])
            s = s[:pos.start()] + ' [NUM] ' + s[pos.end():] # 将数字改写为[NUM] token
            pos = re.search(pattern, s)
        
        if mask:    
            pseg_attr = pseg.cut(s,use_paddle=True) #paddle模式
            seg = jieba.tokenize(s)
            seg = [t[0] for t in seg]
            
            words_attr = {}
            for word, flag in pseg_attr:   
                words_attr[word] = flag

            mask_list = []
            for w in seg:
                if w in words_attr and words_attr[w] in attr_use and not word_is_ignore(w, ignore_ch):
                    mask_list.append('[MASK]')
                else:
                    mask_list.append(w)
            
            mask_text = ''.join(mask_list)
            s = mask_text

        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)
        

        tokens = [] # 最终 tokens
        match_list = []
        num_token = '[num]' # 被tokenizer处理为小写了
        tokens_initial = tokenizer.tokenize(s) # 先使用tokenizer进行初步切割
        # 假设: 被tokenizer切割的'[NUM]'token, 必定以'['开始,以']'结束
        for t in tokens_initial:
            match_text = ''.join(match_list).replace('#', '')
            text_now = match_text + t.replace('#', '')
            if text_now == num_token: # 完全匹配则直接加入
                tokens.append('[NUM]')
                match_list = []
            elif num_token.startswith(text_now): # 匹配前缀
                match_list.append(t)
            else:
                tokens += match_list
                match_list = []
                if num_token.startswith(t.replace('#', '')):
                    match_list.append(t)
                else:
                    tokens.append(t)

        tokens = ['[CLS]'] + tokens + ['[SEP]']

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1: # 如果只出现一次
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equation)
        out_seq_infix = out_seq
        out_seq_prefix = from_infix_to_prefix(out_seq) 
        '''
        if ignore:
            seg = list(jieba.cut(s))

            # 忽略不是1, 2, 3.14的常数
            IGNORE = False
            for op in out_seq:
                pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
                pos = re.search(pattern, op)
                if pos and pos.start() == 0 and op not in ['1', '2', '3.14']:
                    IGNORE = True
                    break
            if IGNORE:
                ignore_const += 1
                continue
            # ------------------------------------------
            # 忽略纯数学问题
            for t in mask_seq:
                if t in ['+', '-', '*', 'residue', 'quotient', 'product', 'divisor', 'dividend']:
                    IGNORE = True
                    break
            for t in seg:
                if t in ['+', '-', '*', '余数', '商', '积', '除数', '被除数', '因数']:
                    IGNORE = True
                    break
            """
            for word in ['余数', '商', '积', '除数', '被除数', '因数']:
                if word in seg["original_text"]: # 这样子做会将 "商店有10本书..."这种文本识别为IGNORE
                    print(word)
                    IGNORE = True
                    break
            """
            if IGNORE:
                # print(line["original_text"])
                ignore_math += 1
                continue
            # ------------------------------------------
            # 忽略奇怪操作符
            for op in out_seq_prefix:
                if op[0] == 'N' or op in ['1', '2', '3.14', '+', '-', '*', '/']:
                    continue
                else:
                    IGNORE = True
                    break
            if IGNORE:
                # print(out_seq_prefix)
                ignore_op += 1
                continue
            # ------------------------------------------
            # 忽略表达式长于9的, 忽略抽取数字个数多于5的
            if len(out_seq_prefix) > 9 or len(nums) > 5: # 建议len(nums)>选择5,6,7
                ignore_output_len += 1
                continue
            # ------------------------------------------
        '''
        if "output_prefix" in line: # 若字段包含output_prefix,则其一定为ground truth  
            out_seq_prefix = line["output_prefix"].split(" ")

        prefix_list = equivalent_expression_old(out_seq_prefix)

        # 需要注意在delq情况下容易出现, line["output_prefix"]中包含被切割掉的[NUM],其没法被统计在nums中
        ignore = False # 筛未检测变量的文本
        for s in out_seq_prefix:  
            if s[0].isdigit() and s not in generate_nums:# and trainset:
                ignore = True
                break
            
            if s[0] == 'N' and int(s[1:]) >= len(nums):
                ignore = True
                break
                
            if s in generate_nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1
        if ignore:
            continue

        count_num_max = max(count_num_max, len(nums))

        num_idx = []
        for i, j in enumerate(tokens):
            if j == "[NUM]":
                num_idx.append(i)
        
        assert len(nums) == len(num_idx)

        # 可解释性数据接口
        inter_prefix = list()
        pairs.append({
            'original_text':s,
            'tokens': tokens, # 对数字改为[NUM]后的token列表
            'output': out_seq_prefix,# 先序遍历后的表达式
            'output_list': prefix_list, # 等价表达式list(前序形式)
            'nums': nums, # 按顺序记录出现的数字 
            'num_idx': num_idx, # 记录input列表中出现[NUM]的下标
            'id':line['id'],
            'interpretation': {}, # line["interpretation"], # 可解释性标注
            'inter_prefix':inter_prefix,# 可解释性标注的前序遍历logic
        })

    # ignore_all = ignore_const + ignore_output_len + ignore_math + ignore_op
    # ignore_list = [ignore_all, ignore_const, ignore_output_len, ignore_math, ignore_op]

    # generate_num_select = [] #　只选择出现过５次以上的 常量
    # for g in generate_nums:
    #     if generate_nums_dict[g] >= 5:
    #         generate_num_select.append(g)
    #     else:
    #         print("find barely used num:", g)

    return pairs, generate_nums, count_num_max#, ignore_list


"""
面向数据:(仅考虑最基础的数据) # 中英文均适配
{
"id":"1",
"original_text":"镇海雅乐学校二年级的小朋友到一条小路的一边植树．小朋友们每隔2米种一棵树（马路两头都种了树），最后发现一共种了11棵，这条小路长多少米．",
"equation":"x=(11-1)*2",
"ans":"20"
}

输出数据:
{
    'original_text':line["original_text"].strip().split(" "),
    'tokens': tokens, # 对数字改为[NUM]后的token列表
    'output': line["output_prefix"].split(" "),# 先序遍历后的表达式
    'nums': line["nums"].split(' '), # 按顺序记录出现的数字 
    'num_idx': num_idx, # 记录input列表中出现[NUM]的下标
    'id':line['id'],
    'interpretation': line["interpretation"], # 可解释性标注
    'inter_prefix':inter_prefix,# 可解释性标注的前序遍历logic
}
"""
def process_data_pipeline_delq(train_data_path, valid_data_path, test_data_path, tokenizer, debug=False, mask=False):
    train_data = read_json(train_data_path)
    valid_data = read_json(valid_data_path)
    if test_data_path:
        test_data = read_json(test_data_path)
    else:
        test_data = []

    if debug:
        train_data = train_data[:100]
        valid_data = valid_data[:30]
        test_data = test_data[:30]

    train_data, generate_nums, copy_nums = transfer_num_delq(train_data, tokenizer, mask, trainset=True)
    valid_data, _, _ = transfer_num_delq(valid_data, tokenizer, mask)
    test_data, _, _ = transfer_num_delq(test_data, tokenizer, mask)

    # ignore_list = [ignore_list_train, ignore_list_valid, ignore_list_test]
    return train_data, valid_data, test_data, generate_nums, copy_nums#, ignore_list

"""
data格式:{
    'original_text':line["original_text"].strip().split(" "),
    'tokens': tokens, # 对数字改为[NUM]后的token列表
    'nums': line["nums"].split(' '), # 按顺序记录出现的数字 
    'num_idx': num_idx, # 记录input列表中出现[NUM]的下标
    'interpretation': line["interpretation"], # 可解释性标注
    'inter_prefix':inter_prefix,# 可解释性标注的前序遍历logic
    'id':line['id'],
    pair["token_ids"] = token_ids
    pair["token_type_ids"] = token_type_ids
    pair["attention_mask"] = attention_mask
    pair["output"] = output_cell
}
"""
def prepare_pairs(pairs, output_lang, tokenizer, max_seq_length, tree=False):
    PAD_id = tokenizer.pad_token_id
    processed_pairs = []
    # ignore_input_len = 0
    """
    pair:{
        'original_text':line["original_text"].strip().split(" "),
        'tokens': tokens, # 对数字改为[NUM]后的token列表
        'output': line["output_prefix"].split(" "),# 中序遍历后的表达式
        'nums': line["nums"].split(' '), # 按顺序记录出现的数字 
        'num_idx': num_idx, # 记录input列表中出现[NUM]的下标
        'id':line['id'],
        'interpretation': line["interpretation"], # 可解释性标注
        'inter_prefix':inter_prefix,# 可解释性标注的前序遍历logic
    }
    """
    for pair in pairs:
        ## 先处理num_stack
        num_stack = []
        for word in pair['output']:  # 处理表达式
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                print("output lang not find: ", word, '||', pair['output'], pair["id"])     
                flag_not = False
                for i, j in enumerate(pair['nums']):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair["nums"]))])

        num_stack.reverse()
        
        '''assert num_stack == []''' # mask question 时需要注释掉

        # 忽略长度＞180的问题, 此处用ignore控制的原因在于：文本过长容易导致内存错误
        assert len(pair["tokens"]) <= max_seq_length
        
        output_cell = indexes_from_sentence_output(output_lang, pair['output'], tree=tree)   

        output_cell_list = list()
        for line in pair['output_list']:
            output_cell_list.append(indexes_from_sentence_output(output_lang, line, tree=tree))

        token_ids = tokenizer.convert_tokens_to_ids(pair["tokens"])
        token_len = len(token_ids)
        # Padding 
        padding_ids = [PAD_id]*(max_seq_length - len(token_ids))
        token_ids += padding_ids
        # token_type_ids
        token_type_ids = [0]*max_seq_length
        # attention_mask
        attention_mask = [1]*token_len + padding_ids
        
        ### Testing num 
        for idx in pair["num_idx"]:
            assert pair["tokens"][idx] == '[NUM]'
        
        pair["token_ids"] = token_ids
        pair["token_type_ids"] = token_type_ids
        pair["attention_mask"] = attention_mask
        pair["output"] = output_cell
        pair["output_list"] = output_cell_list
        pair["nums"] = pair["nums"]
        pair["id"] = pair["id"]
        pair["original_text"] = pair["original_text"]

        processed_pairs.append(pair)

    return processed_pairs#, ignore_input_len


def prepare_bert_data(pairs_train, pairs_valid, pairs_test, generate_nums, 
                      copy_nums, tokenizer, max_seq_length, tree=False):
    output_lang = Lang()
    """
    {
    'original_text':line["original_text"].strip().split(" "),
    'tokens': tokens, # 对数字改为[NUM]后的token列表
    'output': out_seq_prefix, # 先序遍历后的表达式
    'nums': nums, # 按顺序记录出现的数字 
    'num_idx': num_idx, # 记录input列表中出现[NUM]的下标
    'id':line['id'],
    # 'interpretation': interpretation, # 可解释性标注
    }
    """
    ## build lang
    print("Tokenizing/Indexing words...")
    for pair in pairs_train:
        output_lang.add_sen_to_vocab(pair['output'])

    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)
        
    print(output_lang.index2word)
    print('Indexed %d words in output' % (output_lang.n_words))

    train_pairs = prepare_pairs(pairs_train, output_lang, tokenizer, max_seq_length, tree)
    valid_pairs = prepare_pairs(pairs_valid, output_lang, tokenizer, max_seq_length, tree)
    test_pairs = prepare_pairs(pairs_test, output_lang, tokenizer, max_seq_length, tree)

    print('Number of training data %d' % (len(train_pairs)))
    print('Number of validating data %d' % (len(valid_pairs)))
    print('Number of testing data %d' % (len(test_pairs)))
    
    # ignore_len_list = [ignore_input_len_train, ignore_input_len_valid, ignore_input_len_test]
    return output_lang, train_pairs, valid_pairs, test_pairs#, ignore_len_list
