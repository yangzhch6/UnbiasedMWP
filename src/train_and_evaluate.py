from src.masked_cross_entropy import *
from src.pre_data import *
from src.expressions_transfer import *
from src.models import *
import math
import copy
import torch
import torch.optim
import torch.nn.functional as f
import time

MAX_OUTPUT_LENGTH = 45
MAX_INPUT_LENGTH = 120
USE_CUDA = torch.cuda.is_available()


class Beam:  # the class save the beam node
    def __init__(self, score, input_var, hidden, all_output):
        self.score = score
        self.input_var = input_var
        self.hidden = hidden
        self.all_output = all_output
        

def time_since(s):  # compute time
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)

def generate_tree_input(target, num_start):
    target_input = copy.deepcopy(target)
    for i in range(len(target)):
        if target_input[i] >= num_start:
            target_input[i] = 0
    return torch.LongTensor(target_input)

def check_logic(predict, logic):
    acc_num = 0
    logic_num = 0
    predict = [d.cpu().item() for d in predict]
    for p, l  in zip(predict, logic):
        if l == -1:
            continue
        
        if p == l:
            acc_num += 1
        logic_num += 1
    return acc_num, logic_num

def compute_prefix_tree_result(test_res, test_tar, output_lang, num_list):
    if test_res == test_tar:
        return True, True, test_res, test_tar
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang, num_list)

    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_prefix_expression(test) - compute_prefix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar

def get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, max_num_size, hidden_size):
    indices = list()
    masked_index = list()
    sen_len = encoder_outputs.size(0)
    temp_1 = [1 for _ in range(hidden_size)]
    temp_0 = [0 for _ in range(hidden_size)]
    for b in range(batch_size):
        for i in num_pos[b]:
            indices.append(i + b * sen_len)
            masked_index.append(temp_0)
        indices += [0 for _ in range(len(num_pos[b]), max_num_size)]
        masked_index += [temp_1 for _ in range(len(num_pos[b]), max_num_size)]
    indices = torch.LongTensor(indices)
    masked_index = torch.BoolTensor(masked_index)
    masked_index = masked_index.view(batch_size, max_num_size, hidden_size)
    if USE_CUDA:
        indices = indices.cuda()
        masked_index = masked_index.cuda()
    all_outputs = encoder_outputs.transpose(0, 1).contiguous() # B x S x H
    all_embedding = all_outputs.view(-1, encoder_outputs.size(2)) # B x S x H -> (B x S) x H
    all_num = all_embedding.index_select(0, indices)
    all_num = all_num.view(batch_size, max_num_size, hidden_size)
    return all_num.masked_fill_(masked_index, 0.0) 

def get_all_number_encoder_outputs_ddp(encoder_outputs, num_pos, batch_size, num_size, hidden_size, local_rank):
    indices = list()
    sen_len = encoder_outputs.size(0)
    masked_index = []
    temp_1 = [1 for _ in range(hidden_size)]
    temp_0 = [0 for _ in range(hidden_size)]
    for b in range(batch_size):
        for i in num_pos[b]:
            indices.append(i + b * sen_len)
            masked_index.append(temp_0)
        indices += [0 for _ in range(len(num_pos[b]), num_size)]
        masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
    indices = torch.LongTensor(indices)
    masked_index = torch.BoolTensor(masked_index)
    masked_index = masked_index.view(batch_size, num_size, hidden_size)
    if USE_CUDA:
        indices = indices.to(local_rank)
        masked_index = masked_index.to(local_rank)
    all_outputs = encoder_outputs.transpose(0, 1).contiguous() # B x S x H
    all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # B x S x H -> (B x S) x H
    all_num = all_embedding.index_select(0, indices)
    all_num = all_num.view(batch_size, num_size, hidden_size)
    return all_num.masked_fill_(masked_index, 0.0)


def copy_list(l):
    r = []
    if len(l) == 0:
        return r
    for i in l:
        if type(i) is list:
            r.append(copy_list(i))
        else:
            r.append(i)
    return r


class TreeBeam:  # the class save the beam node
    def __init__(self, score, node_stack, embedding_stack, left_childs, out, logic_list = []):
        self.score = score
        self.embedding_stack = copy_list(embedding_stack)
        self.node_stack = copy_list(node_stack)
        self.left_childs = copy_list(left_childs)
        self.out = copy.deepcopy(out)
        self.logic_list = copy.deepcopy(logic_list)


class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False):
        self.embedding = embedding
        self.terminal = terminal

def train_tree_ddp(output, output_len, nums_stack, num_size, generate_nums,
               encoder, predict, generate, merge, encoder_optimizer, encoder_scheduler, 
               predict_optimizer, generate_optimizer,
               merge_optimizer, output_lang, num_idx, 
               token_ids, token_type_ids, attention_mask, local_rank):

    seq_mask = torch.BoolTensor(attention_mask)
    seq_mask = (seq_mask == torch.BoolTensor(torch.zeros_like(seq_mask)))
    num_mask = []
    max_num_size = max(num_size) + len(generate_nums)
    for i in num_size:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.BoolTensor(num_mask)

    unk = output_lang.word2index["[UNK]"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)

    target = torch.LongTensor(output).transpose(0, 1)

    # [ [0.0]*predict.hidden_size ]
    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.module.hidden_size)]).unsqueeze(0)
    batch_size = len(token_ids)

    encoder.module.train()
    predict.module.train()
    generate.module.train()
    merge.module.train()

    if USE_CUDA:
        # print("convert tensor to cuda")
        seq_mask = seq_mask.to(local_rank)
        padding_hidden = padding_hidden.to(local_rank)
        num_mask = num_mask.to(local_rank)
        token_ids = torch.tensor(token_ids, dtype=torch.long).to(local_rank)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).to(local_rank)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(local_rank)

        output_len = torch.LongTensor(output_len).to(local_rank)

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    merge_optimizer.zero_grad()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(token_ids, token_type_ids, attention_mask)
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    max_output_len = max(output_len)

    all_node_outputs = []

    copy_num_len = [len(_) for _ in num_idx]
    num_size = max(copy_num_len)
    all_nums_encoder_outputs = get_all_number_encoder_outputs_ddp(encoder_outputs, num_idx, batch_size, num_size,
                                                              encoder.module.hidden_size, local_rank)

    num_start = output_lang.num_start
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]
    for t in range(max_output_len):
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)

        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack, num_start, unk)
        target[t] = target_t
        if USE_CUDA:
            generate_input = generate_input.to(local_rank)
        left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
        left_childs = []
        for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                               node_stacks, target[t].tolist(), embeddings_stacks):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue

            if i < num_start:
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))
                o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
            else:
                current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                while len(o) > 0 and o[-1].terminal:
                    sub_stree = o.pop()
                    op = o.pop()
                    current_num = merge(op.embedding, sub_stree.embedding, current_num)
                o.append(TreeEmbedding(current_num, True))
            if len(o) > 0 and o[-1].terminal:
                left_childs.append(o[-1].embedding)
            else:
                left_childs.append(None)

    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

    target = target.transpose(0, 1).contiguous()
    if USE_CUDA:
        all_node_outputs = all_node_outputs.to(local_rank)
        target = target.to(local_rank)

    loss, accurate = masked_cross_entropy(all_node_outputs, target, output_len)
    loss.backward()

    # Update parameters with optimizers
    encoder_optimizer.step()
    encoder_scheduler.step()
    
    predict_optimizer.step()
    generate_optimizer.step()
    merge_optimizer.step()
    return loss.item(), accurate.item()

def train_tree(output, output_len, num_size, generate_nums,
               encoder, predict, generate, merge, encoder_optimizer, encoder_scheduler, 
               predict_optimizer, generate_optimizer,
               merge_optimizer, output_lang, num_idx, 
               token_ids, token_type_ids, attention_mask):

    seq_mask = torch.BoolTensor(attention_mask)
    seq_mask = (seq_mask == torch.BoolTensor(torch.zeros_like(seq_mask)))
    num_mask = []
    max_num_size = max(num_size) + len(generate_nums)
    for i in num_size:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.BoolTensor(num_mask)

    unk = output_lang.word2index["[UNK]"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)

    target = torch.LongTensor(output).transpose(0, 1)

    # [ [0.0]*predict.hidden_size ]
    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(token_ids)

    encoder.train()
    predict.train()
    generate.train()
    merge.train()

    if USE_CUDA:
        # print("convert tensor to cuda")
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
        token_ids = torch.tensor(token_ids, dtype=torch.long).cuda()
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).cuda()
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    merge_optimizer.zero_grad()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(token_ids, token_type_ids, attention_mask)
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    max_output_len = max(output_len)

    all_node_predict = []

    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_idx, batch_size, max(num_size),
                                                              encoder.hidden_size)

    num_start = output_lang.num_start
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]
    for t in range(max_output_len):
        num_score, op_score, goal, context, all_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

        ## score
        predict_score = torch.cat((op_score, num_score), 1) # B x Output_size
        all_node_predict.append(predict_score) # [B x Output_size]

        # op's label of each node, nums node will be masked to 0
        node_op_label = generate_tree_input(target[t].tolist(), num_start)
        if USE_CUDA:
            node_op_label = node_op_label.cuda()
        left_child, right_child, node_op_embedding = generate(goal, node_op_label, context)
        left_childs = []
        for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                               node_stacks, target[t].tolist(), embeddings_stacks):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue

            if i < num_start:
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))
                o.append(TreeEmbedding(node_op_embedding[idx].unsqueeze(0), False))
            else:
                # print(i - num_start, all_nums_embeddings.shape)
                current_num = all_nums_embeddings[idx, i - num_start].unsqueeze(0)
                while len(o) > 0 and o[-1].terminal:
                    sub_stree = o.pop()
                    op = o.pop()
                    current_num = merge(op.embedding, sub_stree.embedding, current_num)
                o.append(TreeEmbedding(current_num, True))
            if len(o) > 0 and o[-1].terminal:
                left_childs.append(o[-1].embedding)
            else:
                left_childs.append(None)

    all_node_predict = torch.stack(all_node_predict, dim=1)  # B x max_output_len x Output_size

    target = target.transpose(0, 1).contiguous()
    if USE_CUDA:
        output_len = torch.LongTensor(output_len).cuda()
        all_node_predict = all_node_predict.cuda()
        target = target.cuda()

    loss, accurate = masked_cross_entropy(all_node_predict, target, output_len)
    loss.backward()

    # Update parameters with optimizers
    encoder_optimizer.step()
    encoder_scheduler.step()
    
    predict_optimizer.step()
    generate_optimizer.step()
    merge_optimizer.step()
    return loss.item(), accurate.item()


def find_max_prefix_line(line, output_list, output_len): 
    while output_len > 0:
        for i in range(len(output_list)):
            if line[:output_len] in output_list[i][:output_len]:
                return output_list[i]
        output_len -= 1
    return output_list[0]


def find_max_prefix_list(logits, output_list, output_len):
    B = logits.size()[0]
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = functional.log_softmax(logits_flat, dim=1)     
    _, predicts = torch.max(log_probs_flat, dim=1)
    predicts = predicts.view(B, -1).cpu().tolist()

    target = list()
    for i in range(len(predicts)):
        max_prefix_match = find_max_prefix_line(predicts[i], output_list[i], output_len[i])
        target.append(max_prefix_match)

    return target

# Version 1.0
def train_tree_em(output_list, output_len, num_size, generate_nums,
               encoder, predict, generate, merge, encoder_optimizer, encoder_scheduler, 
               predict_optimizer, generate_optimizer,
               merge_optimizer, output_lang, num_idx, 
               token_ids, token_type_ids, attention_mask):

    seq_mask = torch.BoolTensor(attention_mask)
    seq_mask = (seq_mask == torch.BoolTensor(torch.zeros_like(seq_mask)))
    num_mask = []
    max_num_size = max(num_size) + len(generate_nums)
    for i in num_size:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.BoolTensor(num_mask)

    unk = output_lang.word2index["[UNK]"]

    # [ [0.0]*predict.hidden_size ]
    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(token_ids)

    encoder.train()
    predict.train()
    generate.train()
    merge.train()

    if USE_CUDA:
        # print("convert tensor to cuda")
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
        token_ids = torch.tensor(token_ids, dtype=torch.long).cuda()
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).cuda()
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    merge_optimizer.zero_grad()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(token_ids, token_type_ids, attention_mask)
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    max_output_len = max(output_len)

    all_node_predict = []

    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_idx, batch_size, max(num_size),
                                                              encoder.hidden_size)

    num_start = output_lang.num_start
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]
    for t in range(max_output_len):
        num_score, op_score, goal, context, all_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

        ## score
        predict_score = torch.cat((op_score, num_score), 1) # B x Output_size
        all_node_predict.append(predict_score) # [B x Output_size]

        predict_score_log = functional.log_softmax(predict_score, dim=1)     
        _, predict_node = torch.max(predict_score_log[:,1:], dim=1)
        predict_node += 1
        
        # op's label of each node, nums node will be masked to 0
        node_op_label = generate_tree_input(predict_node.tolist(), num_start)
        if USE_CUDA:
            node_op_label = node_op_label.cuda()
        left_child, right_child, node_op_embedding = generate(goal, node_op_label, context)
        left_childs = []
        for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                               node_stacks, predict_node.tolist(), embeddings_stacks):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue

            if i < num_start:
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))
                o.append(TreeEmbedding(node_op_embedding[idx].unsqueeze(0), False))
            else:
                # print(i - num_start, all_nums_embeddings.shape)
                current_num = all_nums_embeddings[idx, i - num_start].unsqueeze(0)
                while len(o) > 0 and o[-1].terminal:
                    sub_stree = o.pop()
                    op = o.pop()
                    current_num = merge(op.embedding, sub_stree.embedding, current_num)
                o.append(TreeEmbedding(current_num, True))
            if len(o) > 0 and o[-1].terminal:
                left_childs.append(o[-1].embedding)
            else:
                left_childs.append(None)

    all_node_predict = torch.stack(all_node_predict, dim=1)  # B x max_output_len x Output_size

    target = find_max_prefix_list(all_node_predict, output_list, output_len)
    # stop
    if USE_CUDA:
        output_len = torch.LongTensor(output_len).cuda()
        all_node_predict = all_node_predict.cuda()
        target = torch.LongTensor(target).cuda()

    loss, accurate = masked_cross_entropy(all_node_predict, target, output_len)
    loss.backward()

    # Update parameters with optimizers
    encoder_optimizer.step()
    encoder_scheduler.step()
    
    predict_optimizer.step()
    generate_optimizer.step()
    merge_optimizer.step()
    return loss.item(), accurate.item()


# 在生成过程中选择表达式
def find_max_prefix_generating(all_node_predict, output_list, output_len, target):
    find_target = []
    step = len(all_node_predict)

    for p, t, l, exp_list in zip(all_node_predict[-1], target, output_len, output_list):
        find_target.append(t)
        # print("add", t)
        if step > l:
            continue

        for expression in exp_list:
            if expression[:step-1] != t[:step-1]:
                continue
            if expression[step-1] == p:
                find_target[-1] = expression
                break
                
    return find_target

# Version 2.0
def train_tree_em2(output_list, output_len, num_size, generate_nums,
               encoder, predict, generate, merge, encoder_optimizer, encoder_scheduler, 
               predict_optimizer, generate_optimizer,
               merge_optimizer, output_lang, num_idx, 
               token_ids, token_type_ids, attention_mask):

    seq_mask = torch.BoolTensor(attention_mask)
    seq_mask = (seq_mask == torch.BoolTensor(torch.zeros_like(seq_mask)))
    num_mask = []
    max_num_size = max(num_size) + len(generate_nums)
    for i in num_size:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.BoolTensor(num_mask)

    unk = output_lang.word2index["[UNK]"]

    # [ [0.0]*predict.hidden_size ]
    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(token_ids)

    encoder.train()
    predict.train()
    generate.train()
    merge.train()

    if USE_CUDA:
        # print("convert tensor to cuda")
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
        token_ids = torch.tensor(token_ids, dtype=torch.long).cuda()
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).cuda()
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    merge_optimizer.zero_grad()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(token_ids, token_type_ids, attention_mask)
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    max_output_len = max(output_len)

    all_node_feature = []
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_idx, batch_size, max(num_size),
                                                              encoder.hidden_size)

    all_node_predict = [] # step * batch_size
    target = [exp[0] for exp in output_list] # batch_size * step

    num_start = output_lang.num_start
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]
    for t in range(max_output_len):
        num_score, op_score, goal, context, all_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

        ## score
        predict_score = torch.cat((op_score, num_score), 1) # B x Output_size
        all_node_feature.append(predict_score) # [B x Output_size]

        predict_score_log = functional.log_softmax(predict_score, dim=1)     
        _, predict_node = torch.max(predict_score_log[:,1:], dim=1)
        predict_node += 1
        all_node_predict.append(predict_node.tolist())
        
        target = find_max_prefix_generating(all_node_predict, output_list, output_len, target)
        target_step = [line[t] for line in target]
        # find target expression
        # print(all_node_predict)
        # print(output_list)
        # print(target)
        node_op_label = generate_tree_input(target_step, num_start)
        if USE_CUDA:
            node_op_label = node_op_label.cuda()
        # print(node_op_label.shape)
        left_child, right_child, node_op_embedding = generate(goal, node_op_label, context)
        left_childs = []
        for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                               node_stacks, target_step, embeddings_stacks):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue

            if i < num_start:
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))
                o.append(TreeEmbedding(node_op_embedding[idx].unsqueeze(0), False))
            else:
                # print(i - num_start, all_nums_embeddings.shape)
                current_num = all_nums_embeddings[idx, i - num_start].unsqueeze(0)
                while len(o) > 0 and o[-1].terminal:
                    sub_stree = o.pop()
                    op = o.pop()
                    current_num = merge(op.embedding, sub_stree.embedding, current_num)
                o.append(TreeEmbedding(current_num, True))
            if len(o) > 0 and o[-1].terminal:
                left_childs.append(o[-1].embedding)
            else:
                left_childs.append(None)

    all_node_feature = torch.stack(all_node_feature, dim=1)  # B x max_output_len x Output_size

    # stop
    if USE_CUDA:
        output_len = torch.LongTensor(output_len).cuda()
        all_node_feature = all_node_feature.cuda()
        target_tensor = torch.LongTensor(target).cuda()

    loss, accurate = masked_cross_entropy(all_node_feature, target_tensor, output_len)
    loss.backward()

    # Update parameters with optimizers
    encoder_optimizer.step()
    encoder_scheduler.step()
    
    predict_optimizer.step()
    generate_optimizer.step()
    merge_optimizer.step()
    return loss.item(), accurate.item()

def evaluate_tree(generate_nums, encoder, predict, generate, merge, output_lang, 
                  num_pos, token_ids, token_type_ids, attention_mask, input_len_max, 
                  beam_size=5, max_length=MAX_OUTPUT_LENGTH):

    # seq_mask = torch.ByteTensor(attention_mask)
    seq_mask = torch.BoolTensor(1, input_len_max).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    # input_var = torch.LongTensor(input_batch).unsqueeze(1)

    num_mask = torch.BoolTensor(1, len(num_pos) + len(generate_nums)).fill_(0)

    # Set to not-training mode to disable dropout
    encoder.eval()
    predict.eval()
    generate.eval()
    merge.eval()

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        # input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()

        token_ids = torch.tensor(token_ids, dtype=torch.long).cuda()
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).cuda()
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).cuda()

    # Run words through encoder
    encoder_outputs, problem_output = encoder(token_ids, token_type_ids, attention_mask)

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.hidden_size)
    num_start = output_lang.num_start
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

    for t in range(max_length):
        current_beams = []
        while len(beams) > 0:
            b = beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                continue
            # left_childs = torch.stack(b.left_childs)
            left_childs = b.left_childs

            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                seq_mask, num_mask)

            out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

            topv, topi = out_score.topk(beam_size)

            for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                current_node_stack = copy_list(b.node_stack)
                current_left_childs = []
                current_embeddings_stacks = copy_list(b.embedding_stack)
                current_out = copy.deepcopy(b.out)

                out_token = int(ti)
                current_out.append(out_token)

                node = current_node_stack[0].pop()

                if out_token < num_start:
                    generate_input = torch.LongTensor([out_token])
                    if USE_CUDA:
                        generate_input = generate_input.cuda()
                    left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)

                    current_node_stack[0].append(TreeNode(right_child))
                    current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                    current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                else:
                    current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                    while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        sub_stree = current_embeddings_stacks[0].pop()
                        op = current_embeddings_stacks[0].pop()
                        current_num = merge(op.embedding, sub_stree.embedding, current_num)
                    current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                    current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                else:
                    current_left_childs.append(None)
                current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
                                              current_left_childs, current_out))
        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break

    return beams[0].out

def evaluate_tree_ddp(generate_nums, encoder, predict, generate, merge, output_lang, 
                  num_pos, token_ids, token_type_ids, attention_mask, input_len_max, 
                  local_rank, beam_size=5, max_length=MAX_OUTPUT_LENGTH):

    # seq_mask = torch.ByteTensor(attention_mask)
    seq_mask = torch.BoolTensor(1, input_len_max).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    # input_var = torch.LongTensor(input_batch).unsqueeze(1)

    num_mask = torch.BoolTensor(1, len(num_pos) + len(generate_nums)).fill_(0)

    # Set to not-training mode to disable dropout
    encoder.module.eval()
    predict.module.eval()
    generate.module.eval()
    merge.module.eval()

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.module.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        # input_var = input_var.to(local_rank)
        seq_mask = seq_mask.to(local_rank)
        padding_hidden = padding_hidden.to(local_rank)
        num_mask = num_mask.to(local_rank)

        token_ids = torch.tensor(token_ids, dtype=torch.long).to(local_rank)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).to(local_rank)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(local_rank)

    # Run words through encoder
    # ddp的eval中因为全在单卡上计算，各个模型在前向计算时要使用.module以避免使用ddp 
    encoder_outputs, problem_output = encoder.module(token_ids, token_type_ids, attention_mask)

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    all_nums_encoder_outputs = get_all_number_encoder_outputs_ddp(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.module.hidden_size, local_rank)
    num_start = output_lang.num_start
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

    for t in range(max_length):
        current_beams = []
        while len(beams) > 0:
            b = beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                continue
            # left_childs = torch.stack(b.left_childs)
            left_childs = b.left_childs

            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict.module(
                b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                seq_mask, num_mask)

            out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

            topv, topi = out_score.topk(beam_size)

            for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                current_node_stack = copy_list(b.node_stack)
                current_left_childs = []
                current_embeddings_stacks = copy_list(b.embedding_stack)
                current_out = copy.deepcopy(b.out)

                out_token = int(ti)
                current_out.append(out_token)

                node = current_node_stack[0].pop()

                if out_token < num_start:
                    generate_input = torch.LongTensor([out_token])
                    if USE_CUDA:
                        generate_input = generate_input.to(local_rank)
                    left_child, right_child, node_label = generate.module(current_embeddings, generate_input, current_context)

                    current_node_stack[0].append(TreeNode(right_child))
                    current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                    current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                else:
                    current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                    while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        sub_stree = current_embeddings_stacks[0].pop()
                        op = current_embeddings_stacks[0].pop()
                        current_num = merge.module(op.embedding, sub_stree.embedding, current_num)
                    current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                    current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                else:
                    current_left_childs.append(None)
                current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
                                              current_left_childs, current_out))
        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break

    return beams[0].out

