import copy
from itertools import permutations as pmt
from sympy import simplify

def tree_len(tree):
    if tree == {}:
        return 0
    length = 1
    length += tree_len(tree["left"])
    length += tree_len(tree["right"])
    return length


def tree_leaf(tree):
    if tree == {}:
        return
    
    if tree["left"] == {}:
        tree["left"] = {
            "node": "num",
            "left": {},
            "right": {}
        }
    else:
        tree_leaf(tree["left"])

    if tree["right"] == {}:
        tree["right"] = {
            "node": "num",
            "left": {},
            "right": {}
        }
    else:
        tree_leaf(tree["right"])


def construct_bone(tree, root, node_len, target_len, tree_list):
    if node_len == 0:
        return

    node_len -= 1
    # generate this node
    root["node"] = "op"
    root["left"] = {}
    root["right"] = {}
    
    if tree_len(tree) == target_len:
        copy_tree = copy.deepcopy(tree)
        tree_leaf(copy_tree)
        tree_list.append(copy_tree)
    
    for left in range(node_len+1):
        right = node_len - left
        construct_bone(tree, root["left"], left, target_len, tree_list)
        construct_bone(tree, root["right"], right, target_len, tree_list)
        root["left"] = {}
        root["right"] = {}

# 前序遍历表达式树
def preorder_traversal(tree, prefix):
    if tree == {}:
        return 
    prefix.append(tree["node"])
    preorder_traversal(tree["left"], prefix)
    preorder_traversal(tree["right"], prefix)

# 中序遍历表达式树
def inorder_traversal(tree, infix):
    if tree == {}:
        return 
    infix.append('(')
    inorder_traversal(tree["left"], infix)
    infix.append(tree["node"])
    inorder_traversal(tree["right"], infix)
    infix.append(')')

# 前序遍历建立表达式树
def prefix_to_tree(tree, prefix):
    tree["node"] = prefix[0]
    tree["left"] = {}
    tree["right"] = {}
    
    if prefix[0] not in ['+', '-', '*', '/']:
        return prefix[1:]
    prefix_left = prefix_to_tree(tree["left"], prefix[1:])
    prefix_right = prefix_to_tree(tree["right"], prefix_left)
    return prefix_right


def recognize(prefix):
    op = list()
    nums = list()
    for t in prefix:
        if t not in ['+', '-', '*', '/']:
            nums.append(t)
        else:
            op.append(t)
    return op, nums

def expression_permutation(op, nums):
    op_pmt = [list(t) for t in pmt(op)]
    nums_pmt = [list(t) for t in pmt(nums)]
    exp_pmt = list()
    for o in op_pmt:
        for n in nums_pmt:
            exp_pmt.append((o, n))
    return exp_pmt


def fill_bone(bone, op, nums):
    op_index = 0
    num_index = 0

    for i in range(len(bone)):
        if bone[i] == "op":
            bone[i] = op[op_index]
            op_index += 1
        else:
            bone[i] = nums[num_index]
            num_index += 1
    assert op_index == len(op) and num_index == len(nums)
    return bone


def generate_equivalent(expression_prefix):
    op, nums = recognize(expression_prefix)
    exp_pmt = expression_permutation(op, nums)

    bone_list = list() 
    bone_tree = {} 
    construct_bone(bone_tree, bone_tree, len(op), len(op), bone_list)
    prefix_bone = list()
    infix_bone = list()
    for tree in bone_list:
        prefix = list()
        infix = list()
        preorder_traversal(tree, prefix)
        inorder_traversal(tree, infix)
        prefix_bone.append(prefix)
        infix_bone.append(infix)
    tree_list = list()
    for bone in prefix_bone:
        for op, nums in exp_pmt:
            bone_copy = copy.deepcopy(bone)
            fill_bone(bone_copy, op, nums)
            tree = {}
            prefix_to_tree(tree, bone_copy)
            tree_list.append(tree)
    
    prefix_list = list()
    infix_list = list()
    for i in range(len(tree_list)):
        infix_line = list()
        prefix_line = list()
        inorder_traversal(tree_list[i], infix_line)
        preorder_traversal(tree_list[i], prefix_line)
        infix_list.append(infix_line)
        prefix_list.append(prefix_line)

    return infix_list, prefix_list

def prefix_to_infix(prefix):
    tree = dict()
    prefix_to_tree(tree, prefix)
    infix = list()
    inorder_traversal(tree, infix)
    return infix

# Version 3.0
def equivalent_expression(expression_prefix):
    expression_infix = prefix_to_infix(expression_prefix)
    infix_list, prefix_list = generate_equivalent(expression_prefix)
    infix_equivalent = list()
    prefix_equivalent = list()
    for i in range(len(infix_list)):
        if simplify(' '.join(expression_infix)) == simplify(' '.join(infix_list[i])):
            infix_equivalent.append(infix_list[i])
            prefix_equivalent.append(prefix_list[i])
    return infix_equivalent, prefix_equivalent


# 根据一颗表达式树,对等价节点左右交换生成等价表达式list
def variate_tree(tree, root, prefix_all):
    if root == {}:
        return 
        
    variate_tree(tree, root["left"], prefix_all)
    variate_tree(tree, root["right"], prefix_all)
    
    if root["node"] in ['+', '*']:
        tmp = root['left'] 
        root['left'] = root["right"]
        root["right"] = tmp
    
        prefix_all.append(copy.deepcopy(tree))
    
        variate_tree(tree, root["left"], prefix_all)
        variate_tree(tree, root["right"], prefix_all)
    
        tmp = root['left'] 
        root['left'] = root["right"]
        root["right"] = tmp


# 对ground truth前缀表达式构建等价list
# Version 1.0 & 2.0
def equivalent_expression_old(prefix):
    prefix_tree = dict()
    prefix_to_tree(prefix_tree, prefix)
    tree_list = [prefix_tree]
    variate_tree(prefix_tree, prefix_tree, tree_list)
    equ_list = list()
    for tree in tree_list:
        prefix_line = list()
        preorder_traversal(tree, prefix_line)
        equ_list.append(prefix_line)
    return equ_list

if __name__ == "__main__":
    expression_prefix = "- * N2 N0 N3".split()
    infix_equivalent, prefix_equivalent = equivalent_expression(expression_prefix)
    print(infix_equivalent)
