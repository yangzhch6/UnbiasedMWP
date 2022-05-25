This is the repository of our paper: [Unbiased Math Word Problems Benchmark for Mitigating Solving Bias](https://arxiv.org/abs/2205.08108)   
```
Zhicheng Yang, Jinghui Qin, Jiaqi Chen, Xiaodan Liang
Unbiased Math Word Problems Benchmark for Mitigating Solving Bias
Annual Conference of the North American Chapter of the Association for Computational Linguistics, 2022. (Findings of NAACL 2022)
```

# Preparing

download chinese-bert-wwm from https://huggingface.co/hfl/chinese-bert-wwm

change the vocab.txt file as following:
```
[PAD]
[NUM]
[unused2]
[unused3]
[unused4]
...
```

# UnbiasedMWP Dataset
```
data/UnbiasedMWP/UnbiasedMWP-Src/: UnbiasedMWP-Source data in our paper
data/UnbiasedMWP/UnbiasedMWP-All/: UnbiasedMWP-All data in our paper
```

# Training
General:
```
CUDA_VISIBLE_DEVICES=0 python Filename
    --save_path //your model save path
    --save // control whether to save model
    --train_path // train data path
    --valid_path // valid data path
    --test_path // test data path
```

## Math23K
**Equivalent Expression Generation of Math23K will take a long time, please wait for about 15 minutes.**   
Bert2Tree Baseline:
```
CUDA_VISIBLE_DEVICES=7 python run_bert2tree.py --save_path model/math23k/bert2tree-split --save --train_path data/Math23K/Math23K-Split/train.json --valid_path data/Math23K/Math23K-Split/valid.json --test_path data/Math23K/Math23K-Split/test.json
```

Bert2Tree + DTS:
```
CUDA_VISIBLE_DEVICES=4 python run_bert2tree_dts.py --save_path model/math23k/bert2tree_dts-split --save --train_path data/Math23K/Math23K-Split/train.json --valid_path data/Math23K/Math23K-Split/valid.json --test_path data/Math23K/Math23K-Split/test.json
```

Bert2Tree baseline:
```
CUDA_VISIBLE_DEVICES=0 python run_bert2tree.py 
    --save_path model/unbiasedmwp/bert2tree
    --save 
    --train_path data/UnbiasedMWP-Source/train_src.json
    --valid_path data/UnbiasedMWP-Source/valid_src.json
    --test_path data/UnbiasedMWP-Source/test_src.json
```
Bert2Tree + DTS:
```
CUDA_VISIBLE_DEVICES=0 python run_bert2tree_dts.py 
    --save_path model/unbiasedmwp/bert2tree
    --save 
    --train_path data/UnbiasedMWP-Source/train_src.json
    --valid_path data/UnbiasedMWP-Source/valid_src.json
    --test_path data/UnbiasedMWP-Source/test_src.json
```
## Evaluating
Bert2Tree + DST:
```
CUDA_VISIBLE_DEVICES=0 python run_bert2tree_evaluate.py 
    --save_path model/unbiasedmwp/bert2tree
    --save 
    --train_path data/UnbiasedMWP-Source/train_src.json
    --valid_path data/UnbiasedMWP-Source/valid_src.json
    --test_path data/UnbiasedMWP-Source/test_src.json
```
