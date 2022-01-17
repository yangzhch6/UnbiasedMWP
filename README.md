## Preparing

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

## UnbiasedMWP Dataset
```
data/**_src: UnbiasedMWP-Source data in our paper
data/**_all: UnbiasedMWP-All data in our paper
```

## Training
General:
```
CUDA_VISIBLE_DEVICES=0 python Filename
    --save_path //your model save path
    --save // control whether to save model
    --train_path // train data path
    --valid_path // valid data path
    --test_path // test data path
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
Bert2Tree + DST:
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