## File
Only valid.json is different
```
own_processed: our own split validation set
original: Math23K data

the own_processed/count.py proved that:
    own_processed/train_all.json == train_all.json
    own_processed/test.json == test.json

```

## id 8883 in train.json
change the equation from `1-(-(1/2))` to `1+(1/2)`

Our ARR:
```
    {
        "id": "8883",
        "original_text": "计算：1-(0-(1/2))=．",
        "segmented_text": "计算 ： 1 - ( 0 - (1/2) ) = ．",
        "output_original": "x=1-(0-(1/2))",
        "ans": "(3)/(2)"
    },
```