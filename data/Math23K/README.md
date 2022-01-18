## Fix Data 
`processed/` is our ARR math23k data processed from original math23 dataset

The Difference betweeen processed/ and the original data is 
1. the data split (random choose 1k from original/math23k_train.json)
2. fix id 8883 from original data
    id 8883 
    ```
    change id 8883 to the follwing:
        {
            "id": "8883",
            "original_text": "计算：1-(0-(1/2))=．",
            "segmented_text": "计算 ： 1 - ( 0 - (1/2) ) = ．",
            "output_original": "x=1-(0-(1/2))",
            "ans": "(3)/(2)"
        }
    ```