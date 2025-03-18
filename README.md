# Dynamic Speculative Planning

## Dataset
- `dataset_predict_k_cot.json`: openAGI dataset for sft
- `dataset_predict_k_cot_val.json`: openAGI dataset for sft (including middle steps)
- `dataset_sft_travelplanner.json`: travel planner dataset for sft
- `dataset_value_func_cot.json`: openAGI dataset for lambda return value function
- `dataset_value_func_react.json`: travel planner dataset for lambda return value function
  


## Experiment
### SFT
command for running sft
```
python -m OpenAGI.predict_k.sft_ppo_distilbert_train --batch_size 16 --lr 1e-4
```

### Lambda Return
command for running lambda return
```
python -m OpenAGI.lambda_return_train --batch_size 16 --lr 1e-4 --lambda_ 1
```

command for running lambda return with batch sample
```
python -m OpenAGI.lambda_return_bs_train --batch_size 16 --lr 1e-4 --lambda_ 1
```