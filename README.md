# Sufficient Facts


## Installation 

You should run the following command in your environment :
```bash
pip install focal_loss_torch tqdm torch transformers numpy sklearn
``` 

## Files 

* "full.jsonl" is the 3 datasets combined (Hover+fever+vitaminc)
* the "full2.jsonl" file was computed from "full.jsonl" file to rearrange and rebalance the dataset and train/val/test split.

## Run
example command to Train and Evaluate RoBERTa model on the full2.jsonl dataset:

```bash
python model.py --epochs 300 --dataset_dir sufficient_facts --dataset full2.jsonl --batch_size 8 --lr 1e-6 --model roberta
```