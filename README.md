# Sufficient Facts
This work is based on the paper [Fact Checking with Insufficient Evidence, by Atanasova et al.
](https://arxiv.org/abs/2204.02007) as NLP Project for M2 IASD

by Oumaima BENDRAMEZ, Rida LALI, Hamza TOUZANI

## Installation 

To install the required packages to run the code, you should run the following command in your environment :
```bash
pip install focal_loss_torch tqdm torch transformers numpy sklearn
``` 

## Dataset files

* "full.jsonl" is the 3 datasets combined (Hover+fever+vitaminc)
* the "full2.jsonl" file was computed from "full.jsonl" file to rearrange and rebalance the dataset and train/val/test split.

## Run
Here is an example command to Train and Evaluate RoBERTa model on the full2.jsonl dataset:

```bash
python model.py --epochs 300 --dataset_dir sufficient_facts --dataset full2.jsonl --batch_size 8 --lr 1e-6 --model roberta
```

- **epochs** flag is for the number of epochs
- **dataset_dir** flag is to give the local directory of the dataset
- **dataset** flag is to give the dataset file
- **batch_size** and **lr** are learning parameters
- **model** is to select between BERT and RoBERTa models (`bert` or `roberta`)