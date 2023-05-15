You should run the following command in your environment :

pip install focal_loss_torch , tqdm , torch, transformers, numpy, sklearn

"full.jsonl" is the 3 datasets combined (Hover+fever+vitaminc)

the "full2.jsonl" file was computed from "full.jsonl" file to rearrange and rebalance the dataset and train/val/test split.

example command to Train and Evaluate RoBERTa model on the full2.jsonl dataset:

python model.py --epochs 30 --dataset_dir sufficient_facts --dataset full2.jsonl --batch_size 8 --lr 1e-7