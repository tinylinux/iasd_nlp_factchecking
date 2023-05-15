#All usefull imports
import argparse
import math
from argparse import Namespace
from functools import partial
from typing import Dict
import numpy as np
import torch

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast, RobertaConfig
from transformers.optimization import AdamW
import json
from typing import Dict, List, Set, AnyStr

from copy import deepcopy
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from focal_loss.focal_loss import FocalLoss

LABELS_BEFORE={'SUPPORTS': 0,'SUPPORTED':0, 'REFUTES': 1, 'NOT ENOUGH': 2}
LABELS_AFTER={"ENOUGH -- IRRELEVANT":0,"ENOUGH -- REPEATED":1,"NOT ENOUGH":2}


#Taken and modified from https://github.com/copenlu/sufficient_facts/blob/master to better load data with explanations
def collate(instances: List[Dict],
                         tokenizer: AutoTokenizer,
                          max_claim_len:int=None,
                          claim_only=False,
                         device='cuda'):
    """Collates a batch with data from an explanations dataset"""

    # [CLS] claim tokens [SEP] evidence tokens [SEP] label before tokens  [SEP] removed tokens [SEP]
    input_ids = []
    sentence_start_ids = []
    claim_ends = []
   
    #Iterate over dataset
    for instance in instances:
        #Tokenize the claim
        instance_sentence_starts = []
        instance_input_ids = [tokenizer.cls_token_id]
        
        claim_tokens = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(instance['claim']))
        if max_claim_len:
            claim_tokens = claim_tokens[:max_claim_len]
        instance_input_ids.extend(claim_tokens)
        instance_input_ids.append(tokenizer.sep_token_id)
        
        claim_ends.append(len(instance_input_ids))

        if claim_only:
            input_ids.append(instance_input_ids)
            sentence_start_ids.append(instance_sentence_starts)
            
        else:
            #Tokenize the other parts
            for col in ["evidence","label_before","removed"]:
                if col=="evidence":
                    for el in instance[col]:
                        instance_sentence_starts.append(len(instance_input_ids))

                        tokens = tokenizer.convert_tokens_to_ids(
                            tokenizer.tokenize(el[0]))
                        
                        instance_input_ids.extend(tokens)
                        instance_input_ids.append(tokenizer.sep_token_id)
                elif col=="label_before":
                    
                    instance_sentence_starts.append(len(instance_input_ids))
                    tokens = tokenizer.convert_tokens_to_ids(
                        tokenizer.tokenize(instance[col]))
                    
                    instance_input_ids.extend(tokens)
                    instance_input_ids.append(tokenizer.sep_token_id)
                else:
                    instance_sentence_starts.append(len(instance_input_ids))
                    tokens = tokenizer.convert_tokens_to_ids(
                        tokenizer.tokenize(instance[col][0]))
                    
                    instance_input_ids.extend(tokens)
                    instance_input_ids.append(tokenizer.sep_token_id)
                
                

            input_ids.append(instance_input_ids)

            sentence_start_ids.append(instance_sentence_starts)


    #Padd tensors and define sentence start
    batch_max_len = max([len(_s) for _s in input_ids])

    input_ids = [_s[:batch_max_len] for _s in input_ids]
    sentence_start_ids = [[i for i in ids if i < batch_max_len]
                          for ids in sentence_start_ids]

    padded_ids_tensor = torch.tensor(
        [_s + [tokenizer.pad_token_id] * (
                batch_max_len - len(_s)) for _s in
         input_ids])
    #Define labels tensor
    labels = torch.tensor([_x['label_after'] for _x in instances],
                          dtype=torch.long)
    #Define dataset format to return
    result = {
 
        'input_ids_tensor': padded_ids_tensor.cuda(device),
        'target_labels_tensor': labels.cuda(device),
        'sentence_start_ids': sentence_start_ids,
        'ids': [instance['id'] for instance in instances],
        'query_claim_ends': torch.tensor(claim_ends).cuda(device)
    }
    
    return result

#Dataset class definition for handling data
class SFDataset(Dataset):
    def __init__(self, data_dic: list):
        super().__init__()
        self.dataset = []
        
        i=0
        for item in data_dic:
            _dict={}
            _dict["id"]=i
            i+=1
            _dict["claim"]=item["claim"]
            _dict["label_before"]=item["label_before"]       
            _dict["label_after"]=LABELS_AFTER[item["label_after"]]
            _dict["evidence"]=item["evidence"]
            _dict["removed"]=item["removed"] 
            self.dataset.append(_dict)
    #We have to redefine __len__ and __getitem__ for pytorch Dataset
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]

#Create datasets and split into train/validation/test from json file
def create_and_split_dataset(dataset_dir: str,
                 dataset: str,
                 num_labels: int = None):
    datasets = []

    with open(dataset_dir+"/"+dataset,"r") as out:
                data=out.readlines()
                
                data_list = []
                
                for i, line in tqdm(enumerate(data)):
                    data_list.append(json.loads(line))
                
                if dataset=='full2.jsonl': #Our own dataset
                    datasets.append(SFDataset(data_list[300:]))
                    datasets.append(SFDataset(data_list[50:300]))
                    datasets.append(SFDataset(data_list[:50]))
                else: #If other dataset is used
                    datasets.append(SFDataset(data_list[int(0.2*len(data_list)):]))
                    datasets.append(SFDataset(data_list[int(0.1*len(data_list)):int(0.2*len(data_list))]))
                    datasets.append(SFDataset(data_list[:int(0.1*len(data_list))]))
    
    
    return tuple(datasets)

#Main training function
def train_model(args: Namespace,
                model: torch.nn.Module,
                train_dl: DataLoader, val_dl: DataLoader,
                optimizer: torch.optim.Optimizer, scheduler) :

    best_score = {'dev_0_f1': 0,'dev_1_f1':0,'val_2_f1':0, 'Loss_Validation': 1}
    #loss_fct = torch.nn.CrossEntropyLoss()
    loss_fct=FocalLoss(gamma=0.8)  #FocalLoss to help with class imbalance?
    model.train()
    for ep in range(args.epochs):
        step = -1
        for batch_i, batch in enumerate(train_dl): #Iterate over batches
            #print(batch)
            step += 1
            logits = model(batch['input_ids_tensor'], attention_mask=batch['input_ids_tensor']!=tokenizer.pad_token_id).logits

            m = torch.nn.Softmax(dim=-1) #for focalloss
            loss = loss_fct(m(logits.view(-1, args.nb_labels)),  #Compute loss
                            batch['target_labels_tensor'].long().view(-1)) 
            #backward pass and optimizer step
            loss.backward()

            
            optimizer.step()
           

            current_train = {
                'train_loss': loss.cpu().data.numpy(),
                'epoch': ep+1   ,
                'step': step,
            }
            print(
                '\t'.join([f'{k}: {v:.3f}' for k, v in current_train.items()]), flush=True, end='\r')  #Print training advancement

        #Evaluate on validation set    
        current_val = eval_model(args, model, val_dl)
        current_val.update(current_train)
        print(f"epoch {ep+1}, step {step}", current_val, flush=True)
        if current_val['Loss_Validation'] >= best_score['Loss_Validation']:
                    best_score = deepcopy(current_val)
        model.train()
    #Evaluate on test set
    result = eval_model(args, model, test_dl)
    print('Scores on test set: ', result, flush=True)
        

        
    return  best_score

#Scoring a batch and return mean of losses, precision, accuracy, f1 and recall
def score_batch(args,
                model: torch.nn.Module,
                test_dl: DataLoader):
    model.eval()
    pred_class, true_class, losses, ids = [], [], [], []
    inputs = []
    
    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Evaluation"):
            
            optimizer.zero_grad()
            logits = model(batch['input_ids_tensor'], attention_mask=batch['input_ids_tensor']!=tokenizer.pad_token_id).logits
            inputs += [tokenizer.decode(i) for i in batch['input_ids_tensor'].detach().cpu().numpy().tolist()]
            true_class += batch['target_labels_tensor'].detach().cpu().numpy().tolist()
            pred_class += logits.detach().cpu().numpy().tolist()

            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, args.nb_labels),
                            batch['target_labels_tensor'].long().view(-1))
            ids += batch['ids']
            losses.append(loss.item())

        prediction = np.argmax(np.asarray(pred_class).reshape(-1, args.nb_labels),
                               axis=-1)
        p, r, f1, _ = precision_recall_fscore_support(true_class,
                                                      prediction,
                                                      )
        acc = accuracy_score(true_class, prediction)
    return  np.mean(losses), acc, p, r, f1
#Evaluate model performance and return Dict with all 5 metrics : Loss, Precision, Recall, F1, Accuracy
def eval_model(args,
               model: torch.nn.Module,
               test_dl: DataLoader):
    losses, acc, p, r, f1 = score_batch(args, model, test_dl)
    dev_eval = {
        'Loss_Validation': np.mean(losses),
        'Precision_Validation': p,
        'Recall_Validation': r,
        'F1_Validation': f1,
        'Accuracy_Validation': acc
    }

    return dev_eval

if __name__ == "__main__":
    #Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset file name",
                        choices=['hover_sufficient_facts.jsonl', 'fever_sufficient_facts.jsonl', 'vitaminc_sufficient_facts.jsonl','full.jsonl','full2.jsonl'],
                        default='hover_sufficient_facts.jsonl')
    parser.add_argument("--dataset_dir", help="Path to the train datasets",
                        default='data/sufficient_facts', type=str)
    parser.add_argument("--nb_labels",help="Number of labels",default=3,type=int)


    parser.add_argument("--batch_size", help="Batch size", type=int, default=8)
    parser.add_argument("--lr", help="Learning Rate", type=float, default=1e-6)
    parser.add_argument("--epochs", help="Epochs number", type=int, default=4)
    
    args = parser.parse_args()
   
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #Get dataset with split
    train, val, test = create_and_split_dataset(args.dataset_dir,args.dataset,args.nb_labels)
    
    #Initialise tokenizer and classifier
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    config = RobertaConfig.from_pretrained('roberta-base')
    config.num_labels = args.nb_labels
    model = RobertaForSequenceClassification.from_pretrained(
        'roberta-base',
        config=config).to(device)
    #Define collate function
    collate_fn = partial(collate,
                         tokenizer=tokenizer,
                         device=device,
                         
                         max_claim_len=512)
    #Define optimizer to be used
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    
    #Handle the datasets with DataLoader objects
    train_dl = DataLoader(batch_size=args.batch_size,
                          dataset=train, shuffle=True,
                          collate_fn=collate_fn)
    val_dl = DataLoader(#batch_size=args.batch_size,
                            dataset=val,
                            collate_fn=collate_fn,
                            shuffle=False)
    
    test_dl = DataLoader(test,  #batch_size=args.batch_size,
                         collate_fn=collate_fn, shuffle=False)
    

    
    

    step_per_epoch = math.ceil(len(train) / (args.batch_size))
    num_steps = step_per_epoch * args.epochs

    best_perf = train_model(args, model,train_dl, val_dl,optimizer, None)
    #Print best perf
    print(best_perf)
    

