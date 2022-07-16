import os
os.environ['CUDA_VISIBLE_DEVICES']= '1,2' 

import datasets
from datasets import load_dataset
from datasets import Dataset
import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from datasets import load_metric
from datasets import list_metrics
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
import numpy as np
from datasets import load_metric
import random
import pdb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import argparse

parser = argparse.ArgumentParser(description='baseline')
parser.add_argument('--dataset', default='newsela', type=str, nargs='+',
                    help='dataset name')
parser.add_argument('--class_num', default=5, type=int, nargs='+',
                    choices=[3, 5], help='total class num')
args = parser.parse_args()

if type(args.dataset) == type(['aa']):
    args.dataset = args.dataset[0]
if type(args.class_num) == type([3]):
    args.class_num = args.class_num[0]

print(args)

dataset_name = args.dataset
class_num = args.class_num
train_dir = '../'+ dataset_name + '//' + dataset_name + '_full_simp_feature_train.csv'
valid_dir = '../'+ dataset_name + '//' + dataset_name + '_full_simp_feature_valid.csv'
test_dir = '../'+ dataset_name + '//' + dataset_name + '_full_simp_feature_test.csv'
mapper = {'readability': 'label'}
train_df = pd.read_csv(train_dir, usecols=['text', 'class'])
valid_df = pd.read_csv(valid_dir, usecols=['text', 'class'])
test_df = pd.read_csv(test_dir, usecols=['text', 'class'])


val_dataset = load_dataset('csv', data_files=valid_dir)
val_dataset = val_dataset.rename_column('class','labels')
test_dataset = load_dataset('csv', data_files=test_dir)
test_dataset = test_dataset.rename_column('class','labels')
train_dataset = load_dataset('csv', data_files=train_dir)
train_dataset = train_dataset.rename_column('class','labels')
checkpoint = "allenai/longformer-base-4096"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
  return tokenizer(example['text'], truncation=True, padding='longest', max_length=512)

tokenized_train_datasets = train_dataset.map(tokenize_function, batched=True)
tokenized_valid_datasets = val_dataset.map(tokenize_function, batched=True)
tokenized_test_datasets = test_dataset.map(tokenize_function, batched=True)
print(tokenized_train_datasets)
print(tokenized_valid_datasets)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def get_dataloader(tokenized_datasets, usage:str):
    dataloader = DataLoader(
        tokenized_datasets, shuffle=True, batch_size=8, collate_fn=data_collator
    )
    return dataloader


def accuracy(preds, labels):
    a = accuracy_score(labels, preds)
    return a
    
def precision(preds, labels):
    p = precision_score(labels, preds, average='macro')
    return p

def recall(preds, labels):
    r = recall_score(labels, preds, average='macro')
    return r
    
def F1(preds, labels):
    f = f1_score(labels, preds, average='macro')
    return f

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    metrics = {}
    metrics['accuracy'] = accuracy(predictions, labels)
    metrics['recall'] = recall(predictions, labels)
    metrics['f1'] = F1(predictions, labels)
    metrics['precision'] = precision(predictions, labels)
    return metrics

def set_training_args():
    training_args = TrainingArguments(
        output_dir='./' + dataset_name + '//longformer_' + str(class_num),
        overwrite_output_dir = True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=128,
        num_train_epochs=5,
        evaluation_strategy='steps',
        save_strategy='steps',
        save_steps=50, 
        eval_steps=50, 
        load_best_model_at_end=True, 
        metric_for_best_model='eval_accuracy', 
        greater_is_better=True, 
        save_total_limit=8
    )
    return training_args


model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=class_num).cuda()

'''
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        print(inputs)
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits").cuda()
        pdb.set_trace()
        print(labels)
        # compute custom loss (suppose one has 5 labels with different weights)
        # need modification
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss 
'''

def ordinary_trainer(model):
    trainer = Trainer(
    model,
    args=set_training_args(),
    train_dataset=tokenized_train_datasets['train'],
    eval_dataset=tokenized_valid_datasets['train'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    )
    return trainer

def essemble_predict(tokenized_datasets, trainer):
    test_preds = trainer.predict(tokenized_datasets)
    final_preds = np.argmax(test_preds.predictions, axis=-1)
    list_preds = final_preds.tolist()
    list_result = [label for label in list_preds]
    return list_result

def compute_test_metrics(preds, labels):
    metrics = {}
    metrics['accuracy'] = accuracy(preds, labels)
    metrics['recall'] = recall(preds, labels)
    metrics['f1'] = F1(preds, labels)
    metrics['precision'] = precision(preds, labels)
    return metrics

import copy
trainer = ordinary_trainer(model)

trainer.predict(tokenized_test_datasets['train'])
ans = essemble_predict(tokenized_test_datasets['train'], trainer)
print(compute_test_metrics(ans, tokenized_test_datasets['train']['labels']))

if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()

pdb.set_trace()

trainer.train()

def essemble_predict(tokenized_datasets, trainer):
    test_preds = trainer.predict(tokenized_datasets)
    print(test_preds.predictions)
    final_preds = np.argmax(test_preds.predictions, axis=-1)
    print(final_preds)
    list_preds = final_preds.tolist()
    list_result = [label for label in list_preds]
    return list_result, test_preds.predictions

print(tokenized_test_datasets['train'])
list_preds = essemble_predict(tokenized_test_datasets['train'], trainer)
length = len(list_preds)
pdb.set_trace()
print(compute_test_metrics(list_preds[0], tokenized_test_datasets['train']['labels']))

pdb.set_trace()

ans_df = pd.DataFrame()
ans_df['text'] = tokenized_test_datasets['train']['text']
ans_df['pred_label'] = list_preds[0]
ans_df['pred_raw'] = [str(preds) for preds in list_preds[1]]
ans_df['true_label'] = tokenized_test_datasets['train']['label']
ans_df.to_csv('./' + dataset_name + '//longformer_pred.csv')
