# with feature
import os
os.environ['CUDA_VISIBLE_DEVICES']= '4,5,6,7' 
# os.environ['STANFORD_PARSER'] = 'D://stanford-parser-full-2020-11-17//stanford-parser.jar'
# os.environ['STANFORD_MODELS'] = 'D://stanford-parser-full-2020-11-17//stanford-parser-4.2.0-models.jar'
import torch
import math
import pdb
from transformers import BertModel, AutoTokenizer, BertTokenizer, RobertaTokenizer, RobertaModel
from torch.utils.data import Dataset
import torch.utils.data as Data
import torch.nn as nn
from torch.optim import *
import copy
from nltk import sent_tokenize
import re
import os
import pandas as pd
import nltk
import glob
from nltk.corpus import stopwords
from nltk.tag import pos_tag

from collections import Counter
from nltk import sent_tokenize
# from nltk.parse.stanford import StanfordParser
from scipy.stats import entropy
import numpy as np
from transformers import DataCollatorWithPadding
# parser=StanfordParser(model_path="D://stanford-parser-full-2020-11-17//stanford-parser-4.2.0-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

checkpoint = './roberta-base/'
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
best_acc = 0.0

import argparse

parser = argparse.ArgumentParser(description='error analysis argument')
parser.add_argument('--dataset_path', default='./raz//raz_full_trunc_', type=str, nargs='+',
                    help='dataset name')
parser.add_argument('--class_num', default=3, type=int, nargs='+',
                    choices=[3, 5], help='total class num')
parser.add_argument('--gpus', default='5,7')
parser.add_argument('--feature_num', default=70, type=int)
parser.add_argument('--train_steps', default=30, type=int)
parser.add_argument('--data_portion', default=1.0, type=float)
parser.add_argument('--max_seq_len', default=512, type=int)
parser.add_argument('--test_time', default=3, type=int)
args = parser.parse_args()

if type(args.dataset_path) == type(['aa']):
    args.dataset_path = args.dataset_path[0]
if type(args.class_num) == type([3]):
    args.class_num = args.class_num[0]
if type(args.feature_num) == type([3]):
    args.feature_num = args.feature_num[0]
if type(args.train_steps) == type([3]):
    args.train_steps = args.train_steps[0]
if type(args.data_portion) == type([3.0]):
    args.data_portion = args.data_portion
if type(args.max_seq_len) == type([3]):
    args.max_seq_len = args.max_seq_len[0]
if type(args.test_time) == type([3]):
    args.test_time = args.test_time[0]

args.max_seq_len = int(args.max_seq_len)


print(args)

class RobertaModelwithFeature(nn.Module):
    def __init__(self, feature_num=70, num_labels=5):
        super(RobertaModelwithFeature, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(0.1)
        self.statistic_layer = nn.Sequential(nn.Linear(feature_num+768, feature_num+768),
                                          nn.Tanh(),
                                          nn.Dropout(0.1),
                                          nn.Linear(feature_num+768, num_labels))

    def forward(self, batched_data):
        # at first put tokenized text in bert
        # then concate the output of bert model with selected features
        # then go through a linear layer to get autput

        # input_ids, attention_mask, token_type_ids, position_ids
        input_ids, attention_mask, labels, feature = batched_data
        inputs = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        feature = feature.to(torch.float32).cuda()

        roberta_outputs = self.roberta(
                input_ids, attention_mask)
        
        # (batch_size, sequence_length, hidden_size)
        # (batch_size, hidden_size)
        sequence_output = roberta_outputs.last_hidden_state[:,0,:]
        dropout_output = self.dropout(sequence_output)
        full_feature = torch.cat((dropout_output, feature), 1)
        prediction = self.statistic_layer(full_feature)
        return prediction

def train(train_dataloader, model, optimizer, scheduler, criterion, epoch, n_labels):
    labeled_train_iter = iter(train_dataloader)
    
    model.train()
    total_steps = 0
    for batch_idx, batched_data in enumerate(train_dataloader):

        total_steps += 1

        batched_input = labeled_train_iter.next()
        batch_size = len(batched_input[0])
        inputs_x = batched_input[0]
        targets_x = batched_input[-2]
        targets_x = torch.zeros(batch_size, n_labels).scatter_(
            1, targets_x.long().view(-1, 1), 1)

        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)

        all_inputs = inputs_x
        all_targets = targets_x

        logits = model(batched_input)
        
        loss = criterion(logits, batched_input[-2].long().cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print("epoch {}, step {}, loss {}".format(
                epoch, batch_idx, loss.item()))

class FeaturedData(Dataset):
    # Data loader for labeled data
    def __init__(self, dataset_text, dataset_label, dataset_feature, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.labels = dataset_label
        self.feature = dataset_feature
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.text[idx]
        feature = self.feature[idx].astype(float)
        tokenized_text = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_seq_len)
        input_ids = tokenized_text['input_ids']
        attention_mask = tokenized_text['attention_mask']
        return (torch.tensor(input_ids), torch.tensor(attention_mask), self.labels[idx], torch.tensor(feature))

def get_data(data_path):
    # Load the tokenizer for bert
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    max_seq_len = args.max_seq_len

    train_df = pd.read_csv(data_path+'feature_train.csv')
    valid_df = pd.read_csv(data_path+'feature_valid.csv')
    test_df = pd.read_csv(data_path+'feature_test.csv')
    if 'trunced_text' in train_df.columns:
        del train_df['trunced_text']
        del valid_df['trunced_text']
        del test_df['trunced_text']
    if 'Unnamed: 0' in train_df.columns:
        del train_df['Unnamed: 0']
        del valid_df['Unnamed: 0']
        del test_df['Unnamed: 0']
    
    if 'Unnamed: 0' in train_df.columns:
        train_df = train_df.sample(frac=args.data_portion, ignore_index=True)
    else:
        train_df = train_df.sample(frac=args.data_portion)
    if 'Unnamed: 0' in train_df.columns:
        del train_df['Unnamed: 0']
        del valid_df['Unnamed: 0']
        del test_df['Unnamed: 0']
    if 'Unnamed: 0.1' in train_df.columns:
        del train_df['Unnamed: 0.1']
        del valid_df['Unnamed: 0.1']
        del test_df['Unnamed: 0.1']
    print(train_df.columns)
    
    train_labels = []
    train_text = []
    train_feature = []
    
    # pdb.set_trace()
    
    for i in range(len(train_df)):
        train_labels.append(train_df.iloc[i][-1])
        train_text.append(train_df.iloc[i].text)
        if 'trunc' in data_path:
            train_feature.append(train_df.iloc[i][1:-1])
        else:
            train_feature.append(train_df.iloc[i][1:-1])
      
    # Here we only use the bodies and removed titles to do the classifications
    train_labels = np.array(train_labels)
    train_text = np.array(train_text)
    train_feature = np.array(train_feature)
    
    test_labels = []
    test_text = []
    test_feature = []
    
    for i in range(len(test_df)):
        test_labels.append(test_df.iloc[i][-1])
        test_text.append(test_df.iloc[i].text)
        if 'trunc' in data_path:
            test_feature.append(test_df.iloc[i][1:-1])
        else:
            test_feature.append(test_df.iloc[i][1:-1])

    test_labels = np.array(test_labels)
    test_text = np.array(test_text)
    test_feature = np.array(test_feature)
    
    valid_labels = []
    valid_text = []
    valid_feature = []
    
    for i in range(len(valid_df)):
        valid_labels.append(valid_df.iloc[i][-1])
        valid_text.append(valid_df.iloc[i].text)
        if 'trunc' in data_path:
            valid_feature.append(valid_df.iloc[i][1:-1])
        else:
            valid_feature.append(valid_df.iloc[i][1:-1])
            
    
    valid_labels = np.array(valid_labels)
    valid_text = np.array(valid_text)
    valid_feature = np.array(valid_feature)
    
    print(test_labels[:20])
    n_labels = max(test_labels) + 1

    # Build the dataset class for each set
    train_dataset = FeaturedData(
        train_text, train_labels, train_feature, tokenizer, max_seq_len)
    val_dataset = FeaturedData(
        valid_text, valid_labels,valid_feature, tokenizer, max_seq_len)
    test_dataset = FeaturedData(
        test_text, test_labels, test_feature, tokenizer, max_seq_len)

    print("#Labeled: {}, Val {}, Test {}".format(len(
        train_labels), len(valid_labels), len(test_labels)))

    return train_dataset, val_dataset, test_dataset, n_labels


def validate(valloader, model, criterion, epoch):
    model.eval()
    with torch.no_grad():
        loss_total = 0
        total_sample = 0
        acc_total = 0
        correct = 0
        
        tmp_df = pd.DataFrame()
        text = []
        label = []
        prediction = []
        
        for batch_idx, batched_data in enumerate(valloader):
            input_ids, attention_mask, labels, feature = batched_data
            num_input = input_ids.numpy().tolist()
            str_text = [tokenizer.decode(nums) for nums in num_input]
            text.extend(str_text)
            label.extend(labels.numpy().tolist())
            
            inputs, targets = input_ids.cuda(), labels.long().cuda(non_blocking=True)
            outputs = model(batched_data)
            loss = criterion(outputs, targets)

            _, predicted = torch.max(outputs.data, 1)
            
            prediction.extend(predicted.cpu().numpy().tolist())

            if batch_idx % 5 == 0:
                print("Sample some true labeles and predicted labels")
                print(predicted[:20])
                print(targets[:20])

            correct += (np.array(predicted.cpu()) ==
                        np.array(targets.cpu())).sum()
            loss_total += loss.item() * inputs.shape[0]
            total_sample += inputs.shape[0]

        acc_total = correct/total_sample
        loss_total = loss_total/total_sample

        tmp_df['text'] = text
        tmp_df['label'] = label
        tmp_df['prediction'] = prediction

    return loss_total, acc_total, tmp_df

def main():
    global best_acc
    # Read dataset and build dataloaders
    train_set, val_set, test_set, n_labels = get_data(
        args.dataset_path)
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    
    for data in train_set:
        print('?')
        break
    
    train_loader = Data.DataLoader(
        dataset=train_set, batch_size=8, shuffle=True)
    val_loader = Data.DataLoader(
        dataset=val_set, batch_size=512, shuffle=False)
    test_loader = Data.DataLoader(
        dataset=test_set, batch_size=512, shuffle=False)

    # Define the model, set the optimizer
    model = RobertaModelwithFeature(feature_num=args.feature_num, num_labels=args.class_num).cuda()
    model = nn.DataParallel(model)
    optimizer = AdamW(
        [
            {"params": model.module.roberta.parameters(), "lr": 0.00001},
            {"params": model.module.statistic_layer.parameters(), "lr": 0.001},
        ])

    num_warmup_steps = math.floor(50)
    num_total_steps = 50

    scheduler = None
    #WarmupConstantSchedule(optimizer, warmup_steps=num_warmup_steps)
    criterion = nn.CrossEntropyLoss()

    test_accs = []
    max_test_acc = 0.0
    test_df = pd.DataFrame()
    
    '''
    if os.path.exists(args.dataset_path + str(args.max_seq_len) + '_' + str(args.data_portion) + '_' + str(args.test_time) + '_with_feature.pth'):
        model = torch.load(args.dataset_path + str(args.max_seq_len) + '_' + str(args.data_portion) + '_' + str(args.test_time) + '_with_feature.pth')
        test_loss, test_acc, test_df = validate(
                test_loader, model, criterion, 0)
        test_df.to_csv(args.dataset_path + str(args.max_seq_len) +  '_' + str(args.data_portion) +  '_' + str(args.test_time) + '_feature_result.csv')
    pdb.set_trace()
    '''

    # Start training
    for epoch in range(100):

        train(train_loader, model, optimizer,
              scheduler, criterion, epoch, n_labels)

        val_loss, val_acc, _ = validate(
            val_loader, model, criterion, epoch)

        print("epoch {}, val acc {}, val_loss {}".format(
            epoch, val_acc, val_loss))

        if val_acc >= best_acc:
            best_acc = val_acc
            test_loss, test_acc, tmp_test_df = validate(
                test_loader, model, criterion, epoch)
            test_accs.append(test_acc)
            if test_acc > max_test_acc:
                test_df = tmp_test_df
                max_test_acc = test_acc
                torch.save(model, args.dataset_path + str(args.max_seq_len) + '_' + str(args.data_portion) + '_' + str(args.test_time) + '_with_feature.pth')
                print('-------model saved--------')

            print("epoch {}, test acc {},test loss {}".format(
                epoch, test_acc, test_loss))

        print('Epoch: ', epoch)

        print('Best acc:')
        print(best_acc)

        print('Test acc:')
        print(test_accs)

    print("Finished training!")
    print('Best acc:')
    print(best_acc)

    print('Test acc:')
    print(test_accs)
    
    test_df.to_csv(args.dataset_path + str(args.max_seq_len) +  '_' + str(args.data_portion) +  '_' + str(args.test_time) + '_feature_result.csv')


if __name__ == '__main__':
    main()