# feature added

# with feature
import os
os.environ['STANFORD_PARSER'] = 'D://stanford-parser-full-2020-11-17//stanford-parser.jar'
os.environ['STANFORD_MODELS'] = 'D://stanford-parser-full-2020-11-17//stanford-parser-4.2.0-models.jar'
import torch
import math
import pdb
from transformers import BertModel, AutoTokenizer, BertTokenizer, RobertaTokenizer
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
from math import sqrt, log
from nltk import sent_tokenize
from nltk.parse.stanford import StanfordParser
from scipy.stats import entropy
import numpy as np
from transformers import DataCollatorWithPadding
nlp_parser=StanfordParser(model_path="D://stanford-parser-full-2020-11-17//stanford-parser-4.2.0-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

import argparse

parser = argparse.ArgumentParser(description='baseline')
parser.add_argument('--dataset', default='newsela', type=str, nargs='+',
                    help='dataset name')
parser.add_argument('--split', default='test', type=str)
parser.add_argument('--trunc', default=True, type=type(True))
args = parser.parse_args()

if type(args.dataset) == type(['aa']):
    args.dataset = args.dataset[0]
if type(args.split) == type(['aa']):
    args.split = args.split[0]
print(args)

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

class features:
    def __init__(self):
        self.Dale_Chall_List = pd.read_csv('C:\\Users\\housd810\\Desktop\\tmp\\' + "Dale Chall List.txt")
        self.word_difficulity = pd.read_csv('C:\\Users\\housd810\\Desktop\\tmp\\' + "I159729.csv", usecols=["Word", "Freq_HAL", "I_Zscore"])
        
        self.columns = ['Avg Parse Tree Height', 'Max Parse Tree Height', 'POS Distribution', 'MTLD', \
                        'Max Clause Num', 'Mean Clause Num', 'Max SBAR Num', 'Mean SBAR Num', \
                        'Max ratio of Dependency Clause', 'Mean Ratio of Dependency Clause', \
                        'Co-conj Per Passage']
        self.ptcol = ['CC', 'CD', 'NNS', 'VBP', 'NN', 'RB', 'MD', 'VB', 'VBZ', 'VBD', 'VBG', 'IN', 'JJ', 'FW', 'WDT', \
                 'RBR', 'PRP$', 'VBN', 'PRP', 'DT', 'JJS', 'RP', 'JJR', 'WRB', 'WP', 'NNP', 'WP$', \
                 'PDT', 'RBS', 'NNPS', 'SYM', 'EX', 'TO', 'UH']

        self.co_conj = ['for', 'and', 'nor', 'but', 'or', 'yet', 'so']
        self.TTR_mode = ['', 'corrected', 'bi', 'root','uber']
        self.var_mode = [['VB'], ['JJ'], ['RB'], ['NN'], ['VB', 'JJ', 'RB', 'NN']]
        
        self.TTRs = ['TTR ' + str(mode) for mode in self.TTR_mode]
        self.vars = ['Word Variation ' + " ".join(mode) for mode in self.var_mode]
        
        self.columns.extend(self.TTRs)
        self.columns.extend(self.vars)
        self.columns.extend(['class'])
        
    # Preprocessing
    def preprocessing(self, text1):
        clean_txt = re.sub('[^a-zA-Z\.,0-9\!\?:]', ' ', text1)
        clean_txt = [word for word in clean_txt.lower().split() if len(word)]
        return clean_txt
    
    def parse_tree(self, txt):
        clean_txt = " ".join(txt)
        txt = sent_tokenize(clean_txt)
        for i in range(len(txt)):
            if len(txt[i].split()) > 50:
                txt[i] = " ".join(txt[i].split()[:50])
        try:
            list_result = list(nlp_parser.raw_parse_sents(txt))
        except:
            print(clean_txt)
            pdb.set_trace()
        return list_result
    
    def parse_tree_height(self, txt):
        parse_list = self.parse_tree(txt)
        tree_heights = []
        for tree in parse_list:
            for sub_tree in tree:
                tree_heights.append(sub_tree.height())
        return tree_heights
    
    def pos_count_in_list(self, list1):
        pt = pos_tag(list1)
        dictpt = dict(pt)
        dictpt = Counter(dictpt.values())
        vl = []
        for i in self.ptcol:
            if i in dictpt.keys():
                vl.append(dictpt[i])
            else:
                vl.append(0)
        return vl
    
    # from the paper: Linguistic Features in Readability Assessment
    def POSD(self, text):
        # calc the KL divergence between sentence POS count distribution and document POS count distribution
        POSD = 0.0
        sent_pos = []
        str_txt = " ".join(text)
        txt = sent_tokenize(str_txt)
        doc_pos = self.pos_count_in_list(text)
        doc_pos = [pos+1 for pos in doc_pos]
        doc_pos = np.array(doc_pos) / sum(doc_pos)
        for sent in txt:
            sent_tmp_pos = self.pos_count_in_list(sent.split())
            sent_tmp_pos = [pos+1 for pos in sent_tmp_pos]
            sent_pos.append(np.array(sent_tmp_pos) / sum(sent_tmp_pos))
        for s_pos in sent_pos:
            POSD = POSD + entropy(doc_pos, s_pos)
        POSD = POSD / len(sent_pos)
        return POSD
    
    def MTLD(self, pt):
        # processed
        pos_dict = {}
        TTR = 0.72
        factors = 0
        total_token = 1
        cur_TTR = 0
        for pt_token in pt:
            pos_dict[pt_token[1]] = 1
            cur_TTR = len(list(pos_dict.keys())) / total_token
            if cur_TTR < TTR:
                factors = factors + 1
                pos_dict.clear()
                total_token = 0
                cur_TTR = 0
            total_token = total_token + 1
        seq_MTLD = factors + cur_TTR
        
        factors = 0
        total_token = 1
        cur_TTR = 0
        pos_dict.clear()
        pt.reverse()
        for pt_token in pt:
            pos_dict[pt_token[1]] = 1
            cur_TTR = len(list(pos_dict.keys())) / total_token
            if cur_TTR < TTR:
                factors = factors + 1
                pos_dict.clear()
                total_token = 0
            total_token = total_token + 1
        rev_MTLD = factors + cur_TTR
        print(seq_MTLD, rev_MTLD)
        return seq_MTLD + rev_MTLD

    def calc_clause(self, p_tree):
        total_clause = []
        total_SBAR = []
        for tree in p_tree:
            for sub_tree in tree:
                sent_clause = len(list(sub_tree.subtrees(filter=lambda x: (x.label() == 'S' or x.label() == 'SBAR' or x.label == 'SBARQ'))))
                sent_SBAR = len(list(sub_tree.subtrees(filter=lambda x: (x.label() == 'SBAR' or x.label() == 'SBARQ'))))
                total_clause.append(sent_clause)
                total_SBAR.append(sent_SBAR)
        return total_clause, total_SBAR

    def max_clause(self, total_clause):
        return max(total_clause)

    def mean_clause(self, total_clause):
        return np.mean(total_clause)

    def max_SBAR(self, total_SBAR):
        return max(total_SBAR)

    def mean_SBAR(self, total_SBAR):
        return np.mean(total_SBAR)

    def max_ratio_dclause(self, total_clause, total_SBAR):
        ratio = [total_SBAR[i] / total_clause[i] for i in range(len(total_clause)) if total_clause[i]]
        if len(ratio):
            return max(ratio)
        else:
            return 0

    def mean_ratio_dclause(self, total_clause, total_SBAR):
        ratio = [total_SBAR[i] / total_clause[i] for i in range(len(total_clause)) if total_clause[i]]
        if len(ratio):
            return np.mean(ratio)
        else:
            return 0

    def word_variation(self, pt, modes:list):
        full_dict = {}
        content_dict = {}
        for p_token in pt:
            for mode in modes:
                if mode in p_token[1]:
                    content_dict[p_token[1]] = 1
            full_dict[p_token[1]] = 1
        print(full_dict)
        print(content_dict)
        return len(content_dict) / len(full_dict)

    def co_conj_per_passage(self, text):
        conj = 0
        for word in text:
            if text in self.co_conj:
                conj = conj + 1
        return conj

    def TTR(self, pt, mode:str=None):
        pt_dict = {}
        for p_token in pt:
            pt_dict[p_token[1]] = 1
        if len(pt_dict) == 0 or len(pt) == 0 or len(pt_dict) == len(pt):
            return 0
        if mode == '':
            return len(pt_dict) / len(pt)
        elif mode == 'corrected':
            return len(pt_dict) / (sqrt(2) * len(pt))
        elif mode == 'bi':
            return log(len(pt_dict)) / log(len(pt))
        elif mode == 'root':
            return len(pt_dict) / sqrt(len(pt))
        elif mode == 'uber':
            return log(len(pt_dict)) * log(len(pt_dict)) / log(len(pt) / len(pt_dict))

    def fextr(self, text):
        text=str(text)
        c2 = []
        txt = self.preprocessing(text)
        parse_list = self.parse_tree(txt)
        tree_heights = self.parse_tree_height(txt)
        print(tree_heights)
        pt = pos_tag(txt)
        total_clause, total_SBAR = self.calc_clause(parse_list)
        if len(tree_heights):
            c2.append(np.mean(tree_heights))
            c2.append(max(tree_heights))
            c2.append(self.POSD(txt))
            c2.append(self.MTLD(pt))
            c2.append(self.max_clause(total_clause))
            c2.append(self.mean_clause(total_clause))
            c2.append(self.max_SBAR(total_SBAR))
            c2.append(self.mean_SBAR(total_SBAR))
            c2.append(self.max_ratio_dclause(total_clause, total_SBAR))
            c2.append(self.mean_ratio_dclause(total_clause, total_SBAR))
            c2.append(self.co_conj_per_passage(txt))
            for mode in self.TTR_mode:
                c2.append(self.TTR(pt, mode))
            for mode in self.var_mode:
                c2.append(self.word_variation(pt, mode))
        else:
            c2.extend([0, 0])
            c2.append(0)
            c2.append(0)
            c2.append(0)
            c2.append(0)
            c2.append(0)
            c2.append(0)
            c2.append(0)
            c2.append(0)
            c2.append(0)
            for mode in self.TTR_mode:
                c2.append(0)
            for mode in self.var_mode:
                c2.append(0)
        return c2
    def create_dataframe(self, path):
        data = []
        data_csv = pd.read_csv(path)
        print(data_csv)
        for index, row in data_csv.iterrows():
            print(index)
            #print(row[0])
            c2 = [row[1]]
            if args.trunc:
                if len(tokenizer.encode(row[1])) > 512:
                    trunced_text = tokenizer.decode(tokenizer.encode(row[1])[1:511])
                else:
                    trunced_text = tokenizer.decode(tokenizer.encode(row[1])[1:-1])
                print(len(tokenizer.encode(row[1])), len(tokenizer.encode(trunced_text)))
                c2.extend(self.fextr(trunced_text))
            else:
                c2.extend(self.fextr(row[1]))
            # else:
                # trunced_text = ''
            # txt = self.preprocessing(row[1])
            # parse_list = self.parse_tree(txt)
            # total_clause, total_SBAR = self.calc_clause(parse_list)
            # m_dclause = self.mean_ratio_dclause(total_clause, total_SBAR)
            c2.append(row[2])
            data.append(c2)
            
            # assert len(c2) == len(['text'] + text_features.columns)
            if index % 50 == 0:
                print(index)
        df = pd.DataFrame(data, columns=['text'] + self.columns)
        # df = pd.DataFrame(data, columns=['text'] + ['Mean Ratio of Dependency Clause', 'class'])
        return df

text_features = features()
print(len(['text'] + text_features.columns))
df = pd.DataFrame()
try:
        df = text_features.create_dataframe('./' + args.dataset + '//'+ args.dataset + '_' + args.split + '.csv')
    df.to_csv('.//' + args.dataset + '//' + args.dataset + '_new_feature_' + args.split + '.csv')
except:
    df.to_csv('.//' + args.dataset + '//' + args.dataset + '_new_feature_' + args.split + '.csv')
