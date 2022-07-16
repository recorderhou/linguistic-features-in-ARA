# store text and text features in csv files
import os
# os.environ['STANFORD_PARSER'] = 'D://stanford-parser-full-2020-11-17//stanford-parser.jar'
# os.environ['STANFORD_MODELS'] = 'D://stanford-parser-full-2020-11-17//stanford-parser-4.2.0-models.jar'
import torch
import math
import pdb
from transformers import BertModel, AutoTokenizer, BertTokenizer
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
from nltk.parse.stanford import StanfordParser
from scipy.stats import entropy
import numpy as np
from transformers import DataCollatorWithPadding
from math import sqrt

class features:
    def __init__(self):
        self.Dale_Chall_List = pd.read_csv('C:\\Users\\housd810\\Desktop\\tmp\\' + "Dale Chall List.txt")
        self.word_difficulity = pd.read_csv('C:\\Users\\housd810\\Desktop\\tmp\\' + "I159729.csv", usecols=["Word", "Freq_HAL", "I_Zscore"])
        '''
        self.columns = ['Total Number Of Sentences', 'Average Sentence Length', 'Average Word Difficulty',
                   'Average Word Length', \
                   'Number of Uncommon Words', 'Number of Unique Words', 'Words with 1 to 3 syllables', \
                   "Words with 4 syllables", "Words with 5 syllables", "Words with 6 syllables", \
                   "Words with more than 7 syllables", "Average number of syllables", "Average Height of Parse Tree"
            "Max Height of Parse Tree", "POS distribution"]
        '''
        
        # just for test
        self.columns = ['Total Number Of Sentences', 'Average Sentence Length', 'Average Word Difficulty',
                   'Average Word Length', \
                   'Number of Uncommon Words', 'Number of Unique Words', 'Words with 1 to 3 syllables', \
                   "Words with 4 syllables", "Words with 5 syllables", "Words with 6 syllables", \
                   "Words with more than 7 syllables", "Average number of syllables", "ARI", "FRE", "SMOG"]
        self.ptcol = ['CC', 'CD', 'NNS', 'VBP', 'NN', 'RB', 'MD', 'VB', 'VBZ', 'VBD', 'VBG', 'IN', 'JJ', 'FW', 'WDT', \
                 'RBR', 'PRP$', 'VBN', 'PRP', 'DT', 'JJS', 'RP', 'JJR', 'WRB', 'WP', 'NNP', 'WP$', \
                 'PDT', 'RBS', 'NNPS', 'SYM', 'EX', 'TO', 'UH']
        self.ptcol.sort()
        self.columns.extend(self.ptcol)
        self.columns.extend(['class'])

    # Preprocessing
    def preprocessing(self, text1):
        text1 = re.sub('[^a-zA-Z]', ' ', text1)
        return [word for word in text1.lower().split() if not word in set(stopwords.words('english'))]

    # Feature extraction
    def avg_sentence_length(self, text, num_sents):
        avg = float(len(text) / num_sents)
        return avg

    def avg_word_length(self, text):
        s = 0
        for w in text:
            s += len(w)
            a = s / len(text)
        return a

    # word difficulty
    def avg_word_difficulty(self, text):
        diff = 0
        for w in text:
            if w in self.word_difficulity.Word.tolist():
                if float(self.word_difficulity.loc[self.word_difficulity['Word'] == w]['I_Zscore']) > 0:
                    diff += 1
        a = diff / len(text) * 100
        # print(diff)
        return [a]

    # 计算单音节词
    def syllable_count_single_word(self, word):
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count += 1
        return count

    # 计算音节
    def avg_syllables(self, text):
        s = 0
        for w in text:
            s += self.syllable_count_single_word(w)
        a = s / len(text)
        return a

    # 计算词性
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

    def dif_words(self, text):
        frequency = Counter(text)
        return len(frequency)

    def freq_syl(self, text):
        count = [0, 0, 0, 0, 0]
        uniq_words = Counter(text).keys()
        for word in uniq_words:
            x = self.syllable_count_single_word(word)
            if (x > 1 and x <= 3):
                count[0] += 1
            elif (x == 4):
                count[1] += 1
            elif (x == 5):
                count[2] += 1
            elif (x == 6):
                count[3] += 1
            else:
                count[4] += 1
        return count

    def not_in_dale_chall(self, text):
        n = [w for w in text if w not in self.Dale_Chall_List]
        n1 = len(n)
        return n1
    
    '''
    def avg_parse_tree_height(self, text):
        clean_txt = re.sub('[^a-zA-Z.]', ' ', text)
        clean_txt = [word for word in clean_txt.lower().split() if not word in set(stopwords.words('english'))]
        clean_txt = " ".join(clean_txt)
        txt = sent_tokenize(clean_txt)
        list_result = list(parser.raw_parse_sents(txt))
        str_result = []
        for tree in list_result:
            for sub_tree in tree:
                str_result.append(str(sub_tree))
        max_depth = []
        for str_tree in str_result:
            depth = 0
            m_depth = 0
            for chars in str_tree:
                if chars == '(':
                    depth += 1
                    if depth > m_depth:
                        m_depth = depth
                if chars == ')':
                    depth -= 1
            if m_depth > 3:
                max_depth.append(m_depth)
        avg_height = np.mean(np.array(max_depth))
        return avg_height
    
    def max_parse_tree_height(self, text):
        clean_txt = re.sub('[^a-zA-Z.]', ' ', text)
        clean_txt = [word for word in clean_txt.lower().split() if not word in set(stopwords.words('english'))]
        clean_txt = " ".join(clean_txt)
        txt = sent_tokenize(clean_txt)
        list_result = list(parser.raw_parse_sents(txt))
        str_result = []
        for tree in list_result:
            for sub_tree in tree:
                str_result.append(str(sub_tree))
        max_depth = []
        for str_tree in str_result:
            depth = 0
            m_depth = 0
            for chars in str_tree:
                if chars == '(':
                    depth += 1
                    if depth > m_depth:
                        m_depth = depth
                if chars == ')':
                    depth -= 1
            if m_depth > 3:
                max_depth.append(m_depth)
        max_height = np.max(np.array(max_depth))
        return max_height
    '''
    def ARI(self, avg_word_length, avg_sent_length):
        return 4.71 * avg_word_length + 0.5 * avg_sent_length - 21.43
    
    def FRE(self, word_num, sent_num, freq_syl):
        return 206.835 - 1.015 * word_num / sent_num - 84.6 * np.sum(freq_syl) / word_num
    
    def SMOG(self, freq_syl, sent_num):
        poly_syl = np.sum(np.array(freq_syl[1:]))
        return 1.043 * sqrt(poly_syl * 30 / sent_num) + 3.1291
    
    def Coh(self, text):
        return
    
    '''
    # from the paper: Linguistic Features in Readability Assessment
    def POSD(self, text):
        # calc the KL divergence between sentence POS count distribution and document POS count distribution
        POSD = 0.0
        sent_pos = []
        clean_txt = re.sub('[^a-zA-Z.]', ' ', text)
        clean_txt = [word for word in clean_txt.lower().split() if not word in set(stopwords.words('english'))]
        clean_txt = " ".join(clean_txt)
        txt = sent_tokenize(clean_txt)
        doc_pos = self.pos_count_in_list(txt)
        for sent in txt:
            sent_pos.append(self.pos_count_in_list(txt))
        for s_pos in sent_pos:
            POSD = POSD + entropy(s_pos, doc_pos)
        POSD = POSD / len(sent_pos)
        return POSD
    '''

    def fextr(self, text):
        text=str(text)
        c2 = []
        txt = self.preprocessing(text)
        avg1 = len(sent_tokenize(text))
        c2.append(avg1)
        avg_sent_len = self.avg_sentence_length(txt, avg1)
        c2.append(avg_sent_len)
        avg_word_diff = self.avg_word_difficulty(txt)
        c2.extend(avg_word_diff)
        avg_word_len = self.avg_word_length(txt)
        no_dale_hall = self.not_in_dale_chall(txt)
        total_diff_words = self.dif_words(txt)
        c2.extend([avg_word_len, no_dale_hall, total_diff_words])
        freq_syls = self.freq_syl(txt)
        c2.extend(freq_syls)
        avg_syls = self.avg_syllables(txt)
        c2.append(avg_syls)
        ARIc = self.ARI(avg_word_len, avg_sent_len)
        total_word = len([word for word in text.split() if len(word)])
        FREc = self.FRE(total_word, avg1, freq_syls)
        SMOGc = self.SMOG(freq_syls, avg1)
        c2.extend([ARIc, FREc, SMOGc])
        '''
        c2.append(self.avg_parse_tree_height(text))
        c2.append(self.max_parse_tree_height(text))
        c2.append(self.POSD(text))
        '''
        vallist = self.pos_count_in_list(txt)
        c2.extend(vallist)
        return c2
    def create_dataframe(self, path):
        data = []
        data_csv = pd.read_csv(path)
        print(data_csv)
        for index, row in data_csv.iterrows():
            #print(row[0])
            c2 = [row[1]]
            c2.extend(self.fextr(row[1]))
            c2.append(row[2])
            data.append(c2)
        #print(data)
            
            assert len(c2) == len(['text'] + text_features.columns)
            if index % 50 == 0:
                print(index)
        df = pd.DataFrame(data, columns=['text'] + self.columns)
        return df

text_features = features()
print(len(['text'] + text_features.columns))
df = text_features.create_dataframe('./raz//raz_train.csv')
df.to_csv('.//raz//' + 'raz_feature_train.csv')