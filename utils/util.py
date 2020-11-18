import copy
import torch.nn as nn
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import pandas as pd
import time
import re
from datetime import timedelta
import random
from Config import Config
from sentence import get_key_sentences
from importantsentence import ImportantSentence
from sklearn.utils import shuffle
from NWVfinder import *

finder = NWVfinder()



def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


imsen=ImportantSentence(loadw2vmodle=True)


def build_dataset(path):
    dataset=pd.read_csv(path,header=None)
    print(dataset)
    i=0
    data_p=[]
    pos=0
    neg=0
    neu=0
    dataset=dataset.sample(frac=Config.dataset_size)
    #dataset=dataset.head(n=Config.dataset_size)
    l=[]
    c=0
    for tup in tqdm(dataset.itertuples()):
        
        if(tup[3]==0):
            pos+=1
        if(tup[3]==1):
            neu+=1
        if(tup[3]==2):
            neg+=1
        
        
        try:
            context=[]
            #print(tup[1])
            #print(tup[2])
            
            imsen.analyze([[tup[1],tup[2]]],outputrawpara=False,impsentnum1=Config.context_length,delwordcixing=['x','p','u'])
            #print(imsen.improtantsentences)
            context.extend(imsen.improtantsentences[0])
            
            '''
            for i in context:
                if len(i)>80:
                    c+=1
            '''
            #print(context)
            #context= re.split(r'[。！？]',tup[2])[:Config.context_length] 
            #context.extend(get_key_sentences(tup[1],tup[2],Config.context_length))
            #[wordlist, addSentence] = finder.searchSentence(''.join(context))
            
            for s in context:
                [wordlist, addSentence] = finder.searchSentence(s)
                s+=addSentence
                #print(s)
            
            #context.append(addSentence)
            
            context.extend(['']* max(Config.context_length-len(context),0))
            #print(wordlist)
            #print(addSentence)
            #print(len(context))
            data_p.append(([tup[1]],context ,tup[3]))
        
        except:
            print('error')
            continue
        
    
    #print(pos)
    #print(neu)
    #print(neg)
    #print(data_p) 
    #length=len(data_p)
    #print(np.mean(l))
    #print(c)
    return  data_p





class DatasetIterater(object):
    def __init__(self, batches, batch_size=Config.batch_size, device=Config.device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % batch_size != 0:
            self.residue = True
        self.index = 0
        self.device = device
        

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            return batches
            
    def __iter__(self):
        return self
        
    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset):
    #print(Config.batch_size)
    iter = DatasetIterater(dataset, Config.batch_size, Config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
