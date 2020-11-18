# coding: UTF-8
import time
import torch
import numpy as np
from torch import nn
import argparse
from utils.util import build_dataset,build_iterator,get_time_dif
from transformer import make_model

from train import train
from Config import Config
from torchsummary import summary
from simple import ff

if __name__ == '__main__':
    print('in')
    np.random.seed(1)
   
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    PATH='data_balance.csv'
    start_time = time.time()
    print("Loading data...")

    train_data= build_dataset('./train.csv')
    dev_data=build_dataset('./test.csv')
    
    #print(train_data)
    train_iter = build_iterator(train_data)
    dev_iter = build_iterator(dev_data)
    test_iter = dev_iter
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    
    # train
    model=make_model().to(Config.device)
    #model=ff().to('cuda')
    print('init')
    '''
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    '''
    for name, w in model.named_parameters():
        
        if 'embedding' not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                nn.init.xavier_normal_(w)

            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass
    print(model)
    #print(model.parameters)
    
    train(model, train_iter, dev_iter, test_iter)
