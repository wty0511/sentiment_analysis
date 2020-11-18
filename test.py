
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
    
    start_time = time.time()
    print("Loading data...")

        # train
    model=make_model().to(Config.device)
    
    print('init')
    for  param in model.embedding.state_dict():
        print(param)
    

