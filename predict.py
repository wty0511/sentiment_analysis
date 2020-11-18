import time
import torch
import numpy as np
from torch import nn
import argparse
from utils.util import build_dataset,build_iterator,get_time_dif
from transformer_backup import make_model
from train import train
import pandas as pd 
from Config import Config
from torchsummary import summary
from simple import ff
from sentence import get_key_sentences
def build_dataset(path):
    dataset=pd.read_csv(path)
    print(dataset)
    data_p=[]
    for tup in dataset.itertuples():

        try:
            
            context=[]
            #print(tup[3])
            context.extend(get_key_sentences(tup[3],tup[4],Config.context_length))
            context.extend(['']* max(Config.context_length-len(context),0))
            data_p.append(([tup[3]],context ,tup[4]))

        except:
            print('error')
            continue

        
    return  data_p


if __name__ == '__main__':
    print('in')
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True 
    data=build_dataset('FINAL.csv')
    start_time = time.time()
    print("Loading data...")    
    test_iter = build_iterator(data)
    print(len(test_iter))
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    result =pd.DataFrame(columns=('title','content','sentiment'))
    # train
    model = torch.load(Config.save_path).to('cuda')
    print(model)
    model.eval()
    
    #print(model.parameters)
    for batch in test_iter:
        src=[x[1] for x in batch]
        tgt = [x[0] for x in batch]
        content=[x[2] for x in batch]
        outputs = model(src,tgt)
        predic = torch.max(outputs.data, 1)[1].cpu().numpy().tolist()[0]
        #print(src)
        #print(tgt)
        #print(content)
        result = result.append(pd.DataFrame({'title': [tgt[0][0]], 'content': [content[0]],'sentiment': [predic]}), ignore_index=True)
    result.to_csv('predict_final.csv',index=None)
