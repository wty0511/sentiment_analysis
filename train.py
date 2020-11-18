# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from transformers import AdamW
from tensorboardX import SummaryWriter
from utils.util import get_time_dif
from transformer import Config
def f1score(pre, true):
    print(pre)
    print(true)
    pos=0
    neg=0
    nu=0
    pos_pre=0
    neg_pre=0
    nu_pre=0
    pos_acc=0
    neg_acc=0
    nu_acc=0

    for i, j in zip(pre, true):

        if (j == 2):
            neg += 1
        if (j == 1):
            nu += 1
        if (j == 0):
            pos += 1
        if (i == 2 and i == j):
            neg_acc += 1
        if (i == 1 and i == j):
            nu_acc += 1
        if (i == 0 and i == j):
            pos_acc += 1
        if (i == 2):
            neg_pre += 1
        if (i == 1):
            nu_pre += 1
        if (i == 0):
            pos_pre += 1
   
   
    print(pos)
    print(pos_acc)
    print(nu)
    print(nu_acc)
    print(neg)
    print(neg_acc)
    
    try:
        pos_r=pos_acc/pos
    except:
        pos_r=0
    try:
        nu_r=nu_acc/nu
    except:
        nu_r=0
    try:
        neg_r=neg_acc/neg
    except:
        neg_r=0
    try:
        pos_a=pos_acc/pos_pre
    except:
        pos_a=0
    try:
        nu_a=nu_acc/nu_pre
    except:
        nu_a=0
    try:
        neg_a=neg_acc/neg_pre
    except:
        neg_a=0
    print(pos_a)
    print(pos_r)
    print(nu_a)
    print(nu_r)
    print(neg_a)
    print(neg_r)
    try:
        print((pos_a*pos_r)/(pos_a+pos_r)*2)
    except:
        print(0)
    try:
        print((nu_a*nu_r)/(nu_a+nu_r)*2)
    except:
        print(0)
    try:
        print((neg_a*neg_r)/(neg_a+neg_r)*2)
    except:
        print(0)
    

def train(model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    ''' 
    for name,param in model.embedding.model.named_parameters():
            param.requires_grad_(False)
            #print(name)
            #print(param.requires_grad)
    '''
    ls=LabelSmoothing(3,0.3)
    t=0
    model.train()
    optimizer=AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.learning_rate)
    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.learning_rate) 
    '''
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = AdamW(optimizer_grouped_parameters,
                         lr=Config.learning_rate)   
    '''
    dev_acc=0
    dev_loss=0
    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    total_batch = 1  # 记录进行到多少batch
    dev_best_loss=float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    #writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    imporve=''
    for epoch in range(Config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, Config.num_epochs))
        #scheduler.step() # 学习率衰减
        true=[]
        predict=[]
        train_loss=0 
        for i, batch in enumerate(train_iter):
             
            #print('batch',i,'len',len(batch)) 
            #print(batch)
            src=[x[1] for x in batch]
            tgt = [x[0] for x in batch]
            labels=[x[2] for x in batch]
            #print(src)
            #print(tgt)
            #print(labels)
            #print(Config.device)
            true.extend(labels) 
            labels=torch.tensor(labels).to(Config.device).long()
             
            outputs = model(src,tgt)
            #print(outputs)
            model.zero_grad()
            #loss = nn.CrossEntropyLoss()(outputs,labels)
            loss= ls(outputs,labels)
            #print(loss)
            ''' 
            for name, parms in model.named_parameters():
                parms.retain_grad()
                print('-->name:', name,'-->grad_requirs:', parms.requires_grad, '-->grad_value:', parms.grad)
            '''
            #print([x.grad for x in optimizer.param_groups[0]['params']])
            train_loss+=loss
            #print(outputs)
            loss.backward()
            optimizer.step()
            predict.extend(torch.max(outputs.data, 1)[1].cpu().numpy().tolist())
             
        #scheduler.step() 
         
        report = metrics.classification_report(true, predict, labels =[0,1,2],target_names=Config.class_list, digits=4)
        print(report)

        print(train_loss/len(train_iter))
        print('--------------')
        
        dev_acc, dev_loss=evaluate(model,dev_iter,False)
        print(dev_loss)
        print(dev_acc)
        
        if dev_acc>0.65:
                    
            print('train')
            '''
            for name, parms in model.embedding.model.named_parameters():
                    print('-->name:', name,'-->grad_requirs:', parms.requires_grad, '-->grad_value:', parms.grad)
            '''
            if t==0:
                t=1
                for param in model.embedding.model.parameters():
                    param.requires_grad = True
                '''
                for name, parms in model.named_parameters():
                    print('-->name:', name,'-->grad_requirs:', parms.requires_grad, '-->grad_value:', parms.grad)
                '''
                optimizer=AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        
        if(dev_loss<dev_best_loss):
            dev_best_loss=dev_loss
            torch.save(model, Config.save_path)
            #torch.save(model.state_dict(), Config.save_path)
            improve = '*'
            
            last_improve = total_batch
        else:
            improve = ''
        print(improve)
        model.train()
        
    test(model, test_iter)
    


def test(model, test_iter):
    # test
    model.load_state_dict(torch.load(Config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    model.train()

def evaluate(model, data_iter, test=False):
    model.eval()
    loss_total = 0
    true=[]
    predict=[]
     
    with torch.no_grad():
        for i, batch in enumerate(data_iter):
            src = [x[1] for x in batch]
            tgt = [x[0] for x in batch]
            labels = [x[2] for x in batch]
            true.extend(labels)
            labels = torch.tensor(labels).to(Config.device).long()
            outputs = model(src,tgt)
            loss = F.nll_loss(outputs, labels)
            loss_total += loss
            predict.extend(torch.max(outputs.data, 1)[1].cpu().numpy().tolist())
        report = metrics.classification_report(true, predict, labels =[0,1,2],target_names=Config.class_list, digits=4)
        acc = metrics.accuracy_score(true, predict)
        print(report)
    if test:
        report = metrics.classification_report(true, predict, labels =[0,1,2],target_names=Config.class_list, digits=4)
        confusion = metrics.confusion_matrix(true, predict)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")

        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):

        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size ))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        self.true_dist = true_dist

        return self.criterion(x, true_dist)
