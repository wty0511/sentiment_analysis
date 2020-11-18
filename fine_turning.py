from transformers import BertForSequenceClassification
from transformers import AdamW
from transformers import BertTokenizer
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.utils import shuffle
class DatasetIterater(object):
    def __init__(self, batches, batch_size):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  
        if len(batches) % batch_size != 0:
            self.residue = True
        self.index = 0
        


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

                                  
def build_iterator(dataset,batch_size):
    #print(Config.batch_size)
    iter = DatasetIterater(dataset, batch_size)
    return iter
model = BertForSequenceClassification.from_pretrained('bert-base-chinese',num_labels=3).to('cuda')
model.train()
optimizer = AdamW(model.parameters(), lr=1e-7)
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
data=pd.read_csv('./data_balance.csv',sep=',',header=None)


data = shuffle(data)
i=0

b=[]
print(data)
for tup in data.itertuples():
    i+=1
    #if(i==50):
        #break
    try:
        b.append((tup[1]+tup[2],tup[3]))
        
        
    except:
        print(tup[1])
        print(tup[2])
        
#text_batch = ["我爱你", "我恨你"]
data_iter=build_iterator(b,1)

#encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True,max_length=128)
#input_ids = encoding['input_ids'].to('cuda')
#attention_mask = encoding['attention_mask'].to('cuda')
#label = torch.tensor(label).unsqueeze(0).to('cuda')
#label = torch.tensor([1,0]).unsqueeze(0).to('cuda')
#print(label)
#print(input_ids)
acc=0
#print(attention_mask)
for i in range(10):
    t=[]
    p=[]
    for batch in data_iter:
        text_batch=[x[0] for x in batch]
        label=[x[1] for x in batch]
        encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True,max_length=512)
        input_ids = encoding['input_ids'].to('cuda')
        attention_mask = encoding['attention_mask'].to('cuda')
        label = torch.tensor(label).unsqueeze(0).to('cuda')
        outputs = model(input_ids, attention_mask=attention_mask, labels=label)
        #print(outputs[0])
        loss,output = outputs[:2]
        #print(loss)
        predic = torch.max(output.data, 1)[1].cpu()
        label=label.cpu()
        t.extend(label.squeeze(0).numpy())
        p.extend(predic.numpy())
    
        loss.backward()
        optimizer.step()
    train_acc = metrics.accuracy_score(t, p)
    print(train_acc)

    if(acc<=train_acc):
        acc=train_acc
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), './ernie-1.0/pytorch_model.bin')
        model_to_save.config.to_json_file('./ernie-1.0/config.json')
        tokenizer.save_vocabulary('./ernie-1.0/vocab.txt')
'''
model_to_save = model.module if hasattr(model, 'module') else model

torch.save(model_to_save.state_dict(), './ernie-1.0/pytorch_model.bin')
model_to_save.config.to_json_file('./ernie-1.0/config.json')
tokenizer.save_vocabulary('vocab.txt')
'''


