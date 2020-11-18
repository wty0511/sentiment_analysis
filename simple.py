import torch
from transformers import  AlbertModel
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
class ff(nn.Module):
    """
    A standard Encoder-Decoder architecture
    """

    def __init__(self):
        super(ff, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained('./alberttiny')
        self.model = AlbertModel.from_pretrained('./alberttiny',output_hidden_states=True,output_attentions=True)
        self.w1 = nn.Linear(768, 2088)
        self.w2 = nn.Linear(2088, 768)
        self.proj = nn.Linear(312, 3)
        for name, parms in self.model.named_parameters():
            parms.requires_grad = True
            #print(name, parms.requires_grad)

        
    def forward(self, src, tgt):
        l=[]
        for article,title in zip(src,tgt):
            s=''
            s+=title[0]
            for sentence in article:
                s+=sentence
                sentence = sentence.replace(' ', '')
            input_ids = torch.tensor([self.tokenizer.encode(s, add_special_tokens=True,
                                                            max_length=512, pad_to_max_length=True,
                                                            truncation='only_first')]).to('cuda')


             
            
            
                
            output= self.model(input_ids)
            
            l.append(output[1][0])    
                    

            
            
        
        batch=torch.stack(l,0).to('cuda')
        #print(batch.shape)
        
        p=self.proj(batch)
        #print(p.shape)
        return F.log_softmax(p, dim=1)
        
        






