import copy
import torch
import torch.nn as nn
from gensim.models import Word2Vec
import math
import jieba
import gensim
from utils.util import clones
import numpy as np
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from Config import Config
from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers import *
import re
from transformers import  AlbertModel
def isSymbol(inputString):
    return bool(re.match(r'[^\w]', inputString))



class Embedding(object):
    def __init__(self,):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('./alberttiny')
        self.model = AlbertModel.from_pretrained('./alberttiny').to(Config.device)
        self.is_src=None
        #self.w2v= gensim.models.KeyedVectors.load_word2vec_format('../news_comment/baike_26g_news_13g_novel_229g.bin', binary=True)
        '''
        for param in self.model.parameters():
            param.requires_grad = False
        '''
    def set (self,is_src):
        self.is_src=is_src
    def forward(self, sentence_batch):
        #self.model.eval()
        # sentence_batch(batch size, sentence length)
        embedding_batches = []
        id_batches = []
        max_length= Config.src_pad_size if self.is_src else Config.tgt_pad_size
        #print(max_length)
        #print(sentence_batch)

        for sentence in sentence_batch:
            sentence=sentence.replace(' ','')
            #print('sentence')
            #print(sentence)
            input_ids = torch.tensor([self.tokenizer.encode(sentence, add_special_tokens=False,max_length=max_length, pad_to_max_length=True,truncation='only_first')]).to(Config.device)
            #print(input_ids)
        
            if sentence =='':
                input_ids=torch.zeros(1,max_length).long().to(Config.device)
                #print(input_ids)
            #print('ids')
            #print(input_ids)
            

            sequence_output, pooled_output = self.model(input_ids)
            sequence_output = sequence_output.squeeze(0)
            #print(sequence_output)
            embedding_batches.append(sequence_output)
            id_batches.append(input_ids)
        embedding_batches = torch.stack(embedding_batches, 0)* math.sqrt(Config.d_model)
        id_batches = torch.stack(id_batches, 0)
        #print(id_batches)
        id_batches = id_batches == 0
         
        id_batches = id_batches.unsqueeze(1)
        #print(embedding_batches)
        #print(id_batches)
        #print(type(embedding_batches))
        return embedding_batches, id_batches
'''

#w2v= gensim.models.KeyedVectors.load_word2vec_format('../news_comment/baike_26g_news_13g_novel_229g.bin', binary=True)
class Embedding(nn.Module):
    def __init__(self,):
        super().__init__()
        #self.tokenizer = BertTokenizer.from_pretrained('./bert_fineturning')
        #self.model = BertModel.from_pretrained('./bert_fineturning')
        self.is_src=None
        self.w2v= gensim.models.KeyedVectors.load_word2vec_format('./baike_26g_news_13g_novel_229g.bin', binary=True)
        #self.model.train()
    def set (self,is_src):
        self.is_src=is_src
    
    def get_v(self,sentence,max_length):
        #print('getv') 

        vector = []
        mask=[]
        
        try:
            l = list(jieba.cut(sentence, cut_all=False))
            l = [x for x in l if not isSymbol(x)]

            if l==[]:
                #print('not a sentence')
                #print(sentence)
                return torch.stack([torch.zeros(Config.d_model)for _ in range(max_length)],0),torch.tensor([0  for i in range(max_length)])

        except:
            #print('not a sentence')
            #print(sentence)
            return torch.stack([torch.zeros(Config.d_model)for _ in range(max_length)],0),torch.tensor([0  for i in range(max_length)])

        #print('getv')
        count = 0
        i = -1
        #print(len(l))
        while count < max_length:
            i += 1
            if i < len(l):
                #print('w2v')
                #print(torch.tensor(list(self.w2v[l[i]])))
                try:
        
                    vector.append(torch.tensor(list(self.w2v[l[i]])))
                    mask.append(1)
                    count += 1
                except:

                    continue

            else:

                vector.append(torch.tensor([0.0  for i in range(Config.d_model)]))
                count += 1
                mask.append(0)


        return torch.stack(vector,0),torch.tensor(mask)
    def forward(self, sentence_batch):
        # sentence_batch(batch size, sentence length)
        embedding_batches = []
        id_batches = []
        max_length= Config.src_pad_size if self.is_src else Config.tgt_pad_size
        #print(max_length)
        #print(sentence_batch)

        for sentence in sentence_batch:
            sentence=sentence.replace(' ','')
            #print('sentence')
            #print(sentence)

            e,m=self.get_v(sentence,max_length)
            #print('em')
            #print(e)
            #print(m)
            embedding_batches.append(e.to(Config.device))
            id_batches.append(m.to(Config.device))
        embedding_batches = torch.stack(embedding_batches, 0)* math.sqrt(Config.d_model)
        id_batches = torch.stack(id_batches, 0)
        #print(id_batches)
        id_batches=id_batches==0


        #print(embedding_batches)
        #print(id_batches)
        return embedding_batches.to(Config.device), id_batches.to(Config.device)
'''
class PositionalEncoding(nn.Module):
    def __init__(self, ):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=Config.dropout)
        self.is_src=None
        
    def set(self,is_src):
        self.is_src=is_src
    def forward(self, x):
        """
            x: (batch_size, seq_len, d_model)
        """
        max_len= Config.src_pad_size if self.is_src else Config.tgt_pad_size
        pe = torch.zeros(max_len,Config.d_model ).to(Config.device)
        
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0,Config.d_model, 2) * -(math.log(10000.0) / Config.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        x = x + pe
        return self.dropout(x)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout=Config.dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)).to(Config.device) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    #print('attention')
    #print(p_attn)
    #print(p_attn.shape)
    if dropout is not None:
        p_attn = dropout(p_attn)
    #print(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, h=Config.num_head, d_model=Config.d_model, dropout=Config.dropout):
        "Take in #model size and number of heads."
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"

        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # (batch size,head num,seq length,embedding dim)
        #print(query.shape)
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        #print('attention')
        #print(x.shape)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward layers"

    def __init__(self, size, self_attn, feed_forward, dropout=Config.dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """norm -> self_attn -> dropout -> add ->
        norm -> feed_forward -> dropout -> add"""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        #print('encoder')
        #print(x.shape)
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)



class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn and feed forward"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, src_states, src_mask, tgt_mask):
        """norm -> self_attn -> dropout -> add ->
        norm -> src_attn -> dropout -> add ->
        norm -> feed_forward -> dropout -> add"""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        #print('decoder')
        #print(x.shape)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, src_states, src_states, src_mask))
        #print(x.shape)
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking"

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, src_states, src_mask, tgt_mask):
        """
            x: (batch_size, tgt_seq_len, d_model)
            src_states: (batch_size, src_seq_len, d_model)
            src_mask: (batch_size, 1, src_seq_len)
            tgt_mask: (batch_size, tgt_seq_len, tgt_seq_len)
        """

        for layer in self.layers:
            x = layer(x, src_states, src_mask, tgt_mask)
        x = self.norm(x)  # (batch_size, tgt_seq_len, d_model)

        # add max pooling across sequences

        x = F.max_pool1d(x.permute(0, 2, 1), x.shape[1]).squeeze(-1)  # (batch_size, d_model)

        return x



class Generator(nn.Module):
    """
    A standard linear + softmax generation step
    """

    def __init__(self, d_model=Config.d_model, cls=Config.num_classes):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, cls)

    def forward(self, x):

        p=self.proj(x)
        return F.log_softmax(p, dim=1)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model=Config.d_model, d_ff=Config.d_ff, dropout=Config.dropout):
        super(PositionWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
            x: (batch_size, seq_len, d_model)
        """
    
            
        return self.w2(self.dropout(F.relu(self.w1(x))))


def make_model(N=Config.num_encoder, d_model=Config.d_model, d_ff=Config.d_ff, h=Config.num_head,
               output_size=Config.num_classes, dropout=Config.dropout):
    
    print('make_model')
    c = copy.deepcopy
    enc_attn = MultiHeadAttention(h, d_model)
    dec_attn = MultiHeadAttention(h, d_model)
    enc_dec_attn = MultiHeadAttention(h, d_model)
    ff = PositionWiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding()
    emb = Embedding()  # share src and tgt embedding
    print('1')
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(enc_attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(dec_attn), c(enc_dec_attn), c(ff), dropout), N),
        emb,
        Generator(d_model, output_size),
        c(position),
        c(position),
    )
    print('2')
    return model

    
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture
    """

    def __init__(self, encoder, decoder, embedding, generator, src_position_embedding, tgt_position_embedding):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = embedding
        self.generator = generator
        self.src_position_embedding = src_position_embedding
        self.tgt_position_embedding = tgt_position_embedding

    def forward(self, src, tgt):
        #print('in')
        src_mask_list = []
        src_embeding_list = []
        self.src_position_embedding.set(True)
        self.embedding.set(True)

        for x in src:
            #print('src')
            e, m = self.embedding.forward(x)
            m=m.squeeze(1)
            #m=m.unsqueeze(1)
            #print('ems')
            #print(e.shape)
            #print(m.shape)
            
            e = self.src_position_embedding.forward(e)
            src_embeding_list.append(e)
            src_mask_list.append(m)
        
         
        src_embeding_list = torch.stack(src_embeding_list, 0)
        src_mask_list = torch.stack(src_mask_list, 0)
        #print('src')
        #print(src_embeding_list)
        #print(src_mask_list)
        tgt_mask_list = []
        tgt_embeding_list = []
        self.embedding.set(False)
        self.tgt_position_embedding.set(False)
        for x in tgt:
            #print('tgt')
            e, m = self.embedding.forward(x)
            m=m.squeeze(1)
            #m=m.unsqueeze(1)
            #print('emt')
            #print(e.shape)
            #print(m.shape)
            e = self.tgt_position_embedding.forward(e)
            tgt_embeding_list.append(e)
            tgt_mask_list.append(m)
        
        tgt_embeding_list = torch.stack(tgt_embeding_list, 0)
        tgt_mask_list = torch.stack(tgt_mask_list, 0)
         
        #print('tgt')
        #print(tgt_embeding_list)
        #print(tgt_mask_list)
        src_states = []
        for i in range(src_embeding_list.shape[1]):
            # utterance-level self-attention for each src sentence
            #print('src')
            state=self.encoder(src_embeding_list[:, i:i + 1, :, :].squeeze(1), src_mask_list[:, i:i + 1, :, :])
            #print(state.shape)
            src_states.append(state)
        
        src_states = torch.cat(src_states, dim=1)
        #print('src_states')
        #print(src_states)
        #print('context!!!!!!!!!!!!!!!!')
        #print(src_mask_list.shape)
        context_mask=src_mask_list.view(src_mask_list.shape[0],1,1,Config.context_length*Config.src_pad_size)
        '''
        context_mask=src_mask_list.squeeze(2)
        context_mask=context_mask.view((context_mask.shape[0],Config.context_length*Config.src_pad_size))
        context_mask=context_mask.unsqueeze(1)
        context_mask = context_mask.unsqueeze(1)
        '''

        #print('mask')
        #print(context_mask.shape)
        #print(src_states.shape)
        #src_states=src_states.view(src_states.shape[0],Config.context_length*Config.src_pad_size,Config.d_model) 
        # context-level self-attention
        
        src_states = self.encoder(src_states,context_mask)
        #print('decoder')
        #print(src_states)
        '''
        #print('end') 
        d=F.max_pool1d(src_states.permute(0, 2, 1), src_states.shape[1]).squeeze(-1)
        #print(d)
        #print(d.shape)
        d = self.generator(d)
        #print(d)
        return d
        '''
        #print('cross')
        #print(src_states.shape)
        #print(tgt_embeding_list.shape)
        #print(context_mask.shape)
        #print(tgt_mask_list.shape)
        # cross-attention
        #tgt_embeding=self.encoder(tgt_embeding_list.squeeze(1), tgt_mask_list)
        #d = self.decoder(src_states,tgt_embeding_list.squeeze(1), tgt_mask_list, context_mask)
        d = self.decoder(tgt_embeding_list.squeeze(1), src_states, context_mask, tgt_mask_list)
        d = self.generator(d)
        
        return d
        
