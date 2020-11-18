import copy
import torch
import torch.nn as nn
from transformers import AdamW
import math
from utils.util import clones
import numpy as np
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from Config import Config
import numpy as np
from transformers import  AlbertModel
from transformers import AutoTokenizer, AutoModel
class Embedding(nn.Module):
    def __init__(self, max_length=Config.src_pad_size, d_model=Config.d_model):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('./alberttiny')
        self.model = AlbertModel.from_pretrained('./alberttiny').to(Config.device)
        #print(self.model.is_leaf)
        #self.tokenizer = BertTokenizer.from_pretrained("./albert")
        #self.model = AlbertModel.from_pretrained("./albert").to(Config.device)
        self.d_model = d_model
        #self.optimizer = AdamW(self.model.parameters(), lr=1e-5)
        self.is_src=True
        '''        
        for param in self.model.parameters():
            print(param.is_leaf)
        ''' 
        #self.model.train()
        for param in self.model.parameters():
            param.requires_grad = False
    def set(self, is_src):
        self.is_src = is_src

    def forward(self, article_batch):
        # sentence_batch(batch size, sentence length)
        #self.model.eval()
        batch=[]
        mask=[]
        #print(len(article_batch))
        
        if(self.is_src):
            self.max_length=Config.src_pad_size
        else:
            self.max_length = Config.tgt_pad_size
        for article in article_batch:
            m=[]
            b=[]
            for sentence in article:
                sentence = sentence.replace(' ', '')
                input_ids = torch.tensor([self.tokenizer.encode(sentence, add_special_tokens=True,max_length=256, pad_to_max_length=True,truncation='only_first')]).to(Config.device)

                if sentence =='':
                    input_ids=torch.zeros(1,self.max_length).to(Config.device).long()

                 
                sequence_output, pooled_output = self.model(input_ids)
                #print(sequence_output.squeeze(0)[0].is_leaf)     
                b.append(sequence_output.squeeze(0)[0])
                #print(pooled_output)
                if sentence == '':
                    m.append(0)
                else:
                    m.append(1)
                
            b=torch.stack(b,dim=0)
            batch.append(b)
            m=torch.tensor(m)
            mask.append(m)
            #print(len(mask))
        batch = torch.stack(batch,0)
        mask = torch.stack(mask,0)
        mask=mask==0
        #print('bm')
        #print(batch.shape)
        #print(mask.shape)
        #print(batch)
        #print(mask)
        
        return batch, mask.to(Config.device)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=Config.d_model, dropout=Config.dropout, max_len=Config.src_pad_size):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
            x: (batch_size, seq_len, d_model)
        """

        x = x + self.pe
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
    if dropout is not None:
        p_attn = dropout(p_attn)
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
        '''
        print(query.shape)
        print(key.shape)
        print(value.shape)
        print('after')
        '''
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # (batch size,head num,seq length,embedding dim)
        '''
        print(query.shape)
        print(key.shape)
        print(value.shape)
        '''
        # 2) Apply attention on all the projected vectors in batch.
        mask=torch.unsqueeze(mask,1)
        mask = torch.unsqueeze(mask, 1)
        #print('mask shape')
        #print(mask.shape)
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


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn and feed forward"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
        self.aggregation_layer = torch.nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)

    def forward(self, x, src_states, src_mask, tgt_mask):
        """norm -> self_attn -> dropout -> add ->
        norm -> src_attn -> dropout -> add ->
        norm -> feed_forward -> dropout -> add"""

        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        
        #t = self.sublayer[1](x, lambda x: self.src_attn(x, src_states, src_states, src_mask))
        t=self.src_attn(x, src_states, src_states, src_mask)
        c=torch.cat( (x,t),1)
        #print(c.shape)
        c=self.aggregation_layer(c)
        #print(c.shape)
        return self.sublayer[2](c, self.feed_forward)


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
    c = copy.deepcopy
    enc_attn = MultiHeadAttention(h, d_model)
    dec_attn = MultiHeadAttention(h, d_model)
    enc_dec_attn = MultiHeadAttention(h, d_model)
    ff = PositionWiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    emb = Embedding()  # share src and tgt embedding

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(enc_attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(dec_attn), c(enc_dec_attn), c(ff), dropout), N),
        emb,
        Generator(d_model, output_size),

    )

    return model


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture
    """

    def __init__(self, encoder, decoder, embedding, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = embedding
        self.generator = generator


    def forward(self, src, tgt):
        #print('in')
        src_embedding,src_mask = self.embedding(src)
        self.embedding.set(False)
        tgt_embedding, tgt_mask = self.embedding(tgt)
        #src_mask=src_mask.to('cuda')
        #src_embedding = src_embedding.to('cuda')
        #tgt_mask = tgt_mask.to('cuda')
        #tgt_embedding = tgt_embedding.to('cuda')
        context=self.encoder(src_embedding,src_mask)
        #print('context!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #print(context.shape)
        d = self.decoder(tgt_embedding, context, src_mask, tgt_mask)
        d = self.generator(d)

        return d


