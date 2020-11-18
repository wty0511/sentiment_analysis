import copy
import torch
import torch.nn as nn

import math
from utils.util import clones
import numpy as np
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from Config import Config





class Embedding(nn.Module):
    def __init__(self,max_length=Config.pad_size,d_model=Config.d_model):
        super().__init__()
        self.max_length=max_length
        self.tokenizer = BertTokenizer.from_pretrained('../ernie-1.0')
        self.model = BertModel.from_pretrained('ernie-1.0')
        self.d_model=d_model
    def forward(self,sentence_batch):
        #sentence_batch(batch size, sentence length)
        embedding_batches=[]
        id_batches=[]

        for sentence in sentence_batch:

            input_ids = torch.tensor([self.tokenizer.encode(sentence, add_special_tokens=False,max_length=self.max_length,pad_to_max_length=True,truncation='longest_first')])
            with torch.no_grad():
                sequence_output, pooled_output = self.model(input_ids)
            sequence_output=sequence_output.squeeze(0)
            embedding_batches.append(sequence_output)
            id_batches.append(input_ids[0])
        embedding_batches=torch.stack(embedding_batches,0) * math.sqrt(self.d_model)
        id_batches = torch.stack(id_batches, 0)
        id_batches=id_batches==0
        id_batches = id_batches.unsqueeze(1)
        id_batches = id_batches.unsqueeze(1)
        return embedding_batches,id_batches
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=Config.d_model, dropout=Config.dropout, max_len=Config.pad_size):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
            x: (batch_size, seq_len, d_model)
        """

        x = x + self.pe[:, :x.size(1)]
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
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
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

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        #(batch size,head num,seq length,embedding dim)
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

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

    def forward(self, x, src_states, src_mask, tgt_mask):
        """norm -> self_attn -> dropout -> add ->
        norm -> src_attn -> dropout -> add ->
        norm -> feed_forward -> dropout -> add"""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, src_states, src_states, src_mask))
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
        x = self.sublayer[1](x, lambda x: self.src_attn(x, src_states, src_states, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class Generator(nn.Module):
    """
    A standard linear + softmax generation step
    """

    def __init__(self, d_model=Config.d_model, cls=Config.num_classes):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, cls)

    def forward(self, x):

        return F.log_softmax(self.proj(x), dim=1)
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

def make_model(N=Config.num_encoder, d_model=Config.d_model, d_ff=Config.d_ff, h=Config.num_head, output_size=Config.num_classes, dropout=Config.dropout):
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
        c(position),
        c(position),
    )

    return model

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture
    """
    def __init__(self, encoder, decoder, embedding, generator,src_postion_embedding,tgt_postion_embedding):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embedding=embedding
        self.generator = generator
        self.src_postion_embedding=src_postion_embedding
        self.tgt_postion_embedding=tgt_postion_embedding
    def forward(self, src, tgt):

        mask_list=[]
        src_embeding_list = []
        for x in src:
            e,m=self.embedding.forward(x)
            m=self.src_postion_embedding(m)
            src_embeding_list.append(e)
            mask_list.append(m)
        src_embeding_list = torch.stack(src_embeding_list, 0)

        mask_list = torch.stack(mask_list, 0)
        tgt_embedding,tgt_mask=self.embedding().forward(tgt)
        tgt_embedding = self.src_postion_embedding(tgt_embedding)
        src_states=[]
        for s,m in zip(src_embeding_list,mask_list):
        # utterance-level self-attention for each src sentence
            src_states.append(self.encoder(s,m))
        src_states = torch.cat(src_states, dim=1)

        # context-level self-attention
        mask_list=torch.cat(mask_list, dim=1)
        src_states = self.encoder(src_states, mask_list)
        print(src_states)
        # cross-attention
        d = self.decoder(tgt_embedding, src_states, mask_list, tgt_mask)
        return d

