import math
from turtle import forward
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l



#缩放点积注意力，query和key长度一致
class DotProductAttnetion(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttnetion, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        #获得query和key一致的长度
        d=queries.shape[-1]
        #交换keys最后两个维度
        scores=torch.bmm(queries, keys.transpose(1,2))/math.sqrt(d)
        self.attention_weights=masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


#多头注意力

def transpose_qkv(X, num_heads):
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数, num_hiddens/num_heads)
    X = X.permute(0,2,1,3)
    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数, num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0,2,1,3)
    return X.reshape(X.shape[0], X.shape[1], -1)

class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads=num_heads
        self.attention = DotProductAttnetion(dropout)
        #将query，key，value变化成相同长度
        self.W_q=nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k=nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v=nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o=nn.Linear(num_hiddens, num_hiddens, bias=bias)
    
    def forward(self, queries, keys, values, valid_lens):
        #输入query：batch_size, query或key-value个数， num_hiddens
        #valid_lens: (batch,)或(batch, query个数)
        #linear变换后
        #batchsize*num_heads, query/key-value个数，num_hiddens/num_heads
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_hideens)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        #outputs:batch*num_heads,query个数，num_hiddens/num_heads
        output = self.attention(queries, keys, values, valid_lens)

        #batch,querys个数,num_hiddens
        output_concat=transpose_output(output, self.num_heads)
        return self.W_o(output_concat)











#前馈网络
class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        X=self.dense1(X)
        X=self.relu(X)
        return self.dense2(X)

#残差和归一化
class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout=nn.Dropout(dropout)
        self.norm = nn.LayerNorm(normalized_shape)
    
    def forward(self, X, Y):
        return self.norm(self.dropout(Y)+X)

class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        #创造一个足够长的P
        #1:batch_size例子
        self.P = torch.zeros((1,max_len,num_hiddens))
        X=torch.arange(max_len, dtype=torch.float32).reshape(-1,1)/torch.pow(10000,torch.arange(0, num_hiddens, 2, dtype=torch.float32)/num_hiddens)
        self.P[:,:,0::2]=torch.sin(X)
        self.P[:,:,1::2]=torch.cos(X)

    def forward(self, X):
        X = X+self.P[:,:X.shape[1], :].to(X.device)
        return self.dropout(X)






#编码器

#编码器块
class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, 
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention=MultiHeadAttention(key_size, query_size, value_size, 
                            num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1=AddNorm(norm_shape, dropout)
        self.ffn=PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2=AddNorm(norm_shape, dropout)
    
    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

class TransformerEncoder(d2l.Encoder):
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens,
                norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i), 
                EncoderBlock(key_size, query_size, value_size, num_heads,
                            norm_shape, ffn_num_input, ffn_num_hiddens, 
                            num_heads, dropout, use_bias))
    
    def forward(self, X, valid_lens, *args):
        X = self.pos_encoding(self.embedding(X)*math.sqrt(self.num_hiddens))
        self.attention_weight=[None]*len(self.blks)
        for i,blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weight[i] = blk.attention.attention.attention_weights
        return X

class DecoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i=i
        self.attention1=MultiHeadAttention(key_size, query_size, value_size,
                        num_hiddens, num_heads, dropout)
        self.addnorm1=AddNorm(norm_shape, dropout)
        self.attention2=MultiHeadAttention(key_size, query_size, value_size,
                        num_hiddens, num_heads, dropout)
        self.addnorm2=AddNorm(norm_shape, dropout)
        self.ffn=PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_heads)
        self.addnorm3=AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs=state[0]
        enc_valid_lens=state[1]

        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i]=key_values
        if self.training:
            batch_size, num_steps, _ = X.shape

            dec_valid_lens = torch.arange(1, num_steps+1, device=X.device).repeat(batch_size,1)
        else:
            dec_valid_lens=None

        X2=self.attention1(X, key_values, key_values, dec_valid_lens)
        Y=self.addnorm1(X, X2)
        
        Y2=self.attention2(Y, enc_outputs,enc_outputs, enc_valid_lens)
        Z=self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state


