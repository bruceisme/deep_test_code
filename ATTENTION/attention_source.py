import math
from turtle import forward
from matplotlib.cbook import flatten
from matplotlib.pyplot import axis
import  torch
from torch import nn
from d2l import torch as d2l

def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), detype=torch.float32, device=X.device)[None, :]<valid_len[:, None]
    X[~max]=value

def masked_softmax(X, valid_lens):
    #在最后一个轴上遮蔽元素来执行softmax操作
    #X:3D,batchXlenXembedding
    # valid_lens:1D或2D, batchsize=2时，2x2x4
    # [2,3]是一维，2表示第一个样本前两列有效，3表示第二个样本前三列有效
    """[[1,3],[2,4]]是二维,
    [1,3]中1表示第一个样本第一行的第1个样本有效,3表示第二行的前三列有效
    [2,4]中2表示第一个样本第一行的前2个样本有效,4表示第二行的前4行有效"""
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape=X.shape
        #如果是1维
        if valid_lens.dim()==1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        #2维
        else:
            valid_lens = valid_lens.reshape(-1)
        #最后一轴的masked元素使用非常大的负值来替换，softmax后输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)

        return nn.functional.softmax(X.reshape(shape), dim=-1)
#加性注意力,quer和key不同长度时使用
class AdditiveAttention(nn.Block):
    def __init__(self, key_size,query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k=nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q=nn.Linear(query_size, num_hiddens, bias=False)
        self.W_v=nn.Linear(num_hiddens, 1, bias=False)
        self.dropout=nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys=self.W_q(queries), self.W_k(keys)
        #queries:batch,queries个数,1,num_hidden
        #key:batch,1,"key-balue"个数,num_hidden
        features=queries.unsqueeze(2)+keys.unsqueeze(1)
        features=torch.tanh(features)
        #source：batch,quries个数,"key-value"个数
        scores=self.W_v(features).squeeze(-1)
        self.attention_weights=masked_softmax(scores, valid_lens)
        #values:batch,"key-value"个数，值的维度
        return torch.bmm(self.dropout(self.attention_weights), values)

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


#自注意力
num_hiddens, num_heads=100, 5
attention= d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5)
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