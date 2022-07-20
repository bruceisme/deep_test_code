import math
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
