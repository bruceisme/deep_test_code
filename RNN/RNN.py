from ast import increment_lineno
from binascii import b2a_hex

from zmq import device


%matplotlib inline
import math
import torch
from torch import nn
from torch.nn import founctional as F
from d2l import torch as d2l

batch_size, num_steps=32,35
train_iter, vocab=d2l.load_data_time_machine(batch_size, num_steps)
F.one_hot(torch.tensor([0,2]), len(vocab))
#小批量数据维度应为（批量大小x时间步x词向量维度）
#使用的数据维度应为（时间步x批量大小x词向量维度）

#参数定义
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs=vocab_size
    #生成一些数据
    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01
    #h_t=W_hh*h_t-1 + W_hx*x_t-1 + b_h
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h=torch.zeros(num_hiddens, device=device)
    #q=W_hq*h_t+b_q
    W_hq=normal((num_hiddens, num_outputs))
    b_q=torch.zeros(num_outputs, device=device)

    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

#在初始化时返回隐藏状态
def init_rnn_state(batch_size, num_hidden, device):
    return (torch.zeros((batch_size, num_hidden), device=device),)

#state上一层传递的隐藏状态
def rnn(inputs, state, params):
    #h_t=W_hh*h_t-1 + W_hx*x_t-1 + b_h
    #q=W_hq*h_t+b_q
    W_xh, W_hh,b_h, W_hq, b_q = params
    H, =state
    outputs=[]
    for X in inputs:
        H=torch.tanh(torch.mm(X, W_xh)+ torch.mm(H,W_hh)+b_h)
        Y=torch.mm(H, W_hq)+b_q
        outputs.append(Y)
    #列数不变，行数变成btach*时间步x词向量维度
    #H:(batchxhidden_size)
    return torch.cat(outputs,dim=0), (H,)

class RNNModelScreatch:
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size=vocab_size
        self.num_hiddens = num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state=init_state
        #RNN
        self.forward_fn=forward_fn

    def __call__(self, X, state):
        X=F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)
    
    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)

num_hiddens=512
net=RNNModelScreatch(len(vocab), num_hiddens, d2l.try_gpu(), get_params, init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.to(d2l.try_gpu()), state)
Y.shape, len(new_state), new_state[0].shape

