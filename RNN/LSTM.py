import torch
from torch import nn
from d2l import torch as d2l

def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs=vocab_size

    #随机生成数据
    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01
    
    def create():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))
    #输入门、遗忘门、输出门、候选记忆单元
    W_xi, W_hi, b_i=create()        #重置门
    W_xf, W_hf, b_f=create()        #更新门
    W_xo, W_ho, b_o=create()        #输出门状态
    W_xc, W_hc, b_c=create()        #候选记忆单元

    W_hq=normal((num_hiddens, num_outputs))
    b_q=torch.zeros(num_outputs, device=device)

    params=[W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q]
    for param in params:
        param.requires_grad(True)
    return params

#初始化lstm隐藏层状态,lstm会返回两个隐藏状态记忆单元和隐状态
def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))

def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q]=params
    (H,C) = state
    outputs=[]
    for X in inputs:
        I=torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F=torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O=torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)

        C=F*C + I*C_tilda
        H=O*torch.tanh(C)

        Y=(H @ W_hq) + b_q
        outputs.append(Y)

    return torch.cat(outputs, fim=0), (H, C)







