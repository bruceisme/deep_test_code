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
    
    W_xr, W_hr, b_r=create()        #重置门
    W_xz, W_hz, b_z=create()        #更新门
    W_xh, W_hh, b_h=create()        #候选隐状态

    W_hq=normal((num_hiddens, num_outputs))
    b_q=torch.zeros(num_outputs, device=device)

    params=[W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad)(True)
    return params

#初始化第一个隐藏层状态
def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)

def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q=params
    H, =state
    outputs=[]
    for X in inputs:
        Z=torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R=torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H=Z*H +(1-Z)*H_tilda
        #当前时间步的预测
        Y=H @ W_hq + b_q
        outputs.append(Y)
    
    return torch.cat(outputs, dim=0), (H,)




