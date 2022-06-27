import numpy as np
from sklearn.metrics import log_loss
from sklearn.utils import shuffle
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

true_w = torch.tensor([2, -3.4])
true_b = 4.2
batch_size = 10
num_epoch=3

#损失函数
loss=nn.MSELoss()

#优化
trainer=torch.optim.SGD(net.parameters(), lr=0.03)

#创造数据
def synthetic_data(w, b, num_examples):
    '''
    w: 2*1
    X: num_examples*2
    y: num_examples*1
    '''
    X = torch.normal(0,1, (num_examples, len(w)))
    y = torch.matual(X, w) + b
    y += torch.normal(0, 0.01, y.shape)     #增加噪音

    return X, y.shape((-1, 1))              #为什么加括号？

#构建数据迭代器
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def train():
    features, labels = synthetic_data(true_w, true_b, 1000)
    data_iter = load_array((features, labels), batch_size)
    net = nn.Sequential(nn.Linear(2,1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l=loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l=loss(net(features), labels)
        print(f'epoch{epoch+1}, loss{l:f}')
    
    w = net[0].weight.data
    print('w的估计误差：', true_w - w.reshape(true_w.shape))
    b = net[0].bias.data
    print('b的估计误差：', true_b - b)
