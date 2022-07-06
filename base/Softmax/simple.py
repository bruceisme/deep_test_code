#!pip install git+https://github.com/d2l-ai/d2l-zh@release
import torch
from torch import nn
from d2l import torch as d2l



def init_weights(m)
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight, std=0.01)



def train(batch_size, num_epochs ):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    loss = nn.CrossEntropyLoss(reduction='None')
    trainer = torch.optim.SGD(net.pararmeters(), lr=0.1)
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    net.apply(init_weights)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

if __name__ == "__main__":
    batch_size = 256
    num_epochs = 10
    train(batch_size, num_epochs)



