import torch
from torch import nn
from d2l import torch as d2l


batch_size = 256
num_inputs, num_outputs, num_hiddens = 784, 10, 256
num_epochs, lr = 10, 0.1

loss = nn.CrossEntropyLoss(reduction='none')
updater = torch.optim.SGD(params, lr=lr)

def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

def net(X):
    X = X.reshape((-1, num_inpouts))
    H = relu(X@W1 + b1 )
    H2 = H@W2 + b2
    return H2


def train():
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    
    W1 = nn.parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True)*0.01)
    b1 = nn.parameter(torch.zeros(num_hiddens, requires_grad=True))
    W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True)*0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

    params = [W1, b1, W2, b2]

    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
    d2l.predict_ch3(net, test_iter)

